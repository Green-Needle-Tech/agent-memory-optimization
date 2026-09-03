#!/usr/bin/env python3
"""Daily memory optimization — L1/L2/L3 maintenance for Hermes Agent (v3.1).

no-agent cron script: stdout is delivered verbatim; empty stdout = silent.

v3.0 (Sep 2026): LLM-as-judge replaced with deterministic rule-based
heuristics (memory_heuristics.py). Zero external chat-completion calls.
  - heuristic_dedup_pass: exact (SHA-256 of normalized content) + strong
    (structured claim / high lexical threshold) duplicate detection,
    indexed candidate generation — no O(n^2) matrix, cross-batch coverage
  - heuristic_contradiction_scan: structured claim extraction; recency_wins
    only with explicit transitions or reliable timestamps; stable and
    uncertain conflicts flagged for human review, never auto-resolved
  - try_resolve_issues_with_rules: fixed remediation allowlist — the LLM
    resolver that could select arbitrary API actions (including
    invalidation by recall query) is removed
  - recall preserves id/content/created_at/updated_at/tags metadata
  - structured Issue records (code/severity/message/context) — free-form
    problem strings exist only at the output boundary
  - dry-run mode (MEMORY_HEURISTICS_DRY_RUN=1 or --dry-run): analysis and
    reporting only, no memory mutation
  - audit log for every mutation (rule id, confidence, reason, timestamp)

v3.0.1 (Sep 2026): Timeout fix — skip consolidation when pending=0, POLL_DEADLINE 480→240s,
SCRIPT_TIME_BUDGET=540s guard on remaining steps. Fixes 3 consecutive cron timeouts.
v2.3 (Sep 2026): Correctness — structured records, deterministic recency, paginated scanning.
v2.2.1 (Sep 2026): Safety patch — fact_type-aware invalidation, Telegram
HTML escaping, accurate delivery status, env-var config, atomic state writes.
v2.0.1 (Aug 2026): self-pollution fixes — smoke-test retirement, meta-memory
invalidation, shared recall set.

Behavior (per memory-optimization skill):
  L2 (Hindsight, any 0.8+; Knowledge Pages checks activate on 0.9+):
    1. POST /consolidate -> poll operation to terminal state (max 8 min)
    2. Recall smoke-test after consolidation (over-prune check)
    3. Retain smoke-test: success:true AND total_tokens>0
    4. Bank stats: total_nodes>0, failed_operations trend vs last run
    5. Remove expired smoke-test records
    6. Remove known meta-maintenance records
    7. Heuristic dedup (exact + strong), invalidate via PATCH
    8. Re-fetch/filter invalidated records
    9. Structured contradiction detection
   10. Apply high-confidence recency_wins only; report the rest
    11. Knowledge Pages tree: count pages + is_stale pages (0.9+ only)
  L1 (local memory):
    12. MEMORY.md / USER.md capacity check; >=90% triggers Hindsight offload
  L3 (LLM wiki / OKF bundle):
    13. Stale-page lint trigger: >=5 active pages >90 days stale (frontmatter
        `updated` with mtime fallback; _archive/ and index pages excluded)
        runs one read-only lint pass (llmwiki lint --json, python3 -m llmwiki
        fallback, 5-min timeout) and summarizes the JSON report (pages scanned,
        error/warning/info counts, up to 3 sample issues). Lint failures
        (missing CLI, timeout, malformed output, nonzero exit) are reported
        as unresolved issues without crashing the run.
  14. Deterministic remediation rules (allowlist)
  15. Telegram notification for unresolved issues

Output only when something needs attention; else silent. Exit 0 always
(a crash is reported via stdout, never via nonzero exit).
"""

import argparse
import contextlib
import dataclasses
import html
import json
import os
import re
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

# Rule-based heuristics module (v3.0 — replaces llm_judge entirely)
sys.path.insert(0, str(Path(__file__).parent))
import memory_heuristics  # noqa: E402  (always available — ships with this repo)

# v2.3: structured memory records, paginated listing, candidate generation
memory_records: types.ModuleType | None
try:
    import memory_records
except ImportError:
    memory_records = None  # standalone run: use old recall approach

# Reuse the proven offload routine from the 30-min offload cron
# (deployed copy lives next to this file in ~/.hermes/scripts/).
memory_offload: types.ModuleType | None
try:
    import memory_offload
except ImportError:
    memory_offload = None  # standalone run: offload step will be skipped

# === Config (environment-overridable, no hardcoded /root) ===
HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
BASE = os.environ.get("HINDSIGHT_URL", "http://localhost:8888")
BANK = os.environ.get("HINDSIGHT_BANK", "main")
MEM_FILE = Path(os.environ.get("MEMORY_FILE", str(HERMES_HOME / "memories" / "MEMORY.md")))
USER_FILE = Path(os.environ.get("USER_FILE", str(HERMES_HOME / "memories" / "USER.md")))
KB_DIR = Path(os.environ.get("KB_DIR", str(HERMES_HOME / "kb")))
WIKI_DIR = Path(os.environ.get("WIKI_DIR", str(Path.home() / "wiki")))
ENV_FILE = HERMES_HOME / ".env"

POLL_INTERVAL = 10
POLL_DEADLINE = 240          # 4 min max wait for consolidation (was 480 — caused cron timeouts)
SCRIPT_TIME_BUDGET = 540     # global soft limit: skip remaining steps if exceeded (cron timeout is 600s)
MEM_CAP = int(os.environ.get("MEMORY_CHARS", "2200"))
USER_CAP = int(os.environ.get("USER_CHARS", "1375"))
MEM_WARN = 0.90              # warn at 90% capacity
OFFLOAD_AT = 0.90            # trigger Hindsight offload at >=90%
KB_STALE_DAYS = 90           # L3 page staleness threshold (skill: >90 days = stale)
FAILED_OPS_WARN = 10         # failed_operations count worth reporting
KP_STALE_RATIO_WARN = 0.5    # warn when >50% of knowledge pages are stale
KP_EXACT_CHECK_MAX = 25      # use exact per-page mental-model is_stale for KBs up to this size
STATE_FILE = HERMES_HOME / "scripts" / ".daily_memory_opt_state.json"

# Heuristic dedup/contradiction settings
DEDUP_RECALL_LIMIT = 50      # max memories to recall for dedup/contradiction scan
DEDUP_RECALL_TOKENS = 3000   # token budget for recall query

# v2.2.1: destructive mutation disabled by default — requires --allow-destructive
ALLOW_DESTRUCTIVE = False

# v2.0.1: self-pollution guards
SMOKE_TEST_TAG = "daily-memopt-smoke"   # tag on the retain smoke-test memory
SMOKE_TEST_MAX_AGE_S = 172800           # smoke-test memories older than 48h = junk

# v3.0.1: global runtime guard — skip remaining steps if budget exceeded
_RUN_START = time.time()

def _time_remaining():
    """Return True if we still have time budget, False if exceeded."""
    return (time.time() - _RUN_START) < SCRIPT_TIME_BUDGET
META_MEMORY_PATTERNS = [                # reports ABOUT past maintenance runs = noise
    r"stable-attribute conflict was flagged",
    r"conflict was flagged",
    r"duplicates? (?:were|was) invalidated",
    r"memor(?:y|ies) (?:were|was) invalidated",
    r"invalidated \(non-destructive",
    r"recency-wins",
    r"needs? review",
    r"flagged for (?:human )?review",
]
META_MEMORY_RE = re.compile("|".join(META_MEMORY_PATTERNS), re.IGNORECASE)
FLAG_STATE_FILE = HERMES_HOME / "scripts" / ".daily_memory_opt_flags.json"

# Telegram notification config (for unresolved issues)
TG_MAX_MSG_LEN = 4000   # Telegram message limit is 4096; keep margin

# v2.4: Constants for previously-duplicated literals (SonarCloud design issues)
META_NOISE_REASON = "meta_noise: report about a past maintenance run"
ISO_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S"


# === Structured issues (v3.0 spec section 10) ===

@dataclasses.dataclass
class Issue:
    """Structured maintenance issue. Rendering to text happens only at the
    output boundary (render_issue)."""
    code: str
    severity: str            # "info" | "warning" | "critical"
    message: str
    context: dict = dataclasses.field(default_factory=dict)
    auto_resolvable: bool = False


def render_issue(issue: Issue) -> str:
    """Render an Issue to human-readable text (output boundary only)."""
    detail = ""
    if issue.context:
        parts = [f"{k}={v}" for k, v in issue.context.items()]
        detail = f" ({', '.join(parts)})"
    return f"{issue.message}{detail}"


def _read_env_var(var_name):
    """Read a variable from environment or ~/.hermes/.env file."""
    val = os.environ.get(var_name, "")
    if val:
        return val
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(var_name + "="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def _atomic_write_text(path: Path, content: str) -> None:
    """Write content to path atomically via temp file + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram parse_mode=HTML."""
    return html.escape(text, quote=False)


def _is_dry_run(args) -> bool:
    """Dry-run is active via CLI flag or MEMORY_HEURISTICS_DRY_RUN env/config."""
    return bool(getattr(args, "dry_run", False)) or memory_heuristics.is_dry_run()


# === Safe memory invalidation (v2.2.1) ===

def _get_memory(mid):
    """Fetch a single memory by ID. Returns dict or None."""
    try:
        _, resp = http("GET", f"/v1/default/banks/{BANK}/memories/{mid}", timeout=120)
        return resp
    except Exception:
        return None


def curate_memory(memory_id, action="invalidate", reason=""):
    """Safe memory curation: invalidates a memory after checking fact_type.

    Only world and experience memories can be directly invalidated.
    Observations are derived — PATCH returns 400. For observations, we
    find and invalidate their source memories instead.

    Returns True if the memory (or its sources) were invalidated, False otherwise.
    """
    mem = _get_memory(memory_id)
    if mem is None:
        return False

    fact_type = mem.get("fact_type", "world")

    if fact_type in ("world", "experience"):
        # Source fact — safe to invalidate directly
        return _patch_invalidate(memory_id, reason)

    if fact_type == "observation":
        # Derived fact — can't PATCH directly. Find and invalidate sources.
        source_ids = mem.get("source_memory_ids", [])
        if not source_ids:
            # No source IDs available — can't safely invalidate
            return False
        invalidated_any = False
        for sid in source_ids:
            if _patch_invalidate(sid, reason):
                invalidated_any = True
        return invalidated_any

    # Unknown fact type — don't touch it
    return False


def _patch_invalidate(mid, reason=""):
    """Direct PATCH invalidation (world/experience only). Never DELETE."""
    try:
        http("PATCH", f"/v1/default/banks/{BANK}/memories/{mid}", timeout=120,
             body={"state": "invalidated", "reason": reason})
        return True
    except Exception:
        return False


def invalidate_memory(mid, reason="duplicate"):
    """Non-destructive invalidation with fact_type safety check.

    Observations are handled by invalidating their source memories.
    Never DELETE — invalidation preserves audit trail and is recoverable.
    """
    return curate_memory(mid, action="invalidate", reason=reason)


# === Deterministic remediation rules (v3.0 spec section 11) ===
# Fixed allowlist: the LLM resolver that could select arbitrary API
# actions (including invalidation by arbitrary recall query) is removed.
# Everything outside this allowlist remains unresolved and is sent
# through the notification path.

RULE_REMEDIATIONS = {
    "L2_CONSOLIDATION_PENDING": "trigger_consolidation",
    "L1_CAPACITY_EXCEEDED": "run_memory_offload",
    "SMOKE_TEST_EXPIRED": "invalidate_exact_memory_id",
    "META_MEMORY_FOUND": "invalidate_exact_memory_id",
    "EXACT_DUPLICATE": "invalidate_exact_memory_id",
    "STRONG_DUPLICATE": "invalidate_exact_memory_id",
    "STATE_CHANGE_HIGH_CONFIDENCE": "invalidate_exact_older_memory_id",
}


def try_resolve_issues_with_rules(issues):
    """Apply the fixed remediation allowlist to structured issues.

    The resolver must not:
    - Search arbitrary text and invalidate all recall matches
    - Change unknown Hindsight configuration keys
    - Resolve stable conflicts
    - Modify L3 content
    - Execute actions based on generated natural-language instructions

    Returns (resolved_issues, unresolved_issues) lists.
    """
    resolved, unresolved = [], []
    for issue in issues:
        action = RULE_REMEDIATIONS.get(issue.code)
        if action is None:
            unresolved.append(issue)
            continue
        result = _execute_rule_remediation(issue, action)
        if result is not None:
            resolved.append(result)
        else:
            unresolved.append(issue)
    return resolved, unresolved


def _execute_rule_remediation(issue: Issue, action: str):
    """Execute one allowlisted remediation. Returns resolution text or None."""
    try:
        if action == "trigger_consolidation":
            http("POST", f"/v1/default/banks/{BANK}/consolidate", timeout=120, body={})
            return f"{render_issue(issue)} → resolved (consolidation triggered)"

        if action == "run_memory_offload":
            if memory_offload is None:
                return None
            used_before, _ = memory_offload.get_memory_usage()
            memory_offload.main()
            used_after, _ = memory_offload.get_memory_usage()
            if used_after < used_before:
                return (f"{render_issue(issue)} → resolved "
                        f"(offloaded, now {used_after} chars)")
            return None

        if action == "invalidate_exact_memory_id":
            mid = issue.context.get("memory_id")
            if not mid or not ALLOW_DESTRUCTIVE:
                return None
            if invalidate_memory(mid, reason=issue.code.lower()):
                memory_heuristics.audit_log(
                    operation="invalidate", memory_id=mid,
                    rule=issue.code, confidence="high",
                    reason=issue.message,
                )
                return f"{render_issue(issue)} → resolved (invalidated {mid[:12]}…)"
            return None

        if action == "invalidate_exact_older_memory_id":
            mid = issue.context.get("older_memory_id")
            if not mid or not ALLOW_DESTRUCTIVE:
                return None
            if invalidate_memory(mid, reason="superseded_state_change"):
                memory_heuristics.audit_log(
                    operation="invalidate", memory_id=mid,
                    rule=issue.code, confidence="high",
                    reason=issue.message,
                    replacement_id=issue.context.get("newer_memory_id"),
                )
                return f"{render_issue(issue)} → resolved (invalidated older {mid[:12]}…)"
            return None

        return None  # unknown action — never guess
    except Exception:
        return None


def send_telegram_notification(text):
    """Send a message to the user's Telegram DM via Bot API.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_HOME_CHANNEL from ~/.hermes/.env.
    Returns True on success, False on failure (silent — never crash the cron).
    """
    bot_token = _read_env_var("TELEGRAM_BOT_TOKEN")
    chat_id = _read_env_var("TELEGRAM_HOME_CHANNEL")
    if not bot_token or not chat_id:
        return False

    # HTML content is escaped by the caller; only our own tags are raw
    if len(text) > TG_MAX_MSG_LEN:
        text = text[:TG_MAX_MSG_LEN - 20] + "\n\n…(truncated)"

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        body = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            resp.read()
        return True
    except Exception:
        return False


def http(method, path, timeout=120, body=None):
    req = urllib.request.Request(
        BASE + path,
        method=method,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
        return r.status, json.loads(r.read() or b"{}")
    # urllib.error.HTTPError propagates to caller's try/except


def walk_tree(nodes):
    """Yield every node in the Knowledge Pages tree recursively."""
    for n in nodes or []:
        yield n
        yield from walk_tree(n.get("children"))


def recall_recent_memories(query="user preferences, environment, configuration, tools", limit=None):
    """Recall recent memories for heuristic dedup + contradiction scan.

    v3.0: preserves metadata — id, content, created_at, updated_at, tags.
    The contradiction detector must receive these records, not only
    content strings (timestamp safety: recall order is never treated as
    chronological order; timestamps come from metadata).

    Returns list of record dicts, or empty list on failure.
    """
    try:
        _, resp = http("POST", f"/v1/default/banks/{BANK}/memories/recall", timeout=120,
                       body={"query": query, "max_tokens": DEDUP_RECALL_TOKENS})
        hits = resp.get("results") or resp.get("memories") or resp.get("items") or []
        memories = []
        for h in hits:
            mid = h.get("id") or h.get("memory_id")
            content = h.get("content", "") or h.get("text", "")
            if mid and content:
                memories.append({
                    "id": mid,
                    "content": content,
                    "created_at": h.get("created_at"),
                    "updated_at": h.get("updated_at") or h.get("edited_at") or h.get("date"),
                    "tags": h.get("tags", []),
                })
        return memories[:(limit or DEDUP_RECALL_LIMIT)]
    except Exception:
        return []


def recall_all_recent(limit=DEDUP_RECALL_LIMIT):
    """Recall a broad mix of recent memories (single shared set for dedup + contradictions)."""
    return recall_recent_memories(
        query="user preferences, environment, configuration, tools, versions, migrations, decisions",
        limit=limit,
    )


def cleanup_smoke_tests(issues):
    """Invalidate old smoke-test memories (v2.0.1 self-pollution fix).

    Each new smoke-test memory is tagged SMOKE_TEST_TAG; anything tagged
    and older than SMOKE_TEST_MAX_AGE_S is invalidated. Legacy untagged
    ones are matched by content prefix.
    """
    cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - SMOKE_TEST_MAX_AGE_S))
    removed = 0
    try:
        _, resp = http("POST", f"/v1/default/banks/{BANK}/memories/recall", timeout=120,
                       body={"query": "daily memory optimization smoke test", "max_tokens": 2000})
        hits = resp.get("results") or resp.get("memories") or resp.get("items") or []
        for h in hits:
            content = (h.get("content") or h.get("text") or "")
            tags = [str(t).lower() for t in (h.get("tags") or [])]
            is_smoke = SMOKE_TEST_TAG in tags or content.startswith("daily memory optimization smoke test")
            if not is_smoke:
                continue
            # Date embedded in content; invalidate only if older than the cutoff
            m = re.search(r"smoke test (\d{4}-\d{2}-\d{2})", content)
            if m and m.group(1) > cutoff:
                continue  # recent — keep (it's this run's probe)
            mid = h.get("id") or h.get("memory_id")
            if mid and invalidate_memory(mid, reason="smoke_test_junk"):
                removed += 1
                memory_heuristics.audit_log(
                    operation="invalidate", memory_id=mid,
                    rule="SMOKE_TEST_EXPIRED", confidence="high",
                    reason="expired smoke-test record",
                )
    except Exception:
        return  # cleanup is best-effort; never block the run
    if removed:
        issues.append(Issue(
            code="SMOKE_TEST_CLEANUP", severity="info",
            message=f"Self-pollution cleanup: {removed} old smoke-test memories invalidated",
        ))


def invalidate_meta_memories(memories, issues):
    """Invalidate meta-memories: reports ABOUT past maintenance runs (v2.0.1).

    Past runs' problem reports were retained as memories. Those reports
    then flag against every related fact on every subsequent run —
    self-referential noise. They are bookkeeping, not facts; invalidate
    on sight.
    """
    removed = 0
    for m in memories:
        if META_MEMORY_RE.search(m.get("content", "")):
            if invalidate_memory(m["id"], reason=META_NOISE_REASON):
                removed += 1
                memory_heuristics.audit_log(
                    operation="invalidate", memory_id=m["id"],
                    rule="META_MEMORY_FOUND", confidence="high",
                    reason=META_NOISE_REASON,
                )
    if removed:
        issues.append(Issue(
            code="META_MEMORY_CLEANUP", severity="info",
            message=f"Self-pollution cleanup: {removed} meta-memories "
                    f"(reports about past runs) invalidated",
        ))
    return removed


# === Heuristic dedup pass (v3.0 — replaces llm_semantic_dedup_pass) ===

def heuristic_dedup_pass(issues, memories=None):
    """Rule-based semantic dedup pass (exact + strong, v3.0).

    v3.0 spec section 10:
      - Uses memory_heuristics.semantic_dedup() over the full input set —
        duplicates across different recall batches ARE detected (the old
        LLM chunked approach compared only within 30-entry batches).
      - Indexed candidate generation — no unrestricted O(n^2) comparison.
      - Only exact and strong groups are invalidated; "possible" groups
        are reported only.
      - Dry-run: prints proposed actions with rule identifiers, no mutation.

    Research basis:
    - Hindsight blog: fact deduplication — "same claim, different wording"
    - Human-Inspired Memory: dedup-based consolidation achieves 97.2%
      precision, 58% store reduction
    """
    if memories is None:
        memories = recall_all_recent()
    if len(memories) < 2:
        return

    dup_groups = memory_heuristics.semantic_dedup(memories)
    invalidated = 0
    possible_reported = 0

    for group in dup_groups:
        confidence = group.get("confidence")
        if confidence not in ("exact", "strong"):
            possible_reported += 1
            continue
        for dup_idx in group["duplicates"]:
            if not (0 <= dup_idx < len(memories)):
                continue
            mid = memories[dup_idx]["id"]
            reason = f"duplicate of {memories[group['canonical']]['id'][:12]}… ({group['rule']})"
            if _is_dry_run(_current_args):
                print(f"DRY RUN: would invalidate memory {mid}")
                print(f"  Rule: {group['rule']}")
                print(f"  Canonical: {memories[group['canonical']]['id']}")
                print(f"  Reason: {group['reason']}")
                continue
            if memory_records is not None:
                memory_records.append_audit_log(memory_records.AuditEntry(
                    timestamp=time.strftime(ISO_TIMESTAMP_FMT, time.gmtime()),
                    action="invalidate",
                    memory_id=mid,
                    reason=reason,
                ))
            if invalidate_memory(mid, reason=reason):
                memory_heuristics.audit_log(
                    operation="invalidate", memory_id=mid,
                    rule=group["rule"], confidence=confidence,
                    reason=group["reason"],
                    replacement_id=memories[group["canonical"]]["id"],
                )
                invalidated += 1

    if invalidated > 0:
        issues.append(Issue(
            code="STRONG_DUPLICATE" if invalidated else "EXACT_DUPLICATE",
            severity="info",
            message=f"Heuristic dedup: {invalidated} duplicate memories invalidated "
                    f"(non-destructive — retained on disk for audit)",
            context={"scanned": len(memories), "auto_resolvable": True},
        ))
    if possible_reported > 0:
        issues.append(Issue(
            code="POSSIBLE_DUPLICATE_REPORT", severity="info",
            message=f"Heuristic dedup: {possible_reported} possible (report-only) "
                    f"duplicate groups found — no action taken",
        ))


# === Heuristic contradiction scan (v3.0 — replaces llm_contradiction_scan) ===

def heuristic_contradiction_scan(issues, memories=None):
    """Rule-based structured contradiction detection (v3.0).

    v3.0 spec sections 9-10:
      - Structured claim extraction; only reliably-parsed claims compared.
      - recency_wins applied ONLY for high-confidence results (explicit
        transitions or reliable timestamps). Recall order is never
        treated as chronological order.
      - Stable and uncertain conflicts are reported (flag_human), never
        auto-resolved.
      - Complementary facts are not contradictions.
    """
    if memories is None:
        memories = recall_all_recent()
    if len(memories) < 2:
        return

    contradictions = memory_heuristics.detect_contradictions(memories)
    invalidated = 0
    flagged = 0
    prev_flags = load_flag_state()
    new_flags = set()

    for c in contradictions:
        pair = c["pair"]
        if not (0 <= pair[0] < len(memories) and 0 <= pair[1] < len(memories)):
            continue

        resolution = c.get("resolution", "flag_human")

        if resolution == "recency_wins" and c.get("confidence") == "high":
            older_idx = c.get("older_index")
            if older_idx is None or not (0 <= older_idx < len(memories)):
                continue
            older = memories[older_idx]
            newer_id = memories[pair[0] if pair[1] == older_idx else pair[1]]["id"]
            if _is_dry_run(_current_args):
                print(f"DRY RUN: would invalidate older memory {older['id']}")
                print(f"  Rule: {c['rule']}")
                print(f"  Replacement: {newer_id}")
                print(f"  Reason: {c['reason']}")
                continue
            if invalidate_memory(older["id"], reason=f"superseded: {c['reason']}"):
                memory_heuristics.audit_log(
                    operation="invalidate", memory_id=older["id"],
                    rule=c["rule"], confidence="high",
                    reason=c["reason"], replacement_id=newer_id,
                )
                invalidated += 1
        else:
            # flag_human — report once per pair (fingerprint state file)
            pair_contents = [memories[pair[0]]["content"], memories[pair[1]]["content"]]
            fp = flag_fingerprint(pair_contents)
            if fp in prev_flags:
                continue
            new_flags.add(fp)
            flagged += 1
            issues.append(Issue(
                code="CONFLICT_FLAG_HUMAN", severity="warning",
                message=f"Contradiction scan: {c['type']} needs review — "
                        f"[{pair_contents[0][:60]}] vs [{pair_contents[1][:60]}] ({c['reason']})",
                context={"rule": c["rule"], "resolution": resolution},
            ))

    if invalidated > 0:
        issues.append(Issue(
            code="STATE_CHANGE_HIGH_CONFIDENCE", severity="info",
            message=f"Contradiction scan: {invalidated} stale state-change memories "
                    f"invalidated (recency-wins, high confidence only)",
            auto_resolvable=True,
        ))
    if new_flags:
        save_flag_state(prev_flags | new_flags)
    if flagged and not invalidated:
        pass  # flag issues already appended above


def load_flag_state():
    """Load previously-reported stable-conflict fingerprints (cross-run dedup)."""
    try:
        return set(json.loads(FLAG_STATE_FILE.read_text()))
    except Exception:
        return set()


def save_flag_state(flags):
    """Persist stable-conflict fingerprints so unresolved flags report once, not daily."""
    with contextlib.suppress(Exception):
        _atomic_write_text(FLAG_STATE_FILE, json.dumps(sorted(flags)))


def flag_fingerprint(pair_contents):
    """Stable fingerprint for a flagged pair (order-independent, content-based)."""
    a, b = sorted(pair_contents)
    import hashlib
    return hashlib.sha256(f"{a}||{b}".encode()).hexdigest()


def _restore_memory(memory_id: str):
    """Restore a previously invalidated memory (v2.4).

    Re-validates a memory by PATCHing state back to 'valid'.
    Reads the audit log to find the before_state for verification.
    """
    audit_entries = []
    if memory_records is not None:
        audit_entries = memory_records.read_audit_log(limit=10, action="invalidate")
        matching = [e for e in audit_entries if e.get("memory_id", "").startswith(memory_id)]
        if matching:
            entry = matching[-1]  # most recent
            print(f"Found audit entry: {entry.get('timestamp')} — {entry.get('reason')}")
            print(f"  Before state: {entry.get('before_state')}")

    try:
        http("PATCH", f"/v1/default/banks/{BANK}/memories/{memory_id}", timeout=120,
             body={"state": "valid", "reason": "restored from audit log"})
        print(f"Memory {memory_id[:12]}... restored to valid state.")
        if memory_records is not None:
            memory_records.append_audit_log(memory_records.AuditEntry(
                timestamp=time.strftime(ISO_TIMESTAMP_FMT, time.gmtime()),
                action="restore",
                memory_id=memory_id,
                after_state={"state": "valid"},
                reason="manual restore from audit log",
            ))
    except Exception as e:
        print(f"Restore failed: {type(e).__name__}: {e}")


def _print_audit_log():
    """Print recent audit log entries (v2.4)."""
    if memory_records is None:
        print("memory_records module not available — no audit log.")
        return
    entries = memory_records.read_audit_log(limit=50)
    if not entries:
        print("No audit log entries found.")
        return
    print(f"Recent audit log ({len(entries)} entries):\n")
    for e in entries:
        print(f"  {e.get('timestamp', '?')} | {e.get('action', '?'):12s} | "
              f"{e.get('memory_id', '')[:12]}... | {e.get('reason', '')[:60]}")


def _trigger_consolidation(args, issues):
    """Step 1: Trigger consolidation, return operation_id or None.

    v3.0.1: Skip when pending_consolidation == 0 — nothing to consolidate,
    and the POST triggers a no-op that still needs polling.
    """
    if _is_dry_run(args):
        issues.append(Issue(code="L2_CONSOLIDATION_PENDING", severity="info",
                             message="[dry-run] Consolidation skipped"))
        return None
    try:
        # Check if there's anything pending first
        _, stats = http("GET", f"/v1/default/banks/{BANK}/stats", timeout=30)
        pending = stats.get("pending_consolidation", 0)
        if pending == 0:
            # Nothing to consolidate — skip the poll loop entirely
            return None

        status, resp = http("POST", f"/v1/default/banks/{BANK}/consolidate", timeout=120, body={})
        op_id = resp.get("operation_id")
        if not op_id:
            issues.append(Issue(
                code="L2_CONSOLIDATION_NO_OP_ID", severity="warning",
                message=f"Consolidate returned HTTP {status} but no operation_id",
                context={"response": str(resp)[:120]},
            ))
        return op_id
    except Exception as e:
        issues.append(Issue(
            code="L2_CONSOLIDATION_TRIGGER_FAILED", severity="warning",
            message=f"Consolidate trigger failed: {type(e).__name__}",
        ))
        return None


def _poll_consolidation(op_id, issues):
    """Step 2: Poll consolidation operation until terminal state or timeout."""
    deadline = time.time() + POLL_DEADLINE
    while time.time() < deadline:
        try:
            _, op = http("GET", f"/v1/default/banks/{BANK}/operations/{op_id}", timeout=120)
            st = op.get("status", "")
            if st in ("completed", "failed", "error", "cancelled"):
                return op
        except Exception as e:
            issues.append(Issue(
                code="L2_OPERATION_POLL_FAILED", severity="warning",
                message=f"Operation poll failed: {type(e).__name__}",
            ))
            return None
        time.sleep(POLL_INTERVAL)
    return None


def _run_consolidation(args, issues):
    """Steps 1-2: Trigger consolidation and poll for completion."""
    op_id = _trigger_consolidation(args, issues)
    if not op_id:
        return None

    final = _poll_consolidation(op_id, issues)
    if final is None:
        issues.append(Issue(
            code="L2_CONSOLIDATION_TIMEOUT", severity="warning",
            message=f"Consolidation op {op_id[:8]} did not finish within {POLL_DEADLINE}s",
        ))
    elif final.get("status") != "completed":
        issues.append(Issue(
            code="L2_CONSOLIDATION_BAD_STATUS", severity="warning",
            message=f"Consolidation op {op_id[:8]} ended with status={final.get('status')}",
        ))
    return final


def _run_smoke_tests(args, issues, final):
    """Steps 2b + 2c: Recall and retain smoke-tests."""
    # --- Recall smoke-test (consolidation over-prune check) ----------
    if final is not None and final.get("status") == "completed":
        try:
            _, rec = http("POST", f"/v1/default/banks/{BANK}/memories/recall", timeout=120,
                          body={"query": "David's preferred output language and document summary style"})
            hits = rec.get("results") or rec.get("memories") or rec.get("items") or []
            if not hits:
                issues.append(Issue(
                    code="L2_RECALL_SMOKE_TEST_EMPTY", severity="warning",
                    message="Recall smoke-test returned 0 results after consolidation — "
                            "possible over-prune; verify manually",
                ))
        except Exception as e:
            issues.append(Issue(
                code="L2_RECALL_SMOKE_TEST_FAILED", severity="warning",
                message=f"Recall smoke-test failed: {type(e).__name__}",
            ))

    # --- Retain smoke-test (silent write-failure check) -------------
    if not _is_dry_run(args):
        try:
            _, ret = http("POST", f"/v1/default/banks/{BANK}/memories", timeout=120,
                          body={"items": [{
                              "content": f"daily memory optimization smoke test {time.strftime('%Y-%m-%d')}",
                              "tags": [SMOKE_TEST_TAG],
                          }]})
            usage = (ret.get("usage") or {})
            if not ret.get("success"):
                issues.append(Issue(
                    code="L2_RETAIN_SMOKE_TEST_FAILED", severity="critical",
                    message=f"Retain smoke-test failed: success != true ({str(ret)[:120]})",
                ))
            elif not usage.get("total_tokens"):
                issues.append(Issue(
                    code="L2_RETAIN_SMOKE_TEST_NO_TOKENS", severity="critical",
                    message="Retain smoke-test: success but total_tokens=0 — "
                            "fact extraction not running; check LLM provider/auth",
                ))
        except Exception as e:
            issues.append(Issue(
                code="L2_RETAIN_SMOKE_TEST_FAILED", severity="critical",
                message=f"Retain smoke-test failed: {type(e).__name__}",
            ))


def _check_bank_stats(issues):
    """Step 3: Bank stats + failed_operations trend."""
    try:
        _, stats = http("GET", f"/v1/default/banks/{BANK}/stats", timeout=120)
        nodes = stats.get("total_nodes", 0)
        docs = stats.get("total_documents", 0)
        if nodes <= 0:
            issues.append(Issue(
                code="L2_BANK_STATS_SUSPICIOUS", severity="warning",
                message=f"Bank '{BANK}' stats suspicious",
                context={"total_nodes": nodes, "total_documents": docs},
            ))
        failed = stats.get("failed_operations", 0)
        prev_failed = 0
        if STATE_FILE.exists():
            with contextlib.suppress(Exception):
                prev_failed = int(json.loads(STATE_FILE.read_text()).get("failed_operations", 0))
        if failed >= FAILED_OPS_WARN or failed > prev_failed:
            issues.append(Issue(
                code="L2_FAILED_OPERATIONS_INCREASED", severity="warning",
                message="Hindsight failed operations increased — writes failing silently; check backlog",
                context={"current": failed, "previous": prev_failed},
            ))
        _atomic_write_text(STATE_FILE, json.dumps({"failed_operations": failed}))
    except Exception as e:
        issues.append(Issue(
            code="L2_BANK_STATS_FETCH_FAILED", severity="warning",
            message=f"Bank stats fetch failed: {type(e).__name__}",
        ))


def _run_heuristic_passes(args, issues):
    """Steps 5-10: self-pollution cleanup + heuristic dedup + contradiction scan.

    Processing order (v3.0 spec section 10):
      5. Remove expired smoke-test records
      6. Remove known meta-maintenance records
      7. Run exact and strong heuristic dedup
      8. Re-fetch or filter invalidated records
      9. Run structured contradiction detection
     10. Apply only high-confidence recency_wins; report the rest
    """
    dry = _is_dry_run(args)
    try:
        # 5. Expired smoke-test records
        if not dry:
            cleanup_smoke_tests(issues)

        # 6. Meta-maintenance records + shared recall set
        memories = recall_all_recent()
        if memories:
            if not dry:
                invalidate_meta_memories(memories, issues)
            # 8. Filter invalidated records before the scans
            memories = [m for m in memories if not META_MEMORY_RE.search(m["content"])]

        # 7. Exact + strong heuristic dedup
        heuristic_dedup_pass(issues, memories=memories)

        # 9-10. Structured contradiction detection
        heuristic_contradiction_scan(issues, memories=memories)
    except Exception as e:
        issues.append(Issue(
            code="HEURISTIC_PASS_FAILED", severity="warning",
            message=f"Heuristic dedup/contradiction passes failed: {type(e).__name__}: {e}",
        ))


def _get_hindsight_version():
    """Fetch Hindsight API version as a comparable tuple."""
    _, ver = http("GET", "/version", timeout=30)
    api_version = ver.get("api_version", "0.0.0")
    version_parts = tuple(int(x) for x in api_version.split(".")[:3] if x.isdigit())
    return api_version, version_parts


def _detect_stale_pages(pages, version_parts, api_version):
    """Detect stale knowledge pages using version-appropriate strategy.

    For Hindsight >= 0.9.1, tree is_stale is scope-aware — use it directly.
    For older versions with small KBs, fall back to the mental-model workaround.
    """
    use_tree_stale_directly = version_parts >= (0, 9, 1)

    if use_tree_stale_directly or len(pages) > KP_EXACT_CHECK_MAX:
        return [n for n in pages if n.get("is_stale")], use_tree_stale_directly

    # Old workaround: query each page's mental model
    stale = []
    for n in pages:
        mm = n.get("mental_model_id")
        if not mm:
            continue
        with contextlib.suppress(Exception):
            _, m = http("GET", f"/v1/default/banks/{BANK}/mental-models/{mm}", timeout=120)
            if m.get("is_stale"):
                stale.append(n)
    return stale, use_tree_stale_directly


def _check_knowledge_pages(issues):
    """Step 11: Knowledge Pages health (Hindsight >= 0.9 only)."""
    try:
        api_version, version_parts = _get_hindsight_version()

        _, tree = http("GET", f"/v1/default/banks/{BANK}/knowledge-base/tree", timeout=120)
        pages = [n for n in walk_tree(tree.get("roots")) if n.get("kind") == "page"]
        if not pages:
            return

        stale, use_tree_stale_directly = _detect_stale_pages(pages, version_parts, api_version)
        if stale and len(stale) / len(pages) > KP_STALE_RATIO_WARN:
            issues.append(Issue(
                code="KP_PAGES_STALE", severity="warning",
                message=f"Knowledge Pages: {len(stale)}/{len(pages)} pages stale "
                        f"(v{api_version}, {'scope-aware' if use_tree_stale_directly else 'approximate'} signal) "
                        f"(e.g. {', '.join(n['name'] for n in stale[:3])}) — pages falling behind consolidation",
            ))
    except urllib.error.HTTPError:
        pass  # <0.9 or KB disabled — not an error
    except Exception as e:
        issues.append(Issue(
            code="KP_CHECK_FAILED", severity="warning",
            message=f"Knowledge Pages check failed: {type(e).__name__}",
        ))


# --- L3 stale-page lint trigger (v3.1) -----------------------------------

L3_STALE_TRIGGER = 5            # run lint when >= this many active pages are stale
LINT_TIMEOUT_S = 300            # five-minute timeout on the lint pass
LINT_SAMPLE_ISSUES = 3          # max representative issues reported
LINT_ISSUE_KEYS = ("error", "warning", "info")   # lint JSON severity keys

# Frontmatter `updated:` value patterns (stdlib-only parsing — no yaml dep)
_FM_UPDATED_RE = re.compile(
    r"^updated:\s*[\"']?(\d{4}-\d{2}-\d{2})"
    r"(?:[T ](\d{2}:\d{2}(?::\d{2})?))?",
    re.MULTILINE,
)


def _parse_frontmatter_updated(text):
    """Return epoch seconds for frontmatter `updated`, or None when absent/invalid.

    Accepts date-only (YYYY-MM-DD) or datetime (ISO `T` or space separator)
    values. Timezone offsets are ignored (local-time assumption) — staleness
    of 90+ days is insensitive to sub-day precision.
    """
    m = _FM_UPDATED_RE.search(text)
    if not m:
        return None
    date_s, time_s = m.group(1), m.group(2)
    try:
        y, mo, d = (int(x) for x in date_s.split("-"))
        hh = mm = ss = 0
        if time_s:
            parts = time_s.split(":")
            hh, mm = int(parts[0]), int(parts[1])
            ss = int(parts[2]) if len(parts) > 2 else 0
        return time.mktime((y, mo, d, hh, mm, ss, 0, 0, -1))
    except (ValueError, OverflowError):
        return None


def _is_index_page(path):
    """Index pages (index.md, INDEX.md) don't count toward staleness."""
    return path.name.lower() == "index.md"


def _collect_active_pages(wiki_dir):
    """Active (non-archived, non-index) markdown pages under wiki_dir."""
    return [
        p for p in wiki_dir.rglob("*.md")
        if "_archive" not in p.parts and not _is_index_page(p)
    ]


def _page_age_days(path, now=None):
    """Page age in days: frontmatter `updated` when valid, else file mtime."""
    now = time.time() if now is None else now
    try:
        fm_ts = _parse_frontmatter_updated(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        fm_ts = None
    ts = fm_ts if fm_ts is not None and fm_ts <= now + 86400 else path.stat().st_mtime
    return max(0.0, (now - ts) / 86400.0)


def _find_stale_pages(wiki_dir):
    """Active pages strictly older than KB_STALE_DAYS days."""
    now = time.time()
    pages = _collect_active_pages(wiki_dir)
    stale = [p for p in pages if _page_age_days(p, now) > KB_STALE_DAYS]
    return stale, pages


def _build_lint_commands(wiki_dir):
    """Lint command candidates: CLI first, then script-path fallback.

    The llmwiki wrapper script lives at ~/.hermes/scripts/llmwiki. The PyPI
    package (if installed) provides a `llmbase` CLI; our wrapper bridges the
    `lint --wiki-dir --json` interface to it.
    """
    # 1. llmwiki on PATH (symlinked wrapper or installed package)
    # 2. python3 <scripts_dir>/llmwiki (direct script execution)
    scripts_dir = HERMES_HOME / "scripts" / "llmwiki"
    return [
        ["llmwiki", "lint", "--wiki-dir", str(wiki_dir), "--json"],
        [sys.executable or "python3", str(scripts_dir),
         "lint", "--wiki-dir", str(wiki_dir), "--json"],
    ]


def _run_lint_command(wiki_dir):
    """Run one lint pass (CLI, then module fallback). Returns (proc, cmd_kind).

    proc is a CompletedProcess, or None when the CLI is unavailable.
    cmd_kind is "cli" or "module" (for reporting), or None with proc=None.
    """
    import shutil
    import subprocess
    for cmd in _build_lint_commands(wiki_dir):
        try:
            if cmd[0] != sys.executable and not shutil.which(cmd[0]):
                continue
            proc = subprocess.run(  # noqa: S603 (fixed argv, no shell)
                cmd, capture_output=True, text=True, timeout=LINT_TIMEOUT_S,
            )
            return proc, "cli" if cmd[0] == "llmwiki" else "module"
        except subprocess.TimeoutExpired:
            return "timeout", "timeout"
        except OSError:
            continue
    return None, None


def _extract_lint_counts(payload):
    """Best-effort lint counters from arbitrary JSON shapes.

    Returns (pages_scanned, counts dict {error, warning, info}).
    """
    counts = {k: 0 for k in LINT_ISSUE_KEYS}
    pages_scanned = None
    if not isinstance(payload, dict):
        return pages_scanned, counts

    # Pages scanned: common key names, else count of a pages/files array
    for key in ("pages_scanned", "pages", "scanned", "total_pages", "files_scanned"):
        v = payload.get(key)
        if isinstance(v, int):
            pages_scanned = v
            break
    if pages_scanned is None:
        for key in ("pages", "files", "results"):
            v = payload.get(key)
            if isinstance(v, list):
                pages_scanned = len(v)
                break

    # Counts: top-level ints, nested summary dicts, or issue lists
    for sev in LINT_ISSUE_KEYS:
        v = payload.get(sev)
        if isinstance(v, int):
            counts[sev] = v
        elif isinstance(v, list):
            counts[sev] = len(v)
    for key in ("summary", "counts", "stats", "totals"):
        sub = payload.get(key)
        if isinstance(sub, dict):
            for sev in LINT_ISSUE_KEYS:
                v = sub.get(sev)
                if isinstance(v, int):
                    counts[sev] = max(counts[sev], v)
                elif isinstance(v, list):
                    counts[sev] = max(counts[sev], len(v))
    issues_list = None
    for key in ("issues", "problems", "findings", "violations"):
        v = payload.get(key)
        if isinstance(v, list) and issues_list is None:
            issues_list = v
    if issues_list is not None:
        for item in issues_list:
            sev = item.get("severity") or item.get("level") if isinstance(item, dict) else None
            if sev in counts:
                counts[sev] += 1
    return pages_scanned, counts


def _extract_lint_issues(payload, limit=LINT_SAMPLE_ISSUES):
    """Up to `limit` representative issue strings from lint JSON."""
    if not isinstance(payload, dict):
        return []
    for key in ("issues", "problems", "findings", "violations"):
        v = payload.get(key)
        if isinstance(v, list):
            out = []
            for item in v[:limit]:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    sev = item.get("severity") or item.get("level") or "issue"
                    msg = (item.get("message") or item.get("description")
                           or item.get("rule") or item.get("title") or "unspecified issue")
                    page = (item.get("page") or item.get("file")
                            or item.get("path") or item.get("title"))
                    out.append(f"{msg} [{sev}]" if page is None
                               else f"{page}: {msg} [{sev}]")
            return out
    return []


def _report_lint_result(wiki_dir, stale_count, total, issues):
    """Run the lint pass and append report/failure Issues. Never raises."""
    try:
        proc, cmd_kind = _run_lint_command(wiki_dir)
    except Exception as e:  # defensive: lint must never crash maintenance
        issues.append(Issue(
            code="L3_LINT_FAILED", severity="warning",
            message=f"L3 lint pass ({wiki_dir}) failed unexpectedly: {type(e).__name__}",
        ))
        return
    if proc is None:
        issues.append(Issue(
            code="L3_LINT_CLI_UNAVAILABLE", severity="warning",
            message=f"L3 lint trigger ({wiki_dir}): {stale_count} of {total} active pages "
                    f"stale (> {KB_STALE_DAYS} days), but no llmwiki CLI found "
                    f"(llmwiki / python3 -m llmwiki) — install to enable lint",
        ))
        return
    if proc == "timeout":
        issues.append(Issue(
            code="L3_LINT_TIMEOUT", severity="warning",
            message=f"L3 lint pass ({wiki_dir}) timed out after {LINT_TIMEOUT_S}s "
                    f"({stale_count} of {total} active pages stale) — unresolved",
        ))
        return
    stdout, returncode = proc.stdout or "", proc.returncode
    try:
        payload = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError:
        payload = None
    if payload is None:
        issues.append(Issue(
            code="L3_LINT_MALFORMED_OUTPUT", severity="warning",
            message=f"L3 lint pass ({wiki_dir}, {cmd_kind}) produced "
                    f"{'no JSON output' if not stdout.strip() else 'malformed JSON'} "
                    f"(exit {returncode}) — unresolved",
        ))
        return
    pages_scanned, counts = _extract_lint_counts(payload)
    sample = _extract_lint_issues(payload)
    ctx = {f"{k}_count": v for k, v in counts.items()}
    if pages_scanned is not None:
        ctx["pages_scanned"] = pages_scanned
    detail = f"{counts['error']} errors, {counts['warning']} warnings, {counts['info']} info"
    if pages_scanned is not None:
        detail = f"{pages_scanned} pages scanned — " + detail
    if sample:
        detail += "; e.g. " + "; ".join(sample)
    issues.append(Issue(
        code="L3_LINT_REPORT", severity="warning",
        message=f"L3 lint pass ({wiki_dir}): {stale_count} of {total} active pages stale "
                f"(> {KB_STALE_DAYS} days) — {detail}",
        context=ctx,
    ))
    if returncode != 0:
        issues.append(Issue(
            code="L3_LINT_NONZERO_EXIT", severity="warning",
            message=f"L3 lint pass ({wiki_dir}, {cmd_kind}) exited {returncode} "
                    f"(results above may be incomplete) — unresolved",
        ))


def _check_l3_wiki(issues):
    """Step 12: L3 wiki lint trigger — full lint pass on >= 5 stale active pages."""
    try:
        for wiki_dir in (KB_DIR, WIKI_DIR):
            if not wiki_dir.exists():
                continue
            stale, pages = _find_stale_pages(wiki_dir)
            if len(stale) >= L3_STALE_TRIGGER:
                _report_lint_result(wiki_dir, len(stale), len(pages), issues)
    except Exception as e:
        issues.append(Issue(
            code="L3_WIKI_CHECK_FAILED", severity="warning",
            message=f"L3 wiki check failed: {type(e).__name__}",
        ))


def _check_local_memory(args, issues):
    """Step 13: Local memory capacity check."""
    try:
        if MEM_FILE.exists():
            content = MEM_FILE.read_text(encoding="utf-8")
            n = len(content)
            pct = n / MEM_CAP
            if pct >= OFFLOAD_AT:
                _handle_memory_offload(args, issues, n, pct)
            elif pct >= MEM_WARN:
                issues.append(Issue(
                    code="L1_MEMORY_NEAR_CAPACITY", severity="warning",
                    message=f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — approaching capacity",
                ))
        if USER_FILE.exists():
            content = USER_FILE.read_text(encoding="utf-8")
            n = len(content)
            pct = n / USER_CAP
            if pct >= MEM_WARN:
                issues.append(Issue(
                    code="L1_USER_NEAR_CAPACITY", severity="warning",
                    message=f"USER.md at {n}/{USER_CAP} chars ({int(pct*100)}%) — prune or offload to Hindsight",
                ))
    except Exception as e:
        issues.append(Issue(
            code="L1_CHECK_FAILED", severity="warning",
            message=f"Local memory check failed: {type(e).__name__}",
        ))


def _handle_memory_offload(args, issues, n, pct):
    """Handle memory offload when MEMORY.md is over capacity."""
    if memory_offload is None:
        issues.append(Issue(
            code="L1_CAPACITY_EXCEEDED", severity="warning",
            message=f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                    "memory_offload.py not found next to this script; offload skipped",
        ))
    elif not _is_dry_run(args):
        try:
            used_before, _ = memory_offload.get_memory_usage()
            memory_offload.main()
            used_after, _ = memory_offload.get_memory_usage()
            if used_after < used_before:
                issues.append(Issue(
                    code="L1_OFFLOAD_DONE", severity="info",
                    message=f"MEMORY.md was at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                            f"offloaded to Hindsight, now {used_after}/{MEM_CAP} chars",
                ))
            else:
                issues.append(Issue(
                    code="L1_OFFLOAD_NO_PROGRESS", severity="warning",
                    message=f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                            "offload ran but no entries moved (all essential?). Prune manually.",
                ))
        except Exception as e:
            issues.append(Issue(
                code="L1_OFFLOAD_FAILED", severity="warning",
                message=f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                        f"offload failed: {type(e).__name__}",
            ))
    else:
        issues.append(Issue(
            code="L1_CAPACITY_EXCEEDED", severity="info",
            message=f"[dry-run] MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                    "offload would run here",
        ))


def _build_telegram_text(unresolved, resolved):
    """Build the Telegram notification HTML for unresolved issues."""
    timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    tg_text = (
        f"<b>🧠 Daily Memory Optimization</b>\n"
        f"<i>{timestamp}</i>\n\n"
        f"<b>{len(unresolved)} issue(s) need your attention "
        f"(rule-based auto-resolve attempted, {len(resolved)} resolved):</b>\n\n"
    )
    for u in unresolved:
        tg_text += f"  • {_escape_html(render_issue(u) if isinstance(u, Issue) else str(u))}\n"
    return tg_text


def _get_run_mode(args):
    """Determine the current run mode string for output."""
    if _is_dry_run(args):
        return "dry-run"
    if getattr(args, "apply", False):
        return "apply"
    if ALLOW_DESTRUCTIVE:
        return "destructive"
    return "safe"


def _output_results(args, issues):
    """Step 15: Output: rule-based auto-resolve → Telegram fallback."""
    if not issues:
        return  # empty stdout -> cron stays silent

    # Step 14: attempt deterministic rule-based resolution
    resolved, unresolved = try_resolve_issues_with_rules(issues)

    # Build output report
    if resolved:
        print("**🧠 Daily Memory Optimization — issues auto-resolved (rule-based)**\n")
        for r in resolved:
            print(f"  ✅ {r}")

    if not unresolved:
        return

    # Notify user via Telegram DM for remaining issues
    tg_text = _build_telegram_text(unresolved, resolved)
    tg_sent = send_telegram_notification(tg_text)
    delivery_status = (
        "Telegram DM sent" if tg_sent
        else "Telegram DM delivery FAILED (check bot token / chat ID)"
    )

    # Also print to stdout (for cron log visibility)
    mode = _get_run_mode(args)
    print(f"\n**🧠 Daily Memory Optimization — unresolved issues ({delivery_status}, mode={mode})**\n")
    for u in unresolved:
        print(f"  ⚠️ {render_issue(u) if isinstance(u, Issue) else str(u)}")


# Module-level args reference for helper functions (set in main()).
_current_args = argparse.Namespace(dry_run=False)


def main():
    global ALLOW_DESTRUCTIVE, _current_args

    parser = argparse.ArgumentParser(description="Daily memory optimization (v3.0 rule-based)")
    parser.add_argument("--allow-destructive", action="store_true",
                        help="Enable auto-mutation (invalidate, config_tune). "
                             "Default: disabled — rules are advisor only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report issues without taking any action. "
                             "Also active via MEMORY_HEURISTICS_DRY_RUN=1.")
    parser.add_argument("--apply", action="store_true",
                        help="Apply all recommended actions (implies --allow-destructive). "
                             "Use --dry-run first to preview.")
    parser.add_argument("--restore", type=str, metavar="AUDIT_ID",
                        help="Restore a previously invalidated memory from the audit log. "
                             "Pass the memory_id from the audit log entry.")
    parser.add_argument("--audit-log", action="store_true",
                        help="Print recent audit log entries and exit.")
    args = parser.parse_args()

    _current_args = args

    # --apply implies --allow-destructive
    if args.apply:
        ALLOW_DESTRUCTIVE = True
    else:
        ALLOW_DESTRUCTIVE = args.allow_destructive

    # --restore: re-validate a previously invalidated memory
    if args.restore:
        _restore_memory(args.restore)
        return

    # --audit-log: print recent entries and exit
    if args.audit_log:
        _print_audit_log()
        return

    issues = []

    # --- 1-2. Trigger consolidation + poll -------------------------------
    final = _run_consolidation(args, issues)

    # --- 2b + 2c. Smoke tests --------------------------------------------
    _run_smoke_tests(args, issues, final)

    # --- 3. Bank stats + failed_operations trend --------------------------
    _check_bank_stats(issues)

    # --- 5-10. Self-pollution cleanup + heuristic passes ------------------
    if _time_remaining():
        _run_heuristic_passes(args, issues)
    else:
        issues.append(Issue(
            code="SCRIPT_TIME_BUDGET_EXCEEDED", severity="warning",
            message=f"Script time budget ({SCRIPT_TIME_BUDGET}s) exceeded after consolidation+smoke — heuristic passes skipped",
        ))

    # --- 11. Knowledge Pages health (Hindsight >= 0.9 only) ---------------
    if _time_remaining():
        _check_knowledge_pages(issues)

    # --- 12. L3 wiki lint-lite -------------------------------------------
    if _time_remaining():
        _check_l3_wiki(issues)

    # --- 13. Local memory capacity check ---------------------------------
    _check_local_memory(args, issues)

    # --- 14-15. Output: rule-based auto-resolve → Telegram fallback -------
    _output_results(args, issues)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Script crash = real error, always report
        print(f"**🧠 Daily Memory Optimization — script crashed**\n\n  • {type(e).__name__}: {e}")
    sys.exit(0)
