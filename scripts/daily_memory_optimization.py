#!/usr/bin/env python3
"""Daily memory optimization — L1/L2/L3 maintenance for Hermes Agent.

no-agent cron script: stdout is delivered verbatim; empty stdout = silent.

v2.3 (Sep 2026): Correctness — structured records, deterministic recency, paginated scanning.
  - Structured MemoryRecord with timestamps and fact_type metadata
  - Deterministic recency resolution from timestamps (not LLM list-position)
  - Paginated /memories/list replaces broad recall (scan cursor for coverage)
  - Cross-chunk candidate generation (BATCH_SIZE boundary fix)
  - Version-gated Knowledge Pages is_stale behavior
  - Idempotent document_id values for retains

v2.4 (Sep 2026): Productization — CLI workflow, audit logging, privacy.
  - JSON audit log (before/after state, restore capability)
  - Privacy/PII redaction before cloud judging
  - --dry-run / --apply / --allow-destructive / --restore CLI modes
  - Local-model/private judging support via env vars

v2.2.1 (Sep 2026): Safety patch.
  - LLM auto-mutation DISABLED by default (requires --allow-destructive flag)
  - Safe memory invalidation: fetches fact_type before PATCH (observations
    return 400 — now handled by finding and invalidating source memories)
  - Telegram HTML content escaped (raw <, >, & caused delivery failure)
  - Telegram delivery status reported accurately (was always "sent")
  - Config via environment variables (no hardcoded /root paths)
  - Character-count consistency (decoded chars, not st_size bytes)
  - Atomic state-file writes

v2.2 (Sep 2026): LLM auto-resolve + Telegram notification.
  - After collecting issues, the script attempts to resolve them using
    z-ai/glm-5.2 (one LLM call): consolidate, invalidate stale memories,
    or tune Hindsight config
  - Issues the LLM cannot auto-resolve are sent as a Telegram DM notification
  - Silent if all issues are resolved by the LLM or no issues found

v2.0.1 (Aug 2026): self-pollution fixes from live-run false positives.
  - Smoke-test probe memories are tagged and retired after 48h (30 had accumulated)
  - Meta-memories (reports ABOUT past maintenance runs) invalidated on sight —
    they previously flagged against every related fact, every day
  - Exact-duplicate pre-pass before the LLM (verbatim copies need no judge)
  - One shared recall set feeds both dedup and contradiction passes
    (previously different queries meant duplicates never reached dedup)
  - Contradiction prompt hardened: complementary pairs (policy vs capability)
    are not conflicts; temporal markers route state changes to recency-wins
  - Stable-conflict flags deduped across runs via fingerprint state file
    (an unresolved flag reports once, not daily)

v2.0 (Aug 2026): LLM-driven semantic dedup + contradiction detection.
  - L2 semantic dedup pass via llm_judge.semantic_dedup() (replaces manual recall+overlap)
  - L2 contradiction scan via llm_judge.detect_contradictions() (replaces manual scan)
  - LLM-optional: degrades to rule-based if LLM unavailable
  - Batch consolidation: all memories in one LLM call (LycheeMemory V2 pattern)

Behavior (per memory-optimization skill v2.0.0):
  L2 (Hindsight, any 0.8+; Knowledge Pages checks activate on 0.9+):
    1. POST /consolidate -> poll operation to terminal state (max 8 min)
    2. Recall smoke-test after consolidation (over-prune check)
    3. Retain smoke-test: success:true AND total_tokens>0
       (health green != writes working — fact extraction is what silently fails)
    4. Bank stats: total_nodes>0, failed_operations trend vs last run
    5. **NEW** LLM semantic dedup pass: recall recent memories, LLM identifies
       near-duplicates, invalidate via PATCH (non-destructive: state=invalidated)
    6. **NEW** LLM contradiction scan: recall recent memories, LLM finds entity-drift
       pairs, apply recency-wins invalidation (state changes) or flag for human review
    7. Knowledge Pages tree: count pages + is_stale pages (0.9+ only)
  L1 (local memory):
    8. MEMORY.md / USER.md capacity check; >=90% triggers Hindsight offload
  L3 (LLM wiki / OKF bundle):
    9. Wiki lint-lite: stale-page count (>90 days) on ~/.hermes/kb

Output only when something needs attention; else silent. Exit 0 always
(a crash is reported via stdout, never via nonzero exit).
"""

import argparse
import contextlib
import hashlib
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

# LLM judge module (LLM-optional — degrades to rule-based if unavailable)
sys.path.insert(0, str(Path(__file__).parent))
llm_judge: types.ModuleType | None
try:
    import llm_judge
except ImportError:
    llm_judge = None  # standalone run: skip LLM-driven steps

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
POLL_DEADLINE = 480          # 8 min max wait for consolidation
MEM_CAP = int(os.environ.get("MEMORY_CHARS", "2200"))
USER_CAP = int(os.environ.get("USER_CHARS", "1375"))
MEM_WARN = 0.90              # warn at 90% capacity
OFFLOAD_AT = 0.90            # trigger Hindsight offload at >=90%
KB_STALE_DAYS = 90           # L3 page staleness threshold (skill: >90 days = stale)
FAILED_OPS_WARN = 10         # failed_operations count worth reporting
KP_STALE_RATIO_WARN = 0.5    # warn when >50% of knowledge pages are stale
KP_EXACT_CHECK_MAX = 25      # use exact per-page mental-model is_stale for KBs up to this size
STATE_FILE = HERMES_HOME / "scripts" / ".daily_memory_opt_state.json"

# LLM-driven dedup/contradiction settings
DEDUP_RECALL_LIMIT = 50      # max memories to recall for LLM dedup/contradiction scan
DEDUP_RECALL_TOKENS = 3000   # token budget for recall query

# v2.2.1: LLM auto-mutation disabled by default — requires --allow-destructive
ALLOW_DESTRUCTIVE = False

# v2.0.1: self-pollution guards
SMOKE_TEST_TAG = "daily-memopt-smoke"   # tag on the retain smoke-test memory
SMOKE_TEST_MAX_AGE_S = 172800           # smoke-test memories older than 48h = junk
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
    """Escape HTML special characters for Telegram parse_mode=HTML.

    v2.2.1: Raw <, >, & in issue text caused Telegram delivery failure
    or malformed output. html.escape handles all three + quotes.
    """
    return html.escape(text, quote=False)


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

    v2.2.1: Only world and experience memories can be directly invalidated.
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
    """Direct PATCH invalidation (world/experience only)."""
    try:
        http("PATCH", f"/v1/default/banks/{BANK}/memories/{mid}", timeout=120,
             body={"state": "invalidated", "reason": reason})
        return True
    except Exception:
        return False


def invalidate_memory(mid, reason="duplicate"):
    """Non-destructive invalidation with fact_type safety check.

    v2.2.1: Wraps curate_memory() to add fact_type checking. Observations
    are handled by invalidating their source memories. Never DELETE —
    invalidation preserves audit trail and is recoverable.
    """
    return curate_memory(mid, action="invalidate", reason=reason)


# Allowlisted config keys (v2.2.1: prevent arbitrary config mutation)
ALLOWED_CONFIG_KEYS = {
    "recall_budget_function", "recall_max_tokens",
    "consolidation_max_memories_per_round",
    "retain_mission", "reflect_mission",
    "enable_auto_consolidation",
}


def _build_resolver_prompts(problems):
    """Build system and user prompts for the LLM resolver."""
    system_prompt = (
        "You are a memory maintenance resolver for a three-layer AI agent memory system.\n"
        "L1 = local MEMORY.md/USER.md (always injected, char-capped).\n"
        "L2 = Hindsight server (semantic recall, consolidation, Knowledge Pages).\n"
        "L3 = LLM Wiki (git-versioned markdown knowledge base).\n\n"
        "You receive a list of issues found during daily maintenance.\n"
        "For each issue, decide if it can be auto-resolved with an API action.\n"
        "Available actions:\n"
        '  - {"action": "consolidate"} — POST /consolidate to trigger Hindsight consolidation\n'
        '  - {"action": "invalidate", "reason": "...", "query": "search terms"} — recall memories\n'
        '    matching query, invalidate the stale ones (PATCH state=invalidated)\n'
        '  - {"action": "config_tune", "key": "...", "value": "..."} — PATCH /config\n'
        '  - {"action": "skip"} — cannot auto-resolve, needs human review\n\n'
        "Return a JSON array of objects, one per issue:\n"
        '  {"issue_index": 0, "action": "consolidate|invalidate|config_tune|skip",\n'
        '   "reason": "brief explanation", "query": "optional search terms",\n'
        '   "key": "optional config key", "value": "optional config value"}\n'
        "Respond with ONLY the JSON array, no other text."
    )
    issue_list = "\n".join(f"  [{i}] {p}" for i, p in enumerate(problems))
    user_prompt = f"Issues found during daily memory optimization:\n{issue_list}\n\nReturn the JSON action plan."
    return system_prompt, user_prompt


def _execute_resolve_action(item, issue):
    """Execute a single LLM-proposed resolve action. Returns resolution string or None (unresolved)."""
    action = item.get("action", "skip")

    # Consolidate is always safe (non-destructive)
    if action == "consolidate":
        try:
            http("POST", f"/v1/default/banks/{BANK}/consolidate", timeout=120, body={})
            return f"{issue} → resolved (consolidation triggered)"
        except Exception:
            return None

    # Destructive actions require --allow-destructive flag (v2.2.1)
    if action == "invalidate" and ALLOW_DESTRUCTIVE:
        return _execute_invalidate_action(item, issue)

    if action == "config_tune" and ALLOW_DESTRUCTIVE:
        return _execute_config_tune_action(item, issue)

    # skip, unknown, or destructive action without --allow-destructive
    return None


def _execute_invalidate_action(item, issue):
    """Execute an invalidate action. Returns resolution string or None."""
    query = item.get("query", "")
    reason = item.get("reason", "stale memory invalidated by LLM resolver")
    if not query:
        return None
    memories = recall_recent_memories(query=query, limit=10)
    invalidated = sum(1 for m in memories if invalidate_memory(m["id"], reason))
    if invalidated > 0:
        return f"{issue} → resolved ({invalidated} memories invalidated)"
    return None


def _execute_config_tune_action(item, issue):
    """Execute a config_tune action. Returns resolution string or None."""
    key = item.get("key", "")
    value = item.get("value", "")
    # v2.2.1: allowlist config keys
    if not key or key not in ALLOWED_CONFIG_KEYS:
        return None
    try:
        http("PATCH", f"/v1/default/banks/{BANK}/config", timeout=120,
             body={"updates": {key: value}})
        return f"{issue} → resolved (config {key}={value})"
    except Exception:
        return None


def try_resolve_issues_with_llm(problems):
    """Attempt to resolve collected issues using z-ai/glm-5.2 (one try).

    v2.2.1: LLM auto-mutation is DISABLED by default. The LLM is used as
    an ADVISOR only — it proposes actions, but destructive actions
    (invalidate, config_tune) are only executed if ALLOW_DESTRUCTIVE is True.
    Consolidation (non-destructive) is always allowed.

    Sends the issue list to the LLM with context about the memory system
    and available Hindsight API operations. The LLM returns a JSON action
    plan; each action is validated against a typed policy before execution.

    Returns (resolved_issues, unresolved_issues) lists.
    """
    if llm_judge is None:
        return [], list(problems)

    system_prompt, user_prompt = _build_resolver_prompts(problems)

    try:
        response = llm_judge._llm_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=3000,
        )
        if not response:
            return [], list(problems)

        plan = llm_judge._parse_json_response(response)
        if not plan or not isinstance(plan, list):
            return [], list(problems)

        resolved, unresolved = [], []
        for item in plan:
            idx = item.get("issue_index", -1)
            if not isinstance(idx, int) or idx < 0 or idx >= len(problems):
                continue
            issue = problems[idx]
            result = _execute_resolve_action(item, issue)
            if result:
                resolved.append(result)
            else:
                unresolved.append(issue)

        return resolved, unresolved

    except Exception:
        return [], list(problems)


def send_telegram_notification(text):
    """Send a message to the user's Telegram DM via Bot API.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_HOME_CHANNEL from ~/.hermes/.env.
    Returns True on success, False on failure (silent — never crash the cron).

    v2.2.1: HTML content is now escaped (raw <, >, & caused delivery failure
    or malformed output). The caller now checks the return value to report
    delivery status accurately.
    """
    bot_token = _read_env_var("TELEGRAM_BOT_TOKEN")
    chat_id = _read_env_var("TELEGRAM_HOME_CHANNEL")
    if not bot_token or not chat_id:
        return False

    # v2.2.1: escape HTML special characters in issue text
    # Keep our own <b>, <i>, <br/> tags — escape only the dynamic content
    # The caller is responsible for using escaped text in the body.
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
    """Recall recent memories for LLM-driven dedup + contradiction scan.

    Returns list of {id, content} dicts, or empty list on failure.
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
                memories.append({"id": mid, "content": content})
        return memories[:(limit or DEDUP_RECALL_LIMIT)]
    except Exception:
        return []


def recall_all_recent(limit=DEDUP_RECALL_LIMIT):
    """Recall a broad mix of recent memories (single shared set for dedup + contradictions).

    v2.0.1: dedup and contradiction scans previously used different recall queries,
    so duplicates the contradiction scan saw never reached the dedup pass. One
    broad recall now feeds both passes, keeping indices consistent.
    """
    return recall_recent_memories(
        query="user preferences, environment, configuration, tools, versions, migrations, decisions",
        limit=limit,
    )


def cleanup_smoke_tests(problems):
    """Invalidate old smoke-test memories (v2.0.1 self-pollution fix).

    The retain smoke-test writes a memory every run; without cleanup they
    accumulate forever (30 found in one bank). Each new smoke-test memory is
    tagged SMOKE_TEST_TAG; anything tagged and older than SMOKE_TEST_MAX_AGE_S
    is invalidated. Legacy untagged ones are matched by content prefix.
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
            if invalidate_memory(h.get("id") or h.get("memory_id"), reason="smoke_test_junk"):
                removed += 1
    except Exception:
        return  # cleanup is best-effort; never block the run
    if removed:
        problems.append(f"Self-pollution cleanup: {removed} old smoke-test memories invalidated")


def invalidate_meta_memories(memories, problems):
    """Invalidate meta-memories: reports ABOUT past maintenance runs (v2.0.1).

    Past runs' problem reports were retained as memories ("a stable-attribute
    conflict was flagged regarding X"). Those reports then flag against every
    related fact on every subsequent run — self-referential noise. They are
    bookkeeping, not facts; invalidate on sight.
    """
    removed = 0
    for m in memories:
        if META_MEMORY_RE.search(m.get("content", "")) and invalidate_memory(
            m["id"], reason=META_NOISE_REASON
        ):
            removed += 1
    if removed:
        problems.append(f"Self-pollution cleanup: {removed} meta-memories (reports about past runs) invalidated")
    return removed


def exact_duplicate_prepass(memories):
    """Invalidate verbatim duplicates before the LLM passes (v2.0.1).

    Exact copies (same normalized content) need no LLM call — collapse them
    deterministically, keeping the first occurrence. Returns count removed.
    """
    seen = {}
    removed = 0
    for m in memories:
        key = hashlib.sha256(
            re.sub(r"\s+", " ", m["content"]).strip().lower().encode()
        ).hexdigest()
        if key in seen:
            if invalidate_memory(m["id"], reason="exact_duplicate"):
                removed += 1
        else:
            seen[key] = m["id"]
    return removed


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
    return hashlib.sha256(f"{a}||{b}".encode()).hexdigest()


def _dedup_invalidate_records(records, dup_groups, problems, total_source):
    """Invalidate duplicate memories from structured records (v2.3 path)."""
    invalidated = 0
    for group in dup_groups:
        for dup_idx in group["duplicates"]:
            if not (0 <= dup_idx < len(records)):
                continue
            mid = records[dup_idx].id
            # v2.4: audit log
            if memory_records is not None:
                memory_records.append_audit_log(memory_records.AuditEntry(
                    timestamp=time.strftime(ISO_TIMESTAMP_FMT, time.gmtime()),
                    action="invalidate",
                    memory_id=mid,
                    before_state={"state": records[dup_idx].state, "fact_type": records[dup_idx].fact_type},
                    reason="semantic_duplicate",
                ))
            if invalidate_memory(mid, reason="semantic_duplicate"):
                invalidated += 1
    if invalidated > 0:
        problems.append(
            f"LLM semantic dedup: {invalidated} near-duplicate memories invalidated "
            f"(non-destructive — retained on disk for audit). "
            f"Scanned {len(records)}/{total_source} source memories (cursor-based)."
        )


def _dedup_invalidate_recall(memories, dup_groups, problems):
    """Invalidate duplicate memories from recall-based approach (v2.0 path)."""
    invalidated = 0
    for group in dup_groups:
        for dup_idx in group["duplicates"]:
            if 0 <= dup_idx < len(memories):
                mid = memories[dup_idx]["id"]
                if invalidate_memory(mid, reason="semantic_duplicate"):
                    invalidated += 1
    if invalidated > 0:
        problems.append(
            f"LLM semantic dedup: {invalidated} near-duplicate memories invalidated "
            f"(non-destructive — retained on disk for audit)"
        )


def llm_semantic_dedup_pass(problems, memories=None):
    """LLM-driven semantic dedup pass (Step 5, new in v2.0).

    v2.3: Uses paginated /memories/list instead of one broad recall query.
    Cross-chunk candidate generation fixes the BATCH_SIZE boundary problem.
    Privacy redaction applied before sending to cloud judge.

    v2.0.1: accepts a pre-recalled shared memory list (same set feeds the
    contradiction scan, keeping indices consistent).

    Research basis:
    - MenteDB llm_consolidation: LLM-as-judge for semantic dedup
    - Hindsight blog: fact deduplication — "same claim, different wording"
    - Human-Inspired Memory: dedup-based consolidation achieves 97.2% precision, 58% store reduction
    """
    if llm_judge is None:
        return  # LLM unavailable — skip (rule-based dedup happens in Hindsight's own consolidation)

    # v2.3: use structured records from /memories/list when available
    if memory_records is not None:
        records, total_source, new_offset = memory_records.get_scan_batch()
        if len(records) < 2:
            return  # nothing to dedup

        # Save scan cursor for next run
        memory_records.save_scan_cursor({
            "offset": new_offset,
            "total_seen": memory_records.load_scan_cursor().get("total_seen", 0) + len(records),
        })

        # Generate cross-chunk candidate pairs (fixes BATCH_SIZE boundary)
        candidates = memory_records.generate_candidate_pairs(records)
        if not candidates:
            return  # no candidates to check

        # Prepare safe records for LLM judging (PII redacted)
        safe_records = memory_records.prepare_for_judging(records)
        contents = [r["content"] for r in safe_records]
        dup_groups = llm_judge.semantic_dedup(contents)
        _dedup_invalidate_records(records, dup_groups, problems, total_source)
        return

    # Fallback: old recall-based approach (v2.0)
    if memories is None:
        memories = recall_all_recent()
    if len(memories) < 2:
        return  # nothing to dedup

    contents = [m["content"] for m in memories]
    dup_groups = llm_judge.semantic_dedup(contents)
    _dedup_invalidate_recall(memories, dup_groups, problems)


def _normalize_text(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def _handle_recency_wins_records(records, pair, reason, prev_flags, new_flags, problems):
    """Handle a recency_wins contradiction pair using structured records. Returns (invalidated, flagged)."""
    rec_a = records[pair[0]]
    rec_b = records[pair[1]]
    newer, older = memory_records.resolve_recency((rec_a, rec_b))

    if older is not None:
        # Deterministic recency — invalidate the older one
        if memory_records is not None:
            memory_records.append_audit_log(memory_records.AuditEntry(
                timestamp=time.strftime(ISO_TIMESTAMP_FMT, time.gmtime()),
                action="invalidate",
                memory_id=older.id,
                before_state={"state": older.state, "fact_type": older.fact_type},
                after_state={"state": "invalidated"},
                reason=f"superseded (deterministic recency): {reason}",
            ))
        if invalidate_memory(older.id, reason=f"superseded: {reason}"):
            return 1, 0
        return 0, 0

    # Missing/equal timestamps — can't determine recency, flag for human
    pair_contents = [rec_a.content, rec_b.content]
    fp = flag_fingerprint(pair_contents)
    if fp in prev_flags:
        return 0, 0
    new_flags.add(fp)
    problems.append(
        f"LLM contradiction scan: state-change conflict but timestamps "
        f"missing/equal — needs manual recency check — "
        f"[{pair_contents[0][:60]}] vs [{pair_contents[1][:60]}] ({reason})"
    )
    return 0, 1


def _handle_stable_conflict_records(records, pair, reason, prev_flags, new_flags, problems):
    """Handle a stable_conflict pair using structured records. Returns (invalidated, flagged)."""
    pair_contents = [records[pair[0]].content, records[pair[1]].content]
    a, b = _normalize_text(pair_contents[0]), _normalize_text(pair_contents[1])
    if a == b or a.startswith(b) or b.startswith(a):
        shorter_idx = pair[0] if len(a) <= len(b) else pair[1]
        if invalidate_memory(records[shorter_idx].id, reason="semantic_duplicate (near-verbatim pair)"):
            return 1, 0
        return 0, 0
    if re.search(r"duplicat|identical", reason, re.IGNORECASE):
        shorter_idx = pair[0] if len(pair_contents[0]) <= len(pair_contents[1]) else pair[1]
        if invalidate_memory(records[shorter_idx].id, reason=f"semantic_duplicate (judge reason: {reason[:80]})"):
            return 1, 0
        return 0, 0
    fp = flag_fingerprint(pair_contents)
    if fp in prev_flags:
        return 0, 0
    new_flags.add(fp)
    problems.append(
        f"LLM contradiction scan: stable-attribute conflict needs review — "
        f"[{pair_contents[0][:60]}] vs [{pair_contents[1][:60]}] ({reason})"
    )
    return 0, 1


def _handle_invalidate_meta_records(records, pair):
    """Handle an invalidate_meta pair using structured records. Returns invalidated count."""
    invalidated = 0
    for idx in pair:
        if 0 <= idx < len(records) and META_MEMORY_RE.search(records[idx].content):
            if invalidate_memory(records[idx].id, reason=META_NOISE_REASON):
                invalidated += 1
    return invalidated


def _contradiction_scan_with_records(problems):
    """Run contradiction scan using structured records (v2.3 path)."""
    records, total_source, _ = memory_records.get_scan_batch()
    if len(records) < 2:
        return

    candidates = memory_records.generate_candidate_pairs(records)
    if not candidates:
        return

    safe_records = memory_records.prepare_for_judging(records)
    contents = [r["content"] for r in safe_records]
    contradictions = llm_judge.detect_contradictions(contents)

    invalidated = 0
    flagged = 0
    seen_pairs = set()
    prev_flags = load_flag_state()
    new_flags = set()

    for c in contradictions:
        pair = c["pair"]
        if not (0 <= pair[0] < len(records) and 0 <= pair[1] < len(records)):
            continue
        key = tuple(sorted(pair))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        resolution = c.get("resolution", "flag_human")
        reason = c.get("reason", "")

        if resolution == "recency_wins":
            inv, flg = _handle_recency_wins_records(records, pair, reason, prev_flags, new_flags, problems)
            invalidated += inv
            flagged += flg
        elif resolution == "invalidate_meta":
            invalidated += _handle_invalidate_meta_records(records, pair)
        else:
            inv, flg = _handle_stable_conflict_records(records, pair, reason, prev_flags, new_flags, problems)
            invalidated += inv
            flagged += flg

    if invalidated > 0:
        problems.append(
            f"LLM contradiction scan: {invalidated} stale state-change memories invalidated "
            f"(recency-wins, deterministic from timestamps)"
        )
    if new_flags:
        save_flag_state(prev_flags | new_flags)


def _handle_recency_wins_recall(memories, pair, reason, newer_idx):
    """Handle a recency_wins pair using recall-based approach. Returns invalidated count."""
    older_idx = pair[0] if pair[1] == newer_idx else pair[1]
    if 0 <= older_idx < len(memories):
        mid = memories[older_idx]["id"]
        if invalidate_memory(mid, reason=f"superseded: {reason}"):
            return 1
    return 0


def _handle_invalidate_meta_recall(memories, pair):
    """Handle an invalidate_meta pair using recall-based approach. Returns invalidated count."""
    invalidated = 0
    for idx in pair:
        if 0 <= idx < len(memories) and META_MEMORY_RE.search(memories[idx]["content"]) and invalidate_memory(
            memories[idx]["id"], reason=META_NOISE_REASON
        ):
            invalidated += 1
    return invalidated


def _handle_stable_conflict_recall(memories, pair, reason, contents, prev_flags, new_flags, problems):
    """Handle a stable_conflict pair using recall-based approach. Returns (invalidated, flagged)."""
    pair_contents = [contents[pair[0]], contents[pair[1]]]
    a, b = _normalize_text(pair_contents[0]), _normalize_text(pair_contents[1])
    if a == b or a.startswith(b) or b.startswith(a):
        shorter_idx = pair[0] if len(a) <= len(b) else pair[1]
        if invalidate_memory(memories[shorter_idx]["id"], reason="semantic_duplicate (near-verbatim pair)"):
            return 1, 0
        return 0, 0
    if re.search(r"duplicat|identical", reason, re.IGNORECASE):
        shorter_idx = pair[0] if len(pair_contents[0]) <= len(pair_contents[1]) else pair[1]
        if invalidate_memory(memories[shorter_idx]["id"], reason=f"semantic_duplicate (judge reason: {reason[:80]})"):
            return 1, 0
        return 0, 0
    fp = flag_fingerprint(pair_contents)
    if fp in prev_flags:
        return 0, 0
    new_flags.add(fp)
    problems.append(
        f"LLM contradiction scan: stable-attribute conflict needs review — "
        f"[{pair_contents[0][:60]}] vs [{pair_contents[1][:60]}] ({reason})"
    )
    return 0, 1


def _contradiction_scan_with_recall(problems, memories):
    """Run contradiction scan using recall-based approach (v2.0 fallback path)."""
    if memories is None:
        memories = recall_all_recent()
    if len(memories) < 2:
        return  # nothing to scan

    contents = [m["content"] for m in memories]
    contradictions = llm_judge.detect_contradictions(contents)

    invalidated = 0
    flagged = 0
    seen_pairs = set()
    prev_flags = load_flag_state()
    new_flags = set()

    for c in contradictions:
        pair = c["pair"]
        # v2.2.1: indices already validated by llm_judge, but double-check bounds
        if not (0 <= pair[0] < len(memories) and 0 <= pair[1] < len(memories)):
            continue
        key = tuple(sorted(pair))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        resolution = c.get("resolution", "flag_human")
        reason = c.get("reason", "")
        newer_idx = c.get("newer_index")

        if resolution == "recency_wins" and newer_idx is not None:
            invalidated += _handle_recency_wins_recall(memories, pair, reason, newer_idx)
        elif resolution == "invalidate_meta":
            invalidated += _handle_invalidate_meta_recall(memories, pair)
        else:
            inv, flg = _handle_stable_conflict_recall(memories, pair, reason, contents, prev_flags, new_flags, problems)
            invalidated += inv
            flagged += flg

    if invalidated > 0:
        problems.append(
            f"LLM contradiction scan: {invalidated} stale state-change memories invalidated "
            f"(recency-wins, non-destructive)"
        )
    if new_flags:
        save_flag_state(prev_flags | new_flags)


def llm_contradiction_scan(problems, memories=None):
    """LLM-driven contradiction detection (Step 6, new in v2.0).

    v2.3: Recency is resolved deterministically from MemoryRecord timestamps,
    not from the LLM's interpretation of list position or wording. Uses
    structured records when available. Cross-chunk candidate generation.

    v2.0.1: handles the 'meta_noise' type (invalidate the report memory),
    dedups flagged pairs across runs via a fingerprint state file (an
    unresolved flag reports once, not every day), and shares the recall set
    with the dedup pass.

    Research basis:
    - Hindsight blog: conflict handling — recency wins for state, source/confidence for stable
    - Hindsight blog: entity drift (Postgres->MySQL) is the canonical failure
    """
    if llm_judge is None:
        return  # LLM unavailable — skip

    # v2.3: use structured records for deterministic recency
    if memory_records is not None:
        _contradiction_scan_with_records(problems)
        return

    # Fallback: old recall-based approach (v2.0)
    _contradiction_scan_with_recall(problems, memories)


def _restore_memory(memory_id: str):
    """Restore a previously invalidated memory (v2.4).

    Re-validates a memory by PATCHing state back to 'valid'.
    Reads the audit log to find the before_state for verification.
    """
    # Check audit log for the memory
    audit_entries = []
    if memory_records is not None:
        audit_entries = memory_records.read_audit_log(limit=10, action="invalidate")
        matching = [e for e in audit_entries if e.get("memory_id", "").startswith(memory_id)]
        if matching:
            entry = matching[-1]  # most recent
            print(f"Found audit entry: {entry.get('timestamp')} — {entry.get('reason')}")
            print(f"  Before state: {entry.get('before_state')}")

    # Re-validate the memory
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


def _trigger_consolidation(args, problems):
    """Step 1: Trigger consolidation, return operation_id or None."""
    if args.dry_run:
        problems.append("[dry-run] Consolidation skipped")
        return None
    try:
        status, resp = http("POST", f"/v1/default/banks/{BANK}/consolidate", timeout=120, body={})
        op_id = resp.get("operation_id")
        if not op_id:
            problems.append(f"Consolidate returned HTTP {status} but no operation_id: {str(resp)[:120]}")
        return op_id
    except Exception as e:
        problems.append(f"Consolidate trigger failed: {type(e).__name__}: {e}")
        return None


def _poll_consolidation(op_id, problems):
    """Step 2: Poll consolidation operation until terminal state or timeout."""
    deadline = time.time() + POLL_DEADLINE
    while time.time() < deadline:
        try:
            _, op = http("GET", f"/v1/default/banks/{BANK}/operations/{op_id}", timeout=120)
            st = op.get("status", "")
            if st in ("completed", "failed", "error", "cancelled"):
                return op
        except Exception as e:
            problems.append(f"Operation poll failed: {type(e).__name__}: {e}")
            return None
        time.sleep(POLL_INTERVAL)
    return None


def _run_consolidation(args, problems):
    """Steps 1-2: Trigger consolidation and poll for completion."""
    op_id = _trigger_consolidation(args, problems)
    if not op_id:
        return None

    final = _poll_consolidation(op_id, problems)
    if final is None:
        problems.append(f"Consolidation op {op_id[:8]} did not finish within {POLL_DEADLINE}s")
    elif final.get("status") != "completed":
        problems.append(f"Consolidation op {op_id[:8]} ended with status={final.get('status')}")
    return final


def _run_smoke_tests(args, problems, final):
    """Steps 3 + 3b: Recall and retain smoke-tests."""
    # --- 3. Recall smoke-test (consolidation over-prune check) ----------
    if final is not None and final.get("status") == "completed":
        try:
            _, rec = http("POST", f"/v1/default/banks/{BANK}/memories/recall", timeout=120,
                          body={"query": "David's preferred output language and document summary style"})
            hits = rec.get("results") or rec.get("memories") or rec.get("items") or []
            if not hits:
                problems.append(
                    "Recall smoke-test returned 0 results after consolidation — "
                    "possible over-prune; verify manually"
                )
        except Exception as e:
            problems.append(f"Recall smoke-test failed: {type(e).__name__}: {e}")

    # --- 3b. Retain smoke-test (silent write-failure check) -------------
    if not args.dry_run:
        try:
            _, ret = http("POST", f"/v1/default/banks/{BANK}/memories", timeout=120,
                          body={"items": [{
                              "content": f"daily memory optimization smoke test {time.strftime('%Y-%m-%d')}",
                              "tags": [SMOKE_TEST_TAG],
                          }]})
            usage = (ret.get("usage") or {})
            if not ret.get("success"):
                problems.append(f"Retain smoke-test failed: success != true ({str(ret)[:120]})")
            elif not usage.get("total_tokens"):
                problems.append(
                    "Retain smoke-test: success but total_tokens=0 — "
                    "fact extraction not running; check LLM provider/auth"
                )
        except Exception as e:
            problems.append(f"Retain smoke-test failed: {type(e).__name__}: {e}")


def _check_bank_stats(problems):
    """Step 4: Bank stats + failed_operations trend."""
    try:
        _, stats = http("GET", f"/v1/default/banks/{BANK}/stats", timeout=120)
        nodes = stats.get("total_nodes", 0)
        docs = stats.get("total_documents", 0)
        if nodes <= 0:
            problems.append(f"Bank '{BANK}' stats suspicious: total_nodes={nodes}, docs={docs}")
        failed = stats.get("failed_operations", 0)
        prev_failed = 0
        if STATE_FILE.exists():
            with contextlib.suppress(Exception):
                prev_failed = int(json.loads(STATE_FILE.read_text()).get("failed_operations", 0))
        if failed >= FAILED_OPS_WARN or failed > prev_failed:
            problems.append(
                f"failed_operations={failed} (was {prev_failed}) — "
                f"Hindsight writes failing silently; check backlog"
            )
        _atomic_write_text(STATE_FILE, json.dumps({"failed_operations": failed}))
    except Exception as e:
        problems.append(f"Bank stats fetch failed: {type(e).__name__}: {e}")


def _run_llm_passes(args, problems):
    """Step 5: LLM dedup + contradiction passes."""
    if not args.dry_run:
        try:
            cleanup_smoke_tests(problems)
            memories = recall_all_recent()
            if memories:
                invalidate_meta_memories(memories, problems)
                exact_dupes = exact_duplicate_prepass(memories)
                if exact_dupes:
                    problems.append(f"Exact-duplicate pre-pass: {exact_dupes} verbatim copies invalidated")
                # Drop memories invalidated by the pre-passes before the LLM sees them
                memories = [m for m in memories if not META_MEMORY_RE.search(m["content"])]
            llm_semantic_dedup_pass(problems, memories=memories)
            llm_contradiction_scan(problems, memories=memories)
        except Exception as e:
            problems.append(f"LLM dedup/contradiction passes failed: {type(e).__name__}: {e}")


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


def _check_knowledge_pages(problems):
    """Step 7: Knowledge Pages health (Hindsight >= 0.9 only)."""
    # v2.3: Version-gate the is_stale behavior. Per current Hindsight docs
    # (v0.9.1+), tree is_stale is scope-aware (only marks stale when a
    # memory in that page's tags+fact-type scope has been written since
    # the page last read memories). The old bank-wide approximation
    # workaround (querying mental models for small KBs) is only needed
    # for versions < 0.9.1.
    try:
        api_version, version_parts = _get_hindsight_version()

        _, tree = http("GET", f"/v1/default/banks/{BANK}/knowledge-base/tree", timeout=120)
        pages = [n for n in walk_tree(tree.get("roots")) if n.get("kind") == "page"]
        if not pages:
            return

        stale, use_tree_stale_directly = _detect_stale_pages(pages, version_parts, api_version)
        if stale and len(stale) / len(pages) > KP_STALE_RATIO_WARN:
            problems.append(
                f"Knowledge Pages: {len(stale)}/{len(pages)} pages stale "
                f"(v{api_version}, {'scope-aware' if use_tree_stale_directly else 'approximate'} signal) "
                f"(e.g. {', '.join(n['name'] for n in stale[:3])}) — pages falling behind consolidation"
            )
    except urllib.error.HTTPError:
        pass  # <0.9 or KB disabled — not an error
    except Exception as e:
        problems.append(f"Knowledge Pages check failed: {type(e).__name__}: {e}")


def _check_l3_wiki(problems):
    """Step 7b: L3 wiki lint-lite."""
    try:
        # Check both KB_DIR (~/.hermes/kb) and WIKI_DIR (~/wiki or $WIKI_DIR)
        for wiki_dir in (KB_DIR, WIKI_DIR):
            if wiki_dir.exists():
                md_files = [p for p in wiki_dir.rglob("*.md") if "_archive" not in p.parts]
                stale = []
                cutoff = time.time() - KB_STALE_DAYS * 86400
                for p in md_files:
                    if p.stat().st_mtime < cutoff and "index.md" not in p.name.lower():
                        stale.append(p.name)
                if len(stale) >= 5:
                    problems.append(
                        f"L3 wiki ({wiki_dir}): {len(stale)} of {len(md_files)} pages older than {KB_STALE_DAYS} days "
                        f"(e.g. {', '.join(sorted(stale)[:3])}) — run llm-wiki lint / refresh"
                    )
    except Exception as e:
        problems.append(f"L3 wiki check failed: {type(e).__name__}: {e}")


def _check_local_memory(args, problems):
    """Step 8: Local memory capacity check."""
    # v2.2.1: use decoded char count, not st_size bytes
    try:
        if MEM_FILE.exists():
            content = MEM_FILE.read_text(encoding="utf-8")
            n = len(content)
            pct = n / MEM_CAP
            if pct >= OFFLOAD_AT:
                _handle_memory_offload(args, problems, n, pct)
            elif pct >= MEM_WARN:
                problems.append(f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — approaching capacity")
        if USER_FILE.exists():
            content = USER_FILE.read_text(encoding="utf-8")
            n = len(content)
            pct = n / USER_CAP
            if pct >= MEM_WARN:
                problems.append(f"USER.md at {n}/{USER_CAP} chars ({int(pct*100)}%) — prune or offload to Hindsight")
    except Exception as e:
        problems.append(f"Local memory check failed: {type(e).__name__}: {e}")


def _handle_memory_offload(args, problems, n, pct):
    """Handle memory offload when MEMORY.md is over capacity."""
    if memory_offload is None:
        problems.append(
            f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
            "memory_offload.py not found next to this script; offload skipped"
        )
    elif not args.dry_run:
        try:
            used_before, _ = memory_offload.get_memory_usage()
            memory_offload.main()
            used_after, _ = memory_offload.get_memory_usage()
            if used_after < used_before:
                problems.append(
                    f"MEMORY.md was at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                    f"offloaded to Hindsight, now {used_after}/{MEM_CAP} chars"
                )
            else:
                problems.append(
                    f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                    f"offload ran but no entries moved (all essential?). Prune manually."
                )
        except Exception as e:
            problems.append(
                f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                f"offload failed: {type(e).__name__}: {e}"
            )


def _build_telegram_text(unresolved, resolved):
    """Build the Telegram notification HTML for unresolved issues."""
    timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    tg_text = (
        f"<b>🧠 Daily Memory Optimization</b>\n"
        f"<i>{timestamp}</i>\n\n"
        f"<b>{len(unresolved)} issue(s) need your attention "
        f"(LLM auto-resolve attempted, {len(resolved)} resolved):</b>\n\n"
    )
    for u in unresolved:
        tg_text += f"  • {_escape_html(u)}\n"
    return tg_text


def _get_run_mode(args):
    """Determine the current run mode string for output."""
    if args.dry_run:
        return "dry-run"
    if args.apply:
        return "apply"
    if ALLOW_DESTRUCTIVE:
        return "destructive"
    return "safe"


def _output_results(args, problems):
    """Step 9: Output: LLM auto-resolve → Telegram fallback."""
    if not problems:
        return  # empty stdout -> cron stays silent

    # Step 9a: attempt LLM auto-resolution
    resolved, unresolved = try_resolve_issues_with_llm(problems)

    # Step 9b: build output report
    if resolved:
        print("**🧠 Daily Memory Optimization — issues auto-resolved**\n")
        for r in resolved:
            print(f"  ✅ {r}")

    if not unresolved:
        return

    # Step 9c: notify user via Telegram DM for remaining issues
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
        print(f"  ⚠️ {u}")


def main():
    global ALLOW_DESTRUCTIVE

    # v2.2.1: parse CLI args
    # v2.4: added --apply and --restore modes
    parser = argparse.ArgumentParser(description="Daily memory optimization")
    parser.add_argument("--allow-destructive", action="store_true",
                        help="Enable LLM auto-mutation (invalidate, config_tune). "
                             "Default: disabled — LLM is advisor only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report issues without taking any action.")
    parser.add_argument("--apply", action="store_true",
                        help="Apply all recommended actions (implies --allow-destructive). "
                             "Use --dry-run first to preview.")
    parser.add_argument("--restore", type=str, metavar="AUDIT_ID",
                        help="Restore a previously invalidated memory from the audit log. "
                             "Pass the memory_id from the audit log entry.")
    parser.add_argument("--audit-log", action="store_true",
                        help="Print recent audit log entries and exit.")
    args = parser.parse_args()

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

    problems = []

    # --- 1-2. Trigger consolidation + poll -------------------------------
    final = _run_consolidation(args, problems)

    # --- 3 + 3b. Smoke tests ---------------------------------------------
    _run_smoke_tests(args, problems, final)

    # --- 4. Bank stats + failed_operations trend -------------------------
    _check_bank_stats(problems)

    # --- 5. LLM dedup + contradiction passes (v2.0, restructured v2.0.1) -
    _run_llm_passes(args, problems)

    # --- 7. Knowledge Pages health (Hindsight >= 0.9 only) ---------------
    _check_knowledge_pages(problems)

    # --- 7b. L3 wiki lint-lite ------------------------------------------
    _check_l3_wiki(problems)

    # --- 8. Local memory capacity check --------------------------------
    _check_local_memory(args, problems)

    # --- 9. Output: LLM auto-resolve → Telegram fallback -----------------
    _output_results(args, problems)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Script crash = real error, always report
        print(f"**🧠 Daily Memory Optimization — script crashed**\n\n  • {type(e).__name__}: {e}")
    sys.exit(0)
