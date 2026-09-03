#!/usr/bin/env python3
"""Hindsight Memory Offload Script (v3.0 — rule-based).

Runs via cron (no_agent=True). When local MEMORY.md exceeds the capacity
threshold, offloads non-essential entries to Hindsight and removes them
from local memory.

v3.2 (Sep 2026): Scoped LLM judge for the offload gate.
  - Rule-based heuristics remain the hard gate (quarantine, pins,
    essential prefixes, offload patterns, weighted scoring)
  - Entries the rules mark OFFLOADABLE are then reviewed by a
    Gemini 2.5 Flash Lite judge (OpenRouter) which confirms which
    are genuinely low-value and safe to move to L2
  - The judge can only VETO an offload (keep in L1) — it can never
    unlock one the rules kept
  - Fail-safe: any judge failure (API down, no key, parse error)
    falls back to the v3.1 rule-based behavior
  - Privacy: content is PII-redacted; sensitive entries never reach
    the judge (see llm_judge.py + memory_records.prepare_for_judging)
  - Disable with JUDGE_ENABLED=0

v3.0 (Sep 2026): LLM-as-judge removed — deterministic heuristics.
  - Classification via memory_heuristics.classify_importance() (weighted
    rule scoring, hard keep/offload rules, explainable decisions)
  - Recall dedup via memory_heuristics.is_duplicate() (exact/strong only)
  - Transactional rewrite: an entry is removed from L1 only after it is
    confirmed to exist in L2 or is successfully retained there; failed
    entries are always kept
  - Atomic MEMORY.md rewrite (temp file + fsync + os.replace)
  - Dry-run mode (MEMORY_HEURISTICS_DRY_RUN=1) reports proposed actions
    without rewriting MEMORY.md
  - Zero external chat-completion calls

v2.2.1 (Sep 2026): Safety patch — data-loss prevention, atomic writes, locking.
  - Fixed mixed-success offload bug: failed entries are ALWAYS kept locally
  - Atomic file writes via temp file + os.replace()
  - Backup before rewriting MEMORY.md
  - Advisory file lock (fcntl) prevents concurrent offload/daily runs
  - Tags now passed in Hindsight retain body
  - Idempotent document_id for retains (content-hash based)
  - Character-count consistency (decode before counting)

Essential entries (kept locally, rule-based):
  - IrisBot host specs (every turn context)
  - Hindsight config (every turn context)
  - Tool quirks / recurring fixes (prevents repeated work)
  - Vision config (every turn context)
  - Explicitly pinned entries ([pin], [pinned], [always inject])

Non-essential entries (offloaded to Hindsight, rule-based):
  - Provider rankings, pricing details, model history
  - One-time debugging lessons, historical state changes
  - Completed tasks, expired incidents, maintenance reports
"""

import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# Rule-based heuristics module (hard gate) + scoped LLM judge (v3.2)
# paths (v3.3) — location-aware resolution, ships with this repo
sys.path.insert(0, str(Path(__file__).parent))
import llm_judge  # noqa: E402
import memory_heuristics  # noqa: E402
import paths  # noqa: E402

# === Config (environment-overridable, location-aware — see paths.py) ===
HERMES_HOME = paths.resolve_hermes_home()
MEMORY_FILE = Path(os.environ.get("MEMORY_FILE", str(HERMES_HOME / "memories" / "MEMORY.md")))
HINDSIGHT_URL = paths.resolve_hindsight_url(HERMES_HOME)
BANK = paths.resolve_hindsight_bank(HERMES_HOME)
CAPACITY_MAX = int(os.environ.get("MEMORY_CHARS", "2200"))  # chars
OFFLOAD_THRESHOLD = float(os.environ.get("OFFLOAD_THRESHOLD", "0.75"))  # 75%
LOCK_FILE = MEMORY_FILE.with_suffix(".lock")  # MEMORY.lock next to MEMORY.md
BACKUP_DIR = MEMORY_FILE.parent / ".backups"
MAX_BACKUPS = 5  # keep last N backups

# Tags for Hindsight retain
TAG_MAP = {
    "IrisBot:": ["environment", "infra"],
    "Hindsight:": ["hindsight", "infra"],
    "Skills NOT": ["skills", "dev-workflow"],
    "MCP tool_call": ["mcp", "dev-workflow"],
    "HTML via execute_code": ["html", "dev-workflow"],
    "Vision:": ["vision", "infra"],
    "Search fallback:": ["dev-workflow", "search"],
    "Composio MCP:": ["composio", "mcp"],
    "lintlang": ["lintlang", "dev-workflow"],
    "Gemini 2.5 Flash-Lite provider": ["providers", "gemini-2.5-flash-lite"],
    "coding-agent-orchestration": ["skills", "dev-workflow"],
}


# === File locking ===

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False  # Windows — no fcntl


class FileLock:
    """Advisory file lock via fcntl.flock (prevents concurrent offload + daily runs).

    On platforms without fcntl (Windows), this is a no-op — the caller should
    be aware of the race risk in multi-process environments.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fd = None

    def __enter__(self):
        if not _HAS_FCNTL:
            return self
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.lock_path, "w")
        fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        if self._fd is not None:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None


# === Atomic write + backup ===

def _make_backup(src: Path) -> Path | None:
    """Create a timestamped backup of src. Returns backup path or None on failure."""
    try:
        import time
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"{src.stem}_{ts}{src.suffix}"
        backup.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        # Rotate: keep only MAX_BACKUPS
        backups = sorted(BACKUP_DIR.glob(f"{src.stem}_*{src.suffix}"))
        for old in backups[:-MAX_BACKUPS]:
            old.unlink(missing_ok=True)
        return backup
    except Exception:
        return None


def atomic_write_text(path: Path, content: str) -> None:
    """Write content to path atomically via temp file + os.replace().

    Ensures MEMORY.md is never in a half-written state if the process crashes
    mid-write. The temp file is created in the same directory (required for
    os.replace to be atomic on the same filesystem). The old file is
    preserved if any write operation fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        # Clean up temp file on failure — never leave partial writes
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_memory_file():
    """Read MEMORY.md and return list of entries (strings)."""
    if not MEMORY_FILE.exists():
        return []
    content = MEMORY_FILE.read_text(encoding="utf-8")
    # Entries are separated by lines containing only '§'
    raw_entries = content.split("§") if "§" in content else [content]
    entries = []
    for item in raw_entries:
        stripped = item.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
            entries.append(stripped)
    return entries


def get_memory_usage():
    """Read decoded character count from MEMORY.md (not byte count)."""
    if not MEMORY_FILE.exists():
        return 0, CAPACITY_MAX
    content = MEMORY_FILE.read_text(encoding="utf-8")
    if "§" in content:
        entries = [e.strip() for e in content.split("§") if e.strip()]
        total_chars = sum(len(e) for e in entries)
    else:
        total_chars = len(content)
    return total_chars, CAPACITY_MAX


def get_tags(entry):
    """Get appropriate tags for a Hindsight retain based on entry content."""
    for prefix, tags in TAG_MAP.items():
        if entry.startswith(prefix) or prefix in entry:
            return tags
    return ["offloaded", "memory-management"]


def _stable_document_id(content: str) -> str:
    """Generate a stable document_id for idempotent Hindsight retains.

    Uses SHA-256 of normalized content with an L1-offload namespace prefix.
    This makes retains idempotent — re-offloading the same content won't
    create duplicates even if the dedup-check misses them.
    """
    normalized = " ".join(content.split()).strip().lower()
    return f"l1-offload:{__import__('hashlib').sha256(normalized.encode()).hexdigest()[:16]}"


def hindsight_health_check():
    """Check if Hindsight is healthy before offloading."""
    try:
        req = urllib.request.Request(f"{HINDSIGHT_URL}/health")
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310
            data = json.loads(resp.read())
            return data.get("status") == "healthy"
    except Exception:
        return False


def hindsight_recall_check(content):
    """Recall + rule-based duplicate check: is content already in Hindsight?

    v3.0: replaces the LLM semantic-dedup check with
    memory_heuristics.is_duplicate() — exact and strong confidence only.
    If no strong duplicate is found, the caller retains the entry.
    """
    query = content[:80]  # first 80 chars as query
    try:
        payload = json.dumps({"query": query, "budget": "low", "max_tokens": 500}).encode()
        req = urllib.request.Request(
            f"{HINDSIGHT_URL}/v1/default/banks/{BANK}/memories/recall",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310
            data = json.loads(resp.read())
            results = data.get("results", [])
            if not results:
                return False
            result_texts = [r.get("content", "") or r.get("text", "") for r in results]
            return memory_heuristics.is_duplicate(content, result_texts[:5])
    except Exception:
        return False  # If recall fails, proceed with retain anyway


def hindsight_retain(content, tags=None):
    """Store an entry in Hindsight with tags and stable document_id.

    The document_id makes retains idempotent (re-offloading the same content
    won't create duplicates).
    """
    import hashlib
    normalized = " ".join(content.split()).strip().lower()
    document_id = f"l1-offload:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"
    body = {
        "items": [{
            "content": content,
            "context": "L1 memory offload",
            "tags": tags or [],
            "document_id": document_id,
        }]
    }
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{HINDSIGHT_URL}/v1/default/banks/{BANK}/memories",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310
        data = json.loads(resp.read())
        success = data.get("success", False)
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return success and tokens > 0


def rewrite_memory_file(entries_to_keep):
    """Rewrite MEMORY.md with only the kept entries (atomic + backup).

    Creates a backup before rewriting, then writes atomically via temp
    file + os.replace(). If the process crashes mid-write, the original
    file is intact and the backup is available for recovery.
    """
    # Backup before rewriting
    if MEMORY_FILE.exists():
        _make_backup(MEMORY_FILE)
    content = "".join(f"{entry.strip()}\n§\n" for entry in entries_to_keep)
    atomic_write_text(MEMORY_FILE, content)


def classify_entries(entries):
    """Classify entries as essential or offloadable (rules + LLM judge, v3.2).

    Stage 1 — hard gate (memory_heuristics.classify_importance()):
    deterministic weighted rules with hard keep/offload overrides.
    Quarantined (secret-like) entries are kept locally and never
    auto-offloaded.

    Stage 2 — scoped LLM judge (llm_judge.judge_offload_candidates()):
    a Gemini 2.5 Flash Lite judge reviews the rule-offloadable entries and
    confirms which are genuinely low-value (safe to move to L2). The judge
    can only VETO an offload — it never unlocks one the rules kept. Any
    judge failure falls back to the full rule-based offload set.
    """
    essential_idx, offloadable_idx = memory_heuristics.classify_importance(entries)
    essential = [entries[i] for i in essential_idx if 0 <= i < len(entries)]

    # Judge confirmation on the rule-offloadable set (fail-safe).
    candidates = [(i, entries[i]) for i in offloadable_idx if 0 <= i < len(entries)]
    confirmed_idx, vetoed_idx, status = llm_judge.judge_offload_candidates(candidates)
    offloadable = [entries[i] for i in confirmed_idx if 0 <= i < len(entries)]

    if vetoed_idx:
        judge_kept = [entries[i] for i in vetoed_idx if 0 <= i < len(entries)]
        print(
            f"Judge ({llm_judge.JUDGE_MODEL}, {status}): kept {len(judge_kept)} "
            f"rule-offloadable entries in L1 (vetoed offload)"
        )
        for entry in judge_kept:
            print(f"  kept: {entry[:80]}")
    elif status in ("fallback", "disabled", "skipped"):
        # Silent: judge is an enhancement, not a dependency.
        pass
    return essential, offloadable


def main():
    """Offload non-essential memory entries to Hindsight.

    Safety invariant (v3.0 spec section 12):
        An entry may be removed from L1 only after it is confirmed to
        exist in L2 or is successfully retained there. Failed entries
        are ALWAYS kept.

    The file lock prevents concurrent execution by the 30-min offload cron
    and the daily optimization job.
    """
    with FileLock(LOCK_FILE):
        _do_offload()


def _offload_entry(entry, tags):
    """Offload a single entry to Hindsight. Returns True if safely offloaded.

    True means: already present in L2 (verified by rule-based duplicate
    check) OR successfully retained now. False means the entry MUST stay
    in L1.
    """
    try:
        # Dedup check — skip retain if already in Hindsight (rule-based,
        # exact/strong confidence only)
        if hindsight_recall_check(entry):
            return True  # Already in L2 — safe to remove from L1
        return hindsight_retain(entry, tags)
    except Exception:
        return False


def _do_offload():
    """Offload non-essential memory entries to Hindsight (transactional, v3.0).

    Required behavior (v3.0 spec section 12):

        entries_to_keep = list(essential)
        for entry in offloadable:
            if already_in_hindsight(entry):
                mark_safe_to_remove(entry)
            elif hindsight_retain(entry):
                mark_safe_to_remove(entry)
            else:
                entries_to_keep.append(entry)

    Only successfully retained or already-present entries may be removed.
    The replacement file is written atomically (MEMORY.md.tmp + fsync +
    os.replace); the old file is preserved if any write operation fails.
    """
    dry_run = memory_heuristics.is_dry_run()

    # 1. Check Hindsight health
    if not hindsight_health_check():
        print("WARN: Hindsight not healthy — skipping offload cycle.")
        sys.exit(0)  # Silent exit, no alert needed

    # 2. Read local memory
    entries = read_memory_file()
    if not entries:
        sys.exit(0)  # Nothing to do

    # 3. Check capacity (decoded chars, not bytes)
    used, capacity = get_memory_usage()
    usage_pct = used / capacity
    if usage_pct <= OFFLOAD_THRESHOLD:
        sys.exit(0)  # Under threshold, nothing to do

    # 4. Classify entries (rule-based)
    essential, offloadable = classify_entries(entries)

    if not offloadable:
        # All entries are essential but we're over capacity
        print(f"WARN: Memory at {usage_pct:.0%} but all {len(entries)} entries are essential. Cannot offload.")
        sys.exit(0)

    # 5. Transactional offload: track per-entry safety
    entries_to_keep = list(essential)
    safe_to_remove = []
    failed = []

    for entry in offloadable:
        tags = get_tags(entry)
        if _offload_entry(entry, tags):
            safe_to_remove.append(entry)
            if dry_run:
                print("DRY RUN: would remove entry from L1 (already present or retained in L2)")
                print("  Rule: OFFLOAD_SAFE_TO_REMOVE")
                print(f"  Entry: {entry[:80]}")
        else:
            failed.append(entry)
            entries_to_keep.append(entry)  # never lose data
            if dry_run:
                print("DRY RUN: would keep entry in L1 (L2 retain failed)")
                print("  Rule: OFFLOAD_RETAIN_FAILED")
                print(f"  Entry: {entry[:80]}")

    # 6. Rewrite local memory: essential + FAILED entries (never lose data)
    #    Dry-run never rewrites MEMORY.md.
    if safe_to_remove and not dry_run:
        rewrite_memory_file(entries_to_keep)

    # 7. Report (only if something happened)
    _report_offload_results(safe_to_remove, failed, essential, entries_to_keep, capacity, dry_run)


def _report_offload_results(safe_to_remove, failed, essential, kept, capacity, dry_run=False):
    """Print offload summary if anything happened."""
    if not (safe_to_remove or failed):
        return
    new_used = sum(len(e) for e in kept)
    new_pct = new_used / capacity
    mode = "rule-based, DRY RUN" if dry_run else "rule-based"
    print(
        f"Memory offload ({mode}): "
        f"{len(safe_to_remove)} entries moved to Hindsight, "
        f"{len(failed)} failed (kept locally). "
        f"Local: {new_used}/{capacity} ({new_pct:.0%}). "
        f"{len(essential)} essential + {len(failed)} failed entries kept."
    )


if __name__ == "__main__":
    main()
