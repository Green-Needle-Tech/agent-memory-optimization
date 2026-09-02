#!/usr/bin/env python3
"""Hindsight Memory Offload Script.

Runs via cron (no_agent=True). When local MEMORY.md exceeds 75% capacity,
offloads non-essential entries to Hindsight and removes them from local memory.

v2.2.1 (Sep 2026): Safety patch — data-loss prevention, atomic writes, locking.
  - Fixed mixed-success offload bug: failed entries are ALWAYS kept locally
  - Atomic file writes via temp file + os.replace()
  - Backup before rewriting MEMORY.md
  - Advisory file lock (fcntl) prevents concurrent offload/daily runs
  - Tags now passed in Hindsight retain body (TAG_MAP was previously unused)
  - Idempotent document_id for retains (content-hash based)
  - Dedup canonical detection fixed (canonical==0 OR 0 in duplicates)
  - Character-count consistency (decode before counting, not st_size bytes)
  - Config via environment variables (no hardcoded /root)

v2.0 (Aug 2026): LLM-driven classification + semantic dedup.
  - Importance classification via llm_judge.classify_importance() (LLM-as-judge)
  - Semantic dedup via llm_judge.semantic_dedup() (replaces word-overlap >60%)
  - LLM-optional: degrades to rule-based prefix matching if LLM unavailable
  - Batch consolidation: all entries in one LLM call (LycheeMemory V2 pattern)

Essential entries (kept locally, LLM-judged):
  - IrisBot host specs (every turn context)
  - Hindsight config (every turn context)
  - Tool quirks / recurring fixes (prevents repeated work)
  - Vision config (every turn context)

Non-essential entries (offloaded to Hindsight, LLM-judged):
  - Provider rankings, pricing details, model history
  - Cron job IDs, specific version numbers
  - One-time debugging lessons
  - Historical state changes
  - Detailed environment configs that rarely change
"""

import hashlib
import json
import os
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

# LLM judge module (LLM-optional — degrades to rule-based if unavailable)
sys.path.insert(0, str(Path(__file__).parent))
llm_judge: types.ModuleType | None
try:
    import llm_judge
except ImportError:
    llm_judge = None  # standalone run: use rule-based fallbacks

# === Config (environment-overridable, no hardcoded /root) ===
HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
MEMORY_FILE = Path(os.environ.get("MEMORY_FILE", str(HERMES_HOME / "memories" / "MEMORY.md")))
HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL", "http://localhost:8888")
BANK = os.environ.get("HINDSIGHT_BANK", "main")
CAPACITY_MAX = int(os.environ.get("MEMORY_CHARS", "2200"))  # chars
OFFLOAD_THRESHOLD = float(os.environ.get("OFFLOAD_THRESHOLD", "0.75"))  # 75%
LOCK_FILE = MEMORY_FILE.with_suffix(".lock")  # MEMORY.lock next to MEMORY.md
BACKUP_DIR = MEMORY_FILE.parent / ".backups"
MAX_BACKUPS = 5  # keep last N backups

# Rule-based fallback prefixes (used when LLM unavailable)
ESSENTIAL_PREFIXES = [
    "IrisBot:",
    "Hindsight: localhost:8888",
    "Skills NOT autonomously patchable",
    "MCP tool_call JSON fix",
    "HTML via execute_code",
    "Vision: openrouter",
    "Search fallback:",
]
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
    "GLM 5.2 provider": ["providers", "glm-5.2"],
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
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        import time
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
    os.replace to be atomic on the same filesystem).
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
    return f"l1-offload:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"


def hindsight_health_check():
    """Check if Hindsight is healthy before offloading."""
    try:
        req = urllib.request.Request(f"{HINDSIGHT_URL}/health")
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310
            data = json.loads(resp.read())
            return data.get("status") == "healthy"
    except Exception:
        return False


def _check_word_overlap(content: str, result_text: str, threshold: float = 0.6) -> bool:
    """Check if two texts have word overlap above threshold."""
    entry_words = set(content.lower().split())
    result_words = set(result_text.lower().split())
    if entry_words and result_words:
        overlap = len(entry_words & result_words) / len(entry_words)
        return overlap > threshold
    return False


def hindsight_recall_check(content):
    """Semantic recall to check if content is already in Hindsight (dedup).

    v2.2.1: Fixed canonical detection — entry is a duplicate if it is the
    canonical OR appears in the duplicates list (previously only checked
    canonical==0, missing the case where Hindsight's copy is better-worded).

    v2.0: Uses LLM judge for semantic dedup when available.
    Falls back to word-overlap >60% if LLM unavailable.
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
            # Use LLM judge for semantic dedup if available
            if llm_judge is not None:
                return _llm_dedup_check(content, results)
            # Fallback: word overlap
            for r in results:
                result_text = r.get("content", "") or r.get("text", "")
                if _check_word_overlap(content, result_text):
                    return True
            return False
    except Exception:
        return False  # If recall fails, proceed with retain anyway


def _llm_dedup_check(content: str, results: list) -> bool:
    """Check if content is a duplicate using LLM semantic dedup."""
    result_texts = [r.get("content", "") or r.get("text", "") for r in results]
    dup_groups = llm_judge.semantic_dedup([content, *result_texts[:5]])
    for g in dup_groups:
        # v2.2.1: entry (index 0) is a duplicate if it's the canonical
        # OR appears in the duplicates list
        if g["canonical"] == 0 or 0 in g["duplicates"]:
            return True  # entry is already represented in Hindsight
    return False


def hindsight_retain(content, tags=None):
    """Store an entry in Hindsight with tags and stable document_id.

    v2.2.1: Tags and document_id are now included in the retain body
    (previously tags were computed but never sent — TAG_MAP had no effect).
    The document_id makes retains idempotent (re-offloading the same content
    won't create duplicates).
    """
    body = {
        "items": [{
            "content": content,
            "context": "L1 memory offload",
            "tags": tags or [],
            "document_id": _stable_document_id(content),
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

    v2.2.1: Creates a backup before rewriting, then writes atomically via
    temp file + os.replace(). If the process crashes mid-write, the original
    file is intact and the backup is available for recovery.
    """
    # Backup before rewriting
    if MEMORY_FILE.exists():
        _make_backup(MEMORY_FILE)
    content = "".join(f"{entry.strip()}\n§\n" for entry in entries_to_keep)
    atomic_write_text(MEMORY_FILE, content)


def classify_entries(entries):
    """Classify entries as essential or offloadable.

    v2.0: Uses LLM judge (batch importance scoring) when available.
    Falls back to rule-based prefix matching if LLM unavailable.
    """
    if llm_judge is not None:
        essential_idx, offloadable_idx = llm_judge.classify_importance(entries)
        # v2.2.1: strict bounds check on LLM-returned indices
        essential = [entries[i] for i in essential_idx if 0 <= i < len(entries)]
        offloadable = [entries[i] for i in offloadable_idx if 0 <= i < len(entries)]
        return essential, offloadable
    # Fallback: rule-based prefix matching
    essential = [e for e in entries if any(e.startswith(p) for p in ESSENTIAL_PREFIXES)]
    offloadable = [e for e in entries if not any(e.startswith(p) for p in ESSENTIAL_PREFIXES)]
    return essential, offloadable


def main():
    """Offload non-essential memory entries to Hindsight.

    Safety invariant (v2.2.1):
        An entry may be removed from L1 only after durable L2 retention
        or verified existing L2 presence. Failed entries are ALWAYS kept.

    The file lock prevents concurrent execution by the 30-min offload cron
    and the daily optimization job.
    """
    with FileLock(LOCK_FILE):
        _do_offload()


def _offload_entry(entry, tags):
    """Offload a single entry to Hindsight. Returns True if safely offloaded."""
    try:
        # Dedup check — skip if already in Hindsight (LLM semantic or word-overlap)
        if hindsight_recall_check(entry):
            return True  # Already in L2 — safe to remove from L1
        success = hindsight_retain(entry, tags)
        return success
    except Exception:
        return False


def _do_offload():
    """Offload non-essential memory entries to Hindsight.

    Safety invariant (v2.2.1):
        An entry may be removed from L1 only after durable L2 retention
        or verified existing L2 presence. Failed entries are ALWAYS kept.

    The file lock prevents concurrent execution by the 30-min offload cron
    and the daily optimization job.
    """
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

    # 4. Classify entries (LLM-driven or rule-based fallback)
    essential, offloadable = classify_entries(entries)

    if not offloadable:
        # All entries are essential but we're over capacity
        print(f"WARN: Memory at {usage_pct:.0%} but all {len(entries)} entries are essential. Cannot offload.")
        sys.exit(0)

    # 5. Offload each non-essential entry to Hindsight
    # v2.2.1: Track success/failure explicitly — failed entries are KEPT.
    successfully_offloaded = _offload_entries(offloadable)

    # 6. Rewrite local memory: essential + FAILED entries (never lose data)
    failed_offloads = [e for e in offloadable if e not in successfully_offloaded]
    kept = list(essential) + failed_offloads

    if successfully_offloaded:
        rewrite_memory_file(kept)

    # 7. Report (only if something happened)
    _report_offload_results(successfully_offloaded, failed_offloads, essential, kept, capacity)


def _offload_entries(offloadable: list) -> list:
    """Offload each entry, tracking successes. Returns list of successfully offloaded entries."""
    successfully_offloaded = []
    for entry in offloadable:
        tags = get_tags(entry)
        if _offload_entry(entry, tags):
            successfully_offloaded.append(entry)
    return successfully_offloaded


def _report_offload_results(successfully_offloaded, failed_offloads, essential, kept, capacity):
    """Print offload summary if anything happened."""
    if not (successfully_offloaded or failed_offloads):
        return
    new_used = sum(len(e) for e in kept)
    new_pct = new_used / capacity
    # v2.2.1: llm_status now accurately reflects whether LLM was actually used
    llm_available = llm_judge is not None and llm_judge.is_llm_available() if llm_judge else False
    llm_status = "LLM-judged" if llm_available else "rule-based"
    print(
        f"Memory offload ({llm_status}): "
        f"{len(successfully_offloaded)} entries moved to Hindsight, "
        f"{len(failed_offloads)} failed (kept locally). "
        f"Local: {new_used}/{capacity} ({new_pct:.0%}). "
        f"{len(essential)} essential + {len(failed_offloads)} failed entries kept."
    )


if __name__ == "__main__":
    main()
