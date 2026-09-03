#!/usr/bin/env python3
"""Structured memory records and candidate generation for v2.3+.

Replaces the old broad-recall approach with:
  - Paginated /memories/list endpoint (every memory eventually examined)
  - Structured MemoryRecord dataclass with timestamps, fact_type, state
  - Deterministic recency resolution from metadata (not LLM list-position)
  - Cross-chunk candidate generation (BATCH_SIZE boundary fix)
  - Scan cursor for incremental coverage across runs

v2.4 additions:
  - JSON audit log (before/after state, restore capability)
  - Privacy/PII redaction before cloud judging
"""

import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Location-aware resolution (v3.3) — ships with this repo
sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402

# === Config ===
HERMES_HOME = paths.resolve_hermes_home()
BASE = paths.resolve_hindsight_url(HERMES_HOME)
BANK = paths.resolve_hindsight_bank(HERMES_HOME)
SCAN_CURSOR_FILE = HERMES_HOME / "scripts" / ".memory_scan_cursor.json"
AUDIT_LOG_FILE = HERMES_HOME / "scripts" / ".memory_audit_log.jsonl"
AUDIT_MAX_ENTRIES = 500  # keep last N audit entries
LIST_PAGE_SIZE = 200     # memories per /memories/list page
MAX_SCAN_PER_RUN = 2000  # max memories to scan per daily run (time-bounded)


@dataclass
class MemoryRecord:
    """Structured memory record with temporal metadata.

    Replaces the old approach of passing raw content strings to the LLM.
    The LLM now receives structured records so it can reason about
    fact_type, timestamps, and state — not just prose.
    """
    id: str
    content: str
    fact_type: str = "world"  # world, experience, observation
    state: str = "valid"
    date: str = ""  # ISO timestamp from Hindsight
    occurred_start: str | None = None
    occurred_end: str | None = None
    edited_at: str | None = None
    tags: list[str] = field(default_factory=list)
    document_id: str | None = None
    source_memory_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp(self) -> str:
        """Best available timestamp for recency comparison.

        Priority: occurred_start > date > edited_at > ''
        (occurred_start is when the event actually happened;
         date is when it was retained; edited_at is last edit)
        """
        return self.occurred_start or self.date or self.edited_at or ""

    @property
    def is_curable(self) -> bool:
        """Only world/experience facts can be directly PATCHed."""
        return self.fact_type in ("world", "experience")

    @property
    def is_observation(self) -> bool:
        return self.fact_type == "observation"

    def to_llm_dict(self) -> dict:
        """Compact dict for LLM prompts (excludes large/internal fields)."""
        return {
            "id": self.id[:12],  # truncated for prompt brevity
            "content": self.content[:200],
            "fact_type": self.fact_type,
            "timestamp": self.timestamp[:10] if self.timestamp else "unknown",
            "tags": self.tags[:3],
        }


def _http_get(url: str, timeout: int = 120) -> dict:
    """GET JSON from URL."""
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read() or b"{}")


def fetch_memory(mid: str) -> MemoryRecord | None:
    """Fetch a single memory by ID and return a structured record."""
    try:
        data = _http_get(f"{BASE}/v1/default/banks/{BANK}/memories/{mid}")
        return _parse_memory(data)
    except Exception:
        return None


def _parse_memory(data: dict) -> MemoryRecord:
    """Parse a Hindsight memory dict into a MemoryRecord."""
    return MemoryRecord(
        id=data.get("id", ""),
        content=data.get("text", "") or data.get("content", ""),
        fact_type=data.get("fact_type") or data.get("type", "world"),
        state=data.get("state", "valid"),
        date=data.get("date", ""),
        occurred_start=data.get("occurred_start"),
        occurred_end=data.get("occurred_end"),
        edited_at=data.get("edited_at"),
        tags=data.get("tags", []),
        document_id=data.get("document_id"),
        source_memory_ids=data.get("source_memory_ids", []),
        metadata=data.get("metadata", {}),
    )


def list_memories(
    fact_type: str | None = None,
    state: str = "valid",
    limit: int = LIST_PAGE_SIZE,
    offset: int = 0,
    tags: str | None = None,
) -> tuple[list[MemoryRecord], int]:
    """Paginated memory listing via /memories/list endpoint.

    Returns (records, total_count). This replaces the old broad-recall
    approach that could miss memories outside the query's semantic
    neighborhood.
    """
    params = [f"limit={limit}", f"offset={offset}"]
    if state:
        params.append(f"state={state}")
    if fact_type:
        params.append(f"type={fact_type}")
    if tags:
        params.append(f"tags={tags}")

    url = f"{BASE}/v1/default/banks/{BANK}/memories/list?{'&'.join(params)}"
    data = _http_get(url)
    total = data.get("total", 0)
    items = data.get("items", [])
    records = [_parse_memory(item) for item in items]
    return records, total


# === Scan cursor (incremental coverage) ===

def load_scan_cursor() -> dict:
    """Load the scan cursor from the previous run."""
    try:
        return json.loads(SCAN_CURSOR_FILE.read_text())
    except Exception:
        return {"offset": 0, "total_seen": 0, "last_run": ""}


def save_scan_cursor(cursor: dict) -> None:
    """Persist the scan cursor for the next run."""
    cursor["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    SCAN_CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCAN_CURSOR_FILE.write_text(json.dumps(cursor))


def get_scan_batch(max_records: int = MAX_SCAN_PER_RUN) -> tuple[list[MemoryRecord], int, int]:
    """Get the next batch of memories to scan using the cursor.

    Returns (records, total_memories, new_offset). The cursor ensures
    every valid world/experience memory is eventually examined across
    multiple daily runs, rather than only scanning the top-N from one
    broad recall query.
    """
    cursor = load_scan_cursor()
    offset = cursor.get("offset", 0)

    # Get total count of valid source memories (world + experience)
    _, total = list_memories(fact_type="world", state="valid", limit=1, offset=0)
    _, total_exp = list_memories(fact_type="experience", state="valid", limit=1, offset=0)
    total_source = total + total_exp

    # Fetch world memories first, then experience
    records = []
    remaining = max_records

    # World memories
    if remaining > 0:
        world_records, _ = list_memories(
            fact_type="world", state="valid", limit=remaining, offset=offset
        )
        records.extend(world_records)
        remaining -= len(world_records)

    # Experience memories (always from offset 0 — separate list)
    if remaining > 0:
        exp_offset = max(0, cursor.get("experience_offset", 0))
        exp_records, _ = list_memories(
            fact_type="experience", state="valid", limit=remaining, offset=exp_offset
        )
        records.extend(exp_records)

    new_offset = offset + len(records)
    if new_offset >= total_source:
        new_offset = 0  # wrap around — start fresh next run

    return records, total_source, new_offset


# === Cross-chunk candidate generation ===

def _extract_keywords(rec: MemoryRecord) -> set[str]:
    """Extract keywords from a record's tags and content for entity blocking."""
    keywords = set()
    for tag in rec.tags:
        if not tag.startswith(("parent:", "session:")):
            keywords.add(tag.lower())
    words = re.findall(r'\b[A-Z][a-z]+\b', rec.content)
    keywords.update(w.lower() for w in words if len(w) > 3)
    return keywords


def _build_entity_blocks(records: list[MemoryRecord]) -> dict[str, list[int]]:
    """Build entity-based blocking index from records."""
    entity_blocks: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        for kw in _extract_keywords(rec):
            entity_blocks.setdefault(kw, []).append(i)
    return entity_blocks


def _content_similarity_candidates(records: list[MemoryRecord]) -> set[tuple[int, int]]:
    """Generate candidates based on content keyword overlap > 40%."""
    candidates: set[tuple[int, int]] = set()
    contents_normalized = [_normalize(rec.content) for rec in records]
    word_sets = [set(c.split()) for c in contents_normalized]
    for i in range(len(records)):
        if not word_sets[i]:
            continue
        for j in range(i + 1, len(records)):
            if not word_sets[j]:
                continue
            overlap = len(word_sets[i] & word_sets[j]) / max(len(word_sets[i]), 1)
            if overlap > 0.4:
                candidates.add((i, j))
    return candidates


def generate_candidate_pairs(records: list[MemoryRecord]) -> list[tuple[int, int]]:
    """Generate candidate pairs for dedup/contradiction checking.

    v2.3: Fixes the cross-chunk boundary problem. Previously, BATCH_SIZE=30
    meant entry 29 and entry 30 could never be compared. Now we generate
    candidates across ALL records, not just within chunks.

    Candidates are generated by:
    1. Entity overlap (shared entity names)
    2. Content similarity (normalized keyword overlap)
    3. Adjacent records (catches near-duplicates in ingest order)

    The LLM is then asked only to classify these pre-filtered candidates,
    not to scan everything — reducing LLM calls dramatically.
    """
    candidates: set[tuple[int, int]] = set()

    # 1. Entity-based blocking
    entity_blocks = _build_entity_blocks(records)
    for indices in entity_blocks.values():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                candidates.add((indices[a], indices[b]))

    # 2. Content similarity blocking (normalized keyword overlap > 40%)
    candidates |= _content_similarity_candidates(records)

    # 3. Adjacent records (catches near-duplicates in ingest order)
    for i in range(len(records) - 1):
        candidates.add((i, i + 1))

    return sorted(candidates)


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


# === Deterministic recency resolution ===

def resolve_recency(
    pair: tuple[MemoryRecord, MemoryRecord],
) -> tuple[MemoryRecord | None, MemoryRecord | None]:
    """Determine which memory is newer based on metadata.

    v2.3: Recency is resolved from timestamps in metadata, not from
    the LLM's interpretation of list position or wording.

    Returns (newer, older) or (None, None) if timestamps are missing
    or equal (requires human review).
    """
    a, b = pair
    ts_a = a.timestamp
    ts_b = b.timestamp

    if not ts_a or not ts_b:
        return None, None  # missing timestamps — can't determine recency

    if ts_a == ts_b:
        return None, None  # equal timestamps — ambiguous

    if ts_a > ts_b:
        return a, b  # a is newer
    else:
        return b, a  # b is newer


# === Audit log (v2.4) ===

@dataclass
class AuditEntry:
    """Audit log entry for memory mutations."""
    timestamp: str
    action: str  # invalidate, config_tune, consolidate, offload
    memory_id: str = ""
    before_state: dict = field(default_factory=dict)
    after_state: dict = field(default_factory=dict)
    reason: str = ""
    actor: str = "daily_memory_optimization"  # or "memory_offload"

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))


def append_audit_log(entry: AuditEntry) -> None:
    """Append an entry to the JSONL audit log."""
    AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry.to_jsonl() + "\n")

    # Rotate: keep only last AUDIT_MAX_ENTRIES
    try:
        lines = AUDIT_LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) > AUDIT_MAX_ENTRIES:
            rotated = "\n".join(lines[-AUDIT_MAX_ENTRIES:]) + "\n"
            fd, tmp = tempfile.mkstemp(dir=str(AUDIT_LOG_FILE.parent), prefix=".audit_rot_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(rotated)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(AUDIT_LOG_FILE))
    except Exception:  # noqa: S110 - best-effort rotation, never block
        pass


def read_audit_log(limit: int = 50, action: str | None = None) -> list[dict]:
    """Read recent audit log entries."""
    try:
        lines = AUDIT_LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
        entries = []
        for line in reversed(lines):
            try:
                entry = json.loads(line)
                if action and entry.get("action") != action:
                    continue
                entries.append(entry)
                if len(entries) >= limit:
                    break
            except json.JSONDecodeError:
                continue
        return entries
    except Exception:
        return []


# === Privacy/PII redaction (v2.4) ===

# Patterns for common PII
_PII_PATTERNS = [
    # Email
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
    # Phone numbers (various formats)
    (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE]'),
    # SSN
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
    # Credit card (basic pattern)
    (re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'), '[CARD]'),
    # API keys (common formats)
    (re.compile(r'(?:sk-|csk-|sk_nous_)[A-Za-z0-9]{20,}'), '[API_KEY]'),
    # IP addresses
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP]'),
]

# Tags that should never be sent to cloud judging
EXCLUDED_TAGS = {"secret", "private", "credential", "password"}


def redact_pii(text: str) -> str:
    """Redact PII and sensitive patterns from text before cloud judging.

    v2.4: Prevents operational memory containing credentials, personal
    information, or API keys from being sent to a cloud LLM judge.
    """
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def should_exclude_from_judging(record: MemoryRecord) -> bool:
    """Check if a memory should be excluded from cloud judging.

    Excludes memories with sensitive tags or content matching
    credential/password patterns.
    """
    # Check tags
    record_tags_lower = {t.lower() for t in record.tags}
    if record_tags_lower & EXCLUDED_TAGS:
        return True

    # Check content for credential patterns
    content_lower = record.content.lower()
    credential_indicators = ["password", "secret", "api_key", "token", "credential"]
    if any(indicator in content_lower for indicator in credential_indicators):
        # Only exclude if it looks like an actual credential, not just a mention
        # e.g. "the password is X" vs "passwords should be hashed"
        if re.search(r'(?:password|secret|api_?key|token|credential)\s*[=:]\s*\S+', content_lower):
            return True

    return False


def prepare_for_judging(records: list[MemoryRecord]) -> list[dict]:
    """Prepare memory records for LLM judging with privacy redaction.

    Returns a list of dicts safe to include in LLM prompts.
    Records with sensitive tags/content are excluded entirely.
    Content is PII-redacted before sending.
    """
    safe_records = []
    for rec in records:
        if should_exclude_from_judging(rec):
            continue
        redacted = redact_pii(rec.content)
        safe_records.append({
            "id": rec.id[:12],
            "content": redacted[:200],
            "fact_type": rec.fact_type,
            "timestamp": rec.timestamp[:10] if rec.timestamp else "unknown",
            "tags": [t for t in rec.tags if not t.startswith(("parent:", "session:"))][:3],
        })
    return safe_records
