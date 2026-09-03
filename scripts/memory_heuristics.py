#!/usr/bin/env python3
"""Rule-based memory heuristics for agent memory optimization (v3.0).

Replaces the v2 LLM-as-judge (llm_judge.py) with deterministic, local,
stdlib-only heuristics:

  1. classify_importance(entries)  -> essential vs offloadable (weighted rules)
  2. semantic_dedup(entries)       -> high-confidence duplicate groups (indexed)
  3. detect_contradictions(entries)-> structured claim conflicts
  4. is_duplicate(entry, others)   -> high-confidence pairwise check

Design principles (v3.0 spec):
  - Zero external chat-completion calls. No OpenRouter, no /chat/completions,
    no model endpoint anywhere in this module.
  - Deterministic: identical input + config -> identical output.
  - Conservative mutation: only exact and high-confidence strong duplicates
    or reliably-dated state changes may be auto-invalidated. False
    negatives are always preferred over destructive false positives.
  - Indexed candidate generation: dedup and contradiction scans never
    build an unrestricted O(n^2) pair matrix.
  - Timestamp safety: recall order is never treated as chronological order.
  - Non-destructive: callers use PATCH {"state": "invalidated"}; never DELETE.
  - Explainable: every decision carries a rule identifier and reason.
  - Stdlib only.

Inputs: each entry may be a plain string or a record dict:
  {"id": "...", "content": "...", "created_at": "...", "updated_at": "...",
   "tags": [...]}
Output indices always refer to the original input list.

Configuration: optional ~/.hermes/memory_heuristics.json
  {
    "essential_prefixes": [...],   # extra hard-keep prefixes
    "offload_patterns": [...],    # extra regex hard-offload patterns
    "state_attributes": [...],   # extra state-changing attributes
    "dry_run": false
  }
Invalid configuration produces a warning and falls back to defaults.

Dry-run: MEMORY_HEURISTICS_DRY_RUN=1 (env) or "dry_run": true (config file)
enables analysis-only mode; callers must not mutate memory state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# === Config ===

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
CONFIG_FILE = HERMES_HOME / "memory_heuristics.json"

# Hard-keep prefixes — consolidated from the old llm_judge.py and
# memory_offload.py fallback lists (v3.0 spec section 7).
DEFAULT_ESSENTIAL_PREFIXES = (
    "IrisBot:",
    "Hindsight: localhost:8888",
    "Skills NOT autonomously patchable",
    "MCP tool_call JSON fix",
    "HTML via execute_code",
    "Vision:",
    "Search fallback:",
)

# Explicit pin markers — always keep in L1.
PIN_MARKERS = (
    "[pin]",
    "[pinned]",
    "[always inject]",
    "[always-inject]",
)

# Hard-offload content patterns (regex, case-insensitive).
DEFAULT_OFFLOAD_PATTERNS = (
    r"\bcompleted (?:ticket|task|migration|one[- ]time)\b",
    r"\b(?:previously|previous|used to|formerly|no longer)\b.*\b(?:provider|model|backend|endpoint|version|url|port)\b",
    r"\bexpired incident\b",
    r"\bold migration\b",
    r"\bmaintenance report\b",
    r"\bwas (?:resolved|fixed|completed|migrated) (?:on|in) \d{4}\b",
    r"\b(?:past|previous) (?:run|report|maintenance)\b",
    r"\b(?:detailed )?log(?:s| entry| entries)? (?:from|of|about)\b",
    r"\btemporary observation\b",
)

# Weighted scoring rules (v3.0 spec section 7 table).
SCORE_RULES = (
    # (rule_id, regex, score)
    ("SCORE_ACTIVE_ENDPOINT", r"\b(?:localhost:\d{4}|https?://\S+|endpoint is)\b", 4),
    ("SCORE_ACTIVE_STATE", r"\b(?:current|active)\b.*\b(?:provider|model|url|port|version|bank|endpoint|database)\b", 4),
    ("SCORE_RECURRING_WORKAROUND", r"\b(?:quirk|fix:|workaround|recurring)\b", 3),
    ("SCORE_EVERY_TURN_PREFERENCE", r"\b(?:every turn|always injected|must know)\b", 3),
    ("SCORE_RUNTIME_CAPABILITY", r"\b(?:runs on|vcpu|ram|disk|installed)\b", 2),
    ("SCORE_HISTORICAL_MARKER", r"\b(?:previously|previous|used to|formerly|old (?:provider|model|version))\b", -4),
    ("SCORE_COMPLETED_TASK", r"\b(?:completed|one[- ]time|resolved)\b", -3),
    ("SCORE_MAINTENANCE_REPORT", r"\b(?:maintenance report|bookkeeping|audit (?:log|entry))\b", -5),
    ("SCORE_TIME_LIMITED", r"\b(?:temporary|transient|current session)\b", -3),
    ("SCORE_PRICING_HISTORY", r"\b(?:pricing|price per|version history)\b", -2),
)

# Disposition thresholds.
ESSENTIAL_SCORE = 3  # score >= 3 -> essential

# Secret-like content — quarantine, never auto-offload to Hindsight.
SECRET_RE = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|token|credential)\s*[=:]\s*\S+",
    re.IGNORECASE,
)

# === Normalization ===

# Stop words removed for similarity calculation (kept in normalized text).
STOP_WORDS = frozenset(
    "a an the is are was were be been being am do does did have has had "
    "of in on at to for with by from as and or but not no nor so if then "
    "than that this these those it its it's we our you your they their he "
    "she his her them us me my mine there here when where which who whom "
    "whose what how why all any both each few more most other some such "
    "can will just should now into about over under again further once "
    "only own same too very s t don re ve ll d m o y".split()
)

# Protected values — differences here block duplicate collapsing and route
# pairs to contradiction detection instead (v3.0 spec section 6).
# (name, regex, capture group) — group 0 = whole match, else the group.
PROTECTED_VALUE_PATTERNS = (
    ("url", re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE), 0),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0),
    ("port", re.compile(r"\bport\s*(?:is|=|:)?\s*(\d{2,5})\b", re.IGNORECASE), 1),
    ("semver", re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][\w.-]+)?\b"), 0),
    ("model", re.compile(r"\b(?:gpt|glm|claude|gemini|gemma|llama|mistral|qwen|deepseek|gpt-oss)[\w.-]*\b", re.IGNORECASE), 0),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?\b"), 0),
    ("email_like_host", re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b", re.IGNORECASE), 0),
    ("path", re.compile(r"(?:~|\.\.|/)[\w./-]+(?:\.[A-Za-z]{1,8})\b"), 0),
    ("quantity", re.compile(r"\b\d+(?:\.\d+)?\s*(?:gb|mb|kb|tb|vcpu|cpu|cores?|chars?|hours?|days?|minutes?|tokens?|%)\b", re.IGNORECASE), 0),
    ("bool", re.compile(r"\b(?:true|false|enabled|disabled)\b", re.IGNORECASE), 0),
)

# Markdown formatting removed during normalization.
_MARKDOWN_BULLETS_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_QUOTES_DASHES = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                  "\u2013": "-", "\u2014": "-", "\u2212": "-"}
_PUNCT_SEP_RE = re.compile(r"([(),;:!?])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")

# Negation markers — mismatched negation blocks duplicate collapsing.
_NEGATION_RE = re.compile(r"\b(?:not|never|no longer|cannot|can't|n't)\b")

# Temporal markers — mismatched temporal state blocks duplicate collapsing.
_TEMPORAL_RE = re.compile(
    r"\b(?:now|currently|today|as of|no longer|used to|previously|formerly)\b"
)


@dataclass(frozen=True)
class Claim:
    """A structured claim parsed from a memory entry.

    subject/attribute/value are normalized (lowercase, alias-resolved).
    timestamp is a reliable ISO date if one was extracted; temporal_marker
    records explicit transition wording ("explicit_transition").
    is_old_value marks the superseded side of a transition — such claims
    never participate in dedup and only serve contradiction resolution.
    """

    subject: str
    attribute: str
    value: str
    timestamp: str | None = None
    temporal_marker: str | None = None
    confidence: str = "medium"
    is_old_value: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.attribute)


@dataclass(frozen=True)
class NormalizedMemory:
    """Shared normalization result for one memory entry."""

    original: str
    normalized: str
    tokens: frozenset[str]
    significant_tokens: frozenset[str]
    protected_values: frozenset[str]
    claims: tuple[Claim, ...]
    negated: bool
    temporal_marker: bool
    timestamp: str | None = None
    tags: tuple[str, ...] = ()
    record_id: str | None = None


# === Attribute aliases (v3.0 spec section 9) ===

ATTRIBUTE_ALIASES = {
    "backend": "provider",
    "vendor": "provider",
    "endpoint": "url",
    "base_url": "url",
    "baseurl": "url",
    "db": "database",
    "database_engine": "database",
    "engine": "database",
    "llm": "model",
}

# State-changing attributes — recency_wins is permitted for these.
STATE_ATTRIBUTES = frozenset(
    ("provider", "model", "url", "port", "version", "status", "database", "enabled")
)

# Stable attributes — never auto-resolve; always flag_human.
STABLE_ATTRIBUTES = frozenset(
    ("legal_name", "birth_date", "identity", "account_id", "national_id",
     "passport", "name", "birthday")
)

# Claim syntax patterns (v3.0 spec section 9). Applied to normalized,
# date-stripped, trailing-punctuation-stripped segments.
_CLAIM_PATTERNS = (
    # <subject>: <attribute>=<value>   /   <subject>: <attribute>: <value>
    re.compile(
        r"^(?P<subject>[a-z][\w .-]{0,40}?)\s*:\s*(?P<attr>[a-z][\w -]{0,30}?)\s*[=:]\s*(?P<value>.+)$"
    ),
    # <subject> <attribute> is <value>
    re.compile(
        r"^(?P<subject>[a-z][\w .-]{0,40}?)\s+(?P<attr>[a-z][\w -]{0,30}?)\s+is\s+(?P<value>.+)$"
    ),
    # <subject> uses <value> as <attribute>
    re.compile(
        r"^(?P<subject>[a-z][\w .-]{0,40}?)\s+uses\s+(?P<value>.+?)\s+as\s+(?:its\s+)?(?P<attr>[a-z][\w -]{0,30}?)$"
    ),
    # <subject> (was) switched/upgraded/migrated/changed [attr] from <old> to <new>
    re.compile(
        r"^(?P<subject>[a-z][\w .-]{0,40}?)\s+(?:was\s+)?(?:upgraded|switched|migrated|changed)\s+"
        r"(?:(?P<attr>[a-z][\w -]{0,30}?)\s+)?from\s+(?P<old>.+?)\s+to\s+(?P<value>.+)$"
    ),
)

# Full-text pattern (spans ';'): <subject> no longer uses <old>; it uses <new>
_CLAIM_NO_LONGER_RE = re.compile(
    r"^(?P<subject>[a-z][\w .-]{0,40}?)\s+no\s+longer\s+uses\s+(?P<old>.+?)\s*;\s*"
    r"(?:it\s+)?(?:now\s+)?uses\s+(?P<value>.+)$"
)

# Explicit transition wording -> high-confidence state change.
_TRANSITION_RE = re.compile(
    r"\b(?:switched|migrated|upgraded|changed|no longer uses)\b"
)

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?\b")
_LEADING_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*[:,]?\s*")
_SEMVER_VALUE_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?")


# === Normalization pipeline ===

def _extract_protected_values(text: str) -> frozenset[str]:
    """Extract protected values (URLs, ports, versions, dates, ...)."""
    values = set()
    for _name, pattern, group in PROTECTED_VALUE_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(group) if group else m.group()
            values.add(raw.strip().rstrip(".,;:!?").lower())
    return frozenset(values)


def normalize_text(content: str) -> NormalizedMemory:
    """Normalize one memory entry. See module docstring for the pipeline."""
    original = content if isinstance(content, str) else str(content)

    # 1. Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", original)
    # 5. Markdown bullet / harmless formatting removal
    text = _MARKDOWN_BULLETS_RE.sub("", text)
    # 4. Quote and dash normalization
    for src, dst in _QUOTES_DASHES.items():
        text = text.replace(src, dst)
    # 2. Lowercase, 3. whitespace collapse
    text = re.sub(r"\s+", " ", text).strip().lower()
    # 6. Punctuation separation
    text = _PUNCT_SEP_RE.sub(r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Trailing punctuation removal (exact-dup hash consistency)
    text = _TRAILING_PUNCT_RE.sub("", text)

    tokens = frozenset(_TOKEN_RE.findall(text))
    significant = frozenset(t for t in tokens if t not in STOP_WORDS and len(t) > 1)
    protected = _extract_protected_values(original)
    claims = tuple(_parse_claims(text))

    date_m = _DATE_RE.search(original)
    return NormalizedMemory(
        original=original,
        normalized=text,
        tokens=tokens,
        significant_tokens=significant,
        protected_values=protected,
        claims=claims,
        negated=bool(_NEGATION_RE.search(text)),
        temporal_marker=bool(_TEMPORAL_RE.search(text)),
        timestamp=date_m.group(1) if date_m else None,
    )


def _normalize_attribute(attr: str) -> str:
    attr = attr.strip().lower().replace(" ", "_").replace("-", "_")
    return ATTRIBUTE_ALIASES.get(attr, attr)


def _make_claim(subject, attr, value, old, seg, seg_ts):
    """Build Claim(s) from a pattern match, applying transition semantics."""
    subject = re.sub(r"'?s$", "", subject.strip())
    value = value.strip()
    old = (old or "").strip()
    marker = "explicit_transition" if _TRANSITION_RE.search(seg) else None
    confidence = "high" if marker else "medium"
    ts = seg_ts

    if attr:
        attr_n = _normalize_attribute(attr)
    elif _SEMVER_VALUE_RE.match(value) and (not old or _SEMVER_VALUE_RE.match(old)):
        attr_n = "version"
    else:
        attr_n = "provider"

    claims = [Claim(subject=subject, attribute=attr_n, value=value,
                    timestamp=ts, temporal_marker=marker, confidence=confidence)]
    if old:
        claims.append(Claim(subject=subject, attribute=attr_n, value=old,
                            timestamp=ts, temporal_marker=marker,
                            confidence=confidence, is_old_value=True))
    return claims


def _parse_claims(normalized: str) -> list[Claim]:
    """Parse structured claims from normalized text.

    Only high-reliability syntax is supported (v3.0 spec section 9).
    Anything unparsed simply produces no claims — conservative.
    """
    claims: list[Claim] = []

    # Full-text pattern first: "no longer uses X; it uses Y" spans ';'.
    m = _CLAIM_NO_LONGER_RE.match(normalized)
    if m:
        gd = m.groupdict()
        date_m = _DATE_RE.search(normalized)
        seg_ts = date_m.group(1) if date_m else None
        claims.extend(_make_claim(gd["subject"], "provider", gd["value"],
                                  gd["old"], normalized, seg_ts))
        return claims

    # Sentence-ish segmentation.
    segments = [s.strip() for s in re.split(r"[.;] ", normalized) if s.strip()]
    for seg in segments:
        seg = _TRAILING_PUNCT_RE.sub("", seg)
        # Leading date prefix ("2026-08-01: Hindsight ...") -> claim timestamp.
        seg_ts = None
        dm = _LEADING_DATE_RE.match(seg)
        if dm:
            seg_ts = dm.group(1)
            seg = seg[dm.end():].strip()
        if not seg:
            continue
        # Possessive normalization: "user's legal name" -> "users legal name"
        seg = re.sub(r"'s\b", "s", seg)
        for pattern in _CLAIM_PATTERNS:
            m = pattern.match(seg)
            if not m:
                continue
            gd = m.groupdict()
            if not gd.get("value") or len(gd["value"]) > 120:
                break
            claims.extend(_make_claim(gd["subject"], gd.get("attr") or "",
                                      gd["value"], gd.get("old") or "",
                                      seg, seg_ts))
            break
    return claims


# === Input handling ===

def _as_record(entry):
    """Accept str or dict; return (content, tags, record_id, created, updated)."""
    if isinstance(entry, dict):
        content = str(entry.get("content", "") or entry.get("text", ""))
        tags = tuple(str(t).lower() for t in (entry.get("tags") or []))
        rid = entry.get("id")
        return (content, tags, str(rid) if rid else None,
                entry.get("created_at"), entry.get("updated_at"))
    return (str(entry), (), None, None, None)


def _normalize_all(entries):
    """Normalize every entry; returns list of NormalizedMemory in input order."""
    results = []
    for e in entries:
        content, tags, rid, created, updated = _as_record(e)
        nm = normalize_text(content)
        ts = updated or created or nm.timestamp
        results.append(NormalizedMemory(
            original=nm.original, normalized=nm.normalized,
            tokens=nm.tokens, significant_tokens=nm.significant_tokens,
            protected_values=nm.protected_values, claims=nm.claims,
            negated=nm.negated, temporal_marker=nm.temporal_marker,
            timestamp=str(ts) if ts else None, tags=tags, record_id=rid,
        ))
    return results


# === Configuration ===

def load_config():
    """Load ~/.hermes/memory_heuristics.json; warn + defaults on invalid."""
    defaults = {
        "essential_prefixes": [],
        "offload_patterns": [],
        "state_attributes": [],
        "dry_run": False,
    }
    if not CONFIG_FILE.exists():
        return dict(defaults)
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config root must be an object")
        cfg = dict(defaults)
        for key in defaults:
            if key in data:
                cfg[key] = data[key]
        # Validate shapes
        if not isinstance(cfg["essential_prefixes"], list) or \
           not all(isinstance(p, str) for p in cfg["essential_prefixes"]):
            raise ValueError("essential_prefixes must be a list of strings")
        if not isinstance(cfg["offload_patterns"], list) or \
           not all(isinstance(p, str) for p in cfg["offload_patterns"]):
            raise ValueError("offload_patterns must be a list of strings")
        for p in cfg["offload_patterns"]:
            re.compile(p)  # raises on invalid regex
        if not isinstance(cfg["state_attributes"], list) or \
           not all(isinstance(a, str) for a in cfg["state_attributes"]):
            raise ValueError("state_attributes must be a list of strings")
        if not isinstance(cfg["dry_run"], bool):
            raise ValueError("dry_run must be a boolean")
        return cfg
    except (OSError, ValueError, json.JSONDecodeError, re.error) as exc:
        print(
            f"WARN: invalid {CONFIG_FILE} ({exc}); falling back to defaults",
            file=sys.stderr,
        )
        return dict(defaults)


def is_dry_run() -> bool:
    """True if dry-run mode is active (env var or config file)."""
    env = os.environ.get("MEMORY_HEURISTICS_DRY_RUN", "").lower() in ("1", "true", "yes")
    return env or bool(load_config().get("dry_run"))


# === Audit logging ===

AUDIT_LOG_FILE = HERMES_HOME / "scripts" / ".memory_heuristics_audit.jsonl"
AUDIT_MAX_ENTRIES = 500


def audit_log(operation: str, memory_id: str, rule: str, confidence: str,
              reason: str, replacement_id: str | None = None) -> None:
    """Append one audit entry (operation, id, rule, confidence, reason, ts).

    Content is never logged here; callers pass only ids and reasons, and
    secret-looking reasons are redacted.
    """
    if SECRET_RE.search(reason):
        reason = "[REDACTED: secret-like content]"
    entry = {
        "operation": operation,
        "memory_id": memory_id,
        "rule": rule,
        "confidence": confidence,
        "replacement_id": replacement_id,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import tempfile
        AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        # Rotate: keep last AUDIT_MAX_ENTRIES
        lines = AUDIT_LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) > AUDIT_MAX_ENTRIES:
            rotated = "\n".join(lines[-AUDIT_MAX_ENTRIES:]) + "\n"
            fd, tmp = tempfile.mkstemp(dir=str(AUDIT_LOG_FILE.parent), prefix=".audit_rot_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(rotated)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(AUDIT_LOG_FILE))
    except Exception:  # noqa: S110 - audit is best-effort, never block
        pass


# === Importance classification ===

@dataclass(frozen=True)
class ImportanceDecision:
    index: int
    disposition: str      # "essential" | "offloadable" | "quarantined"
    score: int
    matched_rules: tuple[str, ...]
    reason: str


def classify_importance_detailed(entries, context=None):
    """Classify entries with full explainable decisions.

    Returns list of ImportanceDecision in input order.
    """
    cfg = load_config()
    essential_prefixes = tuple(DEFAULT_ESSENTIAL_PREFIXES) + tuple(cfg["essential_prefixes"])
    offload_res = [re.compile(p, re.IGNORECASE) for p in DEFAULT_OFFLOAD_PATTERNS]
    offload_res += [re.compile(p, re.IGNORECASE) for p in cfg["offload_patterns"]]

    decisions = []
    for i, entry in enumerate(entries):
        content, tags, _rid, _c, _u = _as_record(entry)
        stripped = content.strip()

        # Secret-like content: quarantine — never auto-copied to Hindsight.
        if SECRET_RE.search(stripped) or "secret" in tags or "credential" in tags:
            decisions.append(ImportanceDecision(
                index=i, disposition="quarantined", score=0,
                matched_rules=("RULE_SECRET_QUARANTINE",),
                reason="secret-like content stays local; never auto-offloaded",
            ))
            continue

        low = stripped.lower()

        # Hard keep: explicit pin marker or tag.
        if any(marker in low for marker in PIN_MARKERS) or "pin" in tags or "pinned" in tags:
            decisions.append(ImportanceDecision(
                index=i, disposition="essential", score=100,
                matched_rules=("RULE_EXPLICIT_PIN",),
                reason="explicitly pinned entry",
            ))
            continue

        # Hard keep: configured essential prefix.
        if any(stripped.startswith(p) for p in essential_prefixes):
            decisions.append(ImportanceDecision(
                index=i, disposition="essential", score=100,
                matched_rules=("RULE_ESSENTIAL_PREFIX",),
                reason="matches configured essential prefix",
            ))
            continue

        # Hard offload: explicit offload tag.
        if "offload" in tags:
            decisions.append(ImportanceDecision(
                index=i, disposition="offloadable", score=-100,
                matched_rules=("RULE_TAG_OFFLOAD",),
                reason="explicitly tagged offload",
            ))
            continue

        # Hard offload: content pattern.
        offload_hit = next((p.pattern for p in offload_res if p.search(stripped)), None)
        if offload_hit:
            decisions.append(ImportanceDecision(
                index=i, disposition="offloadable", score=-100,
                matched_rules=("RULE_OFFLOAD_PATTERN",),
                reason=f"matches offload pattern: {offload_hit[:60]}",
            ))
            continue

        # Weighted scoring.
        score = 0
        matched = []
        for rule_id, pattern, pts in SCORE_RULES:
            if re.search(pattern, stripped, re.IGNORECASE):
                score += pts
                matched.append(rule_id)
        disposition = "essential" if score >= ESSENTIAL_SCORE else "offloadable"
        if matched:
            reason = f"weighted rules {', '.join(matched)} -> score {score}"
        else:
            reason = f"no rules matched; default score 0 < {ESSENTIAL_SCORE}"
        decisions.append(ImportanceDecision(
            index=i, disposition=disposition, score=score,
            matched_rules=tuple(matched),
            reason=reason,
        ))
    return decisions


def classify_importance(entries, context=None):
    """Return (essential_indices, offloadable_indices).

    Quarantined entries are treated as essential (kept locally, never
    auto-offloaded). Compatible with the old llm_judge signature.
    """
    if not entries:
        return [], []
    essential, offloadable = [], []
    for d in classify_importance_detailed(entries, context):
        if d.disposition == "offloadable":
            offloadable.append(d.index)
        else:
            essential.append(d.index)
    return essential, offloadable


# === Duplicate detection ===

# Automatic (strong) lexical duplicate thresholds (v3.0 spec section 8).
JACCARD_AUTO = 0.82
CONTAINMENT_AUTO = 0.92
MIN_SIGNIFICANT_TOKENS = 4
# Report-only (possible) thresholds.
JACCARD_POSSIBLE = 0.60
CONTAINMENT_POSSIBLE = 0.75
# Candidate generation limits.
MAX_CANDIDATES_PER_ENTRY = 100
RARE_TOKEN_DOC_FREQ = 5   # token appearing in <= N docs is "rare"
TRIGRAM_MIN_TOKENS = 8    # entries with >= N tokens get trigram indexing
MAX_INDEX_LIST = 50       # skip index lists longer than this (pathological)
MAX_CONFLICT_GROUP = 20   # claim-key groups larger than this are pathological


def _trigrams(tokens: frozenset[str]) -> set[tuple[str, str, str]]:
    """Sorted-token trigrams for long-entry candidate blocking."""
    ordered = sorted(tokens)
    return {(ordered[i], ordered[i + 1], ordered[i + 2])
            for i in range(len(ordered) - 2)}


def _generate_candidates(norms):
    """Indexed candidate generation — never an unrestricted O(n^2) matrix.

    Indexes: exact normalized hash, claim key, rare significant tokens,
    token trigrams. Two entries become lexical candidates only if they
    share >= 2 rare significant tokens or >= 1 trigram. Per-entry
    candidate count is capped at MAX_CANDIDATES_PER_ENTRY.
    """
    by_token: dict[str, list[int]] = {}
    by_trigram: dict[tuple[str, str, str], list[int]] = {}
    doc_freq: dict[str, int] = {}

    for i, nm in enumerate(norms):
        for tok in nm.significant_tokens:
            by_token.setdefault(tok, []).append(i)
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
        if len(nm.significant_tokens) >= TRIGRAM_MIN_TOKENS:
            for tri in _trigrams(nm.significant_tokens):
                by_trigram.setdefault(tri, []).append(i)

    pair_counts: dict[tuple[int, int], int] = {}
    for tok, idxs in by_token.items():
        if doc_freq[tok] > RARE_TOKEN_DOC_FREQ or len(idxs) > MAX_INDEX_LIST:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair = (idxs[a], idxs[b])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    for tri, idxs in by_trigram.items():
        if len(idxs) > MAX_INDEX_LIST:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair = (idxs[a], idxs[b])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    candidates = sorted(k for k, n in pair_counts.items() if n >= 2)

    # Enforce per-entry cap deterministically.
    per_entry: dict[int, list[int]] = {}
    kept = []
    for i, j in candidates:
        if len(per_entry.get(i, ())) >= MAX_CANDIDATES_PER_ENTRY:
            continue
        if len(per_entry.get(j, ())) >= MAX_CANDIDATES_PER_ENTRY:
            continue
        per_entry.setdefault(i, []).append(j)
        per_entry.setdefault(j, []).append(i)
        kept.append((i, j))
    return kept


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _containment(a: frozenset, b: frozenset) -> float:
    smallest = min(len(a), len(b))
    if smallest == 0:
        return 0.0
    return len(a & b) / smallest


def _canonical_choice(indices, norms):
    """Deterministic canonical selection (v3.0 spec section 8).

    Order: pinned/canonical tag > provenance (record id) > more complete
    (protected values agree is enforced by callers) > newer timestamp >
    lower original index.
    """
    def rank(i):
        nm = norms[i]
        return (
            0 if ("canonical" in nm.tags or "pinned" in nm.tags) else 1,
            0 if nm.record_id else 1,
            -len(nm.significant_tokens),
            nm.timestamp or "",
            i,
        )
    return sorted(indices, key=rank)[0]


def _lexical_confidence(a: NormalizedMemory, b: NormalizedMemory) -> str | None:
    """Classify lexical similarity; None = not a duplicate at all."""
    if a.negated != b.negated:
        return None  # negation mismatch — never a duplicate
    if a.temporal_marker != b.temporal_marker:
        return None  # temporal-state mismatch
    if a.protected_values != b.protected_values:
        return None  # protected values differ -> contradiction path instead
    if len(a.significant_tokens) < MIN_SIGNIFICANT_TOKENS or \
       len(b.significant_tokens) < MIN_SIGNIFICANT_TOKENS:
        return None
    jac = _jaccard(a.significant_tokens, b.significant_tokens)
    con = _containment(a.significant_tokens, b.significant_tokens)
    if jac >= JACCARD_AUTO and con >= CONTAINMENT_AUTO:
        return "strong"
    if jac >= JACCARD_POSSIBLE and con >= CONTAINMENT_POSSIBLE:
        return "possible"
    return None


def semantic_dedup(entries):
    """Return high-confidence duplicate groups.

    Group shape:
      {"canonical": idx, "duplicates": [idx, ...], "confidence": str,
       "rule": str, "reason": str}

    Confidence levels:
      "exact"   — identical normalized content (auto-invalidate ok)
      "strong"  — same structured claim or very high lexical similarity
                  with agreeing protected values (auto-invalidate ok)
      "possible"— similar but not safely identical (REPORT ONLY — callers
                  must never invalidate on these)
    """
    if len(entries) < 2:
        return []
    norms = _normalize_all(entries)
    used: set[int] = set()
    groups = []

    # 1. Exact normalized duplicates (SHA-256 of normalized content).
    by_hash: dict[str, list[int]] = {}
    for i, nm in enumerate(norms):
        digest = hashlib.sha256(nm.normalized.encode("utf-8")).hexdigest()
        by_hash.setdefault(digest, []).append(i)
    for digest in sorted(by_hash):
        idxs = [i for i in sorted(by_hash[digest]) if i not in used]
        if len(idxs) < 2:
            continue
        canonical = _canonical_choice(idxs, norms)
        dups = [i for i in idxs if i != canonical and i not in used]
        if not dups:
            continue
        used.update(dups)
        groups.append({
            "canonical": canonical,
            "duplicates": sorted(dups),
            "confidence": "exact",
            "rule": "DEDUP_EXACT_NORMALIZED",
            "reason": "normalized contents are identical",
        })

    # 2. Structured claim duplicates — same (subject, attribute) and value.
    #    Only CURRENT claims participate; superseded old-value claims from
    #    transitions are excluded (they describe past state, not the fact).
    by_claim: dict[tuple[str, str, str], list[int]] = {}
    for i, nm in enumerate(norms):
        for claim in nm.claims:
            if claim.is_old_value:
                continue
            by_claim.setdefault((claim.subject, claim.attribute, claim.value), []).append(i)
    for key in sorted(by_claim):
        members = [i for i in sorted(set(by_claim[key])) if i not in used]
        if len(members) < 2:
            continue
        # Protected values must agree.
        protos = [norms[i].protected_values for i in members]
        if any(p != protos[0] for p in protos[1:]):
            continue
        if any(norms[i].negated != norms[members[0]].negated for i in members):
            continue
        canonical = _canonical_choice(members, norms)
        dups = [i for i in members if i != canonical and i not in used]
        if not dups:
            continue
        used.update(dups)
        groups.append({
            "canonical": canonical,
            "duplicates": sorted(dups),
            "confidence": "strong",
            "rule": "DEDUP_STRUCTURED_CLAIM",
            "reason": "same subject, attribute, and normalized value",
        })

    # 3. Lexical duplicates on indexed candidate pairs.
    for i, j in _generate_candidates(norms):
        if i in used or j in used:
            continue
        conf = _lexical_confidence(norms[i], norms[j])
        if conf is None:
            continue
        canonical = _canonical_choice([i, j], norms)
        dup = j if canonical == i else i
        used.add(dup)
        if conf == "strong":
            groups.append({
                "canonical": canonical,
                "duplicates": [dup],
                "confidence": "strong",
                "rule": "DEDUP_LEXICAL_THRESHOLD",
                "reason": (f"jaccard>={JACCARD_AUTO}, containment>={CONTAINMENT_AUTO}, "
                           "protected values agree"),
            })
        else:
            groups.append({
                "canonical": canonical,
                "duplicates": [dup],
                "confidence": "possible",
                "rule": "DEDUP_LEXICAL_POSSIBLE",
                "reason": "similar but not safely identical — report only",
            })
    return groups


def is_duplicate(entry, others):
    """High-confidence duplicate check of one entry against recalled results.

    Returns True only for exact or strong duplicates (never "possible").
    Used by memory_offload's recall dedup step. If no strong duplicate is
    found, the caller retains the entry.
    """
    if not others:
        return False
    groups = semantic_dedup([entry, *others])
    for g in groups:
        if g["confidence"] not in ("exact", "strong"):
            continue
        if 0 in g["duplicates"] or g["canonical"] == 0:
            return True
    return False


# === Contradiction detection ===

def detect_contradictions(entries):
    """Return structured contradiction candidates.

    Only claims that parse reliably are compared (v3.0 spec section 9).
    Unsupported general-logic contradictions are never reported.
    """
    if len(entries) < 2:
        return []
    norms = _normalize_all(entries)

    # Index CURRENT claims by (subject, attribute). Old-value claims from
    # transitions are kept per-entry for explicit-transition resolution but
    # never compared as standalone assertions (they describe past state).
    by_key: dict[tuple[str, str], list[tuple[int, Claim]]] = {}
    old_claims: dict[int, list[Claim]] = {}
    for i, nm in enumerate(norms):
        for claim in nm.claims:
            if claim.is_old_value:
                old_claims.setdefault(i, []).append(claim)
            else:
                by_key.setdefault(claim.key, []).append((i, claim))

    contradictions = []
    seen_pairs: set[tuple[int, int]] = set()

    for key in sorted(by_key):
        members = by_key[key]
        if len(members) > MAX_CONFLICT_GROUP:
            # Pathological input (e.g. 500 entries all claiming the same
            # key with distinct values) — report only the most recent pair
            # deterministically; flag the rest as one aggregate issue.
            newest = max(members, key=lambda mc: mc[1].timestamp or "")
            oldest = min(members, key=lambda mc: mc[1].timestamp or "")
            if newest[0] != oldest[0] and newest[1].value != oldest[1].value:
                pair_key = (min(newest[0], oldest[0]), max(newest[0], oldest[0]))
                seen_pairs.add(pair_key)
                result = _classify_conflict(newest[0], oldest[0], newest[1], oldest[1], old_claims)
                if result is not None:
                    contradictions.append(result)
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, claim_a = members[a]
                j, claim_b = members[b]
                if i == j or claim_a.value == claim_b.value:
                    continue  # same value -> duplicate territory, not conflict
                pair_key = (min(i, j), max(i, j))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                result = _classify_conflict(i, j, claim_a, claim_b, old_claims)
                if result is not None:
                    contradictions.append(result)
    return contradictions


def _classify_conflict(i, j, claim_a, claim_b, old_claims):
    """Apply the resolution policy to one conflicting current-claim pair.

    old_claims maps entry index -> superseded (old-value) claims parsed from
    explicit-transition entries. A transition supersedes exactly the entry
    whose current value equals the transition's OLD value — never an entry
    that agrees with the transition's NEW value.
    """
    attr = claim_a.attribute
    subject = claim_a.subject

    # Stable attributes — always flag_human, never auto-resolve.
    if attr in STABLE_ATTRIBUTES:
        return {
            "pair": [i, j], "type": "stable_conflict",
            "resolution": "flag_human", "confidence": "high",
            "rule": "CONFLICT_STABLE_ATTRIBUTE",
            "reason": f"stable attribute '{attr}' conflicts — human review required",
        }

    is_state = attr in STATE_ATTRIBUTES
    explicit_a = claim_a.temporal_marker == "explicit_transition"
    explicit_b = claim_b.temporal_marker == "explicit_transition"

    if is_state and (explicit_a or explicit_b):
        # Determine which entry (if either) asserts the transition's OLD
        # value — that entry is superseded. Entries agreeing with the NEW
        # value are consistent, not conflicting.
        if explicit_a and not explicit_b:
            old_values = {c.value for c in old_claims.get(i, [])}
            if claim_b.value not in old_values:
                return None  # agrees with the transition's new value
            return _transition_result(i, j, claim_a, claim_b, subject, attr)
        if explicit_b and not explicit_a:
            old_values = {c.value for c in old_claims.get(j, [])}
            if claim_a.value not in old_values:
                return None  # agrees with the transition's new value
            return _transition_result(j, i, claim_b, claim_a, subject, attr)
        # Both transitions or neither — fall back to timestamps.
        newer_idx = _transition_newer(i, j, claim_a, claim_b)
        if newer_idx is not None:
            older_idx = j if newer_idx == i else i
            return {
                "pair": [i, j], "type": "state_change",
                "resolution": "recency_wins", "older_index": older_idx,
                "newer_index": newer_idx, "confidence": "high",
                "rule": "CONFLICT_EXPLICIT_TRANSITION",
                "reason": f"{subject} explicitly changes {attr} from the older value",
            }

    if is_state and claim_a.timestamp and claim_b.timestamp and \
       claim_a.timestamp != claim_b.timestamp:
        newer_idx = i if claim_a.timestamp > claim_b.timestamp else j
        older_idx = j if newer_idx == i else i
        return {
            "pair": [i, j], "type": "state_change",
            "resolution": "recency_wins", "older_index": older_idx,
            "newer_index": newer_idx, "confidence": "high",
            "rule": "CONFLICT_TIMESTAMPED_STATE_CHANGE",
            "reason": f"newer timestamp ({max(claim_a.timestamp, claim_b.timestamp)}) "
                      f"supersedes older {attr}",
        }

    # State attribute but chronology unknown, or unrecognized attribute.
    return {
        "pair": [i, j],
        "type": "possible_conflict" if is_state else "stable_conflict",
        "resolution": "flag_human",
        "confidence": "medium" if is_state else "low",
        "rule": "CONFLICT_SAME_KEY_DIFFERENT_VALUE",
        "reason": "same subject and attribute have different values, "
                  "but chronology is unknown",
    }


def _transition_result(newer_idx, older_idx, newer_claim, older_claim, subject, attr):
    """Build the recency_wins result for an explicit transition pair."""
    return {
        "pair": [min(newer_idx, older_idx), max(newer_idx, older_idx)],
        "type": "state_change",
        "resolution": "recency_wins", "older_index": older_idx,
        "newer_index": newer_idx, "confidence": "high",
        "rule": "CONFLICT_EXPLICIT_TRANSITION",
        "reason": f"{subject} explicitly changes {attr} from the older value",
    }


def _transition_newer(i, j, claim_a, claim_b):
    """For explicit transitions, the entry WITH the transition marker is
    newer; the plain assertion of the old value is superseded."""
    a_trans = claim_a.temporal_marker == "explicit_transition"
    b_trans = claim_b.temporal_marker == "explicit_transition"
    if a_trans and not b_trans:
        return i
    if b_trans and not a_trans:
        return j
    # Both transitions or neither — fall back to timestamps.
    if claim_a.timestamp and claim_b.timestamp and claim_a.timestamp != claim_b.timestamp:
        return i if claim_a.timestamp > claim_b.timestamp else j
    return None


# === Self-test ===

if __name__ == "__main__":
    test_entries = [
        "IrisBot: Linux 6.8, EPYC 4vCPU/16GB/193GB.",
        "Hindsight: localhost:8888, bank main. gpt-oss-120b via Cerebras.",
        "MCP tool_call JSON fix: Exa/Tavily invalid JSON -> call tool_describe first, retry.",
        "David prefers concise responses, 3-5 line summaries, markdown format.",
        "Hindsight port is 8888.",
        "Hindsight: port=8888",
        "Hindsight port is 9999.",
        "User prefers Python.",
        " user   prefers python ",
        "2026-07-01: Hindsight provider is direct.",
        "2026-08-01: Hindsight switched from direct to OpenRouter.",
        "Hindsight provider is OpenRouter.",
        "Previous model provider was z-ai direct, switched to OpenRouter.",
    ]
    print("=== Rule-Based Memory Heuristics Self-Test ===\n")
    if is_dry_run():
        print("DRY RUN mode active — analysis only, no mutations.\n")

    essential, offloadable = classify_importance(test_entries)
    print("Importance classification:")
    print(f"  Essential: {essential}")
    print(f"  Offloadable: {offloadable}\n")

    dups = semantic_dedup(test_entries)
    print("Duplicate groups:")
    for g in dups:
        print(f"  canonical={g['canonical']} dups={g['duplicates']} "
              f"confidence={g['confidence']} rule={g['rule']}")
    if not dups:
        print("  (none)")
    print()

    conflicts = detect_contradictions(test_entries)
    print("Contradiction candidates:")
    for c in conflicts:
        print(f"  pair={c['pair']} type={c['type']} resolution={c['resolution']} "
              f"rule={c['rule']}")
        print(f"    reason: {c['reason']}")
    if not conflicts:
        print("  (none)")
