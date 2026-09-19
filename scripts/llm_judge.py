#!/usr/bin/env python3
"""TypeSafe Jev judge for L1 memory decisions (v3.6).

v3.6 replaces the OpenRouter chat-completions judge (Gemini 2.5 Flash Lite,
v3.2) with TypeSafe AI's System One structured-decision API and the Jev
model (jev-1.13.0). Jev is a decision model, not a text generator: we send
a `state` plus a map of typed questions and get back typed answers with
probability distributions — no prompt scaffolding, no JSON parsing of
generated prose, no temperature tuning.

Two decision points, both rules-first and fail-safe:

1. Importance classification (`judge_importance`) — reviews only the
   entries the rule-based heuristics classified via *weighted scoring*
   (no hard rule matched). Hard-kept (pins, essential prefixes, quarantine)
   and hard-offloaded (explicit tags/patterns) entries are never sent.
   Jev may move a weighted-band entry in either direction
   (essential <-> offloadable).

2. Offload gate (`judge_offload_candidates`) — reviews only entries the
   rules already marked OFFLOADABLE and can only VETO an offload (keep in
   L1); it can never unlock one.

Design invariants (unchanged from v3.2):
  - Rules first, judge second. The rule-based heuristics remain the hard
    gate. The judge is an enhancement, not a dependency.
  - Fail-safe, not fail-loud. Any judge failure (no key, API down,
    timeout, malformed response) falls back to the rule-based result.
    JUDGE_ENABLED=0 disables with zero network calls.
  - Confidence-gated actions. Per TypeSafe's confidence semantics
    ("the answer says what, confidence says whether to act"), a KEEP veto
    at the offload gate is only applied when confidence >=
    JUDGE_MIN_CONFIDENCE (default 0.6). Low-confidence vetoes are ignored
    — the rule-based offload verdict stands.
  - Privacy. Content is PII-redacted (memory_records.redact_pii) and
    sensitive entries (credential-like content/tags) are excluded via
    memory_records.should_exclude_from_judging — they never leave the host.
  - Attribution. Requests carry X-Title / HTTP-Referer set to the project
    name (Green-Needle-Tech/agent-memory-optimization), never localhost.

Model: jev-1.13.0 (alias jev-latest) via api.typesafe.ai/v1/systemone.
API key resolution order (location-aware — see paths.py):
  1. TYPESAFE_API_KEY env var
  2. $HERMES_HOME/.env (HERMES_HOME resolved from the existing deployment)
  3. ~/.hermes/.env
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

sys_path_parent = str(Path(__file__).parent)
import sys  # noqa: E402

if sys_path_parent not in sys.path:
    sys.path.insert(0, sys_path_parent)
import paths  # noqa: E402  (location-aware resolution, ships with this repo)

# === Config (environment-overridable) ===
HERMES_HOME = paths.resolve_hermes_home()
JUDGE_ENABLED = os.environ.get("JUDGE_ENABLED", "1") not in ("0", "false", "no")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "jev-1.13.0")
TYPESAFE_URL = os.environ.get(
    "TYPESAFE_URL", "https://api.typesafe.ai/v1/systemone"
)
JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "30"))
JUDGE_MAX_ENTRIES = int(os.environ.get("JUDGE_MAX_ENTRIES", "40"))
JUDGE_MIN_CONFIDENCE = float(os.environ.get("JUDGE_MIN_CONFIDENCE", "0.6"))
PROJECT_NAME = os.environ.get(
    "JUDGE_PROJECT_NAME", "Green-Needle-Tech/agent-memory-optimization"
)

# Verdict labels (case-insensitive normalization applied on read).
VERDICT_OFFLOAD = "offload"
VERDICT_KEEP = "keep"
IMPORTANCE_ESSENTIAL = "essential"
IMPORTANCE_OFFLOADABLE = "offloadable"

# Criteria follow TypeSafe Choice best practices (docs.typesafe.ai
# /primitives/choice): option descriptions separate confusable options
# with WHAT / NOT FOR / EXAMPLES.
OFFLOAD_CRITERIA = {
    VERDICT_OFFLOAD: (
        "WHAT: durable but rarely needed this-turn — historical facts, "
        "past incidents, completed work, provider/model history, one-time "
        "lessons, preferences that only matter in specific situations. "
        "NOT FOR: anything needed on every turn or likely to go stale soon. "
        "EXAMPLES: old provider ranking; migration completed last month; "
        "a debugging lesson from a fixed bug"
    ),
    VERDICT_KEEP: (
        "WHAT: volatile/operational state that will go stale (container "
        "states, ports, current model, cron job IDs), trivially "
        "re-discoverable info, OR so critical that recall latency is "
        "unacceptable (active credentials, live incident, safety rule). "
        "NOT FOR: settled historical facts nobody needs every turn. "
        "EXAMPLES: service currently running on port 8642; live incident "
        "being debugged; pinned safety rule"
    ),
}

IMPORTANCE_CRITERIA = {
    IMPORTANCE_ESSENTIAL: (
        "WHAT: must sit in the tiny always-injected L1 window (~2KB) — "
        "environment facts, standing conventions, active config, tool "
        "quirks that prevent repeated work every turn. "
        "NOT FOR: history, completed tasks, situational knowledge. "
        "EXAMPLES: host specs; Hindsight endpoint; a recurring tool "
        "workaround"
    ),
    IMPORTANCE_OFFLOADABLE: (
        "WHAT: durable knowledge that is only relevant in some situations "
        "— retrievable on demand from a semantic-recall store (L2). "
        "NOT FOR: facts the agent needs on every single turn. "
        "EXAMPLES: past provider evaluations; one-time debugging lessons; "
        "completed maintenance reports"
    ),
}

GATE_INSTRUCTIONS = (
    "L1 is a tiny always-injected context window. Entries not needed "
    "EVERY turn should move to a semantic-recall store (L2). A rule-based "
    "pre-filter already marked these entries OFFLOADABLE. Confirm or veto "
    "each offload. When in doubt, choose keep — L1 removal is "
    "irreversible this cycle."
)

IMPORTANCE_INSTRUCTIONS = (
    "L1 is a tiny always-injected context window (~2KB). For each entry, "
    "decide whether it must stay in L1 (essential) or can live in the "
    "on-demand recall store L2 (offloadable). Judge only from the entry "
    "text; when in doubt, choose essential."
)


def load_api_key():
    """Resolve the TypeSafe API key (location-aware, see paths.py).

    Order: env var -> $HERMES_HOME/.env -> ~/.hermes/.env.
    Returns None when no key is configured (judge disabled by absence).
    """
    key = paths.read_env_var("TYPESAFE_API_KEY", hermes_home=HERMES_HOME)
    return key or None


def is_available():
    """True when the judge is enabled AND an API key is configured."""
    return JUDGE_ENABLED and bool(load_api_key())


def _prepare(candidates):
    """Filter and redact candidates for sending to the cloud judge.

    Each candidate is (index, content). Sensitive entries are excluded
    entirely; the rest are PII-redacted and truncated to 300 chars.
    Returns (state_lines, sent_indices) — sent_indices maps state
    positions back to candidate indices.
    """
    try:
        from memory_records import MemoryRecord, redact_pii, should_exclude_from_judging
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from memory_records import MemoryRecord, redact_pii, should_exclude_from_judging

    lines = []
    sent_indices = []
    for idx, content in candidates[:JUDGE_MAX_ENTRIES]:
        rec = MemoryRecord(id=str(idx), content=content, fact_type="world")
        if should_exclude_from_judging(rec):
            continue  # sensitive entry — never sent to the cloud judge
        redacted = redact_pii(content)[:300]
        sent_indices.append(idx)
        lines.append(f"{len(sent_indices) - 1}: {redacted}")
    return lines, sent_indices


def _call_systemone(state, questions):
    """POST one request to TypeSafe System One.

    Args:
        state: string state (numbered entry list).
        questions: dict of question id -> typed question object.

    Returns the `answers` dict ({qid: {"choice": ..., "confidence": ...}})
    or None on any failure (network, HTTP, malformed body).
    """
    body = json.dumps({"model": JUDGE_MODEL, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        TYPESAFE_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {load_api_key()}",
            # Attribution by project name + GitHub link (GNT policy) — never localhost.
            "X-Title": PROJECT_NAME,
            "HTTP-Referer": f"https://github.com/{PROJECT_NAME}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=JUDGE_TIMEOUT) as resp:  # nosec B310
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    except json.JSONDecodeError:
        return None
    answers = data.get("answers")
    return answers if isinstance(answers, dict) else None


def _answer_for(answers, qid):
    """Extract (choice, confidence) for one question id. None when absent."""
    ans = answers.get(qid)
    if not isinstance(ans, dict):
        return None
    choice = str(ans.get("choice", "")).strip().lower()
    if not choice:
        return None
    try:
        confidence = float(ans.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    return choice, confidence


def judge_importance(candidates):
    """Jev importance classification for weighted-band entries.

    Args:
        candidates: list of (index, content) tuples — entries the
            rule-based heuristics classified via weighted scoring (no
            hard rule matched). Hard-gated entries must never be passed.

    Returns:
        (verdicts, status) where verdicts maps candidate index ->
        "essential" | "offloadable" for the entries Jev judged (missing
        entries keep their rule verdict), and status is "disabled" |
        "skipped" | "ok" | "fallback".

    Fail-safe: on ANY failure returns ({}, "fallback") — every entry
    keeps its rule-based disposition.
    """
    if not JUDGE_ENABLED:
        return {}, "disabled"
    if not load_api_key():
        return {}, "disabled"

    lines, sent_indices = _prepare(candidates)
    if not lines:
        return {}, "skipped"

    questions = {
        f"e{pos}": {
            "type": "choice",
            "instructions": IMPORTANCE_INSTRUCTIONS,
            "criteria": IMPORTANCE_CRITERIA,
        }
        for pos in range(len(lines))
    }
    answers = _call_systemone("\n".join(lines), questions)
    if answers is None:
        return {}, "fallback"

    verdicts = {}
    for pos, idx in enumerate(sent_indices):
        result = _answer_for(answers, f"e{pos}")
        if result is None:
            continue
        choice, _conf = result
        if choice in (IMPORTANCE_ESSENTIAL, IMPORTANCE_OFFLOADABLE):
            verdicts[idx] = choice
        # Unknown choice: entry keeps its rule-based verdict (conservative).
    return verdicts, "ok" if verdicts else "fallback"


def judge_offload_candidates(entries):
    """Judge which rule-offloadable entries should actually be offloaded.

    Args:
        entries: list of (index, content) tuples — the entries the
            rule-based heuristics marked OFFLOADABLE. Indices refer to
            positions in the caller's full entry list.

    Returns:
        (confirmed_indices, vetoed_indices, status) where:
        - confirmed_indices: entry indices the judge confirmed as offload
        - vetoed_indices: entry indices the judge kept in L1
        - status: "disabled" | "skipped" | "ok" | "fallback"

    Confidence gate: a KEEP verdict only vetoes the offload when its
    confidence >= JUDGE_MIN_CONFIDENCE. Low-confidence keeps are ignored
    (the rule-based offload stands) — TypeSafe's "confidence says whether
    to act" semantics.

    Fail-safe: on ANY failure (disabled, no candidates, API error,
    malformed answers, empty verdicts) returns the full input as
    confirmed with status "fallback" — i.e. the rule-based behavior.
    """
    if not JUDGE_ENABLED:
        return [i for i, _ in entries], [], "disabled"
    if not load_api_key():
        return [i for i, _ in entries], [], "disabled"

    lines, sent_indices = _prepare(entries)
    if not lines:
        # Nothing safe to send (all sensitive or empty) — fall back to rules.
        return [i for i, _ in entries], [], "skipped"

    questions = {
        f"e{pos}": {
            "type": "choice",
            "instructions": GATE_INSTRUCTIONS,
            "criteria": OFFLOAD_CRITERIA,
        }
        for pos in range(len(lines))
    }
    answers = _call_systemone("\n".join(lines), questions)
    if answers is None:
        return [i for i, _ in entries], [], "fallback"

    confirmed, vetoed = [], []
    any_verdict = False
    for idx, _ in entries:
        result = None
        if idx in sent_indices:
            result = _answer_for(answers, f"e{sent_indices.index(idx)}")
        if result is None:
            # Not sent (sensitive/excluded/over cap) or no answer:
            # keep the rule-based verdict — offload.
            confirmed.append(idx)
            continue
        choice, confidence = result
        any_verdict = True
        if choice == VERDICT_KEEP and confidence >= JUDGE_MIN_CONFIDENCE:
            vetoed.append(idx)
        else:
            # offload confirmed, or low-confidence keep (act-gate failed).
            confirmed.append(idx)
    if not any_verdict:
        return [i for i, _ in entries], [], "fallback"
    return confirmed, vetoed, "ok"
