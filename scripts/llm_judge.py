#!/usr/bin/env python3
"""Gemini 2.5 Flash Lite judge for the L1 -> L2 offload decision (v3.2).

v3.0 removed LLM-as-judge entirely in favor of deterministic rule-based
heuristics. v3.2 reintroduces a *scoped* LLM judge for exactly one decision
point: the L1 -> L2 offload gate. The rule-based heuristics remain the hard
gate (quarantine, pins, essential prefixes, offload patterns, weighted
scoring); the judge then reviews the entries the rules marked OFFLOADABLE
and confirms which of them are genuinely low-value — i.e. safe to move to
Hindsight (L2) — before any L1 removal happens.

Design invariants:
  - Rules first, judge second. The judge can only VETO an offload
    (keep entry in L1), never unlock one. Hard-kept and quarantined
    entries never reach the judge at all.
  - Fail-safe, not fail-loud. Any judge failure (API down, timeout,
    malformed response, missing key) falls back to the rule-based
    result — offload proceeds as in v3.1. The judge is an enhancement,
    not a dependency.
  - Privacy. Content is PII-redacted (memory_records.redact_pii) and
    sensitive entries (credential-like content/tags) are excluded via
    memory_records.should_exclude_from_judging — they never leave the host.
  - Attribution. Calls OpenRouter with X-Title / HTTP-Referer set to the
    project name (Green-Needle-Tech/agent-memory-optimization), not localhost.
  - Zero cost when disabled: JUDGE_ENABLED=0 (default when no key is found)
    short-circuits without any network call.

Model: gemini-2.5-flash-lite via OpenRouter (cheap, fast, non-reasoning).
API key resolution order (v3.3, location-aware — see paths.py):
  1. OPENROUTER_API_KEY env var
  2. $HERMES_HOME/.env (HERMES_HOME resolved from the existing deployment,
     not just the current user's home)
  3. ~/.hermes/.env
"""

import json
import os
import re
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
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-2.5-flash-lite")
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "30"))
JUDGE_MAX_ENTRIES = int(os.environ.get("JUDGE_MAX_ENTRIES", "40"))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0"))
PROJECT_NAME = os.environ.get(
    "JUDGE_PROJECT_NAME", "Green-Needle-Tech/agent-memory-optimization"
)

# Verdict labels the judge may return per entry (case-insensitive).
VERDICT_OFFLOAD = "offload"
VERDICT_KEEP = "keep"


def load_api_key():
    """Resolve the OpenRouter API key (v3.3 location-aware, see paths.py).

    Order: env var -> $HERMES_HOME/.env -> ~/.hermes/.env.
    Returns None when no key is configured (judge disabled by absence).
    """
    key = paths.read_env_var("OPENROUTER_API_KEY", hermes_home=HERMES_HOME)
    return key or None


def is_available():
    """True when the judge is enabled AND an API key is configured."""
    return JUDGE_ENABLED and bool(load_api_key())


def _build_prompt(candidates):
    """Build the judge prompt from offloadable candidates.

    Each candidate is (index, content). Content is PII-redacted and
    truncated; sensitive entries are excluded entirely (never sent).
    Returns (prompt, sent_indices) — sent_indices maps judge positions
    back to candidate indices.
    """
    try:
        from memory_records import MemoryRecord, redact_pii, should_exclude_from_judging
    except ImportError:
        # memory_records lives in the same scripts/ dir; sys.path is set by
        # the importing script. Fall back to a local import via __file__.
        import sys
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

    if not lines:
        return None, []

    prompt = (
        "You are the memory-retention judge for an AI agent's local memory (L1).\n"
        "L1 is a tiny always-injected context window (~2KB). Entries that are not "
        "needed EVERY turn should be offloaded to a semantic-recall store (L2) "
        "where they can be retrieved when relevant.\n\n"
        "A rule-based pre-filter has already marked the entries below as "
        "OFFLOADABLE (low-value: historical state, completed tasks, one-time "
        "lessons, maintenance noise, model/provider history). Your job is to "
        "confirm or veto each one.\n\n"
        "Rules:\n"
        "- offload: the entry is durable but rarely needed this-turn "
        "(historical facts, past incidents, completed work, preferences that "
        "only matter in specific situations)\n"
        "- keep: the entry is volatile/operational state that will go stale "
        "(container states, ports, current model, cron job IDs), is trivially "
        "re-discoverable, OR is so critical that recall latency is unacceptable "
        "(active credentials, live incident, safety rule)\n"
        "- When in doubt, choose keep — L1 removal is irreversible this cycle.\n\n"
        "Entries:\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON object, no prose:\n"
        '{"verdicts": [{"id": <number>, "verdict": "offload" | "keep"}]}'
    )
    return prompt, sent_indices


def _parse_verdicts(text, sent_indices):
    """Parse the judge response into {candidate_index: verdict}.

    Tolerant of markdown fences and stray prose around the JSON object.
    Unknown/missing ids default to KEEP (conservative).
    """
    # Strip markdown code fences if present.
    text = re.sub(r"```(?:json)?", "", text)
    # Find the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    raw = data.get("verdicts")
    if not isinstance(raw, list):
        return None

    verdicts = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if raw_id is None:
            continue
        try:
            pos = int(raw_id)
        except (TypeError, ValueError):
            continue
        if not (0 <= pos < len(sent_indices)):
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in (VERDICT_OFFLOAD, VERDICT_KEEP):
            verdict = VERDICT_KEEP  # conservative default
        verdicts[sent_indices[pos]] = verdict
    return verdicts


def _call_openrouter(prompt, api_key):
    """POST the prompt to OpenRouter. Returns raw response text or None."""
    body = json.dumps({
        "model": JUDGE_MODEL,
        "temperature": JUDGE_TEMPERATURE,
        "messages": [
            {"role": "system", "content": "You are a precise memory-retention judge. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
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
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


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

    Fail-safe: on ANY failure (disabled, no candidates, API error, parse
    error, empty verdicts) returns the full input as confirmed with
    status "fallback" — i.e. the v3.1 rule-based behavior.
    """
    if not JUDGE_ENABLED:
        return [i for i, _ in entries], [], "disabled"

    api_key = load_api_key()
    if not api_key:
        return [i for i, _ in entries], [], "disabled"

    prompt, sent_indices = _build_prompt(entries)
    if prompt is None:
        # Nothing safe to send (all sensitive or empty) — fall back to rules.
        return [i for i, _ in entries], [], "skipped"

    text = _call_openrouter(prompt, api_key)
    if text is None:
        return [i for i, _ in entries], [], "fallback"

    verdicts = _parse_verdicts(text, sent_indices)
    if not verdicts:
        return [i for i, _ in entries], [], "fallback"

    confirmed, vetoed = [], []
    for idx, _ in entries:
        # Entries not sent to the judge (sensitive/excluded/over cap)
        # keep the rule-based verdict: offload.
        if verdicts.get(idx, VERDICT_OFFLOAD) == VERDICT_OFFLOAD:
            confirmed.append(idx)
        else:
            vetoed.append(idx)
    return confirmed, vetoed, "ok"
