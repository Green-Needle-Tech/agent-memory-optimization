#!/usr/bin/env python3
"""LLM-as-judge module for agent memory optimization.

Provides three LLM-driven operations that replace brittle rule-based heuristics:
  1. classify_importance(entries) -> essential vs offloadable
  2. semantic_dedup(entries)      -> near-duplicate groups (LLM judge, not word overlap)
  3. detect_contradictions(entries) -> entity-drift pairs with resolution policy

Design principles (grounded in 2026 agent-memory research):
  - **LLM-optional** (MenteDB llm_consolidation pattern): degrades to rule-based
    fallback if the LLM endpoint is unavailable. Callers with no LLM simply get
    the old heuristic behavior — the engine stays functional.
  - **Batch consolidation** (LycheeMemory V2 / RecMem): all entries are sent in
    a single LLM call, not one call per entry. Reduces token cost 75-87% vs
    eager per-turn consolidation.
  - **Recency-wins contradiction resolution** (Hindsight blog, May 2026): for
    state changes, newer facts supersede older ones via invalidation (not
    deletion). For stable attributes that conflict, flag for human review.
  - **Importance scoring** (Park et al. Generative Agents + Hindsight fact
    extraction): the LLM rates each entry's long-term retention value, acting
    as the write-time quality gate.

v2.2.1 (Sep 2026): Strict LLM response validation.
  - All indices validated: 0 <= i < n, no negative indices, no duplicates
  - Canonical must not appear in its own duplicates list
  - Duplicate groups must not overlap
  - Contradiction pairs must have exactly 2 distinct indices in range
  - newer_index must be one of the pair
  - Missing/malformed entries default to KEEP LOCALLY (not offload)
  - Invalid responses rejected entirely (no silent repair of destructive output)

Usage:
  from llm_judge import classify_importance, semantic_dedup, detect_contradictions

  essential, offloadable = classify_importance(memory_entries)
  dup_groups = semantic_dedup(memory_entries)
  contradictions = detect_contradictions(memory_entries)

Configuration:
  Reads OPENROUTER_API_KEY from ~/.hermes/.env (or environment).
  Uses z-ai/glm-5.2 by default.
  Override via env: LLM_JUDGE_MODEL, LLM_JUDGE_BASE_URL, LLM_JUDGE_API_KEY.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

# === Config (environment-overridable) ===
DEFAULT_MODEL = os.environ.get("LLM_JUDGE_MODEL", "z-ai/glm-5.2")
DEFAULT_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MAX_TOKENS = 6000
DEFAULT_TEMPERATURE = 0.1  # low temp for classification consistency
REQUEST_TIMEOUT = 60
BATCH_SIZE = 30  # max entries per LLM call — larger batches overflow the
                 # response token budget, the JSON truncates, parsing fails,
                 # and the caller silently degrades to rule-based fallback
                 # (observed live: 85-entry batch -> fallback flagged 30+ pairs)

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


def _chunk(items, size=BATCH_SIZE):
    """Yield successive size-sized chunks from items."""
    for i in range(0, len(items), size):
        yield i, items[i:i + size]


def _load_api_key():
    """Load OpenRouter API key from .env file or environment."""
    key = os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_file = HERMES_HOME / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _llm_chat(messages, model=None, max_tokens=None):
    """Call LLM chat completions API. Returns content string or None on failure.

    LLM-optional: returns None on any error (network, auth, parse).
    Caller must handle None as "LLM unavailable, use fallback."
    """
    api_key = _load_api_key()
    if not api_key:
        return None

    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS
    base_url = DEFAULT_BASE_URL

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": DEFAULT_TEMPERATURE,
    }

    # HTTP-Referer and X-Title for OpenRouter attribution
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Green-Needle-Tech/agent-memory-optimization",
        "X-Title": "agent-memory-optimization",
    }

    try:
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read())
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return None


def _parse_json_response(text):
    """Extract JSON from LLM response (handles markdown fences, prose)."""
    if not text:
        return None
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first { ... } or [ ... ] block
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return None


# === Strict validation helpers (v2.2.1) ===

def _validate_index(idx, n, context="index"):
    """Validate a single index: must be int, 0 <= idx < n.

    Negative indices are rejected explicitly — Python accepts them for list
    access, but an LLM-hallucinated -1 targeting the last item is destructive.
    """
    if not isinstance(idx, int):
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
    if idx < 0 or idx >= n:
        return None
    return idx


def _validate_indices_list(indices, n, context="indices"):
    """Validate a list of indices. Returns cleaned list or None if any invalid.

    All indices must be in range [0, n). No negative indices allowed.
    """
    result = []
    for idx in indices:
        validated = _validate_index(idx, n, context)
        if validated is None:
            return None
        result.append(validated)
    return result


def _validate_dedup_group(group, n):
    """Validate a single dedup group from LLM output.

    Rules:
    - canonical must be a valid index in [0, n)
    - duplicates must all be valid indices in [0, n)
    - canonical must NOT appear in duplicates (distinct)
    - at least one duplicate required
    - no duplicate index appears more than once

    Returns cleaned group dict or None if invalid.
    """
    if not isinstance(group, dict):
        return None
    if "canonical" not in group or "duplicates" not in group:
        return None

    canonical = _validate_index(group["canonical"], n, "canonical")
    if canonical is None:
        return None

    if not isinstance(group["duplicates"], list):
        return None

    dups = []
    seen = {canonical}  # canonical must not appear in duplicates
    for d in group["duplicates"]:
        validated = _validate_index(d, n, "duplicate")
        if validated is None:
            return None
        if validated in seen:
            return None  # duplicate index in same group or == canonical
        seen.add(validated)
        dups.append(validated)

    if not dups:
        return None  # must have at least one duplicate

    return {"canonical": canonical, "duplicates": dups}


def _validate_dedup_groups(groups, n):
    """Validate a list of dedup groups. Reject overlapping groups.

    A duplicate index may appear in only one group. If any group is invalid,
    the entire response is rejected — we do NOT silently repair destructive
    output.
    """
    validated = []
    used_indices = set()  # track all indices across groups

    for g in groups:
        clean = _validate_dedup_group(g, n)
        if clean is None:
            return None  # reject entire response on any invalid group
        all_group_indices = {clean["canonical"]} | set(clean["duplicates"])
        if all_group_indices & used_indices:
            return None  # overlap between groups — reject
        used_indices |= all_group_indices
        validated.append(clean)

    return validated


def _validate_contradiction_pair(pair_data, n):
    """Validate a single contradiction entry from LLM output.

    Rules:
    - pair must have exactly 2 elements
    - both must be valid indices in [0, n)
    - the two indices must be distinct (no self-pairs)
    - newer_index (if present) must be one of the pair
    - type must be one of the allowed values
    - resolution must be one of the allowed values

    Returns cleaned dict or None if invalid.
    """
    if not isinstance(pair_data, dict):
        return None
    if "pair" not in pair_data:
        return None

    raw_pair = pair_data["pair"]
    if not isinstance(raw_pair, list) or len(raw_pair) != 2:
        return None

    a = _validate_index(raw_pair[0], n, "pair[0]")
    b = _validate_index(raw_pair[1], n, "pair[1]")
    if a is None or b is None:
        return None
    if a == b:
        return None  # self-pair — meaningless

    c_type = pair_data.get("type", "unknown")
    resolution = pair_data.get("resolution", "flag_human")

    # Validate type and resolution are recognized values
    valid_types = {"state_change", "stable_conflict", "meta_noise", "possible_drift", "unknown"}
    valid_resolutions = {"recency_wins", "flag_human", "invalidate_meta"}
    if c_type not in valid_types:
        c_type = "unknown"
    if resolution not in valid_resolutions:
        resolution = "flag_human"  # safe default

    newer_index = pair_data.get("newer_index")
    if newer_index is not None:
        newer_index = _validate_index(newer_index, n, "newer_index")
        if newer_index is None or newer_index not in (a, b):
            newer_index = None  # invalid — don't use, fall back to flag_human

    # If recency_wins but no valid newer_index, downgrade to flag_human
    if resolution == "recency_wins" and newer_index is None:
        resolution = "flag_human"

    return {
        "pair": [a, b],
        "type": c_type,
        "resolution": resolution,
        "reason": str(pair_data.get("reason", ""))[:200],  # cap reason length
        "newer_index": newer_index,
    }


def _validate_contradictions(contradictions, n):
    """Validate a list of contradiction entries. Deduplicate pairs.

    Each unordered pair may appear at most once. If any entry is invalid,
    the entire response is rejected.
    """
    validated = []
    seen_pairs = set()

    for c in contradictions:
        clean = _validate_contradiction_pair(c, n)
        if clean is None:
            return None  # reject entire response
        key = tuple(sorted(clean["pair"]))
        if key in seen_pairs:
            continue  # deduplicate — same pair reported twice
        seen_pairs.add(key)
        validated.append(clean)

    return validated


def _validate_classify_response(parsed, n_entries):
    """Validate classify_importance LLM response.

    Rules:
    - essential and offloadable are lists of indices
    - all indices in [0, n_entries)
    - no overlap between essential and offloadable
    - unclassified entries default to essential (keep locally — safe default)

    Returns (essential, offloadable) or None if response is unparseable.
    """
    if not isinstance(parsed, dict):
        return None
    if "essential" not in parsed or "offloadable" not in parsed:
        return None

    raw_essential = parsed.get("essential", [])
    raw_offloadable = parsed.get("offloadable", [])

    if not isinstance(raw_essential, list) or not isinstance(raw_offloadable, list):
        return None

    essential = _validate_indices_list(raw_essential, n_entries, "essential")
    offloadable = _validate_indices_list(raw_offloadable, n_entries, "offloadable")

    if essential is None or offloadable is None:
        return None  # any invalid index → reject entirely

    # Check for overlap
    if set(essential) & set(offloadable):
        return None  # overlap → reject

    # Unclassified indices default to ESSENTIAL (keep locally — safe default)
    all_indices = set(range(n_entries))
    classified = set(essential) | set(offloadable)
    unclassified = all_indices - classified
    if unclassified:
        essential.extend(sorted(unclassified))

    return essential, offloadable


# === Rule-based fallbacks (used when LLM unavailable) ===

# Entries that must stay in L1 local memory (every-turn context)
_FALLBACK_ESSENTIAL_PREFIXES = [
    "IrisBot:",
    "Hindsight:",
    "MCP tool_call",
    "HTML via execute_code",
    "Vision:",
    "Search fallback:",
]


def _fallback_classify(entries):
    """Rule-based importance classification (prefix matching)."""
    essential, offloadable = [], []
    for i, entry in enumerate(entries):
        is_essential = any(entry.strip().startswith(p) for p in _FALLBACK_ESSENTIAL_PREFIXES)
        if is_essential:
            essential.append(i)
        else:
            offloadable.append(i)
    return essential, offloadable


def _fallback_dedup(entries):
    """Rule-based semantic dedup (word overlap >60%)."""
    groups = []
    used = set()
    for i, entry_a in enumerate(entries):
        if i in used:
            continue
        group = [i]
        used.add(i)
        words_a = set(entry_a.lower().split())
        for j in range(i + 1, len(entries)):
            if j in used:
                continue
            words_b = set(entries[j].lower().split())
            if words_a and words_b:
                overlap = len(words_a & words_b) / len(words_a)
                if overlap > 0.6:
                    group.append(j)
                    used.add(j)
        if len(group) > 1:
            groups.append({"canonical": group[0], "duplicates": group[1:]})
    return groups


def _fallback_contradictions(entries):
    """Rule-based contradiction detection (keyword-based entity drift)."""
    contradictions = []
    drift_keywords = ["provider", "model", "url", "port", "version", "api"]
    for i, entry_a in enumerate(entries):
        for j in range(i + 1, len(entries)):
            entry_b = entries[j]
            # Check for entity drift: same keyword domain, different values
            words_a = set(entry_a.lower().split())
            words_b = set(entry_b.lower().split())
            shared_drift = any(k in words_a and k in words_b for k in drift_keywords)
            if shared_drift:
                # Check if values differ (simplified: different entries mentioning same domain)
                if entry_a.strip() != entry_b.strip():
                    contradictions.append({
                        "pair": [i, j],
                        "type": "possible_drift",
                        "resolution": "flag_human",
                    })
    return contradictions


# === LLM-driven operations ===

def classify_importance(entries, context="L1 local memory for a Hermes AI agent"):
    """Classify memory entries as essential or offloadable using LLM.

    Uses LLM to rate each entry's long-term retention value (importance scoring,
    per Park et al. Generative Agents + Hindsight fact-extraction-as-filter).

    v2.2.1: Strict validation — all indices validated, no overlap allowed,
    unclassified entries default to ESSENTIAL (keep locally, safe default).
    Invalid LLM responses cause fallback to rule-based for that chunk.

    Args:
        entries: list of memory entry strings
        context: description of the memory store for the LLM

    Returns:
        (essential_indices, offloadable_indices) — lists of integer indices
        into the input list. Falls back to rule-based prefix matching if LLM
        is unavailable or response is invalid.
    """
    if not entries:
        return [], []

    essential, offloadable = [], []
    for offset, chunk in _chunk(entries):
        n_chunk = len(chunk)
        # Build a single batched prompt (LycheeMemory segment-level batching)
        numbered = "\n".join(f"[{offset + k}] {e}" for k, e in enumerate(chunk))
        prompt = (
            f"You are a memory optimization judge for an AI agent's local memory ({context}).\n"
            f"Local memory is injected every turn — every character costs attention.\n"
            f"Classify each entry as 'essential' (must stay local: host specs, active config, "
            f"tool quirks, recurring fixes) or 'offloadable' (stale-in-7-days facts, version numbers, "
            f"one-time lessons, historical state — better stored in semantic recall).\n\n"
            f"Entries:\n{numbered}\n\n"
            f"Respond with ONLY a JSON object: {{\"essential\": [indices], \"offloadable\": [indices]}}\n"
            f"Do not include any prose, explanation, or markdown."
        )

        resp = _llm_chat([{"role": "user", "content": prompt}])
        parsed = _parse_json_response(resp)

        # v2.2.1: strict validation of LLM response
        validated = _validate_classify_response(parsed, n_chunk) if parsed else None

        if validated is not None:
            fe, fo = validated
            essential.extend(offset + i for i in fe)
            offloadable.extend(offset + i for i in fo)
        else:
            # Fallback for this chunk: rule-based
            fe, fo = _fallback_classify(chunk)
            essential.extend(offset + i for i in fe)
            offloadable.extend(offset + i for i in fo)

    # Final validation: all indices accounted for
    all_indices = set(range(len(entries)))
    classified = set(essential) | set(offloadable)
    unclassified = all_indices - classified
    # v2.2.1: unclassified go to ESSENTIAL (keep locally — safe default)
    essential.extend(sorted(unclassified))
    # Remove any overlap (shouldn't happen, but belt-and-suspenders)
    offloadable_set = set(offloadable) - set(essential)
    offloadable = sorted(offloadable_set)
    return essential, offloadable


def semantic_dedup(entries):
    """Find semantic near-duplicates using LLM judge.

    Replaces word-overlap dedup. The LLM identifies entries that express the
    same fact with different wording (e.g., "User prefers Python" vs "User's
    primary language is Python") — these evade exact-match dedup but waste
    retrieval slots.

    v2.2.1: Strict validation — indices bounds-checked, no negative indices,
    canonical distinct from duplicates, no overlapping groups. Invalid
    responses cause fallback to word-overlap for that chunk.

    Args:
        entries: list of memory entry strings

    Returns:
        List of duplicate groups: [{"canonical": idx, "duplicates": [idx, ...]}, ...]
        Falls back to word-overlap dedup (>60%) if LLM unavailable or invalid.
    """
    if len(entries) < 2:
        return []

    groups = []
    for offset, chunk in _chunk(entries):
        n_chunk = len(chunk)
        numbered = "\n".join(f"[{offset + k}] {e}" for k, e in enumerate(chunk))
        prompt = (
            f"You are a semantic deduplication judge for AI agent memory.\n"
            f"Identify groups of entries that express the SAME underlying fact or preference.\n"
            f"This includes verbatim or near-verbatim copies, and same fact with different wording\n"
            f"(e.g., \"User prefers Python\" vs \"User's primary language is Python\"). Only group\n"
            f"genuine duplicates — not entries that merely share a topic.\n\n"
            f"Entries:\n{numbered}\n\n"
            f"Respond with ONLY a JSON array of groups: "
            f"[{{\"canonical\": idx, \"duplicates\": [idx, ...]}}, ...]\n"
            f"Each group must have exactly one canonical (the best-worded entry) and at least one duplicate.\n"
            f"If no duplicates exist, respond with: []\n"
            f"Do not include any prose or markdown."
        )

        resp = _llm_chat([{"role": "user", "content": prompt}])
        parsed = _parse_json_response(resp)

        if parsed is not None and isinstance(parsed, list):
            # v2.2.1: strict validation of all groups
            validated = _validate_dedup_groups(parsed, n_chunk)
            if validated is not None:
                for g in validated:
                    groups.append({
                        "canonical": offset + g["canonical"],
                        "duplicates": [offset + d for d in g["duplicates"]],
                    })
            else:
                # Invalid response — fallback for this chunk
                for g in _fallback_dedup(chunk):
                    groups.append({
                        "canonical": offset + g["canonical"],
                        "duplicates": [offset + d for d in g["duplicates"]],
                    })
        else:
            # Fallback for this chunk: word-overlap
            for g in _fallback_dedup(chunk):
                groups.append({
                    "canonical": offset + g["canonical"],
                    "duplicates": [offset + d for d in g["duplicates"]],
                })

    return groups


def detect_contradictions(entries):
    """Detect contradictory memory entries using LLM.

    Identifies entity-drift pairs (old vs new state of the same fact) and
    stable-attribute conflicts. Resolution policy (Hindsight blog, May 2026):
    - State changes → recency wins (invalidate old, keep new)
    - Stable attributes that conflict → flag for human review

    v2.2.1: Strict validation — pairs must have exactly 2 distinct in-range
    indices, newer_index must be one of the pair, type/resolution validated.
    recency_wins without valid newer_index downgrades to flag_human.
    Invalid responses cause fallback to keyword-based for that chunk.

    v2.0.1 prompt hardening (from live-run false positives):
    - Complementary pairs (policy vs capability, scope vs method) are NOT
      contradictions — only flag when both entries assert the same predicate
      about the same subject and cannot both be true.
    - Temporal markers ("now", "no longer", "was upgraded", "as of <date>")
      signal a state_change → recency_wins, not a stable conflict.
    - Meta-memories (reports ABOUT past conflicts/dedup runs) are noise:
      return them as type "meta_noise" so the caller can invalidate them.

    Args:
        entries: list of memory entry strings

    Returns:
        List of contradictions: [{"pair": [i, j], "type": str, "resolution": str, "reason": str}, ...]
        Falls back to keyword-based drift detection if LLM unavailable or invalid.
    """
    if len(entries) < 2:
        return []

    contradictions = []
    for offset, chunk in _chunk(entries):
        n_chunk = len(chunk)
        numbered = "\n".join(f"[{offset + k}] {e}" for k, e in enumerate(chunk))
        prompt = (
            f"You are a contradiction detector for AI agent memory.\n"
            f"Find pairs of entries that genuinely contradict each other: both assert the SAME\n"
            f"predicate about the same subject, and both cannot be true at once.\n\n"
            f"NOT contradictions (do not report these):\n"
            f"  - Complementary facts: one entry states a POLICY or preference (\"preferred method\n"
            f"    is X\"), the other states a CAPABILITY or constraint (\"the agent cannot auto-edit\n"
            f"    config files\"). Different predicates — both can be true.\n"
            f"  - Different scope or aspect of the same system (e.g. one about the reload endpoint,\n"
            f"    another about the file watcher).\n"
            f"  - Entries that merely share a topic or entity name.\n"
            f"  - Duplicates or near-identical entries (same fact, same claim): duplicates are NOT\n"
            f"    contradictions — leave them to the dedup pass. Never report them here.\n"
            f"  - A report ABOUT a past conflict/dedup run (\"a conflict was flagged regarding X\")\n"
            f"    vs any other entry — that is meta-noise, not a fact about the world.\n\n"
            f"Two contradiction types:\n"
            f"  - 'state_change': the newer entry supersedes the old (provider switched, URL changed,\n"
            f"    version upgraded, feature that 'was never read' now works). Temporal markers\n"
            f"    (\"now\", \"no longer\", \"was upgraded\", \"as of <date>\", \"completed on <date>\")\n"
            f"    usually indicate this. Resolution: 'recency_wins' (invalidate the older entry).\n"
            f"  - 'stable_conflict': both entries claim different stable attributes for the same\n"
            f"    entity, with no temporal ordering. Resolution: 'flag_human'.\n"
            f"  - 'meta_noise': one entry is itself a report about a past conflict/dedup run\n"
            f"    (\"a conflict was flagged...\", \"N duplicates were invalidated...\"). Resolution:\n"
            f"    'invalidate_meta' (the report is stale bookkeeping, not a fact).\n\n"
            f"Entries:\n{numbered}\n\n"
            f"Respond with ONLY a JSON array: "
            f"[{{\"pair\": [i, j], \"type\": \"state_change|stable_conflict|meta_noise\", "
            f"\"resolution\": \"recency_wins|flag_human|invalidate_meta\", \"reason\": \"brief explanation\", "
            f"\"newer_index\": i_or_j}}, ...]\n"
            f"Rules: report each unordered pair at most once; prefer the single clearest type per pair;\n"
            f"if no contradictions exist, respond with: []\n"
            f"Do not include any prose or markdown."
        )

        resp = _llm_chat([{"role": "user", "content": prompt}])
        parsed = _parse_json_response(resp)

        if parsed is not None and isinstance(parsed, list):
            # v2.2.1: strict validation of all contradiction entries
            validated = _validate_contradictions(parsed, n_chunk)
            if validated is not None:
                for c in validated:
                    contradictions.append({
                        "pair": [offset + c["pair"][0], offset + c["pair"][1]],
                        "type": c["type"],
                        "resolution": c["resolution"],
                        "reason": c["reason"],
                        "newer_index": offset + c["newer_index"] if c["newer_index"] is not None else None,
                    })
            else:
                # Invalid response — fallback for this chunk
                contradictions.extend(_fallback_contradictions(chunk))
        else:
            # Fallback for this chunk: keyword-based
            contradictions.extend(_fallback_contradictions(chunk))

    return contradictions


def is_llm_available():
    """Check if the LLM endpoint is configured and reachable."""
    if not _load_api_key():
        return False
    # Quick health check: send a minimal prompt
    resp = _llm_chat([{"role": "user", "content": "Respond with: OK"}], max_tokens=5)
    return resp is not None and len(resp) > 0


if __name__ == "__main__":
    # Self-test: classify, dedup, and detect contradictions on sample entries
    test_entries = [
        "IrisBot: Linux 6.8, EPYC 4vCPU/16GB/193GB.",
        "Hindsight: localhost:8888, bank main. gpt-oss-120b via Cerebras.",
        "MCP tool_call JSON fix: Exa/Tavily invalid JSON error -> call tool_describe first, retry.",
        "David prefers concise responses, 3-5 line summaries, markdown format.",
        "David's output language is English only regardless of input language.",
        "User prefers Python for systems programming.",
        "User's primary coding language is Python.",
        "GLM 5.2 provider was switched to OpenRouter on Aug 2026.",
        "Previous model provider was z-ai direct, switched to OpenRouter.",
    ]

    print("=== LLM Judge Self-Test ===\n")
    print(f"LLM available: {is_llm_available()}\n")

    essential, offloadable = classify_importance(test_entries)
    print("Importance classification:")
    print(f"  Essential: {essential}")
    print(f"  Offloadable: {offloadable}\n")

    dups = semantic_dedup(test_entries)
    print("Semantic dedup groups:")
    for g in dups:
        print(f"  canonical=[{g['canonical']}] dups={g['duplicates']}")
    if not dups:
        print("  (none)")
    print()

    contradictions = detect_contradictions(test_entries)
    print("Contradictions:")
    for c in contradictions:
        print(f"  pair={c['pair']} type={c['type']} resolution={c['resolution']} reason={c.get('reason','')}")
    if not contradictions:
        print("  (none)")
