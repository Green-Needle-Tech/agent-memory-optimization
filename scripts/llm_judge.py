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

Usage:
  from llm_judge import classify_importance, semantic_dedup, detect_contradictions

  essential, offloadable = classify_importance(memory_entries)
  dup_groups = semantic_dedup(memory_entries)
  contradictions = detect_contradictions(memory_entries)

Configuration:
  Reads OPENROUTER_API_KEY from ~/.hermes/.env (or environment).
  Uses google/gemini-2.5-flash by default (~$0.0001/call).
  Override via env: LLM_JUDGE_MODEL, LLM_JUDGE_BASE_URL, LLM_JUDGE_API_KEY.
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

# === Config ===
DEFAULT_MODEL = os.environ.get("LLM_JUDGE_MODEL", "google/gemini-2.5-flash")
DEFAULT_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.1  # low temp for classification consistency
REQUEST_TIMEOUT = 60


def _load_api_key():
    """Load OpenRouter API key from .env file or environment."""
    key = os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_file = Path("/root/.hermes/.env")
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

    Args:
        entries: list of memory entry strings
        context: description of the memory store for the LLM

    Returns:
        (essential_indices, offloadable_indices) — lists of integer indices
        into the input list. Falls back to rule-based prefix matching if LLM
        is unavailable.
    """
    if not entries:
        return [], []

    # Build a single batched prompt (LycheeMemory segment-level batching)
    numbered = "\n".join(f"[{i}] {e}" for i, e in enumerate(entries))
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

    if parsed and "essential" in parsed and "offloadable" in parsed:
        essential = [int(i) for i in parsed["essential"] if isinstance(i, (int, float)) or str(i).isdigit()]
        offloadable = [int(i) for i in parsed["offloadable"] if isinstance(i, (int, float)) or str(i).isdigit()]
        # Validate: all indices accounted for
        all_indices = set(range(len(entries)))
        classified = set(essential) | set(offloadable)
        unclassified = all_indices - classified
        # Any unclassified go to offloadable (safe default)
        offloadable.extend(unclassified)
        return essential, offloadable

    # Fallback: rule-based
    return _fallback_classify(entries)


def semantic_dedup(entries):
    """Find semantic near-duplicates using LLM judge.

    Replaces word-overlap dedup. The LLM identifies entries that express the
    same fact with different wording (e.g., "User prefers Python" vs "User's
    primary language is Python") — these evade exact-match dedup but waste
    retrieval slots.

    Args:
        entries: list of memory entry strings

    Returns:
        List of duplicate groups: [{"canonical": idx, "duplicates": [idx, ...]}, ...]
        Falls back to word-overlap dedup (>60%) if LLM unavailable.
    """
    if len(entries) < 2:
        return []

    numbered = "\n".join(f"[{i}] {e}" for i, e in enumerate(entries))
    prompt = (
        f"You are a semantic deduplication judge for AI agent memory.\n"
        f"Identify groups of entries that express the SAME underlying fact or preference "
        f"with different wording. Only group genuine semantic duplicates — not entries that "
        f"merely share a topic.\n\n"
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
        groups = []
        for g in parsed:
            if isinstance(g, dict) and "canonical" in g and "duplicates" in g:
                canonical = int(g["canonical"])
                dups = [int(d) for d in g["duplicates"]]
                if dups:  # only include groups with at least one duplicate
                    groups.append({"canonical": canonical, "duplicates": dups})
        return groups

    # Fallback: word-overlap
    return _fallback_dedup(entries)


def detect_contradictions(entries):
    """Detect contradictory memory entries using LLM.

    Identifies entity-drift pairs (old vs new state of the same fact) and
    stable-attribute conflicts. Resolution policy (Hindsight blog, May 2026):
    - State changes → recency wins (invalidate old, keep new)
    - Stable attributes that conflict → flag for human review

    Args:
        entries: list of memory entry strings

    Returns:
        List of contradictions: [{"pair": [i, j], "type": str, "resolution": str, "reason": str}, ...]
        Falls back to keyword-based drift detection if LLM unavailable.
    """
    if len(entries) < 2:
        return []

    numbered = "\n".join(f"[{i}] {e}" for i, e in enumerate(entries))
    prompt = (
        f"You are a contradiction detector for AI agent memory.\n"
        f"Find pairs of entries that contradict each other — same entity/fact but different values.\n"
        f"Two types:\n"
        f"  - 'state_change': the newer entry supersedes the old (e.g., provider switched, URL changed). "
        f"    Resolution: 'recency_wins' (invalidate the older entry).\n"
        f"  - 'stable_conflict': both entries claim different stable attributes for the same entity. "
        f"    Resolution: 'flag_human' (needs human review).\n\n"
        f"Entries:\n{numbered}\n\n"
        f"Respond with ONLY a JSON array: "
        f"[{{\"pair\": [i, j], \"type\": \"state_change|stable_conflict\", "
        f"\"resolution\": \"recency_wins|flag_human\", \"reason\": \"brief explanation\", "
        f"\"newer_index\": i_or_j}}, ...]\n"
        f"If no contradictions exist, respond with: []\n"
        f"Do not include any prose or markdown."
    )

    resp = _llm_chat([{"role": "user", "content": prompt}])
    parsed = _parse_json_response(resp)

    if parsed is not None and isinstance(parsed, list):
        contradictions = []
        for c in parsed:
            if isinstance(c, dict) and "pair" in c:
                pair = [int(c["pair"][0]), int(c["pair"][1])]
                contradictions.append({
                    "pair": pair,
                    "type": c.get("type", "unknown"),
                    "resolution": c.get("resolution", "flag_human"),
                    "reason": c.get("reason", ""),
                    "newer_index": c.get("newer_index"),
                })
        return contradictions

    # Fallback: keyword-based
    return _fallback_contradictions(entries)


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
