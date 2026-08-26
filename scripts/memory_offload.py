#!/usr/bin/env python3
"""
Hindsight Memory Offload Script
Runs via cron (no_agent=True). When local MEMORY.md exceeds 75% capacity,
offloads non-essential entries to Hindsight and removes them from local memory.

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

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

# LLM judge module (LLM-optional — degrades to rule-based if unavailable)
sys.path.insert(0, str(Path(__file__).parent))
try:
    import llm_judge  # noqa: E402
except ImportError:
    llm_judge = None  # standalone run: use rule-based fallbacks

# === Config ===
MEMORY_FILE = os.path.expanduser("/root/.hermes/memories/MEMORY.md")
HINDSIGHT_URL = "http://localhost:8888"
BANK = "main"
CAPACITY_MAX = 2200  # chars
OFFLOAD_THRESHOLD = 0.75  # 75%

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

def read_memory_file():
    """Read MEMORY.md and return list of entries (strings)."""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        content = f.read()
    # Entries are separated by lines containing only '§'
    if '§' in content:
        raw_entries = content.split('§')
    else:
        raw_entries = [content]
    entries = []
    for entry in raw_entries:
        entry = entry.strip()
        if entry and not entry.startswith('#') and not entry.startswith('---'):
            entries.append(entry)
    return entries

def get_memory_usage():
    """Read raw char count from MEMORY.md."""
    if not os.path.exists(MEMORY_FILE):
        return 0, CAPACITY_MAX
    with open(MEMORY_FILE, "r") as f:
        content = f.read()
    if '§' in content:
        entries = [e.strip() for e in content.split('§') if e.strip()]
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

def hindsight_health_check():
    """Check if Hindsight is healthy before offloading."""
    try:
        req = urllib.request.Request(f"{HINDSIGHT_URL}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "healthy"
    except Exception:
        return False

def hindsight_recall_check(content):
    """Semantic recall to check if content is already in Hindsight (dedup).

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
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            results = data.get("results", [])
            if not results:
                return False
            # Use LLM judge for semantic dedup if available
            if llm_judge is not None:
                result_texts = [r.get("content", "") or r.get("text", "") for r in results]
                # Batch: check all recalled results against the entry in one LLM call
                dup_groups = llm_judge.semantic_dedup([content] + result_texts[:5])
                for g in dup_groups:
                    if g["canonical"] == 0 and len(g["duplicates"]) > 0:
                        return True  # entry is duplicate of a recalled result
                return False
            # Fallback: word overlap
            for r in results:
                result_text = r.get("content", "") or r.get("text", "")
                entry_words = set(content.lower().split())
                result_words = set(result_text.lower().split())
                if entry_words and result_words:
                    overlap = len(entry_words & result_words) / len(entry_words)
                    if overlap > 0.6:
                        return True
            return False
    except Exception:
        return False  # If recall fails, proceed with retain anyway

def hindsight_retain(content, tags):
    """Store an entry in Hindsight."""
    payload = json.dumps({"items": [{"content": content}]}).encode()
    req = urllib.request.Request(
        f"{HINDSIGHT_URL}/v1/default/banks/{BANK}/memories",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        success = data.get("success", False)
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return success and tokens > 0

def rewrite_memory_file(entries_to_keep):
    """Rewrite MEMORY.md with only the kept entries."""
    with open(MEMORY_FILE, "w") as f:
        for entry in entries_to_keep:
            f.write(entry.strip() + "\n§\n")

def classify_entries(entries):
    """Classify entries as essential or offloadable.

    v2.0: Uses LLM judge (batch importance scoring) when available.
    Falls back to rule-based prefix matching if LLM unavailable.
    """
    if llm_judge is not None:
        essential_idx, offloadable_idx = llm_judge.classify_importance(entries)
        essential = [entries[i] for i in essential_idx if i < len(entries)]
        offloadable = [entries[i] for i in offloadable_idx if i < len(entries)]
        return essential, offloadable
    # Fallback: rule-based prefix matching
    essential = [e for e in entries if any(e.startswith(p) for p in ESSENTIAL_PREFIXES)]
    offloadable = [e for e in entries if not any(e.startswith(p) for p in ESSENTIAL_PREFIXES)]
    return essential, offloadable

def main():
    # 1. Check Hindsight health
    if not hindsight_health_check():
        print("WARN: Hindsight not healthy — skipping offload cycle.")
        sys.exit(0)  # Silent exit, no alert needed

    # 2. Read local memory
    entries = read_memory_file()
    if not entries:
        sys.exit(0)  # Nothing to do

    # 3. Check capacity
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
    offloaded = 0
    failed = 0
    for entry in offloadable:
        tags = get_tags(entry)
        try:
            # Dedup check — skip if already in Hindsight (LLM semantic or word-overlap)
            if hindsight_recall_check(entry):
                offloaded += 1  # Count as offloaded (already there)
                continue
            success = hindsight_retain(entry, tags)
            if success:
                offloaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1

    # 6. Rewrite local memory with only essential entries
    if offloaded > 0:
        rewrite_memory_file(essential)

    # 7. Report (only if something happened)
    new_used = sum(len(e) for e in essential)
    new_pct = new_used / capacity
    llm_status = "LLM-judged" if llm_judge is not None else "rule-based"
    if offloaded > 0 or failed > 0:
        print(f"Memory offload ({llm_status}): {offloaded} entries moved to Hindsight, {failed} failed. "
              f"Local: {new_used}/{capacity} ({new_pct:.0%}). {len(essential)} essential entries kept.")


if __name__ == "__main__":
    main()
