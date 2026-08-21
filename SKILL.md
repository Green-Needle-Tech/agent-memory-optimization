---
name: memory-optimization
description: "Optimize L1/L2/L3 memory: prune, offload, dedup, lint."
version: 1.1.0
author: Iris
license: MIT
trigger: >-
  User asks to optimize, clean, prune, or maintain memory (L1/L2/L3), offload local memory to
  Hindsight, dedup Hindsight memories, lint the LLM Wiki, or fix memory capacity warnings.
  Also when MEMORY.md or USER.md exceeds 75% capacity, or when asked to run the periodic
  memory maintenance routine.
metadata:
  hermes:
    tags: [memory, hindsight, llm-wiki, maintenance, l1, l2, l3]
    related_skills: [hindsight-memory, hermes-local-memory, llm-wiki]
    category: productivity
---

# Three-Layer Memory Optimization (L1 / L2 / L3)

## When to Use
- David asks to optimize/clean/prune/maintain memory (L1, L2, L3), offload local memory to Hindsight, dedup Hindsight, or lint the wiki.
- MEMORY.md or USER.md exceeds 75% capacity, or a periodic deep-maintenance run is requested (beyond the automated cron jobs).
- After Hindsight incidents (backlog replay, consolidation over-prune) that need recall-audit + cleanup.

## Output Format
Markdown for explanations, code blocks for commands. Concise, action-oriented.

## Context Boundary
Each maintenance run is independent. Always re-check current state (capacities, stats, wiki size) before acting — do not assume.

## Layer Topology
- **L1 — Local memory** (`memory` tool): MEMORY.md (2,200 chars) + USER.md (1,375 chars). Always injected every turn.
- **L2 — Hindsight** (localhost:8888, bank `main`): semantic recall on demand. Episodes, preferences, decisions.
- **L3 — LLM Wiki / OKF bundle**: compiled knowledge. SA-Copilot bundle at `~/.hermes/kb` (git-versioned); older static wiki at `/root/wiki/`.
- **Division of labor:** L1 = "must know this turn" · L2 = "recall when relevant" · L3 = "compiled knowledge with sources". A fact existing in two layers lives in the deepest layer that surfaces it reliably — usually L2.

## Research Grounding (web research, Aug 2026)
Key findings from agent-memory literature that drive this procedure:

1. **Consolidation > retrieval.** Production memory failures are policy failures (what gets kept/merged/evicted), not retrieval failures. "An agent that remembers everything is an agent that remembers nothing useful" (Hindsight blog, May 2026). Four levers: **importance, merge, decay, eviction**.
2. **Write time is the cheapest quality gate.** Everything that enters the index must be retrieved, reranked, and judged forever. Fact extraction doubles as the importance filter — conversational filler should never become a memory. Don't retain volatile state.
3. **Less memory = better answers.** Measured: add-all agents accumulated 2,400 records and dropped to 13% accuracy; curated agents held 248 records at 39% — a 3× improvement from storing less (TianPan GC article, Apr 2026).
4. **Scale thresholds.** Agent memory becomes unreliable above ~10K interactions without active consolidation. Failure is silent: stale facts are retrieved with full confidence and no error signal (memory pollution). Entity drift (Postgres→MySQL migration) is the canonical failure: the newer fact must supersede via merge, not coexist.
5. **Semantic dedup beats hash dedup.** "User prefers Python" vs "User's primary language is Python" evade exact-match dedup but consume 3 retrieval slots for 1 fact. Background semantic dedup passes typically shrink stores 30–40% with no information loss.
6. **Tiered TTL by category** (generational GC pattern): immutable facts → infinite TTL; procedural knowledge → weeks/months; preferences → days/weeks; transient state → hours/days. Useful memories refresh their own TTL on successful retrieval (spaced-repetition effect).
7. **Contradiction handling**: for state changes, recency wins (supersede, don't delete); for stable attributes, prefer source/confidence. Always resolve at write time, not query time.
8. **Eviction is for compliance only** (GDPR, PII, user request). For performance problems, fix importance/merge/decay upstream instead — Hindsight's design premise: good consolidation makes eviction unnecessary.

## Procedure

### Step 0 — Assess current state
```
memory(action="add", target="memory", content="probe")  # read usage % from error/usage (or read system prompt caps)
curl -s http://localhost:8888/v1/default/banks/main/stats | jq '{nodes, failed_operations, pending_operations}'
du -sh ~/.hermes/kb /root/wiki 2>/dev/null
```
Report: L1 % for both stores, L2 node count + failed ops, L3 size.
Red flags: node count climbing fast with flat recall quality (index pollution); failed_operations rising (silent write failures).

### Step 1 — L1 prune + offload
L1 is the always-injected tier: every char costs attention on every turn. This is write-time importance filtering in its purest form.
1. Classify entries: essential (host specs, active provider, tool quirks) vs offloadable (stale-in-7-days facts, cron IDs, version numbers, one-time lessons).
2. Dedup-check each offloadable entry against Hindsight (`hindsight_recall`, skip if >60% word overlap).
3. Retain to L2 with context + tags: `hindsight_retain(content=..., context="memory offload", tags=[...])`.
4. Batch-remove from local via `memory` tool `operations` array — see hermes-local-memory skill for the all-or-nothing pitfall (copy exact `current_entries` text; watch em-dashes).
5. Densify remaining entries: merge overlapping ones into single compact entries (single atomic batch).
- **Do NOT retain volatile state** (container states, ports, cron job IDs, current model) — it becomes stale recall bait. Only durable preferences, decisions, and conventions.
- USER.md at 90%+ needs a manual prune batch — adds fail when near-full, so plan remove+add in ONE batch.

### Step 2 — L2 Hindsight maintenance
1. Check `GET /stats`: `failed_operations` must not be climbing (health endpoint alone is NOT proof — writes fail silently while green).
2. **Semantic dedup pass** (mark-and-sweep): recall-based scan for near-duplicate facts (same claim, different wording — hash/exact dedup misses these). Invalidate duplicate `world`/`experience` memories via `PATCH /memories/{id}` `{"state":"invalidated","reason":"duplicate"}`. Never PATCH `observation` type (derived — 400; invalidate the source instead). Expect 30–40% reduction on a polluted store.
3. **Contradiction scan**: look for entity-drift pairs (old vs new state of the same fact, e.g. old provider/model/URL). State changes → invalidate the OLD fact (recency wins). Stable attributes that conflict → flag for David rather than auto-resolving.
4. After invalidations: `POST /consolidate`, then **verify recall** — consolidation can over-prune linked facts; re-add lost critical facts in a single grouped retain.
5. Config tune (PATCH `/config` with `{"updates":{...}}` wrapper): `recall_budget_function: adaptive`, `recall_max_tokens: 3000`, `consolidation_max_memories_per_round: 50`, bank-specific `retain_mission`/`reflect_mission`.
6. Set `recall_types: "observation,world,experience"` in the profile's `hindsight/config.json` (NOT via API PATCH — invalid field).
- LLM must be non-reasoning (current: gpt-oss-120b via Cerebras) — see hindsight-memory skill §5.
- Do NOT mass-evict for performance — fix importance/dedup upstream. Eviction (bank purge) only on explicit request/compliance.

### Step 3 — L3 wiki lint
L3 is the "neocortex" tier: compiled, synthesized, source-grounded. Its GC problem is different — staleness and drift, not volume.
Run llm-wiki skill lint procedure:
- Orphan pages (no inbound wikilinks), broken `[[links]]`, index.md completeness, frontmatter validation (required fields incl. `type`), pages >200 lines (split), `updated` >90 days stale, `confidence: low` / `contested: true` pages, tag-taxonomy violations, log.md >500 entries (rotate).
- sha256 drift check on `raw/` sources: unchanged → skip re-ingest; changed → flag drift.
- **Contradiction handling**: when new info conflicts with existing page content, check dates — newer sources supersede; if genuinely contradictory, note BOTH with dates/sources and mark `contradictions: [page]` in frontmatter. Never silently overwrite.
- Archive superseded pages to `_archive/`, update inbound links to "(archived)", update index + log.
- OKF bundle conventions: required frontmatter field `type`; recommended `status`/`stale_after`/`generated`/`verified`/`sources`; diagrams as Mermaid in-doc; binaries in `kb/assets/`. Use `stale_after` as the tiered-TTL mechanism: immutable reference → no stale_after; fast-moving tech → short stale_after.
- Ask before mass-updating 10+ existing pages.

### Step 4 — Report
One summary per layer: what was offloaded/invalidated/linted, before/after L1 %, L2 node delta (dedup savings), L3 issue counts. Include one "memory health" verdict: is recall precision stable, are there active contradictions, any pollution signals. No raw dumps.

## Automation context (already in place — do not duplicate)
- Auto-offload cron (every 30min, `memory_offload.py`): handles L1 >75% automatically.
- Daily memory optimization cron at 8:00 AM SGT (no-agent Python).
- Hindsight health watchdog every 15min.
Manual runs of this skill are for deep maintenance: USER.md prunes, L2 dedup/contradiction passes, L3 lints.

## Pitfalls
- memory batch ops are atomic — one bad `old_text` rejects everything; copy exact text from `current_entries`.
- Health green ≠ writes working; always check failed_operations + smoke-test retain (`total_tokens > 0`).
- `§` separators are display-only; never include in `old_text`.
- Consolidation over-pruning: always recall-verify after `POST /consolidate`.
- Don't create L3 pages for passing mentions (2+ source rule) — the #1 wiki bloat source.
- Near-duplicates evade exact-match dedup — always dedup semantically (recall + overlap check), not by string compare.
- Stale facts are confidently wrong, not obviously wrong — after backlog replays or model switches, actively recall-test known-current facts (e.g. David's trip.com-only rule) and invalidate resurrected stale guidance.

## Sources
- Hindsight blog: The Consolidation Problem in Agent Memory (May 2026) — four-lever framework, write-time filtering, eviction-for-compliance-only.
- TianPan: Agent Memory Garbage Collection (Apr 2026) — generational TTL tiers, semantic dedup as mark-and-sweep, contradiction detection, 3× accuracy result.
- RankSquire: Agent Memory vs RAG at Scale (Mar 2026) — 10K-interaction threshold, memory pollution, hybrid architecture.
- Zylos Research: Selective Retention and Forgetting (Jun 2026) — CLS theory mapping, importance scoring (recency+importance+relevance), framework survey (Letta/Zep/Mem0/Dreaming V3).
