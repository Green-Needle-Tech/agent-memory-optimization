---
name: memory-optimization
description: "Optimize L1/L2/L3 memory: prune, offload, dedup, lint."
version: 2.0.1
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

## L2 ↔ L3 Bridge: Hindsight and the Wiki (research, Aug 2026)

The two systems are complementary, not competing. Hindsight is a **memory engine** (ingest raw turns → auto-extract typed facts + entity graph → consolidate into deduped observations → TEMPR retrieval: semantic + BM25 + graph + temporal, RRF-fused, cross-encoder reranked). The wiki is a **compiled artifact** (knowledge distilled once, kept current, never re-derived). Four integration patterns, in escalating investment:

1. **Pointer retention (cheapest — default).** Retain wiki page references into Hindsight with tags and paths: `retain("Wiki page 'X' at <path> covers A, B, C", tags=["wiki-ref"])`. Recall then surfaces the pointer; the agent opens the page. Hindsight becomes the search layer over the wiki without duplicating content. Do NOT retain page bodies — that breaks L2/L3 separation and doubles consolidation cost.
2. **Wiki as curated raw layer.** Feed wiki pages into Hindsight via the documents API. Hindsight adds what a static wiki can't: temporal reasoning, multi-hop entity traversal, and automatic contradiction reconciliation (a hand-maintained wiki keeps contradictions "a paragraph apart"; consolidation resolves them). Costly — use only for high-value curated corpora.
3. **Dual-store role separation (this deployment's shape).** Wiki = compiled, human-reviewable domain knowledge (index-first lookup, deterministic, cheap). Hindsight = operational memory (fuzzy, temporal, multi-hop). Reference the wiki when the question maps to a known page; fall back to recall for temporal/personal/entity questions.
4. **Knowledge Pages (native, Hindsight ≥ v0.9).** Hindsight projects its own wiki: Knowledge Pages are self-rewriting markdown documents built from consolidated observations only, organized in a folder tree, mounted to disk via `hindsight fs mount --bank <bank>` (real markdown + YAML frontmatter, background refresh loop; export bundle = index.md + one file per page + log.md — structurally parallel to the Karpathy layout). Key property: a page is a **projected view over memory, not storage** — it heals itself because it re-projects; delete it and nothing is lost. Defaults that matter for maintenance: `fact_types: ["observation"]`, `mode: "delta"` (edits, never regenerates), `exclude_mental_models: true` (pages never cite each other — no feedback loops). Tree staleness (`is_stale`) comes from a single bank-wide `last_memory_write_at` signal — cheap but approximate; the page's mental model gives the exact answer.

**Maintenance implication:** in Pattern 4 the wiki layer needs no lint for contradictions/staleness — consolidation does it. But it inherits L2's failure modes: if retains silently fail, pages go stale with no error. Knowledge-page health rides on the same `failed_operations` + smoke-test checks as Step 2. In Patterns 1–3, wiki lint (Step 3) remains necessary and Hindsight pointers must be invalidated when their target page is archived/moved.

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

## LLM-as-Judge (v2.0, Aug 2026)

The v2.0 upgrade replaces brittle rule-based heuristics with **LLM-as-judge** for three operations, grounded in 2026 agent-memory research:

1. **Importance classification** (`llm_judge.classify_importance`) — replaces hardcoded prefix matching with LLM-rated long-term retention value (Park et al. Generative Agents importance scoring + Hindsight fact-extraction-as-filter). All entries classified in a single batched LLM call (LycheeMemory V2 segment-level batching pattern — 75-87% fewer LLM calls vs per-entry).

2. **Semantic dedup** (`llm_judge.semantic_dedup`) — replaces word-overlap >60% with LLM judge that identifies near-duplicates expressing the same fact with different wording ("User prefers Python" vs "User's primary language is Python"). Research: semantic dedup shrinks polluted stores 30-40% with no information loss (TianPan GC article); dedup-based consolidation achieves 97.2% retention precision with 58% store reduction (Human-Inspired Memory, arXiv:2605.08538).

3. **Contradiction detection** (`llm_judge.detect_contradictions`) — replaces manual scan with LLM-driven entity-drift detection. Resolution policy: recency-wins for state changes (invalidate old, keep new); flag-for-human for stable attributes (Hindsight blog, May 2026).

**LLM-optional design** (MenteDB `llm_consolidation` pattern): all three operations degrade to rule-based fallbacks if the LLM endpoint is unavailable. Callers with no LLM simply get the old heuristic behavior — the engine stays functional. The `llm_judge` module reads `OPENROUTER_API_KEY` from `~/.hermes/.env` and uses `google/gemini-2.5-flash` by default (~$0.0001/call).

**Non-destructive invalidation**: dedup and contradiction resolution use `PATCH /memories/{id} {"state":"invalidated"}` (recall-hidden, retained on disk for audit). Never `DELETE` — old state is recoverable.

### New research sources (v2.0)
- MenteDB `llm_consolidation` (docs.rs/mentedb) — LLM-as-judge for semantic dedup, LLM-optional design, non-destructive invalidation via `Derived` edges.
- LycheeMemory V2 (arXiv:2608.12990) — semantic segment-level consolidation, 86% fewer construction tokens vs A-Mem.
- RecMem (ACL 2026 Findings) — recurrence-based consolidation, 87% token reduction, LLM invoked only on sustained recurrence.
- Human-Inspired Memory Architecture (arXiv:2605.08538) — dedup-based consolidation: 97.2% precision, 58% store reduction, +21.8pp over baseline.
- SCM: Sleep-Consolidated Memory (arXiv:2604.20943) — structured forgetting for LLM memory, privacy-aware pruning.
- NirDiamant/Agent_Memory_Techniques — 30 runnable notebooks: consolidation, compaction, self-reflection, forgetting & decay.

## Procedure

### Step 0 — Assess current state
```
memory(action="add", target="memory", content="probe")  # read usage % from error/usage (or read system prompt caps)
curl -s http://localhost:8888/v1/default/banks/main/stats | jq '{nodes, failed_operations, pending_operations}'
du -sh ~/.hermes/kb /root/wiki 2>/dev/null
```
Report: L1 % for both stores, L2 node count + failed ops, L3 size.
Red flags: node count climbing fast with flat recall quality (index pollution); failed_operations rising (silent write failures).

### Step 1 — L1 prune + offload (LLM-driven, v2.0)
L1 is the always-injected tier: every char costs attention on every turn. This is write-time importance filtering in its purest form.
1. Classify entries using `llm_judge.classify_importance(entries)` — LLM rates each entry's long-term retention value in a single batched call. Falls back to prefix matching if LLM unavailable.
2. Dedup-check each offloadable entry against Hindsight (`hindsight_recall` + `llm_judge.semantic_dedup` for semantic match, skip if LLM says duplicate).
3. Retain to L2 with context + tags: `hindsight_retain(content=..., context="memory offload", tags=[...])`.
4. Batch-remove from local via `memory` tool `operations` array — see hermes-local-memory skill for the all-or-nothing pitfall (copy exact `current_entries` text; watch em-dashes).
5. Densify remaining entries: merge overlapping ones into single compact entries (single atomic batch).
- **Do NOT retain volatile state** (container states, ports, cron job IDs, current model) — it becomes stale recall bait. Only durable preferences, decisions, and conventions.
- USER.md at 90%+ needs a manual prune batch — adds fail when near-full, so plan remove+add in ONE batch.

### Step 2 — L2 Hindsight maintenance (LLM-driven, v2.0)
1. Check `GET /stats`: `failed_operations` must not be climbing (health endpoint alone is NOT proof — writes fail silently while green).
2. **LLM semantic dedup pass** (`llm_judge.semantic_dedup`): recall recent memories (max 50), LLM identifies near-duplicate groups (same fact, different wording). Invalidate duplicates via `PATCH /memories/{id} {"state":"invalidated","reason":"semantic_duplicate"}` (non-destructive — recall-hidden, retained on disk). Never PATCH `observation` type (derived — 400; invalidate the source instead). Expect 30-40% reduction on a polluted store.
3. **LLM contradiction scan** (`llm_judge.detect_contradictions`): recall recent memories, LLM finds entity-drift pairs. State changes → invalidate the OLDER fact (recency wins). Stable attributes that conflict → flag for David rather than auto-resolving.
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
- Daily memory optimization cron at 8:00 AM SGT (no-agent Python, `scripts/daily_memory_optimization.py` in this repo — deploy a copy to `~/.hermes/scripts/` next to `memory_offload.py`).
- Hindsight health watchdog every 15min.

### Daily cron behavior (v2.0.0, targets Hindsight 0.8+; LLM-driven dedup/contradiction; KP checks on 0.9+)
1. `POST /consolidate` → poll operation to terminal state (max 8 min)
2. Recall smoke-test after consolidation (over-prune check)
3. **Retain smoke-test**: `success:true` AND `total_tokens > 0` — the exact step that silently fails while health stays green
4. Bank stats: `total_nodes > 0`, `failed_operations` trend vs last run (state file)
5. **LLM semantic dedup pass** (NEW v2.0): recall recent memories (max 50), `llm_judge.semantic_dedup()` identifies near-duplicates, invalidate via PATCH (non-destructive). LLM-optional — skipped if LLM unavailable.
6. **LLM contradiction scan** (NEW v2.0): recall recent memories, `llm_judge.detect_contradictions()` finds entity-drift pairs. Recency-wins invalidation for state changes; flag-for-human for stable conflicts. LLM-optional.
7. **Knowledge Pages health** (`GET /knowledge-base/tree`, 0.9+): warn when >50% of pages are stale — pages are projected views that inherit L2's failure modes. ⚠️ The tree's `is_stale` is a single bank-wide `last_memory_write_at` signal — on a bank that receives daily writes (including this script's own smoke-test retain) every page reads stale (false positive). For KBs ≤25 pages the script queries each page's mental model (`GET /mental-models/{id}`) for the exact per-page `is_stale`. 404 on older versions = skip silently
8. L1 capacity: MEMORY.md/USER.md ≥90% → warn; ≥90% MEMORY.md triggers offload (LLM-driven classification)
9. L3 wiki lint-lite: ≥5 pages older than 90 days → warn

Silent on success (empty stdout); exit 0 always — errors surface via stdout, never via nonzero exit.

## Field research addendum (Aug 2026)
The 2026 Q1 memory-landscape shift: three of four new systems (Supermemory ASMR, Mastra Observational Memory, Hindsight) abandon plain vector-DB retrieval in favor of **compress-and-reason**. Two consequences for this procedure:
- **Compression is a first-class memory strategy**, not an emergency measure (Observational Memory: 5–40× compression, 94.87% LongMemEval with a single small model). L1 densification and L2 consolidation are the same lever — treat them as the primary optimization, not housekeeping.
- **LLM-as-retriever** (Hindsight's Cara reflect agent, ASMR's search agents) raises retrieval quality but makes every retrieval depend on LLM availability — which is why the daily retain smoke-test (`total_tokens > 0`) matters: LLM-layer failure is the dominant silent failure mode in this architecture.

Manual runs of this skill are for deep maintenance: USER.md prunes, L2 dedup/contradiction passes, L3 lints.

## Pitfalls
- memory batch ops are atomic — one bad `old_text` rejects everything; copy exact text from `current_entries`.
- Health green ≠ writes working; always check failed_operations + smoke-test retain (`total_tokens > 0`).
- Knowledge Pages tree endpoint is `/knowledge-base/tree` (not `/knowledge-base`, which 404s); page bodies need a separate `GET /knowledge-base/pages/{id}` fetch.
- `§` separators are display-only; never include in `old_text`.
- Consolidation over-pruning: always recall-verify after `POST /consolidate`.
- Don't create L3 pages for passing mentions (2+ source rule) — the #1 wiki bloat source.
- Near-duplicates evade exact-match dedup — always dedup semantically (recall + overlap check), not by string compare.
- Stale facts are confidently wrong, not obviously wrong — after backlog replays or model switches, actively recall-test known-current facts (e.g. David's trip.com-only rule) and invalidate resurrected stale guidance.

### LLM-judge pitfalls (v2.0.1, from live daily runs)
- **Batch size overflow → silent fallback.** >30 entries per LLM call overflows the response token budget; the JSON truncates, parsing fails, and the judge silently degrades to the noisy rule-based fallback (observed: an 85-entry batch flagged 30+ pairs with empty reasons). Always chunk (`BATCH_SIZE = 30` in `llm_judge.py`); per-chunk fallback is fine.
- **Self-pollution loop.** The daily script's own outputs (smoke-test probes, "a conflict was flagged…" reports) become memories that the next run's scans flag against everything. Fixes: tag smoke-test probes and retire them after 48h; invalidate meta-memories (reports about past runs) on sight; never retain run reports as memories.
- **Derived facts can't be PATCHed.** `observation`-type memories return 400 on invalidation — invalidate their `world`/`experience` sources instead (`GET /memories/{id}` → `source_memory_ids`). Fact extraction also rewrites retained content ("daily memory optimization smoke test 2026-08-29" becomes "David performed a daily memory optimization smoke test"), so content-prefix matching must be fuzzy.
- **Complementary ≠ contradictory.** "Preferred method is X" (policy) vs "agent cannot auto-edit config" (capability) are different predicates — both true. The judge prompt now states this explicitly; temporal markers ("now", "was upgraded", "completed on <date>") route pairs to recency-wins instead of flag-human.
- **Judge mislabels duplicates as conflicts.** When its own reason says "identical/duplicate", trust the diagnosis over the type label — resolve as dedup. Code guards: exact-normalized match, prefix/subset match, and reason-keyword match all resolve as dedup instead of flagging.
- **Shared recall set.** Dedup and contradiction passes must scan the same recalled memories (one broad query) — different queries meant duplicates the contradiction scan saw never reached the dedup pass.
- **Flag fatigue.** Stable-conflict flags are fingerprinted (order-independent content hash) and persisted; an unresolved flag reports once, not every day.

## Sources
- Hindsight blog: The Consolidation Problem in Agent Memory (May 2026) — four-lever framework, write-time filtering, eviction-for-compliance-only.
- TianPan: Agent Memory Garbage Collection (Apr 2026) — generational TTL tiers, semantic dedup as mark-and-sweep, contradiction detection, 3× accuracy result.
- RankSquire: Agent Memory vs RAG at Scale (Mar 2026) — 10K-interaction threshold, memory pollution, hybrid architecture.
- Zylos Research: Selective Retention and Forgetting (Jun 2026) — CLS theory mapping, importance scoring (recency+importance+relevance), framework survey (Letta/Zep/Mem0/Dreaming V3).
- Hindsight docs — Knowledge Pages (hindsight.vectorize.io/developer/knowledge-pages) and Knowledge Pages API (…/developer/api/knowledge-pages), Aug 2026: pages as projected views over memory, `hindsight fs mount`, default trigger (observations-only, delta mode, exclude_mental_models), staleness signal.
- Latimer et al., "Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects," arXiv:2512.12818 (Dec 2025) — four logical networks, retain/recall/reflect, 91.4% LongMemEval.
- Karpathy, "LLM Wiki" gist (Apr 2026) — three layers, index-first scaling without embedding RAG, file-back answers.
- LLM Memory Research, "LLM Memory Systems: 2026 Q1 Update" (lin-guanguo.github.io, Mar 2026) — anti-RAG shift: Supermemory ASMR (LLM-as-retriever, ensemble voting), Mastra Observational Memory (pure compression, 94.87% LongMemEval), Hindsight (TEMPR 4-way retrieval + MPFP traversal + Cara reflect), MemOS.
- MenteDB `llm_consolidation` (docs.rs/mentedb) — LLM-as-judge for semantic dedup, LLM-optional design, non-destructive invalidation via `Derived` edges.
- LycheeMemory V2 (arXiv:2608.12990) — semantic segment-level consolidation, 86% fewer construction tokens vs A-Mem.
- RecMem (ACL 2026 Findings, aclanthology.org/2026.findings-acl.1619) — recurrence-based consolidation, 87% token reduction, LLM invoked only on sustained recurrence.
- Human-Inspired Memory Architecture (arXiv:2605.08538) — dedup-based consolidation: 97.2% precision, 58% store reduction, +21.8pp over baseline.
- SCM: Sleep-Consolidated Memory (arXiv:2604.20943) — structured forgetting for LLM memory, privacy-aware pruning.
- NirDiamant/Agent_Memory_Techniques (github.com/NirDiamant/Agent_Memory_Techniques) — 30 runnable notebooks: consolidation, compaction, self-reflection, forgetting & decay.
- MEM1 (arXiv:2506.15841) — end-to-end RL for memory consolidation via context pruning, ReAct framework extension.
