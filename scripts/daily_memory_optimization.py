#!/usr/bin/env python3
"""Daily memory optimization — L1/L2/L3 maintenance for Hermes Agent.

no-agent cron script: stdout is delivered verbatim; empty stdout = silent.

v2.0 (Aug 2026): LLM-driven semantic dedup + contradiction detection.
  - L2 semantic dedup pass via llm_judge.semantic_dedup() (replaces manual recall+overlap)
  - L2 contradiction scan via llm_judge.detect_contradictions() (replaces manual scan)
  - LLM-optional: degrades to rule-based if LLM unavailable
  - Batch consolidation: all memories in one LLM call (LycheeMemory V2 pattern)

Behavior (per memory-optimization skill v2.0.0):
  L2 (Hindsight, any 0.8+; Knowledge Pages checks activate on 0.9+):
    1. POST /consolidate -> poll operation to terminal state (max 8 min)
    2. Recall smoke-test after consolidation (over-prune check)
    3. Retain smoke-test: success:true AND total_tokens>0
       (health green != writes working — fact extraction is what silently fails)
    4. Bank stats: total_nodes>0, failed_operations trend vs last run
    5. **NEW** LLM semantic dedup pass: recall recent memories, LLM identifies
       near-duplicates, invalidate via PATCH (non-destructive: state=invalidated)
    6. **NEW** LLM contradiction scan: recall recent memories, LLM finds entity-drift
       pairs, apply recency-wins invalidation (state changes) or flag for human review
    7. Knowledge Pages tree: count pages + is_stale pages (0.9+ only)
  L1 (local memory):
    8. MEMORY.md / USER.md capacity check; >=90% triggers Hindsight offload
  L3 (LLM wiki / OKF bundle):
    9. Wiki lint-lite: stale-page count (>90 days) on ~/.hermes/kb

Output only when something needs attention; else silent. Exit 0 always
(a crash is reported via stdout, never via nonzero exit).
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path

# LLM judge module (LLM-optional — degrades to rule-based if unavailable)
sys.path.insert(0, str(Path(__file__).parent))
try:
    import llm_judge  # noqa: E402
except ImportError:
    llm_judge = None  # standalone run: skip LLM-driven steps

# Reuse the proven offload routine from the 30-min offload cron
# (deployed copy lives next to this file in ~/.hermes/scripts/).
try:
    import memory_offload  # noqa: E402
except ImportError:
    memory_offload = None  # standalone run: offload step will be skipped

BASE = "http://localhost:8888"
BANK = "main"
POLL_INTERVAL = 10
POLL_DEADLINE = 480          # 8 min max wait for consolidation
MEM_CAP = 2200
USER_CAP = 1375
MEM_WARN = 0.90              # warn at 90% capacity
OFFLOAD_AT = 0.90            # trigger Hindsight offload at >=90%
KB_DIR = Path("/root/.hermes/kb")
KB_STALE_DAYS = 90           # L3 page staleness threshold (skill: >90 days = stale)
FAILED_OPS_WARN = 10         # failed_operations count worth reporting
KP_STALE_RATIO_WARN = 0.5    # warn when >50% of knowledge pages are stale
KP_EXACT_CHECK_MAX = 25      # use exact per-page mental-model is_stale for KBs up to this size
STATE_FILE = Path("/root/.hermes/scripts/.daily_memory_opt_state.json")

# LLM-driven dedup/contradiction settings
DEDUP_RECALL_LIMIT = 50      # max memories to recall for LLM dedup/contradiction scan
DEDUP_RECALL_TOKENS = 3000   # token budget for recall query


def http(method, path, timeout=120, body=None):
    req = urllib.request.Request(
        BASE + path,
        method=method,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")
    # urllib.error.HTTPError propagates to caller's try/except


def walk_tree(nodes):
    """Yield every node in the Knowledge Pages tree recursively."""
    for n in nodes or []:
        yield n
        yield from walk_tree(n.get("children"))


def recall_recent_memories(query="user preferences, environment, configuration, tools"):
    """Recall recent memories for LLM-driven dedup + contradiction scan.

    Returns list of {id, content} dicts, or empty list on failure.
    """
    try:
        _, resp = http("POST", f"/v1/default/banks/{BANK}/memories/recall", timeout=120,
                       body={"query": query, "max_tokens": DEDUP_RECALL_TOKENS})
        hits = resp.get("results") or resp.get("memories") or resp.get("items") or []
        memories = []
        for h in hits:
            mid = h.get("id") or h.get("memory_id")
            content = h.get("content", "") or h.get("text", "")
            if mid and content:
                memories.append({"id": mid, "content": content})
        return memories[:DEDUP_RECALL_LIMIT]
    except Exception:
        return []


def invalidate_memory(mid, reason="duplicate"):
    """Non-destructive invalidation: PATCH state=invalidated (recall-hidden, retained on disk).

    Never DELETE — invalidation preserves audit trail and is recoverable.
    """
    try:
        http("PATCH", f"/v1/default/banks/{BANK}/memories/{mid}", timeout=120,
             body={"state": "invalidated", "reason": reason})
        return True
    except Exception:
        return False


def llm_semantic_dedup_pass(problems):
    """LLM-driven semantic dedup pass (Step 5, new in v2.0).

    Recalls recent memories, uses LLM to identify semantic near-duplicates
    (same fact, different wording), and invalidates the duplicates via PATCH
    (non-destructive: state=invalidated, retained on disk for audit).

    Research basis:
    - MenteDB llm_consolidation: LLM-as-judge for semantic dedup
    - Hindsight blog: fact deduplication — "same claim, different wording"
    - Human-Inspired Memory: dedup-based consolidation achieves 97.2% precision, 58% store reduction
    """
    if llm_judge is None:
        return  # LLM unavailable — skip (rule-based dedup happens in Hindsight's own consolidation)

    memories = recall_recent_memories()
    if len(memories) < 2:
        return  # nothing to dedup

    contents = [m["content"] for m in memories]
    dup_groups = llm_judge.semantic_dedup(contents)

    invalidated = 0
    for group in dup_groups:
        canonical_idx = group["canonical"]
        for dup_idx in group["duplicates"]:
            if dup_idx < len(memories):
                mid = memories[dup_idx]["id"]
                if invalidate_memory(mid, reason="semantic_duplicate"):
                    invalidated += 1

    if invalidated > 0:
        problems.append(
            f"LLM semantic dedup: {invalidated} near-duplicate memories invalidated "
            f"(non-destructive — retained on disk for audit)"
        )


def llm_contradiction_scan(problems):
    """LLM-driven contradiction detection (Step 6, new in v2.0).

    Recalls recent memories, uses LLM to find entity-drift pairs (old vs new
    state of the same fact), and applies recency-wins invalidation for state
    changes. Stable-attribute conflicts are flagged for human review.

    Research basis:
    - Hindsight blog: conflict handling — recency wins for state, source/confidence for stable
    - Hindsight blog: entity drift (Postgres→MySQL) is the canonical failure
    """
    if llm_judge is None:
        return  # LLM unavailable — skip

    memories = recall_recent_memories(
        query="provider, model, url, port, version, configuration changes, migrations"
    )
    if len(memories) < 2:
        return  # nothing to scan

    contents = [m["content"] for m in memories]
    contradictions = llm_judge.detect_contradictions(contents)

    invalidated = 0
    flagged = 0
    for c in contradictions:
        pair = c["pair"]
        resolution = c.get("resolution", "flag_human")
        reason = c.get("reason", "")
        newer_idx = c.get("newer_index")

        if resolution == "recency_wins" and newer_idx is not None:
            # Invalidate the OLDER entry (the one that's NOT newer_idx)
            older_idx = pair[0] if pair[1] == newer_idx else pair[1]
            if older_idx < len(memories):
                mid = memories[older_idx]["id"]
                if invalidate_memory(mid, reason=f"superseded: {reason}"):
                    invalidated += 1
        else:
            # Stable conflict — flag for human review (don't auto-resolve)
            flagged += 1
            pair_contents = [contents[pair[0]][:60], contents[pair[1]][:60]]
            problems.append(
                f"LLM contradiction scan: stable-attribute conflict needs review — "
                f"[{pair_contents[0]}] vs [{pair_contents[1]}] ({reason})"
            )

    if invalidated > 0:
        problems.append(
            f"LLM contradiction scan: {invalidated} stale state-change memories invalidated "
            f"(recency-wins, non-destructive)"
        )


def main():
    problems = []

    # --- 1. Trigger consolidation -------------------------------------
    op_id = None
    try:
        status, resp = http("POST", f"/v1/default/banks/{BANK}/consolidate", timeout=120, body={})
        op_id = resp.get("operation_id")
        if not op_id:
            problems.append(f"Consolidate returned HTTP {status} but no operation_id: {str(resp)[:120]}")
    except Exception as e:
        problems.append(f"Consolidate trigger failed: {type(e).__name__}: {e}")

    # --- 2. Poll operation until terminal state ------------------------
    final = None
    if op_id:
        deadline = time.time() + POLL_DEADLINE
        while time.time() < deadline:
            try:
                _, op = http("GET", f"/v1/default/banks/{BANK}/operations/{op_id}", timeout=120)
                st = op.get("status", "")
                if st in ("completed", "failed", "error", "cancelled"):
                    final = op
                    break
            except Exception as e:
                problems.append(f"Operation poll failed: {type(e).__name__}: {e}")
                break
            time.sleep(POLL_INTERVAL)
        if final is None:
            problems.append(f"Consolidation op {op_id[:8]} did not finish within {POLL_DEADLINE}s")
        elif final.get("status") != "completed":
            problems.append(f"Consolidation op {op_id[:8]} ended with status={final.get('status')}")

    # --- 3. Recall smoke-test (consolidation over-prune check) ----------
    # Health green != writes/recall working; probe a known-durable fact.
    if final is not None and final.get("status") == "completed":
        try:
            _, rec = http("POST", f"/v1/default/banks/{BANK}/memories/recall", timeout=120,
                          body={"query": "David's preferred output language and document summary style"})
            hits = rec.get("results") or rec.get("memories") or rec.get("items") or []
            if not hits:
                problems.append("Recall smoke-test returned 0 results after consolidation — possible over-prune; verify manually")
        except Exception as e:
            problems.append(f"Recall smoke-test failed: {type(e).__name__}: {e}")

    # --- 3b. Retain smoke-test (silent write-failure check) -------------
    # success:true with total_tokens == 0 means fact extraction did NOT run
    # (the exact step that fails on broken auth / bad LLM while health is green).
    try:
        _, ret = http("POST", f"/v1/default/banks/{BANK}/memories", timeout=120,
                      body={"items": [{"content": f"daily memory optimization smoke test {time.strftime('%Y-%m-%d')}"}]})
        usage = (ret.get("usage") or {})
        if not ret.get("success"):
            problems.append(f"Retain smoke-test failed: success != true ({str(ret)[:120]})")
        elif not usage.get("total_tokens"):
            problems.append("Retain smoke-test: success but total_tokens=0 — fact extraction not running; check LLM provider/auth")
    except Exception as e:
        problems.append(f"Retain smoke-test failed: {type(e).__name__}: {e}")

    # --- 4. Bank stats + failed_operations trend -----------------------
    try:
        _, stats = http("GET", f"/v1/default/banks/{BANK}/stats", timeout=120)
        nodes = stats.get("total_nodes", 0)
        docs = stats.get("total_documents", 0)
        if nodes <= 0:
            problems.append(f"Bank '{BANK}' stats suspicious: total_nodes={nodes}, docs={docs}")
        # failed_operations climbing = silent write failures despite green health
        failed = stats.get("failed_operations", 0)
        prev_failed = 0
        if STATE_FILE.exists():
            try:
                prev_failed = int(json.loads(STATE_FILE.read_text()).get("failed_operations", 0))
            except Exception:
                pass
        if failed >= FAILED_OPS_WARN or failed > prev_failed:
            problems.append(f"failed_operations={failed} (was {prev_failed}) — Hindsight writes failing silently; check backlog")
        STATE_FILE.write_text(json.dumps({"failed_operations": failed}))
    except Exception as e:
        problems.append(f"Bank stats fetch failed: {type(e).__name__}: {e}")

    # --- 5. LLM semantic dedup pass (NEW in v2.0) ----------------------
    # Recall recent memories, LLM identifies near-duplicates, invalidate
    # (non-destructive). Research: 30-40% reduction on polluted stores.
    try:
        llm_semantic_dedup_pass(problems)
    except Exception as e:
        problems.append(f"LLM semantic dedup pass failed: {type(e).__name__}: {e}")

    # --- 6. LLM contradiction scan (NEW in v2.0) -----------------------
    # Recall recent memories, LLM finds entity-drift pairs, apply
    # recency-wins invalidation or flag for human review.
    try:
        llm_contradiction_scan(problems)
    except Exception as e:
        problems.append(f"LLM contradiction scan failed: {type(e).__name__}: {e}")

    # --- 7. Knowledge Pages health (Hindsight >= 0.9 only) -------------
    # A page is a projected view over memory — it inherits L2's failure
    # modes. The tree's is_stale comes from a single bank-wide
    # last_memory_write_at signal, so on a bank that receives daily writes
    # (incl. this script's own smoke-test retain) EVERY page looks stale —
    # a false positive. For small KBs (<= KP_EXACT_CHECK_MAX pages) query
    # each page's mental model for the exact per-page is_stale instead.
    # 404 = older version or feature off: skip.
    try:
        status, tree = http("GET", f"/v1/default/banks/{BANK}/knowledge-base/tree", timeout=120)
        pages = [n for n in walk_tree(tree.get("roots")) if n.get("kind") == "page"]
        if pages:
            if len(pages) <= KP_EXACT_CHECK_MAX:
                stale = []
                for n in pages:
                    mm = n.get("mental_model_id")
                    if not mm:
                        continue
                    try:
                        _, m = http("GET", f"/v1/default/banks/{BANK}/mental-models/{mm}", timeout=120)
                        if m.get("is_stale"):
                            stale.append(n)
                    except Exception:
                        pass  # per-page failure shouldn't kill the check
            else:
                stale = [n for n in pages if n.get("is_stale")]
            if stale and len(stale) / len(pages) > KP_STALE_RATIO_WARN:
                problems.append(
                    f"Knowledge Pages: {len(stale)}/{len(pages)} pages stale "
                    f"(e.g. {', '.join(n['name'] for n in stale[:3])}) — pages falling behind consolidation"
                )
    except urllib.error.HTTPError:
        pass  # <0.9 or KB disabled — not an error
    except Exception as e:
        problems.append(f"Knowledge Pages check failed: {type(e).__name__}: {e}")

    # --- 7b. L3 wiki lint-lite ------------------------------------------
    try:
        if KB_DIR.exists():
            md_files = [p for p in KB_DIR.rglob("*.md") if "_archive" not in p.parts]
            stale = []
            cutoff = time.time() - KB_STALE_DAYS * 86400
            for p in md_files:
                # skip index pages; flag pages whose mtime is old
                if p.stat().st_mtime < cutoff and "index.md" not in p.name.lower():
                    stale.append(p.name)
            if len(stale) >= 5:
                problems.append(
                    f"L3 wiki: {len(stale)} of {len(md_files)} pages older than {KB_STALE_DAYS} days "
                    f"(e.g. {', '.join(sorted(stale)[:3])}) — run llm-wiki lint / refresh"
                )
        elif not Path("/root/wiki").exists():
            pass  # no L3 bundle installed — not an error
    except Exception as e:
        problems.append(f"L3 wiki check failed: {type(e).__name__}: {e}")

    # --- 8. Local memory capacity check --------------------------------
    try:
        mem = Path("/root/.hermes/memories/MEMORY.md")
        usr = Path("/root/.hermes/memories/USER.md")
        if mem.exists():
            n = mem.stat().st_size
            pct = n / MEM_CAP
            if pct >= OFFLOAD_AT:
                # Memory nearly full — run Hindsight offload to reclaim space.
                if memory_offload is None:
                    problems.append(
                        f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                        "memory_offload.py not found next to this script; offload skipped"
                    )
                else:
                    try:
                        used_before, _ = memory_offload.get_memory_usage()
                        memory_offload.main()
                        used_after, _ = memory_offload.get_memory_usage()
                        if used_after < used_before:
                            problems.append(
                                f"MEMORY.md was at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                                f"offloaded to Hindsight, now {used_after}/{MEM_CAP} chars"
                            )
                        else:
                            problems.append(
                                f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                                f"offload ran but no entries moved (all essential?). Prune manually."
                            )
                    except Exception as e:
                        problems.append(
                            f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — "
                            f"offload failed: {type(e).__name__}: {e}"
                        )
            elif pct >= MEM_WARN:
                problems.append(f"MEMORY.md at {n}/{MEM_CAP} chars ({int(pct*100)}%) — approaching capacity")
        if usr.exists():
            n = usr.stat().st_size
            pct = n / USER_CAP
            if pct >= MEM_WARN:
                problems.append(f"USER.md at {n}/{USER_CAP} chars ({int(pct*100)}%) — prune or offload to Hindsight")
    except Exception as e:
        problems.append(f"Local memory check failed: {type(e).__name__}: {e}")

    # --- 9. Output ------------------------------------------------------
    if problems:
        print(f"**🧠 Daily Memory Optimization — issues found**\n")
        for p in problems:
            print(f"  • {p}")
    # else: empty stdout -> cron stays silent


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Script crash = real error, always report
        print(f"**🧠 Daily Memory Optimization — script crashed**\n\n  • {type(e).__name__}: {e}")
    sys.exit(0)
