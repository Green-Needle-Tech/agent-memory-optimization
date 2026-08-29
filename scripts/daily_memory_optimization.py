#!/usr/bin/env python3
"""Daily memory optimization — L1/L2/L3 maintenance for Hermes Agent.

no-agent cron script: stdout is delivered verbatim; empty stdout = silent.

v2.0 (Aug 2026): LLM-driven semantic dedup + contradiction detection.
  - L2 semantic dedup pass via llm_judge.semantic_dedup() (replaces manual recall+overlap)
  - L2 contradiction scan via llm_judge.detect_contradictions() (replaces manual scan)
  - LLM-optional: degrades to rule-based if LLM unavailable
  - Batch consolidation: all memories in one LLM call (LycheeMemory V2 pattern)

v2.0.1 (Aug 2026): self-pollution fixes from live-run false positives.
  - Smoke-test probe memories are tagged and retired after 48h (30 had accumulated)
  - Meta-memories (reports ABOUT past maintenance runs) invalidated on sight —
    they previously flagged against every related fact, every day
  - Exact-duplicate pre-pass before the LLM (verbatim copies need no judge)
  - One shared recall set feeds both dedup and contradiction passes
    (previously different queries meant duplicates never reached dedup)
  - Contradiction prompt hardened: complementary pairs (policy vs capability)
    are not conflicts; temporal markers route state changes to recency-wins
  - Stable-conflict flags deduped across runs via fingerprint state file
    (an unresolved flag reports once, not daily)

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
import json, re, sys, time, hashlib, urllib.request, urllib.error
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

# v2.0.1: self-pollution guards
SMOKE_TEST_TAG = "daily-memopt-smoke"   # tag on the retain smoke-test memory
SMOKE_TEST_MAX_AGE_S = 172800           # smoke-test memories older than 48h = junk
META_MEMORY_PATTERNS = [                # reports ABOUT past maintenance runs = noise
    r"stable-attribute conflict was flagged",
    r"conflict was flagged",
    r"duplicates? (?:were|was) invalidated",
    r"memor(?:y|ies) (?:were|was) invalidated",
    r"invalidated \(non-destructive",
    r"recency-wins",
    r"needs? review",
    r"flagged for (?:human )?review",
]
META_MEMORY_RE = re.compile("|".join(META_MEMORY_PATTERNS), re.IGNORECASE)
FLAG_STATE_FILE = Path("/root/.hermes/scripts/.daily_memory_opt_flags.json")


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


def recall_recent_memories(query="user preferences, environment, configuration, tools", limit=None):
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
        return memories[:(limit or DEDUP_RECALL_LIMIT)]
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


def recall_all_recent(limit=DEDUP_RECALL_LIMIT):
    """Recall a broad mix of recent memories (single shared set for dedup + contradictions).

    v2.0.1: dedup and contradiction scans previously used different recall queries,
    so duplicates the contradiction scan saw never reached the dedup pass. One
    broad recall now feeds both passes, keeping indices consistent.
    """
    return recall_recent_memories(
        query="user preferences, environment, configuration, tools, versions, migrations, decisions",
        limit=limit,
    )


def cleanup_smoke_tests(problems):
    """Invalidate old smoke-test memories (v2.0.1 self-pollution fix).

    The retain smoke-test writes a memory every run; without cleanup they
    accumulate forever (30 found in one bank). Each new smoke-test memory is
    tagged SMOKE_TEST_TAG; anything tagged and older than SMOKE_TEST_MAX_AGE_S
    is invalidated. Legacy untagged ones are matched by content prefix.
    """
    cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - SMOKE_TEST_MAX_AGE_S))
    removed = 0
    try:
        _, resp = http("POST", f"/v1/default/banks/{BANK}/memories/recall", timeout=120,
                       body={"query": "daily memory optimization smoke test", "max_tokens": 2000})
        hits = resp.get("results") or resp.get("memories") or resp.get("items") or []
        for h in hits:
            content = (h.get("content") or h.get("text") or "")
            tags = [str(t).lower() for t in (h.get("tags") or [])]
            is_smoke = SMOKE_TEST_TAG in tags or content.startswith("daily memory optimization smoke test")
            if not is_smoke:
                continue
            # Date embedded in content; invalidate only if older than the cutoff
            m = re.search(r"smoke test (\d{4}-\d{2}-\d{2})", content)
            if m and m.group(1) > cutoff:
                continue  # recent — keep (it's this run's probe)
            if invalidate_memory(h.get("id") or h.get("memory_id"), reason="smoke_test_junk"):
                removed += 1
    except Exception:
        return  # cleanup is best-effort; never block the run
    if removed:
        problems.append(f"Self-pollution cleanup: {removed} old smoke-test memories invalidated")


def invalidate_meta_memories(memories, problems):
    """Invalidate meta-memories: reports ABOUT past maintenance runs (v2.0.1).

    Past runs' problem reports were retained as memories ("a stable-attribute
    conflict was flagged regarding X"). Those reports then flag against every
    related fact on every subsequent run — self-referential noise. They are
    bookkeeping, not facts; invalidate on sight.
    """
    removed = 0
    for m in memories:
        if META_MEMORY_RE.search(m.get("content", "")):
            if invalidate_memory(m["id"], reason="meta_noise: report about a past maintenance run"):
                removed += 1
    if removed:
        problems.append(f"Self-pollution cleanup: {removed} meta-memories (reports about past runs) invalidated")
    return removed


def exact_duplicate_prepass(memories):
    """Invalidate verbatim duplicates before the LLM passes (v2.0.1).

    Exact copies (same normalized content) need no LLM call — collapse them
    deterministically, keeping the first occurrence. Returns count removed.
    """
    seen = {}
    removed = 0
    for m in memories:
        key = hashlib.sha256(
            re.sub(r"\s+", " ", m["content"]).strip().lower().encode()
        ).hexdigest()
        if key in seen:
            if invalidate_memory(m["id"], reason="exact_duplicate"):
                removed += 1
        else:
            seen[key] = m["id"]
    return removed


def load_flag_state():
    """Load previously-reported stable-conflict fingerprints (cross-run dedup)."""
    try:
        return set(json.loads(FLAG_STATE_FILE.read_text()))
    except Exception:
        return set()


def save_flag_state(flags):
    """Persist stable-conflict fingerprints so unresolved flags report once, not daily."""
    try:
        FLAG_STATE_FILE.write_text(json.dumps(sorted(flags)))
    except Exception:
        pass


def flag_fingerprint(pair_contents):
    """Stable fingerprint for a flagged pair (order-independent, content-based)."""
    a, b = sorted(pair_contents)
    return hashlib.sha256(f"{a}||{b}".encode()).hexdigest()


def llm_semantic_dedup_pass(problems, memories=None):
    """LLM-driven semantic dedup pass (Step 5, new in v2.0).

    Recalls recent memories, uses LLM to identify semantic near-duplicates
    (same fact, different wording), and invalidates the duplicates via PATCH
    (non-destructive: state=invalidated, retained on disk for audit).

    v2.0.1: accepts a pre-recalled shared memory list (same set feeds the
    contradiction scan, keeping indices consistent).

    Research basis:
    - MenteDB llm_consolidation: LLM-as-judge for semantic dedup
    - Hindsight blog: fact deduplication — "same claim, different wording"
    - Human-Inspired Memory: dedup-based consolidation achieves 97.2% precision, 58% store reduction
    """
    if llm_judge is None:
        return  # LLM unavailable — skip (rule-based dedup happens in Hindsight's own consolidation)

    if memories is None:
        memories = recall_all_recent()
    if len(memories) < 2:
        return  # nothing to dedup

    contents = [m["content"] for m in memories]
    dup_groups = llm_judge.semantic_dedup(contents)

    invalidated = 0
    for group in dup_groups:
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


def llm_contradiction_scan(problems, memories=None):
    """LLM-driven contradiction detection (Step 6, new in v2.0).

    Recalls recent memories, uses LLM to find entity-drift pairs (old vs new
    state of the same fact), and applies recency-wins invalidation for state
    changes. Stable-attribute conflicts are flagged for human review.

    v2.0.1: handles the 'meta_noise' type (invalidate the report memory),
    dedups flagged pairs across runs via a fingerprint state file (an
    unresolved flag reports once, not every day), and shares the recall set
    with the dedup pass.

    Research basis:
    - Hindsight blog: conflict handling — recency wins for state, source/confidence for stable
    - Hindsight blog: entity drift (Postgres→MySQL) is the canonical failure
    """
    if llm_judge is None:
        return  # LLM unavailable — skip

    if memories is None:
        memories = recall_all_recent()
    if len(memories) < 2:
        return  # nothing to scan

    contents = [m["content"] for m in memories]
    contradictions = llm_judge.detect_contradictions(contents)

    invalidated = 0
    flagged = 0
    seen_pairs = set()
    prev_flags = load_flag_state()
    new_flags = set()

    for c in contradictions:
        pair = c["pair"]
        # Normalize unordered pair; skip if already handled this run
        key = tuple(sorted(pair))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

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
        elif resolution == "invalidate_meta":
            # The entry itself is a report about a past run — invalidate it
            for idx in pair:
                if idx < len(memories) and META_MEMORY_RE.search(memories[idx]["content"]):
                    if invalidate_memory(memories[idx]["id"], reason="meta_noise: report about a past maintenance run"):
                        invalidated += 1
        else:
            # Stable conflict — flag for human review (don't auto-resolve).
            # Guard 1: identical/near-identical contents are duplicates, not
            # conflicts — the judge occasionally mislabels them (observed live).
            pair_contents = [contents[pair[0]], contents[pair[1]]]
            norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
            a, b = norm(pair_contents[0]), norm(pair_contents[1])
            # Guard 2: one content a prefix/subset of the other = same fact with
            # extra metadata — resolve as dedup (invalidate the shorter), not a flag.
            if a == b or a.startswith(b) or b.startswith(a):
                shorter_idx = pair[0] if len(a) <= len(b) else pair[1]
                if invalidate_memory(memories[shorter_idx]["id"], reason="semantic_duplicate (near-verbatim pair)"):
                    invalidated += 1
                continue
            # Guard 3: judge's own reason says duplicate — trust the diagnosis
            # over the type label and resolve as dedup.
            if re.search(r"duplicat|identical", reason, re.IGNORECASE):
                shorter_idx = pair[0] if len(pair_contents[0]) <= len(pair_contents[1]) else pair[1]
                if invalidate_memory(memories[shorter_idx]["id"], reason=f"semantic_duplicate (judge reason: {reason[:80]})"):
                    invalidated += 1
                continue
            # Cross-run dedup: only report pairs not already flagged before.
            fp = flag_fingerprint(pair_contents)
            if fp in prev_flags:
                continue  # already reported in a previous run — stay silent
            new_flags.add(fp)
            flagged += 1
            problems.append(
                f"LLM contradiction scan: stable-attribute conflict needs review — "
                f"[{pair_contents[0][:60]}] vs [{pair_contents[1][:60]}] ({reason})"
            )

    if invalidated > 0:
        problems.append(
            f"LLM contradiction scan: {invalidated} stale state-change memories invalidated "
            f"(recency-wins, non-destructive)"
        )
    if new_flags:
        save_flag_state(prev_flags | new_flags)


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
    # v2.0.1: the probe memory is tagged so cleanup_smoke_tests can retire it
    # once stale — untagged probes accumulated 30+ in the bank.
    try:
        _, ret = http("POST", f"/v1/default/banks/{BANK}/memories", timeout=120,
                      body={"items": [{
                          "content": f"daily memory optimization smoke test {time.strftime('%Y-%m-%d')}",
                          "tags": [SMOKE_TEST_TAG],
                      }]})
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

    # --- 5. LLM dedup + contradiction passes (v2.0, restructured v2.0.1) -
    # One shared recall feeds both passes (consistent indices), after:
    #   a. smoke-test cleanup (old probes invalidated)
    #   b. meta-memory invalidation (reports about past runs = noise)
    #   c. exact-duplicate pre-pass (no LLM needed for verbatim copies)
    try:
        cleanup_smoke_tests(problems)
        memories = recall_all_recent()
        if memories:
            invalidate_meta_memories(memories, problems)
            exact_dupes = exact_duplicate_prepass(memories)
            if exact_dupes:
                problems.append(f"Exact-duplicate pre-pass: {exact_dupes} verbatim copies invalidated")
            # Drop memories invalidated by the pre-passes before the LLM sees them
            memories = [m for m in memories if not META_MEMORY_RE.search(m["content"])]
        llm_semantic_dedup_pass(problems, memories=memories)
        llm_contradiction_scan(problems, memories=memories)
    except Exception as e:
        problems.append(f"LLM dedup/contradiction passes failed: {type(e).__name__}: {e}")

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
