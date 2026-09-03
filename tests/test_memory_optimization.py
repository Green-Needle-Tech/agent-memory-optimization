"""Tests for agent-memory-optimization scripts (v3.0 — rule-based heuristics).

Covers:
  - Memory offload helpers (tags, stable document ID, classification)
  - Offload data-loss prevention (v2.2.1 P0): partial success, all-fail,
    all-success, dedup, retain exception — transactional safety
  - Atomic writes and backup
  - Retain tags and document_id
  - Daily memory optimization: flag fingerprint, walk_tree, HTML escaping
  - Safe memory invalidation fact_type checking (v2.2.1)
  - Destructive flag: rule-based resolver (v3.0)
  - Memory records: structured records, candidate generation, recency (v2.3)
  - Privacy / PII redaction (v2.4)
  - Audit log (v2.4)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# Ensure scripts directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import daily_memory_optimization
import memory_offload
import memory_records

# ============================================================================
# Memory Offload: helper functions
# ============================================================================

class TestMemoryOffloadHelpers:
    def test_get_tags_known_prefix(self):
        assert memory_offload.get_tags("IrisBot: Linux 6.8 specs") == ["environment", "infra"]
        assert memory_offload.get_tags("Hindsight: localhost:8888") == ["hindsight", "infra"]

    def test_get_tags_unknown(self):
        assert memory_offload.get_tags("Unknown entry") == ["offloaded", "memory-management"]

    def test_classify_entries_fallback(self):
        entries = ["IrisBot: Linux 6.8 specs", "Random fact to offload"]
        essential, offloadable = memory_offload.classify_entries(entries)
        assert len(essential) == 1
        assert len(offloadable) == 1
        assert "IrisBot" in essential[0]
        assert "Random fact" in offloadable[0]

    def test_stable_document_id_deterministic(self):
        """Same content should always produce the same document_id."""
        id1 = memory_offload._stable_document_id("User prefers Python")
        id2 = memory_offload._stable_document_id("User prefers Python")
        assert id1 == id2

    def test_stable_document_id_different_content(self):
        id1 = memory_offload._stable_document_id("User prefers Python")
        id2 = memory_offload._stable_document_id("User prefers Java")
        assert id1 != id2

    def test_stable_document_id_normalizes_whitespace(self):
        """Whitespace differences should not change the document_id."""
        id1 = memory_offload._stable_document_id("User   prefers   Python")
        id2 = memory_offload._stable_document_id("User prefers Python")
        assert id1 == id2

    def test_stable_document_id_has_namespace(self):
        """document_id should have the l1-offload namespace prefix."""
        doc_id = memory_offload._stable_document_id("test content")
        assert doc_id.startswith("l1-offload:")


# ============================================================================
# Memory Offload: data-loss prevention (v2.2.1 P0, v3.0 transactional)
# ============================================================================

class TestOffloadDataLossPrevention:
    """Tests that failed offloads NEVER result in local memory loss."""

    def test_partial_success_keeps_failed_entries(self):
        """Mixed success: 1 succeeds, 4 fail — all 4 failed must be kept locally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "MEMORY.md"
            entries = []
            for i in range(5):
                entries.append(f"IrisBot: essential entry {i} with enough text to fill capacity here")
            for i in range(5):
                entries.append(f"Offloadable entry {i} with some text content here for testing")
            content = "".join(f"{e}\n§\n" for e in entries)
            mem_file.write_text(content)

            with patch.object(memory_offload, "MEMORY_FILE", mem_file), \
                 patch.object(memory_offload, "LOCK_FILE", mem_file.with_suffix(".lock")), \
                 patch.object(memory_offload, "BACKUP_DIR", Path(tmpdir) / ".backups"), \
                 patch.object(memory_offload, "CAPACITY_MAX", 100), \
                 patch.object(memory_offload, "hindsight_health_check", return_value=True), \
                 patch.object(memory_offload, "hindsight_recall_check", return_value=False), \
                 patch.object(memory_offload, "hindsight_retain") as mock_retain:

                # Entry 0 succeeds, entries 1-4 fail
                results = [True, False, False, False, False]
                mock_retain.side_effect = results

                with patch.object(memory_offload, "sys") as mock_sys:
                    mock_sys.exit = lambda code=0: None
                    memory_offload._do_offload()

                result = mem_file.read_text()

                # Essential entries must be present
                for i in range(5):
                    assert f"essential entry {i}" in result

                # Failed entries MUST be present (not lost!)
                for i in range(1, 5):
                    assert f"Offloadable entry {i}" in result

                # Successfully offloaded entry should NOT be present
                assert "Offloadable entry 0" not in result

    def test_all_fail_keeps_everything(self):
        """When all retains fail, no entries should be removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "MEMORY.md"
            entries = [
                "IrisBot: essential entry 0 with enough text to fill capacity",
                "Offloadable entry that will fail to offload",
            ]
            content = "".join(f"{e}\n§\n" for e in entries)
            mem_file.write_text(content)

            with patch.object(memory_offload, "MEMORY_FILE", mem_file), \
                 patch.object(memory_offload, "LOCK_FILE", mem_file.with_suffix(".lock")), \
                 patch.object(memory_offload, "BACKUP_DIR", Path(tmpdir) / ".backups"), \
                 patch.object(memory_offload, "CAPACITY_MAX", 50), \
                 patch.object(memory_offload, "hindsight_health_check", return_value=True), \
                 patch.object(memory_offload, "hindsight_recall_check", return_value=False), \
                 patch.object(memory_offload, "hindsight_retain", return_value=False):

                with patch.object(memory_offload, "sys") as mock_sys:
                    mock_sys.exit = lambda code=0: None
                    memory_offload._do_offload()

                result = mem_file.read_text()
                # Both entries must still be present
                assert "essential entry 0" in result
                assert "Offloadable entry" in result

    def test_all_success_removes_offloadable(self):
        """When all retains succeed, only essential entries remain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "MEMORY.md"
            entries = [
                "IrisBot: essential entry 0 with enough text",
                "Offloadable entry to be removed",
            ]
            content = "".join(f"{e}\n§\n" for e in entries)
            mem_file.write_text(content)

            with patch.object(memory_offload, "MEMORY_FILE", mem_file), \
                 patch.object(memory_offload, "LOCK_FILE", mem_file.with_suffix(".lock")), \
                 patch.object(memory_offload, "BACKUP_DIR", Path(tmpdir) / ".backups"), \
                 patch.object(memory_offload, "CAPACITY_MAX", 50), \
                 patch.object(memory_offload, "hindsight_health_check", return_value=True), \
                 patch.object(memory_offload, "hindsight_recall_check", return_value=False), \
                 patch.object(memory_offload, "hindsight_retain", return_value=True):

                with patch.object(memory_offload, "sys") as mock_sys:
                    mock_sys.exit = lambda code=0: None
                    memory_offload._do_offload()

                result = mem_file.read_text()
                assert "essential entry 0" in result
                assert "Offloadable entry" not in result

    def test_dedup_already_exists_counts_as_offloaded(self):
        """When recall check says entry is already in Hindsight, it's safe to remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "MEMORY.md"
            entries = [
                "IrisBot: essential entry 0 with enough text",
                "Offloadable entry already in Hindsight",
            ]
            content = "".join(f"{e}\n§\n" for e in entries)
            mem_file.write_text(content)

            with patch.object(memory_offload, "MEMORY_FILE", mem_file), \
                 patch.object(memory_offload, "LOCK_FILE", mem_file.with_suffix(".lock")), \
                 patch.object(memory_offload, "BACKUP_DIR", Path(tmpdir) / ".backups"), \
                 patch.object(memory_offload, "CAPACITY_MAX", 50), \
                 patch.object(memory_offload, "hindsight_health_check", return_value=True), \
                 patch.object(memory_offload, "hindsight_recall_check", return_value=True), \
                 patch.object(memory_offload, "hindsight_retain") as mock_retain:

                mock_retain.return_value = True  # shouldn't be called

                with patch.object(memory_offload, "sys") as mock_sys:
                    mock_sys.exit = lambda code=0: None
                    memory_offload._do_offload()

                result = mem_file.read_text()
                assert "essential entry 0" in result
                assert "Offloadable entry" not in result
                # retain should NOT have been called (dedup said already there)
                mock_retain.assert_not_called()

    def test_retain_exception_keeps_entry(self):
        """When retain raises an exception, the entry must be kept locally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "MEMORY.md"
            entries = [
                "IrisBot: essential entry 0 with enough text",
                "Offloadable entry that will throw",
            ]
            content = "".join(f"{e}\n§\n" for e in entries)
            mem_file.write_text(content)

            with patch.object(memory_offload, "MEMORY_FILE", mem_file), \
                 patch.object(memory_offload, "LOCK_FILE", mem_file.with_suffix(".lock")), \
                 patch.object(memory_offload, "BACKUP_DIR", Path(tmpdir) / ".backups"), \
                 patch.object(memory_offload, "CAPACITY_MAX", 50), \
                 patch.object(memory_offload, "hindsight_health_check", return_value=True), \
                 patch.object(memory_offload, "hindsight_recall_check", return_value=False), \
                 patch.object(memory_offload, "hindsight_retain", side_effect=Exception("network error")):

                with patch.object(memory_offload, "sys") as mock_sys:
                    mock_sys.exit = lambda code=0: None
                    memory_offload._do_offload()

                result = mem_file.read_text()
                assert "Offloadable entry" in result  # kept because retain failed


# ============================================================================
# Memory Offload: atomic writes and backup (v2.2.1)
# ============================================================================

class TestAtomicWrites:
    def test_atomic_write_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            memory_offload.atomic_write_text(path, "hello world")
            assert path.read_text() == "hello world"

    def test_atomic_write_overwrites(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.write_text("old content")
            memory_offload.atomic_write_text(path, "new content")
            assert path.read_text() == "new content"

    def test_atomic_write_no_temp_left(self):
        """No temp files should be left after successful write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            memory_offload.atomic_write_text(path, "content")
            temps = [f for f in Path(tmpdir).iterdir() if f.name.startswith(".test_")]
            assert len(temps) == 0

    def test_backup_created_before_rewrite(self):
        """rewrite_memory_file should create a backup before overwriting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "MEMORY.md"
            backup_dir = Path(tmpdir) / ".backups"
            mem_file.write_text("original content")

            with patch.object(memory_offload, "MEMORY_FILE", mem_file), \
                 patch.object(memory_offload, "BACKUP_DIR", backup_dir):
                memory_offload.rewrite_memory_file(["new entry"])

                backups = list(backup_dir.glob("MEMORY_*"))
                assert len(backups) == 1
                assert backups[0].read_text() == "original content"
                assert mem_file.read_text() == "new entry\n§\n"


# ============================================================================
# Memory Offload: retain now sends tags (v2.2.1)
# ============================================================================

class TestRetainTags:
    def test_retain_includes_tags_and_document_id(self):
        """hindsight_retain should include tags and document_id in the request body."""
        captured_body = {}

        class MockResp:
            def read(self):
                return json.dumps({"success": True, "usage": {"total_tokens": 100}}).encode()

        class MockUrp:
            def __enter__(self):
                return MockResp()

            def __exit__(self, *args):
                pass

        def mock_urlopen(req, timeout):
            captured_body["data"] = json.loads(req.data.decode())
            return MockUrp()

        with patch.object(memory_offload.urllib.request, "urlopen", mock_urlopen):
            result = memory_offload.hindsight_retain("test content", ["tag1", "tag2"])

        assert result is True
        body = captured_body["data"]
        assert "items" in body
        item = body["items"][0]
        assert item["content"] == "test content"
        assert item["tags"] == ["tag1", "tag2"]
        assert item["context"] == "L1 memory offload"
        assert "document_id" in item
        assert item["document_id"].startswith("l1-offload:")


# ============================================================================
# Daily Memory Optimization: helpers
# ============================================================================

class TestDailyMemoryOptimization:
    def test_flag_fingerprint(self):
        fp1 = daily_memory_optimization.flag_fingerprint(["fact A", "fact B"])
        fp2 = daily_memory_optimization.flag_fingerprint(["fact B", "fact A"])
        assert fp1 == fp2

    def test_walk_tree(self):
        tree = [
            {"name": "root1", "children": [{"name": "child1"}]},
            {"name": "root2"},
        ]
        nodes = list(daily_memory_optimization.walk_tree(tree))
        names = [n["name"] for n in nodes]
        assert names == ["root1", "child1", "root2"]

    def test_issue_dataclass(self):
        """Issue dataclass should have code, severity, message, context."""
        issue = daily_memory_optimization.Issue(
            code="L2_FAILED_OPERATIONS_INCREASED",
            severity="warning",
            message="Hindsight failed operations increased",
            context={"current": 15, "previous": 10},
        )
        assert issue.code == "L2_FAILED_OPERATIONS_INCREASED"
        assert issue.severity == "warning"
        rendered = daily_memory_optimization.render_issue(issue)
        assert "Hindsight failed operations increased" in rendered
        assert "current=15" in rendered

    def test_rule_remediations_allowlist(self):
        """RULE_REMEDIATIONS should contain only the spec-defined allowlist."""
        expected_keys = {
            "L2_CONSOLIDATION_PENDING",
            "L1_CAPACITY_EXCEEDED",
            "SMOKE_TEST_EXPIRED",
            "META_MEMORY_FOUND",
            "EXACT_DUPLICATE",
            "STRONG_DUPLICATE",
            "STATE_CHANGE_HIGH_CONFIDENCE",
        }
        assert set(daily_memory_optimization.RULE_REMEDIATIONS.keys()) == expected_keys


# ============================================================================
# Daily Memory Optimization: Telegram HTML escaping (v2.2.1)
# ============================================================================

class TestTelegramHtmlEscaping:
    def test_escape_html_basic(self):
        assert daily_memory_optimization._escape_html("hello world") == "hello world"

    def test_escape_html_special_chars(self):
        """Raw <, >, & must be escaped to prevent Telegram parse errors."""
        assert daily_memory_optimization._escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"

    def test_escape_html_script_tag(self):
        """Prompt-injected content in memory text must be neutralized."""
        result = daily_memory_optimization._escape_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ============================================================================
# Daily Memory Optimization: safe memory invalidation (v2.2.1)
# ============================================================================

class TestSafeMemoryInvalidation:
    def test_invalidate_world_fact_directly(self):
        """world/experience facts can be invalidated directly."""
        with patch.object(daily_memory_optimization, "_get_memory", return_value={"fact_type": "world"}), \
             patch.object(daily_memory_optimization, "_patch_invalidate", return_value=True) as mock_patch:
            result = daily_memory_optimization.invalidate_memory("mem-123", "stale")
            assert result is True
            mock_patch.assert_called_once_with("mem-123", "stale")

    def test_invalidate_experience_fact_directly(self):
        with patch.object(daily_memory_optimization, "_get_memory", return_value={"fact_type": "experience"}), \
             patch.object(daily_memory_optimization, "_patch_invalidate", return_value=True) as mock_patch:
            result = daily_memory_optimization.invalidate_memory("mem-456", "duplicate")
            assert result is True
            mock_patch.assert_called_once_with("mem-456", "duplicate")

    def test_invalidate_observation_finds_sources(self):
        """Observations can't be PATCHed directly — find and invalidate source memories."""
        observation = {
            "fact_type": "observation",
            "source_memory_ids": ["src-1", "src-2"],
        }
        with patch.object(daily_memory_optimization, "_get_memory", return_value=observation), \
             patch.object(daily_memory_optimization, "_patch_invalidate", return_value=True) as mock_patch:
            result = daily_memory_optimization.invalidate_memory("obs-789", "stale_observation")
            assert result is True
            assert mock_patch.call_count == 2
            mock_patch.assert_any_call("src-1", "stale_observation")
            mock_patch.assert_any_call("src-2", "stale_observation")

    def test_invalidate_observation_no_sources_returns_false(self):
        """Observation with no source_memory_ids can't be safely invalidated."""
        observation = {
            "fact_type": "observation",
            "source_memory_ids": [],
        }
        with patch.object(daily_memory_optimization, "_get_memory", return_value=observation), \
             patch.object(daily_memory_optimization, "_patch_invalidate") as mock_patch:
            result = daily_memory_optimization.invalidate_memory("obs-000", "stale")
            assert result is False
            mock_patch.assert_not_called()

    def test_invalidate_unknown_fact_type_returns_false(self):
        """Unknown fact types should not be touched."""
        with patch.object(daily_memory_optimization, "_get_memory", return_value={"fact_type": "unknown_type"}), \
             patch.object(daily_memory_optimization, "_patch_invalidate") as mock_patch:
            result = daily_memory_optimization.invalidate_memory("mem-???")
            assert result is False
            mock_patch.assert_not_called()

    def test_invalidate_memory_fetch_failure_returns_false(self):
        """If the GET /memories/{id} fails, don't attempt invalidation."""
        with patch.object(daily_memory_optimization, "_get_memory", return_value=None), \
             patch.object(daily_memory_optimization, "_patch_invalidate") as mock_patch:
            result = daily_memory_optimization.invalidate_memory("mem-fail")
            assert result is False
            mock_patch.assert_not_called()


# ============================================================================
# Daily Memory Optimization: --allow-destructive flag (v3.0 — rule-based)
# ============================================================================

class TestDestructiveFlag:
    def test_destructive_disabled_by_default(self):
        """ALLOW_DESTRUCTIVE should be False by default."""
        assert daily_memory_optimization.ALLOW_DESTRUCTIVE is False

    def test_rule_resolver_skips_invalidate_without_flag(self):
        """When ALLOW_DESTRUCTIVE is False, invalidate actions return None (unresolved)."""
        issue = daily_memory_optimization.Issue(
            code="EXACT_DUPLICATE",
            severity="info",
            message="exact duplicate found",
            context={"memory_id": "mem-123"},
        )
        with patch.object(daily_memory_optimization, "ALLOW_DESTRUCTIVE", False), \
             patch.object(daily_memory_optimization, "invalidate_memory") as mock_invalidate:

            resolved, unresolved = daily_memory_optimization.try_resolve_issues_with_rules([issue])

            assert len(unresolved) == 1
            assert len(resolved) == 0
            mock_invalidate.assert_not_called()

    def test_rule_resolver_allows_consolidate_without_flag(self):
        """Consolidation (non-destructive) should always be allowed."""
        issue = daily_memory_optimization.Issue(
            code="L2_CONSOLIDATION_PENDING",
            severity="info",
            message="consolidation pending",
        )
        with patch.object(daily_memory_optimization, "ALLOW_DESTRUCTIVE", False), \
             patch.object(daily_memory_optimization, "http", return_value=(200, {"ok": True})):

            resolved, unresolved = daily_memory_optimization.try_resolve_issues_with_rules([issue])

            assert len(resolved) == 1
            assert len(unresolved) == 0

    def test_rule_resolver_unknown_code_unresolved(self):
        """Issues with codes not in the allowlist must remain unresolved."""
        issue = daily_memory_optimization.Issue(
            code="UNKNOWN_PROBLEM",
            severity="warning",
            message="something we don't know how to fix",
        )
        resolved, unresolved = daily_memory_optimization.try_resolve_issues_with_rules([issue])
        assert len(unresolved) == 1
        assert len(resolved) == 0


# ============================================================================
# Memory Records: structured records, candidate generation, recency (v2.3)
# ============================================================================

class TestMemoryRecord:
    def test_record_creation(self):
        mr = memory_records
        rec = mr.MemoryRecord(
            id="abc-123",
            content="David prefers concise responses",
            fact_type="world",
            date="2026-09-02T10:00:00+00:00",
        )
        assert rec.fact_type == "world"
        assert rec.is_curable is True
        assert rec.is_observation is False

    def test_record_timestamp_priority(self):
        """timestamp should prefer occurred_start > date > edited_at."""
        mr = memory_records
        rec = mr.MemoryRecord(
            id="abc",
            content="test",
            date="2026-09-02",
            occurred_start="2026-08-01",
        )
        assert rec.timestamp == "2026-08-01"

    def test_record_timestamp_fallback_to_date(self):
        mr = memory_records
        rec = mr.MemoryRecord(id="abc", content="test", date="2026-09-02")
        assert rec.timestamp == "2026-09-02"

    def test_record_timestamp_empty(self):
        mr = memory_records
        rec = mr.MemoryRecord(id="abc", content="test")
        assert rec.timestamp == ""

    def test_observation_not_curable(self):
        mr = memory_records
        rec = mr.MemoryRecord(id="abc", content="test", fact_type="observation")
        assert rec.is_curable is False
        assert rec.is_observation is True

    def test_to_llm_dict_truncates_content(self):
        mr = memory_records
        long_content = "x" * 500
        rec = mr.MemoryRecord(id="abcdef123456", content=long_content)
        d = rec.to_llm_dict()
        assert len(d["content"]) <= 200
        assert len(d["id"]) <= 12


class TestCrossChunkCandidates:
    def test_adjacent_pairs_generated(self):
        """Adjacent records should always be candidates (cross-chunk fix)."""
        mr = memory_records
        records = [
            mr.MemoryRecord(id=f"r{i}", content=f"fact number {i}") for i in range(35)
        ]
        candidates = mr.generate_candidate_pairs(records)
        adjacent = [(i, i + 1) for i in range(34)]
        for pair in adjacent:
            assert pair in candidates, f"Adjacent pair {pair} missing from candidates"

    def test_entity_overlap_generates_candidates(self):
        """Records sharing entities should be candidates."""
        mr = memory_records
        records = [
            mr.MemoryRecord(id="r0", content="Python is preferred", tags=["python"]),
            mr.MemoryRecord(id="r1", content="Java is also used", tags=["java"]),
            mr.MemoryRecord(id="r2", content="Python version updated", tags=["python"]),
        ]
        candidates = mr.generate_candidate_pairs(records)
        assert (0, 2) in candidates

    def test_no_candidates_with_no_overlap(self):
        """Completely unrelated records should only have adjacent pairs."""
        mr = memory_records
        records = [
            mr.MemoryRecord(id="r0", content="zzzzz unrelated"),
            mr.MemoryRecord(id="r1", content="yyyyy different"),
        ]
        candidates = mr.generate_candidate_pairs(records)
        assert (0, 1) in candidates


class TestDeterministicRecency:
    def test_recency_resolved_from_timestamps(self):
        """Newer memory should be identified from timestamps."""
        mr = memory_records
        older = mr.MemoryRecord(id="old", content="old fact", date="2026-01-01")
        newer = mr.MemoryRecord(id="new", content="new fact", date="2026-09-01")
        result_newer, result_older = mr.resolve_recency((older, newer))
        assert result_newer.id == "new"
        assert result_older.id == "old"

    def test_recency_missing_timestamps_returns_none(self):
        """Missing timestamps should return (None, None) — flag for human."""
        mr = memory_records
        a = mr.MemoryRecord(id="a", content="fact a")
        b = mr.MemoryRecord(id="b", content="fact b")
        result_newer, result_older = mr.resolve_recency((a, b))
        assert result_newer is None
        assert result_older is None

    def test_recency_equal_timestamps_returns_none(self):
        """Equal timestamps should return (None, None) — ambiguous."""
        mr = memory_records
        a = mr.MemoryRecord(id="a", content="fact a", date="2026-09-01")
        b = mr.MemoryRecord(id="b", content="fact b", date="2026-09-01")
        result_newer, result_older = mr.resolve_recency((a, b))
        assert result_newer is None
        assert result_older is None

    def test_recency_order_independent(self):
        """Recency resolution should work regardless of pair order."""
        mr = memory_records
        older = mr.MemoryRecord(id="old", content="old", date="2026-01-01")
        newer = mr.MemoryRecord(id="new", content="new", date="2026-09-01")
        n1, o1 = mr.resolve_recency((older, newer))
        n2, o2 = mr.resolve_recency((newer, older))
        assert n1.id == n2.id == "new"
        assert o1.id == o2.id == "old"


# ============================================================================
# Privacy / PII redaction (v2.4)
# ============================================================================

class TestPIIRedaction:
    def test_redact_email(self):
        mr = memory_records
        result = mr.redact_pii("contact me at [EMAIL]")
        assert "[EMAIL]" not in result or result.count("[EMAIL]") == 1
        assert "[EMAIL]" in mr.redact_pii("email: [EMAIL]")

    def test_redact_api_key(self):
        mr = memory_records
        result = mr.redact_pii("the key is [API_KEY]")
        assert "[API_KEY]" in result

    def test_redact_ip(self):
        mr = memory_records
        result = mr.redact_pii("server at [IP]")
        assert "[IP]" in result

    def test_redact_multiple_types(self):
        mr = memory_records
        text = "Email: [EMAIL], IP: [IP], key: [API_KEY]"
        result = mr.redact_pii(text)
        assert "[EMAIL]" in result
        assert "[IP]" in result
        assert "[API_KEY]" in result

    def test_should_exclude_credential_tag(self):
        mr = memory_records
        rec = mr.MemoryRecord(id="x", content="some content", tags=["secret"])
        assert mr.should_exclude_from_judging(rec) is True

    def test_should_exclude_credential_content(self):
        mr = memory_records
        rec = mr.MemoryRecord(id="x", content="the password = hunter2")
        assert mr.should_exclude_from_judging(rec) is True

    def test_should_not_exclude_mention(self):
        mr = memory_records
        rec = mr.MemoryRecord(id="x", content="passwords should be hashed")
        assert mr.should_exclude_from_judging(rec) is False

    def test_prepare_for_judging_excludes_sensitive(self):
        mr = memory_records
        records = [
            mr.MemoryRecord(id="safe", content="safe content", tags=["env"]),
            mr.MemoryRecord(id="secret", content="secret stuff", tags=["credential"]),
        ]
        safe = mr.prepare_for_judging(records)
        assert len(safe) == 1
        assert safe[0]["id"] == "safe"[:12]


# ============================================================================
# Audit log (v2.4)
# ============================================================================

class TestAuditLog:
    def test_audit_entry_serialization(self):
        mr = memory_records
        entry = mr.AuditEntry(
            timestamp="2026-09-02T10:00:00",
            action="invalidate",
            memory_id="abc-123",
            reason="test reason",
        )
        jsonl = entry.to_jsonl()
        parsed = json.loads(jsonl)
        assert parsed["action"] == "invalidate"
        assert parsed["memory_id"] == "abc-123"
        assert parsed["reason"] == "test reason"

    def test_audit_log_write_and_read(self):
        mr = memory_records
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = Path(tmpdir) / "audit.jsonl"
            with patch.object(mr, "AUDIT_LOG_FILE", audit_file):
                entry = mr.AuditEntry(
                    timestamp="2026-09-02T10:00:00",
                    action="invalidate",
                    memory_id="test-123",
                    reason="duplicate",
                )
                mr.append_audit_log(entry)
                entries = mr.read_audit_log(limit=10)
                assert len(entries) == 1
                assert entries[0]["memory_id"] == "test-123"
                assert entries[0]["action"] == "invalidate"


# ============================================================================
# L3 stale-page lint trigger (v3.1)
# ============================================================================

class TestL3StalePageLintTrigger:
    """Acceptance criteria for the >=5-stale-pages lint trigger."""

    def _make_wiki(self, tmpdir, stale_count, mtime_days_ago=120,
                    updated_days_ago=None):
        """Create a wiki dir with `stale_count` stale pages + 2 fresh pages."""
        wiki = Path(tmpdir) / "kb"
        wiki.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for i in range(stale_count):
            p = wiki / f"stale{i}.md"
            if updated_days_ago is not None:
                ts = now - updated_days_ago * 86400
                d = time.strftime("%Y-%m-%d", time.localtime(ts))
                p.write_text(f"---\ntitle: Stale {i}\nupdated: {d}\n---\nbody\n")
            else:
                p.write_text(f"---\ntitle: Stale {i}\n---\nbody\n")
                os.utime(p, (now - mtime_days_ago * 86400, now - mtime_days_ago * 86400))
        for i in range(2):
            p = wiki / f"fresh{i}.md"
            p.write_text(f"---\ntitle: Fresh {i}\n---\nbody\n")
        return wiki

    def test_parse_frontmatter_updated_date_only(self):
        ts = daily_memory_optimization._parse_frontmatter_updated(
            "---\nupdated: 2026-01-15\n---\n")
        assert ts is not None
        assert time.strftime("%Y-%m-%d", time.localtime(ts)) == "2026-01-15"

    def test_parse_frontmatter_updated_datetime(self):
        ts = daily_memory_optimization._parse_frontmatter_updated(
            "---\nupdated: 2026-01-15T10:30:00\n---\n")
        assert ts is not None
        assert time.strftime("%Y-%m-%d", time.localtime(ts)) == "2026-01-15"

    def test_parse_frontmatter_updated_invalid(self):
        text = "---\nupdated: not-a-date\n---\n"
        assert daily_memory_optimization._parse_frontmatter_updated(text) is None
        assert daily_memory_optimization._parse_frontmatter_updated("no frontmatter") is None

    def test_page_age_uses_frontmatter_over_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "page.md"
            p.write_text("---\nupdated: 2026-08-30\n---\nbody\n")
            old = time.time() - 120 * 86400
            os.utime(p, (old, old))   # mtime says 120 days; frontmatter says ~4
            age = daily_memory_optimization._page_age_days(p)
            assert age < 10, f"frontmatter should win, got {age} days"

    def test_page_age_invalid_updated_falls_back_to_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "page.md"
            p.write_text("---\nupdated: garbage\n---\nbody\n")
            old = time.time() - 100 * 86400
            os.utime(p, (old, old))
            age = daily_memory_optimization._page_age_days(p)
            assert 99 < age < 101

    def test_page_age_exactly_90_days_not_stale(self):
        # Deterministic clock: integer-valued `now` makes the 90-day boundary
        # exact in float arithmetic (no sub-second drift across the threshold).
        now = float(int(time.time()))
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = self._make_wiki(tmpdir, stale_count=5, mtime_days_ago=91)
            # All five pages exactly 90 days old -> none stale (strictly >)
            for p in wiki.glob("stale*.md"):
                os.utime(p, (now - 90 * 86400, now - 90 * 86400))
            with patch.object(daily_memory_optimization.time, "time",
                              return_value=now), \
                 patch.object(daily_memory_optimization, "KB_DIR", wiki), \
                 patch.object(daily_memory_optimization, "WIKI_DIR",
                              Path(tmpdir) / "nonexistent"):
                stale, pages = daily_memory_optimization._find_stale_pages(wiki)
                assert stale == []
                # and the trigger therefore does not fire
                with patch.object(daily_memory_optimization,
                                  "_run_lint_command") as run:
                    issues = []
                    daily_memory_optimization._check_l3_wiki(issues)
                    run.assert_not_called()
                    assert issues == []

    def test_archive_and_index_pages_excluded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = Path(tmpdir) / "kb"
            wiki.mkdir(parents=True)
            now = time.time()
            old = now - 120 * 86400
            # 3 genuinely stale active pages
            for i in range(3):
                p = wiki / f"active{i}.md"
                p.write_text("body\n")
                os.utime(p, (old, old))
            # 2 stale pages in _archive/ (must not count)
            arch = wiki / "_archive"
            arch.mkdir()
            for i in range(2):
                p = arch / f"arch{i}.md"
                p.write_text("body\n")
                os.utime(p, (old, old))
            # stale index page (must not count)
            idx = wiki / "index.md"
            idx.write_text("index\n")
            os.utime(idx, (old, old))
            stale, pages = daily_memory_optimization._find_stale_pages(wiki)
            assert len(stale) == 3
            assert all("_archive" not in s.parts for s in stale)
            assert all(s.name.lower() != "index.md" for s in stale)

    def test_four_stale_pages_do_not_run_lint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = self._make_wiki(tmpdir, stale_count=4)
            with patch.object(daily_memory_optimization, "KB_DIR", wiki), \
                 patch.object(daily_memory_optimization, "WIKI_DIR",
                              Path(tmpdir) / "nonexistent"):
                with patch.object(daily_memory_optimization,
                                  "_run_lint_command") as run:
                    issues = []
                    daily_memory_optimization._check_l3_wiki(issues)
                    run.assert_not_called()
                    assert issues == []

    def test_five_stale_pages_run_exactly_one_lint_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = self._make_wiki(tmpdir, stale_count=5)
            with patch.object(daily_memory_optimization, "KB_DIR", wiki), \
                 patch.object(daily_memory_optimization, "WIKI_DIR",
                              Path(tmpdir) / "nonexistent"):
                with patch.object(daily_memory_optimization,
                                  "_run_lint_command",
                                  return_value=(None, None)) as run:
                    issues = []
                    daily_memory_optimization._check_l3_wiki(issues)
                    run.assert_called_once()
                    assert len(issues) == 1
                    assert issues[0].code == "L3_LINT_CLI_UNAVAILABLE"

    def test_lint_json_report_summarized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = self._make_wiki(tmpdir, stale_count=6)
            payload = {
                "pages_scanned": 8,
                "issues": [
                    {"severity": "error", "page": "a.md", "message": "broken link"},
                    {"severity": "warning", "page": "b.md", "message": "missing type"},
                    {"severity": "warning", "page": "c.md", "message": "long page"},
                    {"severity": "info", "page": "d.md", "message": "low confidence"},
                    {"severity": "error", "page": "e.md", "message": "orphan"},
                ],
            }
            proc = subprocess.CompletedProcess(
                ["llmwiki"], 0, stdout=json.dumps(payload), stderr="")
            with patch.object(daily_memory_optimization, "KB_DIR", wiki), \
                 patch.object(daily_memory_optimization, "WIKI_DIR",
                              Path(tmpdir) / "nonexistent"):
                with patch.object(daily_memory_optimization,
                                  "_run_lint_command",
                                  return_value=(proc, "cli")):
                    issues = []
                    daily_memory_optimization._check_l3_wiki(issues)
                    assert len(issues) == 1
                    issue = issues[0]
                    assert issue.code == "L3_LINT_REPORT"
                    assert "8 pages scanned" in issue.message
                    assert "2 errors, 2 warnings, 1 info" in issue.message
                    # up to 3 representative issues, not the full dump
                    assert issue.message.count("[error]") <= 3
                    assert "e.md" not in issue.message  # 4th+ issue not included
                    assert issue.context["error_count"] == 2
                    assert issue.context["pages_scanned"] == 8

    def test_lint_nonzero_exit_reported_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = self._make_wiki(tmpdir, stale_count=5)
            payload = {"pages_scanned": 7, "error": 1, "warning": 0, "info": 0}
            proc = subprocess.CompletedProcess(
                ["llmwiki"], 2, stdout=json.dumps(payload), stderr="boom")
            with patch.object(daily_memory_optimization, "KB_DIR", wiki), \
                 patch.object(daily_memory_optimization, "WIKI_DIR",
                              Path(tmpdir) / "nonexistent"):
                with patch.object(daily_memory_optimization,
                                  "_run_lint_command",
                                  return_value=(proc, "cli")):
                    issues = []
                    daily_memory_optimization._check_l3_wiki(issues)
                    codes = [i.code for i in issues]
                    assert "L3_LINT_REPORT" in codes
                    assert "L3_LINT_NONZERO_EXIT" in codes

    def test_lint_malformed_output_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = self._make_wiki(tmpdir, stale_count=5)
            proc = subprocess.CompletedProcess(
                ["llmwiki"], 0, stdout="not json at all", stderr="")
            with patch.object(daily_memory_optimization, "KB_DIR", wiki), \
                 patch.object(daily_memory_optimization, "WIKI_DIR",
                              Path(tmpdir) / "nonexistent"):
                with patch.object(daily_memory_optimization,
                                  "_run_lint_command",
                                  return_value=(proc, "cli")):
                    issues = []
                    daily_memory_optimization._check_l3_wiki(issues)
                    assert len(issues) == 1
                    assert issues[0].code == "L3_LINT_MALFORMED_OUTPUT"

    def test_lint_timeout_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = self._make_wiki(tmpdir, stale_count=5)
            with patch.object(daily_memory_optimization, "KB_DIR", wiki), \
                 patch.object(daily_memory_optimization, "WIKI_DIR",
                              Path(tmpdir) / "nonexistent"):
                with patch.object(daily_memory_optimization,
                                  "_run_lint_command",
                                  return_value=("timeout", "timeout")):
                    issues = []
                    daily_memory_optimization._check_l3_wiki(issues)
                    assert len(issues) == 1
                    assert issues[0].code == "L3_LINT_TIMEOUT"

    def test_lint_command_candidates_order(self):

        cmds = daily_memory_optimization._build_lint_commands(Path("/data/kb"))
        assert cmds[0] == ["llmwiki", "lint", "--wiki-dir", "/data/kb", "--json"]
        # fallback: python <scripts_dir>/llmwiki (same interpreter as this script)
        assert cmds[1][0] == (sys.executable or "python3")
        assert cmds[1][2:] == ["lint", "--wiki-dir", "/data/kb", "--json"]

    def test_extract_lint_counts_nested_summary(self):
        payload = {"summary": {"error": 2, "warning": 3, "info": 1},
                   "total_pages": 10}
        pages, counts = daily_memory_optimization._extract_lint_counts(payload)
        assert pages == 10
        assert counts == {"error": 2, "warning": 3, "info": 1}

    def test_extract_lint_counts_flat_lists(self):
        payload = {"pages": ["a.md", "b.md"],
                   "error": [{"x": 1}], "warning": [], "info": [1, 2]}
        pages, counts = daily_memory_optimization._extract_lint_counts(payload)
        assert pages == 2
        assert counts == {"error": 1, "warning": 0, "info": 2}

    def test_check_l3_wiki_never_crashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = self._make_wiki(tmpdir, stale_count=5)
            with patch.object(daily_memory_optimization, "KB_DIR", wiki), \
                 patch.object(daily_memory_optimization, "WIKI_DIR",
                              Path(tmpdir) / "nonexistent"):
                with patch.object(daily_memory_optimization,
                                  "_find_stale_pages",
                                  side_effect=RuntimeError("boom")):
                    issues = []
                    daily_memory_optimization._check_l3_wiki(issues)
                    assert len(issues) == 1
                    assert issues[0].code == "L3_WIKI_CHECK_FAILED"
