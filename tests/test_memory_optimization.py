"""Tests for agent-memory-optimization scripts.

Covers:
  - Helper functions (json parsing, fallback classification/dedup/contradiction)
  - LLM response validation (v2.2.1): negative indices, out-of-range, overlaps,
    invalid pairs, canonical-in-duplicates, missing fields
  - Offload safety (v2.2.1): partial success, all-fail, all-success, dedup
    canonical detection, tag handling, atomic writes, file locking
  - Telegram HTML escaping (v2.2.1)
  - Safe memory invalidation fact_type checking (v2.2.1)
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Ensure scripts directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import daily_memory_optimization
import llm_judge
import memory_offload

# ============================================================================
# LLM Judge: JSON parsing
# ============================================================================

class TestLLMJudgeParsing:
    def test_parse_json_response_direct(self):
        text = '{"essential": [0, 1], "offloadable": [2]}'
        parsed = llm_judge._parse_json_response(text)
        assert parsed == {"essential": [0, 1], "offloadable": [2]}

    def test_parse_json_response_markdown_fence(self):
        text = '```json\n[{"pair": [0, 1], "type": "state_change"}]\n```'
        parsed = llm_judge._parse_json_response(text)
        assert parsed == [{"pair": [0, 1], "type": "state_change"}]

    def test_parse_json_response_with_surrounding_prose(self):
        text = 'Here is the result:\n```\n{"canonical": 0, "duplicates": [1]}\n```\nDone.'
        parsed = llm_judge._parse_json_response(text)
        assert parsed == {"canonical": 0, "duplicates": [1]}

    def test_parse_json_response_empty_or_invalid(self):
        assert llm_judge._parse_json_response("") is None
        assert llm_judge._parse_json_response("not valid json at all") is None


# ============================================================================
# LLM Judge: Fallback functions
# ============================================================================

class TestLLMJudgeFallbacks:
    def test_fallback_classify(self):
        entries = [
            "IrisBot: Linux 6.8 specs",
            "Hindsight: localhost:8888 config",
            "Random project note about some task",
            "MCP tool_call JSON fix",
        ]
        essential, offloadable = llm_judge._fallback_classify(entries)
        assert essential == [0, 1, 3]
        assert offloadable == [2]

    def test_fallback_dedup(self):
        entries = [
            "User prefers Python programming language always",
            "User prefers Python programming language deeply",
            "Unrelated note about server uptime",
        ]
        groups = llm_judge._fallback_dedup(entries)
        assert len(groups) == 1
        assert groups[0]["canonical"] == 0
        assert groups[0]["duplicates"] == [1]

    def test_fallback_contradictions(self):
        entries = [
            "LLM provider is openrouter default",
            "LLM provider is anthropic direct",
        ]
        contradictions = llm_judge._fallback_contradictions(entries)
        assert len(contradictions) == 1
        assert contradictions[0]["pair"] == [0, 1]
        assert contradictions[0]["resolution"] == "flag_human"

    def test_classify_importance_empty(self):
        assert llm_judge.classify_importance([]) == ([], [])


# ============================================================================
# LLM Judge: Strict validation (v2.2.1)
# ============================================================================

class TestLLMJudgeValidation:
    """Tests for strict LLM response validation — the core safety layer."""

    # --- Index validation ---

    def test_validate_index_valid(self):
        assert llm_judge._validate_index(0, 5) == 0
        assert llm_judge._validate_index(4, 5) == 4

    def test_validate_index_negative(self):
        """Negative indices must be rejected — Python accepts them for list access."""
        assert llm_judge._validate_index(-1, 5) is None
        assert llm_judge._validate_index(-5, 5) is None

    def test_validate_index_out_of_range(self):
        assert llm_judge._validate_index(5, 5) is None
        assert llm_judge._validate_index(10, 5) is None

    def test_validate_index_non_integer(self):
        assert llm_judge._validate_index("abc", 5) is None
        assert llm_judge._validate_index(None, 5) is None
        assert llm_judge._validate_index(3.7, 5) == 3  # float truncates to int

    def test_validate_indices_list_valid(self):
        result = llm_judge._validate_indices_list([0, 1, 2], 5)
        assert result == [0, 1, 2]

    def test_validate_indices_list_with_negative(self):
        """A single negative index invalidates the entire list."""
        assert llm_judge._validate_indices_list([0, -1, 2], 5) is None

    def test_validate_indices_list_with_out_of_range(self):
        assert llm_judge._validate_indices_list([0, 5, 2], 5) is None

    # --- Dedup group validation ---

    def test_validate_dedup_group_valid(self):
        group = {"canonical": 0, "duplicates": [1, 2]}
        result = llm_judge._validate_dedup_group(group, 5)
        assert result == {"canonical": 0, "duplicates": [1, 2]}

    def test_validate_dedup_group_canonical_in_duplicates(self):
        """Canonical must not appear in its own duplicates list."""
        group = {"canonical": 0, "duplicates": [0, 1]}
        assert llm_judge._validate_dedup_group(group, 5) is None

    def test_validate_dedup_group_negative_index(self):
        group = {"canonical": 0, "duplicates": [-1]}
        assert llm_judge._validate_dedup_group(group, 5) is None

    def test_validate_dedup_group_out_of_range(self):
        group = {"canonical": 0, "duplicates": [10]}
        assert llm_judge._validate_dedup_group(group, 5) is None

    def test_validate_dedup_group_no_duplicates(self):
        """A group with no duplicates is meaningless."""
        group = {"canonical": 0, "duplicates": []}
        assert llm_judge._validate_dedup_group(group, 5) is None

    def test_validate_dedup_group_duplicate_index_in_group(self):
        """Same index appearing twice in duplicates is invalid."""
        group = {"canonical": 0, "duplicates": [1, 1]}
        assert llm_judge._validate_dedup_group(group, 5) is None

    def test_validate_dedup_groups_overlapping(self):
        """Overlapping groups (same index in two groups) must be rejected."""
        groups = [
            {"canonical": 0, "duplicates": [1]},
            {"canonical": 1, "duplicates": [2]},  # 1 appears in both
        ]
        assert llm_judge._validate_dedup_groups(groups, 5) is None

    def test_validate_dedup_groups_non_overlapping(self):
        groups = [
            {"canonical": 0, "duplicates": [1]},
            {"canonical": 2, "duplicates": [3]},
        ]
        result = llm_judge._validate_dedup_groups(groups, 5)
        assert result is not None
        assert len(result) == 2

    def test_validate_dedup_groups_any_invalid_rejects_all(self):
        """If any group is invalid, the entire response is rejected."""
        groups = [
            {"canonical": 0, "duplicates": [1]},
            {"canonical": -1, "duplicates": [2]},  # invalid
        ]
        assert llm_judge._validate_dedup_groups(groups, 5) is None

    # --- Contradiction pair validation ---

    def test_validate_contradiction_pair_valid(self):
        pair_data = {"pair": [0, 1], "type": "state_change", "resolution": "recency_wins", "newer_index": 1}
        result = llm_judge._validate_contradiction_pair(pair_data, 5)
        assert result is not None
        assert result["pair"] == [0, 1]
        assert result["resolution"] == "recency_wins"

    def test_validate_contradiction_pair_self_pair(self):
        """A pair where both indices are the same is meaningless."""
        pair_data = {"pair": [2, 2], "type": "state_change", "resolution": "recency_wins"}
        assert llm_judge._validate_contradiction_pair(pair_data, 5) is None

    def test_validate_contradiction_pair_wrong_length(self):
        """Pairs must have exactly 2 elements."""
        assert llm_judge._validate_contradiction_pair({"pair": [0], "type": "x"}, 5) is None
        assert llm_judge._validate_contradiction_pair({"pair": [0, 1, 2], "type": "x"}, 5) is None

    def test_validate_contradiction_pair_negative_index(self):
        pair_data = {"pair": [0, -1], "type": "state_change", "resolution": "recency_wins"}
        assert llm_judge._validate_contradiction_pair(pair_data, 5) is None

    def test_validate_contradiction_pair_newer_index_not_in_pair(self):
        """newer_index must be one of the pair indices."""
        pair_data = {"pair": [0, 1], "type": "state_change", "resolution": "recency_wins", "newer_index": 3}
        result = llm_judge._validate_contradiction_pair(pair_data, 5)
        assert result is not None
        assert result["newer_index"] is None
        # recency_wins without valid newer_index should downgrade to flag_human
        assert result["resolution"] == "flag_human"

    def test_validate_contradiction_pair_recency_wins_no_newer_index(self):
        """recency_wins without newer_index should downgrade to flag_human."""
        pair_data = {"pair": [0, 1], "type": "state_change", "resolution": "recency_wins"}
        result = llm_judge._validate_contradiction_pair(pair_data, 5)
        assert result is not None
        assert result["resolution"] == "flag_human"

    def test_validate_contradiction_pair_invalid_type_normalized(self):
        """Unknown type values are normalized to 'unknown'."""
        pair_data = {"pair": [0, 1], "type": "something_wrong", "resolution": "flag_human"}
        result = llm_judge._validate_contradiction_pair(pair_data, 5)
        assert result is not None
        assert result["type"] == "unknown"

    def test_validate_contradiction_pair_invalid_resolution_defaults_safe(self):
        """Unknown resolution values default to flag_human (safe)."""
        pair_data = {"pair": [0, 1], "type": "stable_conflict", "resolution": "auto_delete_everything"}
        result = llm_judge._validate_contradiction_pair(pair_data, 5)
        assert result is not None
        assert result["resolution"] == "flag_human"

    def test_validate_contradictions_dedup_pairs(self):
        """Same pair reported twice should be deduplicated."""
        contradictions = [
            {"pair": [0, 1], "type": "state_change", "resolution": "flag_human"},
            {"pair": [1, 0], "type": "stable_conflict", "resolution": "flag_human"},  # same unordered pair
        ]
        result = llm_judge._validate_contradictions(contradictions, 5)
        assert result is not None
        assert len(result) == 1

    def test_validate_contradictions_any_invalid_rejects_all(self):
        contradictions = [
            {"pair": [0, 1], "type": "state_change", "resolution": "flag_human"},
            {"pair": [0, -1], "type": "state_change", "resolution": "flag_human"},  # invalid
        ]
        assert llm_judge._validate_contradictions(contradictions, 5) is None

    # --- Classify response validation ---

    def test_validate_classify_response_valid(self):
        parsed = {"essential": [0, 1], "offloadable": [2]}
        result = llm_judge._validate_classify_response(parsed, 3)
        assert result is not None
        assert result[0] == [0, 1]
        assert result[1] == [2]

    def test_validate_classify_response_overlap_rejected(self):
        """Overlap between essential and offloadable must be rejected."""
        parsed = {"essential": [0, 1], "offloadable": [1, 2]}  # 1 in both
        assert llm_judge._validate_classify_response(parsed, 3) is None

    def test_validate_classify_response_negative_index(self):
        parsed = {"essential": [0, -1], "offloadable": [2]}
        assert llm_judge._validate_classify_response(parsed, 3) is None

    def test_validate_classify_response_unclassified_defaults_essential(self):
        """Unclassified entries should default to essential (keep locally — safe)."""
        parsed = {"essential": [0], "offloadable": [1]}
        result = llm_judge._validate_classify_response(parsed, 4)  # 2, 3 unclassified
        assert result is not None
        assert 2 in result[0]  # unclassified → essential
        assert 3 in result[0]

    def test_validate_classify_response_missing_keys(self):
        assert llm_judge._validate_classify_response({"essential": [0]}, 3) is None
        assert llm_judge._validate_classify_response({"offloadable": [0]}, 3) is None

    def test_validate_classify_response_not_dict(self):
        assert llm_judge._validate_classify_response([0, 1], 3) is None
        assert llm_judge._validate_classify_response("string", 3) is None


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
# Memory Offload: data-loss prevention (v2.2.1 P0)
# ============================================================================

class TestOffloadDataLossPrevention:
    """Tests that failed offloads NEVER result in local memory loss."""

    def test_partial_success_keeps_failed_entries(self):
        """Mixed success: 1 succeeds, 4 fail — all 4 failed must be kept locally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "MEMORY.md"
            # Write 5 essential + 5 offloadable entries (over 75% capacity)
            entries = []
            for i in range(5):
                entries.append(f"IrisBot: essential entry {i} with enough text to fill capacity here")
            for i in range(5):
                entries.append(f"Offloadable entry {i} with some text content here for testing")
            content = "".join(f"{e}\n§\n" for e in entries)
            mem_file.write_text(content)

            # Patch config to use temp dir
            with patch.object(memory_offload, "MEMORY_FILE", mem_file), \
                 patch.object(memory_offload, "LOCK_FILE", mem_file.with_suffix(".lock")), \
                 patch.object(memory_offload, "BACKUP_DIR", Path(tmpdir) / ".backups"), \
                 patch.object(memory_offload, "CAPACITY_MAX", 100), \
                 patch.object(memory_offload, "hindsight_health_check", return_value=True), \
                 patch.object(memory_offload, "hindsight_recall_check", return_value=False), \
                 patch.object(memory_offload, "hindsight_retain") as mock_retain, \
                 patch.object(memory_offload, "llm_judge", None):

                # Entry 0 succeeds, entries 1-4 fail
                results = [True, False, False, False, False]
                mock_retain.side_effect = results

                with patch.object(memory_offload, "sys") as mock_sys:
                    mock_sys.exit = lambda code=0: None  # prevent sys.exit
                    memory_offload._do_offload()

                # Read back the file
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
                 patch.object(memory_offload, "hindsight_retain", return_value=False), \
                 patch.object(memory_offload, "llm_judge", None):

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
                 patch.object(memory_offload, "hindsight_retain", return_value=True), \
                 patch.object(memory_offload, "llm_judge", None):

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
                 patch.object(memory_offload, "hindsight_retain") as mock_retain, \
                 patch.object(memory_offload, "llm_judge", None):

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
                 patch.object(memory_offload, "hindsight_retain", side_effect=Exception("network error")), \
                 patch.object(memory_offload, "llm_judge", None):

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

                # Backup should exist
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

    def test_normalize_text(self):
        raw = "  Hello   World \n  Test  "
        assert daily_memory_optimization._normalize_text(raw) == "hello world test"

    def test_walk_tree(self):
        tree = [
            {"name": "root1", "children": [{"name": "child1"}]},
            {"name": "root2"},
        ]
        nodes = list(daily_memory_optimization.walk_tree(tree))
        names = [n["name"] for n in nodes]
        assert names == ["root1", "child1", "root2"]


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
# Daily Memory Optimization: --allow-destructive flag (v2.2.1)
# ============================================================================

class TestDestructiveFlag:
    def test_destructive_disabled_by_default(self):
        """ALLOW_DESTRUCTIVE should be False by default."""
        # The global is set in main() from argparse; verify the default
        assert daily_memory_optimization.ALLOW_DESTRUCTIVE is False

    def test_llm_resolver_skips_invalidate_without_flag(self):
        """When ALLOW_DESTRUCTIVE is False, invalidate actions are treated as unresolved."""
        problems = ["test issue"]
        plan = [{"issue_index": 0, "action": "invalidate", "query": "test", "reason": "stale"}]

        with patch.object(daily_memory_optimization, "ALLOW_DESTRUCTIVE", False), \
             patch.object(daily_memory_optimization, "llm_judge") as mock_judge, \
             patch.object(daily_memory_optimization, "recall_recent_memories", return_value=[]), \
             patch.object(daily_memory_optimization, "invalidate_memory") as mock_invalidate:

            mock_judge._llm_chat.return_value = json.dumps(plan)
            mock_judge._parse_json_response.return_value = plan

            resolved, unresolved = daily_memory_optimization.try_resolve_issues_with_llm(problems)

            assert len(unresolved) == 1
            assert len(resolved) == 0
            mock_invalidate.assert_not_called()

    def test_llm_resolver_allows_consolidate_without_flag(self):
        """Consolidation (non-destructive) should always be allowed."""
        problems = ["test issue"]
        plan = [{"issue_index": 0, "action": "consolidate"}]

        with patch.object(daily_memory_optimization, "ALLOW_DESTRUCTIVE", False), \
             patch.object(daily_memory_optimization, "llm_judge") as mock_judge, \
             patch.object(daily_memory_optimization, "http", return_value=(200, {"ok": True})):

            mock_judge._llm_chat.return_value = json.dumps(plan)
            mock_judge._parse_json_response.return_value = plan

            resolved, unresolved = daily_memory_optimization.try_resolve_issues_with_llm(problems)

            assert len(resolved) == 1
            assert len(unresolved) == 0
