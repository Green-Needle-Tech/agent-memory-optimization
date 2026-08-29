import sys
from pathlib import Path

# Ensure scripts directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import daily_memory_optimization
import llm_judge
import memory_offload


class TestLLMJudge:
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


class TestMemoryOffload:
    def test_get_tags(self):
        assert memory_offload.get_tags("IrisBot: Linux 6.8 specs") == ["environment", "infra"]
        assert memory_offload.get_tags("Hindsight: localhost:8888") == ["hindsight", "infra"]
        assert memory_offload.get_tags("Unknown entry") == ["offloaded", "memory-management"]

    def test_classify_entries_fallback(self):
        entries = [
            "IrisBot: Linux 6.8 specs",
            "Random fact to offload",
        ]
        essential, offloadable = memory_offload.classify_entries(entries)
        assert len(essential) == 1
        assert len(offloadable) == 1
        assert "IrisBot" in essential[0]
        assert "Random fact" in offloadable[0]


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
