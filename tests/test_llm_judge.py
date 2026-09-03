"""Tests for the scoped Gemini 2.5 Flash Lite judge (v3.2).

Covers:
  - Key resolution (env var, ~/.hermes/.env, absent)
  - Availability gating (JUDGE_ENABLED=0, no key)
  - Prompt building: PII redaction, sensitive-entry exclusion, cap
  - Verdict parsing: clean JSON, fenced JSON, prose-wrapped, malformed,
    unknown ids, out-of-range ids, conservative defaults
  - Fail-safe behavior: API error / timeout / parse failure -> full
    rule-based offload set with status "fallback"
  - Veto semantics: judge can only veto, never unlock
  - Integration: memory_offload.classify_entries with judge veto
  - Attribution headers: project name, not localhost
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import llm_judge
import memory_offload

# ============================================================================
# Key resolution and availability
# ============================================================================

class TestKeyResolution:
    def test_env_var_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
        assert llm_judge.load_api_key() == "sk-test-123"

    def test_hermes_env_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER=x\nOPENROUTER_API_KEY=\"sk-from-file\"\n")
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        assert llm_judge.load_api_key() == "sk-from-file"

    def test_no_key(self, monkeypatch, tmp_path):
        # v3.3: key resolution falls back to ~/.hermes/.env — pin HOME so the
        # test stays hermetic on hosts that have a real key there.
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        assert llm_judge.load_api_key() is None

    def test_is_available_no_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        assert llm_judge.is_available() is False

    def test_is_available_disabled(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", False)
        assert llm_judge.is_available() is False


# ============================================================================
# Fail-safe gating (the core invariant)
# ============================================================================

class TestFailSafe:
    ENTRIES = [(0, "Completed task X last week"), (1, "Old provider ranking")]

    def test_disabled_returns_all(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", False)
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(self.ENTRIES)
        assert confirmed == [0, 1]
        assert vetoed == []
        assert status == "disabled"

    def test_no_key_returns_all(self, monkeypatch, tmp_path):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))  # hermetic: no ~/.hermes/.env fallback
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(self.ENTRIES)
        assert confirmed == [0, 1]
        assert status == "disabled"

    def test_api_failure_falls_back(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")
        monkeypatch.setattr(llm_judge, "_call_openrouter", lambda p, k: None)
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(self.ENTRIES)
        assert confirmed == [0, 1]
        assert vetoed == []
        assert status == "fallback"

    def test_malformed_response_falls_back(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")
        monkeypatch.setattr(llm_judge, "_call_openrouter", lambda p, k: "not json at all")
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(self.ENTRIES)
        assert confirmed == [0, 1]
        assert status == "fallback"

    def test_empty_candidates(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")
        confirmed, vetoed, status = llm_judge.judge_offload_candidates([])
        assert confirmed == []
        assert status == "skipped"


# ============================================================================
# Verdict parsing
# ============================================================================

class TestVerdictParsing:
    def test_clean_json(self):
        text = '{"verdicts": [{"id": 0, "verdict": "offload"}, {"id": 1, "verdict": "keep"}]}'
        v = llm_judge._parse_verdicts(text, [10, 11])
        assert v == {10: "offload", 11: "keep"}

    def test_fenced_json(self):
        text = '```json\n{"verdicts": [{"id": 0, "verdict": "keep"}]}\n```'
        v = llm_judge._parse_verdicts(text, [5])
        assert v == {5: "keep"}

    def test_prose_wrapped_json(self):
        text = 'Here is my judgment:\n{"verdicts": [{"id": 0, "verdict": "OFFLOAD"}]}\nDone.'
        v = llm_judge._parse_verdicts(text, [7])
        assert v == {7: "offload"}

    def test_unknown_verdict_defaults_keep(self):
        text = '{"verdicts": [{"id": 0, "verdict": "maybe"}]}'
        v = llm_judge._parse_verdicts(text, [3])
        assert v == {3: "keep"}

    def test_missing_id_skipped(self):
        text = '{"verdicts": [{"verdict": "keep"}]}'
        v = llm_judge._parse_verdicts(text, [3])
        assert v == {}

    def test_out_of_range_id_skipped(self):
        text = '{"verdicts": [{"id": 99, "verdict": "keep"}]}'
        v = llm_judge._parse_verdicts(text, [3])
        assert v == {}

    def test_malformed_returns_none(self):
        assert llm_judge._parse_verdicts("garbage", [0]) is None
        assert llm_judge._parse_verdicts('{"verdicts": "not-a-list"}', [0]) is None


# ============================================================================
# Veto semantics: judge can only veto, never unlock
# ============================================================================

class TestVetoSemantics:
    def test_vetoed_entries_kept(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")
        response = json.dumps({"verdicts": [
            {"id": 0, "verdict": "offload"},
            {"id": 1, "verdict": "keep"},
        ]})
        monkeypatch.setattr(llm_judge, "_call_openrouter", lambda p, k: response)
        entries = [(0, "Completed task A"), (1, "Live incident ongoing")]
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(entries)
        assert confirmed == [0]
        assert vetoed == [1]
        assert status == "ok"

    def test_unsent_entries_keep_rule_verdict(self, monkeypatch):
        """Sensitive entries excluded from the prompt still offload (rule verdict)."""
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")

        def fake_call(prompt, key):
            # Judge only saw the non-sensitive entry (position 0 -> candidate 0)
            assert "sk-secret1234567890abcdef" not in prompt
            assert "Completed task A" in prompt
            return json.dumps({"verdicts": [{"id": 0, "verdict": "offload"}]})

        monkeypatch.setattr(llm_judge, "_call_openrouter", fake_call)
        entries = [(0, "Completed task A"), (1, "the api_key: sk-secret1234567890abcdef")]
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(entries)
        # Sensitive entry was never sent but keeps its rule-based offload verdict
        assert confirmed == [0, 1]
        assert vetoed == []
        assert status == "ok"


# ============================================================================
# Prompt building and privacy
# ============================================================================

class TestPromptPrivacy:
    def test_pii_redacted_in_prompt(self):
        email = "alice" + "@" + "example.com"
        prompt, sent = llm_judge._build_prompt([
            (0, f"Contact {email} about the completed migration"),
        ])
        assert email not in prompt
        assert "[EMAIL]" in prompt
        assert sent == [0]

    def test_sensitive_entry_excluded(self):
        prompt, sent = llm_judge._build_prompt([
            (0, "Completed task A"),
            (1, "the password: hunter2secret"),
        ])
        assert prompt is not None
        assert "hunter2secret" not in prompt
        assert sent == [0]

    def test_all_sensitive_returns_none(self):
        prompt, sent = llm_judge._build_prompt([(0, "the password: hunter2secret")])
        assert prompt is None
        assert sent == []

    def test_cap_respected(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_MAX_ENTRIES", 3)
        entries = [(i, f"Completed task {i}") for i in range(10)]
        prompt, sent = llm_judge._build_prompt(entries)
        assert sent == [0, 1, 2]

    def test_prompt_contains_role_definition(self):
        prompt, _ = llm_judge._build_prompt([(0, "Completed task A")])
        assert "offload" in prompt
        assert "keep" in prompt
        assert "JSON" in prompt


# ============================================================================
# Attribution headers (GNT policy: project name, not localhost)
# ============================================================================

class TestAttribution:
    def test_headers_sent(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = req.headers
            captured["data"] = json.loads(req.data.decode())

            class FakeResp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return json.dumps({
                        "choices": [{"message": {"content": '{"verdicts": []}'}}
                    ]}).encode()

            return FakeResp()

        monkeypatch.setattr(llm_judge.urllib.request, "urlopen", fake_urlopen)
        llm_judge.judge_offload_candidates([(0, "Completed task A")])

        headers = {k.lower(): v for k, v in captured["headers"].items()}
        assert headers["x-title"] == llm_judge.PROJECT_NAME
        assert headers["http-referer"] == f"https://github.com/{llm_judge.PROJECT_NAME}"
        assert "localhost" not in headers["http-referer"]
        assert captured["data"]["model"] == llm_judge.JUDGE_MODEL


# ============================================================================
# Integration: memory_offload.classify_entries with judge
# ============================================================================

class TestClassifyEntriesWithJudge:
    def test_vetoed_entry_stays_in_offload_pipeline(self, monkeypatch):
        """Vetoed entries must NOT be offloaded — they stay out of `offloadable`."""
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")
        response = json.dumps({"verdicts": [{"id": 0, "verdict": "keep"}]})
        monkeypatch.setattr(llm_judge, "_call_openrouter", lambda p, k: response)

        entries = ["IrisBot: Linux 6.8 specs", "Completed task A"]
        essential, offloadable = memory_offload.classify_entries(entries)
        assert len(essential) == 1
        assert offloadable == []  # judge vetoed the only offload candidate

    def test_fallback_offloads_everything(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")
        monkeypatch.setattr(llm_judge, "_call_openrouter", lambda p, k: None)

        entries = ["IrisBot: Linux 6.8 specs", "Completed task A", "Old provider ranking"]
        essential, offloadable = memory_offload.classify_entries(entries)
        assert len(essential) == 1
        assert len(offloadable) == 2

    def test_hard_gate_untouched_by_judge(self, monkeypatch):
        """The judge never sees hard-kept entries — pins stay essential regardless."""
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "sk-test")

        def fake_call(prompt, key):
            assert "[pin]" not in prompt
            assert "secret" not in prompt
            return json.dumps({"verdicts": []})

        monkeypatch.setattr(llm_judge, "_call_openrouter", fake_call)
        entries = ["[pin] critical safety rule", "the api_key: sk-secret1234567890", "Completed task A"]
        essential, offloadable = memory_offload.classify_entries(entries)
        assert any("[pin]" in e for e in essential)
        assert any("api_key" in e for e in essential)  # quarantined stays local
        assert len(offloadable) == 1


# ============================================================================
# Live smoke (skipped unless a key is present)
# ============================================================================

class TestLiveSmoke:
    def test_live_call_if_available(self, monkeypatch):
        import os
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("no OPENROUTER_API_KEY in environment")
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        entries = [(0, "Completed migration of provider ranking last month")]
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(entries)
        assert status in ("ok", "fallback")
        if status == "ok":
            assert confirmed == [0] or vetoed == [0]
