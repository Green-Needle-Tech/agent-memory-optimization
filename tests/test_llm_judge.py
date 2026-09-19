"""Tests for the scoped TypeSafe Jev judge (v3.6).

Covers:
  - Key resolution (env var, ~/.hermes/.env, absent)
  - Availability gating (JUDGE_ENABLED=0, no key)
  - State building: PII redaction, sensitive-entry exclusion, cap
  - Offload gate: fail-safe (API error / malformed answers), veto-only
    semantics, confidence gate on KEEP vetoes, unsent entries keep rule
    verdict
  - Importance classification: essential/offloadable verdicts, unknown
    choices ignored, fail-safe
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
        monkeypatch.setenv("TYPESAFE_API_KEY", "apikey-test-123")
        assert llm_judge.load_api_key() == "apikey-test-123"

    def test_hermes_env_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER=x\nTYPESAFE_API_KEY=\"apikey-from-file\"\n")
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        assert llm_judge.load_api_key() == "apikey-from-file"

    def test_no_key(self, monkeypatch, tmp_path):
        # Key resolution falls back to ~/.hermes/.env — pin HOME so the
        # test stays hermetic on hosts that have a real key there.
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        assert llm_judge.load_api_key() is None

    def test_is_available_no_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        assert llm_judge.is_available() is False

    def test_is_available_disabled(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "apikey-test-123")
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", False)
        assert llm_judge.is_available() is False


# ============================================================================
# Fail-safe gating (the core invariant)
# ============================================================================

class TestFailSafe:
    ENTRIES = [(0, "Completed task X last week"), (1, "Old provider ranking")]

    def _armed(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "apikey-test")

    def test_disabled_returns_all(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", False)
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(self.ENTRIES)
        assert confirmed == [0, 1]
        assert vetoed == []
        assert status == "disabled"

    def test_no_key_returns_all(self, monkeypatch, tmp_path):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))  # hermetic: no ~/.hermes/.env fallback
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(self.ENTRIES)
        assert confirmed == [0, 1]
        assert status == "disabled"

    def test_api_failure_falls_back(self, monkeypatch):
        self._armed(monkeypatch)
        monkeypatch.setattr(llm_judge, "_call_systemone", lambda s, q: None)
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(self.ENTRIES)
        assert confirmed == [0, 1]
        assert vetoed == []
        assert status == "fallback"

    def test_malformed_answers_fall_back(self, monkeypatch):
        self._armed(monkeypatch)
        monkeypatch.setattr(llm_judge, "_call_systemone", lambda s, q: {"unexpected": True})
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(self.ENTRIES)
        assert confirmed == [0, 1]
        assert status == "fallback"

    def test_empty_candidates(self, monkeypatch):
        self._armed(monkeypatch)
        confirmed, vetoed, status = llm_judge.judge_offload_candidates([])
        assert confirmed == []
        assert status == "skipped"


# ============================================================================
# Veto semantics + confidence gate
# ============================================================================

def _answers(verdicts):
    """Build a System One `answers` body: {qid: {choice, confidence}}."""
    return {f"e{i}": v for i, v in enumerate(verdicts)}


class TestVetoSemantics:
    def _armed(self, monkeypatch, answers):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "apikey-test")
        monkeypatch.setattr(llm_judge, "_call_systemone", lambda s, q: answers)

    def test_vetoed_entries_kept(self, monkeypatch):
        self._armed(monkeypatch, _answers([
            {"choice": "offload", "confidence": 0.9},
            {"choice": "keep", "confidence": 0.95},
        ]))
        entries = [(0, "Completed task A"), (1, "Live incident ongoing")]
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(entries)
        assert confirmed == [0]
        assert vetoed == [1]
        assert status == "ok"

    def test_low_confidence_keep_ignored(self, monkeypatch):
        """KEEP below JUDGE_MIN_CONFIDENCE does not veto — rule offload stands."""
        self._armed(monkeypatch, _answers([
            {"choice": "keep", "confidence": 0.4},
        ]))
        confirmed, vetoed, status = llm_judge.judge_offload_candidates([(0, "Completed task A")])
        assert confirmed == [0]
        assert vetoed == []
        assert status == "ok"

    def test_unsent_entries_keep_rule_verdict(self, monkeypatch):
        """Sensitive entries excluded from the state still offload (rule verdict)."""
        captured = {}

        def fake_call(state, questions):
            captured["state"] = state
            return _answers([{"choice": "offload", "confidence": 0.9}])

        self._armed(monkeypatch, None)
        monkeypatch.setattr(llm_judge, "_call_systemone", fake_call)
        entries = [(0, "Completed task A"), (1, "the api_key: «redacted:sk-…»")]
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(entries)
        assert "redacted:sk" not in captured["state"]
        assert "Completed task A" in captured["state"]
        assert confirmed == [0, 1]
        assert vetoed == []
        assert status == "ok"

    def test_missing_answer_keeps_rule_verdict(self, monkeypatch):
        self._armed(monkeypatch, {})  # no answers at all
        confirmed, vetoed, status = llm_judge.judge_offload_candidates([(0, "Completed task A")])
        assert confirmed == [0]
        assert status == "fallback"


# ============================================================================
# State building and privacy
# ============================================================================

class TestStatePrivacy:
    def test_pii_redacted_in_state(self):
        email = "alice" + "@" + "example.com"
        lines, sent = llm_judge._prepare([
            (0, f"Contact {email} about the completed migration"),
        ])
        assert email not in lines[0]
        assert "[EMAIL]" in lines[0]
        assert sent == [0]

    def test_sensitive_entry_excluded(self):
        lines, sent = llm_judge._prepare([
            (0, "Completed task A"),
            (1, "the password: hunter2secret"),
        ])
        assert lines
        assert "hunter2secret" not in lines[0]
        assert sent == [0]

    def test_all_sensitive_returns_empty(self):
        lines, sent = llm_judge._prepare([(0, "the password: hunter2secret")])
        assert lines == []
        assert sent == []

    def test_cap_respected(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_MAX_ENTRIES", 3)
        entries = [(i, f"Completed task {i}") for i in range(10)]
        lines, sent = llm_judge._prepare(entries)
        assert sent == [0, 1, 2]


# ============================================================================
# Importance classification (v3.6)
# ============================================================================

class TestJudgeImportance:
    def _armed(self, monkeypatch, answers):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "apikey-test")
        monkeypatch.setattr(llm_judge, "_call_systemone", lambda s, q: answers)

    def test_essential_and_offloadable_verdicts(self, monkeypatch):
        self._armed(monkeypatch, _answers([
            {"choice": "essential", "confidence": 0.9},
            {"choice": "offloadable", "confidence": 0.85},
        ]))
        verdicts, status = llm_judge.judge_importance([
            (0, "Hindsight endpoint localhost 8888"), (1, "Old provider ranking"),
        ])
        assert verdicts == {0: "essential", 1: "offloadable"}
        assert status == "ok"

    def test_unknown_choice_ignored(self, monkeypatch):
        self._armed(monkeypatch, _answers([{"choice": "maybe", "confidence": 0.9}]))
        verdicts, status = llm_judge.judge_importance([(0, "some entry")])
        assert verdicts == {}
        assert status == "fallback"

    def test_disabled(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", False)
        verdicts, status = llm_judge.judge_importance([(0, "some entry")])
        assert verdicts == {}
        assert status == "disabled"

    def test_api_failure_falls_back(self, monkeypatch):
        self._armed(monkeypatch, None)
        verdicts, status = llm_judge.judge_importance([(0, "some entry")])
        assert verdicts == {}
        assert status == "fallback"


# ============================================================================
# Attribution headers (GNT policy: project name, not localhost)
# ============================================================================

class TestAttribution:
    def test_headers_and_payload(self, monkeypatch):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "apikey-test")
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
                        "answers": {"e0": {"choice": "offload", "confidence": 0.9}}
                    }).encode()

            return FakeResp()

        monkeypatch.setattr(llm_judge.urllib.request, "urlopen", fake_urlopen)
        llm_judge.judge_offload_candidates([(0, "Completed task A")])

        headers = {k.lower(): v for k, v in captured["headers"].items()}
        assert headers["x-title"] == llm_judge.PROJECT_NAME
        assert headers["http-referer"] == f"https://github.com/{llm_judge.PROJECT_NAME}"
        assert "localhost" not in headers["http-referer"]
        assert headers["authorization"] == "Bearer apikey-test"
        data = captured["data"]
        assert data["model"] == llm_judge.JUDGE_MODEL
        assert "state" in data and "questions" in data
        q = data["questions"]["e0"]
        assert q["type"] == "choice"
        assert set(q["criteria"]) == {"offload", "keep"}


# ============================================================================
# Integration: memory_offload.classify_entries with judge
# ============================================================================

class TestClassifyEntriesWithJudge:
    def _armed(self, monkeypatch, answers):
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        monkeypatch.setattr(llm_judge, "load_api_key", lambda: "apikey-test")
        monkeypatch.setattr(llm_judge, "_call_systemone", lambda s, q: answers)

    def test_vetoed_entry_stays_in_offload_pipeline(self, monkeypatch):
        """Vetoed entries must NOT be offloaded — they stay out of `offloadable`."""
        self._armed(monkeypatch, _answers([{"choice": "keep", "confidence": 0.95}]))
        entries = ["IrisBot: Linux 6.8 specs", "Completed task A"]
        essential, offloadable = memory_offload.classify_entries(entries)
        assert len(essential) == 1
        assert offloadable == []  # judge vetoed the only offload candidate

    def test_fallback_offloads_everything(self, monkeypatch):
        self._armed(monkeypatch, None)
        entries = ["IrisBot: Linux 6.8 specs", "Completed task A", "Old provider ranking"]
        essential, offloadable = memory_offload.classify_entries(entries)
        assert len(essential) == 1
        assert len(offloadable) == 2

    def test_hard_gate_untouched_by_judge(self, monkeypatch):
        """The judge never sees hard-kept entries — pins stay essential regardless."""
        captured = {}

        def fake_call(state, questions):
            captured["state"] = state
            return {}

        self._armed(monkeypatch, None)
        monkeypatch.setattr(llm_judge, "_call_systemone", fake_call)
        entries = ["[pin] critical safety rule", "the api_key: «redacted:sk-…»", "Completed task A"]
        essential, offloadable = memory_offload.classify_entries(entries)
        assert "[pin]" not in captured["state"]
        assert "redacted:sk" not in captured["state"]
        assert any("[pin]" in e for e in essential)
        assert any("api_key" in e for e in essential)  # quarantined stays local
        assert len(offloadable) == 1


# ============================================================================
# Live smoke (skipped unless a key is present)
# ============================================================================

class TestLiveSmoke:
    def test_live_call_if_available(self, monkeypatch):
        import os
        if not os.environ.get("TYPESAFE_API_KEY"):
            pytest.skip("no TYPESAFE_API_KEY in environment")
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        entries = [(0, "Completed migration of provider ranking last month")]
        confirmed, vetoed, status = llm_judge.judge_offload_candidates(entries)
        assert status in ("ok", "fallback")
        if status == "ok":
            assert confirmed == [0] or vetoed == [0]

    def test_live_importance_if_available(self, monkeypatch):
        import os
        if not os.environ.get("TYPESAFE_API_KEY"):
            pytest.skip("no TYPESAFE_API_KEY in environment")
        monkeypatch.setattr(llm_judge, "JUDGE_ENABLED", True)
        verdicts, status = llm_judge.judge_importance([
            (0, "Hindsight memory server runs at localhost 8888 with bank main"),
        ])
        assert status in ("ok", "fallback")
        if status == "ok":
            assert verdicts.get(0) in ("essential", "offloadable")
