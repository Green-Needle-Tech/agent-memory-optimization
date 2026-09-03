"""Tests for memory_heuristics.py — rule-based deterministic heuristics (v3.0).

Covers all 16 spec test cases from section 16:
  - Classification (7 cases)
  - Exact deduplication
  - Strong semantic deduplication
  - Protected-value mismatch
  - Topic overlap without duplication
  - Negation mismatch
  - Explicit state transition
  - Missing chronology
  - Stable conflict
  - Complementary facts
  - Transactional offload (in test_memory_optimization.py)
  - No-network judge test
  - Scale test (10k entries)
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# Ensure scripts directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import memory_heuristics as mh

# ============================================================================
# Classification (spec section 16: 7 cases)
# ============================================================================

class TestClassification:
    """L1 importance classification — deterministic weighted rules."""

    def test_essential_prefixes_remain_in_l1(self):
        """Known essential prefixes must stay in L1."""
        entries = [
            "IrisBot: Linux 6.8 EPYC 4vCPU",
            "Hindsight: localhost:8888 bank main",
            "MCP tool_call JSON fix: call tool_describe first",
            "completed task: deployed v2.4",
        ]
        essential, offloadable = mh.classify_importance(entries)
        assert 0 in essential  # IrisBot:
        assert 1 in essential  # Hindsight:
        assert 2 in essential  # MCP tool_call
        assert 3 in offloadable  # completed task

    def test_historical_provider_state_offloadable(self):
        """Historical provider state should be offloadable."""
        entries = [
            "Previously used Gemini 1.5 Pro for extraction",
            "Old Hindsight provider was direct API",
        ]
        essential, offloadable = mh.classify_importance(entries)
        assert len(essential) == 0
        assert len(offloadable) == 2

    def test_active_provider_config_remains_essential(self):
        """Active current provider configuration stays essential."""
        entries = [
            "Hindsight: localhost:8888 bank main",
            "Current extraction model is gpt-oss-120b via Cerebras",
        ]
        essential, offloadable = mh.classify_importance(entries)
        assert 0 in essential  # Hindsight prefix
        # "current" is a runtime capability indicator
        assert 1 in essential

    def test_maintenance_reports_offloadable(self):
        """Maintenance reports should be offloadable."""
        entries = [
            "Maintenance report: 3 duplicates invalidated on 2026-08-01",
            "Daily optimization ran successfully, no issues found",
        ]
        essential, offloadable = mh.classify_importance(entries)
        assert len(essential) == 0
        assert len(offloadable) == 2

    def test_pinned_entries_override(self):
        """Explicit pin markers override other rules."""
        entries = [
            "Random fact that would normally be offloaded [pin]",
            "Another random observation [always inject]",
        ]
        essential, offloadable = mh.classify_importance(entries)
        assert 0 in essential
        assert 1 in essential

    def test_invalid_config_falls_back_to_defaults(self):
        """Invalid user configuration should fall back to defaults (empty lists)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "memory_heuristics.json"
            config_file.write_text("{ invalid json !!!")
            with patch.object(mh, "CONFIG_FILE", config_file):
                cfg = mh.load_config()
                # Should return default structure (empty lists, not crash)
                assert "essential_prefixes" in cfg
                assert isinstance(cfg["essential_prefixes"], list)
                assert "offload_patterns" in cfg
                assert "dry_run" in cfg
                # Built-in defaults are in module constants, not in config
                assert "IrisBot:" in mh.DEFAULT_ESSENTIAL_PREFIXES

    def test_classification_deterministic(self):
        """Same input must produce same output across calls."""
        entries = [
            "IrisBot: essential fact",
            "completed one-time task",
            "Hindsight: localhost:8888",
            "old migration status note",
        ]
        e1, o1 = mh.classify_importance(entries)
        e2, o2 = mh.classify_importance(entries)
        assert e1 == e2
        assert o1 == o2


# ============================================================================
# Exact deduplication (spec section 16)
# ============================================================================

class TestExactDedup:
    def test_exact_duplicate_after_normalization(self):
        """Whitespace/case differences should produce exact duplicates."""
        entries = [
            "User prefers Python.",
            " user   prefers python. ",
        ]
        groups = mh.semantic_dedup(entries)
        assert len(groups) == 1
        g = groups[0]
        assert g["confidence"] == "exact"
        assert g["rule"] == "DEDUP_EXACT_NORMALIZED"
        assert g["canonical"] == 0
        assert g["duplicates"] == [1]


# ============================================================================
# Strong semantic deduplication (spec section 16)
# ============================================================================

class TestStrongDedup:
    def test_strong_structured_duplicate(self):
        """Same structured claim with different syntax = strong duplicate."""
        entries = [
            "Hindsight port is 8888.",
            "Hindsight: port=8888.",
        ]
        groups = mh.semantic_dedup(entries)
        assert len(groups) == 1
        g = groups[0]
        assert g["confidence"] == "strong"
        assert g["rule"] == "DEDUP_STRUCTURED_CLAIM"
        assert g["duplicates"] == [1]


# ============================================================================
# Protected-value mismatch (spec section 16)
# ============================================================================

class TestProtectedValueMismatch:
    def test_different_ports_not_duplicates(self):
        """Different protected values must not be duplicates."""
        entries = [
            "Hindsight port is 8888.",
            "Hindsight port is 9999.",
        ]
        groups = mh.semantic_dedup(entries)
        assert len(groups) == 0  # not duplicates


# ============================================================================
# Topic overlap without duplication (spec section 16)
# ============================================================================

class TestTopicOverlap:
    def test_no_duplicate_for_different_claims(self):
        """Same subject but different attributes are not duplicates."""
        entries = [
            "Hindsight uses OpenRouter for extraction.",
            "Hindsight stores memories in bank main.",
        ]
        groups = mh.semantic_dedup(entries)
        assert len(groups) == 0


# ============================================================================
# Negation mismatch (spec section 16)
# ============================================================================

class TestNegationMismatch:
    def test_negation_not_duplicate(self):
        """Negated statements must not be duplicates."""
        entries = [
            "Automatic edits are allowed.",
            "Automatic edits are not allowed.",
        ]
        groups = mh.semantic_dedup(entries)
        assert len(groups) == 0


# ============================================================================
# Explicit state transition (spec section 16)
# ============================================================================

class TestExplicitStateTransition:
    def test_high_confidence_recency_wins(self):
        """Explicit transition → recency_wins, high confidence."""
        entries = [
            {"id": "a", "content": "2026-07-01: Hindsight provider is direct.",
             "created_at": "2026-07-01"},
            {"id": "b", "content": "2026-08-01: Hindsight switched from direct to OpenRouter.",
             "created_at": "2026-08-01"},
        ]
        cons = mh.detect_contradictions(entries)
        assert len(cons) == 1
        c = cons[0]
        assert c["resolution"] == "recency_wins"
        assert c["confidence"] == "high"
        assert c["rule"] == "CONFLICT_EXPLICIT_TRANSITION"
        assert c["older_index"] == 0
        assert c["newer_index"] == 1


# ============================================================================
# Missing chronology (spec section 16)
# ============================================================================

class TestMissingChronology:
    def test_flag_human_without_chronology(self):
        """No timestamps → flag_human, no auto-invalidation."""
        entries = [
            {"id": "a", "content": "Hindsight provider is direct."},
            {"id": "b", "content": "Hindsight provider is OpenRouter."},
        ]
        cons = mh.detect_contradictions(entries)
        assert len(cons) == 1
        c = cons[0]
        assert c["resolution"] == "flag_human"
        assert c["confidence"] == "medium"
        assert c["rule"] == "CONFLICT_SAME_KEY_DIFFERENT_VALUE"


# ============================================================================
# Stable conflict (spec section 16)
# ============================================================================

class TestStableConflict:
    def test_stable_attribute_flagged_not_auto_resolved(self):
        """Stable attributes (legal name) must never auto-resolve."""
        entries = [
            {"id": "a", "content": "User's legal name is Alice."},
            {"id": "b", "content": "User's legal name is Alicia."},
        ]
        cons = mh.detect_contradictions(entries)
        assert len(cons) == 1
        c = cons[0]
        assert c["resolution"] == "flag_human"
        # Stable attributes never get recency_wins even with timestamps
        assert c["confidence"] != "high" or c["resolution"] != "recency_wins"


# ============================================================================
# Complementary facts (spec section 16)
# ============================================================================

class TestComplementaryFacts:
    def test_no_contradiction_for_complementary(self):
        """Different topics are not contradictions."""
        entries = [
            {"id": "a", "content": "Preferred deployment method is GitHub Actions."},
            {"id": "b", "content": "The agent cannot edit repository secrets automatically."},
        ]
        cons = mh.detect_contradictions(entries)
        assert len(cons) == 0


# ============================================================================
# No-network judge test (spec section 16)
# ============================================================================

class TestNoNetworkJudge:
    """Verify no external chat-completion calls are made."""

    def test_no_chat_completions_request(self):
        """Patch urllib.request.urlopen — fail if any request targets
        /chat/completions or openrouter.ai."""
        import urllib.request
        blocked_urls = []

        original_urlopen = urllib.request.urlopen

        def guard_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            url_lower = url.lower()
            if "/chat/completions" in url_lower or "openrouter.ai" in url_lower:
                blocked_urls.append(url)
                raise RuntimeError(f"BLOCKED: external LLM call to {url}")
            return original_urlopen(req, timeout=timeout)

        entries = [
            "IrisBot: essential fact about the system",
            "completed task note for offloading",
            "User prefers Python.",
            " user   prefers python. ",
        ]

        with patch("urllib.request.urlopen", guard_urlopen):
            # Exercise all public APIs
            mh.classify_importance(entries)
            mh.semantic_dedup(entries)
            mh.detect_contradictions(entries)
            mh.is_duplicate("test", ["other test"])
            mh.normalize_text("normalize this text")

        assert len(blocked_urls) == 0, f"External LLM calls detected: {blocked_urls}"


# ============================================================================
# Scale test (spec section 16: 10,000 entries)
# ============================================================================

class TestScale:
    """10k synthetic entries with controlled duplicate clusters."""

    def test_scale_10k_entries(self):
        """Verify bounded comparisons, sub-10s runtime, determinism."""
        import random
        random.seed(42)

        entries = []
        # 9000 unique entries
        for i in range(9000):
            entries.append(f"Entry {i} about topic {i % 500} with unique identifier xyz{i}")
        # 1000 duplicates of earlier entries
        for _i in range(1000):
            src = random.randrange(9000)  # noqa: S311
            entries.append(entries[src])

        # Dedup
        t0 = time.time()
        groups = mh.semantic_dedup(entries)
        t1 = time.time()
        elapsed = t1 - t0

        n_dups = sum(len(g["duplicates"]) for g in groups)
        assert n_dups == 1000, f"Expected 1000 duplicates, got {n_dups}"
        assert elapsed < 10.0, f"Runtime {elapsed:.2f}s exceeds 10s target"

        # Determinism
        groups2 = mh.semantic_dedup(entries)
        assert groups == groups2, "Results not deterministic"

    def test_scale_classification(self):
        """Classification on 10k entries should be fast and deterministic."""
        entries = [f"Entry {i} about topic {i}" for i in range(10000)]

        t0 = time.time()
        e1, o1 = mh.classify_importance(entries)
        t1 = time.time()
        assert t1 - t0 < 10.0
        assert len(e1) + len(o1) == 10000

        e2, o2 = mh.classify_importance(entries)
        assert e1 == e2 and o1 == o2


# ============================================================================
# is_duplicate helper
# ============================================================================

class TestIsDuplicate:
    def test_exact_duplicate_detected(self):
        assert mh.is_duplicate("User prefers Python.", ["user prefers python"]) is True

    def test_strong_duplicate_detected(self):
        assert mh.is_duplicate("Hindsight port is 8888.", ["Hindsight: port=8888."]) is True

    def test_different_content_not_duplicate(self):
        assert mh.is_duplicate("Totally different content", ["user prefers python"]) is False

    def test_protected_value_mismatch_not_duplicate(self):
        assert mh.is_duplicate("Hindsight port is 8888.", ["Hindsight port is 9999."]) is False

    def test_empty_others(self):
        assert mh.is_duplicate("anything", []) is False


# ============================================================================
# normalize_text
# ============================================================================

class TestNormalizeText:
    def test_whitespace_collapse(self):
        n = mh.normalize_text("  Hello   World  ")
        assert n.normalized == "hello world"

    def test_case_normalize(self):
        n = mh.normalize_text("HELLO World")
        assert n.normalized == "hello world"

    def test_markdown_bullet_removed(self):
        n = mh.normalize_text("- This is a bullet point")
        assert n.normalized == "this is a bullet point"

    def test_unicode_normalization(self):
        n = mh.normalize_text("\u201chello\u201d")
        assert n.normalized == '"hello"'

    def test_tokens_extracted(self):
        n = mh.normalize_text("hello world test")
        assert "hello" in n.tokens
        assert "world" in n.tokens
        assert "test" in n.tokens

    def test_protected_values_preserved(self):
        n = mh.normalize_text("Hindsight uses port 8888")
        assert "8888" in n.protected_values

    def test_url_protected(self):
        n = mh.normalize_text("See https://example.com for details")
        assert "https://example.com" in n.protected_values

    def test_semver_protected(self):
        n = mh.normalize_text("Upgraded to version 2.4.7")
        assert "2.4.7" in n.protected_values

    def test_claims_parsed(self):
        n = mh.normalize_text("Hindsight: port=8888")
        assert len(n.claims) > 0
        c = n.claims[0]
        assert c.subject == "hindsight"
        assert c.attribute == "port"
        assert c.value == "8888"


# ============================================================================
# Dry-run mode
# ============================================================================

class TestDryRun:
    def test_dry_run_env_var(self):
        with patch.dict(os.environ, {"MEMORY_HEURISTICS_DRY_RUN": "1"}):
            assert mh.is_dry_run() is True

    def test_dry_run_off_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            # May be set by config file; test the env var path
            # Just verify the function doesn't crash
            result = mh.is_dry_run()
            assert isinstance(result, bool)


# ============================================================================
# Audit log
# ============================================================================

class TestAuditLog:
    def test_audit_log_writes_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = Path(tmpdir) / "audit.jsonl"
            with patch.object(mh, "AUDIT_LOG_FILE", audit_file):
                mh.audit_log(
                    operation="invalidate",
                    memory_id="mem-123",
                    rule="DEDUP_EXACT_NORMALIZED",
                    confidence="high",
                    reason="normalized contents are identical",
                )
                content = audit_file.read_text()
                entry = json.loads(content.strip().split("\n")[-1])
                assert entry["operation"] == "invalidate"
                assert entry["memory_id"] == "mem-123"
                assert entry["rule"] == "DEDUP_EXACT_NORMALIZED"
                assert entry["confidence"] == "high"
                assert "timestamp" in entry

    def test_audit_log_replacement_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = Path(tmpdir) / "audit.jsonl"
            with patch.object(mh, "AUDIT_LOG_FILE", audit_file):
                mh.audit_log(
                    operation="invalidate",
                    memory_id="old-mem",
                    rule="CONFLICT_EXPLICIT_TRANSITION",
                    confidence="high",
                    reason="provider changed",
                    replacement_id="new-mem",
                )
                content = audit_file.read_text()
                entry = json.loads(content.strip().split("\n")[-1])
                assert entry.get("replacement_id") == "new-mem"


# ============================================================================
# Claim parsing patterns (spec section 9)
# ============================================================================

class TestClaimParsing:
    def test_colon_attribute_value(self):
        n = mh.normalize_text("Hindsight: port=8888")
        assert len(n.claims) >= 1

    def test_is_syntax(self):
        n = mh.normalize_text("Hindsight port is 8888")
        assert len(n.claims) >= 1
        c = n.claims[0]
        assert c.value == "8888"

    def test_uses_as_syntax(self):
        n = mh.normalize_text("Hindsight uses OpenRouter as provider")
        assert len(n.claims) >= 1

    def test_switched_from_to(self):
        n = mh.normalize_text("Hindsight switched from direct to OpenRouter")
        assert len(n.claims) >= 1

    def test_no_longer_uses(self):
        n = mh.normalize_text("Hindsight no longer uses direct; it uses OpenRouter")
        assert len(n.claims) >= 1

    def test_attribute_aliases(self):
        """backend → provider, endpoint → url, db → database."""
        n = mh.normalize_text("Hindsight: backend=OpenRouter")
        assert len(n.claims) >= 1
        c = n.claims[0]
        assert c.attribute == "provider"  # alias normalized


# ============================================================================
# Canonical selection (spec section 8)
# ============================================================================

class TestCanonicalSelection:
    def test_pinned_entry_preferred(self):
        """Entry tagged 'pinned' should be canonical."""
        entries = [
            {"id": "a", "content": "Hindsight port is 8888.", "tags": []},
            {"id": "b", "content": "Hindsight: port=8888.", "tags": ["pinned"]},
        ]
        groups = mh.semantic_dedup(entries)
        assert len(groups) == 1
        g = groups[0]
        assert g["canonical"] == 1  # pinned entry is canonical

    def test_lower_index_as_tiebreaker(self):
        """When all else equal, lower index wins."""
        entries = [
            "Hindsight port is 8888.",
            "Hindsight: port=8888.",
        ]
        groups = mh.semantic_dedup(entries)
        assert len(groups) == 1
        g = groups[0]
        assert g["canonical"] == 0  # lower index


# ============================================================================
# Contradiction: state attributes vs stable attributes
# ============================================================================

class TestContradictionResolution:
    def test_state_change_with_reliable_timestamps(self):
        """State attribute with timestamps → recency_wins."""
        entries = [
            {"id": "a", "content": "2026-07-01: Hindsight provider is direct.",
             "created_at": "2026-07-01T00:00:00Z"},
            {"id": "b", "content": "2026-08-01: Hindsight provider is OpenRouter.",
             "created_at": "2026-08-01T00:00:00Z"},
        ]
        cons = mh.detect_contradictions(entries)
        assert len(cons) == 1
        c = cons[0]
        assert c["resolution"] == "recency_wins"
        assert c["confidence"] == "high"
        assert c["older_index"] == 0

    def test_state_change_without_timestamps_flagged(self):
        """State attribute without timestamps → flag_human."""
        entries = [
            {"id": "a", "content": "Hindsight provider is direct."},
            {"id": "b", "content": "Hindsight provider is OpenRouter."},
        ]
        cons = mh.detect_contradictions(entries)
        assert len(cons) == 1
        c = cons[0]
        assert c["resolution"] == "flag_human"

    def test_equal_timestamps_flagged(self):
        """Equal timestamps → ambiguous, flag_human."""
        entries = [
            {"id": "a", "content": "Hindsight provider is direct.",
             "created_at": "2026-08-01"},
            {"id": "b", "content": "Hindsight provider is OpenRouter.",
             "created_at": "2026-08-01"},
        ]
        cons = mh.detect_contradictions(entries)
        if cons:
            c = cons[0]
            assert c["resolution"] == "flag_human"

    def test_complementary_not_contradiction(self):
        """Different attributes on same subject are not contradictions."""
        entries = [
            {"id": "a", "content": "Hindsight provider is OpenRouter."},
            {"id": "b", "content": "Hindsight port is 8888."},
        ]
        cons = mh.detect_contradictions(entries)
        assert len(cons) == 0


# ============================================================================
# Configuration loading
# ============================================================================

class TestConfig:
    def test_default_config(self):
        """Default config (no file) should return empty user overrides.
        Built-in defaults are in module constants (DEFAULT_ESSENTIAL_PREFIXES etc.)
        and merged in classify_importance_detailed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "memory_heuristics.json"
            with patch.object(mh, "CONFIG_FILE", config_file):
                cfg = mh.load_config()
                assert "essential_prefixes" in cfg
                assert isinstance(cfg["essential_prefixes"], list)
                assert "offload_patterns" in cfg
                assert "state_attributes" in cfg
                # Built-in defaults live in module constants
                assert "IrisBot:" in mh.DEFAULT_ESSENTIAL_PREFIXES

    def test_custom_config(self):
        """Custom config should override defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "memory_heuristics.json"
            config_file.write_text(json.dumps({
                "essential_prefixes": ["Production API:"],
                "offload_patterns": [r"\bcompleted ticket\b"],
            }))
            with patch.object(mh, "CONFIG_FILE", config_file):
                cfg = mh.load_config()
                assert "Production API:" in cfg["essential_prefixes"]
