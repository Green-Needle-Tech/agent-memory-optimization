"""Tests for location-aware path resolution (v3.3, scripts/paths.py).

Covers:
  - HERMES_HOME: env var always wins; deployment-dir detection (a copy of
    the scripts inside <hermes_home>/scripts resolves to <hermes_home>);
    current-user home; other-user home (/home/*/.hermes) when run as root;
    repo checkout NOT mistaken for a Hermes home; fallback when nothing
    looks like a Hermes install
  - Hindsight URL/bank: env var wins; $HERMES_HOME/hindsight/config.json
    (api_url / bank_id) probed; defaults; malformed config falls back
  - read_env_var: env var wins; $HERMES_HOME/.env; ~/.hermes/.env fallback
    when HERMES_HOME points elsewhere; comments skipped; missing -> ""
  - WIKI_DIR: env var wins; Hermes-user-home wiki preferred over the
    current user's ~/wiki
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import paths

# ============================================================================
# HERMES_HOME resolution
# ============================================================================

class TestHermesHome:
    def test_env_var_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert paths.resolve_hermes_home() == tmp_path

    def test_env_var_wins_even_when_nonexistent(self, monkeypatch, tmp_path):
        """An explicit HERMES_HOME is honored even if it doesn't exist."""
        explicit = tmp_path / "nowhere"
        monkeypatch.setenv("HERMES_HOME", str(explicit))
        assert paths.resolve_hermes_home() == explicit

    def test_deployment_dir_detection(self, monkeypatch, tmp_path):
        """A copy of the scripts inside <hermes_home>/scripts resolves
        HERMES_HOME to <hermes_home> (the existing deployment, not ~)."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        hermes_home = tmp_path / "hermes"
        (hermes_home / "scripts").mkdir(parents=True)
        (hermes_home / "config.yaml").write_text("model: test\n")
        monkeypatch.setattr(paths, "__file__", str(hermes_home / "scripts" / "paths.py"))
        # Current user's ~/.hermes also exists in the probe order but the
        # deployment dir is probed first.
        assert paths.resolve_hermes_home() == hermes_home

    def test_repo_checkout_not_mistaken_for_hermes_home(self, monkeypatch, tmp_path):
        """The repo checkout (scripts/ + tests/, no Hermes markers) must not
        be detected as a Hermes home via the deployment-dir probe."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        repo = tmp_path / "agent-memory-optimization"
        (repo / "scripts").mkdir(parents=True)
        (repo / "tests").mkdir(parents=True)
        monkeypatch.setattr(paths, "__file__", str(repo / "scripts" / "paths.py"))
        # No other candidate exists -> falls back to ~/.hermes
        assert paths.resolve_hermes_home() == Path.home() / ".hermes"

    def test_other_user_home_probed(self, monkeypatch, tmp_path):
        """When run as a different user (e.g. root), an existing install
        under /home/<user>/.hermes is found via the /home scan."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        # Neutralize the deployment-dir and current-user probes
        monkeypatch.setattr(paths, "__file__", str(tmp_path / "lib" / "paths.py"))
        monkeypatch.setenv("HOME", str(tmp_path / "current_user"))

        other_home = tmp_path / "home" / "ubuntu" / ".hermes"
        other_home.mkdir(parents=True)
        (other_home / "config.yaml").write_text("model: test\n")
        monkeypatch.setattr(paths, "_OTHER_HOME_BASES", (str(tmp_path / "home"),))

        assert paths.resolve_hermes_home() == other_home

    def test_markers_required_for_other_user_home(self, monkeypatch, tmp_path):
        """A /home/*/.hermes directory without Hermes markers is skipped."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(paths, "__file__", str(tmp_path / "lib" / "paths.py"))
        monkeypatch.setenv("HOME", str(tmp_path / "current_user"))

        bare = tmp_path / "home" / "ubuntu" / ".hermes"
        bare.mkdir(parents=True)  # exists but has no markers
        monkeypatch.setattr(paths, "_OTHER_HOME_BASES", (str(tmp_path / "home"),))

        assert paths.resolve_hermes_home() == tmp_path / "current_user" / ".hermes"

    def test_looks_like_hermes_home(self, tmp_path):
        assert paths.looks_like_hermes_home(tmp_path) is False
        (tmp_path / "memories").mkdir()
        assert paths.looks_like_hermes_home(tmp_path) is True

    def test_missing_dir_not_hermes_home(self, tmp_path):
        assert paths.looks_like_hermes_home(tmp_path / "nope") is False


# ============================================================================
# Hindsight URL / bank
# ============================================================================

class TestHindsightResolution:
    def test_env_url_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HINDSIGHT_URL", "http://hindsight.local:9999")
        assert paths.resolve_hindsight_url(tmp_path) == "http://hindsight.local:9999"

    def test_env_bank_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HINDSIGHT_BANK", "custom-bank")
        assert paths.resolve_hindsight_bank(tmp_path) == "custom-bank"

    def test_config_file_probed(self, monkeypatch, tmp_path):
        """api_url / bank_id from $HERMES_HOME/hindsight/config.json are used
        when no env var is set (the existing deployment's own config)."""
        monkeypatch.delenv("HINDSIGHT_URL", raising=False)
        monkeypatch.delenv("HINDSIGHT_BANK", raising=False)
        hs_dir = tmp_path / "hindsight"
        hs_dir.mkdir()
        (hs_dir / "config.json").write_text(json.dumps({
            "api_url": "http://192.168.1.50:8888",
            "bank_id": "prod",
        }))
        assert paths.resolve_hindsight_url(tmp_path) == "http://192.168.1.50:8888"
        assert paths.resolve_hindsight_bank(tmp_path) == "prod"

    def test_defaults_without_config(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HINDSIGHT_URL", raising=False)
        monkeypatch.delenv("HINDSIGHT_BANK", raising=False)
        assert paths.resolve_hindsight_url(tmp_path) == "http://localhost:8888"
        assert paths.resolve_hindsight_bank(tmp_path) == "main"

    def test_malformed_config_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HINDSIGHT_URL", raising=False)
        monkeypatch.delenv("HINDSIGHT_BANK", raising=False)
        hs_dir = tmp_path / "hindsight"
        hs_dir.mkdir()
        (hs_dir / "config.json").write_text("{not json")
        assert paths.resolve_hindsight_url(tmp_path) == "http://localhost:8888"
        assert paths.resolve_hindsight_bank(tmp_path) == "main"

    def test_blank_config_value_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HINDSIGHT_URL", raising=False)
        monkeypatch.delenv("HINDSIGHT_BANK", raising=False)
        hs_dir = tmp_path / "hindsight"
        hs_dir.mkdir()
        (hs_dir / "config.json").write_text(json.dumps({"api_url": "", "bank_id": None}))
        assert paths.resolve_hindsight_url(tmp_path) == "http://localhost:8888"
        assert paths.resolve_hindsight_bank(tmp_path) == "main"


# ============================================================================
# .env values (OPENROUTER_API_KEY etc.)
# ============================================================================

class TestReadEnvVar:
    def test_env_var_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
        assert paths.read_env_var("OPENROUTER_API_KEY", hermes_home=tmp_path) == "sk-env"

    def test_hermes_env_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        (tmp_path / ".env").write_text(
            "OTHER=x\nOPENROUTER_API_KEY=\"sk-from-file\"\n"
        )
        assert paths.read_env_var("OPENROUTER_API_KEY", hermes_home=tmp_path) == "sk-from-file"

    def test_comments_skipped(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        (tmp_path / ".env").write_text(
            "# OPENROUTER_API_KEY=sk-commented\nOPENROUTER_API_KEY=sk-real\n"
        )
        assert paths.read_env_var("OPENROUTER_API_KEY", hermes_home=tmp_path) == "sk-real"

    def test_current_user_fallback(self, monkeypatch, tmp_path):
        """When HERMES_HOME is elsewhere, ~/.hermes/.env is still probed."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        home = tmp_path / "current_user"
        (home / ".hermes").mkdir(parents=True)
        (home / ".hermes" / ".env").write_text("OPENROUTER_API_KEY=sk-home\n")
        monkeypatch.setenv("HOME", str(home))
        # hermes_home points elsewhere and has no .env
        assert paths.read_env_var("OPENROUTER_API_KEY", hermes_home=tmp_path) == "sk-home"

    def test_missing_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "nouser"))
        assert paths.read_env_var("OPENROUTER_API_KEY", hermes_home=tmp_path) == ""

    def test_llm_judge_uses_paths(self, monkeypatch, tmp_path):
        """llm_judge.load_api_key resolves via paths (deployment .env)."""
        import llm_judge
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-judge\n")
        monkeypatch.setattr(llm_judge, "HERMES_HOME", tmp_path)
        assert llm_judge.load_api_key() == "sk-judge"


# ============================================================================
# WIKI_DIR resolution
# ============================================================================

class TestWikiDir:
    def test_env_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WIKI_DIR", str(tmp_path / "custom-wiki"))
        assert paths.resolve_wiki_dir(tmp_path) == tmp_path / "custom-wiki"

    def test_hermes_user_home_wiki_preferred(self, monkeypatch, tmp_path):
        """The wiki in the Hermes install owner's home dir wins over the
        current user's ~/wiki."""
        monkeypatch.delenv("WIKI_DIR", raising=False)
        hermes_home = tmp_path / "home" / "ubuntu" / ".hermes"
        hermes_home.mkdir(parents=True)
        (hermes_home.parent / "wiki").mkdir()  # <hermes user home>/wiki

        current = tmp_path / "current_user"
        (current / "wiki").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(current))

        assert paths.resolve_wiki_dir(hermes_home) == hermes_home.parent / "wiki"

    def test_current_user_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WIKI_DIR", raising=False)
        current = tmp_path / "current_user"
        (current / "wiki").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(current))
        # hermes_home with no sibling wiki
        hermes_home = tmp_path / "elsewhere" / ".hermes"
        assert paths.resolve_wiki_dir(hermes_home) == current / "wiki"

    def test_default_when_nothing_exists(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WIKI_DIR", raising=False)
        current = tmp_path / "current_user"
        monkeypatch.setenv("HOME", str(current))
        hermes_home = tmp_path / "elsewhere" / ".hermes"
        assert paths.resolve_wiki_dir(hermes_home) == current / "wiki"
