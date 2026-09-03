#!/usr/bin/env python3
"""Location-aware path resolution for the agent-memory-optimization scripts.

v3.3: locations are no longer resolved from env vars plus a current-user-only
(`~`) default. Every location now probes the *existing* deployment first, so
the scripts work when run as a different user (e.g. root cron jobs against an
install under /home/ubuntu) or when Hermes already carries the answer in its
own config files.

Resolution chains (first hit wins):

  HERMES_HOME:
    1. HERMES_HOME env var (explicit — always wins, even if it doesn't exist)
    2. this module's own deployment dir: <parent>/scripts/.. validated by
       Hermes markers (config.yaml, memories/, hindsight/, skills/)
    3. ~/.hermes (current user)
    4. first existing /home/*/.hermes or /Users/*/.hermes that has markers
    5. fallback ~/.hermes (may not exist — same as the pre-v3.3 default)

  HINDSIGHT_URL / HINDSIGHT_BANK:
    1. HINDSIGHT_URL / HINDSIGHT_BANK env var
    2. $HERMES_HOME/hindsight/config.json -> api_url / bank_id
    3. defaults http://localhost:8888 / main

  Environment values (OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, ...):
    1. process env
    2. $HERMES_HOME/.env
    3. ~/.hermes/.env (covers HERMES_HOME resolved to a non-default location)

  WIKI_DIR:
    1. WIKI_DIR env var
    2. <Hermes user's home>/wiki  (HERMES_HOME.parent — the wiki lives in
       the home directory of the user who owns the Hermes install, which is
       not necessarily the current user)
    3. ~/wiki (current user)
    4. fallback ~/wiki

Stdlib only. Import-safe from every script in this directory (they all
sys.path.insert their own dir). All functions accept an optional explicit
hermes_home so callers and tests can pin the base directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Directories that may hold another user's Hermes install (root cron jobs).
_OTHER_HOME_BASES = ("/home", "/Users")

# A real Hermes home has at least one of these. The repo checkout itself
# (agent-memory-optimization/) has scripts/ and tests/ only — it must NOT
# be mistaken for a Hermes home, so "scripts" is deliberately absent.
_HERMES_MARKERS = ("config.yaml", "memories", "hindsight", "skills")

DEFAULT_HINDSIGHT_URL = "http://localhost:8888"
DEFAULT_HINDSIGHT_BANK = "main"


# === Hermes home ===

def looks_like_hermes_home(path: Path) -> bool:
    """True when path is a directory containing Hermes markers."""
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    return any((path / marker).exists() for marker in _HERMES_MARKERS)


def _probe_hermes_homes():
    """Yield candidate Hermes homes in priority order (env var excluded)."""
    # 1. This module's own deployment dir: deployed scripts live in
    #    <hermes_home>/scripts/, so the parent is the install root.
    here = Path(__file__).resolve().parent
    if here.name == "scripts":
        yield here.parent
    # 2. Current user's home
    yield Path.home() / ".hermes"
    # 3. Other users' homes (running as root against an install under /home/*)
    for base in _OTHER_HOME_BASES:
        base_path = Path(base)
        try:
            entries = sorted(base_path.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                yield entry / ".hermes"


def resolve_hermes_home() -> Path:
    """Resolve the Hermes home directory (see module docstring)."""
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env)  # explicit — always wins
    for candidate in _probe_hermes_homes():
        if looks_like_hermes_home(candidate):
            return candidate
    return Path.home() / ".hermes"


# === Hindsight endpoint / bank ===

def load_hindsight_config(hermes_home: Path | None = None) -> dict:
    """Load $HERMES_HOME/hindsight/config.json; {} when absent/invalid."""
    home = hermes_home if hermes_home is not None else resolve_hermes_home()
    config_file = home / "hindsight" / "config.json"
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _config_str(config: dict, key: str) -> str:
    """String value from a Hindsight config dict; '' when absent/None/non-str."""
    value = config.get(key)
    return value.strip() if isinstance(value, str) else ""


def resolve_hindsight_url(hermes_home: Path | None = None) -> str:
    """Resolve the Hindsight API URL (env -> hindsight config -> default)."""
    env = os.environ.get("HINDSIGHT_URL", "").strip()
    if env:
        return env
    home = hermes_home if hermes_home is not None else resolve_hermes_home()
    return _config_str(load_hindsight_config(home), "api_url") or DEFAULT_HINDSIGHT_URL


def resolve_hindsight_bank(hermes_home: Path | None = None) -> str:
    """Resolve the Hindsight bank name (env -> hindsight config -> default)."""
    env = os.environ.get("HINDSIGHT_BANK", "").strip()
    if env:
        return env
    home = hermes_home if hermes_home is not None else resolve_hermes_home()
    return _config_str(load_hindsight_config(home), "bank_id") or DEFAULT_HINDSIGHT_BANK


# === .env file values ===

def env_file_candidates(hermes_home: Path | None = None) -> list[Path]:
    """Candidate .env files in priority order (existing files only)."""
    home = hermes_home if hermes_home is not None else resolve_hermes_home()
    candidates = [home / ".env", Path.home() / ".hermes" / ".env"]
    seen, out = set(), []
    for path in candidates:
        try:
            if path.is_file() and str(path) not in seen:
                seen.add(str(path))
                out.append(path)
        except OSError:
            continue
    return out


def read_env_var(name: str, hermes_home: Path | None = None) -> str:
    """Read a variable: process env first, then candidate .env files."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    for env_file in env_file_candidates(hermes_home):
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or not line.startswith(name + "="):
                    continue
                return line.split("=", 1)[1].strip().strip("\"'")
        except OSError:
            continue
    return ""


# === Wiki directory ===

def resolve_wiki_dir(hermes_home: Path | None = None) -> Path:
    """Resolve the static LLM wiki directory (see module docstring)."""
    env = os.environ.get("WIKI_DIR", "").strip()
    if env:
        return Path(env)
    home = hermes_home if hermes_home is not None else resolve_hermes_home()
    # The wiki lives in the home dir of the user owning the Hermes install
    # (HERMES_HOME is <home>/.hermes), which is not necessarily the current
    # user's home.
    hermes_user_wiki = home.parent / "wiki"
    if hermes_user_wiki.is_dir():
        return hermes_user_wiki
    current_user_wiki = Path.home() / "wiki"
    if current_user_wiki.is_dir():
        return current_user_wiki
    return current_user_wiki
