"""
Per-user Anthropic API key storage.

Keys are persisted in user_keys.json (CWD), keyed by Clerk user id.
The resolution order for any API call is:
  1. User's personal key (this file)
  2. System-wide fallback (ANTHROPIC_API_KEY env var / user_config.json)
  3. Error — prompt the user to add their key in Settings
"""
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_KEYS_FILE = Path("user_keys.json")


def _load() -> dict:
    if _KEYS_FILE.exists():
        try:
            return json.loads(_KEYS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save(data: dict):
    _KEYS_FILE.write_text(json.dumps(data, indent=2))


def get_user_key(user_id: str) -> str | None:
    """Return the saved Anthropic API key for this user, or None."""
    return _load().get(user_id)


def save_user_key(user_id: str, api_key: str):
    """Persist an Anthropic API key for this user."""
    data = _load()
    data[user_id] = api_key
    _save(data)
    log.info("Saved API key for user %s", user_id)


def delete_user_key(user_id: str):
    """Remove a stored key (e.g. when the user wants to clear it)."""
    data = _load()
    if user_id in data:
        del data[user_id]
        _save(data)
