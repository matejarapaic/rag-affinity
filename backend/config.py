import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# When bundled with PyInstaller, __file__ is in a temp dir.
# sys._MEIPASS is the actual extraction directory where data files land.
if getattr(sys, 'frozen', False):
    _base_dir = Path(sys._MEIPASS)
else:
    _base_dir = Path(__file__).parent

load_dotenv(dotenv_path=_base_dir / ".env", override=True)

# user_config.json lives in CWD (writable, same as .ingest_state.json)
USER_CONFIG_PATH = Path("user_config.json")

# ── Clerk ──────────────────────────────────────────────────────────────────────
# Set these in your .env after creating a Clerk application at clerk.com.
# Leave blank to run in single-user dev mode (no auth required).
CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "")
CLERK_SECRET_KEY      = os.getenv("CLERK_SECRET_KEY", "")
# JWKS URL format: https://<your-clerk-frontend-api>/.well-known/jwks.json
# Found in Clerk dashboard → API Keys → Advanced
CLERK_JWKS_URL        = os.getenv("CLERK_JWKS_URL", "")

# ── Anthropic ──────────────────────────────────────────────────────────────────
# System-wide fallback key (optional when users bring their own via Clerk).
def load_user_config() -> dict:
    if USER_CONFIG_PATH.exists():
        try:
            with open(USER_CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_user_config(data: dict):
    existing = load_user_config()
    existing.update(data)
    with open(USER_CONFIG_PATH, "w") as f:
        json.dump(existing, f, indent=2)


_user_config = load_user_config()
ANTHROPIC_API_KEY = _user_config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY", "")

CHAT_MODEL = "claude-sonnet-4-5"


def validate_config():
    """
    Clerk mode: users supply their own Anthropic key per-account — no system key required.
    Dev mode (no Clerk): falls back to ANTHROPIC_API_KEY env var.
    Only hard-fail if neither Clerk nor a fallback key is present.
    """
    if not CLERK_JWKS_URL and not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "No ANTHROPIC_API_KEY set and Clerk is not configured. "
            "Either set ANTHROPIC_API_KEY in .env or configure Clerk "
            "(CLERK_PUBLISHABLE_KEY, CLERK_SECRET_KEY, CLERK_JWKS_URL)."
        )
