import os
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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CHAT_MODEL = "claude-sonnet-4-5"

def validate_config():
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
