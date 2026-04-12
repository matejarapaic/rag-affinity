"""
Drive Poller
============
Runs the ingestion pipeline every POLL_INTERVAL seconds.
Picks up new files, edited files, and renamed files automatically.

Usage:
    python poller.py             # polls every 5 minutes (default)
    python poller.py --interval 60   # polls every 60 seconds
"""

import time
import argparse
import logging
from ingest_gdrive import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INTERVAL = 300  # 5 minutes


def poll(interval: int):
    log.info(f"Poller started — checking Drive every {interval}s")
    while True:
        log.info("--- Running ingestion ---")
        try:
            run()
        except Exception as e:
            log.error(f"Ingestion error: {e}")
        log.info(f"--- Done. Next run in {interval}s ---")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Poll interval in seconds (default: 300)")
    args = parser.parse_args()
    poll(args.interval)
