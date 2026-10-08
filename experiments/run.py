#!/usr/bin/env python3
"""Unified method runner. Same CLI for every arm.

    python experiments/run.py --method e8q|rabitq|me8q|pq|saq|caq \\
        --data-root datasets --dataset gist --bits 4 --out results/<run>/stable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import METHODS, main  # noqa: E402


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", required=True, choices=METHODS)
    ns, rest = pre.parse_known_args()
    raise SystemExit(main(ns.method, rest))
