"""Resume-gate tests for the construction sweeps.

`ledger_complete` decides whether a resumed sweep reruns a cell. Two ledger
shapes must be accepted: a finished search ledger (survivor triple present)
and a finished `--build-only` cell, which carries a single `nprobe = -1`
placeholder row and no search results at all. The second shape used to be
rejected for the prune-stats methods, which made a resumed construction sweep
rebuild every cell it had already measured.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from protocol import ledger_complete  # noqa: E402


def write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_search_ledger_with_survivor_triple_is_complete(tmp_path):
    p = write(
        tmp_path,
        "msmarco_b4_t12_e8q.json",
        {"rows": [{"nprobe": 64, "survivor_rate": 0.71, "recall": 0.9}]},
    )
    assert ledger_complete(p)


def test_search_ledger_without_survivor_triple_is_rerun(tmp_path):
    # Pre-stats-patch ledger: real search rows, no survivor_rate.
    p = write(
        tmp_path,
        "msmarco_b4_t12_e8q.json",
        {"rows": [{"nprobe": 64, "recall": 0.9}]},
    )
    assert not ledger_complete(p)


@pytest.mark.parametrize("method", ["e8q", "me8q", "rabitq"])
def test_build_only_ledger_is_complete(tmp_path, method):
    p = write(
        tmp_path,
        f"msmarco_b4_t12_{method}.json",
        {"rows": [{"nprobe": -1}], f"{method}_build_s": 67.0},
    )
    assert ledger_complete(p)


def test_build_only_ledger_without_timing_is_rerun(tmp_path):
    # A crash before the build finished leaves no ledger; a leftover without a
    # timing (e.g. -1.0 from a failed build) must not pass as complete.
    p = write(
        tmp_path,
        "msmarco_b4_t12_e8q.json",
        {"rows": [{"nprobe": -1}], "e8q_build_s": -1.0},
    )
    assert not ledger_complete(p)


def test_saq_build_only_ledger_is_complete(tmp_path):
    # saq/caq never needed the survivor triple, and must stay complete too.
    p = write(
        tmp_path,
        "msmarco_b4_t12_saq.json",
        {"rows": [{"nprobe": -1}], "saq_build_s": 55.6},
    )
    assert ledger_complete(p)


def test_truncated_ledger_is_rerun(tmp_path):
    p = tmp_path / "msmarco_b4_t12_e8q.json"
    p.write_text('{"rows": [')
    assert not ledger_complete(p)
