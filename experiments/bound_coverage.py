"""Empirical coverage of the E8Q lower-bound screen (Appendix A.4).

For each (corpus, rate) it samples uniform (query, base-vector) pairs, evaluates
the normalized coarse-to-full discrepancy

    w = (d_c^2 - d^2) / (r_q * Omega)

from the stored codes (exact double-precision probe, no implementation margin),
and reports the empirical violation rate against the theoretical per-candidate
risk delta.  Theorem 3 of the paper predicts

    Pr[w > lambda] <= delta,      lambda = Phi^{-1}(1 - delta),

so the empirical curve must sit below the diagonal on a (delta, rate) plot.

Outputs one JSON and two CSVs under --out:
    bound_coverage.json          per-config summary (N, max|w|, quantiles, rates)
    bound_coverage_curve.csv     long-form (dataset, bits, delta, lambda, rate)
    bound_coverage_quantile.csv  long-form (dataset, bits, prob, w-quantile)

Usage:
    python experiments/bound_coverage.py --out results/bound_coverage
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "datasets"))

from protocol import load_corpus  # noqa: E402

try:
    from scipy.stats import norm  # noqa: E402

    PHI_INV = norm.ppf
except ImportError:  # pragma: no cover - scipy is present in the e8q env
    from math import erf, sqrt

    def PHI_INV(p):  # noqa: D103
        # Acklam-free bisection on erf; only used if scipy is missing.
        lo, hi = -40.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# Risk grid: a decade around the paper's operating point (delta = alpha/k_eff
# with alpha = 1, k = 10 gives 0.025) plus the tails the figure needs.
DELTAS = np.concatenate(
    [
        np.array([0.001, 0.002, 0.005]),
        np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1]),
        np.array([0.15, 0.2]),
    ]
)

PROBS = np.array([0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=REPO / "datasets")
    p.add_argument("--index-root", type=Path,
                   default=REPO / "results" / "adr_full_stage" / "indexes")
    p.add_argument("--datasets", nargs="+", default=[
        "sift1M", "deep1M", "gist", "openai1536", "msmarco10M",
        "stella-trec24-1024",
    ])
    p.add_argument("--bits", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--pairs", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--chunk-bytes", type=int, default=200_000_000,
                   help="gathered-rows buffer per probe call (bytes)")
    p.add_argument("--out", type=Path, default=REPO / "results" / "bound_coverage")
    return p.parse_args(argv)


def index_path(root: Path, ds: str, bits: int) -> Path:
    for suffix in ("_t12_e8q.idx", "_t12_e8q.index"):
        p = root / f"{ds}_b{bits}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no cached E8Q index for {ds} b{bits} under {root}")


class FvecsRows:
    """Random row access to an .fvecs file through an mmap.

    Row i is ``[int32 dim][dim float32]``, so a structured view gathers
    arbitrary rows without packing the corpus to a second file.  On the
    ten-million-scale corpora the whole file becomes resident as clean page
    cache after a few chunks --- tens of gigabytes charged to this process.
    ``drop_resident`` gives that back: ``MADV_DONTNEED`` discards the clean
    file pages of the mapping and the next access faults them straight back
    from the file, so the working set stays at one chunk instead of the
    corpus.  ``prefetch`` issues advisory read-ahead for the next chunk, so
    the random reads are already in flight when the probe needs them; it is
    skipped for files small enough that the cache holds them anyway, where
    the per-row syscall would cost more than the fault it saves.
    """

    def __init__(self, path: Path, dim: int, n: int):
        self.dim = dim
        self.n = n
        self.row_bytes = 4 + 4 * dim
        self.view = np.memmap(
            path, dtype=np.dtype([("d", np.int32), ("v", np.float32, dim)]), mode="r"
        )
        if self.view.shape[0] != n:
            raise ValueError(f"{path}: {self.view.shape[0]} rows, expected {n}")
        self._fd = os.open(path, os.O_RDONLY)
        self._prefetch = os.stat(path).st_size > 4 << 30

    def gather(self, ids: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.view["v"][ids])

    def prefetch(self, ids: np.ndarray) -> None:
        if not self._prefetch:
            return
        advise = os.posix_fadvise
        row_bytes = self.row_bytes
        fd = self._fd
        for i in ids.tolist():
            advise(fd, i * row_bytes, row_bytes, os.POSIX_FADV_WILLNEED)

    def drop_resident(self) -> None:
        # MADV_DONTNEED detaches the chunk's pages from this mapping, but a
        # clean file page goes back to the page cache rather than away, so
        # the corpus still accumulates as cache.  The advisory DONTNEED that
        # follows can only invalidate pages that are no longer mapped, hence
        # the order.
        self.view._mmap.madvise(mmap.MADV_DONTNEED)
        os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)


def run_config(args, E8qIndex, ds: str, bits: int) -> dict:
    ipath = index_path(args.index_root, ds, bits)
    index = E8qIndex.load(str(ipath))
    corp = load_corpus(args.data_root, ds, 10_000, load_base=False)
    n, dim, nq = corp["n"], corp["dim"], corp["nq"]
    queries = np.ascontiguousarray(corp["queries"])
    rows = FvecsRows(corp["ddir"] / f"{ds}_base.fvecs", dim, n)

    chunk = max(1024, args.chunk_bytes // (4 * dim))
    rng = np.random.default_rng(args.seed + bits)

    def draw(take):
        return (rng.integers(0, nq, size=take, dtype=np.uint32),
                rng.integers(0, n, size=take, dtype=np.uint32))

    chunks = []
    done = 0
    pending = draw(min(chunk, args.pairs))
    rows.prefetch(pending[1])
    while pending is not None:
        qidx, ids = pending
        rest = args.pairs - done - ids.size
        nxt = draw(min(chunk, rest)) if rest > 0 else None
        w = index.probe_bound_w(
            queries, rows.gather(ids), qidx, ids, num_threads=args.threads
        )
        w = np.asarray(w, dtype=np.float64)
        chunks.append(w[np.isfinite(w)])
        done += ids.size
        # Release the pages this chunk faulted in before the next chunk
        # faults in its own, so the resident set stays at one chunk rather
        # than growing to the whole corpus.
        rows.drop_resident()
        print(f"  {ds} b{bits}: {done}/{args.pairs}", flush=True)
        if nxt is not None:
            rows.prefetch(nxt[1])
        pending = nxt
    del rows
    ws = np.concatenate(chunks)

    lam = PHI_INV(1.0 - DELTAS)
    rate = (ws[None, :] > lam[:, None]).mean(axis=1)
    quant = np.quantile(ws, PROBS)
    summary = {
        "dataset": ds,
        "bits": bits,
        "index": str(ipath),
        "n": int(n),
        "dim": int(dim),
        "nq": int(nq),
        "pairs": int(ws.size),
        "mean": float(ws.mean()),
        "std": float(ws.std()),
        "skew": float(((ws - ws.mean()) ** 3).mean() / ws.std() ** 3),
        "kurtosis": float(((ws - ws.mean()) ** 4).mean() / ws.std() ** 4),
        "min": float(ws.min()),
        "max": float(ws.max()),
        "abs_p9999": float(np.quantile(np.abs(ws), 0.9999)),
        "abs_p999999": float(np.quantile(np.abs(ws), 0.999999)),
        "rates": {f"{d:.4g}": float(r) for d, r in zip(DELTAS, rate)},
        "max_rate_ratio": float(np.max(rate / DELTAS)),
    }
    return summary, ws, rate, quant


def write_outputs(out: Path, summaries: list[dict], curve_rows: list[str],
                  quant_rows: list[str]) -> None:
    (out / "bound_coverage.json").write_text(
        json.dumps({"deltas": DELTAS.tolist(), "probs": PROBS.tolist(),
                    "configs": summaries}, indent=2)
    )
    (out / "bound_coverage_curve.csv").write_text("\n".join(curve_rows) + "\n")
    (out / "bound_coverage_quantile.csv").write_text("\n".join(quant_rows) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from e8qlib import E8qIndex

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    curve_rows = ["dataset,bits,delta,lambda,empirical_rate,theoretical_rate"]
    quant_rows = ["dataset,bits,prob,w_quantile"]
    for ds in args.datasets:
        for bits in args.bits:
            print(f"[{ds} b{bits}]", flush=True)
            summary, _ws, rate, quant = run_config(args, E8qIndex, ds, bits)
            summaries.append(summary)
            lam = PHI_INV(1.0 - DELTAS)
            for d, l, r in zip(DELTAS, lam, rate):
                curve_rows.append(f"{ds},{bits},{d:.6g},{l:.6f},{r:.8g},{d:.6g}")
            for p, q in zip(PROBS, quant):
                quant_rows.append(f"{ds},{bits},{p:.6g},{q:.6f}")
            print(f"  max rate/delta = {summary['max_rate_ratio']:.4f}", flush=True)
            # Persist after every config: the sweep is long enough that a
            # restart must resume from the last completed config.
            write_outputs(args.out, summaries, curve_rows, quant_rows)

    write_outputs(args.out, summaries, curve_rows, quant_rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
