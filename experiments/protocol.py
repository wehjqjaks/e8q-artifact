"""Shared measurement protocol for every method runner.

The full data contract is README.md. Summary:

  MAIN  (exp_stable.sh) — every method (e8q, me8q, rabitq, saq, caq,
        pq) × 14 datasets × b in 1..8 × the 53-point nprobe grid. One JSON
        per (dataset, method, bits) holds every per-cell stat AND every
        per-arm stat (memory, build time, both relative-error probes) so
        a missing field never forces a rerun.

  ABLATION (the Section 4.3 sweeps) — e8q only: prune-alpha sweep +
        no-prune, thread scaling, k sweep. Reuses the same row schema.
        Extra search-time knobs (alpha, no-prune, threads, k) are collected
        here so a later figure never forces a second MAIN grid.

Filename (every arm)::

    <out>/<ds>_b<bits>_t<T>_<method>.json

Index path (when persisted)::

    <run>/indexes/<ds>_b<bits>_t<T>_<method>.idx   (pq uses .faiss)

JSON header (live method only)::

    dataset, n, dim, nq, bits, threads, k, seed, warmup, repeats, min_ms
    <method>_build_s, <method>_mem_gb, <method>_bytes_per_vector
    <method>_relerr_avg, <method>_relerr_max          (SAQ def, nprobe=200)

JSON row::

    nprobe, <method>_recall, <method>_qps, <method>_qps_p25, <method>_qps_p75,
    <method>_nrounds, <method>_ratio,
    <method>_relerr_avg, <method>_relerr_max   (per cell, same SAQ definition)
    e8q/me8q/rabitq also: survivor_rate, <method>_candidates, <method>_survivors
    (collected in a separate untimed pass; never measured with instrumentation
    on)

Relative error, SAQ paper definition: err = |D_hat - Delta| / Delta over
every returned (query, rank) pair. D_hat is the distance the method's own
search path reports; Delta is exact squared L2 in the SAME domain the
estimator lives in (original vectors for e8q/rabitq/pq/saq/caq;
full-rank PCA domain for me8q). Computed on every nprobe cell from the
last timed round's (ids, dists) so a later figure never needs a rerun.
Header <m>_relerr_* is the nprobe = RELERR_PROBE (200) slice. SAQ/CAQ
read the same quantity from test_relative_error at nprobe=200.

Floats in the ledger are written with 10 decimal places.

Memory: <method>_mem_gb is the total in-memory index size at rest; SAQ/CAQ
measure the persisted quantized index file (their in-memory layout is
owned by the binary).

Machine gating: ram_guard() compares a per-arm memory estimate against
/proc/meminfo before anything is loaded and exits 0 with a `ram-skip:`
line (no ledger file → the arm is retried on a bigger machine).

Timing: warmup untimed passes, then rounds until >= repeats AND >= min-ms
(capped at 64). Headline QPS = nq / median(round seconds). Recall and ratio
use the last timed round's ids.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# Longer method tokens first so parse_stem('_me8q') does not match '_e8q'.
METHODS = ("me8q", "rabitq", "e8q", "saq", "caq", "pq")
DEFAULT_GRID = (
    "1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,28,32,36,40,44,48,"
    "56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,"
    "200,208,240,256,272,304,336,368,400,432,464,496,512"
)
RELERR_PROBE = 200  # nprobe of the SAQ-definition relative-error probe


def _fmt_bits(bits) -> str:
    """Stem-safe bit label: 1 -> '1', 0.5 -> '0.5' (mixed grid, no collisions)."""
    b = float(bits)
    i = int(b)
    return str(i) if b == i else f"{b:g}"


def ledger_filename(dataset: str, bits, threads: int, method: str, trial: int = 1) -> str:
    """Trials > 1 append `_r<m>` before the method infix."""
    if trial and trial > 1:
        return f"{dataset}_b{_fmt_bits(bits)}_t{threads}_r{int(trial)}_{method}.json"
    return f"{dataset}_b{_fmt_bits(bits)}_t{threads}_{method}.json"


def index_filename(dataset: str, bits, threads: int, method: str) -> str:
    ext = ".faiss" if method == "pq" else ".idx"
    return f"{dataset}_b{_fmt_bits(bits)}_t{threads}_{method}{ext}"


def parse_stem(stem: str) -> tuple[str, str, int | float, int] | None:
    """`<ds>_b<bits>_t<T>[_r<m>]_<method>` → (ds, method, bits, threads).

    Bits may be fractional (mixed-precision budget grid, e.g. `b0.5`);
    the per-segment quantizer rates stay integers — the fractional budget
    is only the DP-level total. The trial index is not part of the identity
    tuple (plotters aggregate trials).
    """
    stem = re.sub(r"_r\d+$", "", stem) if stem.count("_") >= 4 else stem
    for method in METHODS:
        suffix = f"_{method}"
        if not stem.endswith(suffix):
            continue
        rest = stem[: -len(suffix)]
        rest = re.sub(r"_r\d+$", "", rest)
        t_at = rest.rfind("_t")
        b_at = rest.rfind("_b")
        if t_at < 0 or b_at < 0 or b_at > t_at:
            continue
        ds = rest[:b_at]
        try:
            bits = float(rest[b_at + 2 : t_at])
            bits = int(bits) if bits.is_integer() else bits
            threads = int(rest[t_at + 2 :])
        except ValueError:
            continue
        if ds:
            return ds, method, bits, threads
    for method in METHODS:
        token = f"_{method}_b"
        i = stem.rfind(token)
        if i < 0:
            continue
        ds = stem[:i]
        rest = stem[i + len(token):]
        try:
            b_part, t_part = rest.split("_t", 1)
            return ds, method, int(b_part), int(t_part)
        except ValueError:
            continue
    return None


def parse_grid(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            a, b = tok.split(":")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def corpus_shape(data_root: Path, dataset: str) -> tuple[int, int, int]:
    """Cheap (n, dim, nq) from fvecs headers — no data loaded."""
    sys.path.insert(0, str(REPO / "datasets"))
    from prepare import fvecs_shape  # noqa: WPS433

    ddir = data_root / dataset
    n, dim = fvecs_shape(ddir / f"{dataset}_base.fvecs")
    nq, _ = fvecs_shape(ddir / f"{dataset}_query.fvecs")
    return int(n), int(dim), int(nq)


def mem_estimate(method: str, n: int, dim: int, bits: float) -> float:
    """Index-at-rest bytes for the RAM guard (README.md)."""
    b = float(bits)
    payload = {"e8q": b * 2.0 - 1.0, "me8q": b, "rabitq": b,
               "pq": b, "saq": b, "caq": b}[method]
    per_vec = dim * payload / 8.0 + 64.0
    return per_vec * n


def ram_guard(need_bytes: float, label: str) -> bool:
    """Exit 0 (ram-skip) when the arm cannot fit in this machine's RAM.

    The runner writes no ledger file, so the arm is retried on a machine
    with more memory instead of being silently dropped. Margin (GiB) is
    E8Q_RAM_MARGIN (default 6) — corpus + index + runtime overhead.
    """
    margin = float(os.environ.get("E8Q_RAM_MARGIN", "6")) * 2**30
    try:
        with open("/proc/meminfo") as fh:
            total_kb = int(next(line for line in fh if line.startswith("MemTotal")).split()[1])
        total = total_kb * 1024
    except (OSError, StopIteration, ValueError):
        return True  # no /proc/meminfo (non-Linux): let the loader decide
    if need_bytes + margin > total:
        print(
            f"ram-skip: {label} need={need_bytes / 2**30:.1f}G "
            f"total={total / 2**30:.1f}G margin={margin / 2**30:.0f}G",
            flush=True,
        )
        return False
    return True


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data-root", type=Path, default=REPO / "datasets")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bits", type=float, default=4)
    ap.add_argument("--grid", default=DEFAULT_GRID)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--min-ms", type=float, default=400.0)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="persist/reuse this index file (flat under <run>/indexes/)",
    )
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--relerr", type=int, default=RELERR_PROBE,
                    help="relative-error probe switch: >0 runs the SAQ "
                         "binary's test_relative_error (saq/caq arms); the "
                         "python methods collect the nprobe=200 probe "
                         "automatically")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--trial", type=int, default=1,
                    help="inter-batch trial index (stems gain _r<m> when > 1)")
    ap.add_argument("--adr", action="store_true",
                    help="also record the sum-ratio ADR@k (README.md)")
    ap.add_argument("--latency-nq", type=int, default=0,
                    help=">0: time this many single queries, save a .npy "
                         "sidecar and per-row latency quantiles (§6.2)")
    ap.add_argument("--phases", action="store_true",
                    help="record per-phase us/query rows (§6.3) when the "
                         "method exposes SearchStats phases")
    ap.add_argument("--cold-load", action="store_true",
                    help="record index load seconds after the caller dropped "
                         "the page cache (§5.4)")
    ap.add_argument("--mmap-base", action="store_true",
                    help="keep the corpus on disk: score ratios from a "
                         "strided fvecs mapping and build from a packed copy "
                         "(for corpora larger than this machine's RAM)")
    ap.add_argument("--build-only", action="store_true",
                    help="build the index, record its construction time and "
                         "memory, then stop: no nprobe sweep, no QPS rounds. "
                         "The ledger carries one placeholder row and all the "
                         "build columns (build s, KVPS, index bytes/vector, "
                         f"peak RSS) -- the construction tables' data path")


def load_corpus(data_root: Path, dataset: str, nq: int, *, load_base: bool = True,
                load_pca: bool = False, load_pca_base: bool = True):
    """Return a dict with n, dim, nq, queries, gt, centroids, cids, optional base/PCA."""
    sys.path.insert(0, str(REPO / "datasets"))
    from prepare import fvecs_shape, read_fvecs, read_ivecs  # noqa: WPS433

    ddir = data_root / dataset
    n, dim = fvecs_shape(ddir / f"{dataset}_base.fvecs")
    nq_all, _ = fvecs_shape(ddir / f"{dataset}_query.fvecs")
    nq = min(nq, nq_all)
    out: dict[str, Any] = {
        "n": int(n),
        "dim": int(dim),
        "nq": int(nq),
        "ddir": ddir,
        "queries": np.ascontiguousarray(read_fvecs(ddir / f"{dataset}_query.fvecs")[:nq]),
        "gt": read_ivecs(ddir / f"{dataset}_groundtruth.ivecs")[:nq],
        "centroids": np.ascontiguousarray(read_fvecs(ddir / f"{dataset}_centroid_4096.fvecs")),
        "cids": read_ivecs(ddir / f"{dataset}_cluster_id_4096.ivecs")[:, 0].astype(np.uint32),
        "base": None,
        "queries_pca": None,
        "centroids_pca": None,
        "base_pca": None,
        "vars": None,
    }
    if load_base:
        out["base"] = np.ascontiguousarray(read_fvecs(ddir / f"{dataset}_base.fvecs"))
    if load_pca:
        # `load_pca_base=False` leaves the big PCA matrix out (a caller that
        # streams it from a packed mmap instead); everything else in this
        # block is small: queries, centroids, and the per-dimension variances.
        if load_pca_base:
            out["base_pca"] = np.ascontiguousarray(
                read_fvecs(ddir / f"{dataset}_base_pca.fvecs")
            )
        out["queries_pca"] = np.ascontiguousarray(
            read_fvecs(ddir / f"{dataset}_query_pca.fvecs")[:nq]
        )
        out["centroids_pca"] = np.ascontiguousarray(
            read_fvecs(ddir / f"{dataset}_centroid_4096_pca.fvecs")
        )
        vars_path = ddir / f"{dataset}_base_pca.vars.fvecs"
        if vars_path.exists():
            out["vars"] = np.ascontiguousarray(read_fvecs(vars_path)[0])
    return out


class FvecsRowReader:
    """Random access to an .fvecs file without materializing the matrix.

    Rows decode on demand from an mmap of the file (fvecs layout: each row
    is [int32 dim][dim x float32]); peak memory is only the touched rows.
    Used by ADR@k when the raw-space base would not co-reside with the PCA
    copy the search itself needs (stella-trec24-1024: two full copies plus
    the index exceed host RAM)."""

    def __init__(self, path: Path):
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        advise_random(self._mm)
        self.dim = int(np.frombuffer(self._mm[:4].tobytes(), dtype=np.int32)[0])
        self._row = 4 + 4 * self.dim
        self.shape = (self._mm.size // self._row, self.dim)

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Return the (len(ids), dim) float32 matrix of the given row ids."""
        out = np.empty((len(ids), self.dim), dtype=np.float32)
        for i, ix in enumerate(ids):
            off = int(ix) * self._row + 4
            out[i] = np.frombuffer(
                self._mm[off:off + 4 * self.dim].tobytes(), dtype=np.float32)
        return out


def _mmap_backing(obj: Any) -> Any:
    """The ``mmap.mmap`` behind a numpy array or view, or None if it has none."""
    seen = 0
    while obj is not None and seen < 8:
        if isinstance(obj, mmap.mmap):
            return obj
        backing = getattr(obj, "_mmap", None)
        if isinstance(backing, mmap.mmap):
            return backing
        obj = getattr(obj, "base", None)
        seen += 1
    return None


def advise_random(obj: Any) -> bool:
    """Turn read-ahead off for a mapping that is read by scattered row picks.

    A fault's default 128 KiB read-ahead pulls in 31 pages nobody asked for.
    On a resident corpus that is invisible; on msmarco10M and stella, whose
    gather touches ~10k rows spread across 40/68 GiB, it is how a working set
    of a few tens of megabytes turns into tens of gigabytes of page cache.
    Returns False for anything that is not a mapping (the resident case).
    """
    backing = _mmap_backing(obj)
    if backing is None:
        return False
    try:
        backing.madvise(mmap.MADV_RANDOM)
    except (AttributeError, OSError, ValueError):
        return False
    return True


def _mapped_path(obj: Any) -> Any:
    """The file behind a numpy memmap/view, or None if it was not one."""
    seen = 0
    while obj is not None and seen < 8:
        name = getattr(obj, "filename", None)
        if name is not None:
            return name
        obj = getattr(obj, "base", None)
        seen += 1
    return None


def release_mapped_pages(obj: Any, path: Any = None) -> bool:
    """Give back the clean file pages behind an mmap'd array, when done with it.

    A corpus streamed through ``--mmap-base`` leaves every page it touched in
    the page cache, so a cell's search ends with the host still holding the
    build's pass over the packed copy plus the ratio probes' gathers -- tens of
    gigabytes that the next cell does not want to inherit.  ``MADV_DONTNEED``
    detaches those pages from this mapping; the advisory ``POSIX_FADV_DONTNEED``
    that follows then invalidates them, which it can only do once they are
    unmapped -- hence the order.  An mmap carries no fd of its own, so the file
    is reopened for that second call; pass ``path`` for a view whose producer
    (a plain ``mmap``) no longer remembers the name.  Returns False for a
    resident array, which is the no-op case.
    """
    backing = _mmap_backing(obj)
    if backing is None:
        return False
    try:
        backing.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError, ValueError):
        return False
    target = path if path is not None else _mapped_path(obj)
    if target is not None:
        try:
            fd = os.open(os.fspath(target), os.O_RDONLY)
        except OSError:
            return True
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        finally:
            os.close(fd)
    return True


def memmap_fvecs(path: Path) -> np.ndarray:
    """Read-only, row-strided view of an ``.fvecs`` file.

    Fancy indexing (``view[ids]``) reads only the requested records, so a
    corpus larger than RAM can still be scored for distance ratio and
    relative error; the pages it touches are file-backed and evictable, and
    peak RSS is the handful of rows a query touches.  The view is
    deliberately not contiguous --- the per-record dim header makes the row
    stride ``dim + 1`` floats --- so it must not be handed to a routine that
    wants a packed matrix (see :func:`ensure_packed_base` for the build).
    Read-ahead is off (see :func:`advise_random`): every access here is a
    scattered row pick, so pulling in the 31 neighbouring pages is pure cache
    growth.
    """
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    advise_random(raw)
    dim = int(np.frombuffer(raw[:4].tobytes(), dtype=np.int32)[0])
    record = 4 + 4 * dim
    if dim <= 0 or raw.size % record:
        raise ValueError(f"{path}: not a whole number of {dim}-d fvecs records")
    view = np.ndarray(shape=(raw.size // record, dim), dtype=np.float32,
                      buffer=raw, offset=4, strides=(4 * (dim + 1), 4))
    view.flags.writeable = False
    return view


def packed_base_path(ddir: Path, dataset: str, tag: str = "base") -> Path:
    return ddir / f"{dataset}_{tag}_packed.f32"


def ensure_packed_matrix(ddir: Path, dataset: str, tag: str, n: int, dim: int,
                         block_rows: int = 32768) -> np.memmap:
    """Contiguous float32 copy of ``<dataset>_<tag>.fvecs``, built in blocks.

    The index builders take a packed ``const float *``.  Handing them the
    strided fvecs view would make the binding's ``forcecast`` materialize the
    whole corpus on the heap, which is exactly what a large corpus cannot
    afford; a packed file costs ``n * dim * 4`` bytes of DISK and is then
    mapped, so the build streams through the page cache instead.  The
    conversion itself runs one block at a time, so it also stays inside a
    small constant footprint.  ``tag`` selects the fvecs stem: ``base`` for
    the raw corpus (E8Q), ``base_pca`` for the PCA copy (ME8Q, whose search
    and build both live in the PCA domain).
    """
    path = packed_base_path(ddir, dataset, tag)
    want = n * dim * 4
    if not (path.exists() and path.stat().st_size == want):
        sys.path.insert(0, str(REPO / "datasets"))
        from prepare import read_fvecs_block  # noqa: WPS433

        src = ddir / f"{dataset}_{tag}.fvecs"
        part = path.with_name(path.name + ".part")
        print(f"packing {src.name} -> {path.name} ({want / 2**30:.1f} GiB)", flush=True)
        with part.open("wb") as fh:
            for start in range(0, n, block_rows):
                take = min(block_rows, n - start)
                fh.write(np.ascontiguousarray(
                    read_fvecs_block(src, start, take, dim), dtype=np.float32
                ).tobytes())
        part.replace(path)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, dim))


def ensure_packed_base(ddir: Path, dataset: str, n: int, dim: int,
                       block_rows: int = 32768) -> np.memmap:
    """The raw-corpus special case of :func:`ensure_packed_matrix`."""
    return ensure_packed_matrix(ddir, dataset, "base", n, dim, block_rows)


def recall_at_k(gt: np.ndarray, ids: np.ndarray, k: int = 10) -> float:
    nq = min(gt.shape[0], ids.shape[0])
    hits = 0
    kk = min(k, ids.shape[1], gt.shape[1])
    for i in range(nq):
        hits += len(set(gt[i, :kk].tolist()) & set(ids[i, :kk].tolist()))
    return hits / (float(k) * nq)


def avg_distance_ratio(
    base: np.ndarray, queries: np.ndarray, gt: np.ndarray, ids: np.ndarray, k: int
) -> float:
    """Mean true L2² of returned k / NN L2², averaged over queries.

    Queries whose NN is an exact/near duplicate (d_nn numerically ~0 —
    e.g. arxiv-abstracts768 carries duplicated vectors) are skipped by a
    RELATIVE epsilon (d_nn < 1e-6 x median d_nn); an absolute cutoff lets
    float32 noise through and blows the ratio up to ~1e8."""
    nq = queries.shape[0]
    d_nns = []
    for qi in range(nq):
        nn = int(gt[qi, 0])
        d_nns.append(float(np.sum((queries[qi] - base[nn]) ** 2)))
    d_nns = np.asarray(d_nns)
    positive = d_nns[d_nns > 0.0]
    floor = max(1e-30, 1e-6 * float(np.median(positive))) if positive.size else 1e-30
    acc = 0.0
    used = 0
    for qi in range(nq):
        d_nn = d_nns[qi]
        if d_nn <= floor:
            continue
        row = np.asarray(ids[qi, :k])
        valid = row[(row >= 0) & (row < base.shape[0])]
        if valid.size == 0:
            continue
        diffs = queries[qi] - base[valid]
        mean_d = float(np.sum(diffs * diffs, axis=1).mean())
        acc += mean_d / d_nn
        used += 1
    return acc / used if used else 0.0


def saq_relerr(
    base: np.ndarray, queries: np.ndarray, ids: np.ndarray, dists: np.ndarray, k: int
) -> tuple[float, float]:
    """|D_hat - Delta| / Delta over the returned top-k (SAQ definition).

    `dists` are the estimated distances the method's own search path
    returned for `ids`; Delta is the exact squared L2 recomputed from the
    raw base. Returns (avg, max); (-1, -1) when nothing valid returned.
    """
    nq = min(ids.shape[0], dists.shape[0], queries.shape[0])
    acc = 0.0
    mx = 0.0
    cnt = 0
    for qi in range(nq):
        row = np.asarray(ids[qi, :k])
        valid = (row >= 0) & (row < base.shape[0])
        if not valid.any():
            continue
        vids = row[valid]
        dd = np.asarray(dists[qi, :k])[valid]
        diffs = queries[qi] - base[vids]
        delta = np.sum(diffs * diffs, axis=1)
        ok = (delta > 1e-12) & np.isfinite(dd)
        if not ok.any():
            continue
        err = np.abs(dd[ok] - delta[ok]) / delta[ok]
        acc += float(err.sum())
        mx = max(mx, float(err.max()))
        cnt += int(ok.sum())
    if cnt == 0:
        return -1.0, -1.0
    return acc / cnt, mx


def time_rounds(
    search_fn: Callable[[], Any],
    *,
    nq: int,
    warmup: int,
    repeats: int,
    min_ms: float,
    max_rounds: int = 64,
) -> tuple[np.ndarray, dict, Any]:
    """Return (ids, stats, extra) from last timed round. extra is search_fn's rest."""
    for _ in range(warmup):
        search_fn()
    rounds: list[float] = []
    ids = None
    extra = None
    while len(rounds) < repeats or sum(rounds) * 1000.0 < min_ms:
        if len(rounds) >= max_rounds:
            break
        t0 = time.perf_counter()
        result = search_fn()
        rounds.append(time.perf_counter() - t0)
        if isinstance(result, tuple):
            ids, extra = result[0], result[1] if len(result) > 1 else None
        else:
            ids, extra = result, None
    if ids is None:
        result = search_fn()
        if isinstance(result, tuple):
            ids, extra = result[0], result[1] if len(result) > 1 else None
        else:
            ids, extra = result, None
    qps_all = [nq / t for t in rounds]
    stats = {
        "qps": nq / float(np.median(rounds)),
        "qps_p25": float(np.percentile(qps_all, 25)),
        "qps_p75": float(np.percentile(qps_all, 75)),
        "nrounds": len(rounds),
    }
    return np.asarray(ids), stats, extra


def _json_float(x: float):
    if isinstance(x, (np.floating,)):
        x = float(x)
    if not isinstance(x, float):
        return x
    if not math.isfinite(x):
        return x
    return float(f"{x:.10f}")


def _sanitize_json(obj):
    """Native JSON with 10 decimal places on every float (README.md)."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _json_float(float(obj))
    return obj


def load_or_none(load_fn, path: Path | None):
    """Load a persisted index; on any failure unlink and return None (rebuild)."""
    if path is None or not path.exists():
        return None
    try:
        index = load_fn(str(path))
        print(f"loaded {path}", flush=True)
        return index
    except Exception as exc:  # noqa: BLE001 — any corrupt/incompatible file
        print(f"load failed ({path}): {exc}; rebuilding", flush=True)
        try:
            path.unlink()
        except OSError:
            pass
        return None


def write_ledger(
    dest: Path,
    *,
    method: str,
    dataset: str,
    n: int,
    dim: int,
    nq: int,
    bits: int | float,
    threads: int,
    seed: int,
    warmup: int,
    repeats: int,
    min_ms: float,
    rows: list[dict],
    k: int = 10,
    build_s: float = -1.0,
    mem_gb: float | None = None,
    bytes_per_vector: float | None = None,
    relerr_avg: float | None = None,
    relerr_p99: float | None = None,
    relerr_max: float | None = None,
    extra: dict | None = None,
) -> Path:
    payload: dict = {
        "dataset": dataset,
        "n": int(n),
        "dim": int(dim),
        "nq": int(nq),
        "bits": int(bits) if float(bits) == int(bits) else float(bits),
        "threads": int(threads),
        "k": int(k),
        "seed": int(seed),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "min_ms": float(min_ms),
        "trial": int(extra.get("trial", 1)) if extra else 1,
        f"{method}_build_s": float(build_s),
    }
    if mem_gb is not None:
        payload[f"{method}_mem_gb"] = float(mem_gb)
    if bytes_per_vector is not None:
        payload[f"{method}_bytes_per_vector"] = float(bytes_per_vector)
    if relerr_avg is not None:
        payload[f"{method}_relerr_avg"] = float(relerr_avg)
    if relerr_p99 is not None:
        payload[f"{method}_relerr_p99"] = float(relerr_p99)
    if relerr_max is not None:
        payload[f"{method}_relerr_max"] = float(relerr_max)
    if extra:
        payload.update(extra)
    payload["rows"] = rows
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace: a crash mid-write leaves dest untouched (or absent),
    # never a truncated JSON that resume would skip as "done".
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(_sanitize_json(payload), indent=2) + "\n")
    os.replace(tmp, dest)
    print(f"wrote {dest}", flush=True)
    return dest


def ledger_complete(path: Path) -> bool:
    """True iff `path` is parseable JSON with a non-empty `rows` list.

    Resume skips only complete ledgers. A truncated / empty / non-JSON file
    (crash mid-arm) is treated as missing so the arm is rerun. Ledgers of the
    prune-stats methods (e8q, me8q, rabitq) are additionally required to carry
    the survivor triple on every row: a ledger without them predates the
    stats patch and must be regenerated (README.md resume). A
    ``--build-only`` ledger is the exception: it holds one placeholder row and
    no search results at all, and is complete once ``<method>_build_s`` exists.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    rows = data.get("rows") if isinstance(data, dict) else None
    if not (isinstance(rows, list) and rows):
        return False
    if isinstance(data.get("method"), str) and "metric" in data:
        # A distance-ratio side-car (experiments/adr_fill.py): one column of
        # one arm plus the recall it was scored from, deliberately not a
        # sweep ledger, so the survivor triple below does not apply to it.
        return True
    parsed = parse_stem(path.stem)
    if parsed is not None and parsed[1] in ("e8q", "me8q", "rabitq"):
        if all("survivor_rate" in r for r in rows if isinstance(r, dict)):
            return True
        return _build_only_complete(data, rows, parsed[1])
    return True


def _build_only_complete(data: dict, rows: list, method: str) -> bool:
    """True for a finished ``--build-only`` ledger (construction sweeps).

    The placeholder row carries ``nprobe = -1`` and no recall/survivor fields,
    so the survivor gate above cannot recognise it; the build timing is the
    completion signal instead. Without this a resumed construction sweep
    rebuilds every cell it already measured.
    """
    if not all(isinstance(r, dict) and r.get("nprobe", 0) < 0 for r in rows):
        return False
    try:
        return float(data.get(f"{method}_build_s") or -1.0) > 0.0
    except (TypeError, ValueError):
        return False


def ledger_dest(out: Path, dataset: str, bits: int | float, threads: int, method: str,
                trial: int = 1) -> Path:
    return out / ledger_filename(dataset, bits, threads, method, trial)


def make_row(
    method: str,
    nprobe: int,
    recall: float,
    stats: dict,
    ratio: float | None,
    extra: dict | None = None,
) -> dict:
    row = {
        "nprobe": int(nprobe),
        f"{method}_recall": float(recall),
        f"{method}_qps": float(stats["qps"]),
        f"{method}_qps_p25": float(stats["qps_p25"]),
        f"{method}_qps_p75": float(stats["qps_p75"]),
        f"{method}_nrounds": int(stats["nrounds"]),
    }
    if ratio is not None:
        row[f"{method}_ratio"] = float(ratio)
    if extra:
        row.update(extra)
    return row


def banner(dataset: str, n: int, dim: int, nq: int, bits, k: int, threads: int, seed: int) -> None:
    print(
        f"dataset={dataset} n={n} dim={dim} nq={nq} bits={bits} k={k} threads={threads}",
        flush=True,
    )
    print(f"seed={seed}", flush=True)


def run_nprobe_grid(
    *,
    method: str,
    grid: list[int],
    search_fn: Callable[[int], Any],
    queries: np.ndarray,
    base: np.ndarray,
    gt: np.ndarray,
    k: int,
    warmup: int,
    repeats: int,
    min_ms: float,
    stats_fn: Callable[[int], dict | None] | None = None,
    relerr_probe: int = RELERR_PROBE,
    adr: bool = False,
) -> tuple[list[dict], float | None, float | None]:
    """Run the nprobe grid; return (rows, relerr_avg, relerr_max).

    The timed pass is stats-free so instrumentation cannot perturb QPS.
    When `stats_fn` is given, one extra untimed pass per nprobe collects
    counters. search_fn may return ids or (ids, dists); the estimated
    distances of the last timed round at nprobe == relerr_probe feed the
    SAQ-definition relative error (README.md).
    """
    nq = queries.shape[0]
    rows = []
    relerr: tuple[float | None, float | None] = (None, None)
    relerr_p99: float | None = None
    print(f"{'nprobe':>8} {method + '_recall':>12} {method + '_qps':>12}  (qps IQR nrounds in JSON)")
    for np_ in grid:
        def _one(nprobe=np_):
            return search_fn(nprobe)

        ids, stats, extra = time_rounds(
            _one, nq=nq, warmup=warmup, repeats=repeats, min_ms=min_ms
        )
        recall = recall_at_k(gt, ids, k)
        ratio = avg_distance_ratio(base, queries, gt, ids, k)
        dists = np.asarray(extra) if extra is not None else None
        cell_rel = (None, None)
        cell_p99 = None
        if dists is not None and getattr(dists, "ndim", 0) == 2:
            cell_rel = saq_relerr(base, queries, ids, dists, k)
            _, cell_p99, _ = relerr_full(base, queries, ids, dists, k)
            if np_ == relerr_probe:
                relerr = cell_rel
                relerr_p99 = cell_p99
                print(
                    f"  relerr@np{np_}: avg={relerr[0]:.10f} p99={cell_p99:.10f} "
                    f"max={relerr[1]:.10f}",
                    flush=True,
                )
        row_extra: dict = {}
        if cell_rel[0] is not None:
            row_extra[f"{method}_relerr_avg"] = float(cell_rel[0])
            row_extra[f"{method}_relerr_max"] = float(cell_rel[1])
            if cell_p99 is not None and cell_p99 >= 0:
                row_extra[f"{method}_relerr_p99"] = float(cell_p99)
        if stats_fn is not None:
            estats = stats_fn(np_)
            if isinstance(estats, dict):
                n = float(nq)
                row_extra["survivor_rate"] = float(estats.get("survivor_rate", 0.0))
                row_extra[f"{method}_candidates"] = float(estats.get("candidates", 0.0)) / n
                row_extra[f"{method}_survivors"] = float(estats.get("survivors", 0.0)) / n
        if adr:
            # Sum-ratio distance ratio on the ids this row was scored from,
            # so no second search is needed and the ratio always belongs to
            # the operating point the recall above describes (README.md).
            row_extra[f"{method}_adr"] = float(
                adr_sum_ratio(base, queries, gt, ids, k))
        rows.append(make_row(method, np_, recall, stats, ratio, row_extra or None))
        print(f"{np_:>8} {recall:>12.6f} {stats['qps']:>12.6f}", flush=True)
    if relerr[0] is None:
        # 200 missing from the grid, or this method returned no dists: one
        # untimed probe so the header field is never silently empty.
        result = search_fn(relerr_probe)
        if isinstance(result, tuple) and len(result) >= 2:
            ids_p, dists_p = np.asarray(result[0]), np.asarray(result[1])
            if dists_p.ndim == 2:
                relerr = saq_relerr(base, queries, ids_p, dists_p, k)
                _, relerr_p99, _ = relerr_full(base, queries, ids_p, dists_p, k)
                print(
                    f"  relerr@np{relerr_probe} (untimed): avg={relerr[0]:.10f} "
                    f"max={relerr[1]:.10f}",
                    flush=True,
                )
    return rows, relerr[0], relerr[1], relerr_p99


def main(method: str, argv: list[str] | None = None) -> int:
    if method == "e8q":
        from run_e8q import run as run_e8q  # noqa: WPS433

        return run_e8q(argv)
    if method == "rabitq":
        from run_rabitq import run as run_rabitq  # noqa: WPS433

        return run_rabitq(argv)
    if method == "me8q":
        from run_me8q import run as run_me8q  # noqa: WPS433

        return run_me8q(argv)
    if method == "pq":
        from run_pq import run as run_pq  # noqa: WPS433

        return run_pq(argv)
    if method in ("saq", "caq"):
        from run_saq import run as run_saq  # noqa: WPS433

        return run_saq(method, argv)
    raise SystemExit(f"unknown method {method}")


# ---------------------------------------------------------------------------
# Extra statistic helpers (README.md).
# ---------------------------------------------------------------------------

def adr_sum_ratio(
    base: np.ndarray, queries: np.ndarray, gt: np.ndarray, ids: np.ndarray, k: int
) -> float:
    """ADR@k, sum-ratio form: mean over queries of
    sum_j d(q, approx_j) / sum_j d(q, exact_j), d = raw-space L2.

    Queries whose exact NN is a numerical duplicate (d_nn below a
    relative epsilon of the median) are excluded — the arxiv-duplicate
    guard (README.md)."""
    nq = min(ids.shape[0], queries.shape[0])
    acc = 0.0
    used = 0
    d_nn_all = np.empty(nq, dtype=np.float64)
    for qi in range(nq):
        nn = int(gt[qi, 0])
        d = queries[qi] - base[nn]
        d_nn_all[qi] = float(np.sum(d * d))
    positive = d_nn_all[d_nn_all > 0.0]
    floor = max(1e-30, 1e-6 * float(np.median(positive))) if positive.size else 1e-30
    for qi in range(nq):
        if d_nn_all[qi] <= floor:
            continue
        row = np.asarray(ids[qi, :k])
        valid = row[(row >= 0) & (row < base.shape[0])]
        if valid.size == 0:
            continue
        diffs = queries[qi] - base[valid]
        approx = float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))
        # exact k distances (recompute the GT rows; gt holds top-100 ids)
        g = np.asarray(gt[qi, :k])
        gd = queries[qi] - base[g]
        exact = float(np.sum(np.sqrt(np.sum(gd * gd, axis=1))))
        if exact > 0.0:
            acc += approx / exact
            used += 1
    return acc / used if used else 0.0


def relerr_full(
    base: np.ndarray, queries: np.ndarray, ids: np.ndarray, dists: np.ndarray, k: int
) -> tuple[float, float, float]:
    """Vectorized |D_hat - Delta| / Delta over the returned top-k.

    Returns (avg, p99, max); (-1, -1, -1) when nothing valid returned.
    p99 replaces max as the plotted tail statistic (README.md)."""
    nq = min(ids.shape[0], dists.shape[0], queries.shape[0])
    errs = []
    for qi in range(nq):
        row = np.asarray(ids[qi, :k])
        valid = (row >= 0) & (row < base.shape[0])
        if not valid.any():
            continue
        vids = row[valid]
        dd = np.asarray(dists[qi, :k])[valid]
        diffs = queries[qi] - base[vids]
        delta = np.sum(diffs * diffs, axis=1)
        ok = (delta > 1e-12) & np.isfinite(dd)
        if not ok.any():
            continue
        errs.append(np.abs(dd[ok] - delta[ok]) / delta[ok])
    if not errs:
        return -1.0, -1.0, -1.0
    e = np.concatenate(errs)
    return (float(e.mean()), float(np.percentile(e, 99)), float(e.max()))


def peak_rss_gb() -> float:
    """VmHWM (peak resident set, GB) of this process — read after build to
    capture the construction peak (README.md)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return -1.0


def latency_quantiles(us: np.ndarray) -> dict:
    """p50/p90/p95/p99 of per-query latencies in microseconds (§6.2)."""
    if us.size == 0:
        return {q: -1.0 for q in ("p50", "p90", "p95", "p99")}
    return {q: float(np.percentile(us, int(q[1:]))) for q in
            ("p50", "p90", "p95", "p99")}


def parse_perf_stat(path) -> dict:
    """Parse a `perf stat` text report into the Table-7 metrics (§6.1)."""
    out: dict[str, float] = {}
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        m = re.match(r"\s*([\d,.]+)\s+([\w\-]+)", line)
        if not m:
            continue
        val = float(m.group(1).replace(",", ""))
        key = m.group(2)
        if key in ("instructions", "cycles", "L1-dcache-load-misses",
                   "LLC-load-misses", "branch-misses"):
            out[key] = val
    if "cycles" in out and out["cycles"] > 0:
        out["ipc"] = out.get("instructions", 0.0) / out["cycles"]
    return out
