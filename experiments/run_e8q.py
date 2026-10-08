#!/usr/bin/env python3
"""E8Q arm under the paper's measurement protocol."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

from e8qlib import E8qIndex  # noqa: E402
from protocol import (  # noqa: E402
    add_common_args,
    banner,
    corpus_shape,
    ensure_packed_base,
    latency_quantiles,
    ledger_dest,
    load_corpus,
    load_or_none,
    mem_estimate,
    memmap_fvecs,
    parse_grid,
    peak_rss_gb,
    ram_guard,
    release_mapped_pages,
    run_nprobe_grid,
    write_ledger,
)


def run(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--prune-alpha", type=float, default=1.0,
                    help="union-bound per-query prune-loss budget (paper default 1.0)")
    args = ap.parse_args(argv)
    bits = int(args.bits)
    n, dim, nq_all = corpus_shape(args.data_root, args.dataset)
    # --mmap-base keeps the corpus on disk: the arm then costs the index plus
    # a few rows per query, not a second copy of the raw vectors, which is
    # what lets a 39 GiB (msmarco10M) or 69 GiB (stella) corpus run here.
    disk_base = bool(getattr(args, "mmap_base", False))
    resident_bytes = 0.0 if disk_base else 4.0 * n * dim
    if not ram_guard(
        resident_bytes + mem_estimate("e8q", n, dim, bits),
        f"{args.dataset} e8q b{bits}",
    ):
        return 0
    corp = load_corpus(args.data_root, args.dataset, args.nq,
                       load_base=not disk_base)
    n, dim, nq = corp["n"], corp["dim"], corp["nq"]
    queries, gt, cent, cids = (
        corp["queries"], corp["gt"], corp["centroids"], corp["cids"]
    )
    if disk_base:
        base = memmap_fvecs(corp["ddir"] / f"{args.dataset}_base.fvecs")
        build_base = ensure_packed_base(corp["ddir"], args.dataset, n, dim)
    else:
        base = build_base = corp["base"]

    build_s = -1.0
    index = load_or_none(E8qIndex.load, args.index_path)
    if index is None:
        index = E8qIndex(dim=dim, nbits=bits, seed=args.seed)
        t0 = time.time()
        index.build(
            build_base, cent, cids, num_threads=max(1, args.threads),
            prune_alpha=args.prune_alpha,
        )
        build_s = time.time() - t0
        if args.index_path is not None:
            args.index_path.parent.mkdir(parents=True, exist_ok=True)
            index.save(str(args.index_path))
    if disk_base:
        # The build just streamed the whole packed corpus through the page
        # cache; the search that follows reads the index and a few raw rows
        # per query instead, so hand those pages back before the sweep.
        release_mapped_pages(build_base)
    if args.no_prune:
        index.set_disable_prune(True)
    if args.prune_alpha > 0:
        index.set_prune_alpha(args.prune_alpha)

    banner(args.dataset, n, dim, nq, bits, args.k, args.threads, args.seed)
    print(f"prune_alpha={args.prune_alpha}")
    mem_bytes = index.memory_bytes
    print(
        f"e8q_build_s={build_s:.1f} mem_gb={mem_bytes / 1e9:.4f} "
        f"bytes_per_vector={mem_bytes / n:.2f}"
    )

    def search_fn(nprobe: int):
        return index.search(queries, args.k, nprobe, num_threads=args.threads)

    def stats_fn(nprobe: int):
        return index.search(
            queries, args.k, nprobe, num_threads=args.threads, return_stats=True
        )[2]

    if getattr(args, "build_only", False):
        # Construction-only run (--build-only): the nprobe sweep is what the
        # construction tables do not need, and skipping it takes the warm-up
        # rounds and the whole grid out of the cell's wall clock.  One
        # placeholder row keeps the artifact a parseable ledger.
        rows, rel_avg, rel_max, rel_p99 = [{"nprobe": -1}], None, None, None
    else:
        rows, rel_avg, rel_max, rel_p99 = run_nprobe_grid(
            method="e8q",
            grid=parse_grid(args.grid),
            search_fn=search_fn,
            queries=queries,
            base=base,
            gt=gt,
            k=args.k,
            warmup=args.warmup,
            repeats=args.repeats,
            min_ms=args.min_ms,
            stats_fn=stats_fn,
            adr=getattr(args, "adr", False),
        )
    if disk_base:
        # Same for the strided raw-space view the ratio probes scored from:
        # the sweep is over, so its pages are pure cache on a host that is
        # already short of RAM.
        release_mapped_pages(base, corp["ddir"] / f"{args.dataset}_base.fvecs")

    # Per-phase us/query rows (§6.3): route = prepare+rotate+LUT, scan covers
    # coarse+bound, refine is the exact re-score — the three spans SearchStats
    # actually times. Absent spans (negative sentinels) are dropped, not zeroed.
    if getattr(args, "phases", False):
        for r in rows:
            st = stats_fn(r["nprobe"])
            route = float(st.get("route_us", -1.0))
            rotate = float(st.get("rotate_us", -1.0))
            route_sum = (route if route >= 0 else 0.0) + (rotate if rotate >= 0 else 0.0)
            if route >= 0 or rotate >= 0:
                r["e8q_us_route"] = route_sum
            scan = float(st.get("scan_us", -1.0))
            refine = float(st.get("refine_us", -1.0))
            if scan >= 0:
                r["e8q_us_scan"] = scan
            if refine >= 0:
                r["e8q_us_refine"] = refine

    # Per-query latency capture (§6.2): single-query timed loop + .npy
    # sidecar; quantiles land on every row (single-cell grids in practice).
    if getattr(args, "latency_nq", 0) > 0:
        nlat = min(args.latency_nq, queries.shape[0])
        nprobe_lat = rows[-1]["nprobe"] if rows else 64
        lats = np.empty(nlat, dtype=np.float64)
        for qi in range(nlat):
            t0 = time.perf_counter()
            index.search(queries[qi : qi + 1], args.k, nprobe_lat,
                         num_threads=1)
            lats[qi] = (time.perf_counter() - t0) * 1e6
        sidecar = ledger_dest(args.out, args.dataset, bits, args.threads, "e8q",
                              getattr(args, "trial", 1)).with_suffix(".latency.npy")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.save(sidecar, lats)
        lat_q = latency_quantiles(lats)
        for r in rows:
            for qname, qval in lat_q.items():
                r[f"e8q_lat_{qname}"] = qval

    write_ledger(
        ledger_dest(args.out, args.dataset, bits, args.threads, "e8q",
                    getattr(args, "trial", 1)),
        method="e8q",
        dataset=args.dataset,
        n=n,
        dim=dim,
        nq=nq,
        bits=bits,
        threads=args.threads,
        seed=args.seed,
        warmup=args.warmup,
        repeats=args.repeats,
        min_ms=args.min_ms,
        rows=rows,
        k=args.k,
        build_s=build_s,
        mem_gb=mem_bytes / 1e9,
        bytes_per_vector=mem_bytes / n,
        relerr_avg=rel_avg,
        relerr_p99=rel_p99,
        relerr_max=rel_max,
        extra={"e8q_prune_alpha": args.prune_alpha,
               "e8q_no_prune": bool(args.no_prune),
               "e8q_projection": os.environ.get("E8Q_PROJECTION", "zpr"),
               "e8q_bpv_code": dim * bits / 8.0,
               "e8q_bpv_factor": mem_bytes / n - dim * bits / 8.0,
               "e8q_bits_per_dim": mem_bytes * 8.0 / n / dim,
               "e8q_mem_total_bytes": mem_bytes,
               "e8q_build_peak_gb": peak_rss_gb(),
               "e8q_build_kvps": (n / build_s / 1000.0) if build_s > 0 else -1.0},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
