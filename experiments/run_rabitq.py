#!/usr/bin/env python3
"""RaBitQ arm under the paper's measurement protocol.

Uses the vendored RaBitQ-Library Python IVF (same shared centroids / lists /
rotator seed as every other arm). Search is single-query full-dim FP32 L2,
squared L2, no rerank.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "baselines" / "RaBitQ-Library" / "python_bindings"))

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


def _import_ivf():
    try:
        from rabitqlib import IvfIndex  # type: ignore  # noqa: WPS433

        return IvfIndex
    except ImportError:
        from _rabitqlib import IvfIndex  # type: ignore  # noqa: WPS433

        return IvfIndex


def run(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args(argv)
    bits = int(args.bits)
    n, dim, nq_all = corpus_shape(args.data_root, args.dataset)
    # --mmap-base keeps the corpus on disk (see run_e8q): resident cost is the
    # index plus the rows a query touches, so corpora past this host's RAM run.
    disk_base = bool(getattr(args, "mmap_base", False))
    resident_bytes = 0.0 if disk_base else 4.0 * n * dim
    if not ram_guard(
        resident_bytes + mem_estimate("rabitq", n, dim, bits),
        f"{args.dataset} rabitq b{bits}",
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
    num_clusters = cent.shape[0]
    IvfIndex = _import_ivf()

    os.environ["RABITQ_ROTATOR_SEED"] = str(args.seed)
    build_s = -1.0
    build_threads = 0
    index = load_or_none(IvfIndex.load, args.index_path)
    if index is None:
        index = IvfIndex(dim=dim, max_elements=n, num_clusters=num_clusters, nbits=bits)
        # Construction width follows --threads, the width every other arm of
        # the campaign builds with.  The fallback used to be a hardcoded 20,
        # which oversubscribed a sweep pinned to 12 CPUs and handed RaBitQ a
        # build budget no other method got.  The env var still overrides, for
        # a host whose build wants a different width.  OMP_NUM_THREADS cannot
        # do this job: IVF::construct carries an explicit num_threads clause
        # and clamps against hardware_concurrency(), not the environment.
        build_threads = int(
            os.environ.get("RABITQ_CONSTRUCT_THREADS", str(args.threads))
        )
        t0 = time.time()
        index.build(build_base, cent, cids, num_threads=build_threads)
        build_s = time.time() - t0
        if args.index_path is not None:
            args.index_path.parent.mkdir(parents=True, exist_ok=True)
            index.save(str(args.index_path))
    if disk_base:
        # The build just streamed the whole packed corpus through the page
        # cache; the search that follows reads the index and a few raw rows
        # per query instead, so hand those pages back before the sweep.
        release_mapped_pages(build_base)

    banner(args.dataset, n, dim, nq, bits, args.k, args.threads, args.seed)
    mem_bytes = index.memory_bytes
    built = f", {build_threads} threads" if build_threads else ""
    print(
        f"rabitq_build_s={build_s:.1f} (rotator seeded{built}) "
        f"mem_gb={mem_bytes / 1e9:.4f} bytes_per_vector={mem_bytes / n:.2f}"
    )

    def search_fn(nprobe: int):
        return index.search(
            queries, args.k, nprobe, high_accuracy=True, num_threads=args.threads
        )

    def stats_fn(nprobe: int):
        return index.search(
            queries, args.k, nprobe, high_accuracy=True,
            num_threads=args.threads, return_stats=True,
        )[2]

    if getattr(args, "build_only", False):
        # Construction-only run (--build-only): the nprobe sweep is what the
        # construction tables do not need, and skipping it takes the warm-up
        # rounds and the whole grid out of the cell's wall clock.  One
        # placeholder row keeps the artifact a parseable ledger.
        rows, rel_avg, rel_max, rel_p99 = [{"nprobe": -1}], None, None, None
    else:
        rows, rel_avg, rel_max, rel_p99 = run_nprobe_grid(
            method="rabitq",
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

    # Per-query latency capture (§6.2): single-query timed loop + .npy
    # sidecar; quantiles land on every row (single-cell grids in practice).
    if getattr(args, "latency_nq", 0) > 0:
        import numpy as np

        nlat = min(args.latency_nq, queries.shape[0])
        nprobe_lat = rows[-1]["nprobe"] if rows else 64
        lats = np.empty(nlat, dtype=np.float64)
        for qi in range(nlat):
            t0 = time.perf_counter()
            index.search(queries[qi : qi + 1], args.k, nprobe_lat,
                         high_accuracy=True, num_threads=1)
            lats[qi] = (time.perf_counter() - t0) * 1e6
        sidecar = ledger_dest(args.out, args.dataset, bits, args.threads, "rabitq",
                              getattr(args, "trial", 1)).with_suffix(".latency.npy")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.save(sidecar, lats)
        lat_q = latency_quantiles(lats)
        for r in rows:
            for qname, qval in lat_q.items():
                r[f"rabitq_lat_{qname}"] = qval

    write_ledger(
        ledger_dest(args.out, args.dataset, bits, args.threads, "rabitq",
                    getattr(args, "trial", 1)),
        method="rabitq",
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
        extra={"rabitq_bpv_code": dim * bits / 8.0,
               "rabitq_bpv_factor": mem_bytes / n - dim * bits / 8.0,
               "rabitq_bits_per_dim": mem_bytes * 8.0 / n / dim,
               "rabitq_mem_total_bytes": mem_bytes,
               "rabitq_build_peak_gb": peak_rss_gb(),
               "rabitq_build_threads": build_threads,
               "rabitq_build_kvps": (n / build_s / 1000.0) if build_s > 0 else -1.0},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
