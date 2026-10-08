#!/usr/bin/env python3
"""PQ baseline at a matched bit budget (paper ANNS panels).

Product quantization under the paper's fairness contract: the SAME shared
IVF k-means centroids and list assignments produced by datasets/prepare.py
(a faiss IndexFlatL2 quantizer preloaded with those centroids; PQ trains
only its sub-codebooks), squared L2, k=10, no rerank.

Settings follow the SAQ paper's baseline section: faiss IndexIVFPQ with
8-bit sub-quantizers (SAQ reports nbits=8, citing that it consistently
beats nbits=4) and the sub-space count M = b*D/8 varied to hit the target
budget, b*dim bits per vector — D must be divisible by M, so budgets PQ
cannot express are skipped with a notice. RaBitQ's own PQ arm additionally
reranks raw vectors; our no-rerank protocol (shared by every method here)
follows SAQ's ANNS evaluation instead.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import faiss
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

from protocol import (
    adr_sum_ratio,  # noqa: E402
    add_common_args,
    banner,
    corpus_shape,
    ledger_dest,
    load_corpus,
    load_or_none,
    mem_estimate,
    parse_grid,
    peak_rss_gb,
    ram_guard,
    run_nprobe_grid,
    write_ledger,
)


def run(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--train-size", type=int, default=256_000)
    args = ap.parse_args(argv)

    bits = int(args.bits)
    n, dim, nq_all = corpus_shape(args.data_root, args.dataset)
    base_bytes = 4.0 * n * dim
    if not ram_guard(
        base_bytes + mem_estimate("pq", n, dim, bits),
        f"{args.dataset} pq b{bits}",
    ):
        return 0
    corp = load_corpus(args.data_root, args.dataset, args.nq, load_base=True)
    n, dim, nq = corp["n"], corp["dim"], corp["nq"]
    queries, gt, cent, cids, base = (
        corp["queries"], corp["gt"], corp["centroids"], corp["cids"], corp["base"]
    )
    num_clusters = cent.shape[0]
    assert cids.shape[0] == n

    faiss.omp_set_num_threads(args.threads)
    quantizer = faiss.IndexFlatL2(dim)
    quantizer.add(np.ascontiguousarray(cent))
    # SAQ's baseline setting: 8-bit sub-quantizers, M = b*D/8.
    nbits = 8
    m = bits * dim // nbits
    if dim % m != 0:
        print(
            f"pq: b={bits} not expressible with {nbits}-bit sub-quantizers "
            f"on dim={dim} (M={m} must divide D); skipping",
            flush=True,
        )
        return 0
    index = load_or_none(faiss.read_index, args.index_path)
    build_s = -1.0
    if index is None:
        index = faiss.IndexIVFPQ(quantizer, dim, num_clusters, m, nbits)
        train = np.ascontiguousarray(base[: min(args.train_size, n)])
        t0 = time.time()
        index.train(train)
        build_s = time.time() - t0
        order = np.argsort(cids, kind="stable")
        index.add_with_ids(np.ascontiguousarray(base[order]), order.astype(np.int64))
        if args.index_path is not None:
            args.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(args.index_path))
    banner(args.dataset, n, dim, nq, bits, args.k, args.threads, args.seed)
    # Total at-rest size: codes + codebooks + per-vector ids + coarse
    # quantizer (README.md).
    mem_bytes = index.code_size * n + m * 256 * 4 + n * 8 + 4 * num_clusters * dim
    print(
        f"pq: nbits={nbits} m={m} train={build_s:.1f}s "
        f"mem_gb={mem_bytes / 1e9:.4f} bytes_per_vector={mem_bytes / n:.2f}",
        flush=True,
    )

    def search_fn(nprobe: int):
        index.nprobe = nprobe
        dists, ids = index.search(queries, args.k)
        return np.asarray(ids), np.asarray(dists)

    rows, rel_avg, rel_max, rel_p99 = run_nprobe_grid(
        method="pq",
        grid=parse_grid(args.grid),
        search_fn=search_fn,
        queries=queries,
        base=base,
        gt=gt,
        k=args.k,
        warmup=args.warmup,
        repeats=args.repeats,
        min_ms=args.min_ms,
    )
    # ADR@k at the Table-5 depths (README.md): raw-space
    # sum-ratio form, the same depths e8q/rabitq use. faiss returns ADC
    # estimates as the distances; only the ids feed ADR.
    if getattr(args, "adr", False):
        for r in rows:
            if r["nprobe"] in (16, 64, 128, 256):
                index.nprobe = r["nprobe"]
                _, ids_a = index.search(queries, args.k)
                r["pq_adr"] = adr_sum_ratio(base, queries, gt, ids_a, args.k)

    dest = ledger_dest(args.out, args.dataset, bits, args.threads, "pq",
                       getattr(args, "trial", 1))
    write_ledger(
        dest,
        method="pq",
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
        extra={"pq_nbits": nbits, "pq_m": m,
               "pq_bpv_code": float(index.code_size),
               "pq_bpv_overhead": mem_bytes / n - index.code_size,
               "pq_bits_per_dim": mem_bytes * 8.0 / n / dim,
               "pq_mem_total_bytes": mem_bytes,
               "pq_build_peak_gb": peak_rss_gb(),
               "pq_build_kvps": (n / build_s / 1000.0) if build_s > 0 else -1.0},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
