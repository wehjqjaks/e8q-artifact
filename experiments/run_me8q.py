#!/usr/bin/env python3
"""ME8Q arm under the paper's measurement protocol.

SAQ mixed-precision pipeline (PCA + DP allocation) with the E8Q codec.
Searches in the full-rank PCA domain; ground-truth ids stay valid.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

from e8qlib import Me8qIndex  # noqa: E402
from protocol import (
    FvecsRowReader,  # noqa: E402
    adr_sum_ratio,  # noqa: E402
    add_common_args,
    advise_random,
    banner,
    corpus_shape,
    ensure_packed_matrix,
    ledger_dest,
    load_corpus,
    load_or_none,
    mem_estimate,
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
    ap.add_argument("--no-drop-energy", action="store_true",
                    help="ablation: fold no residual energy of dropped 0-bit dims")
    ap.add_argument("--match-bpv", type=float, default=0.0,
                    help="byte-matching target: total B/vec of the baseline "
                         "(saq_bytes_per_vector). The DP budget is padded so "
                         "me8q spends the same actual memory; 0 = nominal SAQ "
                         "budget (paper B)")
    ap.add_argument("--match-total-budget", action="store_true",
                    help="true-bit-budget protocol: total memory must equal "
                         "b*dim/8 per vector (code + metadata). The DP budget "
                         "is padded — negatively if needed — to hit it; the "
                         "saved metadata converts into code bits")
    args = ap.parse_args(argv)
    bits = float(args.bits)
    n, dim, nq_all = corpus_shape(args.data_root, args.dataset)
    base_bytes = 4.0 * n * dim
    # --mmap-base streams the PCA copy from disk instead of holding it: the arm
    # then costs the index plus the rows a query touches, not a resident PCA
    # corpus on top of the raw-space scoring reads, which is what lets a 39 GiB
    # (msmarco10M) or 69 GiB (stella) corpus run here.
    disk_base = bool(getattr(args, "mmap_base", False))
    resident_bytes = 0.0 if disk_base else 2.0 * base_bytes
    if not ram_guard(
        resident_bytes + mem_estimate("me8q", n, dim, bits),
        f"{args.dataset} me8q b{bits}",
    ):
        return 0
    corp = load_corpus(
        args.data_root, args.dataset, args.nq, load_base=False, load_pca=True,
        load_pca_base=not disk_base,
    )
    n, dim, nq = corp["n"], corp["dim"], corp["nq"]
    queries, gt = corp["queries"], corp["gt"]
    q_pca, c_pca, base_pca, cids, var = (
        corp["queries_pca"], corp["centroids_pca"], corp["base_pca"],
        corp["cids"], corp["vars"],
    )
    if q_pca is None:
        print("me8q: PCA files missing; run datasets/prepare.sh", flush=True)
        return 1
    if base_pca is None:
        if not disk_base:
            print("me8q: PCA files missing; run datasets/prepare.sh", flush=True)
            return 1
        # Contiguous on-disk copy of the PCA matrix (the strided fvecs view
        # would force the binding to materialize the corpus on the heap).
        base_pca = ensure_packed_matrix(
            corp["ddir"], args.dataset, "base_pca", n, dim
        )

    build_s = -1.0
    # The last build call is the deployed index when the true-bit-budget
    # protocol asks for a second pass; build_s stays the fit total
    # (README.md), so a construction table can report either.
    last_build_s = -1.0
    unpadded_bpv = -1.0
    index = load_or_none(Me8qIndex.load, args.index_path)
    if index is None:
        index = Me8qIndex(dim=dim, avg_bits=bits, seed=args.seed)
        t0 = time.time()
        index.build(
            base_pca, c_pca, cids, num_threads=max(1, args.threads),
            variance=var, prune_alpha=args.prune_alpha,
            drop_energy=not args.no_drop_energy,
        )
        build_s = last_build_s = time.time() - t0
        match_target = 0.0
        if args.match_bpv > 0:
            match_target = args.match_bpv
        elif args.match_total_budget:
            match_target = bits * dim / 8.0
        if match_target > 0:
            # True-bit-budget protocol: charge me8q's own per-vector overhead
            # against the total so code + metadata == b*dim bits exactly.
            # A negative pad shrinks the code; the DP stays in charge. The
            # pad is computed from the nominal build's measured overhead, so
            # verify the padded build lands under budget and correct if a
            # different plan shifted the overhead (a couple of rebuilds at
            # most; the plan ladder shrinks monotonically with the pad).
            unpadded_bpv = index.memory_bytes / n
            final_bpv = unpadded_bpv
            extra_bits = (match_target - unpadded_bpv) * 8.0
            attempts = 1
            if abs(extra_bits) >= 8.0:
                for _attempt in range(3):
                    index = Me8qIndex(dim=dim, avg_bits=bits, seed=args.seed)
                    t0 = time.time()
                    index.build(
                        base_pca, c_pca, cids, num_threads=max(1, args.threads),
                        variance=var, prune_alpha=args.prune_alpha,
                        drop_energy=not args.no_drop_energy,
                        extra_budget_bits=extra_bits,
                    )
                    last_build_s = time.time() - t0
                    build_s += last_build_s
                    attempts += 1
                    final_bpv = index.memory_bytes / n
                    if not any(b > 0 for _, b in index.plan):
                        # The budget cannot buy a single coded run: the DP
                        # produced no allocation plan at this b — record and
                        # skip (marker file, no ledger).
                        import json as _json
                        dest = ledger_dest(args.out, args.dataset, bits,
                                           args.threads, "me8q",
                                           getattr(args, "trial", 1))
                        dest.with_suffix(".skipped").write_text(_json.dumps({
                            "reason": "budget_too_low",
                            "b": bits,
                            "budget_bpv": match_target,
                            "unpadded_bpv": unpadded_bpv,
                            "pad_bits": extra_bits,
                            "builds": attempts,
                        }))
                        print(f"me8q: budget too low at b={bits} "
                              f"(target {match_target:.1f} B/vec after metadata "
                              f"~{(unpadded_bpv * 8.0 - bits * dim):.0f} bits) "
                              "— skipped, no plan", flush=True)
                        return 0
                    if final_bpv <= match_target + 0.5:
                        break
                    extra_bits -= (final_bpv - match_target) * 8.0
            print(f"me8q total-budget: target={match_target:.2f} "
                  f"unpadded={unpadded_bpv:.2f} pad_bits={extra_bits:.0f} "
                  f"final={final_bpv:.2f} B/vec builds={attempts}")
        if args.index_path is not None:
            args.index_path.parent.mkdir(parents=True, exist_ok=True)
            index.save(str(args.index_path))
    # The raw-space base is never resident: ADR@k gathers the few vectors
    # it touches from the fvecs on demand (README.md needs raw-space
    # sums only for the returned/GT ids). Keeping the copy resident is what
    # OOM-killed stella — raw + PCA + index exceeded host RAM.

    banner(args.dataset, n, dim, nq, bits, args.k, args.threads, args.seed)
    print(f"prune_alpha={args.prune_alpha}")
    mem_bytes = index.memory_bytes
    print(
        f"me8q_build_s={build_s:.1f} last={last_build_s:.1f} "
        f"mem_gb={mem_bytes / 1e9:.4f} "
        f"bytes_per_vector={mem_bytes / n:.2f}"
    )

    def search_fn(nprobe: int):
        return index.search(q_pca, args.k, nprobe, num_threads=args.threads)

    def stats_fn(nprobe: int):
        return index.search(
            q_pca, args.k, nprobe, num_threads=args.threads, return_stats=True
        )[2]

    # Relerr Delta must live in the same domain as D_hat: ME8Q estimates in
    # the full-rank PCA space (README.md). Recall still uses original
    # GT ids, which the PCA transform does not permute.
    if disk_base:
        # The build read the packed PCA copy front to back; every read from
        # here on is a scattered row pick, so read-ahead off (and the pages
        # themselves back) is what keeps a 39/69 GiB corpus from camping in
        # this host's page cache.
        advise_random(base_pca)
    if getattr(args, "build_only", False):
        # Construction-only run (--build-only): the nprobe sweep is what the
        # construction tables do not need, and skipping it takes the warm-up
        # rounds and the whole grid out of the cell's wall clock.  One
        # placeholder row keeps the artifact a parseable ledger.
        rows, rel_avg, rel_max, rel_p99 = [{"nprobe": -1}], None, None, None
    else:
        rows, rel_avg, rel_max, rel_p99 = run_nprobe_grid(
            method="me8q",
            grid=parse_grid(args.grid),
            search_fn=search_fn,
            queries=q_pca,
            base=base_pca,
            gt=gt,
            k=args.k,
            warmup=args.warmup,
            repeats=args.repeats,
            min_ms=args.min_ms,
            stats_fn=stats_fn,
        )
    if disk_base:
        release_mapped_pages(base_pca)
    # ADR@k at the Table-5 depths (README.md): raw-space
    # sum-ratio form, the same depths e8q/rabitq use. The raw-space vectors
    # enter only through the returned/GT ids, so gather their union from the
    # fvecs on demand and score the identical adr_sum_ratio math on that
    # dense submatrix — bit-identical to the resident-base path without
    # holding the second 4*n*dim copy.
    if getattr(args, "adr", False):
        import numpy as np

        ids_by_np: dict[int, np.ndarray] = {}
        for r in rows:
            if r["nprobe"] in (16, 64, 128, 256):
                ids_a = index.search(
                    q_pca, args.k, r["nprobe"], num_threads=args.threads
                )
                if isinstance(ids_a, tuple):
                    ids_a = ids_a[0]
                ids_by_np[r["nprobe"]] = np.asarray(ids_a)[:, :args.k]
        gt_k = np.asarray(gt)[:, :args.k]
        need = np.unique(np.concatenate(
            [gt_k.ravel()]
            + [v[v >= 0].ravel() for v in ids_by_np.values()]))
        reader = FvecsRowReader(corp["ddir"] / f"{args.dataset}_base.fvecs")
        sub = reader.gather(need)
        lut = np.full(reader.shape[0], -1, dtype=np.int64)
        lut[need] = np.arange(need.size)
        gt_m = lut[gt_k]
        for npb, v in ids_by_np.items():
            row = next(r for r in rows if r["nprobe"] == npb)
            row["me8q_adr"] = adr_sum_ratio(sub, queries, gt_m, lut[v], args.k)
    write_ledger(
        ledger_dest(args.out, args.dataset, bits, args.threads, "me8q",
                    getattr(args, "trial", 1)),
        method="me8q",
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
        extra={"me8q_prune_alpha": args.prune_alpha,
               "me8q_drop_energy": not args.no_drop_energy,
               "me8q_match_bpv": args.match_bpv,
               "me8q_match_total_budget": bool(args.match_total_budget),
               "me8q_build_last_s": last_build_s,
               "me8q_unpadded_bpv": unpadded_bpv,
               "me8q_plan": ",".join(f"{l}:{b}" for l, b in index.plan),
               "me8q_bpv_code": sum(l * b for l, b in index.plan) / 8.0,
               "me8q_bpv_factor": 8.0 * sum(1 for _, b in index.plan if b > 0),
               "me8q_bpv_other": (mem_bytes / n
                                  - sum(l * b for l, b in index.plan) / 8.0
                                  - 8.0 * sum(1 for _, b in index.plan if b > 0)),
               "me8q_bits_per_dim": mem_bytes * 8.0 / n / dim,
               "me8q_mem_total_bytes": mem_bytes,
               "me8q_build_peak_gb": peak_rss_gb(),
               "me8q_build_kvps": (n / build_s / 1000.0) if build_s > 0 else -1.0},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
