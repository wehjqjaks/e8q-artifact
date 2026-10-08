#!/usr/bin/env python3
"""SAQ / CAQ baseline under the paper's measurement protocol (AVX-512 only).

Wraps the vendored SAQ binaries (`create_index`, `test_qps`,
`test_relative_error`). SAQ = CAQ encoder + segmentation; CAQ =
segmentation off. Both consume the same PCA / IVF files as ME8Q.

The SAQ binary hard-codes `./data/<ds>/` relative to its CWD, so this
wrapper symlinks the prepared dataset into a scratch CWD and converts
the CSV it writes into the shared ledger JSON:

    <out>/<ds>_b<bits>_t<T>_<saq|caq>.json
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
SAQ = REPO / "baselines" / "SAQ"


def saq_search_m(dataset: str) -> int:
    """`-searcher_vars_bound_m` per dataset.

    SAQ's paper default is m=4 (m=2 on GIST). msmarco10M and stella need
    m=16: at m=4 their coarse-to-fine bound false-prunes far below b=4
    (msmarco b=4 recall@10 0.79 -> 0.89, stella 0.71 -> 0.97 at nprobe 64,
    at ~40-50% QPS), which would compare ME8Q against a crippled baseline.
    Below b=2 even m=16 does not move SAQ's recall (the failure is the
    quantizer, not the bound), so those rungs stay as the paper's.
    """
    if dataset.startswith("gist"):
        return 2
    if dataset.startswith("msmarco10M") or dataset.startswith("stella-trec24-1024"):
        return 16
    return 4


from protocol import (  # noqa: E402
    add_common_args,
    banner,
    corpus_shape,
    ledger_dest,
    load_corpus,
    make_row,
    mem_estimate,
    parse_grid,
    ram_guard,
    write_ledger,
)


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def prepare_cwd(data_root: Path, dataset: str, scratch: Path) -> Path:
    src = data_root / dataset
    dst = scratch / "data" / dataset
    dst.mkdir(parents=True, exist_ok=True)
    (scratch / "results" / "saq").mkdir(parents=True, exist_ok=True)
    names = [
        f"{dataset}_base.fvecs",
        f"{dataset}_base_pca.fvecs",
        f"{dataset}_base_pca.vars.fvecs",
        f"{dataset}_query.fvecs",
        f"{dataset}_query_pca.fvecs",
        f"{dataset}_centroid_4096.fvecs",
        f"{dataset}_centroid_4096_pca.fvecs",
        f"{dataset}_cluster_id_4096.ivecs",
        f"{dataset}_groundtruth.ivecs",
    ]
    for name in names:
        p = src / name
        if p.exists():
            _link(p.resolve(), dst / name)
    return dst


def _run(cmd: list[str], cwd: Path, env: dict | None = None,
         label: str = "saq") -> str:
    # Label each invocation with the method it actually runs (saq vs caq) —
    # both share the same vendored binaries, and an unlabelled "saq:" prefix
    # on a caq command line reads like a mixed-precision arm running at a
    # fixed bit width.
    print(f"{label}:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True,
                       text=True, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    print(out, flush=True)
    return out


def _run_peak(cmd: list[str], cwd: Path, env: dict | None = None,
              label: str = "saq") -> tuple[str, int]:
    """Run a build subprocess and track its VmHWM peak (README.md).

    VmHWM is the kernel's high-water RSS mark, so a 50 ms poll can only
    miss the tail of the final interval — never under-report a plateau.
    """
    print(f"{label}:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    peak = 0
    while p.poll() is None:
        try:
            with open(f"/proc/{p.pid}/status") as fh:
                for ln in fh:
                    if ln.startswith("VmHWM:"):
                        peak = max(peak, int(ln.split()[1]) * 1024)
                        break
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    out = (p.stdout.read() or "") if p.stdout else ""
    p.wait()
    print(out, flush=True)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, out)
    return out, peak


def _index_args(index_path: Path | None) -> list[str]:
    if index_path is None:
        return []
    return [f"-index_file={index_path.resolve()}"]


def run(method: str = "saq", argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--method", choices=("saq", "caq"), default=method)
    ap.add_argument("--saq-extra", action="append", default=[],
                    help="extra flag forwarded verbatim to the SAQ binaries "
                         "(e.g. --saq-extra=-searcher_vars_bound_m=16); does "
                         "not affect budget calibration")
    args = ap.parse_args(argv)
    method = args.method

    create = SAQ / "bin" / "create_index"
    test_qps = SAQ / "bin" / "test_qps"
    test_rel = SAQ / "bin" / "test_relative_error"
    if not create.is_file() or not test_qps.is_file():
        print("saq: binaries missing; build baselines/SAQ first", flush=True)
        return 1

    bits = float(args.bits)
    bits = int(bits) if bits.is_integer() else bits
    n, dim, nq_all = corpus_shape(args.data_root, args.dataset)
    if not ram_guard(
        mem_estimate("saq", n, dim, bits) + 2.0 * 4 * n * dim,
        f"{args.dataset} {method} b{bits}",
    ):
        return 0
    corp = load_corpus(args.data_root, args.dataset, args.nq, load_base=False)
    n, dim, nq = corp["n"], corp["dim"], corp["nq"]
    enable_seg = method == "saq"
    grid = parse_grid(args.grid)
    if getattr(args, "build_only", False):
        # Construction-only run (--build-only): the construction tables need
        # the build, not the sweep.  SAQ's search is a subprocess and its
        # build is timed separately, so the cheapest correct form is to keep
        # one grid point (the ledger keeps its row shape) and let the caller
        # drop the relative-error probe.
        grid = grid[:1]
    dest = ledger_dest(args.out, args.dataset, bits, args.threads, method,
                       getattr(args, "trial", 1))

    banner(args.dataset, n, dim, nq, bits, args.k, args.threads, args.seed)
    # Batch master seed -> SAQ's rotation. Upstream draws it from an
    # unseeded std::rand() stream (a fixed matrix for every process); the
    # vendored hook (patch 11) reproduces one exact rotation per batch so
    # the shared-seed contract holds for saq/caq like it does for e8q,
    # me8q and rabitq. Unset reproduces upstream bitwise.
    os.environ["SAQ_ROTATOR_SEED"] = str(args.seed)
    idx_flags = _index_args(args.index_path)
    if args.index_path is not None:
        args.index_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="saqwrap_") as tmp:
        scratch = Path(tmp)
        prepare_cwd(args.data_root, args.dataset, scratch)
        common = [
            f"-dataset={args.dataset}",
            f"-B={args.bits}",
            "-K=4096",
            f"-num_threads={args.threads}",
            f"-enable_segmentation={'true' if enable_seg else 'false'}",
            "-enable_PCA=true",
            "-rand_rotate=true",
            *idx_flags,
            # Per-dataset m (see saq_search_m) unless an explicit --saq-extra
            # overrides it. Extras come FIRST: gflags keeps the first
            # occurrence of a repeated flag.
            *(args.saq_extra or [
                f"-searcher_vars_bound_m={saq_search_m(args.dataset)}"]),
        ]
        have_index = args.index_path is not None and args.index_path.exists()
        build_s = -1.0
        plan_str = ""
        payload_bits = None
        # Filled by the saq budget calibration below; the reuse path keeps
        # the nominal knobs (the persisted index already fixed the plan).
        B_used = float(args.bits)
        search_builds = 0
        build_peak_bytes = 0

        def _do_build(flags, env=None):
            nonlocal build_peak_bytes
            t0 = time.time()
            out, peak = _run_peak([str(create), *flags], cwd=scratch, env=env,
                                  label=method)
            build_peak_bytes = max(build_peak_bytes, peak)
            dt = time.time() - t0
            # Parse the DP's own segmentation print: the header line
            # ("Dynamic bits allocation plan:") is followed by the plan
            # ("1bits: | 0 -> 64 (64d 4b) | 64 -> 256 (192d 0b)").
            lines = out.splitlines()
            p_at = next((i for i, ln in enumerate(lines)
                         if "allocation plan" in ln), -1)
            segs = []
            for ln in lines[p_at + 1 : p_at + 3]:
                segs = re.findall(r"(\d+)d\s+(\d+)b", ln)
                if segs:
                    break
            pstr = ",".join(f"{d}d:{b}b" for d, b in segs) if segs else ""
            pbits = sum(int(d) * int(b) for d, b in segs) if segs else None
            bs = dt
            for p in (scratch / "results" / "saq").glob("*.index.csv"):
                with p.open() as fh:
                    recs = list(csv.DictReader(fh))
                if recs and "index_time_s" in recs[0]:
                    bs = float(recs[0]["index_time_s"])
                    break
            return pstr, pbits, bs

        def _index_files():
            return sorted((scratch / "data" / args.dataset).glob("*.index"))

        def _coded_segs(pstr):
            out = []
            for tok in pstr.split(",") if pstr else []:
                d, b = tok[:-1].split("d:")
                if int(b) > 0:
                    out.append((int(d), int(b)))
            return out

        def _flags_for(b_val):
            return [f for f in common if not f.startswith("-B=")] \
                + [f"-B={b_val:g}"]

        if not have_index:
            plan_str, payload_bits, build_s = _do_build(common)
            B_used = float(args.bits)
            search_builds = 1
            actual_bpv = None
            if method == "saq":
                # True-bit-budget calibration (README.md): users pay
                # TOTAL memory, so SAQ gets b bits per dim per vector of
                # code + metadata. Its own -B is the documented code-rate
                # quota and the serialized index adds per-vector metadata
                # (~46 B/vec on deep1M) on top, so at the nominal B = b the
                # REAL total already exceeds b*dim/8. Bisect B downward
                # from b for the largest quota whose MEASURED total still
                # fits: the achieved bits_per_dim lands as close to b as
                # SAQ's plan ladder allows without exceeding it. No SAQ
                # source is modified — only its own documented budget knob
                # is calibrated. When not even the smallest coded plan
                # fits, the cell is SKIPPED (marker file, no ledger): a
                # recorded capability gap, never a floor run.
                target_bpv = args.bits * dim / 8.0
                B_lo, B_hi = 0.0, None
                best = None
                cur_B = float(args.bits)
                seen = {round(cur_B, 4)}
                while True:
                    files = _index_files()
                    actual_bpv = files[0].stat().st_size / n if files else None
                    if actual_bpv is None:
                        break
                    if _coded_segs(plan_str) and actual_bpv <= target_bpv + 1e-9:
                        best = (cur_B, plan_str, payload_bits, build_s,
                                actual_bpv)
                        B_lo = cur_B
                    elif actual_bpv > target_bpv:
                        B_hi = cur_B
                    else:
                        # All-0 plan: the quota buys no coded segment at all.
                        B_lo = cur_B
                    if search_builds >= 9:
                        break
                    if best is not None and B_hi is not None \
                            and B_hi - best[0] < 0.05:
                        break
                    if best is not None and B_hi is not None:
                        nxt = 0.5 * (best[0] + B_hi)          # bracketed
                    else:
                        # Slope-1 estimate: total bits/dim move 1:1 with the
                        # quota, so the measured excess/shortfall IS the step.
                        nxt = cur_B + (target_bpv - actual_bpv) * 8.0 / dim
                        if abs(nxt - cur_B) < 0.25:
                            nxt = cur_B - 0.25 if actual_bpv > target_bpv \
                                else cur_B + 0.25
                    nxt = round(min(max(nxt, 0.05), float(args.bits)), 4)
                    if nxt in seen:
                        if best is None or B_hi is None:
                            break
                        nxt = round(0.5 * (best[0] + B_hi), 4)
                        if nxt in seen:
                            break
                    seen.add(nxt)
                    cur_B = nxt
                    for p in _index_files():
                        p.unlink()
                    for p in (scratch / "results" / "saq").glob("*.index.csv"):
                        p.unlink()
                    plan_str, payload_bits, build_s = _do_build(_flags_for(cur_B))
                    search_builds += 1
                if best is None:
                    dest.with_suffix(".skipped").write_text(json.dumps({
                        "reason": "budget_too_low",
                        "b": args.bits,
                        "budget_bpv": target_bpv,
                        "best_bpv": actual_bpv,
                        "plan": plan_str,
                        "builds": search_builds,
                    }))
                    print(f"saq: budget too low at b={args.bits} "
                          f"(budget {target_bpv:.1f} B/vec — no coded plan "
                          f"fits) — skipped, no plan", flush=True)
                    return 0
                B_used, plan_str, payload_bits, build_s, actual_bpv = best
                if abs(cur_B - B_used) > 1e-9:
                    # Leave the winning plan's index on disk for the search
                    # and relerr phases (they reload by -B-derived filename).
                    for p in _index_files():
                        p.unlink()
                    for p in (scratch / "results" / "saq").glob("*.index.csv"):
                        p.unlink()
                    plan_str, payload_bits, build_s = _do_build(
                        _flags_for(B_used))
                    search_builds += 1
                print(f"saq budget b={args.bits}: target={target_bpv:.2f} "
                      f"B*={B_used:g} plan={plan_str or '-'} "
                      f"total={actual_bpv:.2f} B/vec builds={search_builds} "
                      f"bpd={actual_bpv * 8.0 / dim:.4f}", flush=True)
                common = _flags_for(B_used)
        else:
            print(f"saq: reuse {args.index_path}", flush=True)

        rel_avg = rel_max = rel_p99 = None
        relerr_by_np: dict[int, tuple[float, float, float | None]] = {}
        if test_rel.is_file() and args.relerr:
            try:
                # One process probes the SAQ-paper point (nprobe=200 -> the
                # ledger header) plus the Table-5 depths whose rows carry
                # per-cell relative error (README.md): vendored patch 13
                # (-err_nprobe_list) runs upstream's own run_error at each
                # depth; recall/QPS are untouched by this pass.
                _run(
                    [str(test_rel), *common,
                     "-err_nprobe_list=200,16,64,128,256"],
                    cwd=scratch, label=method,
                )
                csvs = list((scratch / "results" / "saq").glob("*.csv"))
                for p in csvs:
                    with p.open() as fh:
                        reader = csv.DictReader(fh)
                        if reader.fieldnames and "err_tot_avg" in reader.fieldnames:
                            for rec in reader:
                                try:
                                    np_v = int(float(rec["nprobe"]))
                                    avg_v = float(rec["err_tot_avg"])
                                    max_v = float(rec["err_tot_max"])
                                    p99v = rec.get("err_tot_p99")
                                    relerr_by_np[np_v] = (
                                        avg_v, max_v,
                                        float(p99v) if p99v not in (None, "") else None,
                                    )
                                except (KeyError, TypeError, ValueError):
                                    continue
                            break
            except subprocess.CalledProcessError as exc:
                print(f"saq: relative-error failed ({exc}); continuing", flush=True)
            hdr = relerr_by_np.get(200) \
                or next(iter(relerr_by_np.values()), None)
            if hdr is not None:
                rel_avg, rel_max, rel_p99 = hdr

        rows = []
        qps_csv_dir = scratch / "results" / "saq"
        # One process sweeps the whole nprobe grid (vendored patch 12, upstream's
        # own single-process sweep restricted to our points). Loading the index
        # once per arm instead of once per point is what the mixed-precision
        # cells spend most of their time on; per-point search semantics are
        # unchanged.
        before = {p.resolve() for p in qps_csv_dir.glob("qps_*.csv")}
        _run(
            [
                str(test_qps),
                *common,
                f"-fix_thread={args.threads}",
                f"-nprobe_list={','.join(str(np_) for np_ in grid)}",
                f"-topk={args.k}",
            ],
            cwd=scratch,
            label=method,
        )
        after = [p for p in qps_csv_dir.glob("qps_*.csv") if p.resolve() not in before]
        if not after:
            after = sorted(qps_csv_dir.glob("qps_*.csv"), key=lambda p: p.stat().st_mtime)
        by_nprobe: dict[int, dict] = {}
        if after:
            with after[-1].open() as fh:
                for rec in csv.DictReader(fh):
                    try:
                        by_nprobe[int(float(rec["nprobe"]))] = rec
                    except (KeyError, TypeError, ValueError):
                        continue
        for np_ in grid:
            rec = by_nprobe.get(np_)
            if rec is None:
                print(f"saq: no CSV row for nprobe={np_}", flush=True)
                continue
            recall = float(rec.get("recall", "nan"))
            qps = float(rec.get("QPS", "nan"))
            ratio = float(rec.get("ratio", "nan"))
            # sum-ratio ADR from the vendored test_qps patch; absent on
            # unpatched binaries (column missing) -> no field, not 0.
            adr_raw = rec.get("adr")
            # qps_p25/qps_p75 from the vendored per-round patch; fall back
            # to the headline mean on unpatched binaries.
            p25 = rec.get("qps_p25")
            p75 = rec.get("qps_p75")
            stats = {
                "qps": qps,
                "qps_p25": float(p25) if p25 not in (None, "") else qps,
                "qps_p75": float(p75) if p75 not in (None, "") else qps,
                "nrounds": 10,
            }
            row_extra = None
            if adr_raw not in (None, ""):
                row_extra = {f"{method}_adr": float(adr_raw)}
            rows.append(make_row(method, np_, recall, stats, ratio, row_extra))
            print(f"{np_:>8} {recall:>12.4f} {qps:>12.1f}", flush=True)

        # Per-cell relerr rows at the Table-5 depths (§4.2), from the
        # patch-13 probe above; other nprobes carry no field, never 0.
        for r in rows:
            v = relerr_by_np.get(r["nprobe"])
            if v is not None:
                r[f"{method}_relerr_avg"] = v[0]
                r[f"{method}_relerr_max"] = v[1]
                if v[2] is not None:
                    r[f"{method}_relerr_p99"] = v[2]

        # Memory: the quantized index file (README.md). With
        # --index-path it is the persisted artifact; otherwise stat the
        # file create_index just wrote inside the scratch data dir before
        # the scratch tree is removed.
        mem_bytes = None
        if args.index_path is not None and args.index_path.exists():
            mem_bytes = args.index_path.stat().st_size
        else:
            for p in (scratch / "data" / args.dataset).glob("*.index"):
                mem_bytes = p.stat().st_size
                break
        mem_gb = None if mem_bytes is None else mem_bytes / 1e9
        # Effective search m: the per-dataset default unless --saq-extra
        # overrode the flag (gflags keeps the first occurrence).
        m_eff = saq_search_m(args.dataset)
        for tok in (args.saq_extra or []):
            if tok.startswith("-searcher_vars_bound_m="):
                try:
                    m_eff = int(tok.split("=", 1)[1])
                except ValueError:
                    pass
        extra = {
            "note": (
                "SAQ binary TOPK via -topk (vendored; default 10). Searcher is "
                "SAQ's own; the nprobe grid is swept in ONE process via "
                "-nprobe_list (vendored patch 12 — upstream's own single-process "
                "sweep restricted to our points), so the index loads once per arm "
                "instead of once per point, the same way e8q/rabitq/pq sweep "
                "theirs. QPS is the binary's mean over "
                "10 internal rounds; qps_p25/qps_p75 are the IQR of the same "
                "rounds (vendored patch). relerr comes from one "
                "test_relative_error process probing nprobe 200 (header; the "
                "SAQ paper definition) plus 16/64/128/256 (rows, §4.2) via "
                "vendored patch 13; ADR is the vendored sum-ratio column on "
                "every grid point."
            ),
        }
        if mem_bytes is not None:
            extra[f"{method}_bytes_per_vector"] = mem_bytes / n
            extra[f"{method}_bits_per_dim"] = mem_bytes * 8.0 / n / dim
            extra[f"{method}_mem_total_bytes"] = mem_bytes
            if payload_bits is not None:
                # Decomposition from the parsed DP plan: code payload vs
                # everything else (factors, norms, codebooks, serialization).
                extra[f"{method}_bpv_code"] = payload_bits / 8.0
                extra[f"{method}_bpv_overhead"] = mem_bytes / n - payload_bits / 8.0
            if plan_str:
                extra[f"{method}_plan"] = plan_str
            if build_s > 0:
                extra[f"{method}_build_kvps"] = n / build_s / 1000.0
            if build_peak_bytes > 0:
                extra[f"{method}_build_peak_gb"] = build_peak_bytes / 1e9
            if method == "saq":
                extra[f"{method}_b_used"] = B_used
                extra[f"{method}_search_builds"] = search_builds
                extra[f"{method}_budget_bpv"] = target_bpv
                extra[f"{method}_searcher_vars_bound_m"] = m_eff
        write_ledger(
            dest,
            method=method,
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
            mem_gb=mem_gb,
            relerr_avg=rel_avg,
            relerr_p99=rel_p99,
            relerr_max=rel_max,
            extra=extra,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run("saq"))
