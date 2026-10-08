# Baseline vendoring notes

Third-party baselines whose sources we modified live in this repository as vendored trees (plain directories, no submodules) so the modifications ship with the reproduction.
Every change is marked `[e8qlib vendored patch ...]` at the edit site and listed here.
FAISS is **not** vendored and is **not** a submodule: the PQ arm uses the `faiss-cpu` Python wheel (`requirements.txt`), which already exposes `IndexIVFPQ` / ADC distances. There is no `baselines/faiss` tree and no FAISS source patch.

## baselines/RaBitQ-Library

Upstream: https://github.com/VectorDB-NTU/RaBitQ-Library (pinned at the
commit this tree was vendored from, `af29d8d`; converted from a submodule on
2026-09-20).

Patches:

1. `include/rabitqlib/utils/rotator.hpp` — `FhtKacRotator` constructor now
   honors the `RABITQ_ROTATOR_SEED` environment variable (decimal uint64) to
   make its otherwise per-construct random rotation reproducible. Rationale:
   the paper protocol shares one random rotation seed across every method in
   a run (E8Q/ME8Q consume it natively; RaBitQ has no upstream seeding API).
   When the variable is unset or unparseable, behavior is bitwise the
   upstream `random_device` path. `experiments/run_rabitq.py` exports the
   per-test seed through this variable before constructing the IVF.

2. `include/rabitqlib/index/ivf/ivf.hpp` + `python_bindings/ivf_bindings.cpp`
   — prune-internals stats: `IVF::SearchStats {candidates, survivors}` accrues
   scanned points and lower-bound admissions through optional
   `SearchStats*` parameters on `search` / `search_cluster` /
   `scan_one_batch` (null = no counting, timed path unchanged). A dedicated
   `search(..., float*, SearchStats*)` overload keeps a stats pointer from
   binding to the bool `use_hacc` overload. `IvfIndex.search(...,
   return_stats=True)` returns `(ids, dists, {candidates, survivors,
   survivor_rate})` with per-thread accumulators, so the e8q-vs-rabitq prune
   comparison runs on the same 8-thread batched path. Replaces the earlier
   `RABITQ_SR` env-gated stderr probe (global atomics; raced under parallel
   search). b=1 (`ex_bits_ == 0`, no rerank) inserts every 1-bit estimate,
   so survivors = candidates there by construction.

3. `include/rabitqlib/index/ivf/ivf.hpp` + `python_bindings/ivf_bindings.cpp`
   — `IVF::memory_bytes()` / `IvfIndex.memory_bytes`: total in-memory index
   bytes (batch + ex codes, ids, coarse quantizer, rotator, cluster table)
   for the paper's memory-vs-bits figures. Read-only computation; no layout
   change.

## baselines/SAQ

Upstream: https://github.com/howarlii/SAQ (pinned at the commit this tree
was vendored from, `2163ebc`; converted from a submodule on 2026-09-21).
SAQ/CAQ require AVX-512 (their fastscan LUT path uses `__m512` and their
build enables `-mavx512*` unconditionally), so the SAQ/CAQ arms exist only
in AVX-512 builds on AVX-512 CPUs.

Patches:

1. `src/define_options.h` — `FLAGS_index_file` overrides `DataFilePaths::quant_file`
   so `experiments/run_saq.py --index-path` can persist/reuse a flat
   `<run>/indexes/<ds>_b<bits>_t<T>_<saq|caq>.idx`. Empty keeps the
   upstream `./data/<ds>/<args>.index` path.

2. `src/test_qps.cpp` — `FLAGS_topk` (default 10) replaces the hardcoded
   `TOPK = 100`. Paper STABLE protocol is Recall@10 / k=10; the binary's
   own QPS/ratio now use that k. Wrapper passes `-topk=` from `--k`.

3. `src/test_qps.cpp` — sum-ratio ADR@topk: `Stats` gains `adr`, a new `adr`
   CSV column after `qps_p75` — mean over queries of
   $\sum_j L_2(q, \mathrm{ann}_j) / \sum_j L_2(q, \mathrm{gt}_j)$ with physical
   (sqrt) L2, computed once per round and averaged over the ROUND rounds.
   Queries whose exact NN is a numerical duplicate
   ($d_{nn} \le 10^{-6} \times \mathrm{median}\ d_{nn}$) are excluded (the
   duplicate-vector guard). Rationale: the paper protocol's ADR@k is a
   sum-ratio; upstream `ratio` is a rank-paired sqrt-mean ratio per rank and
   stays untouched (column `ratio` unchanged). `experiments/run_saq.py` maps
   the `adr` column to `<method>_adr` ledger rows (missing column on old
   binaries ⇒ no field, never 0).

4. `saqlib/utils/pool.hpp` — `AvgMaxRecorder` keeps a sample of the inserted
   values and gains `p_percentile(p)` (linear interpolation);
   `AvgMaxGroup::tot_p99()` exposes the merged 99th percentile. Rationale:
   the protocol reports relative-error p99; avg/max alone cannot express the
   tail. `insert` is now single-threaded by contract (the sample owns a heap
   buffer), and `merge` carries the sample through — see patches 7 and 8.
   The sample is capped at `kMaxKept = 4096` values per recorder: once the
   cap is hit, every other stored value is dropped and the sampling stride
   doubles, so the retained set stays a uniform stride-$2^k$ subsample of the
   stream. Without the cap `test_relative_error` would hold
   $NQ \times \text{visited}$ doubles per nprobe row (~4 GB on sift1M, ~40 GB
   for a 10M-point set); with it the peak is
   $NQ \times kMaxKept \times 8$ bytes (~300 MB on sift1M, measured peak RSS
   1.1 GB, no measurable change to the reported p99: 0.02375 vs 0.02391
   exact on sift1M b=4). These recorders only run in the offline eval
   binaries, never in a timing loop.

5. `src/test_relative_error.cpp` — the CSV grows an `err_tot_p99` column
   (from patch 4) appended after `err_q_mx_avg`.
   `experiments/run_saq.py` reads it into `<method>_relerr_p99`.

6. `src/test_qps.cpp` — per-round QPS IQR: `Stats` gains `qps_p25` /
   `qps_p75` (percentiles of the `ROUND = 10` internal per-round QPS
   values), and the result CSV grows `qps_p25,qps_p75` columns. Headline
   `QPS` stays the upstream mean of the 10 rounds. Rationale: the paper
   protocol reports IQR p25/p75 for every method; SAQ's own timing loop is
   internal to the binary, so the distribution is exported instead of
   being re-measured in python (which would perturb the thread pool).

Rebuild `baselines/SAQ` after these patches (`cmake --build` in its tree)
so `bin/create_index` / `bin/test_qps` pick them up.

7. `saqlib/index/ivf.hpp` + `saqlib/quantization/saq_quantizer.hpp` —
   `IVF::construct` step 4 gives every thread-pool task its own
   `SAQuantizer` and folds the private metrics back under a mutex
   (`SAQuantizer::merge_metrics`). Upstream shares one quantizer across the
   pool, whose per-segment `QuantMetrics` recorder each worker mutates;
   that is survivable while the recorder is three plain scalars but corrupts
   the heap once patch 4 makes it own a buffer (`free(): unaligned chunk
   detected in tcache 2`, aborting every 12-thread `create_index`). The
   metrics are the same ones upstream collects; index content is unaffected
   (verified on sift1M b=4 with `-num_threads=1` against a pristine build:
   `test_qps -fix_nprobe=32` reports recall 0.87883, `ratio` 1.0031847 and
   `adr` 1.003169 for both, to the last digit). Note that the index *file*
   is not byte-comparable: `SaqCluData::save` serializes the fastscan
   padding lanes without initializing them, so two runs of the same binary
   already differ in ~1.3 kB of 89.8 MB, and comparing across builds shows
   those same padding bytes, never the codes.

8. `saqlib/quantization/caq/caq_encoder.hpp` — `encode_and_fac` guards the
   code/factor export with `num_bits_ > 0` instead of `num_bits_ > 1`.
   Rationale: upstream bug, independent of patches 1–7. A 1-bit segment
   packs its LSB through the same short-code path as any other bit width, so
   the packer needs `base_code.code`; upstream leaves it an empty
   `Eigen::VectorXi`, and `ClusterPacker::store_and_pack` then reads
   `code[0]` out of bounds. In the release build (asserts off) that is a
   segfault in `create_index`; in a debug build it trips
   `num_dim_pad_ == base_code.code.size()`. It hits exactly the
   `caq` arm at b=1, whose plan is a single unsegmented 128d 1-bit segment
   (`saq` at b=1 is unaffected because its plan splits into 2-bit and 0-bit
   segments). 0-bit segments still skip the export: `encode()` leaves the
   code empty there and the packer returns before touching it.

9. `saqlib/quantization/saq_data.hpp` — `SaqData::load` checks
   `data_variance.cols()` against `rd_up_to_multiple_of(num_dim,
   kDimPaddingSize)` instead of `num_dim`. Upstream bug, independent of
   patches 1–8: the build side (`set_variance` + `prepare_quantizers`)
   zero-pads the variance to the padded width and `save()` serializes the
   padded vector, so `load()`'s check against the *original* dim aborts
   every index load for datasets whose dim is not a multiple of 64
   (`Check failed: data_variance.cols() == num_dim (256 vs. 200)` on
   glove200; msong 420→512 is the same case). Dims already 64-aligned are
   byte-identical in behavior (`num_dim_padded == num_dim`); index content
   and all timing paths are untouched — only the load-side assertion moves.

10. ~~`saqlib/quantization/saq_data.hpp` budget hook~~ — **withdrawn
    2026-09-23.** An earlier revision of this tree subtracted the measured
    per-vector metadata from the DP's bit quota through a
    `SAQ_BUDGET_META_BITS` environment variable. The true-bit-budget
    protocol (PROTOCOL.md §3) does **not** modify the allocator: the
    wrapper searches SAQ's own documented budget knob (`-B`, real-valued)
    upward until the *measured* serialized total fits `b·dim/8` per vector,
    so the DP itself stays byte-identical to upstream. No patch is needed
    and the environment hook is removed.

11. `saqlib/utils/rotator.hpp` — `Rotator::orthogonalize()` honors the
    `SAQ_ROTATOR_SEED` environment variable (decimal uint32) by calling
    `std::srand` before Eigen's `Random()`. Rationale: Eigen's random
    device is `std::rand()` (vendored `Eigen/src/Core/RandomImpl.h`) and
    upstream never seeds it, so every process draws the *same* rotation
    from the default rand() stream — deterministic, but a fixed constant
    that no experiment seed can move. The paper protocol shares one master
    seed per batch across every method and bit width (PROTOCOL.md §1.1);
    e8q/me8q take it natively and RaBitQ through its own vendoring patch,
    so SAQ's rotation must follow the same seed to keep the batch
    reproducible. It is not an algorithmic change — the rotation is drawn
    exactly as upstream draws it, only from a seeded stream (same class of
    hook as RaBitQ patch 1); unset/absent leaves behavior bitwise upstream.
    Index content is otherwise untouched: with one fixed seed, two builds
    agree on the header + rotation region, and the only residual
    differences between runs are the known uninitialized fastscan padding
    lanes (~1.3 kB, patch 7's note).

12. `src/test_qps.cpp` — `FLAGS_nprobe_list` (comma-separated): sweeps an
    explicit nprobe grid inside **one** process. Rationale: upstream already
    sweeps its whole nprobe list in a single process (`fix_nprobe=0`, one
    index load per run); our wrapper had been invoking the binary once per
    grid point, which reloaded the quantized index 52 times per arm and made
    every mixed-precision cell cost ~5 minutes even on a 1M-vector corpus —
    and reloads grow with the index, so the 10M-vector corpora would have
    dominated the whole campaign. With `-nprobe_list` the arm loads the index
    once and calls the *same* `run_search_multi` per point in the same order,
    so per-point search semantics, recall, ratio and ADR are unchanged; only
    process setup and index loading are amortized (which is also how our own
    e8q/rabitq/pq arms already sweep their grids, so this makes the baseline
    arm *more* comparable, not less). `-fix_nprobe` still works. Verified
    bit-identical recall/ratio/ADR against the per-point invocation on
    sift1M b=4 caq (same seed).


13. `src/test_relative_error.cpp` — `FLAGS_err_nprobe_list` (comma-separated):
    measures relative error at explicit nprobe depths in one process.
    Upstream measures only nprobe=200 (its one hardcoded `run_error` call);
    the paper's ledger schema (PROTOCOL.md §4.2) also wants per-cell relerr
    at the Table-5 depths {16, 64, 128, 256}. The wrapper probes
    `200,16,64,128,256`: the 200 row stays the header field (SAQ paper
    definition), the other four merge into the matching grid rows. Same
    spirit as patch 12; recall/QPS passes are untouched by this probe.

Vendoring detail: `saqlib/defines.hpp` includes `third/Eigen/Dense`, but
upstream's `.gitignore` excludes `saqlib/third/`, so a fresh checkout of
the upstream repository is missing the Eigen headers the build needs. The
vendored tree therefore carries `saqlib/third/Eigen/` (headers only)
explicitly.

## FAISS (Python wheel, not in `baselines/`)

Upstream: https://github.com/facebookresearch/faiss.
Consumed as `faiss-cpu>=1.8` from `requirements.txt` (`import faiss`). A source checkout under `baselines/faiss` is unnecessary: we never patch FAISS, never rebuild it, and never call its C++ binaries.

PQ (`experiments/run_pq.py`) follows the SAQ paper's baseline section: `faiss.IndexIVFPQ` with 8-bit sub-quantizers (`nbits=8`), $M = \lfloor b \cdot D / 8 \rfloor$ to hit the bit budget, the shared IVF $k$-means centroids from `datasets/prepare.py` preloaded into an `IndexFlatL2` quantizer (PQ trains only the sub-codebooks), squared L2, $k=10$, no raw rerank. Budgets with `dim % M != 0` are skipped. The estimated distance for the relative-error probe is the ADC distance `IndexIVFPQ.search()` already returns.

`datasets/prepare.py` also uses the same wheel for the offline IVF $k$-means (`nlist=4096`) that every method shares.
