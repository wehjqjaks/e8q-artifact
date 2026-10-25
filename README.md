# e8qlib — code and reproduction package

For the paper

> **Escaping the Hypercube: Fast $E_8$ Lattice Quantization for Approximate Nearest Neighbor Search**

This package accompanies the manuscript.  e8qlib implements
**E8Q**, a training-free quantizer for IVF approximate nearest neighbor
search built on a finite $E_8$ lattice codebook, and **ME8Q**, its
mixed-rate extension (E8Q's codec under SAQ's variance-adaptive bit
allocation).  The method, theory, and results are the paper's; this README
covers building the code and reproducing the measurements.  Script comments
that cite "README.md" for the protocol mean the measurement protocol
summarized under *Reproduce the paper* below.

## Layout

| Path | Responsibility |
| --- | --- |
| `include/`, `src/`, `python_bindings/`, `CMakeLists.txt`, `cmake/` | The e8qlib C++17 library (E8Q codec, ME8Q codec, IVF indexes, SIMD kernels) and the pybind11 module |
| `tests/`, `sample/` | C++ unit tests (`e8q_test`, `me8q_test`, `scan_kernel_test`), Python tests, C++/Python usage samples |
| `datasets/` | Download and preprocessing drivers; `datasets/README.md` documents the corpus registry and sources |
| `experiments/` | Every experiment the paper reads, the method runners, and the figure/table generators (`experiments/plots/`, `experiments/shaping/`) |
| `baselines/` | Vendored RaBitQ-Library and SAQ sources (pinned commits, patches documented in `baselines/CHANGES.md`); software FAISS via the `faiss-cpu` wheel |
| `run_all.sh`, `plot_all.sh` | One-click entry points: measurements, then figures + tables |
| `README.md` | This file: build, run, and reproduction guide |

Corpora, built indexes, measurement outputs, and figures are **not**
shipped — the package is code and scripts only; everything is regenerated
locally by the commands below.  QPS and wall-clock numbers are properties of
the machine that took them; the recall/ratio/bound numbers are not.

## Reference environment

The paper's numbers were taken on this machine class:

| | |
| --- | --- |
| CPU | x86-64 with AVX-512 (F, BW, DQ, VL, VBMI); 12 physical cores / 24 logical |
| Threads | 12 for search and for construction (one per physical core, `taskset -c 0,2,...,22`) |
| RAM | ≥ 64 GiB for the full campaign; the two large corpora (MSMARCO-10M, Stella) auto-skip on smaller hosts with a `ram-skip:` log line and are picked up on re-run (`E8Q_MMAP_BASE=1` streams them from disk instead, recall-identical, QPS not) |
| OS / toolchain | Linux, Python ≥ 3.11, CMake ≥ 3.15, a C++17 compiler |

The SAQ/CAQ baseline arms require an AVX-512 host (their fastscan path is
AVX-512-only); on other machines those arms are skipped by protocol with a
logged notice and no metrics file.

## Build

C++ (Release; compile-time ISA via `E8Q_ISA`):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DE8Q_ISA=avx2
cmake --build build -j
./build/tests/e8q_test
./build/tests/me8q_test
./build/tests/scan_kernel_test
```

| `E8Q_ISA` | What is compiled |
| --- | --- |
| `generic` | scalar kernels only |
| `avx2` (default) | AVX2 kernel TUs |
| `avx512` | AVX-512 kernel TUs (F+BW+DQ+VL+VBMI) — what the paper's runs use |

Runtime `Avx2Enabled()` / `Avx512Enabled()` never report a backend that was
not compiled; detection uses `cpuid` + `xgetbv`, never by executing a
high-ISA kernel.

Python (any venv or conda env):

```bash
python -m pip install -r requirements.txt
python -m pip install -e .            # pyproject turns the bindings on
python -m pytest tests/python -q
```

For the AVX-512 build add `-C cmake.define.E8Q_ISA=avx512` to the `pip
install` line.

## Quick start

```python
import numpy as np
from e8qlib import E8qIndex, Me8qIndex

rng = np.random.default_rng(42)
data = rng.standard_normal((500, 128)).astype(np.float32)
queries = rng.standard_normal((5, 128)).astype(np.float32)
cluster_ids = (np.arange(len(data)) % 5).astype(np.uint32)
centroids = np.stack(
    [data[cluster_ids == cluster].mean(axis=0) for cluster in range(5)]
).astype(np.float32)

index = E8qIndex(dim=128, nbits=4, seed=1)
index.build(data, centroids, cluster_ids)
ids, dists = index.search(queries, k=10, nprobe=5)
print(ids.shape, dists.shape)

mixed = Me8qIndex(dim=128, avg_bits=4.0)
mixed.build(data, centroids, cluster_ids)
ids, dists = mixed.search(queries, k=10, nprobe=5)
```

C++ samples (`E8Q_BUILD_SAMPLES=ON`, default):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DE8Q_ISA=avx2
cmake --build build --target e8q_index me8q_index -j
./build/sample/cpp/e8q_index
./build/sample/cpp/me8q_index
```

## Baselines

`baselines/RaBitQ-Library` (upstream `VectorDB-NTU/RaBitQ-Library`, pinned
commit `af29d8d`) and `baselines/SAQ` (upstream `howarlii/SAQ`, pinned
commit `2163ebc`) ship as vendored source trees; every modification is
marked `[e8qlib vendored patch ...]` at the edit site and documented in
`baselines/CHANGES.md`.  The patches are protocol hooks (shared rotation
seeds, memory/prune statistics, a top-$k$ flag, budget/load fixes) — none
changes the measured search semantics of the baselines.

Build them on the machine that runs the campaign:

```bash
# RaBitQ-Library (python module used by experiments/run_rabitq.py)
python -m pip install ./baselines/RaBitQ-Library

# SAQ / CAQ (binaries used by experiments/run_saq.py; needs AVX-512)
cd baselines/SAQ && mkdir -p build bin && cd build && cmake .. && make -j && cd ../..
```

The baseline arms run at the paper's own settings.  RaBitQ is Extended
RaBitQ in high-accuracy mode without reranking.  SAQ/CAQ use `-K=4096`,
PCA and random rotation enabled, dimension segmentation on for SAQ and off
for CAQ under the recommended adjustment defaults, and SAQ's multi-stage
estimator at the $m$ the paper states ($m = 16$ on MSMARCO-10M and Stella,
$m = 2$ on GIST, $m = 4$ elsewhere; `run_saq.py` sets
`-searcher_vars_bound_m`).  FAISS is a wheel (`faiss-cpu`, in
`requirements.txt`); there is no vendored FAISS tree.  The PQ arm uses
`faiss.IndexIVFPQ` (8-bit sub-quantizers, $M = bD/8$, trained on 256k
vectors), and `datasets/prepare.py` uses the same wheel for the shared IVF
k-means (`nlist = 4096`).

## Reproduce the paper from scratch

### 1. Corpora (public; ~200 GB of downloads for the full set)

```bash
./datasets/prepare.sh msong            # one corpus, for a smoke test
./datasets/prepare.sh --all            # every corpus the downloader knows
```

The paper uses six: `sift1M` (1M × 128), `deep1M` (1M × 256), `gist`
(1M × 960), `openai1536` (999k × 1536), `msmarco10M` (10M × 1024),
`stella-trec24-1024` (17.8M × 1024).  `datasets/README.md` documents each
corpus's public source (texmex / GQR / UCI / Hugging Face / Cohere) and the
`--refill` recovery path.  Each corpus is then preprocessed: exact top-100
ground truth, shared IVF k-means (`nlist = 4096`), and a full-rank PCA (a
rotation, not a reduction, so one ground truth serves both the original and
the rotated domain).  The paper's canonical corpora were prepared with
SAQ's GPU pipeline; `prepare.py` reproduces the same recipe on CPU —
equivalent, not bit-identical, since k-means seeds differ, which can shift
recall margins slightly on low-dimensional corpora.

### 2. Experiments (the full six-corpus grid is an order-of-a-day job on the reference host)

```bash
./run_all.sh --data-root datasets --results results
```

Each method runs in its own process and writes one metrics JSON per
(corpus, rate, threads, method) into `results/{fixed,mixed,ablation,adr_full,
bound_coverage,shaping}/`.  `--stages` picks a subset (default
`main adr bound shaping`; the optional `diag` stage adds the
phases/latency/perf micro-benchmarks, which the paper does not plot).
Resume is idempotent: a completed ledger is skipped.

Measurement protocol the runners implement (paper §4.1; the same contract
in one place, since the scripts cite this README):

- **Search**: $k = 10$, 1,000 queries, squared L2, no raw-vector rerank,
  12 threads, the 53-point $n_\mathrm{probe}$ grid over [1, 512], screening
  margin factor $\alpha = 1.0$ (no per-dataset tuning).
- **Timing**: 3 warm-up passes, then repeats until at least 9 runs and
  400 ms accumulate; **median** QPS with p25/p75 IQR.
- **Fairness**: one seed per (corpus, trial) drives every method and every
  rate of that corpus — the rotation and query order are identical across
  the methods of a cell (E8Q/ME8Q consume the seed natively; the vendored
  RaBitQ reads `RABITQ_ROTATOR_SEED`, the vendored SAQ/CAQ
  `SAQ_ROTATOR_SEED`).  All methods share the same IVF centroids, PCA, and
  rotation; every arm constructs at the run's `--threads`.  Reported
  values are medians across trials (`E8Q_TRIALS`, default 1 for the
  submitted campaign, up to 5).
- **Memory accounting**: e8q vs RaBitQ/CAQ/PQ compare at equal code
  budgets.  ME8Q vs SAQ runs under the true-bit-budget protocol: total
  memory (code + metadata) must match the budget $b$ bits/dim per vector —
  ME8Q via `--match-total-budget`, SAQ via bisecting its own `-B` knob
  until the measured serialized total fits; when no coded plan fits, the
  cell is written as a `<stem>.skipped` marker (`budget_too_low`), never a
  floor-run result.
- **Gating, never silent**: missing corpora, insufficient RAM (`ram-skip:`
  lines), and non-AVX-512 hosts (saq/caq arms) are skipped with a logged
  notice and no metrics file; re-running on a suitable host fills exactly
  those gaps.

### 3. Figures and tables

```bash
./plot_all.sh --results results --out figures_reproduced
```

The manuscript's main text has **8 figures and 6 tables**.  Six figures and
two tables are generated by this pipeline:

| Output | Paper | Script |
| --- | --- | --- |
| `fig_fixed_accuracy`, `fig_fixed_frontier` | Figures 3–4 | `plot_chapter4.py` |
| `fig_bound_selectivity`, `fig_margin_sensitivity`, `fig_scalability` | Figures 5–7 | `plot_chapter4.py` |
| `fig_mixed_frontier` | Figure 8 | `plot_chapter4.py` |
| `tab_fixed_headline`, `tab_memory` | Tables 3–4 | `plot_chapter4.py` |
| `tab_full_spectrum_all`, `fig_bound_coverage`, `fig_shaping_gain_mse`, `fig_all_frontiers_grid` | supplementary material | `plot_chapter4.py`, `plot_bound_coverage.py`, `plot_appendix_spectrum.py` |

The remaining items: Figure 1 (architecture diagram) and Figure 2 (the
orthogonal-residual schematic) are hand-supplied drawings, not part of this
pipeline; the encoding-scale schedule is fixed by
`experiments/shaping/measure_scale_schedule.cpp`, and Table 5 is the
construction sweep of `experiments/table3_build.sh` — its cells are
wall-clock build seconds, so only the machine holding the corpora can
take them (the sweep writes its LaTeX/CSV bodies for a manual copy into
the paper).

## Verification

After step 3, `figures_reproduced/` holds every generated figure and table
(PDF + PNG, CSV + LaTeX), directly comparable against the submitted paper;
`plot_all.sh` exits non-zero if any figure the LaTeX includes is missing.
`python experiments/verify_abstract_claims.py <results-dir>` (the directory
`run_all.sh` wrote, default `./results`) re-derives the abstract's margin
numbers from the ledgers.

Pre-hand-off checks run on this package (2026-10-08, AVX2 host, Python
3.12): C++ build + `e8q_test` / `me8q_test` / `scan_kernel_test` pass; the
wheel builds and `pytest tests/python` reports 10 passed; the E8Q/ME8Q
runners complete on a generated corpus and write valid ledgers; the plot
pipeline ran end-to-end and reproduced the submitted figure files
pixel-identically; `verify_abstract_claims.py` re-derived the paper's
abstract margins (+83.4% CAQ, +51.0% RaBitQ at 90% recall; +2.56 pts
ceiling; GIST b=8 99.35%).

Known machine-dependence, recorded for honest reading:

- QPS columns are the reference host's; the paper's text ties throughput
  comparisons to it.  Re-running elsewhere reproduces recall/ratio numbers
  and the direction of throughput effects, not absolute QPS.
- The paper's $\alpha$-sweep ablation was measured on a different desktop
  (≈30% lower absolute QPS) — the paper reads it for its shape across
  $\alpha$, as Section 4.3 says.
- Table 3 is the reference server's construction times.

## CMake options

| Option | Default | Meaning |
| --- | --- | --- |
| `E8Q_ENABLE_NATIVE_OPTIMIZATION` | ON | `-march=native` on generic TUs |
| `E8Q_BUILD_TESTS` | ON | `e8q_test`, `me8q_test`, `scan_kernel_test` |
| `E8Q_BUILD_SAMPLES` | ON | `e8q_index`, `me8q_index` |
| `E8Q_BUILD_PYTHON_BINDINGS` | OFF | pybind11 module |
| `E8Q_ENABLE_SANITIZERS` | OFF | ASan/UBSan on the library and tests |

## License and third-party code

The e8qlib code is Apache-2.0 (`LICENSE`).  The vendored trees retain their
upstream licenses: `baselines/RaBitQ-Library/LICENSE` and
`baselines/SAQ/LICENSE` (both Apache-2.0), including their bundled Eigen
copies.  `faiss-cpu` is consumed as a wheel under its own license.
