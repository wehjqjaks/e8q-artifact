# Datasets

One folder per corpus, next to the two driver scripts:

```
datasets/
  prepare.sh            # environment bootstrap -> prepare.py
  prepare.py            # every dataset operation (prepare / refill)
  README.md
  <name>/
    <name>_base.fvecs
    <name>_query.fvecs
    <name>_groundtruth.ivecs
    <name>_centroid_4096.fvecs
    <name>_cluster_id_4096.ivecs
    <name>_pca_mean.fvecs
    <name>_pca_matrix.fvecs
    <name>_base_pca.fvecs
    <name>_query_pca.fvecs
    <name>_centroid_4096_pca.fvecs
    <name>_base_pca.vars.fvecs
```

Each folder holds **only the processed files** — raw downloads and archives
are staged under `<name>/raw/` while a corpus is being prepared and deleted
once its processed files are complete (`--keep-raw` preserves them for
debugging).

`prepare.sh` only finds a Python with numpy + faiss (conda env `e8q`, or
`E8Q_PYTHON=...`) and execs `prepare.py`; all flags below are `prepare.py`
flags. Default `--data-root` is this directory. Override only if you want
corpora elsewhere:

```bash
./datasets/prepare.sh msong               # -> datasets/msong/
./datasets/prepare.sh --all               # the 15 paper corpora
./datasets/prepare.sh msong --steps gt,ivf,pca   # derived artifacts only
./datasets/prepare.sh --list              # the corpus registry
./datasets/prepare.sh --refill --status   # what is missing from prepared trees
```

When huggingface.co is unreachable, point `HF_ENDPOINT` at a mirror:
`HF_ENDPOINT=https://hf-mirror.com ./datasets/prepare.sh --all`.

`experiments/run.py` and every experiment script read
`<data-root>/<name>/<name>_*.fvecs`; their default data-root is also
`datasets/`.

Corpora are gitignored (`datasets/*/`, `*.fvecs`, `*.ivecs`). The Python and
bash drivers stay tracked. A `datasets/<name>` symlink to a prepared copy is
used directly when valid, and replaced by a real directory when it dangles
(e.g. after cloning this repo on a server).

## The 15 paper corpora

| Key | n | dim | Source (`prepare.py --list`) |
| --- | --- | --- | --- |
| `sift1M` | 1,000,000 | 128 | texmex SIFT1M |
| `sift10m` | 10,000,000 | 128 | UCI archive 353 (`sift10m.zip`) |
| `deep1M` | 1,000,000 | 256 | GQR deep |
| `msong` | 1,000,000 | 420 | GQR msong |
| `word2vec` | 1,000,000 | 300 | GQR word2vec |
| `glove200` | 399,000 | 200 | Stanford GloVe 6B 200-d |
| `imagenet` | 1,385,000 | 150 | GQR imagenet |
| `gist` | 1,000,000 | 960 | GQR gist |
| `youtube` | 1,177,090 | 1779 | manual (below) |
| `vogue768` | 933,677 | 768 | Hugging Face parquet (JHQ recipe) |
| `arxiv-abstracts768` | 1,693,531 | 768 | Hugging Face parquet |
| `openai1536` | 999,000 | 1536 | Hugging Face parquet (Qdrant dbpedia) |
| `openai3072` | 1,000,000 | 3072 | Hugging Face parquet (Qdrant dbpedia) |
| `msmarco10M` | 10,000,000 | 1024 | Cohere passage shards, streamed |
| `stella-trec24-1024` | 17,732,000 | 1024 | Hugging Face parquet |

HF corpora follow the JHQ parquet recipe (L2-normalize, seeded split).
PCA is **full rank** (rotation, not reduction): `y = (x - mean) @ matrix.T`,
so one ground truth serves both the original and the rotated domain.
Do not download every dataset just to smoke-test the pipeline; `msong` or a
tiny generated fvecs file is enough.

## Special sources

* **`sift10m`** — UCI archive #353 (`sift10m.zip`, base/query fvecs). The
  server ignores Range, so a cut download restarts from scratch.
* **`youtube`** — 1779-d Fisher vectors; the GQR and ann-benchmarks mirrors
  are dead. Place `youtube_{base,query}.fvecs` in `datasets/youtube/`
  directly, then run `prepare.sh youtube --steps gt,ivf,pca`.
* **`msmarco10M`** — streams the public `CohereLabs/msmarco-v2.1-embed-english-v3`
  shards itself (SAQ recipe): the base is the first 10,000,000 passage
  embeddings in shard order (`--base-size`), queries are the corpus queries
  whose published `top1k_offsets` keep at least `--gt-k` neighbors inside
  that prefix, with the filtered offsets as ground truth. The recipe shares
  the common flags; its IVF trains on `--train-size` rows (default 1M for
  this corpus, 512k elsewhere).

## Refilling a partially lost tree

If a prepared tree lost only its large files (`*_base.fvecs`,
`*_base_pca.fvecs`) — e.g. pruned after an upload — `--refill` restores
exactly those: it re-fetches the base from the public source, verifies it
against the stored ground truth and IVF artifacts (shape, sampled exact
top-100, sampled centroid agreement), and rebuilds `*_base_pca.fvecs` from
the stored `pca_matrix`/`pca_mean` so the rotation stays consistent with
the stored `query_pca` / `centroid_pca`.

```bash
./datasets/prepare.sh --refill --status    # what is missing
./datasets/prepare.sh --refill             # refill everything missing
./datasets/prepare.sh --refill sift10m msong   # a subset
```

Only corpora whose files are actually missing are touched, downloads resume
with wget -c, and a failed verification leaves the partial download under
`datasets/<name>/raw/` for inspection. Safe to rerun.

## Prepared-corpora provenance

The paper's corpora were prepared by the SAQ preprocessing pipeline (GPU
k-means `nlist=4096`, full-rank CuPy PCA; per-dataset `*_ivf_4096.json` /
`*_pca.json` metadata record the run). `prepare.py` here reproduces the same
recipe on CPU (faiss k-means, numpy PCA), which is equivalent but not
bit-identical — k-means seeds differ, and E8Q-vs-RaBitQ recall margins on
low-dim corpora are sensitive to the assignment. If you maintain a copy of
the canonical (GPU-prepared) files, point any stage at it directly:

```bash
./experiments/exp_stable.sh /path/to/canonical/datasets <run-id>
```

**Format caveat:** despite the `.fvecs` extension, a few canonical files are
raw float32 with no per-row headers (the canonical `sift1m_*_pca.fvecs` and
`sift1m_centroid_4096_pca.fvecs`); the other corpora (e.g. vogue768) are true
fvecs. A `datasets/sift1M/` tree prepared by this repo's driver holds
research-parity files: real fvecs PCA regenerated from the canonical
`pca_matrix`/`pca_mean` (variance check 7e-8).
