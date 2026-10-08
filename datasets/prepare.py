#!/usr/bin/env python3
"""Download, preprocess, and repair the E8Q / ME8Q benchmark corpora.

One driver for every dataset operation the pipeline needs: preparing a
corpus, streaming the MSMARCO-10M corpus, and refilling the base files of
an already-prepared tree.

A prepared corpus lands in ``<data-root>/<name>/`` (default: this
directory) and holds the files the C++ bench and the SAQ-style mixed-rate
methods read:

    <name>_base.fvecs            float32 corpus
    <name>_query.fvecs           held-out queries
    <name>_groundtruth.ivecs     exact top-100 L2 neighbors
    <name>_centroid_4096.fvecs   shared IVF k-means centroids (all threads)
    <name>_cluster_id_4096.ivecs shared list assignment
    <name>_{pca_mean,pca_matrix,base_pca,query_pca,centroid_4096_pca,
            base_pca.vars}.fvecs  full-rank PCA artifacts (ME8Q / SAQ)

Full-rank PCA keeps squared-L2 distances, so one ground truth serves both
the original and the rotated domain.

Download artifacts and raw archives are staged under ``<name>/raw/`` while
a corpus is being prepared and deleted once its processed files are
complete (keep them with ``--keep-raw``); each corpus folder ends up
holding only the processed files listed above.

Usage:
    python prepare.py --list                 # the corpus registry
    python prepare.py msong sift             # a subset (aliases work)
    python prepare.py --all                  # every registry entry
    python prepare.py --refill --status      # what is missing, then exit
    python prepare.py --refill msong         # restore it from its source

Sources follow the two upstream pipelines the paper builds on: the GQR /
texmex archives and the UCI sift10m archive for the classic corpora, and
the Hugging Face parquet recipes of the JHQ collection (parquet shards,
L2-normalized, seeded split) for the embedding corpora.  MSMARCO-10M
streams the Cohere passage shards through the SAQ recipe; ``youtube`` has
no public mirror left, so its fvecs have to be placed by hand and the
derived artifacts built with ``--steps gt,ivf,pca`` (see README.md).

``--refill`` re-fetches only ``<name>_base.fvecs`` / ``<name>_base_pca.fvecs``
of a processed tree that lost them (e.g. pruned after an upload).  The
stored ground truth and IVF assignment pin the base corpus down exactly,
so the fresh copy is verified against them before installation, and the
rotated base is rebuilt from the stored ``pca_matrix``/``pca_mean`` —
never by re-fitting PCA — so the rotation stays consistent with the
stored ``query_pca`` / ``centroid_pca``.

When huggingface.co is unreachable, point HF_ENDPOINT at a mirror, e.g.
    HF_ENDPOINT=https://hf-mirror.com python prepare.py --all

Steps are idempotent: existing outputs are kept unless --force is given.
Everything in this directory is a driver, not data.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

try:  # optional accelerator; the numpy fallbacks below are always available
    import faiss  # type: ignore

    HAVE_FAISS = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_FAISS = False

HERE = Path(__file__).resolve().parent
# Streamed I/O block: one record row of this many vectors, 64k x d floats.
RECORD_BLOCK = 65536

GQR_URL = "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/{name}.tar.gz"
SIFT10M_URL = "https://archive.ics.uci.edu/static/public/353/sift10m.zip"
TEXMEX_URLS = [
    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
    "http://corpus-texmex.irisa.fr/local/texmex/corpus/sift.tar.gz",
]

# name -> (dim, source)
#   gqr     : CUHK GQR tarball, ships base/query/groundtruth
#   texmex  : SIFT1M (texmex layout)
#   sift10m : UCI archive #353 (sift10m.zip; no Range support, so a cut
#             download restarts from scratch)
#   manual  : no public mirror left; place <name>_{base,query}.fvecs by hand
#   glove   : Stanford GloVe 6B (200-d slice)
#   hf      : Hugging Face parquet (JHQ recipe: L2-normalize, seeded split)
#   msmarco : Cohere passage shards, streamed (SAQ recipe)
DATASETS: dict[str, tuple[int, str]] = {
    "msong": (420, "gqr"),
    "sift1M": (128, "texmex"),
    "sift10m": (128, "sift10m"),
    "deep1M": (256, "gqr"),
    "word2vec": (300, "gqr"),
    "gist": (960, "gqr"),
    "glove200": (200, "glove"),
    "imagenet": (150, "gqr"),
    "youtube": (1779, "manual"),
    "openai1536": (1536, "hf"),
    "openai3072": (3072, "hf"),
    "msmarco10M": (1024, "msmarco"),
    "vogue768": (768, "hf"),
    "arxiv-abstracts768": (768, "hf"),
    "stella-trec24-1024": (1024, "hf"),
}
ALIASES = {
    "sift": "sift1M",
    "deep": "deep1M",
    "image": "imagenet",
    "glove": "glove200",
    "openai-1536": "openai1536",
    "openai-3072": "openai3072",
    "msmarco": "msmarco10M",
    "vogue": "vogue768",
    "arxiv-abstracts": "arxiv-abstracts768",
    "stella-trec24": "stella-trec24-1024",
}

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_REPO = {
    "openai1536": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
    "openai3072": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
    "arxiv-abstracts768": "macrocosm/arxiv_abstracts",
    "stella-trec24-1024": "ielabgroup/stella_trec24_biogen_embedding",
    "vogue768": "tonyassi/vogue933k-embeddings",
}
HF_INCLUDE = {
    "openai1536": "*.parquet",
    "openai3072": "data/train-*.parquet",
    "arxiv-abstracts768": "abstracts_*.parquet",
    "stella-trec24-1024": "data/corpus-*.parquet",
    "vogue768": "data/train-*.parquet",
}
HF_EMBED_COL = {
    "openai1536": "text-embedding-3-large-1536-embedding",
    "openai3072": "text-embedding-3-large-3072-embedding",
}
HF_QUERY_SIZE = {name: 1000 for name in HF_REPO}
HF_BASE_SIZE = {"openai1536": 999_000}
# One seed for the HF query split and the IVF k-means, so every method and
# every rate of a corpus sees the same artifacts.
SPLIT_SEED = 20240911

# The Cohere org was renamed to CohereLabs on the Hub; resolve through the
# new name so a mirror does not have to follow the redirect per file.
MSMARCO_REPO = "CohereLabs/msmarco-v2.1-embed-english-v3"
MSMARCO_SHARD_FMT = "passages_npy/msmarco_v2.1_doc_segmented_{:02d}.npy"
MSMARCO_QUERY_FILE = "queries_parquet/queries.parquet"
MSMARCO_SHARDS = 60
MSMARCO_BASE = 10_000_000

# Refill assumes the stored artifacts are the paper's: nlist = 4096 IVF and
# the shared layout above.
REFILL_CLUSTERS = 4096
# Small corpora first, long-pole runners (sift10m's non-resumable UCI zip,
# the 10M-row streaming recipes) last.
REFILL_ORDER = [
    "msong", "imagenet", "gist",
    "vogue768", "arxiv-abstracts768", "openai1536", "openai3072",
    "sift10m", "msmarco10M", "stella-trec24-1024",
]
DOWNLOAD_WORKERS = 6


class VerifyFailed(RuntimeError):
    """A fetched file does not match the stored artifacts, or a source is
    unusable; the caller aborts that corpus and keeps going."""


# ---------------------------------------------------------------------------
# Logging and shell helpers.


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_url(repo: str, path: str) -> str:
    return f"{HF_ENDPOINT}/datasets/{repo}/resolve/main/{path}"


def train_size_for(kind: str, given: int | None) -> int:
    """IVF k-means training-set size; the streaming recipes use 1M rows."""
    if given:
        return given
    return 1_000_000 if kind == "msmarco" else 512_000


# ---------------------------------------------------------------------------
# fvecs / ivecs I/O (int32 dim header per record, texmex compatible).


def fvecs_shape(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    record = 4 + 4 * dim
    if dim <= 0 or size % record != 0:
        raise ValueError(f"{path}: not a whole number of {dim}-d fvecs records")
    return size // record, dim


def read_fvecs_block(path: Path, start_row: int, rows: int, dim: int) -> np.ndarray:
    """Read consecutive records, dropping the per-record dim headers."""
    rec = 4 + 4 * dim
    with path.open("rb") as f:
        f.seek(start_row * rec)
        buf = f.read(rows * rec)
    if len(buf) != rows * rec:
        raise ValueError(f"{path}: short read at row {start_row}")
    arr = np.frombuffer(buf, dtype=np.float32)
    return arr.reshape(rows, dim + 1)[:, 1:].copy()


def read_fvecs(path: Path) -> np.ndarray:
    n, dim = fvecs_shape(path)
    return read_fvecs_block(path, 0, n, dim)


def write_fvecs(path: Path, data: np.ndarray) -> None:
    data = np.ascontiguousarray(data, dtype=np.float32)
    n, dim = data.shape
    header = np.full((n, 1), dim, dtype=np.int32)
    np.hstack([header, data.view(np.int32)]).tofile(path)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:]


def write_ivecs(path: Path, data: np.ndarray) -> None:
    data = np.ascontiguousarray(data, dtype=np.int32)
    header = np.full((data.shape[0], 1), data.shape[1], dtype=np.int32)
    np.hstack([header, data]).tofile(path)


def _write_fvecs_with_header(fh, header: float, block: np.ndarray) -> None:
    """Append one fvecs record block (float32 rows prefixed by dim header)."""
    out = np.empty((len(block), block.shape[1] + 1), dtype=np.float32)
    out[:, 0] = header
    out[:, 1:] = block
    out.tofile(fh)


# ---------------------------------------------------------------------------
# Downloads.


def http_size(url: str) -> int | None:
    """Content-Length of url if the server reports one, else None."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001 - resume is best-effort
        return None


def wget(url: str, dest: Path, expect_size: int | None = None) -> None:
    """Download url to dest, resuming partial files (wget -c).

    expect_size, when known, keeps a partial file from being mistaken for
    a cached complete one.  --timeout bounds a silently dead connection
    (wget's default read timeout is 15 min); with -c, retries resume where
    the last one stopped instead of surfacing transient stalls as hard
    failures.  Without wget on PATH, a single-shot urllib copy is used.
    """
    if dest.exists() and dest.stat().st_size > 0:
        if expect_size is None or dest.stat().st_size == expect_size:
            log(f"  cached {dest}")
            return
        log(f"  resuming {dest} ({dest.stat().st_size}/{expect_size} bytes)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        run([
            "wget", "-c", "--timeout=60", "--tries=20", "--waitretry=10",
            url, "-O", str(dest),
        ])
    else:  # pragma: no cover - hosts without wget
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=16 * 1024 * 1024)
        tmp.replace(dest)
    if expect_size is not None and dest.stat().st_size != expect_size:
        raise OSError(
            f"{dest}: {dest.stat().st_size} bytes, expected {expect_size}"
        )


def hf_download(repo: str, include: str, dest: Path) -> None:
    """Fetch repo files matching include into dest via the hub tooling."""
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("hf"):
        run(
            [
                "hf", "download", repo, "--repo-type", "dataset",
                "--include", include, "--local-dir", str(dest),
            ]
        )
        return
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "need the `hf` CLI or `pip install huggingface_hub` for HF datasets"
        ) from exc
    snapshot_download(
        repo_id=repo, repo_type="dataset", allow_patterns=[include], local_dir=str(dest)
    )


def hf_tree(repo: str) -> dict[str, int]:
    """repo-relative path -> size in bytes, for every file in the repo."""
    url = f"{HF_ENDPOINT}/api/datasets/{repo}/tree/main?recursive=1"
    with urllib.request.urlopen(url, timeout=120) as resp:
        entries = json.loads(resp.read())
    if len(entries) >= 1000:  # pagination guard; not expected for our repos
        raise VerifyFailed(f"{repo}: tree listing truncated at 1000 entries")
    return {
        e["path"]: int(e.get("size") or 0)
        for e in entries
        if e.get("type") == "file"
    }


def hf_download_files(
    repo: str,
    files: list[tuple[str, int]],
    dest: Path,
    workers: int = DOWNLOAD_WORKERS,
) -> None:
    """Parallel, size-checked fetch of specific repo files (with retries)."""
    dest.mkdir(parents=True, exist_ok=True)

    def one(item: tuple[str, int]) -> str | None:
        path, size = item
        out = dest / path
        if out.exists() and size > 0 and out.stat().st_size == size:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        url = resolve_url(repo, path)
        proc = subprocess.run(
            [
                "wget", "-c", "-q",
                "--tries=5", "--timeout=60", "--waitretry=15",
                url, "-O", str(out),
            ]
        )
        if proc.returncode != 0:
            return f"{path}: wget rc={proc.returncode}"
        if size > 0 and out.stat().st_size != size:
            return f"{path}: {out.stat().st_size} != {size} bytes"
        return None

    for round_no in range(4):
        failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            for res in pool.map(one, files):
                if res:
                    failures.append(res)
        if not failures:
            return
        log(
            f"{repo}: {len(failures)} file(s) failed (round {round_no + 1}): "
            f"{failures[:3]}{'...' if len(failures) > 3 else ''}"
        )
    raise VerifyFailed(f"{repo}: unfinished downloads after 4 rounds")


# ---------------------------------------------------------------------------
# Ground truth: exact top-k squared L2.


def exact_top_k(base_path: Path, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k squared-L2 ids per query row, streamed over the base.

    The distance matrix is accumulated in 64-bit: the per-block broadcast
    of ``||x||^2`` alone can exceed float32 range on large corpora.
    """
    n, dim = fvecs_shape(base_path)
    k = min(k, n)
    Q = np.ascontiguousarray(queries, dtype=np.float64)
    m = len(Q)
    q2 = (Q * Q).sum(axis=1)
    best_d = np.full((m, k), np.inf)
    best_i = np.zeros((m, k), dtype=np.int64)
    for b0 in range(0, n, RECORD_BLOCK):
        bn = min(RECORD_BLOCK, n - b0)
        x = read_fvecs_block(base_path, b0, bn, dim)
        d = (x.astype(np.float64) ** 2).sum(axis=1)[None, :] - 2.0 * (Q @ x.T)
        d += q2[:, None]
        cand_d = np.concatenate([best_d, d], axis=1)
        cand_i = np.concatenate(
            [best_i, np.broadcast_to(np.arange(b0, b0 + bn), (m, bn))], axis=1
        )
        part = np.argpartition(cand_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(cand_d, part, axis=1)
        best_i = np.take_along_axis(cand_i, part, axis=1)
    order = np.argsort(best_d, axis=1)
    return np.take_along_axis(best_i, order, axis=1)


def compute_ground_truth(
    base_path: Path, query_path: Path, out_path: Path, k: int = 100
) -> None:
    n, dim = fvecs_shape(base_path)
    nq, qdim = fvecs_shape(query_path)
    if qdim != dim:
        raise ValueError(f"{query_path}: dim {qdim} != base dim {dim}")
    if HAVE_FAISS:
        index = faiss.IndexFlatL2(dim)
        index.add(read_fvecs(base_path))
        _, gt = index.search(read_fvecs(query_path), k)
        write_ivecs(out_path, gt.astype(np.int32))
        return
    # Numpy fallback: one query batch at a time, so each streamed top-k
    # keeps a bounded (batch x block) distance matrix.
    out_gt = np.empty((nq, k), dtype=np.int32)
    t0 = time.time()
    for q0 in range(0, nq, 32):
        qc = min(32, nq - q0)
        q = read_fvecs_block(query_path, q0, qc, dim)
        out_gt[q0 : q0 + qc] = exact_top_k(base_path, q, k)
        log(f"  gt: {q0 + qc}/{nq} queries ({time.time() - t0:.0f}s)")
    write_ivecs(out_path, out_gt)


# ---------------------------------------------------------------------------
# IVF: shared k-means centroids + assignment (identical across codecs).


def sample_rows(base_path: Path, rows: np.ndarray, dim: int) -> np.ndarray:
    """Gather strided training rows in contiguous block reads."""
    out = np.empty((len(rows), dim), dtype=np.float32)
    i = 0
    while i < len(rows):
        j = i
        while (
            j + 1 < len(rows)
            and rows[j + 1] == rows[j] + 1
            and (rows[j + 1] - rows[i]) < RECORD_BLOCK
        ):
            j += 1
        out[i : j + 1] = read_fvecs_block(
            base_path, int(rows[i]), int(rows[j] - rows[i] + 1), dim
        )
        i = j + 1
    return out


def _kmeans_numpy(
    x: np.ndarray, k: int, iters: int, seed: int, verbose: bool
) -> np.ndarray:
    """k-means++ seeding + Lloyd iterations, faiss-free fallback."""
    rng = np.random.default_rng(seed)
    sub = x[rng.choice(len(x), size=min(len(x), 32 * k), replace=False)]
    cent = np.empty((k, x.shape[1]), dtype=np.float32)
    cent[0] = sub[rng.integers(len(sub))]
    closest = ((sub - cent[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        p = closest / max(closest.sum(), 1e-12)
        cent[i] = sub[rng.choice(len(sub), p=p)]
        closest = np.minimum(closest, ((sub - cent[i]) ** 2).sum(axis=1))
    x = np.ascontiguousarray(x, dtype=np.float32)
    cent = cent.astype(np.float32, copy=False)
    c2 = (cent * cent).sum(axis=1)
    x2 = (x * x).sum(axis=1)
    for it in range(iters):
        sums = np.zeros_like(cent)
        counts = np.zeros(k, dtype=np.int64)
        for b0 in range(0, len(x), RECORD_BLOCK):
            xb = x[b0 : b0 + RECORD_BLOCK]
            d = (xb @ cent.T) * np.float32(-2.0)
            d += c2[None, :]
            d += x2[b0 : b0 + RECORD_BLOCK, None]
            a = d.argmin(axis=1)
            np.add.at(sums, a, xb)
            np.add.at(counts, a, 1)
        empty = counts == 0
        counts[empty] = 1
        cent = sums / counts[:, None]
        if empty.any():
            cent[empty] = x[rng.choice(len(x), size=int(empty.sum()), replace=False)]
        if verbose:
            log(f"  kmeans iter {it + 1}/{iters}")
    return cent


def train_ivf(
    base_path: Path,
    out_centroid: Path,
    clusters: int,
    train_size: int = 512_000,
    iters: int = 20,
    seed: int = SPLIT_SEED,
) -> None:
    """Train shared k-means centroids on a deterministic strided sample."""
    n, dim = fvecs_shape(base_path)
    take = min(train_size, n)
    stride = max(1, n // take)
    rows = np.arange(0, n, stride)[:take]
    x = sample_rows(base_path, rows, dim)
    if HAVE_FAISS:
        kmeans = faiss.Kmeans(dim, clusters, niter=iters, seed=seed, verbose=True)
        kmeans.train(np.ascontiguousarray(x))
        cent = kmeans.centroids.astype(np.float32)
    else:
        log(f"  faiss unavailable; numpy k-means on {len(x)} x {dim}, k={clusters}")
        cent = _kmeans_numpy(x, clusters, iters, seed, verbose=True)
    write_fvecs(out_centroid, cent)


def assign_ivf(base_path: Path, centroid_path: Path, out_ids: Path) -> None:
    """Assign every base vector to its nearest shared centroid."""
    n, dim = fvecs_shape(base_path)
    cent = read_fvecs(centroid_path)
    ids = np.empty(n, dtype=np.int32)
    if HAVE_FAISS:
        quantizer = faiss.IndexFlatL2(dim)
        quantizer.add(np.ascontiguousarray(cent))
        for b0 in range(0, n, RECORD_BLOCK):
            bn = min(RECORD_BLOCK, n - b0)
            _, found = quantizer.search(read_fvecs_block(base_path, b0, bn, dim), 1)
            ids[b0 : b0 + bn] = found[:, 0]
    else:
        c2 = (cent.astype(np.float64) ** 2).sum(axis=1)
        for b0 in range(0, n, RECORD_BLOCK):
            bn = min(RECORD_BLOCK, n - b0)
            x = read_fvecs_block(base_path, b0, bn, dim)
            d = (x.astype(np.float64) ** 2).sum(axis=1)[:, None] - 2.0 * (
                x @ cent.T
            ) + c2[None, :]
            ids[b0 : b0 + bn] = d.argmin(axis=1)
            if b0 % (RECORD_BLOCK * 16) == 0:
                log(f"  assign: {b0 + bn}/{n}")
    write_ivecs(out_ids, ids[:, None])


# ---------------------------------------------------------------------------
# PCA (SAQ preprocessing): full-rank rotation, y = (x - mean) @ evecs^T.


def compute_pca(
    base_path: Path,
    query_path: Path,
    centroid_path: Path | None,
    ddir: Path,
    ds: str,
    clusters: int,
) -> None:
    n, dim = fvecs_shape(base_path)
    mu = np.zeros(dim, dtype=np.float64)
    cov = np.zeros((dim, dim), dtype=np.float64)
    for b0 in range(0, n, RECORD_BLOCK):
        bn = min(RECORD_BLOCK, n - b0)
        x = read_fvecs_block(base_path, b0, bn, dim).astype(np.float64)
        mu += x.sum(axis=0)
        cov += x.T @ x
    mu /= n
    cov = cov / n - np.outer(mu, mu)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    evals = evals[::-1].clip(min=0)
    evecs = evecs[:, ::-1]  # descending; row i = eigenvector i
    write_fvecs(ddir / f"{ds}_pca_mean.fvecs", mu[None, :].astype(np.float32))
    write_fvecs(ddir / f"{ds}_pca_matrix.fvecs", evecs.T.astype(np.float32))
    write_fvecs(ddir / f"{ds}_base_pca.vars.fvecs", evals[None, :].astype(np.float32))

    header = np.array([dim], dtype=np.int32).view(np.float32)[0]

    def transform(src: Path, dst: Path, rows: int) -> None:
        # Stream headered fvecs records (int32 dim + float32 row), matching
        # every texmex-layout reader; whole-file loads are avoided.
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as fh:
            for b0 in range(0, rows, RECORD_BLOCK):
                bn = min(RECORD_BLOCK, rows - b0)
                x = read_fvecs_block(src, b0, bn, dim).astype(np.float64)
                _write_fvecs_with_header(fh, header, ((x - mu) @ evecs).astype(np.float32))
        tmp.replace(dst)

    transform(base_path, ddir / f"{ds}_base_pca.fvecs", n)
    nq, _ = fvecs_shape(query_path)
    transform(query_path, ddir / f"{ds}_query_pca.fvecs", nq)
    if centroid_path is not None and centroid_path.exists():
        nc, _ = fvecs_shape(centroid_path)
        transform(centroid_path, ddir / f"{ds}_centroid_{clusters}_pca.fvecs", nc)


# ---------------------------------------------------------------------------
# Preparing a corpus: download -> convert -> gt -> ivf -> pca.


def extract_into(tarball: Path, ddir: Path, ds: str) -> None:
    """Unpack a corpus tarball, renaming every archive member to <ds>_*."""
    with tempfile.TemporaryDirectory() as td:
        run(["tar", "-xzf", str(tarball), "-C", td])
        for p in list(Path(td).rglob("*.fvecs")) + list(Path(td).rglob("*.ivecs")):
            dst_name = p.name
            for tail in ("_base.fvecs", "_query.fvecs", "_groundtruth.ivecs",
                         "_learn.fvecs"):
                # GQR ships some archives as deep_* instead of deep1M_*.
                if dst_name.endswith(tail):
                    dst_name = ds + tail
                    break
            shutil.move(str(p), ddir / dst_name)


def download(ds: str, ddir: Path, raw: Path) -> None:
    dim, kind = DATASETS[ds]
    log(f"[{ds}] download ({kind})")
    if kind == "gqr":
        tarball = raw / f"{ds}.tar.gz"
        wget(GQR_URL.format(name=ds), tarball)
        extract_into(tarball, ddir, ds)
    elif kind == "texmex":
        tarball = raw / "sift1m-texmex.tar.gz"
        if not tarball.exists() or tarball.stat().st_size == 0:
            last: Exception | None = None
            for url in TEXMEX_URLS:
                try:
                    wget(url, tarball)
                    last = None
                    break
                except Exception as exc:  # noqa: BLE001 - try the next mirror
                    last = exc
            if last is not None:
                raise SystemExit(f"sift1M: all mirrors failed ({last})")
        with tempfile.TemporaryDirectory() as td:
            run(["tar", "-xzf", str(tarball), "-C", td])
            for p in Path(td).rglob("*"):
                if not p.is_file():
                    continue
                name = p.name.replace("sift_", "sift1M_")
                if "learn" in name:
                    continue
                shutil.move(str(p), ddir / name)
    elif kind == "sift10m":
        # UCI archive #353 ships the circulated 10M-SIFT corpus as a zip
        # (base/query fvecs). The server ignores Range, so a truncated
        # download has to be restarted from scratch.
        zpath = raw / "sift10m.zip"
        wget(SIFT10M_URL, zpath)
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(td)
            for p in list(Path(td).rglob("*.fvecs")) + list(
                Path(td).rglob("*.ivecs")
            ):
                dst_name = p.name
                for tail in ("_base.fvecs", "_query.fvecs",
                             "_groundtruth.ivecs", "_learn.fvecs"):
                    if dst_name.endswith(tail) and not dst_name.startswith(ds + "_"):
                        dst_name = ds + tail
                        break
                if "learn" in dst_name:
                    continue
                shutil.move(str(p), ddir / dst_name)
    elif kind == "manual":
        raise SystemExit(
            f"[{ds}] no public mirror remains (the GQR and ann-benchmarks "
            "links are dead). Place {ds}_base.fvecs and {ds}_query.fvecs in "
            f"{ddir} manually, then rerun with --steps gt,ivf,pca."
        )
    elif kind == "glove":
        wget("https://nlp.stanford.edu/data/glove.6B.zip", raw / "glove.6B.zip")
    elif kind == "hf":
        hf_download(HF_REPO[ds], HF_INCLUDE[ds], raw)
    elif kind == "msmarco":
        pass  # streamed by prepare_msmarco(), not through raw staging


def convert_hf(
    ds: str, raw: Path, base_path: Path, query_path: Path, want: int | None
) -> None:
    """Stream HF parquet shards into base/query (JHQ recipe, bounded memory).

    One pass over the shards in sorted order: L2-normalize each batch,
    hold out ``min(HF_QUERY_SIZE, rows // 100)`` seeded query rows, and
    write the rest to base_path as fvecs.  The refill path regenerates the
    base through this same function, so the two stay byte-compatible.
    """
    import pyarrow.parquet as pq  # type: ignore

    dim = DATASETS[ds][0]
    pq_files = sorted(raw.rglob("*.parquet"))
    if not pq_files:
        raise SystemExit(f"[{ds}] no parquet under {raw}; run download first")
    total = 0
    for p in pq_files:
        total += pq.ParquetFile(p).metadata.num_rows
        if want is not None and total >= want:
            total = want
            break
    nq = min(HF_QUERY_SIZE[ds], total // 100)
    rng = np.random.default_rng(SPLIT_SEED)
    qidx = np.sort(rng.choice(total, size=nq, replace=False))
    log(f"[{ds}] convert: {total} rows, held-out {nq} queries (seeded split)")

    header = np.array([dim], dtype=np.int32).view(np.float32)[0]
    tmp = base_path.parent / (base_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    query_rows: list[np.ndarray] = []
    pos = qi = 0
    next_log = 2_000_000
    t0 = time.time()
    try:
        with tmp.open("wb") as fh:
            for p in pq_files:
                pf = pq.ParquetFile(p)
                col = HF_EMBED_COL.get(ds)
                if col is None:
                    for field in pf.schema_arrow:
                        if str(field.type).startswith("list") or "fixed" in str(field.type):
                            col = field.name
                            break
                for batch in pf.iter_batches(batch_size=8192, columns=[col]):
                    arr = np.stack(
                        batch.column(0).to_numpy(zero_copy_only=False)
                    ).astype(np.float32)
                    if arr.shape[1] != dim:
                        raise VerifyFailed(
                            f"[{ds}] {p.name}: embedding dim {arr.shape[1]} != {dim}"
                        )
                    arr /= np.maximum(
                        np.linalg.norm(arr, axis=1, keepdims=True), 1e-12
                    )
                    if want is not None and pos + len(arr) > want:
                        arr = arr[: want - pos]
                    m = len(arr)
                    q0 = qi
                    while qi < nq and qidx[qi] < pos + m:
                        qi += 1
                    if qi > q0:
                        loc = qidx[q0:qi] - pos
                        query_rows.append(arr[loc].copy())
                        keep = np.ones(m, dtype=bool)
                        keep[loc] = False
                        block = arr[keep]
                    else:
                        block = arr
                    if len(block):
                        _write_fvecs_with_header(fh, header, block)
                    pos += m
                    if pos >= next_log:
                        log(
                            f"  [{ds}] converted {pos}/{total} rows "
                            f"({time.time() - t0:.0f}s)"
                        )
                        next_log = pos + 2_000_000
                    if want is not None and pos >= want:
                        break
                if want is not None and pos >= want:
                    break
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if pos != total or qi != nq:
        raise VerifyFailed(f"[{ds}] streamed {pos}/{total} rows, {qi}/{nq} queries")
    tmp.replace(base_path)
    query = np.vstack(query_rows) if query_rows else np.empty((0, dim), np.float32)
    write_fvecs(query_path, query)


def convert(ds: str, ddir: Path, raw: Path, force: bool, max_rows: int | None) -> None:
    dim, kind = DATASETS[ds]
    base = ddir / f"{ds}_base.fvecs"
    query = ddir / f"{ds}_query.fvecs"
    if base.exists() and query.exists() and not force:
        log(f"[{ds}] base/query exist; skipping convert")
        return
    if kind == "glove":
        log(f"[{ds}] convert (GloVe 6B 200-d)")
        with zipfile.ZipFile(raw / "glove.6B.zip") as zf:
            with zf.open("glove.6B.200d.txt") as fh:
                vecs = np.array(
                    [line.split()[1:] for line in fh.read().decode().splitlines()],
                    dtype=np.float32,
                )
        nq = 1000
        write_fvecs(base, vecs[:-nq])
        write_fvecs(query, vecs[-nq:])
    elif kind == "hf":
        convert_hf(ds, raw, base, query, max_rows or HF_BASE_SIZE.get(ds))
    else:
        log(f"[{ds}] no converter for kind {kind}; using the downloaded files")


def ensure_corpus_dir(root: Path, ds: str) -> Path:
    """Return the working corpus dir, replacing a dangling or foreign symlink
    (e.g. one pointing at a checkout-local canonical tree that does not exist
    on this machine) with a real directory."""
    ddir = root / ds
    if ddir.is_symlink():
        if ddir.resolve().is_dir() and (ddir / f"{ds}_base.fvecs").exists():
            log(f"[{ds}] using prepared symlink {ddir} -> {ddir.resolve()}")
            return ddir
        log(f"[{ds}] replacing unusable symlink {ddir}")
        ddir.unlink()
    ddir.mkdir(parents=True, exist_ok=True)
    return ddir


def prepare(ds: str, root: Path, args) -> None:
    dim, kind = DATASETS[ds]
    ddir = ensure_corpus_dir(root, ds)
    if kind == "msmarco":
        prepare_msmarco(ds, ddir, args)
        return
    base = ddir / f"{ds}_base.fvecs"
    query = ddir / f"{ds}_query.fvecs"
    gt = ddir / f"{ds}_groundtruth.ivecs"
    cent = ddir / f"{ds}_centroid_{args.clusters}.fvecs"
    cids = ddir / f"{ds}_cluster_id_{args.clusters}.ivecs"
    raw = ddir / "raw"

    if "download" in args.steps:
        download(ds, ddir, raw)
    if "convert" in args.steps:
        convert(ds, ddir, raw, args.force, args.max_rows)
    if "gt" in args.steps and base.exists() and query.exists() and (
        not gt.exists() or args.force
    ):
        log(f"[{ds}] ground truth (top-{args.gt_k})")
        compute_ground_truth(base, query, gt, k=args.gt_k)
    if "ivf" in args.steps and base.exists() and (
        not cent.exists() or not cids.exists() or args.force
    ):
        log(f"[{ds}] ivf k={args.clusters} (shared by every method)")
        train_ivf(base, cent, args.clusters,
                  train_size_for(kind, args.train_size), args.iterations, args.seed)
        assign_ivf(base, cent, cids)
    if "pca" in args.steps and base.exists() and not (
        ddir / f"{ds}_base_pca.fvecs"
    ).exists():
        log(f"[{ds}] pca (full-rank rotation)")
        compute_pca(base, query, cent if cent.exists() else None, ddir, ds,
                    args.clusters)

    # The corpus folder holds only processed files: raw staging is removed
    # once every artifact exists. --keep-raw preserves it for debugging.
    artifacts = [base, query, gt, cent, cids,
                 ddir / f"{ds}_pca_mean.fvecs", ddir / f"{ds}_pca_matrix.fvecs",
                 ddir / f"{ds}_base_pca.fvecs", ddir / f"{ds}_query_pca.fvecs",
                 ddir / f"{ds}_centroid_{args.clusters}_pca.fvecs"]
    if raw.exists() and all(p.exists() for p in artifacts) and not args.keep_raw:
        log(f"[{ds}] removing raw staging {raw}")
        shutil.rmtree(raw)


# ---------------------------------------------------------------------------
# MSMARCO-10M: streaming recipe over the Cohere passage shards.


def npy_rows(url: str) -> int:
    """Read (row count) from the npy header via a range request."""
    import io

    req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        head = resp.read()
    buf = io.BytesIO(head)
    version = np.lib.format.read_magic(buf)
    if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(buf)
    elif version == (2, 0):
        shape, _, _ = np.lib.format.read_array_header_2_0(buf)
    else:  # pragma: no cover
        raise VerifyFailed(f"unsupported npy version {version} at {url}")
    return int(shape[0])


def msmarco_fetch_shards(raw_root: Path, base_size: int) -> list[str]:
    """Download the npy shards covering the base_size-row prefix (+queries).

    Shard row counts come from the npy headers, so only the shards that
    actually fall inside the prefix are fetched.
    """
    rows = 0
    names: list[str] = []
    for shard in range(MSMARCO_SHARDS):
        name = MSMARCO_SHARD_FMT.format(shard)
        rows += npy_rows(resolve_url(MSMARCO_REPO, name))
        names.append(name)
        if rows >= base_size:
            break
    if rows < base_size:
        raise VerifyFailed(f"msmarco: shards hold {rows} rows < {base_size}")
    log(
        f"msmarco: {len(names)} npy shard(s) cover the {base_size}-row prefix "
        f"({rows} rows available)"
    )
    tree = hf_tree(MSMARCO_REPO)
    files = [(n, tree.get(n, 0)) for n in names]
    files.append((MSMARCO_QUERY_FILE, tree.get(MSMARCO_QUERY_FILE, 0)))
    hf_download_files(MSMARCO_REPO, files, raw_root)
    return names


def materialize_base(
    ddir: Path, ds: str, raw_root: Path, names: list[str], base_size: int
) -> Path:
    """Write the shard prefix into <ds>_base.fvecs (streamed, atomic)."""
    dim = DATASETS[ds][0]
    fvecs_path = ddir / f"{ds}_base.fvecs"
    if fvecs_path.exists():
        return fvecs_path
    header = np.array([dim], dtype=np.int32).view(np.float32)[0]
    tmp = ddir / (fvecs_path.name + ".tmp")
    pos = 0
    try:
        with tmp.open("wb") as fh:
            for name in names:
                arr = np.load(raw_root / name, mmap_mode="r")
                take = min(base_size - pos, int(arr.shape[0]))
                for begin in range(0, take, 50_000):
                    end = min(begin + 50_000, take)
                    block = np.asarray(arr[begin:end], dtype=np.float32)
                    _write_fvecs_with_header(fh, header, block)
                pos += take
                log(f"  [{ds}] copied {name} rows={take} total={pos}")
                if pos >= base_size:
                    break
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if pos != base_size:
        tmp.unlink(missing_ok=True)
        raise VerifyFailed(f"[{ds}] materialized {pos} rows, expected {base_size}")
    tmp.replace(fvecs_path)
    return fvecs_path


def prepare_msmarco_queries(
    ddir: Path, ds: str, raw_root: Path, base_size: int, gt_k: int
) -> None:
    """Query embeddings + published top1k offsets -> query/gt pair.

    Queries keep only the neighbors that fall inside the base prefix;
    those whose filtered offsets drop below gt_k are skipped.
    """
    import pyarrow.parquet as pq  # type: ignore

    queries_parquet = raw_root / MSMARCO_QUERY_FILE
    if not queries_parquet.exists():
        raise VerifyFailed(f"[{ds}] missing {queries_parquet}; run download first")
    table = pq.read_table(queries_parquet, columns=["_id", "emb", "top1k_offsets"])
    ids = table["_id"].to_pylist()
    embeddings = table["emb"].to_pylist()
    offsets = table["top1k_offsets"].to_pylist()
    selected_ids: list[str] = []
    selected_queries: list[np.ndarray] = []
    selected_gt: list[list[int]] = []
    for qid, emb, off in zip(ids, embeddings, offsets):
        filtered = [int(x) for x in off if 0 <= int(x) < base_size]
        if len(filtered) >= gt_k:
            selected_ids.append(str(qid))
            selected_queries.append(np.asarray(emb, dtype=np.float32))
            selected_gt.append(filtered[:gt_k])
    if not selected_queries:
        raise VerifyFailed(f"[{ds}] no queries keep >= {gt_k} neighbors in the prefix")
    write_fvecs(
        ddir / f"{ds}_query.fvecs",
        np.vstack(selected_queries).astype(np.float32, copy=False),
    )
    write_ivecs(ddir / f"{ds}_groundtruth.ivecs", np.asarray(selected_gt, dtype=np.int32))
    (ddir / f"{ds}_query_ids.json").write_text(json.dumps(selected_ids) + "\n")
    log(f"[{ds}] selected {len(selected_ids)}/{len(ids)} queries")


def msmarco_ivf_and_pca(ddir: Path, ds: str, args) -> None:
    """Shared-layout IVF + full-rank faiss PCA over the streamed base."""
    dim = DATASETS[ds][0]
    base = ddir / f"{ds}_base.fvecs"
    n, _ = fvecs_shape(base)
    k = args.clusters
    cent_path = ddir / f"{ds}_centroid_{k}.fvecs"
    cid_path = ddir / f"{ds}_cluster_id_{k}.ivecs"
    if args.force or not (cent_path.exists() and cid_path.exists()):
        train = read_fvecs_block(
            base, 0, min(train_size_for("msmarco", args.train_size), n), dim
        )
        index = faiss.index_factory(dim, f"IVF{k},Flat")
        index.verbose = True
        index.train(train)
        write_fvecs(cent_path, index.quantizer.reconstruct_n(0, index.nlist))
        tmp = ddir / (cid_path.name + ".tmp")
        with tmp.open("wb") as fh:
            for begin in range(0, n, 100_000):
                end = min(begin + 100_000, n)
                _, found = index.quantizer.search(
                    read_fvecs_block(base, begin, end - begin, dim), 1
                )
                out = np.empty((end - begin, 2), dtype=np.int32)
                out[:, 0] = 1
                out[:, 1] = found[:, 0]
                out.tofile(fh)
                log(f"  [{ds}] assigned {end}/{n}")
        tmp.replace(cid_path)
    base_pca_path = ddir / f"{ds}_base_pca.fvecs"
    if args.force or not base_pca_path.exists():
        pca = faiss.PCAMatrix(dim, dim)
        pca.train(read_fvecs_block(base, 0, min(300_000, n), dim))
        write_fvecs(ddir / f"{ds}_pca_mean.fvecs",
                    faiss.vector_to_array(pca.mean).reshape(1, dim))
        write_fvecs(ddir / f"{ds}_pca_matrix.fvecs",
                    faiss.vector_to_array(pca.PCAMat).reshape(dim, dim))
        sums = np.zeros(dim, dtype=np.float64)
        sums_sq = np.zeros(dim, dtype=np.float64)
        header = np.array([dim], dtype=np.int32).view(np.float32)[0]
        tmp = ddir / (base_pca_path.name + ".tmp")
        with tmp.open("wb") as fh:
            for begin in range(0, n, 50_000):
                end = min(begin + 50_000, n)
                y = pca.apply(
                    read_fvecs_block(base, begin, end - begin, dim)
                ).astype(np.float32)
                sums += y.sum(axis=0, dtype=np.float64)
                sums_sq += np.square(y, dtype=np.float64).sum(axis=0)
                _write_fvecs_with_header(fh, header, y)
        tmp.replace(base_pca_path)
        variance = sums_sq / n - (sums / n) ** 2
        write_fvecs(ddir / f"{ds}_base_pca.vars.fvecs",
                    variance.reshape(1, -1).astype(np.float32))
        nq, _ = fvecs_shape(ddir / f"{ds}_query.fvecs")
        query = read_fvecs_block(ddir / f"{ds}_query.fvecs", 0, nq, dim)
        write_fvecs(ddir / f"{ds}_query_pca.fvecs",
                    pca.apply(np.ascontiguousarray(query)))
        cent = read_fvecs_block(cent_path, 0, k, dim)
        write_fvecs(ddir / f"{ds}_centroid_{k}_pca.fvecs",
                    pca.apply(np.ascontiguousarray(cent)))


def prepare_msmarco(ds: str, ddir: Path, args) -> None:
    """Full MSMARCO-10M pipeline: shards -> base -> queries -> ivf/pca.

    The heavy passes stream through shard memmaps so the 10M x 1024 matrix
    is never held as one in-RAM ndarray (the refill path reuses the same
    fetchers and materializer).
    """
    if not HAVE_FAISS:
        raise SystemExit(
            "[msmarco10M] needs faiss-cpu and pyarrow; install them or use "
            "another corpus"
        )
    raw_root = ddir / "raw" / MSMARCO_REPO.replace("/", "_")
    base = ddir / f"{ds}_base.fvecs"
    query = ddir / f"{ds}_query.fvecs"
    if "download" in args.steps or "convert" in args.steps:
        names = msmarco_fetch_shards(raw_root, args.base_size)
        materialize_base(ddir, ds, raw_root, names, args.base_size)
    if ("gt" in args.steps and not query.exists()) or args.force:
        prepare_msmarco_queries(ddir, ds, raw_root, args.base_size, args.gt_k)
    if ("ivf" in args.steps or "pca" in args.steps) and base.exists():
        msmarco_ivf_and_pca(ddir, ds, args)
    # The corpus folder holds only processed files: raw shards (~34 GB) are
    # staging, removed once every artifact exists (--keep-raw preserves
    # them for debugging).
    k = args.clusters
    artifacts = [base, query, ddir / f"{ds}_groundtruth.ivecs",
                 ddir / f"{ds}_centroid_{k}.fvecs", ddir / f"{ds}_cluster_id_{k}.ivecs",
                 ddir / f"{ds}_pca_mean.fvecs", ddir / f"{ds}_pca_matrix.fvecs",
                 ddir / f"{ds}_base_pca.fvecs", ddir / f"{ds}_query_pca.fvecs",
                 ddir / f"{ds}_centroid_{k}_pca.fvecs"]
    if (raw_root.parent.exists() and not args.keep_raw
            and all(p.exists() for p in artifacts)):
        log(f"[{ds}] removing raw staging {raw_root.parent}")
        shutil.rmtree(raw_root.parent)


# ---------------------------------------------------------------------------
# Refilling a partially lost tree: re-fetch the base, verify it against the
# stored artifacts, rebuild the rotated base from the stored PCA basis.


def ddir_of(root: Path, ds: str) -> Path:
    return root / ds


def expected_rows(root: Path, ds: str) -> int:
    """Base row count implied by the stored IVF assignment (one id per row)."""
    cid = ddir_of(root, ds) / f"{ds}_cluster_id_{REFILL_CLUSTERS}.ivecs"
    if not cid.exists():
        raise VerifyFailed(f"{ds}: {cid.name} missing; cannot pin the base size")
    return cid.stat().st_size // 8


def expected_base_bytes(root: Path, ds: str) -> int:
    dim, _ = DATASETS[ds]
    return expected_rows(root, ds) * (4 + 4 * dim)


def missing_parts(root: Path, ds: str) -> list[str]:
    ddir = ddir_of(root, ds)
    parts = []
    if not (ddir / f"{ds}_base.fvecs").exists():
        parts.append("base")
    if not (ddir / f"{ds}_base_pca.fvecs").exists():
        parts.append("base_pca")
    return parts


def verify_shape(root: Path, ds: str, base_path: Path) -> None:
    n, dim = fvecs_shape(base_path)
    exp_n, exp_dim = expected_rows(root, ds), DATASETS[ds][0]
    if (n, dim) != (exp_n, exp_dim):
        raise VerifyFailed(
            f"{ds}: base is {n}x{dim}, stored artifacts imply {exp_n}x{exp_dim}"
        )
    log(f"{ds}: base shape {n} x {dim} matches the stored artifacts")


def compare_arrays(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    """(bit-exact, max abs diff) for two fvecs/ivecs arrays."""
    if a.shape != b.shape:
        return False, float("inf")
    if a.size == 0:
        return True, 0.0
    diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return bool(np.array_equal(a, b)), diff


def check_assignment(root: Path, ds: str, base_path: Path, sample: int = 256) -> None:
    """Sampled rows must sit closest to the centroid the IVF file recorded."""
    ddir = ddir_of(root, ds)
    dim = DATASETS[ds][0]
    cent = np.ascontiguousarray(
        read_fvecs(ddir / f"{ds}_centroid_{REFILL_CLUSTERS}.fvecs")
    )
    n = expected_rows(root, ds)
    rng = np.random.default_rng(SPLIT_SEED)
    rows = np.sort(rng.choice(n, size=min(sample, n), replace=False))
    x = sample_rows(base_path, rows, dim)
    if HAVE_FAISS:
        index = faiss.IndexFlatL2(dim)
        index.add(cent)
        _, found = index.search(x, 1)
        ids = found[:, 0].astype(np.int64)
    else:  # pragma: no cover - faiss ships in the paper's env
        d = (
            (x.astype(np.float64) ** 2).sum(axis=1)[:, None]
            - 2.0 * x @ cent.astype(np.float64).T
            + (cent.astype(np.float64) ** 2).sum(axis=1)[None, :]
        )
        ids = d.argmin(axis=1)
    cid = np.memmap(
        ddir / f"{ds}_cluster_id_{REFILL_CLUSTERS}.ivecs", dtype=np.int32, mode="r"
    ).reshape(-1, 2)[rows, 1]
    rate = float((ids == cid).mean())
    if rate < 0.9:
        raise VerifyFailed(
            f"{ds}: only {rate:.1%} of {len(rows)} sampled rows match the "
            "stored IVF assignment"
        )
    log(f"{ds}: IVF assignment agreement {rate:.1%} on {len(rows)} sampled rows")


def check_groundtruth(root: Path, ds: str, base_path: Path, n_queries: int = 3) -> None:
    """Sampled queries must reproduce the stored exact top-100 neighbours."""
    ddir = ddir_of(root, ds)
    q_all = read_fvecs(ddir / f"{ds}_query.fvecs")
    gt = read_ivecs(ddir / f"{ds}_groundtruth.ivecs")
    rng = np.random.default_rng(777)
    qsel = np.sort(
        rng.choice(len(q_all), size=min(n_queries, len(q_all)), replace=False)
    )
    tops = exact_top_k(base_path, q_all[qsel], k=gt.shape[1])
    worst = 1.0
    for row, qi in enumerate(qsel):
        overlap = len(set(tops[row].tolist()) & set(gt[qi].tolist())) / gt.shape[1]
        worst = min(worst, overlap)
    if worst < 0.95:
        raise VerifyFailed(
            f"{ds}: sampled exact top-{gt.shape[1]} overlaps the stored "
            f"ground truth only {worst:.0%}"
        )
    log(
        f"{ds}: exact search reproduces the stored ground truth "
        f"(worst overlap {worst:.0%} over {len(qsel)} queries)"
    )


def verify_base(root: Path, ds: str, base_path: Path) -> None:
    verify_shape(root, ds, base_path)
    check_assignment(root, ds, base_path)
    check_groundtruth(root, ds, base_path)


def regenerate_base_pca(root: Path, ds: str, base_path: Path) -> None:
    """Apply the stored PCA basis to the fresh base, never re-fit it."""
    ddir = ddir_of(root, ds)
    dim = DATASETS[ds][0]
    out_path = ddir / f"{ds}_base_pca.fvecs"
    mu = read_fvecs(ddir / f"{ds}_pca_mean.fvecs")
    mat = read_fvecs(ddir / f"{ds}_pca_matrix.fvecs")
    if mu.shape != (1, dim) or mat.shape != (dim, dim):
        raise VerifyFailed(
            f"{ds}: pca_mean {mu.shape} / pca_matrix {mat.shape} do not "
            f"match dim {dim}"
        )

    mat64 = mat.astype(np.float64)
    orth = float(np.max(np.abs(mat64 @ mat64.T - np.eye(dim))))
    if orth > 1e-3:
        raise VerifyFailed(
            f"{ds}: stored pca_matrix is not orthonormal (max |MM^T - I| = {orth:g})"
        )

    # Pick the multiplication convention that reproduces the stored
    # query_pca rows; guard against silently transposing the rotation.
    mu64 = mu[0].astype(np.float64)
    nq = fvecs_shape(ddir / f"{ds}_query.fvecs")[0]
    rng = np.random.default_rng(11)
    qsel = np.sort(rng.choice(nq, size=min(3, nq), replace=False))
    chosen, best_resid = None, float("inf")
    for name, m in (("matrix.T", mat64.T), ("matrix", mat64)):
        resid = 0.0
        for qi in qsel:
            q = read_fvecs_block(ddir / f"{ds}_query.fvecs", int(qi), 1, dim)[
                0
            ].astype(np.float64)
            y_stored = read_fvecs_block(
                ddir / f"{ds}_query_pca.fvecs", int(qi), 1, dim
            )[0]
            y = (q - mu64) @ m
            scale = float(np.max(np.abs(y_stored))) + 1.0
            resid = max(resid, float(np.max(np.abs(y - y_stored))) / scale)
        if resid < best_resid:
            chosen, best_resid = name, resid
    if best_resid > 1e-3:
        raise VerifyFailed(
            f"{ds}: neither rotation convention reproduces query_pca "
            f"(best residual {best_resid:g}); refusing to write base_pca"
        )
    log(
        f"{ds}: rotation convention y = (x - mean) @ pca_{chosen} verified "
        f"against query_pca (residual {best_resid:.1e})"
    )
    m64 = mat64.T if chosen == "matrix.T" else mat64

    n, _ = fvecs_shape(base_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    header = np.array([dim], dtype=np.int32).view(np.float32)[0]
    t0 = time.time()
    with tmp.open("wb") as fh:
        for b0 in range(0, n, RECORD_BLOCK):
            bn = min(RECORD_BLOCK, n - b0)
            x = read_fvecs_block(base_path, b0, bn, dim).astype(np.float64)
            _write_fvecs_with_header(fh, header, ((x - mu64) @ m64).astype(np.float32))
            if b0 % (RECORD_BLOCK * 16) == 0:
                log(
                    f"  {ds}: rotated {b0 + bn}/{n} rows "
                    f"({100.0 * (b0 + bn) / n:.0f}%, {time.time() - t0:.0f}s)"
                )
    tmp.replace(out_path)
    if fvecs_shape(out_path) != (n, dim):
        raise VerifyFailed(f"{ds}: base_pca written with wrong shape")
    log(
        f"{ds}: base_pca written ({out_path.stat().st_size / 1e9:.2f} GB, "
        f"{time.time() - t0:.0f}s)"
    )


# ---------------------------------------------------------------------------
# Refill source fetchers.  Each returns the path of a candidate base; the
# verification above happens in refill_one() before anything is installed.


def _candidates(ext: Path, tail: str) -> list[Path]:
    return sorted(
        p
        for p in ext.rglob("*")
        if p.is_file()
        and p.name.endswith(tail)
        and "_pca" not in p.name
        and ".vars" not in p.name
    )


def pick_from_tree(root: Path, ds: str, ext: Path, query_required: bool) -> Path:
    """Find the base in an extracted archive and cross-check its companions."""
    dim = DATASETS[ds][0]
    exp_n = expected_rows(root, ds)
    bases = [
        p for p in _candidates(ext, "_base.fvecs") if fvecs_shape(p) == (exp_n, dim)
    ]
    if not bases:
        found = [p.name for p in _candidates(ext, "_base.fvecs")]
        raise VerifyFailed(
            f"{ds}: archive holds no {exp_n}x{dim} *_base.fvecs "
            f"(found {found or 'none'})"
        )
    if len(bases) > 1:
        log(
            f"{ds}: WARNING multiple matching bases {[p.name for p in bases]}; "
            "using the first"
        )
    base = bases[0]

    queries = _candidates(ext, "_query.fvecs")
    if queries:
        stored = read_fvecs(ddir_of(root, ds) / f"{ds}_query.fvecs")
        exact, diff = compare_arrays(read_fvecs(queries[0]), stored)
        if exact:
            log(f"{ds}: archive query is bit-identical to the stored one")
        else:
            msg = f"{ds}: archive query differs from stored (max |d| = {diff:g})"
            if query_required:
                raise VerifyFailed(msg)
            log(f"{msg}; the stored ground truth will decide")
    elif query_required:
        raise VerifyFailed(f"{ds}: archive holds no *_query.fvecs to compare")

    gts = _candidates(ext, "_groundtruth.ivecs")
    if gts:
        stored = read_ivecs(ddir_of(root, ds) / f"{ds}_groundtruth.ivecs")
        exact, diff = compare_arrays(read_ivecs(gts[0]), stored)
        state = (
            "is bit-identical to"
            if exact
            else f"deviates from (max |d| = {diff:g})"
        )
        log(f"{ds}: archive ground truth {state} the stored one")
    return base


def fetch_gqr(root: Path, ds: str, raw: Path) -> Path:
    url = GQR_URL.format(name=ds)
    tarball = raw / f"{ds}.tar.gz"
    size = http_size(url)
    for attempt in range(3):
        try:
            wget(url, tarball, expect_size=size)
        except (OSError, subprocess.CalledProcessError) as exc:
            log(f"{ds}: wget failed (attempt {attempt + 1}, {exc}); retrying")
            continue
        ext = raw / "extract"
        if ext.exists():
            shutil.rmtree(ext)
        ext.mkdir(parents=True)
        try:
            run(["tar", "-xzf", str(tarball), "-C", str(ext)])
        except subprocess.CalledProcessError:
            log(
                f"{ds}: tarball extraction failed (attempt {attempt + 1}); "
                "re-downloading"
            )
            tarball.unlink(missing_ok=True)
            continue
        return pick_from_tree(root, ds, ext, query_required=True)
    raise VerifyFailed(f"{ds}: tarball unusable after 3 downloads")


def fetch_sift10m(root: Path, ds: str, raw: Path) -> Path:
    zpath = raw / "sift10m.zip"
    # The UCI server ignores Range, so a partial download cannot resume and
    # any truncated zip is restarted from scratch.
    for attempt in range(5):
        try:
            if not (zpath.exists() and zpath.stat().st_size > 0):
                wget(SIFT10M_URL, zpath)
            ext = raw / "extract"
            if ext.exists():
                shutil.rmtree(ext)
            ext.mkdir(parents=True)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(ext)
        except (zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
            log(
                f"{ds}: zip incomplete (attempt {attempt + 1}, {exc}); "
                "restarting the download"
            )
            zpath.unlink(missing_ok=True)
            continue
        return pick_from_tree(root, ds, ext, query_required=False)
    raise VerifyFailed(f"{ds}: zip unusable after 5 downloads")


def hf_materialize(root: Path, ds: str, raw: Path) -> Path:
    """Rebuild the base with the prepare convert recipe, streamed."""
    ddir = ddir_of(root, ds)
    repo, include = HF_REPO[ds], HF_INCLUDE[ds]
    tree = hf_tree(repo)
    files = sorted((p, s) for p, s in tree.items() if fnmatch.fnmatch(p, include))
    if not files:
        raise VerifyFailed(f"{ds}: no file in {repo} matches {include}")
    log(
        f"{ds}: {repo}: {len(files)} parquet shards, "
        f"{sum(s for _, s in files) / 1e9:.1f} GB"
    )
    hf_download_files(repo, files, raw)

    tmp_base = ddir / f"{ds}_base.fvecs.tmp"
    tmp_query = ddir / f"{ds}_query.fvecs.tmp"
    convert_hf(ds, raw, tmp_base, tmp_query, HF_BASE_SIZE.get(ds))
    stored = read_fvecs(ddir / f"{ds}_query.fvecs")
    exact, diff = compare_arrays(read_fvecs(tmp_query), stored)
    if exact:
        log(f"{ds}: regenerated query is bit-identical to the stored one")
    else:
        log(
            f"{ds}: WARNING regenerated query differs from stored "
            f"(max |d| = {diff:g}); the stored ground truth decides"
        )
    tmp_query.unlink(missing_ok=True)
    return tmp_base


def fetch_msmarco(root: Path, ds: str, raw: Path, base_size: int) -> Path:
    ddir = ddir_of(root, ds)
    raw_root = raw / MSMARCO_REPO.replace("/", "_")
    names = msmarco_fetch_shards(raw_root, base_size)
    base = materialize_base(ddir, ds, raw_root, names, base_size)

    # Rebuild the query/ground-truth pair from the published offsets and
    # compare; they are independent of the shards, so agreement certifies
    # the whole recipe.
    check_dir = Path(tempfile.mkdtemp(prefix=".qcheck_", dir=ddir))
    try:
        prepare_msmarco_queries(check_dir, ds, raw_root, base_size, gt_k=100)
        for name in (f"{ds}_query.fvecs", f"{ds}_groundtruth.ivecs"):
            reader = read_fvecs if name.endswith(".fvecs") else read_ivecs
            exact, diff = compare_arrays(reader(check_dir / name), reader(ddir / name))
            state = (
                "is bit-identical to"
                if exact
                else f"differs from (max |d| = {diff:g})"
            )
            log(f"{ds}: regenerated {name} {state} the stored one")
    finally:
        shutil.rmtree(check_dir)
    return base


def refill_one(root: Path, ds: str, keep_raw: bool) -> str:
    ddir = ddir_of(root, ds)
    raw = ddir / "raw"
    dim, kind = DATASETS[ds]
    miss = missing_parts(root, ds)
    if not miss:
        return "complete"
    if not (ddir / f"{ds}_cluster_id_{REFILL_CLUSTERS}.ivecs").exists():
        raise VerifyFailed(
            f"{ds}: nothing to refill from — {ddir} holds no prepared "
            f"artifacts (run datasets/prepare.sh {ds} first)"
        )
    log(f"{ds}: missing {' and '.join(miss)} — refilling from the {kind} source")

    need = expected_base_bytes(root, ds) * (2 if "base_pca" in miss else 1) * 3
    free = shutil.disk_usage(root).free
    if free < need:
        raise VerifyFailed(
            f"{ds}: needs ~{need / 1e9:.0f} GB headroom, {free / 1e9:.0f} GB free"
        )

    ddir.mkdir(parents=True, exist_ok=True)
    base_final = ddir / f"{ds}_base.fvecs"
    # msmarco's materializer writes the base in place (it streams 10M rows),
    # so a failure after that point must remove it or it would look whole.
    in_place = kind == "msmarco" and "base" in miss

    try:
        if "base" in miss:
            if kind == "gqr":
                candidate = fetch_gqr(root, ds, raw)
            elif kind == "sift10m":
                candidate = fetch_sift10m(root, ds, raw)
            elif kind == "hf":
                candidate = hf_materialize(root, ds, raw)
            elif kind == "msmarco":
                candidate = fetch_msmarco(root, ds, raw, MSMARCO_BASE)
            else:
                raise VerifyFailed(
                    f"{ds}: source kind '{kind}' is not refillable here; "
                    f"use datasets/prepare.sh {ds}"
                )
        else:
            log(f"{ds}: base already present; re-verifying it")
            candidate = base_final

        verify_base(root, ds, candidate)
        if candidate != base_final:
            os.replace(candidate, base_final)
        log(f"{ds}: base installed ({base_final.stat().st_size / 1e9:.2f} GB)")

        if not (ddir / f"{ds}_base_pca.fvecs").exists():
            regenerate_base_pca(root, ds, base_final)
    except Exception:
        if in_place:
            base_final.unlink(missing_ok=True)
        raise

    if raw.exists() and not keep_raw:
        log(f"{ds}: removing raw staging {raw}")
        shutil.rmtree(raw)
    return "filled"


# ---------------------------------------------------------------------------
# Command line.


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datasets", nargs="*",
                    help="corpus names or aliases (default: every registry entry)")
    ap.add_argument(
        "--data-root", type=Path, default=HERE,
        help="parent of the per-corpus folders; default is this datasets/ directory",
    )
    ap.add_argument("--all", action="store_true", help="prepare every registry entry")
    ap.add_argument("--steps", default="download,convert,gt,ivf,pca",
                    help="comma-separated subset of the preparation steps")
    ap.add_argument("--clusters", type=int, default=4096,
                    help="shared IVF nlist")
    ap.add_argument("--gt-k", type=int, default=100)
    ap.add_argument("--train-size", type=int, default=None,
                    help="IVF k-means training rows (default 512k; msmarco10M 1M)")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap HF rows read (default: the paper's corpus sizes)")
    ap.add_argument("--base-size", type=int, default=MSMARCO_BASE,
                    help="msmarco10M prefix and refill size (rows)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild existing outputs instead of keeping them")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep <name>/raw after a corpus is whole")
    ap.add_argument("--list", action="store_true",
                    help="show the corpus registry, then exit")
    ap.add_argument("--refill", action="store_true",
                    help="re-fetch only the missing base files of prepared corpora")
    ap.add_argument("--status", action="store_true",
                    help="with --refill: list what is missing, then exit")
    args = ap.parse_args()
    args.steps = args.steps.split(",")
    return args


def prepare_all(root: Path, args) -> None:
    names = [ALIASES.get(n, n) for n in args.datasets]
    if args.all or not names:
        names = list(DATASETS)
    for name in names:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset {name}; --list shows the registry")
        prepare(name, root, args)
    log("done")


def refill(root: Path, args) -> None:
    if args.status:
        for ds, (dim, kind) in DATASETS.items():
            ddir = ddir_of(root, ds)
            if not ddir.is_dir():
                log(f"{ds:22s} d={dim:<5d} {kind:8s} absent")
                continue
            miss = missing_parts(root, ds)
            log(
                f"{ds:22s} d={dim:<5d} {kind:8s} "
                f"{'complete' if not miss else 'missing ' + ', '.join(miss)}"
            )
        return

    names = [ALIASES.get(n, n) for n in args.datasets] or [
        ds for ds in REFILL_ORDER
        if ddir_of(root, ds).is_dir() and missing_parts(root, ds)
    ]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}")

    log(
        f"refill: {len(names)} corpus/corpora, "
        f"HF_ENDPOINT={HF_ENDPOINT}, data root {root}"
    )
    t0 = time.time()
    results: list[tuple[str, str]] = []
    for ds in names:
        try:
            results.append((ds, refill_one(root, ds, args.keep_raw)))
        except Exception as exc:  # noqa: BLE001 - keep going through the list
            log(f"{ds}: FAILED — {exc}")
            traceback.print_exc()
            results.append((ds, f"FAILED: {exc}"))

    log("=" * 70)
    for ds, res in results:
        log(f"  {ds:22s} {res}")
    free = shutil.disk_usage(root).free
    log(f"elapsed {time.time() - t0:.0f}s, {free / 1e9:.0f} GB free")
    if any(res.startswith("FAILED") for _, res in results):
        sys.exit(1)


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()

    if args.list:
        for name, (dim, kind) in DATASETS.items():
            print(f"{name:22s} d={dim:<5d} {kind}")
        return

    if args.refill or args.status:
        refill(root, args)
        return

    prepare_all(root, args)


if __name__ == "__main__":
    main()
