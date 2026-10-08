"""Python tests for E8qIndex and Me8qIndex."""

import numpy as np
import pytest
from e8qlib import E8qIndex, Me8qIndex

DIM = 64
N_VECTORS = 400
N_QUERIES = 16
N_CLUSTERS = 8
TOPK = 10


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    scale = np.power(0.97, np.arange(DIM)).astype(np.float32)
    base = (rng.standard_normal((N_VECTORS, DIM)) * scale).astype(np.float32)
    queries = (rng.standard_normal((N_QUERIES, DIM)) * scale).astype(np.float32)
    cluster_ids = (np.arange(N_VECTORS) % N_CLUSTERS).astype(np.uint32)
    centroids = np.stack(
        [base[cluster_ids == c].mean(axis=0) for c in range(N_CLUSTERS)]
    ).astype(np.float32)
    return base, queries, centroids, cluster_ids


def test_e8q_build_search_roundtrip(data, tmp_path):
    base, queries, centroids, cluster_ids = data
    idx = E8qIndex(dim=DIM, nbits=4, seed=1)
    idx.build(base, centroids, cluster_ids, num_threads=2)
    assert idx.is_built
    assert idx.n == N_VECTORS
    assert idx.num_clusters == N_CLUSTERS
    ids, dists = idx.search(queries, k=TOPK, nprobe=N_CLUSTERS, num_threads=2)
    assert ids.shape == (N_QUERIES, TOPK)
    assert dists.shape == (N_QUERIES, TOPK)
    assert np.all(ids < N_VECTORS)
    path = tmp_path / "e8q.bin"
    idx.save(str(path))
    loaded = E8qIndex.load(str(path))
    ids2, _ = loaded.search(queries, k=TOPK, nprobe=N_CLUSTERS)
    np.testing.assert_array_equal(ids, ids2)


def test_me8q_build_search_roundtrip(data, tmp_path):
    base, queries, centroids, cluster_ids = data
    idx = Me8qIndex(dim=DIM, avg_bits=3.0)
    idx.build(base, centroids, cluster_ids, num_threads=2)
    assert idx.is_built
    assert idx.n == N_VECTORS
    ids, dists = idx.search(queries, k=TOPK, nprobe=N_CLUSTERS, num_threads=2)
    assert ids.shape == (N_QUERIES, TOPK)
    assert dists.shape == (N_QUERIES, TOPK)
    path = tmp_path / "me8q.bin"
    idx.save(str(path))
    loaded = Me8qIndex.load(str(path))
    ids2, _ = loaded.search(queries, k=TOPK, nprobe=N_CLUSTERS)
    np.testing.assert_array_equal(ids, ids2)
