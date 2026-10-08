"""Minimal IVF indexing example for E8Q."""

import argparse

import numpy as np
from e8qlib import E8qIndex, Me8qIndex


def cluster_round_robin(data: np.ndarray, k: int):
    ids = (np.arange(len(data)) % k).astype(np.uint32)
    cents = np.stack([data[ids == c].mean(axis=0) for c in range(k)]).astype(np.float32)
    return cents, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("e8q", "me8q"), default="e8q")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--nlist", type=int, default=8)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    data = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    queries = rng.standard_normal((8, args.dim)).astype(np.float32)
    centroids, cluster_ids = cluster_round_robin(data, args.nlist)

    if args.method == "e8q":
        index = E8qIndex(dim=args.dim, nbits=args.bits, seed=1)
        index.build(data, centroids, cluster_ids, num_threads=2)
    else:
        index = Me8qIndex(dim=args.dim, avg_bits=float(args.bits))
        index.build(data, centroids, cluster_ids, num_threads=2)

    ids, dists = index.search(queries, k=10, nprobe=args.nlist)
    print(ids.shape, dists.shape)
    print(ids[0])


if __name__ == "__main__":
    main()
