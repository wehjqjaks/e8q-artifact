// Minimal E8Q IVF example: synthetic data, build, save/load, search.
//
//   ./e8q_index [n=1000] [dim=64] [nlist=8] [bits=4] [nprobe=8] [k=10]
//
// Centroids are per-list means of a round-robin assignment (toy clustering).
// Production indexes use offline k-means (`datasets/prepare.sh`).

#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "e8qlib/index/e8q.hpp"

int run(int argc, char** argv) {
    const size_t n = argc > 1 ? static_cast<size_t>(std::stoul(argv[1])) : 1000;
    const size_t dim = argc > 2 ? static_cast<size_t>(std::stoul(argv[2])) : 64;
    const size_t nlist = argc > 3 ? static_cast<size_t>(std::stoul(argv[3])) : 8;
    const int bits = argc > 4 ? std::stoi(argv[4]) : 4;
    const size_t nprobe = argc > 5 ? static_cast<size_t>(std::stoul(argv[5])) : nlist;
    const size_t k = argc > 6 ? static_cast<size_t>(std::stoul(argv[6])) : 10;

    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.0F, 1.0F);

    std::vector<float> data(n * dim);
    for (float& v : data) {
        v = nd(rng);
    }

    std::vector<uint32_t> cluster_ids(n);
    std::vector<float> centroids(nlist * dim, 0.0F);
    std::vector<size_t> counts(nlist, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t c = static_cast<uint32_t>(i % nlist);
        cluster_ids[i] = c;
        ++counts[c];
        for (size_t d = 0; d < dim; ++d) {
            centroids[c * dim + d] += data[i * dim + d];
        }
    }
    for (size_t c = 0; c < nlist; ++c) {
        const float inv = 1.0F / static_cast<float>(counts[c]);
        for (size_t d = 0; d < dim; ++d) {
            centroids[c * dim + d] *= inv;
        }
    }

    e8qlib::E8qIndex index;
    index.Build(
        data.data(),
        centroids.data(),
        cluster_ids.data(),
        n,
        dim,
        nlist,
        bits,
        /*seed=*/1,
        /*threads=*/2
    );

    const char* path = "e8q_sample.idx";
    index.Save(path);
    e8qlib::E8qIndex loaded;
    loaded.Load(path);

    std::vector<float> query(dim);
    for (float& v : query) {
        v = nd(rng);
    }

    e8qlib::QueryScratch scratch;
    e8qlib::SearchStats stats;
    std::vector<uint32_t> ids(k);
    std::vector<float> dists(k);
    loaded.Search(query.data(), k, nprobe, ids.data(), scratch, &stats, dists.data());

    std::cout << "E8Q n=" << loaded.n() << " dim=" << loaded.dim() << " bits=" << loaded.bits()
              << " nlist=" << loaded.num_clusters() << " mem=" << loaded.memory_bytes()
              << " B\n";
    std::cout << "search candidates=" << stats.candidates << " survivors=" << stats.survivors
              << "\n";
    for (size_t i = 0; i < k; ++i) {
        std::cout << "  " << ids[i] << "  " << dists[i] << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
