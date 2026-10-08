#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bindings_common.hpp"
#include "e8qlib/index/me8q.hpp"

namespace py = pybind11;

namespace e8qlib::python_bindings {

class Me8qIndex {
   public:
    Me8qIndex(size_t dim, float avg_bits = 4.0F, uint64_t seed = 0)
        : dim_(dim), avg_bits_(avg_bits), seed_(seed) {
        if (dim == 0) {
            throw std::invalid_argument("dim must be positive");
        }
        if (avg_bits <= 0.0F) {
            throw std::invalid_argument("avg_bits must be positive");
        }
    }

    void build(
        py::handle data,
        py::handle centroids,
        py::handle cluster_ids,
        size_t num_threads = 1,
        py::handle variance = py::none(),
        bool drop_energy = true,
        float prune_alpha = 0.0F,
        std::string plan_spec = "",
        double extra_budget_bits = 0.0
    ) {
        auto data_array = ensure_2d_array<float>(data, "data");
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");
        auto cluster_ids_array = ensure_1d_array<uint32_t>(cluster_ids, "cluster_ids");

        if (static_cast<size_t>(data_array.shape(1)) != dim_) {
            throw std::invalid_argument("data dimension does not match index dim");
        }
        if (static_cast<size_t>(centroids_array.shape(1)) != dim_) {
            throw std::invalid_argument("centroid dimension does not match index dim");
        }
        if (static_cast<size_t>(cluster_ids_array.shape(0)) !=
            static_cast<size_t>(data_array.shape(0))) {
            throw std::invalid_argument("cluster_ids length must match number of rows in data");
        }

        const float* var_ptr = nullptr;
        py::array_t<float, py::array::c_style | py::array::forcecast> var_array;
        if (!variance.is_none()) {
            var_array = ensure_1d_array<float>(variance, "variance");
            if (static_cast<size_t>(var_array.shape(0)) != dim_) {
                throw std::invalid_argument("variance length must match dim");
            }
            var_ptr = var_array.data();
        }

        e8qlib::me8q::BuildOptions opt;
        opt.avg_bits = avg_bits_;
        opt.threads = num_threads;
        opt.drop_energy = drop_energy;
        opt.prune_alpha = prune_alpha;
        opt.plan_spec = plan_spec;
        opt.extra_budget_bits = extra_budget_bits;
        if (seed_ != 0) {
            opt.rot_seed = seed_;
        }

        index_ = std::make_unique<e8qlib::me8q::Me8qIndex>();
        index_->Build(
            data_array.data(),
            static_cast<size_t>(data_array.shape(1)),
            centroids_array.data(),
            static_cast<size_t>(centroids_array.shape(1)),
            cluster_ids_array.data(),
            static_cast<size_t>(data_array.shape(0)),
            dim_,
            static_cast<size_t>(centroids_array.shape(0)),
            opt,
            var_ptr
        );
        built_ = true;
    }

    py::tuple search(
        py::handle queries,
        size_t k,
        size_t nprobe,
        size_t num_threads = 1,
        bool return_stats = false
    ) {
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (!built_ || index_ == nullptr) {
            throw std::runtime_error("Me8qIndex must be built or loaded before search");
        }
        if (static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (k == 0) {
            throw std::invalid_argument("k must be positive");
        }
        if (nprobe == 0) {
            throw std::invalid_argument("nprobe must be positive");
        }

        const size_t nq = static_cast<size_t>(query_array.shape(0));
        const auto shape =
            std::vector<ssize_t>{static_cast<ssize_t>(nq), static_cast<ssize_t>(k)};
        auto ids = py::array_t<uint32_t>(shape);
        auto dists = py::array_t<float>(shape);
        auto* ids_data = ids.mutable_data();
        auto* dists_data = dists.mutable_data();
        std::fill(ids_data, ids_data + ids.size(), std::numeric_limits<uint32_t>::max());
        std::fill(
            dists_data, dists_data + dists.size(), std::numeric_limits<float>::infinity()
        );

        const int threads = static_cast<int>(std::max<size_t>(1, num_threads));
        std::vector<e8qlib::me8q::SearchStats> per_thread(
            return_stats ? static_cast<size_t>(threads) : 0
        );
        if (nq > 1) {
            // Batched entry point: the same per-query results, but one
            // centroid sweep serves a whole group of queries (the scan
            // streams the index through L3 and evicts the centroid matrix,
            // so a per-query sweep pays DRAM for all 2 MiB).
            index_->SearchIVFFastBatch(
                query_array.data(), nq, k, nprobe, ids_data,
                return_stats ? &per_thread[0] : nullptr, dists_data, threads
            );
        } else {
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
// dynamic(1): survivor-count variance (0-500 exact-refine rows per query)
// makes a 16-chunk straggler set the round wall — same fix as the e8q
// binding.
#pragma omp for schedule(dynamic, 1)
#endif
            for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
                const size_t idx = static_cast<size_t>(qi);
                index_->SearchIVFFast(
                    query_array.data() + idx * dim_,
                    k,
                    nprobe,
                    ids_data + idx * k,
                    return_stats ? &per_thread[static_cast<size_t>(tid)] : nullptr,
                    dists_data + idx * k
                );
            }
        }
        }
        if (!return_stats) {
            return py::make_tuple(ids, dists);
        }
        e8qlib::me8q::SearchStats acc;
        for (const auto& s : per_thread) {
            acc.candidates += s.candidates;
            acc.survivors += s.survivors;
            acc.bits_scored += s.bits_scored;
            acc.bits_total += s.bits_total;
            acc.pruned += s.pruned;
            acc.rotate_us += s.rotate_us;
            acc.route_us += s.route_us;
            acc.scan_us += s.scan_us;
            acc.refine_us += s.refine_us;
            acc.lut_us += s.lut_us;
            acc.filter_us += s.filter_us;
            acc.lane_us += s.lane_us;
            acc.rest_us += s.rest_us;
        }
        py::dict stats;
        stats["candidates"] = acc.candidates;
        stats["survivors"] = acc.survivors;
        stats["bits_scored"] = acc.bits_scored;
        stats["bits_total"] = acc.bits_total;
        stats["pruned"] = acc.pruned;
        stats["rotate_us"] = acc.rotate_us / static_cast<double>(nq);
        stats["route_us"] = acc.route_us / static_cast<double>(nq);
        stats["scan_us"] = acc.scan_us / static_cast<double>(nq);
        stats["refine_us"] = acc.refine_us / static_cast<double>(nq);
        stats["lut_us"] = acc.lut_us / static_cast<double>(nq);
        stats["filter_us"] = acc.filter_us / static_cast<double>(nq);
        stats["lane_us"] = acc.lane_us / static_cast<double>(nq);
        stats["rest_us"] = acc.rest_us / static_cast<double>(nq);
        stats["survivor_rate"] =
            acc.candidates > 0
                ? static_cast<double>(acc.survivors) / static_cast<double>(acc.candidates)
                : 0.0;
        return py::make_tuple(ids, dists, stats);
    }

    float estimate(py::handle query, uint32_t id) const {
        if (!built_ || index_ == nullptr) {
            throw std::runtime_error("Me8qIndex must be built or loaded before estimate");
        }
        auto q = ensure_1d_array<float>(query, "query");
        if (static_cast<size_t>(q.shape(0)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        return index_->Estimate(q.data(), id);
    }

    void save(const std::string& path) const {
        if (!built_ || index_ == nullptr) {
            throw std::runtime_error("Me8qIndex must be built or loaded before save");
        }
        index_->Save(path.c_str());
    }

    static Me8qIndex load(const std::string& path) {
        Me8qIndex wrapper(1, 4.0F);
        wrapper.index_ = std::make_unique<e8qlib::me8q::Me8qIndex>();
        wrapper.index_->Load(path.c_str());
        wrapper.dim_ = wrapper.index_->dim();
        wrapper.avg_bits_ = static_cast<float>(wrapper.index_->bits_per_dim());
        wrapper.built_ = true;
        return wrapper;
    }

    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] float avg_bits() const { return avg_bits_; }
    [[nodiscard]] bool is_built() const { return built_; }
    [[nodiscard]] size_t n() const { return index_ != nullptr ? index_->n() : 0; }
    [[nodiscard]] size_t num_clusters() const {
        return index_ != nullptr ? index_->num_clusters() : 0;
    }
    [[nodiscard]] double bits_per_dim() const {
        return index_ != nullptr ? index_->bits_per_dim() : 0.0;
    }
    [[nodiscard]] size_t memory_bytes() const {
        return index_ != nullptr ? index_->memory_bytes() : 0;
    }
    // DP-chosen segmentation (len, bits per run) for ledger memory
    // decomposition on the python side.
    [[nodiscard]] std::vector<std::pair<size_t, int>> plan() const {
        if (index_ == nullptr) {
            return {};
        }
        std::vector<std::pair<size_t, int>> out;
        for (const auto& seg : index_->shared_plan()) {
            out.emplace_back(seg.len, seg.bits);
        }
        return out;
    }

   private:
    size_t dim_ = 0;
    float avg_bits_ = 4.0F;
    uint64_t seed_ = 0;
    bool built_ = false;
    std::unique_ptr<e8qlib::me8q::Me8qIndex> index_;
};

}  // namespace e8qlib::python_bindings

void register_me8q(py::module_& m) {
    using namespace e8qlib::python_bindings;
    py::class_<Me8qIndex>(m, "Me8qIndex")
        .def(
            py::init<size_t, float, uint64_t>(),
            py::arg("dim"),
            py::arg("avg_bits") = 4.0F,
            py::arg("seed") = 0
        )
        .def(
            "build",
            &Me8qIndex::build,
            py::arg("data"),
            py::arg("centroids"),
            py::arg("cluster_ids"),
            py::arg("num_threads") = 1,
            py::arg("variance") = py::none(),
            py::arg("drop_energy") = true,
            py::arg("prune_alpha") = 0.0F,
            py::arg("plan_spec") = "",
            py::arg("extra_budget_bits") = 0.0
        )
        .def(
            "search",
            &Me8qIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("nprobe"),
            py::arg("num_threads") = 1,
            py::arg("return_stats") = false
        )
        .def("estimate", &Me8qIndex::estimate, py::arg("query"), py::arg("id"))
        .def("save", &Me8qIndex::save, py::arg("path"))
        .def_static("load", &Me8qIndex::load, py::arg("path"))
        .def_property_readonly("dim", &Me8qIndex::dim)
        .def_property_readonly("avg_bits", &Me8qIndex::avg_bits)
        .def_property_readonly("is_built", &Me8qIndex::is_built)
        .def_property_readonly("n", &Me8qIndex::n)
        .def_property_readonly("num_clusters", &Me8qIndex::num_clusters)
        .def_property_readonly("bits_per_dim", &Me8qIndex::bits_per_dim)
        .def_property_readonly("memory_bytes", &Me8qIndex::memory_bytes)
        .def_property_readonly("plan", &Me8qIndex::plan);
}
