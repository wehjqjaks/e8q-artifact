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
#include "e8qlib/index/e8q.hpp"

namespace py = pybind11;

namespace e8qlib::python_bindings {

class E8qIndex {
   public:
    E8qIndex(size_t dim, size_t nbits, uint64_t seed = 0)
        : dim_(dim), nbits_(nbits), seed_(seed) {
        if (dim == 0) {
            throw std::invalid_argument("dim must be positive");
        }
        if (nbits < 1 || nbits > 13) {
            throw std::invalid_argument("nbits must be in [1, 13]");
        }
    }

    void build(
        py::handle data,
        py::handle centroids,
        py::handle cluster_ids,
        size_t num_threads = 1,
        float prune_alpha = 0.0F
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

        index_ = std::make_unique<e8qlib::E8qIndex>();
        e8qlib::E8qBuildOptions opt;
        opt.prune_alpha = prune_alpha;
        index_->Build(
            data_array.data(),
            centroids_array.data(),
            cluster_ids_array.data(),
            static_cast<size_t>(data_array.shape(0)),
            dim_,
            static_cast<size_t>(centroids_array.shape(0)),
            static_cast<int>(nbits_),
            seed_,
            num_threads,
            opt
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
            throw std::runtime_error("E8qIndex must be built or loaded before search");
        }
        if (static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (k == 0) {
            throw std::invalid_argument("k must be positive");
        }
        // Validate against the core's cap before entering the OpenMP region:
        // an exception thrown inside the parallel region would terminate the
        // process instead of surfacing as a Python error.
        if (k > e8qlib::kMaxK) {
            throw std::invalid_argument(
                "k must be <= " + std::to_string(e8qlib::kMaxK)
            );
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
        std::vector<e8qlib::SearchStats> per_thread(
            return_stats ? static_cast<size_t>(threads) : 0
        );
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
        {
            e8qlib::QueryScratch scratch;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
// Chunk 1, not 16: per-query cost varies with the survivor count (0-500+
            // exact-refine rows), so a 16-query chunk hands one thread a straggler
            // tail that sets the whole round's wall time at low nprobe.
#pragma omp for schedule(dynamic, 1)
#endif
            for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
                const size_t idx = static_cast<size_t>(qi);
                index_->Search(
                    query_array.data() + idx * dim_,
                    k,
                    nprobe,
                    ids_data + idx * k,
                    scratch,
                    return_stats ? &per_thread[static_cast<size_t>(tid)] : nullptr,
                    dists_data + idx * k
                );
            }
        }
        if (!return_stats) {
            return py::make_tuple(ids, dists);
        }
        e8qlib::SearchStats acc;
        for (const auto& s : per_thread) {
            acc.candidates += s.candidates;
            acc.survivors += s.survivors;
            acc.rotate_us += s.rotate_us;
            acc.prepare_route_us += s.prepare_route_us;
            acc.scan_us += s.scan_us;
            acc.refine_us += s.refine_us;
        }
        py::dict stats;
        stats["candidates"] = acc.candidates;
        stats["survivors"] = acc.survivors;
        stats["rotate_us"] = acc.rotate_us / static_cast<double>(nq);
        stats["route_us"] = acc.prepare_route_us / static_cast<double>(nq);
        stats["scan_us"] = acc.scan_us / static_cast<double>(nq);
        stats["refine_us"] = acc.refine_us / static_cast<double>(nq);
        stats["survivor_rate"] =
            acc.candidates > 0
                ? static_cast<double>(acc.survivors) / static_cast<double>(acc.candidates)
                : 0.0;
        return py::make_tuple(ids, dists, stats);
    }

    float estimate(py::handle query, uint32_t id) const {
        if (!built_ || index_ == nullptr) {
            throw std::runtime_error("E8qIndex must be built or loaded before estimate");
        }
        auto q = ensure_1d_array<float>(query, "query");
        if (static_cast<size_t>(q.shape(0)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        return index_->Estimate(q.data(), id);
    }

    // Batched ZPR bound probe: for pair i, probes the row base_rows[i] (raw
    // build input of global id ids[i]) with queries[qidx[i]] and writes the
    // normalized discrepancy (d_c^2 - d^2) / (r_q * Omega) into out[i].
    // Lives here rather than in Python so the sampling loop costs one call,
    // not one call per pair; rows are passed gathered, so the caller never
    // has to materialize or pack a whole corpus.
    py::array_t<double> probe_bound_w(
        py::handle queries, py::handle base_rows, py::handle qidx, py::handle ids,
        size_t num_threads = 1
    ) const {
        if (!built_ || index_ == nullptr) {
            throw std::runtime_error("E8qIndex must be built or loaded before probe_bound_w");
        }
        auto q = ensure_2d_array<float>(queries, "queries");
        auto b = ensure_2d_array<float>(base_rows, "base_rows");
        auto qi = ensure_1d_array<uint32_t>(qidx, "qidx");
        auto ii = ensure_1d_array<uint32_t>(ids, "ids");
        if (static_cast<size_t>(q.shape(1)) != dim_) {
            throw std::invalid_argument("queries dimension does not match index dim");
        }
        if (static_cast<size_t>(b.shape(1)) != dim_) {
            throw std::invalid_argument("base_rows dimension does not match index dim");
        }
        const size_t m = static_cast<size_t>(ii.shape(0));
        if (static_cast<size_t>(qi.shape(0)) != m ||
            static_cast<size_t>(b.shape(0)) != m) {
            throw std::invalid_argument(
                "qidx, ids and base_rows must have the same length"
            );
        }
        const float* qp = q.data();
        const float* bp = b.data();
        const uint32_t* qip = qi.data();
        const uint32_t* iip = ii.data();
        py::array_t<double> out(static_cast<py::ssize_t>(m));
        double* op = out.mutable_data();
        const size_t nq = static_cast<size_t>(q.shape(0));
        const size_t nb = index_->n();
#ifdef _OPENMP
#pragma omp parallel for num_threads(static_cast<int>(std::max<size_t>(1, num_threads)))
#endif
        for (long long i = 0; i < static_cast<long long>(m); ++i) {
            const size_t j = static_cast<size_t>(i);
            if (qip[j] >= nq || iip[j] >= nb) {
                op[j] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            op[j] = index_->ProbeBoundW(
                qp + static_cast<size_t>(qip[j]) * dim_,
                bp + j * dim_, iip[j]
            );
        }
        return out;
    }

    void set_prune_alpha(float alpha) {
        if (!built_ || index_ == nullptr) {
            throw std::runtime_error("E8qIndex must be built or loaded before set_prune_alpha");
        }
        if (!(alpha > 0.0F)) {
            throw std::invalid_argument("prune_alpha must be positive");
        }
        index_->SetPruneAlpha(alpha);
    }

    void set_disable_prune(bool on) {
        if (!built_ || index_ == nullptr) {
            throw std::runtime_error("E8qIndex must be built or loaded before set_disable_prune");
        }
        index_->SetDisablePrune(on);
    }

    [[nodiscard]] bool disable_prune() const {
        return index_ != nullptr ? index_->disable_prune() : false;
    }

    void save(const std::string& path) const {
        if (!built_ || index_ == nullptr) {
            throw std::runtime_error("E8qIndex must be built or loaded before save");
        }
        index_->Save(path.c_str());
    }

    static E8qIndex load(const std::string& path) {
        E8qIndex wrapper(1, 1, 0);
        wrapper.index_ = std::make_unique<e8qlib::E8qIndex>();
        wrapper.index_->Load(path.c_str());
        wrapper.dim_ = wrapper.index_->dim();
        wrapper.nbits_ = static_cast<size_t>(wrapper.index_->bits());
        wrapper.built_ = true;
        return wrapper;
    }

    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t nbits() const { return nbits_; }
    [[nodiscard]] bool is_built() const { return built_; }
    [[nodiscard]] size_t n() const { return index_ != nullptr ? index_->n() : 0; }
    [[nodiscard]] size_t num_clusters() const {
        return index_ != nullptr ? index_->num_clusters() : 0;
    }
    [[nodiscard]] size_t memory_bytes() const {
        return index_ != nullptr ? index_->memory_bytes() : 0;
    }

   private:
    size_t dim_ = 0;
    size_t nbits_ = 0;
    uint64_t seed_ = 0;
    bool built_ = false;
    std::unique_ptr<e8qlib::E8qIndex> index_;
};

}  // namespace e8qlib::python_bindings

void register_e8q(py::module_& m) {
    using namespace e8qlib::python_bindings;
    py::class_<E8qIndex>(m, "E8qIndex")
        .def(
            py::init<size_t, size_t, uint64_t>(),
            py::arg("dim"),
            py::arg("nbits"),
            py::arg("seed") = 0
        )
        .def(
            "build",
            &E8qIndex::build,
            py::arg("data"),
            py::arg("centroids"),
            py::arg("cluster_ids"),
            py::arg("num_threads") = 1,
            py::arg("prune_alpha") = 0.0F
        )
        .def(
            "search",
            &E8qIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("nprobe"),
            py::arg("num_threads") = 1,
            py::arg("return_stats") = false
        )
        .def("estimate", &E8qIndex::estimate, py::arg("query"), py::arg("id"))
        .def(
            "probe_bound_w",
            &E8qIndex::probe_bound_w,
            py::arg("queries"),
            py::arg("base"),
            py::arg("qidx"),
            py::arg("ids"),
            py::arg("num_threads") = 1
        )
        .def("set_disable_prune", &E8qIndex::set_disable_prune, py::arg("on"))
        .def("set_prune_alpha", &E8qIndex::set_prune_alpha, py::arg("alpha"))
        .def("save", &E8qIndex::save, py::arg("path"))
        .def_static("load", &E8qIndex::load, py::arg("path"))
        .def_property_readonly("dim", &E8qIndex::dim)
        .def_property_readonly("nbits", &E8qIndex::nbits)
        .def_property_readonly("is_built", &E8qIndex::is_built)
        .def_property_readonly("n", &E8qIndex::n)
        .def_property_readonly("num_clusters", &E8qIndex::num_clusters)
        .def_property_readonly("memory_bytes", &E8qIndex::memory_bytes)
        .def_property_readonly("disable_prune", &E8qIndex::disable_prune);
}
