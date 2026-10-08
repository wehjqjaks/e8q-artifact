// ME8Q: SAQ's mixed-precision IVF pipeline with the CAQ encoder replaced by
// the E8Q lattice codec. Segmentation, DP bit allocation, per-segment
// factors, and the search path follow SAQ; only the codebook is E8Q.
//
// Storage per vector (cluster-major, per-list stride): the plan's segment
// codes at b bits/dim, plus one fp32 (amplitude coefficient, h2) pair per
// coded segment — the same 64-bit-per-segment factor budget SAQ charges.
//
// Estimator (segment-additive E8Q estimator, per segment s of residual
// r = x - c, c = centroid, q = query):
//   D_hat = ||q - c||^2 + sum_s [ h2_s - 2 * coef_s * S_s(q) ]
//   S_s(q) = sum_{i in s} q_i * level[br_i][digit_i]   (raw level units)
//   coef_s = ||r_s||^2 / <r_s, levels_s>  (dot-normalized fit)
//   h2_s   = ||r_s||^2 + 2 * coef_s * <c_s, levels_s>
#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "e8qlib/quantization/me8q_plan.hpp"
#include "e8qlib/quantization/me8q_codec.hpp"
#include "e8qlib/utils/aligned_alloc.hpp"
#include "e8qlib/utils/rotator.hpp"

namespace e8qlib::me8q {

struct BuildOptions {
    float avg_bits = 4.0F;
    // "L:B,L:B,..." manual plan (dims:bits, must cover the padded dimension),
    // for tests the allocator cannot express. Overrides SAQ DP.
    std::string plan_spec;
    // Seed of the per-run FHT–Kac-4 rotators (same geometry as E8Q's
    // whole-vector rotation, applied per coded plan run).
    uint64_t rot_seed = 0x5EED1234ULL;
    int max_bits = kMaxBits;
    // Fold the true residual energy of the dropped (0-bit) dimensions into
    // the last coded run's h2: the estimate of a dropped run is then its
    // minimum-MSE value (energy, no alignment) instead of plain zero.
    bool drop_energy = true;
    // Union-bound per-query prune-loss budget of the ZPR bound
    // (lambda = PhiInv(1 - alpha/k_eff), k_eff = max(4k, 32); 0 = 0.1).
    float prune_alpha = 0.0F;
    // Byte-matching pad (bits per vector) added to the DP budget so the
    // plan's total spend equals a target method's actual per-vector
    // memory (see PlanInput::extra_bits). 0 = SAQ-parity nominal budget.
    double extra_budget_bits = 0.0;
    size_t threads = 1;
};

struct SearchStats {
    size_t candidates = 0;
    size_t clusters = 0;
    double route_us = 0.0;
    double scan_us = 0.0;
    double rotate_us = 0.0;  // per-run query rotation
    double lut_us = 0.0;     // per-query u8 FastScan LUT build + quantisation
    double filter_us = 0.0;  // vectorized coarse prune over lanes
    double lane_us = 0.0;    // scalar lane pass (coarse recompute + cascade)
    double rest_us = 0.0;    // refine-window remainder (batch bookkeeping)
    double refine_us = 0.0;  // lane loop: coarse score + certified cascade
    size_t survivors = 0;   // lanes that entered exact refinement (fast path)
    size_t bits_scored = 0;
    size_t bits_total = 0;
    size_t pruned = 0;  // vectors whose tail was skipped
};

class Me8qIndex {
   public:
    // `base` is row-major with leading dimension `ld` (the PCA-domain
    // dimension); it is read only during Build. Dimensions beyond `dim` are
    // treated as zero (the padded domain the plan covers).
    void Build(
        const float* base, size_t ld, const float* centroids, size_t ld_c,
        const uint32_t* cluster_ids, size_t n, size_t dim, size_t num_clusters,
        const BuildOptions& opt, const float* data_variance = nullptr
    );

    void Save(const char* path) const;
    void Load(const char* path);

    void SearchIVF(
        const float* query, size_t k, size_t nprobe, uint32_t* out_ids,
        SearchStats* stats = nullptr, float* out_dists = nullptr
    ) const;

    // Two-level scan. Every coded block is stored as a 1-bit top-bit plane
    // (nibble-LUT FastScan over 32-vector batches) + (rate-1)-bit refine rows.
    // Build/Load call BuildFastLayout(); the public method remains for tests.
    void BuildFastLayout();
    // Single-query FastScan path (paper serving): 1-bit coarse plane, then
    // per-segment prune/cascade, then exact refine of survivors. Routes over
    // the plan's coded prefix with fp32 L2 (`ivf::RouteL2`).
    void SearchIVFFast(
        const float* query, size_t k, size_t nprobe, uint32_t* out_ids,
        SearchStats* stats = nullptr, float* out_dists = nullptr
    ) const;
    // Batched FastScan path: identical results query for query, but the route
    // phase sweeps the centroid matrix once per group of queries
    // (`ivf::RouteL2Group`) instead of once per query, and the three phases
    // (rotate / route / scan) run as separate parallel regions so the sweep
    // is shared. Used by the Python binding for nq > 1.
    void SearchIVFFastBatch(
        const float* queries, size_t nq, size_t k, size_t nprobe, uint32_t* out_ids,
        SearchStats* stats, float* out_dists, int num_threads
    ) const;
    // Scan phase of SearchIVFFast: everything after routing, for one query.
    // `rot_qv` is the padded, rotated query; `cd`/`take` the probe list.
    void ScanPreRouted(
        const float* rot_qv, const std::pair<float, uint32_t>* cd, size_t take, size_t k,
        uint32_t* out_ids, SearchStats* stats, float* out_dists, double route_us,
        double rot_us
    ) const;
    // Exact fp32 routing over the padded, rotated query (one query).
    void RouteQuery(
        const float* q, size_t nprobe, std::vector<std::pair<float, uint32_t>>* cd
    ) const;

    // Estimated-distance top-k over every vector (no routing).
    void ExhaustiveTopK(
        const float* query, size_t k, std::vector<std::pair<float, uint32_t>>* out
    ) const;

    [[nodiscard]] size_t n() const { return n_; }
    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t padded_dim() const { return padded_dim_; }
    // Padded dim minus the plan's trailing zero-bit dims: the route only needs
    // the coded prefix (the tail is exact zeros in the query and the rotated
    // centroids).
    [[nodiscard]] size_t route_dim() const { return route_dim_; }
    [[nodiscard]] size_t num_clusters() const { return num_clusters_; }
    [[nodiscard]] const Plan& shared_plan() const { return shared_plan_; }
    [[nodiscard]] double bits_per_dim() const;
    [[nodiscard]] size_t memory_bytes() const;
    // Estimator value and per-segment factors for one stored vector (accuracy protocol).
    [[nodiscard]] float Estimate(const float* query, uint32_t id) const;

   private:
    struct SegSlot {
        uint32_t code_off = 0;    // byte offset of the segment code in the row
        uint32_t factor_off = 0;  // byte offset of the (coef, h2) pair
        uint32_t dim_off = 0;     // dimension offset of the run
        uint16_t len = 0;         // segment dimensions
        uint8_t bits = 0;         // 0 = dropped (no slot emitted)
    };
    struct ListLayout {
        std::vector<SegSlot> segs;
        size_t stride = 0;  // bytes per vector row
    };
    struct FastLayout {
        std::vector<uint32_t> run_rate;
        std::vector<uint32_t> run_blk0;
        std::vector<uint32_t> run_nblk;
        std::vector<uint32_t> run_dim_off;
        std::vector<uint32_t> run_len;
        std::vector<uint8_t> block_run;
        size_t nrun = 0;
        size_t blocks_total = 0;
        size_t plane_cols = 0;
        size_t plane_bytes = 0;
        size_t batches = 0;
        size_t cnt = 0;
        size_t refine_stride = 0;
        std::vector<uint32_t> refine_run_off;
        // Scan record per batch: [plane_cols*32 data rows][per run: 128 B
        // of u8 banks (base/weight/perp + unused 4th) + 32 B of 8 fp32
        // scales]. Search uses the first three banks (ZPR: base, W, Omega).
        std::vector<uint8_t> plane;
        std::vector<uint8_t> refine;
        // Per-vector per-run ZPR banks packed so
        //   weight = 2*coef*gain,  base = h2 - 2*coef*vpc
        // matches the cascade undo (signed coef) and, for the usual
        // coef > 0, yields W_x and rn^2 + W_x <c',w> (no b_i). perp
        // stores Omega / (2*|coef|) with Omega = ||R_x|| / sqrt(L_r - 1).
        std::vector<float> ortho_gain;
        std::vector<float> ortho_vpc;
        std::vector<float> ortho_perp;
        // Per-run centroid-slice energy ||rc_r||^2 (the per-query d2_r term).
        std::vector<double> rc_n2;
        size_t bank_run_bytes = 160;  // per-run bank region (128B banks + 32B scales)
    };

    [[nodiscard]] const float* Row(size_t c, size_t i) const {
        return reinterpret_cast<const float*>(
            codes_.data() + byte_begin_[c] + i * layouts_[c].stride
        );
    }
    [[nodiscard]] double EstimateRow(
        const float* row, const ListLayout& lay, double d2, const float* q_rot
    ) const;
    void EstimateRow4(
        const uint8_t* const rows[4], const ListLayout& lay, double d2,
        const float* q_rot, double out[4]
    ) const;
    void PrepareRotatedQuery(const float* q, float* rot) const;
    void PrepareCentroidsRotated();
    void PadQuery(const float* query, std::vector<float>* padded) const;
    void PushCandidate(
        float dist, uint32_t id, size_t k, std::vector<std::pair<float, uint32_t>>* out
    ) const;

    size_t n_ = 0;
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    size_t num_clusters_ = 0;
    BuildOptions opt_;
    SegmentCodec codec_;
    // One FHT–Kac-4 per coded plan run, spanning the whole run. A run is the
    // unit the single per-run amplitude factor prices, so the rotation must
    // spread the run's energy over all of its dimensions (per-64-tile
    // rotation leaves block energies segregated and cost ~33% extra relative
    // estimator error on uniform plans).
    std::vector<FhtKac4Rotator> run_rot_;
    void BuildRunRotators();
    // 64B-aligned (AVX-512 route GEMV streams cent_rot_ a line at a time;
    // the coarse accumulator's unpack stores are vmovdqa32 — the e8q A1
    // alignment fix applies to these members too).
    AlignedVector<float> cent_rot_;
    size_t route_dim_ = 0;
    AlignedVector<float> centroid_n2_;

    std::vector<float> centroids_;
    AlignedVector<uint32_t> ids_;
    std::vector<uint64_t> list_begin_;
    std::vector<uint64_t> byte_begin_;
    std::vector<uint8_t> codes_;
    std::vector<ListLayout> layouts_;
    Plan shared_plan_;
    size_t bytes_per_vector_ = 0;
    std::vector<FastLayout> fast_;
    bool fast_ready_ = false;
    // Per-query prune-loss budget of the orthogonal-projection union bound.
    float ortho_alpha_ = 1.0F;
    std::array<SegmentCodec::ClassModel, kMaxBits + 1> fast_cls_ = {};
    mutable std::vector<uint32_t> id_to_row_;
    mutable std::vector<uint32_t> id_to_list_;
    mutable std::mutex id_map_mu_;
    mutable bool id_map_ready_ = false;
    void EnsureIdMap() const;

    [[nodiscard]] const Plan& PlanOf(size_t /*c*/) const { return shared_plan_; }
};

}  // namespace e8qlib::me8q
