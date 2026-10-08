// E8Q IVF index: build, persistence, streamed search.
//
// Storage (cluster-major; per 32-vector batch):
//   scan record  : [ortho header 128 B: base/weight/perp u8 banks (32 B each)
//                   + 8 fp32 scales][rowA: blocks*32 B, one byte per lane per
//                   block]
//   refine record: [low digit bits, vector-major][g1 fp32 * 32][h3 fp32 * 32]
//   ids          : global point ids, cluster-major
// plus the FhtKac-4 rotator and rotated centroids for routing.
//
// One two-stage pipeline at every rate b in [1, 13]:
//   * Coarse gate: rowA prices ONE byte per block. The byte carries the digit
//     top bit of dims 0..6; bit 0 is the BRANCH at b == 1 (dim 7's sign is the
//     E8 parity of digits 0..6) and dim 7's top bit at b >= 2. The scan LUT
//     prices per-dim branch-averaged class means w_cls of the top-bit class:
//     branch-blind, one LUT, one stream, for every b.
//   * Refine: exact E8 reconstruction. b >= 2 stores the low (b-1) bits of
//     digits 0..6 plus [branch | digit-7 bits 1..b-2] in digit 7's slot (its
//     true LSB is parity-derived); storage is exactly b bits/dim. b == 1
//     decodes the sign byte directly (digit 7 = parity, branch = bit 0).
//   * Prune bound (zero-parallel residual + union bound): per vector,
//       dlo = d2 + base - W_x * S - |W|*lut_err - W_round
//             - lambda * r_horizon * Omega
//     W_x = 2*rn/<u,w_scan> so <u,R>==0 (NN bias rides on S, not base);
//     lut_err = 2 sqrt(blocks) * LUT quantum (CLT, not linear M);
//     lambda = PhiInv(1 - alpha / k_eff), k_eff = max(4k, 32);
//     r_horizon = min(sqrt(d2), sqrt(distk0)). alpha = prune_alpha, a
//     disclosed per-query prune-loss budget; no calibration.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#include "e8qlib/utils/aligned_alloc.hpp"
#include "e8qlib/utils/rotator.hpp"

namespace e8qlib {

// Maximum top-k per Search call. The per-query heaps and the sort scratch are
// sized by this; callers that need a larger k must re-rank in two passes.
constexpr size_t kMaxK = 128;

// Build-time knobs. The effective values are persisted with the index.
struct E8qBuildOptions {
    float prune_alpha = 0.0F;  // per-query prune-loss budget for the union bound (0 = 1.0)
};

namespace search_simd {
struct Access;
}

struct SearchStats {
    size_t candidates = 0;  // logical points scanned by the coarse pass
    size_t survivors = 0;   // lanes that entered exact refinement
    double rotate_us = 0.0;
    double prepare_route_us = 0.0;
    double scan_us = 0.0;
    double refine_us = 0.0;
};

// One candidacy row produced by the coarse scan, refined (exactly re-scored)
// later. Lives at namespace scope because QueryScratch stages rows across
// clusters before the flush (see FlushSurvivors). POD; size matters, it is
// copied per survivor.
struct SurvivorRow {
    // b >= 2: this lane's low-digit bits at block 0 (block b at +b*(bits-1)
    // bytes, vector-major). b == 1: unused (the rowA byte is the whole code).
    const uint8_t* refine;
    // rowA base of the batch record (byte per lane at blk*32 + lane).
    const uint8_t* plane;
    uint32_t lane;  // in-batch lane index
    float g1;
    // h3 with the row's cluster d2 ALREADY FOLDED IN (h3 := d2 + h3 at push):
    // d2 and h3 only ever appear as the sum in refine, and folding makes the
    // row self-contained for cross-cluster staging.
    float h3;
    uint32_t id;
};

// Per-thread scratch; reuse across queries (no allocation inside Search).
class QueryScratch {
   public:
    // Re-read by every one of the kc route dot products: cache-line aligned.
    AlignedVector<float> rot_q;       // padded rotated query
    // Stage-1 LUT over the rowA bytes (per block, 64 u8: low nibble = dims
    // 4..7, high nibble = dims 0..3, each 16-entry half duplicated), the
    // w_cls class means priced by the query, symmetric u8 quantization.
    std::vector<uint8_t> lut_c;
    // lut_c pre-permuted into the pair kernels' lo/hi register layout
    // (128 B per block pair: [lo | hi], see FillPairLut): the scan hot loop
    // then loads its two LUT registers with plain 512-bit loads instead of
    // rebuilding them with shuffle_i64x2 at every block pair of every batch.
    std::vector<uint8_t> lut_pair;
    float prep_delta_c = 0.0F;        // lut_c quantization step
    float prep_base_c = 0.0F;         // per-lane score = delta*S + base
    // PrepareQuery float stages, pre-sized once and reused.
    std::vector<float> cf_buf;
    std::vector<float> probe_dist;    // num_clusters
    std::vector<uint32_t> probe_id;   // num_clusters
    std::vector<uint64_t> probe_key;  // packed (dist bits, id) selection keys
    bool disable_prune = false;       // test hook: always refine every lane
    float ortho_z = 0.0F;             // union-bound quantile, computed per query
    float ortho_lut_err = 0.0F;       // per-query LUT rounding bound
    float ortho_score_max = 0.0F;     // per-query |S| bound (W-rounding cover)
    std::vector<float> heap_dist;     // k
    std::vector<uint32_t> heap_id;    // k
    size_t heap_count = 0;
    double refine_ns = 0.0;           // per-probe refine time accumulator
    // Cross-cluster survivor staging (see FlushSurvivors): rows accumulate
    // here until the flush threshold, so RefineBatch always runs at a size
    // that keeps the four-way gather pipeline busy.
    std::vector<SurvivorRow> staged_rows;
    size_t staged_n = 0;

    friend class E8qIndex;
};

class E8qIndex {
   public:
    void Build(
        const float* base,
        const float* centroids,
        const uint32_t* cluster_ids,
        size_t n,
        size_t dim,
        size_t num_clusters,
        int bits,
        uint64_t seed,
        size_t threads,
        const E8qBuildOptions& options = {}
    );

    void Save(const char* path) const;
    void Load(const char* path);

    // Top-k search over `nprobe` lists; out_ids receives k ids.
    void Search(
        const float* query, size_t k, size_t nprobe, uint32_t* out_ids, QueryScratch& scratch,
        SearchStats* stats = nullptr, float* out_dists = nullptr
    ) const;

    // Exact estimator over every vector (no routing, no pruning). Same
    // estimator as Search's refinement. Tests and exhaustive probes.
    void ExhaustiveTopK(
        const float* query, size_t k, std::vector<std::pair<float, uint32_t>>& out
    ) const;

    // Estimated squared L2 of one stored vector (accuracy protocol).
    [[nodiscard]] float Estimate(const float* query, uint32_t id) const;

    // Diagnostic probe for the ZPR screening bound (coverage study).  For the
    // pair (query, raw vector with global id `id`) it returns the normalized
    // coarse-to-full discrepancy
    //     w = (d_c^2 - d^2) / (r_q * Omega),
    // where d^2 is the exact estimator, d_c^2 the Coarse-Plane estimator,
    // r_q = ||query - centroid||_2 the query residual norm and
    // Omega = ||R_x||_2 / sqrt(D - 1) the orthogonal discrepancy scale.  The
    // screening bound of Theorem 3 (d^2 >= d_c^2 - r_q * lambda * Omega) is
    // violated exactly when w > lambda, so one pass of w gives the empirical
    // violation rate at every tail quantile lambda.
    //
    // Both estimates are recomputed in double precision from the stored codes
    // (no implementation margin, no fixed-point LUT error): the probe measures
    // the mathematical bound, not the padded kernel bound.  `base_row` must be
    // the raw build input of `id`; returns NaN for a degenerate pair.
    [[nodiscard]] double ProbeBoundW(
        const float* query, const float* base_row, uint32_t id
    ) const;

    // When true, Search copies the flag onto QueryScratch so every lane is
    // refined (ablation: no coarse-to-refine screen).
    void SetDisablePrune(bool on) { disable_prune_ = on; }
    [[nodiscard]] bool disable_prune() const { return disable_prune_; }
    // Search-time override of the union-bound prune-loss budget (the lambda
    // quantile is evaluated per query, so the budget is a query-time knob;
    // Build only persists the default).
    void SetPruneAlpha(float alpha) { ortho_alpha_ = alpha; }

    [[nodiscard]] size_t n() const { return n_; }
    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t padded_dim() const { return padded_dim_; }
    [[nodiscard]] int bits() const { return bits_; }
    [[nodiscard]] size_t num_clusters() const { return num_clusters_; }
    [[nodiscard]] size_t memory_bytes() const;

    // Scan-record ortho header: u8 banks base/weight/perp at 0/32/64
    // (32 lanes each) and fp32 scales at 96 (sc[0..5] = base/weight/perp
    // min and step pairs, sc[6] = max |weight|). Layout is part of the
    // on-disk record; kernels and tests read it.
    static constexpr size_t kH3QOff = 0;
    static constexpr size_t kCqOff = 32;
    static constexpr size_t kAqOff = 64;
    static constexpr size_t kScalesOff = 96;
    static constexpr size_t kHdrQBytes = 128;

   private:
    friend struct search_simd::Access;
    size_t n_ = 0;
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    size_t num_clusters_ = 0;
    int bits_ = 0;
    int blocks_ = 0;  // padded_dim / 8

    FhtKac4Rotator rotator_;
    // Route GEMV source, streamed once per query with 512-bit loads: kept
    // cache-line aligned so no load straddles two lines.
    AlignedVector<float> centroid_rot_;  // num_clusters * padded_dim
    AlignedVector<float> centroid_n2_;   // num_clusters
    std::vector<uint64_t> cluster_begin_;  // num_clusters + 1 (vector offsets)

    // CSR-P32 scan arena: one contiguous record per 32-vector batch,
    //   [ortho header 128 B][rowA: blocks * 32 B]
    // so the batch scan streams a compact record with no cold bytes in the
    // stride. The survivor-only refine data (low digit bits + g1/h3) lives in
    // a separate refine arena read only for the ~1% of lanes that survive
    // the coarse mask.
    AlignedVector<uint8_t> arena_;
    std::vector<uint64_t> arena_begin_;  // num_clusters + 1 (byte offsets)
    // rot(mu) of the build set's mean: subtracted from every rotated vector
    // (base, centroids, queries) so the banks/LUTs see a centered domain.
    AlignedVector<float> mean_rot_;
    size_t batch_bytes_ = 0;             // scan record bytes (64 B aligned)
    AlignedVector<uint8_t> ref_arena_;
    std::vector<uint64_t> ref_arena_begin_;  // num_clusters + 1 (byte offsets)
    size_t ref_batch_bytes_ = 0;             // refine record bytes (64 B aligned)

    // Level tables, read with 512-bit loads at b >= 6: 64-byte aligned so a
    // load never straddles two cache lines.
    AlignedVector<float> refine_tab_;  // 2*stride level tables (stride 256 at b<=8, 2^b above)
    // Scan class-mean basis: gate_mid_[2*d + t] = branch-averaged mean level
    // of digit top-bit class t (same value for every dim, one scalar per
    // class; at b == 1 dim 7 is 0 — that scan bit is the BRANCH, not the
    // parity sign). InitBitTables() fills it once at Build/Load so the
    // query path runs zero math-library work for it.
    std::array<double, 16> gate_mid_{};
    void InitBitTables();
    bool disable_prune_ = false;
    // Effective build options (persisted with the index for reproducibility).
    float ortho_alpha_ = 1.0F;  // per-query prune-loss budget (union bound)
    mutable std::vector<uint32_t> inv_ids_;  // global id -> cluster-major offset
    mutable std::mutex inv_ids_mu_;          // guards the lazy inv_ids_ fill
    AlignedVector<uint32_t> ids_;  // per vector, cluster-major

    [[nodiscard]] size_t P1Off() const { return kHdrQBytes; }
    // Refine-arena record: [low digit bits (vector-major, (b-1) B per block
    // per lane; empty at b == 1)][g1[32] fp32][h3[32]]. Digit top bits come
    // from the scan plane byte, which the scan has just touched.
    [[nodiscard]] size_t RefRefineOff() const { return 0; }
    [[nodiscard]] size_t RefG1Off() const {
        return RefRefineOff() + RefineStride() * 32;
    }
    [[nodiscard]] size_t RefH3Off() const { return RefG1Off() + 128; }
    // Whole refine-record bytes: refine payload + fp32 g1/h3.
    [[nodiscard]] size_t RefRecordBytes() const { return RefH3Off() + 128; }
    // PER-LANE refine payload bytes: (b-1) per block, vector-major
    // ((lane*blocks + blk) * (b-1)); empty at b == 1. The batch's 32-lane
    // region is RefG1Off()'s `* 32`.
    [[nodiscard]] size_t RefineStride() const {
        return bits_ >= 2
                   ? static_cast<size_t>(blocks_) * static_cast<size_t>(bits_ - 1)
                   : 0;
    }
    void PrepareQuery(const float* rot_q, QueryScratch& s) const;
    // Full-dim fp32 L2 against every centroid, then partial_sort top-nprobe
    // (RaBitQ FlatInitializer shape). No int8, no dim prefix, no rerank cap.
    void RouteQuery(const float* rot_q, size_t nprobe, QueryScratch& s) const;
    [[nodiscard]] float ExactScore(
        const uint8_t* ref_vec, const uint8_t* plane_vec, uint32_t lane, const float* q
    ) const;
    // Exact re-scoring of staged survivors. Rows carry h3 with d2 already
    // folded in (see SurvivorRow), so no cluster-level d2 parameter.
    void RefineBatch(const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s) const;
    // Flush the staged survivor rows through the exact refiner: cold-start
    // top-C cap, then RefineBatch.
    void FlushSurvivors(QueryScratch& s, const float* q, SearchStats* stats) const;
    void InsertHeap(float dist, uint32_t id, QueryScratch& s) const;
    void ExtractTopK(
        QueryScratch& s, size_t k, uint32_t* out_ids, float* out_dists = nullptr
    ) const;
    void ScanCluster(
        size_t cid, float d2, const float* q, QueryScratch& s, SearchStats* stats
    ) const;

    // Arena-addressed row pointers for one vector of a known cluster.
    [[nodiscard]] SurvivorRow RowsAt(size_t c, size_t i) const {
        const uint8_t* rp = ref_arena_.data() + ref_arena_begin_[c] +
                            (i / 32) * ref_batch_bytes_;
        const float* g1b = reinterpret_cast<const float*>(rp + RefG1Off());
        const float* h3b = reinterpret_cast<const float*>(rp + RefH3Off());
        const size_t lane = i % 32;
        const uint8_t* pr =
            arena_.data() + arena_begin_[c] + (i / 32) * batch_bytes_ + P1Off();
        return SurvivorRow{
            bits_ >= 2
                ? rp + RefRefineOff() + lane * RefineStride()
                : nullptr,
            pr,
            static_cast<uint32_t>(lane),
            g1b[lane], h3b[lane],
            ids_[static_cast<size_t>(cluster_begin_[c]) + i]
        };
    }
};

// ---------------------------------------------------------------------------
// Coarse-payload helpers (shared by the index and the white-box tests).
// Pack one block's rowA byte: bit (7-i) = digit top bit of dim i for i in
// 0..6; bit 0 = the branch at b == 1 (dim 7's sign is parity-derived) and
// dim 7's top bit at b >= 2. `row` must have blocks bytes.
void PackCoarseRow(
    const uint16_t digits[8], int branch, int bits, int blk, uint8_t* row
);
// Reconstruct one (lane, block) of the rowA + refine payload into true
// digits (+ branch): the low (b-1) field bits per dim plus the rowA top bit;
// digit 7's LSB is the parity of digits 0..6 and its stored field bit 0 is
// the branch. The 16-bit variant is the b = 9..13 twin (digit values above
// 255, refine fields up to 12 bytes).
void DecodeSplitBlockLane(
    int bits, const uint8_t* rowa_batch, const uint8_t* refine_lane, uint32_t blk,
    uint32_t lane, uint8_t digits[8], int* branch
);
void DecodeSplitBlockLane(
    int bits, const uint8_t* rowa_batch, const uint8_t* refine_lane, uint32_t blk,
    uint32_t lane, uint16_t digits[8], int* branch
);

// RowA accumulate: one u8 LUT stream (64 B per block) over the byte-per-lane
// rows; the per-lane u8 sums come back i32-exact across backends.
void AccumulateCoarseHi(
    const uint8_t* __restrict__ payload, const uint8_t* __restrict__ lut_c,
    const uint8_t* __restrict__ lut_pair, size_t blocks, int32_t out[32]
);
// Permute lut_c into the pair kernels' lo/hi layout once per query:
// lut_pair[128*p .. 64) = [a.q0 a.q1 b.q0 b.q1], [64..128) = [a.q2 a.q3 b.q2
// b.q3] with a/b the pair's two 64-byte lut_c lines. Same bytes, same order
// the kernels' own shuffles produced; odd trailing blocks are not permuted
// (the tail path reads lut_c directly).
void FillPairLut(const uint8_t* __restrict__ lut_c, size_t blocks, uint8_t* __restrict__ lut_pair);
// Two-batch form: one block loop scans two consecutive 32-lane batches and
// shares the per-block LUT loads (the LUT stream is the majority of the
// single-batch kernel's loads, so pairing cuts the scan's load-port pressure).
void AccumulateCoarseHiPair(
    const uint8_t* __restrict__ payload1, const uint8_t* __restrict__ payload2,
    const uint8_t* __restrict__ lut_c, const uint8_t* __restrict__ lut_pair, size_t blocks,
    int32_t out1[32], int32_t out2[32]
);

}  // namespace e8qlib
