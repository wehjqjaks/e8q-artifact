#include "e8qlib/quantization/e8q_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace e8qlib {
namespace {

// Acklam's inverse normal CDF (|rel err| < 1.15e-9), same rational
// approximation as the reference implementation used by the production e8q
// encoder.
double InvNormalCdf(double p) {
    static constexpr double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                    -2.759285104469687e+02, 1.383577518672690e+02,
                                    -3.066479806614716e+01, 2.506628277459239e+00};
    static constexpr double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                    -1.556989798598866e+02, 6.680131188771972e+01,
                                    -1.328068155288572e+01};
    static constexpr double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                    -2.400758277161838e+00, -2.549732539343734e+00,
                                    4.374664141464968e+00, 2.938163982698783e+00};
    static constexpr double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                    2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double kPlow = 0.02425;
    if (p < kPlow) {
        const double q = std::sqrt(-2 * std::log(std::max(p, 1e-300)));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p <= 1 - kPlow) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    const double q = std::sqrt(-2 * std::log(std::max(1 - p, 1e-300)));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

// Bounded-bin index over the level table. NearestInTable was 69% of the
// encoder: 32 calls per 8-dimension block, each a log2(2^(b-1)) chain of
// dependent loads. The level values are quantiles of one normal, so their
// smallest gap is known once the table is built; a uniform bin no wider than
// that gap leaves at most two levels inside a bin, and the insertion point is
// then a couple of compare-adds from the bin's own bracket. Exact by
// construction -- bin_lo[i] is the true lower_bound at the bin's left edge, so
// the insertion point for any value in the bin lies in [bin_lo[i],
// bin_lo[i+1]] -- and the binary search stays as the fallback for rates whose
// bins would outgrow the cap.
constexpr size_t kLutMaxBins = 16384;
// The merged table interleaves four classes, so its levels sit about twice as
// close and span about twice the range of one parity table's: holding a bin to
// the tightest gap costs it roughly four times the bins.  Its own cap keeps
// rate 12 (25k bins) on the index instead of dropping it to the binary search,
// which measured 37-43% slower there.
constexpr size_t kLutMaxBinsMerged = 32768;
// Widest bin the bin index may produce, in levels.
constexpr size_t kLutMaxSpan = 2;

// The bounded-bin index, shared by the per-parity tables and the merged table
// below.  Builds bin_lo over [lev[0], lev[n-1]] and reports whether the result
// is usable: bins no wider than the tightest level gap, so the walk that
// resolves a bin steps at most kLutMaxSpan times.
void BuildBinIndexRaw(
    const double* lev, size_t n, std::vector<uint32_t>& bin_lo, double& bin_lo_v,
    double& bin_scale, size_t& nbins, bool& usable, size_t max_bins = kLutMaxBins
) {
    usable = false;
    bin_lo.clear();
    // Below this size the binary search costs less than the bin lookup.
    if (n < 16) {
        return;
    }
    const double lo_v = lev[0];
    const double hi_v = lev[n - 1];
    if (!(hi_v > lo_v)) {
        return;
    }
    double min_gap = std::numeric_limits<double>::max();
    for (size_t i = 1; i < n; ++i) {
        min_gap = std::min(min_gap, lev[i] - lev[i - 1]);
    }
    if (!(min_gap > 0.0)) {
        return;
    }
    const double range = hi_v - lo_v;
    size_t bins = 1;
    while (static_cast<double>(bins) * min_gap < range && bins < max_bins) {
        bins *= 2;
    }
    // Only usable while a bin really is no wider than the tightest level gap;
    // past the cap the bracket would stop being a couple of entries wide and
    // the binary search is the better search.
    if (static_cast<double>(bins) * min_gap < range) {
        return;
    }
    nbins = bins;
    bin_lo_v = lo_v;
    bin_scale = static_cast<double>(bins) / range;
    bin_lo.assign(bins + 1, 0);
    size_t j = 0;
    for (size_t b = 0; b <= bins; ++b) {
        const double edge = lo_v + static_cast<double>(b) / bin_scale;
        while (j < n && lev[j] < edge) {
            ++j;
        }
        bin_lo[b] = static_cast<uint32_t>(j);
    }
    // A bin holds span <= 2 levels, which InsertPos then resolves with two
    // unrolled compare-adds; anything wider means the cap bit and the binary
    // search is the better search.
    size_t max_span = 0;
    for (size_t b = 0; b < bins; ++b) {
        max_span = std::max(max_span, static_cast<size_t>(bin_lo[b + 1] - bin_lo[b]));
    }
    usable = max_span <= kLutMaxSpan;
}

struct ParityTable {
    std::vector<double> lev;       // sorted level values, then +inf padding
    std::vector<uint16_t> dig;     // digit of each sorted level
    std::vector<uint32_t> bin_lo;  // size bins+1; empty when !usable
    size_t n = 0;                  // real level count (lev holds n + 2 entries)
    double bin_lo_v = 0.0;         // value at bin 0's left edge (lev.front())
    double bin_scale = 0.0;        // bins / (lev.back() - lev.front())
    size_t nbins = 1;
    bool usable = false;
    bool ready = false;
};

void BuildBinIndex(ParityTable& t) {
    BuildBinIndexRaw(t.lev.data(), t.n, t.bin_lo, t.bin_lo_v, t.bin_scale, t.nbins,
                     t.usable);
}

const ParityTable& GetParityTable(
    const std::vector<double>& levels, int branch, bool odd
) {
    // Keyed by (bits, branch, parity): branch-0 and branch-1 level sets
    // differ, so both must be cached separately. Index 0 unused; 1..13.
    static thread_local ParityTable tables[kMaxQuantizeBits + 1][2][2];
    const int bits = __builtin_ctz(static_cast<unsigned>(levels.size()));
    ParityTable& t = tables[bits][branch][odd ? 1 : 0];
    if (!t.ready) {
        const int q = 1 << bits;
        t.lev.clear();
        t.dig.clear();
        for (int d = odd ? 1 : 0; d < q; d += 2) {
            t.lev.push_back(levels[static_cast<size_t>(d)]);
            t.dig.push_back(static_cast<uint16_t>(d));
        }
        // insertion sort by (level, digit) - ties keep the smaller digit first
        const size_t n = t.lev.size();
        for (size_t i = 1; i < n; ++i) {
            double lv = t.lev[i];
            uint16_t dg = t.dig[i];
            size_t j = i;
            while (j > 0 && (t.lev[j - 1] > lv ||
                             (t.lev[j - 1] == lv && t.dig[j - 1] > dg))) {
                t.lev[j] = t.lev[j - 1];
                t.dig[j] = t.dig[j - 1];
                --j;
            }
            t.lev[j] = lv;
            t.dig[j] = dg;
        }
        t.n = n;
        BuildBinIndex(t);
        // Two +inf past the last level. The bin walk below steps up to twice
        // from the bin's bracket start -- `usable` caps a bin at two levels, so
        // it can reach index n but never n + 1 -- and the padding makes those
        // reads defined and false instead of a bounds test. Appended after the
        // index build so it is invisible to it, and counted out of `n` so no
        // search sees it.
        t.lev.push_back(std::numeric_limits<double>::infinity());
        t.lev.push_back(std::numeric_limits<double>::infinity());
        t.ready = true;
    }
    return t;
}

// First index with lev[i] >= v, the same insertion point std::lower_bound
// returns (ties therefore resolve exactly as before).
inline size_t InsertPos(const ParityTable& t, double v) {
    const double* const lev = t.lev.data();
    const size_t n = t.n;
    if (t.usable) {
        if (v <= t.bin_lo_v) {
            return 0;
        }
        if (v > lev[n - 1]) {
            return n;
        }
        size_t bin = static_cast<size_t>((v - t.bin_lo_v) * t.bin_scale);
        if (bin >= t.nbins) {
            bin = t.nbins - 1;
        }
        // Two steps out of the bin's own bracket, unguarded: `usable` caps a
        // bin at two levels, so the walk can only reach index n, and the +inf
        // padding makes that read defined and false rather than a bounds test.
        // The walk is monotone, so it cannot step past the true insertion
        // point, at which lev[pos] >= v already holds.
        // (Prefetching the bracket into a per-bin copy of the two level
        // values, so both loads address off `bin` instead of off the previous
        // comparison, measured 3-11% SLOWER: it adds 8 KB per table and the
        // dependent pair below is already covered by the surrounding
        // independent searches.)
        size_t pos = t.bin_lo[bin];
        pos += (lev[pos] < v) ? 1 : 0;
        pos += (lev[pos] < v) ? 1 : 0;
        return pos;
    }
    // Branchless binary search: the branchy lower_bound mispredicts on the
    // probe that splits on the sign of v, which is wrong about half the time.
    size_t pos = 0;
    size_t step = 1;
    while (step * 2 <= n) {
        step *= 2;
    }
    for (; step != 0; step >>= 1) {
        const size_t next = pos + step;
        pos += (next <= n && lev[next - 1] < v) ? step : 0;
    }
    return pos;
}

void NearestInTable(
    double v, const ParityTable& t, uint16_t* digit, double* level, double* cost
) {
    const double* const lev = t.lev.data();
    const size_t n = t.n;
    const size_t pos = InsertPos(t, v);
    size_t best = pos < n ? pos : n - 1;
    if (pos > 0) {
        const double dc = std::abs(v - lev[pos - 1]);
        const double db = std::abs(v - lev[best]);
        // Written as a single bitwise condition so the compiler emits a select
        // rather than a branch: which side of the interval v falls on is a coin
        // flip, and a mispredicting branch here costs more than the whole
        // search it follows.
        const uint16_t dig_c = t.dig[pos - 1];
        const uint16_t dig_b = t.dig[best];
        const bool take_below = (dc < db) | ((dc == db) & (dig_c < dig_b));
        best = take_below ? pos - 1 : best;
    }
    *digit = t.dig[best];
    *level = lev[best];
    const double dv = v - lev[best];
    *cost = dv * dv;
}

// --------------------------------------------------------------------------
// Merged level table: one search per coordinate instead of four.
//
// Branch j's table places digit d at the normal quantile
// 1/2 + (d - 2^(b-1) + eta_j)/2^b with eta_0 = 1/4 and eta_1 = 3/4, so for
// every digit the four levels the encoder searched through four separate
// tables interleave exactly,
//   level(d,0) < level(d,1) < level(d+1,0) < level(d+1,1),
// and the merged array of 2^(b+1) levels is the sorted union of all four.
// In that order the class of the level at position i -- branch (i & 1), digit
// parity (i >> 1) & 1 -- cycles with period four, so with the residues
// numbered
//   class r = 2 * branch + parity,
// the level of class r sits at positions i with (i & 3) == r.  One insertion
// point then brackets the nearest level of all four classes within four
// steps, and the four the encoder needs are simply r = 0..3.
//
// The build verifies that order rather than trusting it, and the encoder keeps
// the four-table path for any rate where the check fails.  The four front and
// four back sentinels make the class walk unguarded: a class's neighbour on
// either side is at most four positions away, so the reads stay inside the
// allocation and the -inf/+inf sentinel loses every distance comparison for
// the side that falls off the table.
constexpr long kMergedPad = 4;

struct MergedTable {
    std::vector<double> buf;        // 4 x -inf | 2^(b+1) levels | 4 x +inf
    std::vector<uint16_t> dig;      // digit per level, zeros on the padding
    std::vector<uint32_t> bin_lo;   // size bins+1; empty when !usable
    size_t n = 0;                   // 2^(b+1) real levels
    double bin_lo_v = 0.0;          // value at bin 0's left edge
    double bin_scale = 0.0;         // bins / range
    size_t nbins = 1;
    bool usable = false;
    bool alternating = false;       // the interleaving above was verified
    bool ready = false;
    const double* lev() const { return buf.data() + kMergedPad; }
};

const MergedTable& GetMergedTable(
    const std::vector<double>& l0, const std::vector<double>& l1
) {
    static thread_local MergedTable tabs[kMaxQuantizeBits + 1];
    const int bits = __builtin_ctz(static_cast<unsigned>(l0.size()));
    MergedTable& t = tabs[bits];
    if (t.ready) {
        return t;
    }
    const size_t q = l0.size();
    const size_t n = 2 * q;
    struct Entry {
        double v;
        uint16_t dig;
        uint8_t branch;
    };
    std::vector<Entry> raw;
    raw.reserve(n);
    for (size_t d = 0; d < q; ++d) {
        raw.push_back({l0[d], static_cast<uint16_t>(d), 0});
        raw.push_back({l1[d], static_cast<uint16_t>(d), 1});
    }
    std::stable_sort(raw.begin(), raw.end(), [](const Entry& a, const Entry& b) {
        return a.v < b.v;
    });
    bool alt = true;
    for (size_t i = 0; i < n; ++i) {
        if (i > 0 && !(raw[i].v > raw[i - 1].v)) {
            alt = false;
            break;
        }
        if (raw[i].branch != static_cast<uint8_t>(i & 1U) ||
            (raw[i].dig & 1U) != static_cast<uint16_t>((i >> 1) & 1U)) {
            alt = false;
            break;
        }
    }
    t.alternating = alt;
    const double inf = std::numeric_limits<double>::infinity();
    t.buf.assign(static_cast<size_t>(2 * kMergedPad) + n, inf);
    for (long i = 0; i < kMergedPad; ++i) {
        t.buf[static_cast<size_t>(i)] = -inf;
    }
    t.dig.assign(static_cast<size_t>(2 * kMergedPad) + n, 0);
    t.n = n;
    for (size_t i = 0; i < n; ++i) {
        t.buf[static_cast<size_t>(kMergedPad) + i] = raw[i].v;
        t.dig[static_cast<size_t>(kMergedPad) + i] = raw[i].dig;
    }
    if (alt) {
        BuildBinIndexRaw(t.lev(), n, t.bin_lo, t.bin_lo_v, t.bin_scale, t.nbins,
                         t.usable, kLutMaxBinsMerged);
    }
    t.ready = true;
    return t;
}

inline size_t InsertPosMerged(const MergedTable& t, double v) {
    const double* const lev = t.lev();
    const size_t n = t.n;
    if (t.usable) {
        if (v <= t.bin_lo_v) {
            return 0;
        }
        if (v > lev[n - 1]) {
            return n;
        }
        size_t bin = static_cast<size_t>((v - t.bin_lo_v) * t.bin_scale);
        if (bin >= t.nbins) {
            bin = t.nbins - 1;
        }
        // Two unguarded steps, exactly as in InsertPos: a bin holds at most
        // kLutMaxSpan levels and the +inf padding covers the read past the end.
        size_t pos = t.bin_lo[bin];
        pos += (lev[pos] < v) ? 1 : 0;
        pos += (lev[pos] < v) ? 1 : 0;
        return pos;
    }
    size_t pos = 0;
    size_t step = 1;
    while (step * 2 <= n) {
        step *= 2;
    }
    for (; step != 0; step >>= 1) {
        const size_t next = pos + step;
        pos += (next <= n && lev[next - 1] < v) ? step : 0;
    }
    return pos;
}

// Nearest level of each of the four classes for one coordinate, indexed by the
// class residue r of the merged order: r = 0 is branch 0 even-digit, 1 branch
// 1 even-digit, 2 branch 0 odd-digit, 3 branch 1 odd-digit.  dig[r] and
// cost[r] receive that class's nearest digit and squared distance -- the pair
// NearestInTable returns for one parity table.
inline void NearestMerged(double v, const MergedTable& t, uint16_t dig[4], double cost[4]) {
    const double* const lev = t.lev();
    const uint16_t* const dg = t.dig.data() + kMergedPad;
    const ptrdiff_t p = static_cast<ptrdiff_t>(InsertPosMerged(t, v));
    const ptrdiff_t p4 = p & 3;
    for (ptrdiff_t r = 0; r < 4; ++r) {
        // Largest index below the insertion point whose residue is r, and the
        // smallest at or above it: both are within four positions of p
        // because the classes cycle with period four.
        const ptrdiff_t i_b = p - 1 - ((p4 + 3 - r) & 3);
        const ptrdiff_t i_a = p + ((r + 4 - p4) & 3);
        const uint16_t d_b = dg[i_b];
        const uint16_t d_a = dg[i_a];
        const double dv_b = v - lev[i_b];
        const double dv_a = lev[i_a] - v;
        const double c_b = dv_b * dv_b;
        const double c_a = dv_a * dv_a;
        // Same bitwise tie-break NearestInTable uses: a select, not a branch.
        const bool take_b = (c_b < c_a) | ((c_b == c_a) & (d_b < d_a));
        dig[r] = take_b ? d_b : d_a;
        cost[r] = take_b ? c_b : c_a;
    }
}

// One branch's two parity tables, resolved once per encode call instead of
// once per dimension. The cache behind GetParityTable is thread_local, so the
// shared-library build pays a __tls_get_addr for every lookup, and the
// per-dimension path made 16 of them per 8-dimension block.
struct BranchTables {
    const ParityTable* even = nullptr;
    const ParityTable* odd = nullptr;
};

}  // namespace

double FixedScale(int bits) {
    // The per-rate encoding-scale schedule s(b): the grid alignment that
    // maximizes the expected squared directional cosine E[cos^2(u, v_hat)] of
    // a standard isotropic Gaussian block, evaluated offline directly on the
    // level tables (experiments/shaping/measure_scale_schedule.cpp ->
    // results/shaping/scale_schedule.csv).  The objective is the one the
    // estimator itself is exposed to: g1 of src/index/e8q_build.cpp is a
    // dot-normalized fit, g1 = r_n / <u, v_hat>, so the reconstruction
    // amplitude is absorbed exactly and the grid alignment controls only the
    // angle between the decoded block and the true one,
    // |e_x|^2 = r_x^2 tan^2(theta).
    //
    // The schedule is NOT serialized: the per-vector g1 absorbs amplitude, so
    // the constant only shapes which codeword the encoder selects, and it
    // costs zero storage.  No corpus, query set, or recall target enters the
    // evaluation -- the table is a geometric constant of the level tables and
    // the Gaussian block premise, so it is fixed once for every corpus,
    // dimension, and bit budget at that rate.
    //
    // b = 1 keeps the Zador constant sigma_c that the two-level sign code has
    // always used: the objective is flat there (it varies by 4.5e-4 of cos^2
    // over the whole band [1.00, 1.12], and sigma_c sits 3.4e-4 below the
    // band's own argmax at 1.06), so the analytic constant is kept rather
    // than replaced by a measured point on the same ridge.
    //
    // Every entry was taken from the argmax of the objective on a
    // 0.0025-wide grid. The ridge is flat at high rate -- cos^2 changes by
    // < 1e-5 over +-0.05 at b >= 6 -- which is why the last entries are not
    // monotone; the 0.005 spread between them is worth < 1e-5 of cos^2.
    static constexpr double kScaleSchedule[kMaxQuantizeBits + 1] = {
        0.0,
        1.118034, 0.895, 0.825, 0.785, 0.755, 0.735, 0.735, 0.735,
        0.725, 0.715, 0.705, 0.710, 0.705,
    };
    // E8Q_SCALE_SCHEDULE="s1,s2,...,s13" overrides the table for the offline
    // schedule study (experiments/_sched_sweep.sh, _sched_b123.sh); parsing
    // happens once.
    static const std::array<double, kMaxQuantizeBits + 1> table = [] {
        std::array<double, kMaxQuantizeBits + 1> t{};
        for (int i = 0; i <= kMaxQuantizeBits; ++i) {
            t[static_cast<size_t>(i)] = kScaleSchedule[i];
        }
        const char* env = std::getenv("E8Q_SCALE_SCHEDULE");
        if (env != nullptr) {
            const char* p = env;
            for (int i = 1; i <= kMaxQuantizeBits && *p != '\0'; ++i) {
                char* end = nullptr;
                const double v = std::strtod(p, &end);
                if (end == p) {
                    break;
                }
                t[static_cast<size_t>(i)] = v;
                p = end;
                while (*p == ',' || *p == ' ') {
                    ++p;
                }
            }
        }
        return t;
    }();
    if (bits < 1) {
        return table[1];
    }
    if (bits > kMaxQuantizeBits) {
        return table[static_cast<size_t>(kMaxQuantizeBits)];
    }
    return table[static_cast<size_t>(bits)];
}

std::vector<double> BranchLevels(int bits, int branch) {
    const int q = 1 << bits;
    const double eta = 0.25 + 0.5 * static_cast<double>(branch);
    std::vector<double> out(static_cast<size_t>(q));
    const int start = static_cast<int>(std::ceil(-0.5 * static_cast<double>(q) - eta));
    for (int i = 0; i < q; ++i) {
        const long t = static_cast<long>(start + i);
        const int digit = static_cast<int>(((t % q) + q) % q);
        const double p = 0.5 + (static_cast<double>(t) + eta) / static_cast<double>(q);
        out[static_cast<size_t>(digit)] =
            kSigmaC * InvNormalCdf(std::min(1.0 - 1e-12, std::max(1e-12, p)));
    }
    return out;
}

void ScanClassMeans(int bits, double w_cls[2]) {
    const std::vector<double> l0 = BranchLevels(bits, 0);
    const std::vector<double> l1 = BranchLevels(bits, 1);
    const int cvals = 1 << (bits - 1);
    for (int t2 = 0; t2 < 2; ++t2) {
        double s0 = 0.0;
        double s1 = 0.0;
        for (int lo = 0; lo < cvals; ++lo) {
            s0 += l0[static_cast<size_t>((t2 << (bits - 1)) | lo)];
            s1 += l1[static_cast<size_t>((t2 << (bits - 1)) | lo)];
        }
        w_cls[t2] = 0.5 * (s0 / cvals + s1 / cvals);
    }
}

// levels depend only on (bits, branch) -- kSigmaC and the Phi^-1 quantile
// sequence -- so the encoder's per-call rebuild was pure overhead: 2^bits
// InvNormalCdf evaluations and two heap allocations per vector (e8q) or per
// segment (me8q), for tables the encoder only ever reads through the
// (bits, branch, parity) caches.  Built once, on first use.
struct LevelSet {
    std::array<std::vector<double>, kMaxQuantizeBits + 1> lev[2];
    LevelSet() {
        for (int b = 1; b <= kMaxQuantizeBits; ++b) {
            lev[0][static_cast<size_t>(b)] = BranchLevels(b, 0);
            lev[1][static_cast<size_t>(b)] = BranchLevels(b, 1);
        }
    }
};

const LevelSet& GetLevelSet() {
    static const LevelSet s;
    return s;
}

// Per-branch selection and the single-flip parity repair (Theorem 3.2),
// shared by the four-table and the merged search so the two can only differ in
// which (digit, cost) pair each parity class contributes.
inline BlockCode AssembleBlock(
    const std::array<uint16_t, 8>& dig_e, const std::array<double, 8>& cost_e,
    const std::array<uint16_t, 8>& dig_o, const std::array<double, 8>& cost_o
) {
    BlockCode out;
    int parity = 0;
    for (int i = 0; i < 8; ++i) {
        const size_t k = static_cast<size_t>(i);
        const bool use_odd = cost_o[k] < cost_e[k];
        out.digits[k] = use_odd ? dig_o[k] : dig_e[k];
        out.cost += use_odd ? cost_o[k] : cost_e[k];
        parity ^= use_odd ? 1 : 0;
    }
    if (parity != 0) {
        // Even-parity repair keeps every codeword on the E8 lattice quotient
        // shell at EVERY rate b >= 1 (the paper's construction, and the fair
        // comparison: digits 8b-1 bits + branch 1 bit = 8b logical bits per
        // block, capacity 2^(8b) codewords). Dropping the constraint at any
        // rate would double the book at the same logical cost — unfair vs
        // rabitq; a b >= 2 drop (+0.0005..+0.0059 recall on gist b3) and the
        // earlier b == 1 exemption were both measured and REVERTED for this
        // reason. The b == 1 case stores the 8 sign bits in the plane byte
        // exactly as before; only the repair loop applies again.
        // Repair = single flip of argmin_i |c^o_i - c^e_i|: the per-coordinate
        // costs are separable and the only coupling is the mod-2 constraint,
        // so any feasible assignment differs from the independent optimum in
        // an ODD number of coordinates and costs >= min_i |d_i| extra; the
        // single flip attains the bound. (A 2-state DP over the same 8x2
        // candidate set was implemented and provably returns the same optimum
        // — reverted as dead cost.)
        int sel = 0;
        double best_extra = std::numeric_limits<double>::max();
        for (int i = 0; i < 8; ++i) {
            const bool was_odd =
                out.digits[static_cast<size_t>(i)] == dig_o[static_cast<size_t>(i)];
            const double extra =
                was_odd ? cost_e[static_cast<size_t>(i)] - cost_o[static_cast<size_t>(i)]
                        : cost_o[static_cast<size_t>(i)] - cost_e[static_cast<size_t>(i)];
            if (extra < best_extra) {
                best_extra = extra;
                sel = i;
            }
        }
        const bool was_odd = out.digits[static_cast<size_t>(sel)] ==
                             dig_o[static_cast<size_t>(sel)];
        out.digits[static_cast<size_t>(sel)] =
            was_odd ? dig_e[static_cast<size_t>(sel)] : dig_o[static_cast<size_t>(sel)];
        out.cost += std::max(best_extra, 0.0);
    }
    return out;
}

inline BlockCode EncodeBlockIn(const double* v, const BranchTables& bt) {
    std::array<double, 8> cost_e{};
    std::array<double, 8> cost_o{};
    std::array<double, 8> lev_e{};
    std::array<double, 8> lev_o{};
    std::array<uint16_t, 8> dig_e{};
    std::array<uint16_t, 8> dig_o{};
    for (int i = 0; i < 8; ++i) {
        NearestInTable(v[i], *bt.even, &dig_e[static_cast<size_t>(i)],
                       &lev_e[static_cast<size_t>(i)], &cost_e[static_cast<size_t>(i)]);
        NearestInTable(v[i], *bt.odd, &dig_o[static_cast<size_t>(i)],
                       &lev_o[static_cast<size_t>(i)], &cost_o[static_cast<size_t>(i)]);
    }
    return AssembleBlock(dig_e, cost_e, dig_o, cost_o);
}

// Both branches of one block from a single merged search per coordinate: the
// four classes NearestMerged returns are the two branches' even and odd
// tables, so one pass over the eight coordinates produces both candidate
// codewords the branch index chooses between.
inline void EncodeBlockBothBranches(
    const double* v, const MergedTable& mt, BlockCode* out0, BlockCode* out1
) {
    std::array<uint16_t, 8> de0{}, do0{}, de1{}, do1{};
    std::array<double, 8> ce0{}, co0{}, ce1{}, co1{};
    uint16_t dg[4];
    double cs[4];
    for (int i = 0; i < 8; ++i) {
        NearestMerged(v[i], mt, dg, cs);
        // Residue order of the merged table: 0 = branch 0 even, 1 = branch 1
        // even, 2 = branch 0 odd, 3 = branch 1 odd.
        const size_t k = static_cast<size_t>(i);
        de0[k] = dg[0];
        ce0[k] = cs[0];
        de1[k] = dg[1];
        ce1[k] = cs[1];
        do0[k] = dg[2];
        co0[k] = cs[2];
        do1[k] = dg[3];
        co1[k] = cs[3];
    }
    *out0 = AssembleBlock(de0, ce0, do0, co0);
    *out1 = AssembleBlock(de1, ce1, do1, co1);
}

BlockCode EncodeBlock(const double* v, int bits, const std::vector<double>& levels, int branch) {
    (void)bits;
    const BranchTables bt{
        &GetParityTable(levels, branch, false), &GetParityTable(levels, branch, true)
    };
    return EncodeBlockIn(v, bt);
}

std::vector<double> DefaultScales() {
    // Log-spaced fallback grid (white-box tests only).
    std::vector<double> s(kNumScale);
    const double base = std::log(0.55 / FixedScale(3));
    const double step = std::log(1.35 / 0.55) / static_cast<double>(kNumScale - 1);
    for (int i = 0; i < kNumScale; ++i) {
        s[static_cast<size_t>(i)] = FixedScale(3) * std::exp(base + step * i);
    }
    return s;
}

// The whole-vector encode. Everything above this line is deliberately one
// compile unit and nothing else: a second entry point sharing these tables
// (ME8Q's residual form, which divides by the segment norm on the way in)
// cost this loop 20% -- GCC stopped inlining the block helpers once the file
// held two copies of it -- so ME8Q divides into its own buffer and calls
// here, and an encoder microbenchmark (`bench_encode`) guards the number.
void EncodeVectorAtScaleInto(
    const float* unit_dir, int padded_dim, int bits, double s, VectorCode& vc
) {
    if (bits < 1 || bits > kMaxQuantizeBits) {
        throw std::invalid_argument("EncodeVectorAtScale: bits must be in [1, 13]");
    }
    const int blocks = padded_dim / 8;
    const double rd = std::sqrt(static_cast<double>(padded_dim));
    const LevelSet& ls = GetLevelSet();
    const MergedTable& mt =
        GetMergedTable(ls.lev[0][static_cast<size_t>(bits)],
                       ls.lev[1][static_cast<size_t>(bits)]);
    vc.digits.resize(static_cast<size_t>(blocks));
    vc.branch.resize(static_cast<size_t>(blocks));
    if (mt.alternating) {
        for (int blk = 0; blk < blocks; ++blk) {
            double v[8];
            for (int i = 0; i < 8; ++i) {
                v[i] = static_cast<double>(unit_dir[blk * 8 + i]) * (s * rd);
            }
            BlockCode e0;
            BlockCode e1;
            EncodeBlockBothBranches(v, mt, &e0, &e1);
            if (e1.cost < e0.cost) {
                vc.digits[static_cast<size_t>(blk)] = e1.digits;
                vc.branch[static_cast<size_t>(blk)] = 1;
            } else {
                vc.digits[static_cast<size_t>(blk)] = e0.digits;
                vc.branch[static_cast<size_t>(blk)] = 0;
            }
        }
        return;
    }
    const std::vector<double>* lev[2] = {
        &ls.lev[0][static_cast<size_t>(bits)], &ls.lev[1][static_cast<size_t>(bits)]
    };
    const BranchTables bt[2] = {
        {&GetParityTable(*lev[0], 0, false), &GetParityTable(*lev[0], 0, true)},
        {&GetParityTable(*lev[1], 1, false), &GetParityTable(*lev[1], 1, true)},
    };
    for (int blk = 0; blk < blocks; ++blk) {
        double v[8];
        for (int i = 0; i < 8; ++i) {
            v[i] = static_cast<double>(unit_dir[blk * 8 + i]) * (s * rd);
        }
        const BlockCode e0 = EncodeBlockIn(v, bt[0]);
        const BlockCode e1 = EncodeBlockIn(v, bt[1]);
        if (e1.cost < e0.cost) {
            vc.digits[static_cast<size_t>(blk)] = e1.digits;
            vc.branch[static_cast<size_t>(blk)] = 1;
        } else {
            vc.digits[static_cast<size_t>(blk)] = e0.digits;
            vc.branch[static_cast<size_t>(blk)] = 0;
        }
    }
}


VectorCode EncodeVectorAtScale(
    const float* unit_dir, int padded_dim, int bits, double s
) {
    VectorCode vc;
    EncodeVectorAtScaleInto(unit_dir, padded_dim, bits, s, vc);
    return vc;
}

VectorCode EncodeVector(
    const float* unit_dir, int padded_dim, int bits, const float* scales_in
) {
    if (bits < 1 || bits > kMaxQuantizeBits) {
        throw std::invalid_argument("EncodeVector: bits must be in [1, 13]");
    }
    const int blocks = padded_dim / 8;
    const double rd = std::sqrt(static_cast<double>(padded_dim));
    const LevelSet& ls = GetLevelSet();
    const std::vector<double>* levels[2] = {
        &ls.lev[0][static_cast<size_t>(bits)], &ls.lev[1][static_cast<size_t>(bits)]
    };
    const BranchTables bt[2] = {
        {&GetParityTable(*levels[0], 0, false), &GetParityTable(*levels[0], 0, true)},
        {&GetParityTable(*levels[1], 1, false), &GetParityTable(*levels[1], 1, true)},
    };
    std::vector<float> fallback;
    const float* scales = scales_in;
    if (scales == nullptr) {
        const std::vector<double> def = DefaultScales();
        fallback.assign(def.begin(), def.end());
        scales = fallback.data();
    }

    VectorCode vc;
    vc.digits.resize(static_cast<size_t>(blocks));
    vc.branch.resize(static_cast<size_t>(blocks));
    // Midpoint of the learned grid: only used to seed the closed-form scale
    // estimate (the snapped scale is re-encoded from scratch below).
    const double s_mid = scales[kNumScale / 2];
    for (int blk = 0; blk < blocks; ++blk) {
        double v[8];
        for (int i = 0; i < 8; ++i) {
            v[i] = static_cast<double>(unit_dir[blk * 8 + i]) * (s_mid * rd);
        }
        // Probe both branches at the seed scale; estimate s* from the better
        // probe: unit ~= lev / (s * rd)  =>  s* = (lev.lev) / ((unit.lev) * rd).
        double best_cost = std::numeric_limits<double>::max();
        double best_s = s_mid;
        for (int br = 0; br < 2; ++br) {
            const BlockCode e = EncodeBlockIn(v, bt[br]);
            double ul = 0.0;
            double ll = 0.0;
            for (int i = 0; i < 8; ++i) {
                const double lev =
                    (*levels[br])[static_cast<size_t>(e.digits[static_cast<size_t>(i)])];
                ul += static_cast<double>(unit_dir[blk * 8 + i]) * lev;
                ll += lev * lev;
            }
            if (e.cost < best_cost && ul > 1e-12) {
                best_cost = e.cost;
                best_s = ll / (ul * rd);
            }
        }
        best_s = std::min(2.0, std::max(0.25, best_s));
        double s = static_cast<double>(scales[0]);
        double sd = std::numeric_limits<double>::max();
        for (int i = 0; i < kNumScale; ++i) {
            const double dd = std::abs(static_cast<double>(scales[i]) - best_s);
            if (dd < sd) {
                sd = dd;
                s = static_cast<double>(scales[i]);
            }
        }
        for (int i = 0; i < 8; ++i) {
            v[i] = static_cast<double>(unit_dir[blk * 8 + i]) * (s * rd);
        }
        const BlockCode e0 = EncodeBlockIn(v, bt[0]);
        const BlockCode e1 = EncodeBlockIn(v, bt[1]);
        const bool take1 = e1.cost < e0.cost;
        vc.digits[static_cast<size_t>(blk)] = take1 ? e1.digits : e0.digits;
        vc.branch[static_cast<size_t>(blk)] = take1 ? 1 : 0;
    }
    return vc;
}

void PackRefineBlock(const uint16_t digits[8], int branch, int bits, uint8_t* out) {
    if (bits == 1) {
        // The rowA byte is the whole b == 1 code: sign bits of dims 0..6,
        // branch in bit 0 (dim 7's sign is parity-derived at decode).
        uint8_t byte = 0;
        for (int i = 0; i < 7; ++i) {
            if (digits[i] & 1u) {
                byte = static_cast<uint8_t>(byte | static_cast<uint8_t>(1u << (7 - i)));
            }
        }
        byte = static_cast<uint8_t>(byte | static_cast<uint8_t>(branch & 1));
        out[0] = byte;
        return;
    }
    // Low c = b-1 bits of digits 0..6; digit 7 bit 0 replaced by the branch.
    // b <= 8 fits in 56 bits; b = 9..13 uses up to 96 bits (__int128).
    const int c = bits - 1;
    const unsigned __int128 mask = (static_cast<unsigned __int128>(1) << c) - 1;
    unsigned __int128 x = 0;
    for (int i = 0; i < 7; ++i) {
        x |= (static_cast<unsigned __int128>(digits[i]) & mask) << (c * i);
    }
    x |= (static_cast<unsigned __int128>(digits[7] & ~1u) & mask) << (c * 7);
    x |= static_cast<unsigned __int128>(branch & 1) << (c * 7);
    std::memcpy(out, &x, static_cast<size_t>(c));
}

void PackRefineBlock(const uint8_t digits[8], int branch, int bits, uint8_t* out) {
    uint16_t wide[8];
    for (int i = 0; i < 8; ++i) {
        wide[i] = digits[i];
    }
    PackRefineBlock(wide, branch, bits, out);
}

void UnpackRefineBlock(
    const uint8_t* in, int bits, uint8_t plane_byte, uint8_t digits[8], int* branch
) {
    if (bits > 8) {
        // Digits exceed 8 bits and the low fields span up to 96 bits; the
        // uint8_t staging path cannot represent them. Use the uint16_t twin.
        throw std::invalid_argument("UnpackRefineBlock: bits must be <= 8");
    }
    if (bits == 1) {
        // The rowA byte is the whole b == 1 code: sign bits of dims 0..6,
        // branch in bit 0, dim 7's sign = the parity of the other seven.
        const uint8_t byte = in[0];
        int par = 0;
        for (int i = 0; i < 7; ++i) {
            digits[i] = static_cast<uint8_t>((byte >> (7 - i)) & 1u);
            par ^= digits[i] & 1;
        }
        digits[7] = static_cast<uint8_t>(par);
        if (branch != nullptr) {
            *branch = static_cast<int>(byte & 1u);
        }
        return;
    }
    // Low fields from the row, top bit from the scan plane byte; digit 7's
    // LSB is the parity of digits 0..6, and its stored bit 0 is the branch.
    const int c = bits - 1;
    uint64_t x = 0;
    std::memcpy(&x, in, static_cast<size_t>(c));
    const uint64_t mask = (uint64_t{1} << c) - 1;
    int par = 0;
    for (int i = 0; i < 7; ++i) {
        par ^= static_cast<int>(((x >> (c * i)) & mask) & 1u);
    }
    if (branch != nullptr) {
        *branch = static_cast<int>((x >> (c * 7)) & 1u);
    }
    for (int i = 0; i < 7; ++i) {
        const uint64_t f = (x >> (c * i)) & mask;
        const uint64_t top = (plane_byte >> (7 - i)) & 1u;
        digits[i] = static_cast<uint8_t>(f | (top << c));
    }
    const uint64_t f7 = ((x >> (c * 7)) & mask) & ~uint64_t{1};
    const uint64_t top7 = (plane_byte >> 0) & 1u;
    digits[7] = static_cast<uint8_t>(f7 | static_cast<uint64_t>(par) | (top7 << c));
}

void UnpackRefineBlock(
    const uint8_t* in, int bits, uint8_t plane_byte, uint16_t digits[8], int* branch
) {
    // Wide twin of the 8-bit unpack above (b = 9..13): the low fields span
    // up to 96 bits, so the staging integer is 128-bit.
    if (bits == 1) {
        const uint8_t byte = in[0];
        int par = 0;
        for (int i = 0; i < 7; ++i) {
            digits[i] = static_cast<uint16_t>((byte >> (7 - i)) & 1u);
            par ^= digits[i] & 1;
        }
        digits[7] = static_cast<uint16_t>(par);
        if (branch != nullptr) {
            *branch = static_cast<int>(byte & 1u);
        }
        return;
    }
    const int c = bits - 1;
    unsigned __int128 x = 0;
    std::memcpy(&x, in, static_cast<size_t>(c));
    const unsigned __int128 mask = (static_cast<unsigned __int128>(1) << c) - 1;
    int par = 0;
    for (int i = 0; i < 7; ++i) {
        par ^= static_cast<int>(((x >> (c * i)) & mask) & 1u);
    }
    if (branch != nullptr) {
        *branch = static_cast<int>((x >> (c * 7)) & 1u);
    }
    for (int i = 0; i < 7; ++i) {
        const unsigned __int128 f = (x >> (c * i)) & mask;
        const unsigned __int128 top = (plane_byte >> (7 - i)) & 1u;
        digits[i] = static_cast<uint16_t>(f | (top << c));
    }
    const unsigned __int128 f7 = ((x >> (c * 7)) & mask) & ~static_cast<unsigned __int128>(1);
    const unsigned __int128 top7 = (plane_byte >> 0) & 1u;
    digits[7] = static_cast<uint16_t>(f7 | static_cast<unsigned __int128>(par) | (top7 << c));
}

std::vector<float> BuildRefineTables(int bits, int padded_dim) {
    const size_t q = static_cast<size_t>(1 << bits);
    const size_t stride = std::max(size_t{256}, q);
    std::vector<float> tab(2 * stride, 0.0F);
    const double rd = std::sqrt(static_cast<double>(padded_dim));
    for (int br = 0; br < 2; ++br) {
        const std::vector<double> lev = BranchLevels(bits, br);
        float* t = &tab[static_cast<size_t>(br) * stride];
        for (size_t d = 0; d < lev.size(); ++d) {
            t[d] = static_cast<float>(lev[d] / rd);
        }
    }
    return tab;
}

}  // namespace e8qlib
