#include "simd/me8q_prune.hpp"

#include <vector>

#include <immintrin.h>

namespace e8qlib::me8q {

size_t CoarsePruneAvx2(const CoarsePruneArgs& a) {
    if (a.nrun == 0 || a.lane_cnt < 8) {
        return 0;
    }
    const __m256 lam_v = _mm256_set1_ps(a.lambda);
    const __m256 distk_v = _mm256_set1_ps(a.distk);
    const __m256 vdelta = _mm256_set1_ps(a.lut_delta);
    const __m256 dc0_v = _mm256_set1_ps(a.d2);
    const __m256 sign_mask = _mm256_set1_ps(-0.0F);
    // Per-run scalars do not depend on the lane group.
    thread_local PruneRunScalars rs;
    FillPruneRunScalars(a, rs);
    const std::vector<float>& rh_r = rs.rh;
    const std::vector<float>& qerr_f = rs.qerr;
    const std::vector<float>& wpad_f = rs.wpad;
    const std::vector<float>& lutmin_f = rs.lutmin;
    const size_t covered = (a.lane_cnt / 8) * 8;
    for (size_t l0 = 0; l0 + 8 <= a.lane_cnt; l0 += 8) {
        __m256 dc = dc0_v;
        __m256 pool = _mm256_setzero_ps();
        __m256 qa = _mm256_setzero_ps();
        __m256 wpad = _mm256_setzero_ps();
        for (size_t r = 0; r < a.nrun; ++r) {
            const __m256i ai = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(a.acc + r * 32 + l0)
            );
            const __m256 s_c_v = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(ai), vdelta, _mm256_set1_ps(lutmin_f[r])
            );
            const uint8_t* bp = a.banks + r * a.bank_run_bytes;
            const float* sc = reinterpret_cast<const float*>(bp + 128);
            const __m256 bmin = _mm256_set1_ps(sc[0]);
            const __m256 bsc = _mm256_set1_ps(sc[1]);
            const __m256 wmin = _mm256_set1_ps(sc[2]);
            const __m256 wsc = _mm256_set1_ps(sc[3]);
            const __m256 pmin = _mm256_set1_ps(sc[4]);
            const __m256 psc = _mm256_set1_ps(sc[5]);
            const __m256 bq = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bp + l0))
            ));
            const __m256 wq = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bp + 32 + l0))
            ));
            const __m256 pq = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bp + 64 + l0))
            ));
            const __m256 base_q = _mm256_fmadd_ps(bq, bsc, bmin);
            const __m256 weight_q = _mm256_fmadd_ps(wq, wsc, wmin);
            const __m256 perp_q = _mm256_fmadd_ps(pq, psc, pmin);
            dc = _mm256_fnmadd_ps(weight_q, s_c_v, dc);
            dc = _mm256_add_ps(dc, base_q);
            const __m256 t = _mm256_mul_ps(perp_q, _mm256_set1_ps(rh_r[r]));
            _mm256_storeu_ps(a.lane_sc + 32 * r + l0, s_c_v);
            _mm256_storeu_ps(a.lane_tr + 32 * r + l0, t);
            pool = _mm256_fmadd_ps(t, t, pool);
            const __m256 abs_w = _mm256_andnot_ps(sign_mask, weight_q);
            qa = _mm256_fmadd_ps(abs_w, _mm256_set1_ps(qerr_f[r]), qa);
            wpad = _mm256_add_ps(wpad, _mm256_set1_ps(wpad_f[r]));
        }
        const __m256 slack = _mm256_fmadd_ps(
            lam_v, _mm256_sqrt_ps(pool), _mm256_add_ps(qa, wpad)
        );
        _mm256_storeu_ps(a.lane_dc + l0, dc);
        _mm256_storeu_ps(a.lane_pool + l0, pool);
        // The weight pad rides in lane_qa so the cascade's death test covers
        // it with the rest of the additive slack.
        _mm256_storeu_ps(a.lane_qa + l0, _mm256_add_ps(qa, wpad));
        const int bits_mask =
            _mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(dc, slack), distk_v, _CMP_GE_OQ));
        if (bits_mask == 0) {
            for (int j = 0; j < 8; ++j) {
                a.keep_lanes->push_back(static_cast<uint32_t>(l0 + static_cast<size_t>(j)));
            }
            continue;
        }
        const int np_ = __builtin_popcount(static_cast<unsigned>(bits_mask));
        *a.candidates += static_cast<size_t>(np_);
        *a.bits_total += static_cast<size_t>(np_) * a.bits_row;
        // bits_mask != 0 here (all-keep already continued), so keep_mask != 0xFF.
        const int keep_mask = (~bits_mask) & 0xFF;
        if (keep_mask != 0) {
            for (int j = 0; j < 8; ++j) {
                if (((keep_mask >> j) & 1) != 0) {
                    a.keep_lanes->push_back(static_cast<uint32_t>(l0 + static_cast<size_t>(j)));
                }
            }
        }
    }
    return covered;
}

}  // namespace e8qlib::me8q
