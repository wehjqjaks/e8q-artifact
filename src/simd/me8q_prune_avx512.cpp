#include "simd/me8q_prune.hpp"

#include <vector>

#include <immintrin.h>

namespace e8qlib::me8q {

size_t CoarsePruneAvx512(const CoarsePruneArgs& a) {
    if (a.nrun == 0 || a.lane_cnt < 16) {
        return 0;
    }
    const __m512 lam_v = _mm512_set1_ps(a.lambda);
    const __m512 distk_v = _mm512_set1_ps(a.distk);
    const __m512 vdelta = _mm512_set1_ps(a.lut_delta);
    const __m512 dc0_v = _mm512_set1_ps(a.d2);
    thread_local PruneRunScalars rs;
    FillPruneRunScalars(a, rs);
    const std::vector<float>& rh_r = rs.rh;
    const std::vector<float>& qerr_f = rs.qerr;
    const std::vector<float>& wpad_f = rs.wpad;
    const std::vector<float>& lutmin_f = rs.lutmin;
    const size_t covered = (a.lane_cnt / 16) * 16;
    for (size_t l0 = 0; l0 + 16 <= a.lane_cnt; l0 += 16) {
        __m512 dc = dc0_v;
        __m512 pool = _mm512_setzero_ps();
        __m512 qa = _mm512_setzero_ps();
        __m512 wpad = _mm512_setzero_ps();
        for (size_t r = 0; r < a.nrun; ++r) {
            const __m512i ai = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(a.acc + r * 32 + l0)
            );
            const __m512 s_c_v = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(ai), vdelta, _mm512_set1_ps(lutmin_f[r])
            );
            const uint8_t* bp = a.banks + r * a.bank_run_bytes;
            const float* sc = reinterpret_cast<const float*>(bp + 128);
            const __m512 bq = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bp + l0))
            ));
            const __m512 wq = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bp + 32 + l0))
            ));
            const __m512 pq = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bp + 64 + l0))
            ));
            const __m512 base_q = _mm512_fmadd_ps(bq, _mm512_set1_ps(sc[1]),
                                                  _mm512_set1_ps(sc[0]));
            const __m512 weight_q = _mm512_fmadd_ps(wq, _mm512_set1_ps(sc[3]),
                                                    _mm512_set1_ps(sc[2]));
            const __m512 perp_q = _mm512_fmadd_ps(pq, _mm512_set1_ps(sc[5]),
                                                  _mm512_set1_ps(sc[4]));
            dc = _mm512_fnmadd_ps(weight_q, s_c_v, dc);
            dc = _mm512_add_ps(dc, base_q);
            const __m512 t = _mm512_mul_ps(perp_q, _mm512_set1_ps(rh_r[r]));
            _mm512_storeu_ps(a.lane_sc + 32 * r + l0, s_c_v);
            _mm512_storeu_ps(a.lane_tr + 32 * r + l0, t);
            pool = _mm512_fmadd_ps(t, t, pool);
            const __m512 abs_w = _mm512_abs_ps(weight_q);
            qa = _mm512_fmadd_ps(abs_w, _mm512_set1_ps(qerr_f[r]), qa);
            wpad = _mm512_add_ps(wpad, _mm512_set1_ps(wpad_f[r]));
        }
        const __m512 slack = _mm512_fmadd_ps(
            lam_v, _mm512_sqrt_ps(pool), _mm512_add_ps(qa, wpad)
        );
        _mm512_storeu_ps(a.lane_dc + l0, dc);
        _mm512_storeu_ps(a.lane_pool + l0, pool);
        // The weight pad rides in lane_qa so the cascade's death test covers
        // it with the rest of the additive slack.
        _mm512_storeu_ps(a.lane_qa + l0, _mm512_add_ps(qa, wpad));
        const __mmask16 bits_mask =
            _mm512_cmp_ps_mask(_mm512_sub_ps(dc, slack), distk_v, _CMP_GE_OQ);
        if (bits_mask == 0) {
            for (int j = 0; j < 16; ++j) {
                a.keep_lanes->push_back(static_cast<uint32_t>(l0 + static_cast<size_t>(j)));
            }
            continue;
        }
        const int np_ = __builtin_popcount(static_cast<unsigned>(bits_mask));
        *a.candidates += static_cast<size_t>(np_);
        *a.bits_total += static_cast<size_t>(np_) * a.bits_row;
        const uint16_t keep_mask = static_cast<uint16_t>(~bits_mask);
        for (int j = 0; j < 16; ++j) {
            if (((keep_mask >> j) & 1) != 0) {
                a.keep_lanes->push_back(static_cast<uint32_t>(l0 + static_cast<size_t>(j)));
            }
        }
    }
    return covered;
}

}  // namespace e8qlib::me8q
