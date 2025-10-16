// CUDA kernels for OTTO (Optimized Trend Tracker Oscillator)
//
// Category: Recurrence/IIR with per-row sequential scan.
// Implements the common case where MA type = VAR (VIDYA on LOTT).
// For other MA types, the host wrapper may fall back to a multi-stage pipeline.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float nzf(float x) {
    return isfinite(x) ? x : 0.0f;
}

extern "C" __global__
void otto_batch_f32(
    const float* __restrict__ prices,     // len = series_len
    const float* __restrict__ cabs,       // len = series_len (abs CMO(9) on price)
    const int*   __restrict__ ott_periods,// len = n_combos
    const float* __restrict__ ott_percents,// len = n_combos
    const int*   __restrict__ fast_vidyas,// len = n_combos
    const int*   __restrict__ slow_vidyas,// len = n_combos
    const float* __restrict__ cocos,      // len = n_combos (correcting_constant)
    int series_len,
    int n_combos,
    int /*first_valid (reserved)*/,
    float* __restrict__ hott_out,         // len = n_combos * series_len
    float* __restrict__ lott_out          // len = n_combos * series_len
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) return;

    // Load per-row params
    const int slow = max(__ldg(slow_vidyas + combo), 1);
    const int fast = max(__ldg(fast_vidyas + combo), 1);
    const int p1 = max(slow / 2, 1);
    const int p2 = slow;
    const int p3 = max(slow * fast, 1);

    const double a1_base = 2.0 / (static_cast<double>(p1) + 1.0);
    const double a2_base = 2.0 / (static_cast<double>(p2) + 1.0);
    const double a3_base = 2.0 / (static_cast<double>(p3) + 1.0);

    const int ott_p = max(__ldg(ott_periods + combo), 1);
    const double a_base_lott = 2.0 / (static_cast<double>(ott_p) + 1.0);
    const double ott_percent = static_cast<double>(__ldg(ott_percents + combo));
    const double coco = static_cast<double>(__ldg(cocos + combo));

    const double fark = ott_percent * 0.01;
    const double scale_up = (200.0 + ott_percent) / 200.0;
    const double scale_dn = (200.0 - ott_percent) / 200.0;

    float* __restrict__ hott_row = hott_out + combo * series_len;
    float* __restrict__ lott_row = lott_out + combo * series_len;

    // VIDYA tracks for source
    double v1 = 0.0, v2 = 0.0, v3 = 0.0;
    // For VAR(LOTT): CMO(9) on lott diffs
    const int CMO_P = 9;
    double ring_up[CMO_P];
    double ring_dn[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) { ring_up[k] = 0.0; ring_dn[k] = 0.0; }
    double sum_up = 0.0, sum_dn = 0.0;
    int head = 0;

    double prev_lott = 0.0;
    double ma_prev = 0.0;
    double long_stop_prev = NAN, short_stop_prev = NAN;
    int dir_prev = 1;

    for (int i = 0; i < series_len; ++i) {
        const double x = nzf(__ldg(prices + i));
        const double c_abs = static_cast<double>(__ldg(cabs + i));

        // Adaptive alphas
        const double a1 = a1_base * c_abs;
        const double a2 = a2_base * c_abs;
        const double a3 = a3_base * c_abs;

        // VIDYA on source
        v1 = fma(a1, x, (1.0 - a1) * v1);
        v2 = fma(a2, x, (1.0 - a2) * v2);
        v3 = fma(a3, x, (1.0 - a3) * v3);

        // LOTT = v1 / ((v2 - v3) + coco)
        const double denom_l = (v2 - v3) + coco;
        const double lott = denom_l != 0.0 ? (v1 / denom_l) : 0.0;
        lott_row[i] = static_cast<float>(lott);

        // CMO(9) on LOTT diffs (VIDYA on LOTT)
        if (i > 0) {
            const double d = lott - prev_lott;
            if (i >= CMO_P) {
                sum_up -= ring_up[head];
                sum_dn -= ring_dn[head];
            }
            const double up = d > 0.0 ? d : 0.0;
            const double dn = d > 0.0 ? 0.0 : -d;
            ring_up[head] = up;
            ring_dn[head] = dn;
            sum_up += up;
            sum_dn += dn;
            head = (head + 1) == CMO_P ? 0 : (head + 1);
        }
        prev_lott = lott;

        const double denom = sum_up + sum_dn;
        const double c2 = (i >= CMO_P && denom != 0.0) ? fabs((sum_up - sum_dn) / denom) : 0.0;
        const double a_lott = a_base_lott * c2;
        const double ma = fma(a_lott, lott, (1.0 - a_lott) * ma_prev);
        ma_prev = ma;

        if (i == 0) {
            long_stop_prev = ma * (1.0 - fark);
            short_stop_prev = ma * (1.0 + fark);
            const double mt = long_stop_prev;
            hott_row[i] = static_cast<float>(ma > mt ? mt * scale_up : mt * scale_dn);
        } else {
            const double ls = ma * (1.0 - fark);
            const double ss = ma * (1.0 + fark);
            const double long_stop = (ma > long_stop_prev) ? fmax(ls, long_stop_prev) : ls;
            const double short_stop = (ma < short_stop_prev) ? fmin(ss, short_stop_prev) : ss;
            const int dir = (dir_prev == -1 && ma > short_stop_prev)
                                ? 1
                                : ((dir_prev == 1 && ma < long_stop_prev) ? -1 : dir_prev);
            const double mt = (dir == 1) ? long_stop : short_stop;
            hott_row[i] = static_cast<float>(ma > mt ? mt * scale_up : mt * scale_dn);
            long_stop_prev = long_stop;
            short_stop_prev = short_stop;
            dir_prev = dir;
        }
    }
}

extern "C" __global__
void otto_many_series_one_param_f32(
    const float* __restrict__ prices_tm, // time-major [t * rows + s]
    int cols,                            // series_len (time)
    int rows,                            // num_series
    int ott_period,
    float ott_percent_f,
    int fast_vidya,
    int slow_vidya,
    float coco_f,
    float* __restrict__ hott_tm,
    float* __restrict__ lott_tm
) {
    const int series = blockIdx.x;
    if (series >= rows || threadIdx.x != 0) return;

    const int p1 = max(slow_vidya / 2, 1);
    const int p2 = max(slow_vidya, 1);
    const int p3 = max(slow_vidya * max(fast_vidya, 1), 1);
    const double a1_base = 2.0 / (static_cast<double>(p1) + 1.0);
    const double a2_base = 2.0 / (static_cast<double>(p2) + 1.0);
    const double a3_base = 2.0 / (static_cast<double>(p3) + 1.0);
    const double a_base_lott = 2.0 / (static_cast<double>(max(ott_period, 1)) + 1.0);
    const double coco = static_cast<double>(coco_f);
    const double ott_percent = static_cast<double>(ott_percent_f);
    const double fark = ott_percent * 0.01;
    const double scale_up = (200.0 + ott_percent) / 200.0;
    const double scale_dn = (200.0 - ott_percent) / 200.0;

    // Precompute c_abs on price locally (CMO 9)
    const int CMO_P = 9;
    double ring_up_p[CMO_P];
    double ring_dn_p[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) { ring_up_p[k] = 0.0; ring_dn_p[k] = 0.0; }
    double sum_up_p = 0.0, sum_dn_p = 0.0; int head_p = 0;

    double v1 = 0.0, v2 = 0.0, v3 = 0.0;
    double prev_price = 0.0;

    // For VAR(LOTT)
    double ring_up_l[CMO_P];
    double ring_dn_l[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) { ring_up_l[k] = 0.0; ring_dn_l[k] = 0.0; }
    double sum_up_l = 0.0, sum_dn_l = 0.0; int head_l = 0;
    double prev_lott = 0.0;
    double ma_prev = 0.0;
    double long_stop_prev = NAN, short_stop_prev = NAN; int dir_prev = 1;

    for (int t = 0; t < cols; ++t) {
        const double x = nzf(prices_tm[t * rows + series]);
        if (t > 0) {
            const double d = x - prev_price;
            if (t >= CMO_P) { sum_up_p -= ring_up_p[head_p]; sum_dn_p -= ring_dn_p[head_p]; }
            const double up = d > 0.0 ? d : 0.0;
            const double dn = d > 0.0 ? 0.0 : -d;
            ring_up_p[head_p] = up; ring_dn_p[head_p] = dn;
            sum_up_p += up; sum_dn_p += dn; head_p = (head_p + 1) == CMO_P ? 0 : (head_p + 1);
        }
        prev_price = x;
        const double denom_p = sum_up_p + sum_dn_p;
        const double c_abs = (t >= CMO_P && denom_p != 0.0) ? fabs((sum_up_p - sum_dn_p)/denom_p) : 0.0;

        const double a1 = a1_base * c_abs;
        const double a2 = a2_base * c_abs;
        const double a3 = a3_base * c_abs;
        v1 = fma(a1, x, (1.0 - a1) * v1);
        v2 = fma(a2, x, (1.0 - a2) * v2);
        v3 = fma(a3, x, (1.0 - a3) * v3);
        const double denom_l = (v2 - v3) + coco;
        const double lott = denom_l != 0.0 ? (v1 / denom_l) : 0.0;
        lott_tm[t * rows + series] = static_cast<float>(lott);

        if (t > 0) {
            const double d = lott - prev_lott;
            if (t >= CMO_P) { sum_up_l -= ring_up_l[head_l]; sum_dn_l -= ring_dn_l[head_l]; }
            const double up = d > 0.0 ? d : 0.0;
            const double dn = d > 0.0 ? 0.0 : -d;
            ring_up_l[head_l] = up; ring_dn_l[head_l] = dn;
            sum_up_l += up; sum_dn_l += dn; head_l = (head_l + 1) == CMO_P ? 0 : (head_l + 1);
        }
        prev_lott = lott;
        const double denom_lc = sum_up_l + sum_dn_l;
        const double c2 = (t >= CMO_P && denom_lc != 0.0) ? fabs((sum_up_l - sum_dn_l)/denom_lc) : 0.0;
        const double a_lott = a_base_lott * c2;
        const double ma = fma(a_lott, lott, (1.0 - a_lott) * ma_prev);
        ma_prev = ma;

        if (t == 0) {
            long_stop_prev = ma * (1.0 - fark);
            short_stop_prev = ma * (1.0 + fark);
            const double mt = long_stop_prev;
            hott_tm[t * rows + series] = static_cast<float>(ma > mt ? mt * scale_up : mt * scale_dn);
        } else {
            const double ls = ma * (1.0 - fark);
            const double ss = ma * (1.0 + fark);
            const double long_stop = (ma > long_stop_prev) ? fmax(ls, long_stop_prev) : ls;
            const double short_stop = (ma < short_stop_prev) ? fmin(ss, short_stop_prev) : ss;
            const int dir = (dir_prev == -1 && ma > short_stop_prev)
                                ? 1
                                : ((dir_prev == 1 && ma < long_stop_prev) ? -1 : dir_prev);
            const double mt = (dir == 1) ? long_stop : short_stop;
            hott_tm[t * rows + series] = static_cast<float>(ma > mt ? mt * scale_up : mt * scale_dn);
            long_stop_prev = long_stop; short_stop_prev = short_stop; dir_prev = dir;
        }
    }
}

