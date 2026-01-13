





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
    const float* __restrict__ prices,
    const float* __restrict__ cabs,
    const int*   __restrict__ ott_periods,
    const float* __restrict__ ott_percents,
    const int*   __restrict__ fast_vidyas,
    const int*   __restrict__ slow_vidyas,
    const float* __restrict__ cocos,
    int series_len,
    int n_combos,
    int ,
    float* __restrict__ hott_out,
    float* __restrict__ lott_out
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) return;


    const int slow = max(__ldg(slow_vidyas + combo), 1);
    const int fast = max(__ldg(fast_vidyas + combo), 1);
    const int p1 = max(slow / 2, 1);
    const int p2 = slow;
    const int p3 = max(slow * fast, 1);

    const float a1_base = 2.0f / (static_cast<float>(p1) + 1.0f);
    const float a2_base = 2.0f / (static_cast<float>(p2) + 1.0f);
    const float a3_base = 2.0f / (static_cast<float>(p3) + 1.0f);

    const int ott_p = max(__ldg(ott_periods + combo), 1);
    const float a_base_lott = 2.0f / (static_cast<float>(ott_p) + 1.0f);
    const float ott_percent = __ldg(ott_percents + combo);
    const float coco = __ldg(cocos + combo);

    const float fark = ott_percent * 0.01f;
    const float scale_up = (200.0f + ott_percent) / 200.0f;
    const float scale_dn = (200.0f - ott_percent) / 200.0f;

    float* __restrict__ hott_row = hott_out + combo * series_len;
    float* __restrict__ lott_row = lott_out + combo * series_len;


    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;

    const int CMO_P = 9;
    float ring_up[CMO_P];
    float ring_dn[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) { ring_up[k] = 0.0f; ring_dn[k] = 0.0f; }
    float sum_up = 0.0f, sum_dn = 0.0f;
    int head = 0;

    float prev_lott = 0.0f;
    float ma_prev = 0.0f;
    float long_stop_prev = NAN, short_stop_prev = NAN;
    int dir_prev = 1;

    for (int i = 0; i < series_len; ++i) {
        const float x = nzf(__ldg(prices + i));
        const float c_abs = __ldg(cabs + i);


        const float a1 = a1_base * c_abs;
        const float a2 = a2_base * c_abs;
        const float a3 = a3_base * c_abs;


        v1 = fmaf(a1, x, (1.0f - a1) * v1);
        v2 = fmaf(a2, x, (1.0f - a2) * v2);
        v3 = fmaf(a3, x, (1.0f - a3) * v3);


        const float denom_l = (v2 - v3) + coco;
        const float lott = denom_l != 0.0f ? (v1 / denom_l) : 0.0f;
        lott_row[i] = lott;


        if (i > 0) {
            const float d = lott - prev_lott;
            if (i >= CMO_P) {
                sum_up -= ring_up[head];
                sum_dn -= ring_dn[head];
            }
            const float up = d > 0.0f ? d : 0.0f;
            const float dn = d > 0.0f ? 0.0f : -d;
            ring_up[head] = up;
            ring_dn[head] = dn;
            sum_up += up;
            sum_dn += dn;
            head = (head + 1) == CMO_P ? 0 : (head + 1);
        }
        prev_lott = lott;

        const float denom = sum_up + sum_dn;
        const float c2 = (i >= CMO_P && denom != 0.0f) ? fabsf((sum_up - sum_dn) / denom) : 0.0f;
        const float a_lott = a_base_lott * c2;
        const float ma = fmaf(a_lott, lott, (1.0f - a_lott) * ma_prev);
        ma_prev = ma;

        if (i == 0) {
            long_stop_prev = ma * (1.0f - fark);
            short_stop_prev = ma * (1.0f + fark);
            const float mt = long_stop_prev;
            hott_row[i] = (ma > mt ? mt * scale_up : mt * scale_dn);
        } else {
            const float ls = ma * (1.0f - fark);
            const float ss = ma * (1.0f + fark);
            const float long_stop = (ma > long_stop_prev) ? fmaxf(ls, long_stop_prev) : ls;
            const float short_stop = (ma < short_stop_prev) ? fminf(ss, short_stop_prev) : ss;
            const int dir = (dir_prev == -1 && ma > short_stop_prev)
                                ? 1
                                : ((dir_prev == 1 && ma < long_stop_prev) ? -1 : dir_prev);
            const float mt = (dir == 1) ? long_stop : short_stop;
            hott_row[i] = (ma > mt ? mt * scale_up : mt * scale_dn);
            long_stop_prev = long_stop;
            short_stop_prev = short_stop;
            dir_prev = dir;
        }
    }
}

extern "C" __global__
void otto_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
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
    const float a1_base = 2.0f / (static_cast<float>(p1) + 1.0f);
    const float a2_base = 2.0f / (static_cast<float>(p2) + 1.0f);
    const float a3_base = 2.0f / (static_cast<float>(p3) + 1.0f);
    const float a_base_lott = 2.0f / (static_cast<float>(max(ott_period, 1)) + 1.0f);
    const float coco = coco_f;
    const float ott_percent = ott_percent_f;
    const float fark = ott_percent * 0.01f;
    const float scale_up = (200.0f + ott_percent) / 200.0f;
    const float scale_dn = (200.0f - ott_percent) / 200.0f;


    const int CMO_P = 9;
    float ring_up_p[CMO_P];
    float ring_dn_p[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) { ring_up_p[k] = 0.0f; ring_dn_p[k] = 0.0f; }
    float sum_up_p = 0.0f, sum_dn_p = 0.0f; int head_p = 0;

    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    float prev_price = 0.0f;


    float ring_up_l[CMO_P];
    float ring_dn_l[CMO_P];
    #pragma unroll
    for (int k = 0; k < CMO_P; ++k) { ring_up_l[k] = 0.0f; ring_dn_l[k] = 0.0f; }
    float sum_up_l = 0.0f, sum_dn_l = 0.0f; int head_l = 0;
    float prev_lott = 0.0f;
    float ma_prev = 0.0f;
    float long_stop_prev = NAN, short_stop_prev = NAN; int dir_prev = 1;

    for (int t = 0; t < cols; ++t) {
        const float x = nzf(prices_tm[t * rows + series]);
        if (t > 0) {
            const float d = x - prev_price;
            if (t >= CMO_P) { sum_up_p -= ring_up_p[head_p]; sum_dn_p -= ring_dn_p[head_p]; }
            const float up = d > 0.0f ? d : 0.0f;
            const float dn = d > 0.0f ? 0.0f : -d;
            ring_up_p[head_p] = up; ring_dn_p[head_p] = dn;
            sum_up_p += up; sum_dn_p += dn; head_p = (head_p + 1) == CMO_P ? 0 : (head_p + 1);
        }
        prev_price = x;
        const float denom_p = sum_up_p + sum_dn_p;
        const float c_abs = (t >= CMO_P && denom_p != 0.0f) ? fabsf((sum_up_p - sum_dn_p) / denom_p) : 0.0f;

        const float a1 = a1_base * c_abs;
        const float a2 = a2_base * c_abs;
        const float a3 = a3_base * c_abs;
        v1 = fmaf(a1, x, (1.0f - a1) * v1);
        v2 = fmaf(a2, x, (1.0f - a2) * v2);
        v3 = fmaf(a3, x, (1.0f - a3) * v3);
        const float denom_l = (v2 - v3) + coco;
        const float lott = denom_l != 0.0f ? (v1 / denom_l) : 0.0f;
        lott_tm[t * rows + series] = lott;

        if (t > 0) {
            const float d = lott - prev_lott;
            if (t >= CMO_P) { sum_up_l -= ring_up_l[head_l]; sum_dn_l -= ring_dn_l[head_l]; }
            const float up = d > 0.0f ? d : 0.0f;
            const float dn = d > 0.0f ? 0.0f : -d;
            ring_up_l[head_l] = up; ring_dn_l[head_l] = dn;
            sum_up_l += up; sum_dn_l += dn; head_l = (head_l + 1) == CMO_P ? 0 : (head_l + 1);
        }
        prev_lott = lott;
        const float denom_lc = sum_up_l + sum_dn_l;
        const float c2 = (t >= CMO_P && denom_lc != 0.0f) ? fabsf((sum_up_l - sum_dn_l) / denom_lc) : 0.0f;
        const float a_lott = a_base_lott * c2;
        const float ma = fmaf(a_lott, lott, (1.0f - a_lott) * ma_prev);
        ma_prev = ma;

        if (t == 0) {
            long_stop_prev = ma * (1.0f - fark);
            short_stop_prev = ma * (1.0f + fark);
            const float mt = long_stop_prev;
            hott_tm[t * rows + series] = (ma > mt ? mt * scale_up : mt * scale_dn);
        } else {
            const float ls = ma * (1.0f - fark);
            const float ss = ma * (1.0f + fark);
            const float long_stop = (ma > long_stop_prev) ? fmaxf(ls, long_stop_prev) : ls;
            const float short_stop = (ma < short_stop_prev) ? fminf(ss, short_stop_prev) : ss;
            const int dir = (dir_prev == -1 && ma > short_stop_prev)
                                ? 1
                                : ((dir_prev == 1 && ma < long_stop_prev) ? -1 : dir_prev);
            const float mt = (dir == 1) ? long_stop : short_stop;
            hott_tm[t * rows + series] = (ma > mt ? mt * scale_up : mt * scale_dn);
            long_stop_prev = long_stop; short_stop_prev = short_stop; dir_prev = dir;
        }
    }
}
