










#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>



#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif


__device__ __forceinline__ float lerp_fma(float prev, float x, float a) {
    return __fmaf_rn(a, x - prev, prev);
}

template <typename T>
__device__ __forceinline__ T clampT(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

extern "C" __global__
void ehlers_itrend_batch_f32(const float* __restrict__ prices,
                             const int* __restrict__ warmups,
                             const int* __restrict__ max_dcs,
                             int series_len,
                             int first_valid,
                             int n_combos,
                             int max_shared_dc,
                             float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int warmup = warmups[combo];
    const int max_dc = max_dcs[combo];
    if (warmup <= 0 || max_dc <= 0 || max_shared_dc <= 0) return;
    if (max_shared_dc < max_dc) return;


    if (threadIdx.x != 0) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    float* __restrict__ pfx = reinterpret_cast<float*>(shraw);
    const int cap = max_dc;
    for (int i = 0; i < cap; ++i) pfx[i] = 0.0f;

    const int row_offset = combo * series_len;


    float fir_buf[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float det_buf[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float i1_buf[7]  = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float q1_buf[7]  = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float prev_i2 = 0.0f, prev_q2 = 0.0f;
    float prev_re = 0.0f, prev_im = 0.0f;
    float prev_mesa = 0.0f, prev_smooth = 0.0f;
    float prev_it1 = 0.0f, prev_it2 = 0.0f, prev_it3 = 0.0f;

    int ring_ptr = 0;
    int pidx = 0;
    float pcur = 0.0f;

    const int warm_threshold = first_valid + warmup;
    const float c0962 = 0.0962f;
    const float c5769 = 0.5769f;

    for (int i = 0; i < series_len; ++i) {
        const float x0 = prices[i];
        const float x1 = (i >= 1) ? prices[i - 1] : 0.0f;
        const float x2 = (i >= 2) ? prices[i - 2] : 0.0f;
        const float x3 = (i >= 3) ? prices[i - 3] : 0.0f;

        const float fir_val = (4.0f * x0 + 3.0f * x1 + 2.0f * x2 + x3) * 0.1f;
        fir_buf[ring_ptr] = fir_val;


        const int c  = ring_ptr;
        const int c2 = (c >= 2) ? (c - 2) : (c + 5);
        const int c4 = (c >= 4) ? (c - 4) : (c + 3);
        const int c6 = (c >= 6) ? (c - 6) : (c + 1);
        const int c3 = (c >= 3) ? (c - 3) : (c + 4);

        const float fir_0 = fir_buf[c];
        const float fir_2 = fir_buf[c2];
        const float fir_4 = fir_buf[c4];
        const float fir_6 = fir_buf[c6];

        const float period_mult = 0.075f * prev_mesa + 0.54f;
        const float h_in = c0962 * fir_0 + c5769 * fir_2 - c5769 * fir_4 - c0962 * fir_6;

        const float det_val = h_in * period_mult;
        det_buf[c] = det_val;

        const float i1_val = det_buf[c3];
        i1_buf[c] = i1_val;

        const float det_0 = det_buf[c];
        const float det_2 = det_buf[c2];
        const float det_4 = det_buf[c4];
        const float det_6 = det_buf[c6];

        const float h_in_q1 = c0962 * det_0 + c5769 * det_2 - c5769 * det_4 - c0962 * det_6;
        const float q1_val = h_in_q1 * period_mult;
        q1_buf[c] = q1_val;

        const float i1_0 = i1_buf[c];
        const float i1_2 = i1_buf[c2];
        const float i1_4 = i1_buf[c4];
        const float i1_6 = i1_buf[c6];
        const float j_i_val = (c0962 * i1_0 + c5769 * i1_2 - c5769 * i1_4 - c0962 * i1_6) * period_mult;

        const float q1_0 = q1_buf[c];
        const float q1_2 = q1_buf[c2];
        const float q1_4 = q1_buf[c4];
        const float q1_6 = q1_buf[c6];
        const float j_q_val = (c0962 * q1_0 + c5769 * q1_2 - c5769 * q1_4 - c0962 * q1_6) * period_mult;

        const float i2_cur = 0.2f * (i1_val - j_q_val) + 0.8f * prev_i2;
        const float q2_cur = 0.2f * (q1_val + j_i_val) + 0.8f * prev_q2;

        const float re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        const float im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        const float re_smooth = prev_re + 0.2f * (re_val - prev_re);
        const float im_smooth = prev_im + 0.2f * (im_val - prev_im);
        prev_re = re_smooth;
        prev_im = im_smooth;

        float new_mesa = 0.0f;
        if (re_smooth != 0.0f || im_smooth != 0.0f) {
            const float phase = atan2f(im_smooth, re_smooth);
            if (phase != 0.0f) new_mesa = (2.0f * CUDART_PI_F) / phase;
        }

        const float up_lim  = 1.5f * prev_mesa;
        const float low_lim = 0.67f * prev_mesa;
        new_mesa = clampT(new_mesa, low_lim, up_lim);
        new_mesa = clampT(new_mesa, 6.0f, 50.0f);
        const float final_mesa = prev_mesa + 0.2f * (new_mesa - prev_mesa);
        prev_mesa = final_mesa;
        const float sp_val = prev_smooth + 0.33f * (final_mesa - prev_smooth);
        prev_smooth = sp_val;

        int dcp = __float2int_rn(sp_val);
        dcp = clampT(dcp, 1, max_dc);



        float old = pfx[pidx];
        pidx += 1; if (pidx >= cap) pidx = 0;
        pcur += x0;
        int pback = pidx - dcp; if (pback < 0) pback += cap;
        const float prev_prefix = (pback == pidx) ? old : pfx[pback];
        const float sum_src = pcur - prev_prefix;
        pfx[pidx] = pcur;
        const float it_val  = sum_src / (float)dcp;

        const float eit_val = (i < warmup)
            ? x0
            : (4.0f * it_val + 3.0f * prev_it1 + 2.0f * prev_it2 + prev_it3) * 0.1f;

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        out[row_offset + i] = (i >= warm_threshold) ? eit_val : CUDART_NAN_F;

        ring_ptr = (c == 6) ? 0 : (c + 1);
    }
}

extern "C" __global__
void ehlers_itrend_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int warmup,
    int max_dc,
    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) return;
    if (warmup <= 0 || max_dc <= 0) return;

    const int stride = num_series;


    if (threadIdx.x != 0) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    float* __restrict__ pfx = reinterpret_cast<float*>(shraw);
    const int cap = max_dc;
    for (int i = 0; i < cap; ++i) pfx[i] = 0.0f;

    float fir_buf[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float det_buf[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float i1_buf[7]  = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float q1_buf[7]  = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float prev_i2 = 0.0f, prev_q2 = 0.0f;
    float prev_re = 0.0f, prev_im = 0.0f;
    float prev_mesa = 0.0f, prev_smooth = 0.0f;
    float prev_it1 = 0.0f, prev_it2 = 0.0f, prev_it3 = 0.0f;
    int ring_ptr = 0;
    int pidx = 0;
    float pcur = 0.0f;

    const int first_valid = first_valids[series_idx];
    const int warm_threshold = first_valid + warmup;
    const float c0962 = 0.0962f;
    const float c5769 = 0.5769f;

    for (int t = 0; t < series_len; ++t) {
        const int idx = t * stride + series_idx;
        const float x0 = prices_tm[idx];
        const float x1 = (t >= 1) ? prices_tm[(t - 1) * stride + series_idx] : 0.0f;
        const float x2 = (t >= 2) ? prices_tm[(t - 2) * stride + series_idx] : 0.0f;
        const float x3 = (t >= 3) ? prices_tm[(t - 3) * stride + series_idx] : 0.0f;

        const float fir_val = (4.0f * x0 + 3.0f * x1 + 2.0f * x2 + x3) * 0.1f;
        fir_buf[ring_ptr] = fir_val;

        const int c  = ring_ptr;
        const int c2 = (c >= 2) ? (c - 2) : (c + 5);
        const int c4 = (c >= 4) ? (c - 4) : (c + 3);
        const int c6 = (c >= 6) ? (c - 6) : (c + 1);
        const int c3 = (c >= 3) ? (c - 3) : (c + 4);

        const float fir_0 = fir_buf[c];
        const float fir_2 = fir_buf[c2];
        const float fir_4 = fir_buf[c4];
        const float fir_6 = fir_buf[c6];

        const float period_mult = 0.075f * prev_mesa + 0.54f;
        const float h_in = c0962 * fir_0 + c5769 * fir_2 - c5769 * fir_4 - c0962 * fir_6;

        const float det_val = h_in * period_mult;
        det_buf[c] = det_val;

        const float i1_val = det_buf[c3];
        i1_buf[c] = i1_val;

        const float det_0 = det_buf[c];
        const float det_2 = det_buf[c2];
        const float det_4 = det_buf[c4];
        const float det_6 = det_buf[c6];

        const float h_in_q1 = c0962 * det_0 + c5769 * det_2 - c5769 * det_4 - c0962 * det_6;
        const float q1_val = h_in_q1 * period_mult;
        q1_buf[c] = q1_val;

        const float i1_0 = i1_buf[c];
        const float i1_2 = i1_buf[c2];
        const float i1_4 = i1_buf[c4];
        const float i1_6 = i1_buf[c6];
        const float j_i_val = (c0962 * i1_0 + c5769 * i1_2 - c5769 * i1_4 - c0962 * i1_6) * period_mult;

        const float q1_0 = q1_buf[c];
        const float q1_2 = q1_buf[c2];
        const float q1_4 = q1_buf[c4];
        const float q1_6 = q1_buf[c6];
        const float j_q_val = (c0962 * q1_0 + c5769 * q1_2 - c5769 * q1_4 - c0962 * q1_6) * period_mult;

        const float i2_cur = 0.2f * (i1_val - j_q_val) + 0.8f * prev_i2;
        const float q2_cur = 0.2f * (q1_val + j_i_val) + 0.8f * prev_q2;

        const float re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        const float im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        const float re_smooth = prev_re + 0.2f * (re_val - prev_re);
        const float im_smooth = prev_im + 0.2f * (im_val - prev_im);
        prev_re = re_smooth;
        prev_im = im_smooth;

        float new_mesa = 0.0f;
        if (re_smooth != 0.0f || im_smooth != 0.0f) {
            const float phase = atan2f(im_smooth, re_smooth);
            if (phase != 0.0f) new_mesa = (2.0f * CUDART_PI_F) / phase;
        }
        const float up_lim  = 1.5f * prev_mesa;
        const float low_lim = 0.67f * prev_mesa;
        new_mesa = clampT(new_mesa, low_lim, up_lim);
        new_mesa = clampT(new_mesa, 6.0f, 50.0f);
        const float final_mesa = prev_mesa + 0.2f * (new_mesa - prev_mesa);
        prev_mesa = final_mesa;
        const float sp_val = prev_smooth + 0.33f * (final_mesa - prev_smooth);
        prev_smooth = sp_val;

        int dcp = __float2int_rn(sp_val);
        dcp = clampT(dcp, 1, max_dc);


        float old = pfx[pidx];
        pidx += 1; if (pidx >= cap) pidx = 0;
        pcur += x0;
        int pback = pidx - dcp; if (pback < 0) pback += cap;
        const float prev_prefix = (pback == pidx) ? old : pfx[pback];
        const float sum_src = pcur - prev_prefix;
        pfx[pidx] = pcur;
        const float it_val  = sum_src / (float)dcp;

        const float eit_val = (t < warmup)
            ? x0
            : (4.0f * it_val + 3.0f * prev_it1 + 2.0f * prev_it2 + prev_it3) * 0.1f;

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        out_tm[idx] = (t >= warm_threshold) ? eit_val : CUDART_NAN_F;

        ring_ptr = (c == 6) ? 0 : (c + 1);
    }
}
