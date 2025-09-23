// CUDA kernels for the Ehlers Instantaneous Trend (ITrend) indicator.
//
// Mirrors the ALMA-style CUDA API by providing FP32 implementations for
// single-series × many-parameter sweeps and time-major many-series execution.
// Kernels follow the scalar CPU reference closely, reusing ring buffers and
// dynamic dominant-cycle sizing per row.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// FMA-based linear interpolation: prev + a * (x - prev)
__device__ __forceinline__ float lerp_fma(float prev, float x, float a) {
    return __fmaf_rn(a, x - prev, prev);
}

__device__ __forceinline__ float ring_get(const float buf[7], int center, int offset) {
    int idx = center - offset;
    idx += 7;
    idx %= 7;
    return buf[idx];
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
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    const int warmup = warmups[combo];
    const int max_dc = max_dcs[combo];
    if (warmup <= 0 || max_dc <= 0 || max_shared_dc <= 0) {
        return;
    }

    extern __shared__ __align__(16) unsigned char shraw[];
    float* sum_ring = reinterpret_cast<float*>(shraw);

    for (int idx = threadIdx.x; idx < max_shared_dc; idx += blockDim.x) {
        sum_ring[idx] = 0.0f;
    }
    __syncthreads();

    const int row_offset = combo * series_len;
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float fir_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float det_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float i1_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float q1_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float prev_i2 = 0.0f;
    float prev_q2 = 0.0f;
    float prev_re = 0.0f;
    float prev_im = 0.0f;
    float prev_mesa = 0.0f;
    float prev_smooth = 0.0f;
    float prev_it1 = 0.0f;
    float prev_it2 = 0.0f;
    float prev_it3 = 0.0f;
    int sum_idx = 0;
    int ring_ptr = 0;

    const int warm_threshold = first_valid + warmup;
    const float coeff_0962 = 0.0962f;
    const float coeff_5769 = 0.5769f;

    for (int i = 0; i < series_len; ++i) {
        const float x0 = prices[i];
        const float x1 = (i >= 1) ? prices[i - 1] : 0.0f;
        const float x2 = (i >= 2) ? prices[i - 2] : 0.0f;
        const float x3 = (i >= 3) ? prices[i - 3] : 0.0f;

        const float fir_val = 0.1f *
            __fmaf_rn(4.0f, x0,
            __fmaf_rn(3.0f, x1,
            __fmaf_rn(2.0f, x2, x3)));
        fir_buf[ring_ptr] = fir_val;

        const float fir_0 = ring_get(fir_buf, ring_ptr, 0);
        const float fir_2 = ring_get(fir_buf, ring_ptr, 2);
        const float fir_4 = ring_get(fir_buf, ring_ptr, 4);
        const float fir_6 = ring_get(fir_buf, ring_ptr, 6);

        const float h_in = __fmaf_rn(coeff_0962, fir_0,
                           __fmaf_rn(coeff_5769, fir_2,
                           __fmaf_rn(-coeff_5769, fir_4, -coeff_0962 * fir_6)));
        const float period_mult = __fmaf_rn(0.075f, prev_mesa, 0.54f);
        const float det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        const float i1_val = ring_get(det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        const float det_0 = ring_get(det_buf, ring_ptr, 0);
        const float det_2 = ring_get(det_buf, ring_ptr, 2);
        const float det_4 = ring_get(det_buf, ring_ptr, 4);
        const float det_6 = ring_get(det_buf, ring_ptr, 6);
        const float h_in_q1 = __fmaf_rn(coeff_0962, det_0,
                               __fmaf_rn(coeff_5769, det_2,
                               __fmaf_rn(-coeff_5769, det_4, -coeff_0962 * det_6)));
        const float q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        const float i1_0 = ring_get(i1_buf, ring_ptr, 0);
        const float i1_2 = ring_get(i1_buf, ring_ptr, 2);
        const float i1_4 = ring_get(i1_buf, ring_ptr, 4);
        const float i1_6 = ring_get(i1_buf, ring_ptr, 6);
        const float j_i_val = __fmaf_rn(coeff_0962, i1_0,
                               __fmaf_rn(coeff_5769, i1_2,
                               __fmaf_rn(-coeff_5769, i1_4, -coeff_0962 * i1_6))) * period_mult;

        const float q1_0 = ring_get(q1_buf, ring_ptr, 0);
        const float q1_2 = ring_get(q1_buf, ring_ptr, 2);
        const float q1_4 = ring_get(q1_buf, ring_ptr, 4);
        const float q1_6 = ring_get(q1_buf, ring_ptr, 6);
        const float j_q_val = __fmaf_rn(coeff_0962, q1_0,
                               __fmaf_rn(coeff_5769, q1_2,
                               __fmaf_rn(-coeff_5769, q1_4, -coeff_0962 * q1_6))) * period_mult;

        const float i2_cur = __fmaf_rn(0.2f, (i1_val - j_q_val), 0.8f * prev_i2);
        const float q2_cur = __fmaf_rn(0.2f, (q1_val + j_i_val), 0.8f * prev_q2);

        const float re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        const float im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        const float re_smooth = lerp_fma(prev_re, re_val, 0.2f);
        const float im_smooth = lerp_fma(prev_im, im_val, 0.2f);
        prev_re = re_smooth;
        prev_im = im_smooth;

        float new_mesa = 0.0f;
        if (re_smooth != 0.0f && im_smooth != 0.0f) {
            const float phase = atan2f(im_smooth, re_smooth);
            new_mesa = 2.0f * CUDART_PI_F / phase;
        }
        const float up_lim = 1.5f * prev_mesa;
        if (new_mesa > up_lim) {
            new_mesa = up_lim;
        }
        const float low_lim = 0.67f * prev_mesa;
        if (new_mesa < low_lim) {
            new_mesa = low_lim;
        }
        if (new_mesa < 6.0f) {
            new_mesa = 6.0f;
        } else if (new_mesa > 50.0f) {
            new_mesa = 50.0f;
        }
        const float final_mesa = lerp_fma(prev_mesa, new_mesa, 0.2f);
        prev_mesa = final_mesa;
        const float sp_val = lerp_fma(prev_smooth, final_mesa, 0.33f);
        prev_smooth = sp_val;

        int dcp = static_cast<int>(floorf(sp_val + 0.5f));
        if (dcp < 1) {
            dcp = 1;
        }
        if (dcp > max_dc) {
            dcp = max_dc;
        }

        sum_ring[sum_idx] = x0;
        sum_idx += 1;
        if (sum_idx >= max_dc) {
            sum_idx = 0;
        }

        float sum_src = 0.0f, c = 0.0f;
        int idx2 = sum_idx;
        for (int cnt = 0; cnt < dcp; ++cnt) {
            idx2 = (idx2 == 0) ? (max_dc - 1) : (idx2 - 1);
            float y = sum_ring[idx2] - c;
            float t = sum_src + y;
            c = (t - sum_src) - y;
            sum_src = t;
        }
        const float it_val = sum_src / static_cast<float>(dcp);

        const float eit_val = (i < warmup)
            ? x0
            : 0.1f * __fmaf_rn(4.0f, it_val,
                      __fmaf_rn(3.0f, prev_it1,
                      __fmaf_rn(2.0f, prev_it2, prev_it3)));

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        if (i >= warm_threshold) {
            out[row_offset + i] = eit_val;
        }

        ring_ptr = (ring_ptr + 1) % 7;
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
    if (series_idx >= num_series || series_len <= 0) {
        return;
    }
    if (warmup <= 0 || max_dc <= 0) {
        return;
    }

    const int stride = num_series;

    extern __shared__ __align__(16) unsigned char shraw[];
    float* sum_ring = reinterpret_cast<float*>(shraw);
    for (int idx = threadIdx.x; idx < max_dc; idx += blockDim.x) {
        sum_ring[idx] = 0.0f;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float fir_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float det_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float i1_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float q1_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float prev_i2 = 0.0f;
    float prev_q2 = 0.0f;
    float prev_re = 0.0f;
    float prev_im = 0.0f;
    float prev_mesa = 0.0f;
    float prev_smooth = 0.0f;
    float prev_it1 = 0.0f;
    float prev_it2 = 0.0f;
    float prev_it3 = 0.0f;
    int sum_idx = 0;
    int ring_ptr = 0;

    const int first_valid = first_valids[series_idx];
    const int warm_threshold = first_valid + warmup;
    const float coeff_0962 = 0.0962f;
    const float coeff_5769 = 0.5769f;

    for (int t = 0; t < series_len; ++t) {
        const int idx = t * stride + series_idx;
        const float x0 = prices_tm[idx];
        const float x1 = (t >= 1) ? prices_tm[(t - 1) * stride + series_idx] : 0.0f;
        const float x2 = (t >= 2) ? prices_tm[(t - 2) * stride + series_idx] : 0.0f;
        const float x3 = (t >= 3) ? prices_tm[(t - 3) * stride + series_idx] : 0.0f;

        const float fir_val = 0.1f *
            __fmaf_rn(4.0f, x0,
            __fmaf_rn(3.0f, x1,
            __fmaf_rn(2.0f, x2, x3)));
        fir_buf[ring_ptr] = fir_val;

        const float fir_0 = ring_get(fir_buf, ring_ptr, 0);
        const float fir_2 = ring_get(fir_buf, ring_ptr, 2);
        const float fir_4 = ring_get(fir_buf, ring_ptr, 4);
        const float fir_6 = ring_get(fir_buf, ring_ptr, 6);

        const float h_in = __fmaf_rn(coeff_0962, fir_0,
                           __fmaf_rn(coeff_5769, fir_2,
                           __fmaf_rn(-coeff_5769, fir_4, -coeff_0962 * fir_6)));
        const float period_mult = __fmaf_rn(0.075f, prev_mesa, 0.54f);
        const float det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        const float i1_val = ring_get(det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        const float det_0 = ring_get(det_buf, ring_ptr, 0);
        const float det_2 = ring_get(det_buf, ring_ptr, 2);
        const float det_4 = ring_get(det_buf, ring_ptr, 4);
        const float det_6 = ring_get(det_buf, ring_ptr, 6);
        const float h_in_q1 = __fmaf_rn(coeff_0962, det_0,
                               __fmaf_rn(coeff_5769, det_2,
                               __fmaf_rn(-coeff_5769, det_4, -coeff_0962 * det_6)));
        const float q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        const float i1_0 = ring_get(i1_buf, ring_ptr, 0);
        const float i1_2 = ring_get(i1_buf, ring_ptr, 2);
        const float i1_4 = ring_get(i1_buf, ring_ptr, 4);
        const float i1_6 = ring_get(i1_buf, ring_ptr, 6);
        const float j_i_val = __fmaf_rn(coeff_0962, i1_0,
                               __fmaf_rn(coeff_5769, i1_2,
                               __fmaf_rn(-coeff_5769, i1_4, -coeff_0962 * i1_6))) * period_mult;

        const float q1_0 = ring_get(q1_buf, ring_ptr, 0);
        const float q1_2 = ring_get(q1_buf, ring_ptr, 2);
        const float q1_4 = ring_get(q1_buf, ring_ptr, 4);
        const float q1_6 = ring_get(q1_buf, ring_ptr, 6);
        const float j_q_val = __fmaf_rn(coeff_0962, q1_0,
                               __fmaf_rn(coeff_5769, q1_2,
                               __fmaf_rn(-coeff_5769, q1_4, -coeff_0962 * q1_6))) * period_mult;

        const float i2_cur = __fmaf_rn(0.2f, (i1_val - j_q_val), 0.8f * prev_i2);
        const float q2_cur = __fmaf_rn(0.2f, (q1_val + j_i_val), 0.8f * prev_q2);

        const float re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        const float im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        const float re_smooth = lerp_fma(prev_re, re_val, 0.2f);
        const float im_smooth = lerp_fma(prev_im, im_val, 0.2f);
        prev_re = re_smooth;
        prev_im = im_smooth;

        float new_mesa = 0.0f;
        if (re_smooth != 0.0f && im_smooth != 0.0f) {
            const float phase = atan2f(im_smooth, re_smooth);
            new_mesa = 2.0f * CUDART_PI_F / phase;
        }
        const float up_lim = 1.5f * prev_mesa;
        if (new_mesa > up_lim) {
            new_mesa = up_lim;
        }
        const float low_lim = 0.67f * prev_mesa;
        if (new_mesa < low_lim) {
            new_mesa = low_lim;
        }
        if (new_mesa < 6.0f) {
            new_mesa = 6.0f;
        } else if (new_mesa > 50.0f) {
            new_mesa = 50.0f;
        }
        const float final_mesa = lerp_fma(prev_mesa, new_mesa, 0.2f);
        prev_mesa = final_mesa;
        const float sp_val = lerp_fma(prev_smooth, final_mesa, 0.33f);
        prev_smooth = sp_val;

        int dcp = static_cast<int>(floorf(sp_val + 0.5f));
        if (dcp < 1) {
            dcp = 1;
        }
        if (dcp > max_dc) {
            dcp = max_dc;
        }

        sum_ring[sum_idx] = x0;
        sum_idx += 1;
        if (sum_idx >= max_dc) {
            sum_idx = 0;
        }

        float sum_src = 0.0f, c = 0.0f;
        int idx2 = sum_idx;
        for (int cnt = 0; cnt < dcp; ++cnt) {
            idx2 = (idx2 == 0) ? (max_dc - 1) : (idx2 - 1);
            float y = sum_ring[idx2] - c;
            float t = sum_src + y;
            c = (t - sum_src) - y;
            sum_src = t;
        }
        const float it_val = sum_src / static_cast<float>(dcp);

        const float eit_val = (t < warmup)
            ? x0
            : 0.1f * __fmaf_rn(4.0f, it_val,
                      __fmaf_rn(3.0f, prev_it1,
                      __fmaf_rn(2.0f, prev_it2, prev_it3)));

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        if (t >= warm_threshold) {
            out_tm[idx] = eit_val;
        }

        ring_ptr = (ring_ptr + 1) % 7;
    }
}
