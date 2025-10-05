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

// Use templated ring accessor to support both float and double local rings
template <typename T>
__device__ __forceinline__ T ring_get_t(const T buf[7], int center, int offset) {
    int idx = center - offset;
    idx += 7;
    idx %= 7;
    return buf[idx];
}

// FMA-based linear interpolation: prev + a * (x - prev)
__device__ __forceinline__ float lerp_fma(float prev, float x, float a) {
    return __fmaf_rn(a, x - prev, prev);
}

// Back-compat helper for float rings
__device__ __forceinline__ float ring_get(const float buf[7], int center, int offset) {
    return ring_get_t<float>(buf, center, offset);
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

    float  fir_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Promote numerically sensitive paths to FP64 for CPU parity
    double det_buf[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double i1_buf[7]  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double q1_buf[7]  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double prev_i2 = 0.0;
    double prev_q2 = 0.0;
    double prev_re = 0.0;
    double prev_im = 0.0;
    double prev_mesa = 0.0;
    double prev_smooth = 0.0;
    double prev_it1 = 0.0;
    double prev_it2 = 0.0;
    double prev_it3 = 0.0;
    int sum_idx = 0;
    int ring_ptr = 0;

    const int warm_threshold = first_valid + warmup;
    const double coeff_0962 = 0.0962;
    const double coeff_5769 = 0.5769;

    for (int i = 0; i < series_len; ++i) {
        const float x0 = prices[i];
        const float x1 = (i >= 1) ? prices[i - 1] : 0.0f;
        const float x2 = (i >= 2) ? prices[i - 2] : 0.0f;
        const float x3 = (i >= 3) ? prices[i - 3] : 0.0f;

        const double fir_val_d = (4.0 * (double)x0 + 3.0 * (double)x1 + 2.0 * (double)x2 + (double)x3) / 10.0;
        const float  fir_val   = (float)fir_val_d;
        fir_buf[ring_ptr] = fir_val;

        const double fir_0 = (double)ring_get(fir_buf, ring_ptr, 0);
        const double fir_2 = (double)ring_get(fir_buf, ring_ptr, 2);
        const double fir_4 = (double)ring_get(fir_buf, ring_ptr, 4);
        const double fir_6 = (double)ring_get(fir_buf, ring_ptr, 6);

        const double h_in = coeff_0962 * fir_0 + coeff_5769 * fir_2 - coeff_5769 * fir_4 - coeff_0962 * fir_6;
        const double period_mult = 0.075 * prev_mesa + 0.54;
        const double det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        const double i1_val = ring_get_t<double>(det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        const double det_0 = ring_get_t<double>(det_buf, ring_ptr, 0);
        const double det_2 = ring_get_t<double>(det_buf, ring_ptr, 2);
        const double det_4 = ring_get_t<double>(det_buf, ring_ptr, 4);
        const double det_6 = ring_get_t<double>(det_buf, ring_ptr, 6);
        const double h_in_q1 = coeff_0962 * det_0 + coeff_5769 * det_2 - coeff_5769 * det_4 - coeff_0962 * det_6;
        const double q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        const double i1_0 = ring_get_t<double>(i1_buf, ring_ptr, 0);
        const double i1_2 = ring_get_t<double>(i1_buf, ring_ptr, 2);
        const double i1_4 = ring_get_t<double>(i1_buf, ring_ptr, 4);
        const double i1_6 = ring_get_t<double>(i1_buf, ring_ptr, 6);
        const double j_i_val = (coeff_0962 * i1_0 + coeff_5769 * i1_2 - coeff_5769 * i1_4 - coeff_0962 * i1_6) * period_mult;

        const double q1_0 = ring_get_t<double>(q1_buf, ring_ptr, 0);
        const double q1_2 = ring_get_t<double>(q1_buf, ring_ptr, 2);
        const double q1_4 = ring_get_t<double>(q1_buf, ring_ptr, 4);
        const double q1_6 = ring_get_t<double>(q1_buf, ring_ptr, 6);
        const double j_q_val = (coeff_0962 * q1_0 + coeff_5769 * q1_2 - coeff_5769 * q1_4 - coeff_0962 * q1_6) * period_mult;

        const double i2_cur_d = 0.2 * (i1_val - j_q_val) + 0.8 * prev_i2;
        const double q2_cur_d = 0.2 * (q1_val + j_i_val) + 0.8 * prev_q2;

        const double re_val_d = i2_cur_d * prev_i2 + q2_cur_d * prev_q2;
        const double im_val_d = i2_cur_d * prev_q2 - q2_cur_d * prev_i2;
        prev_i2 = i2_cur_d;
        prev_q2 = q2_cur_d;

        const double re_smooth = prev_re + 0.2 * (re_val_d - prev_re);
        const double im_smooth = prev_im + 0.2 * (im_val_d - prev_im);
        prev_re = re_smooth;
        prev_im = im_smooth;

        double new_mesa_d = 0.0;
        if (re_smooth != 0.0 && im_smooth != 0.0) {
            const double phase = atan(im_smooth / re_smooth);
            new_mesa_d = 2.0 * M_PI / phase;
        }
        const double up_lim_d = 1.5 * prev_mesa;
        if (new_mesa_d > up_lim_d) { new_mesa_d = up_lim_d; }
        const double low_lim_d = 0.67 * prev_mesa;
        if (new_mesa_d < low_lim_d) { new_mesa_d = low_lim_d; }
        if (new_mesa_d < 6.0) new_mesa_d = 6.0; else if (new_mesa_d > 50.0) new_mesa_d = 50.0;
        const double final_mesa_d = prev_mesa + 0.2 * (new_mesa_d - prev_mesa);
        prev_mesa = final_mesa_d;
        const double sp_val = prev_smooth + 0.33 * (final_mesa_d - prev_smooth);
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

        double sum_src = 0.0;
        int idx2 = sum_idx;
        for (int cnt = 0; cnt < dcp; ++cnt) {
            idx2 = (idx2 == 0) ? (max_dc - 1) : (idx2 - 1);
            sum_src += (double)sum_ring[idx2];
        }
        const double it_val = sum_src / (double)dcp;

        const double eit_val = (i < warmup)
            ? (double)x0
            : (4.0 * it_val + 3.0 * prev_it1 + 2.0 * prev_it2 + prev_it3) / 10.0;

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        if (i >= warm_threshold) {
            out[row_offset + i] = (float)eit_val;
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

    float  fir_buf[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    double det_buf[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double i1_buf[7]  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double q1_buf[7]  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double prev_i2 = 0.0;
    double prev_q2 = 0.0;
    double prev_re = 0.0;
    double prev_im = 0.0;
    double prev_mesa = 0.0;
    double prev_smooth = 0.0;
    double prev_it1 = 0.0;
    double prev_it2 = 0.0;
    double prev_it3 = 0.0;
    int sum_idx = 0;
    int ring_ptr = 0;

    const int first_valid = first_valids[series_idx];
    const int warm_threshold = first_valid + warmup;
    const double coeff_0962 = 0.0962;
    const double coeff_5769 = 0.5769;

    for (int t = 0; t < series_len; ++t) {
        const int idx = t * stride + series_idx;
        const float  x0 = prices_tm[idx];
        const float  x1 = (t >= 1) ? prices_tm[(t - 1) * stride + series_idx] : 0.0f;
        const float  x2 = (t >= 2) ? prices_tm[(t - 2) * stride + series_idx] : 0.0f;
        const float  x3 = (t >= 3) ? prices_tm[(t - 3) * stride + series_idx] : 0.0f;

        const double fir_val_d = (4.0 * (double)x0 + 3.0 * (double)x1 + 2.0 * (double)x2 + (double)x3) / 10.0;
        const float  fir_val   = (float)fir_val_d;
        fir_buf[ring_ptr] = fir_val;

        const double fir_0 = (double)ring_get(fir_buf, ring_ptr, 0);
        const double fir_2 = (double)ring_get(fir_buf, ring_ptr, 2);
        const double fir_4 = (double)ring_get(fir_buf, ring_ptr, 4);
        const double fir_6 = (double)ring_get(fir_buf, ring_ptr, 6);

        const double h_in = coeff_0962 * fir_0 + coeff_5769 * fir_2 - coeff_5769 * fir_4 - coeff_0962 * fir_6;
        const double period_mult = 0.075 * prev_mesa + 0.54;
        const double det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        const double i1_val = ring_get_t<double>(det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        const double det_0 = ring_get_t<double>(det_buf, ring_ptr, 0);
        const double det_2 = ring_get_t<double>(det_buf, ring_ptr, 2);
        const double det_4 = ring_get_t<double>(det_buf, ring_ptr, 4);
        const double det_6 = ring_get_t<double>(det_buf, ring_ptr, 6);
        const double h_in_q1 = coeff_0962 * det_0 + coeff_5769 * det_2 - coeff_5769 * det_4 - coeff_0962 * det_6;
        const double q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        const double i1_0 = ring_get_t<double>(i1_buf, ring_ptr, 0);
        const double i1_2 = ring_get_t<double>(i1_buf, ring_ptr, 2);
        const double i1_4 = ring_get_t<double>(i1_buf, ring_ptr, 4);
        const double i1_6 = ring_get_t<double>(i1_buf, ring_ptr, 6);
        const double j_i_val = (coeff_0962 * i1_0 + coeff_5769 * i1_2 - coeff_5769 * i1_4 - coeff_0962 * i1_6) * period_mult;

        const double q1_0 = ring_get_t<double>(q1_buf, ring_ptr, 0);
        const double q1_2 = ring_get_t<double>(q1_buf, ring_ptr, 2);
        const double q1_4 = ring_get_t<double>(q1_buf, ring_ptr, 4);
        const double q1_6 = ring_get_t<double>(q1_buf, ring_ptr, 6);
        const double j_q_val = (coeff_0962 * q1_0 + coeff_5769 * q1_2 - coeff_5769 * q1_4 - coeff_0962 * q1_6) * period_mult;

        const double i2_cur_d2 = 0.2 * (i1_val - j_q_val) + 0.8 * prev_i2;
        const double q2_cur_d2 = 0.2 * (q1_val + j_i_val) + 0.8 * prev_q2;

        const double re_val_d2 = i2_cur_d2 * prev_i2 + q2_cur_d2 * prev_q2;
        const double im_val_d2 = i2_cur_d2 * prev_q2 - q2_cur_d2 * prev_i2;
        prev_i2 = i2_cur_d2;
        prev_q2 = q2_cur_d2;

        const double re_smooth2 = prev_re + 0.2 * (re_val_d2 - prev_re);
        const double im_smooth2 = prev_im + 0.2 * (im_val_d2 - prev_im);
        prev_re = re_smooth2;
        prev_im = im_smooth2;

        double new_mesa_d = 0.0;
        if (re_smooth2 != 0.0 && im_smooth2 != 0.0) {
            const double phase = atan(im_smooth2 / re_smooth2);
            new_mesa_d = 2.0 * M_PI / phase;
        }
        const double up_lim_d = 1.5 * prev_mesa;
        if (new_mesa_d > up_lim_d) { new_mesa_d = up_lim_d; }
        const double low_lim_d = 0.67 * prev_mesa;
        if (new_mesa_d < low_lim_d) { new_mesa_d = low_lim_d; }
        if (new_mesa_d < 6.0) new_mesa_d = 6.0; else if (new_mesa_d > 50.0) new_mesa_d = 50.0;
        const double final_mesa_d = prev_mesa + 0.2 * (new_mesa_d - prev_mesa);
        prev_mesa = final_mesa_d;
        const double sp_val = prev_smooth + 0.33 * (final_mesa_d - prev_smooth);
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

        double sum_src = 0.0;
        int idx2 = sum_idx;
        for (int cnt = 0; cnt < dcp; ++cnt) {
            idx2 = (idx2 == 0) ? (max_dc - 1) : (idx2 - 1);
            sum_src += (double)sum_ring[idx2];
        }
        const double it_val = sum_src / (double)dcp;

        const double eit_val = (t < warmup)
            ? (double)x0
            : (4.0 * it_val + 3.0 * prev_it1 + 2.0 * prev_it2 + prev_it3) / 10.0;

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        if (t >= warm_threshold) {
            out_tm[idx] = (float)eit_val;
        }

        ring_ptr = (ring_ptr + 1) % 7;
    }
}
