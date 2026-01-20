







#include <cuda_runtime.h>
#include <math.h>
#include "ds_float2.cuh"





#ifndef ZSCORE_COMBO_TILE
#define ZSCORE_COMBO_TILE 4
#endif


__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }
__device__ __forceinline__ bool nonpos_or_nan(float x) { return !(x > 0.0f); }


__device__ __forceinline__ dsf load_dsf_f2(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return ds_make(v.x, v.y);
}



extern "C" __global__ void zscore_sma_prefix_f32ds(
    const float*  __restrict__ data,
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*   __restrict__ periods,
    const float* __restrict__ nbdevs,
    int n_combos,
    float* __restrict__ out
) {
    const int group = blockIdx.y;
    const int co_base = group * ZSCORE_COMBO_TILE;

    __shared__ int s_period[ZSCORE_COMBO_TILE];
    __shared__ int s_warm[ZSCORE_COMBO_TILE];
    __shared__ float s_inv_n[ZSCORE_COMBO_TILE];
    __shared__ float s_inv_nb[ZSCORE_COMBO_TILE];

    if (threadIdx.x < ZSCORE_COMBO_TILE) {
        const int c = co_base + (int)threadIdx.x;
        if (c < n_combos) {
            const int p = periods[c];
            const float nb = nbdevs[c];
            s_period[threadIdx.x] = p;
            s_warm[threadIdx.x] = first_valid + p - 1;
            s_inv_n[threadIdx.x] = (p > 0) ? (1.0f / (float)p) : 0.0f;
            s_inv_nb[threadIdx.x] = (nb != 0.0f) ? (1.0f / nb) : 0.0f;
        } else {
            s_period[threadIdx.x] = 0;
            s_warm[threadIdx.x] = 0;
            s_inv_n[threadIdx.x] = 0.0f;
            s_inv_nb[threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        const int end = t + 1;
        const dsf ex = load_dsf_f2(prefix_sum, end);
        const dsf ex2 = load_dsf_f2(prefix_sum_sq, end);
        const int end_bad = prefix_nan[end];
        const float x = data[t];

#pragma unroll
        for (int k = 0; k < ZSCORE_COMBO_TILE; ++k) {
            const int combo = co_base + k;
            if (combo >= n_combos) break;

            float out_val = nan_f32();
            const int period = s_period[k];
            if (period > 0) {
                const int warm = s_warm[k];
                const float invN = s_inv_n[k];
                const float inv_nbdev = s_inv_nb[k];

                if (t >= warm && inv_nbdev != 0.0f) {
                    int start = end - period;
                    if (start < 0) start = 0;
                    const int nan_count = end_bad - prefix_nan[start];
                    if (nan_count == 0) {

                        const dsf s1 = ds_sub(ex, load_dsf_f2(prefix_sum, start));
                        const dsf s2 = ds_sub(ex2, load_dsf_f2(prefix_sum_sq, start));
                        const dsf mean_ds = ds_scale(s1, invN);
                        const dsf ex2_ds = ds_scale(s2, invN);
                        const dsf var_ds = ds_sub(ex2_ds, ds_mul(mean_ds, mean_ds));
                        const float mean = ds_to_f(mean_ds);
                        const float var = ds_to_f(var_ds);
                        if (var > 0.0f && isfinite(var)) {
                            const float sd = sqrtf(var);
                            const float denom_inv = (sd > 0.0f) ? (inv_nbdev / sd) : 0.0f;
                            out_val = (x - mean) * denom_inv;
                        }
                    }
                }
            }
            out[combo * len + t] = out_val;
        }

        t += stride;
    }
}


extern "C" __global__
void pack_prefix_double_to_float2(const double* __restrict__ src,
                                  float2* __restrict__ dst,
                                  int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double v = src[i];
    float hi = (float)v;
    float lo = (float)(v - (double)hi);
    dst[i] = make_float2(hi, lo);
}


extern "C" __global__ void zscore_sma_prefix_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int* __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ nbdevs,
    int n_combos,
    float* __restrict__ out) {
    const float nan_f = __int_as_float(0x7fffffff);

    const int group = blockIdx.y;
    const int co_base = group * ZSCORE_COMBO_TILE;

    __shared__ int s_period[ZSCORE_COMBO_TILE];
    __shared__ int s_warm[ZSCORE_COMBO_TILE];
    __shared__ double s_inv_n[ZSCORE_COMBO_TILE];
    __shared__ float s_nbdev[ZSCORE_COMBO_TILE];

    if (threadIdx.x < ZSCORE_COMBO_TILE) {
        const int c = co_base + (int)threadIdx.x;
        if (c < n_combos) {
            const int p = periods[c];
            s_period[threadIdx.x] = p;
            s_warm[threadIdx.x] = first_valid + p - 1;
            s_inv_n[threadIdx.x] = (p > 0) ? (1.0 / (double)p) : 0.0;
            s_nbdev[threadIdx.x] = nbdevs[c];
        } else {
            s_period[threadIdx.x] = 0;
            s_warm[threadIdx.x] = 0;
            s_inv_n[threadIdx.x] = 0.0;
            s_nbdev[threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        const int end = t + 1;
        const double ps_end = prefix_sum[end];
        const double ps2_end = prefix_sum_sq[end];
        const int end_bad = prefix_nan[end];
        const double x = (double)data[t];

#pragma unroll
        for (int k = 0; k < ZSCORE_COMBO_TILE; ++k) {
            const int combo = co_base + k;
            if (combo >= n_combos) break;

            float out_val = nan_f;
            const int period = s_period[k];
            if (period > 0) {
                const int warm = s_warm[k];
                const float nbdev = s_nbdev[k];
                const double inv_n = s_inv_n[k];

                if (t >= warm && nbdev != 0.0f) {
                    int start = end - period;
                    if (start < 0) start = 0;

                    const int nan_count = end_bad - prefix_nan[start];
                    if (nan_count == 0) {
                        const double sum = ps_end - prefix_sum[start];
                        const double sum2 = ps2_end - prefix_sum_sq[start];
                        const double mean = sum * inv_n;
                        const double variance = (sum2 * inv_n) - (mean * mean);
                        if (variance > 0.0) {
                            const double denom = sqrt(variance) * (double)nbdev;
                            if (denom != 0.0 && !isnan(denom)) {
                                out_val = (float)((x - mean) / denom);
                            }
                        }
                    }
                }
            }

            out[combo * len + t] = out_val;
        }

        t += stride;
    }
}







extern "C" __global__ void zscore_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    const int* __restrict__ first_valids,
    int period,
    float nbdev,
    int cols,
    int rows,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.x;
    if (series >= cols || period <= 0) return;
    const int stride = cols;


    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= rows) return;

    const int warm = first_valid + period - 1;
    if (nbdev == 0.0f) {

        return;
    }

    const double inv_n = 1.0 / (double)period;


    double s1 = 0.0, s2 = 0.0;
    int nan_in_win = 0;
    const int init_end = min(warm + 1, rows);
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else { const double d = (double)v; s1 += d; s2 += d * d; }
    }

    if (warm < rows && nan_in_win == 0) {
        const double mean = s1 * inv_n;
        const double var = (s2 * inv_n) - (mean * mean);
        if (var > 1e-30) {
            const double sd_nb = sqrt(var) * (double)nbdev;
            const double x = (double)data_tm[warm * stride + series];
            out_tm[warm * stride + series] = (float)((x - mean) / sd_nb);
        }

    }


    for (int t = warm + 1; t < rows; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];

        if (isnan(old_v) || isnan(new_v)) {

            s1 = 0.0; s2 = 0.0; nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float vv = data_tm[k * stride + series];
                if (isnan(vv)) { nan_in_win++; }
                else { const double d = (double)vv; s1 += d; s2 += d * d; }
            }
        } else {

            const double od = (double)old_v;
            const double nd = (double)new_v;
            s1 += nd - od;
            s2 += (nd * nd) - (od * od);
        }

        if (nan_in_win == 0) {
            const double mean = s1 * inv_n;
            const double var  = (s2 * inv_n) - (mean * mean);
            if (var > 1e-30) {
                const double sd_nb = sqrt(var) * (double)nbdev;
                const double x = (double)new_v;
                out_tm[t * stride + series] = (float)((x - mean) / sd_nb);
            } else {

            }
        } else {

        }
    }
}

