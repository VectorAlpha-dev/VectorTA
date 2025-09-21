// CUDA kernels for the Volume Weighted Moving Average (VWMA).
//
// Each parameter combination (period) is assigned to a block in the Y dimension
// while the X dimension iterates over time indices. The kernel operates on
// precomputed prefix sums of price*volume and volume so that every thread only
// performs two subtractions and one division per output value.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void vwma_batch_f32(const float* __restrict__ pv_prefix,
                    const float* __restrict__ vol_prefix,
                    const int* __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int warm = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        float value;
        if (t < warm) {
            value = NAN;
        } else {
            const int start = t - period + 1;
            const int prev = start - 1;

            float sum_pv = pv_prefix[t];
            float sum_vol = vol_prefix[t];

            if (prev >= 0) {
                sum_pv -= pv_prefix[prev];
                sum_vol -= vol_prefix[prev];
            }

            value = sum_pv / sum_vol;
        }

        out[base_out + t] = value;
        t += stride;
    }
}

extern "C" __global__
void vwma_multi_series_one_param_f32(const float* __restrict__ pv_prefix_tm,
                                     const float* __restrict__ vol_prefix_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    const int warm = first_valids[series_idx] + period - 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t - period + 1;
            const int prev = start - 1;
            const int idx = t * num_series + series_idx;

            float sum_pv = pv_prefix_tm[idx];
            float sum_vol = vol_prefix_tm[idx];

            if (prev >= 0) {
                const int prev_idx = prev * num_series + series_idx;
                sum_pv -= pv_prefix_tm[prev_idx];
                sum_vol -= vol_prefix_tm[prev_idx];
            }

            out_tm[out_idx] = sum_pv / sum_vol;
        }

        t += stride;
    }
}
