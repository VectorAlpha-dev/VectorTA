// CUDA kernels for MinMax (Local Extrema)
//
// Semantics mirror the scalar Rust implementation in src/indicators/minmax.rs:
// - Inputs: high, low (f32)
// - Parameter: order (window radius). A point i is a local min if:
//     - low[i] is finite
//     - all k neighbors on both sides are finite (low)
//     - low[i] < min(low[i-1..i-k]) and low[i] < min(low[i+1..i+k])
//   Similarly for local max using high and strict > with max over neighbors.
// - Warmup/NaN rules:
//     - Write NaN for indices < first_valid
//     - is_min/is_max remain NaN unless the strict inequalities are satisfied
//     - last_min/last_max forward-fill the most recent extrema, starting from NaN
// - Many-series kernel expects time-major layout: index = t * num_series + s

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void minmax_batch_f32(const float* __restrict__ high,
                      const float* __restrict__ low,
                      int series_len,
                      int first_valid,
                      const int* __restrict__ orders,
                      int n_combos,
                      float* __restrict__ out_is_min,
                      float* __restrict__ out_is_max,
                      float* __restrict__ out_last_min,
                      float* __restrict__ out_last_max) {
    if (series_len <= 0) return;
    const int row = blockIdx.y;
    if (row >= n_combos) return;

    const int base = row * series_len;
    // Initialize this row's outputs to NaN in parallel
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += blockDim.x * gridDim.x) {
        out_is_min[base + i] = CUDART_NAN_F;
        out_is_max[base + i] = CUDART_NAN_F;
        out_last_min[base + i] = CUDART_NAN_F;
        out_last_max[base + i] = CUDART_NAN_F;
    }
    __syncthreads();

    // Single thread per row does the sequential scan to maintain forward-fill state
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int order = orders[row];
    if (order <= 0) return;
    if (first_valid >= series_len) return;

    float last_min = CUDART_NAN_F;
    float last_max = CUDART_NAN_F;

    for (int i = max(0, first_valid); i < series_len; ++i) {
        float min_here = CUDART_NAN_F;
        float max_here = CUDART_NAN_F;

        const bool in_bounds = (i >= order) && (i + order < series_len);
        const float ch = high[i];
        const float cl = low[i];
        if (in_bounds && isfinite(ch) && isfinite(cl)) {
            // Check LOW neighbors on both sides are finite and strictly greater than center
            bool left_ok_low = true, right_ok_low = true;
            float lmin = CUDART_INF_F, rmin = CUDART_INF_F;
            for (int o = 1; o <= order; ++o) {
                const float ll = low[i - o];
                const float rl = low[i + o];
                if (!isfinite(ll)) { left_ok_low = false; break; }
                if (!isfinite(rl)) { right_ok_low = false; break; }
                lmin = fminf(lmin, ll);
                rmin = fminf(rmin, rl);
            }
            if (left_ok_low && right_ok_low && cl < lmin && cl < rmin) {
                min_here = cl;
            }

            // Check HIGH neighbors on both sides are finite and strictly less than center
            bool left_ok_high = true, right_ok_high = true;
            float lmax = -CUDART_INF_F, rmax = -CUDART_INF_F;
            for (int o = 1; o <= order; ++o) {
                const float lh = high[i - o];
                const float rh = high[i + o];
                if (!isfinite(lh)) { left_ok_high = false; break; }
                if (!isfinite(rh)) { right_ok_high = false; break; }
                lmax = fmaxf(lmax, lh);
                rmax = fmaxf(rmax, rh);
            }
            if (left_ok_high && right_ok_high && ch > lmax && ch > rmax) {
                max_here = ch;
            }
        }

        out_is_min[base + i] = min_here;
        out_is_max[base + i] = max_here;
        if (isfinite(min_here)) { last_min = min_here; }
        if (isfinite(max_here)) { last_max = max_here; }
        out_last_min[base + i] = last_min;
        out_last_max[base + i] = last_max;
    }
}

// Many-series × one-param (time-major): [t][series]
extern "C" __global__
void minmax_many_series_one_param_time_major_f32(const float* __restrict__ high_tm,
                                                 const float* __restrict__ low_tm,
                                                 const int* __restrict__ first_valids,
                                                 int num_series,
                                                 int series_len,
                                                 int order,
                                                 float* __restrict__ out_is_min_tm,
                                                 float* __restrict__ out_is_max_tm,
                                                 float* __restrict__ out_last_min_tm,
                                                 float* __restrict__ out_last_max_tm) {
    const int s = blockIdx.x; // one block per series
    if (s >= num_series || series_len <= 0 || order <= 0) return;
    const int stride = num_series;
    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];

    // init outputs to NaN for this series
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        const int idx = t * stride + s;
        out_is_min_tm[idx] = CUDART_NAN_F;
        out_is_max_tm[idx] = CUDART_NAN_F;
        out_last_min_tm[idx] = CUDART_NAN_F;
        out_last_max_tm[idx] = CUDART_NAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    if (fv >= series_len) return;

    float last_min = CUDART_NAN_F;
    float last_max = CUDART_NAN_F;

    for (int t = max(0, fv); t < series_len; ++t) {
        const int idx = t * stride + s;
        float min_here = CUDART_NAN_F;
        float max_here = CUDART_NAN_F;

        const bool in_bounds = (t >= order) && (t + order < series_len);
        const float ch = high_tm[idx];
        const float cl = low_tm[idx];
        if (in_bounds && isfinite(ch) && isfinite(cl)) {
            bool left_ok_low = true, right_ok_low = true;
            float lmin = CUDART_INF_F, rmin = CUDART_INF_F;
            for (int o = 1; o <= order; ++o) {
                const float ll = low_tm[(t - o) * stride + s];
                const float rl = low_tm[(t + o) * stride + s];
                if (!isfinite(ll)) { left_ok_low = false; break; }
                if (!isfinite(rl)) { right_ok_low = false; break; }
                lmin = fminf(lmin, ll);
                rmin = fminf(rmin, rl);
            }
            if (left_ok_low && right_ok_low && cl < lmin && cl < rmin) {
                min_here = cl;
            }

            bool left_ok_high = true, right_ok_high = true;
            float lmax = -CUDART_INF_F, rmax = -CUDART_INF_F;
            for (int o = 1; o <= order; ++o) {
                const float lh = high_tm[(t - o) * stride + s];
                const float rh = high_tm[(t + o) * stride + s];
                if (!isfinite(lh)) { left_ok_high = false; break; }
                if (!isfinite(rh)) { right_ok_high = false; break; }
                lmax = fmaxf(lmax, lh);
                rmax = fmaxf(rmax, rh);
            }
            if (left_ok_high && right_ok_high && ch > lmax && ch > rmax) {
                max_here = ch;
            }
        }

        out_is_min_tm[idx] = min_here;
        out_is_max_tm[idx] = max_here;
        if (isfinite(min_here)) { last_min = min_here; }
        if (isfinite(max_here)) { last_max = max_here; }
        out_last_min_tm[idx] = last_min;
        out_last_max_tm[idx] = last_max;
    }
}

