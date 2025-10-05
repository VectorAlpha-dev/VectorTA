// CUDA kernels for the Midway Weighted Exponential (MWDX) indicator.
//
// Category: Recurrence/IIR (EMA-like). Each parameter combo or series is
// processed sequentially in time by a single thread (lane 0) to preserve the
// exact scalar semantics. Other threads in the CTA help initialize outputs to
// NaN before the warm region. This mirrors ALMA's public CUDA API surface for
// batch and many-series entry points while keeping the math strictly
// sequential where required.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Batch: one price series × many parameters (grid.y = combos)
extern "C" __global__
void mwdx_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ facs,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.y; // combos live on grid.y (ALMA convention)
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    const float fac = facs[combo];
    const float beta = 1.0f - fac;
    const int row_offset = combo * series_len;

    // Initialize to NaN prior to warmup. Threads cooperate across the row.
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0 || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    float prev = prices[first_valid];
    out[row_offset + first_valid] = prev;

    for (int t = first_valid + 1; t < series_len; ++t) {
        const float price = prices[t];
        prev = __fmaf_rn(price, fac, beta * prev);
        out[row_offset + t] = prev;
    }
}

extern "C" __global__
void mwdx_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    float fac,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) {
        return;
    }

    const float beta = 1.0f - fac;
    const int stride = num_series;
    const int first_valid = first_valids[series_idx];

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0 || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    int offset = first_valid * stride + series_idx;
    float prev = prices_tm[offset];
    out_tm[offset] = prev;

    for (int t = first_valid + 1; t < series_len; ++t) {
        offset = t * stride + series_idx;
        const float price = prices_tm[offset];
        prev = __fmaf_rn(price, fac, beta * prev);
        out_tm[offset] = prev;
    }
}

// Many-series, one-parameter: 2D tiled variant (time-major)
// Threads only help with NaN initialization; lane 0 of each series does the
// sequential recurrence. Provided for launch-geometry symmetry with ALMA.
template<int TX, int TY>
__device__ void mwdx_many_series_one_param_tiled2d_f32_core(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    const int s_base = blockIdx.y * TY;
    const int s_local = s_base + threadIdx.y;
    if (s_local >= num_series || series_len <= 0) return;

    const float beta = 1.0f - fac;
    const int stride = num_series;
    const int first_valid = first_valids[s_local];

    // Initialize NaNs across the entire time range for this series row.
    // Threads cooperate by striding over time with TX.
    for (int t = threadIdx.x; t < series_len; t += TX) {
        int out_idx = t * stride + s_local;
        out_tm[out_idx] = NAN;
    }
    __syncthreads();

    // Sequential scan for this series handled by lane x==0
    if (threadIdx.x == 0) {
        if (first_valid >= 0 && first_valid < series_len) {
            int off0 = first_valid * stride + s_local;
            float prev = prices_tm[off0];
            out_tm[off0] = prev;
            for (int t = first_valid + 1; t < series_len; ++t) {
                int off = t * stride + s_local;
                float price = prices_tm[off];
                prev = __fmaf_rn(price, fac, beta * prev);
                out_tm[off] = prev;
            }
        }
    }
}

extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f32_tx128_ty2(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f32_core<128, 2>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}

extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f32_tx128_ty4(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f32_core<128, 4>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}
