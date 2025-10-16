// CUDA kernels for Chaikin's Volatility (CVI)
//
// CVI is defined as 100 * (EMA(H-L)_t - EMA(H-L)_{t-p}) / EMA(H-L)_{t-p}
// Warmup semantics (matching scalar): first valid output at index
//   warm = first_valid + (2*period - 1)
// Values prior to warm are NaN.
//
// Implementation notes:
// - Per-combo sequential time scan (EMA recurrence) per block.
// - Two-pass approach: first compute EMA(H-L) across time into the output row,
//   then convert in-place to CVI by differencing with the EMA lag of `period`.
// - FP32 with FMA for update; outputs are FP32.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void cvi_batch_f32(const float* __restrict__ high,
                   const float* __restrict__ low,
                   const int*   __restrict__ periods,
                   const float* __restrict__ alphas,
                   const int*   __restrict__ warm_indices,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int   period = periods[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || first_valid < 0 || warm >= series_len) return;

    const int base = combo * series_len;

    // Initialize entire row to NaN cooperatively
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    // First pass: compute EMA(H-L) into out[base + t]
    float y = high[first_valid] - low[first_valid];
    out[base + first_valid] = y;
    for (int t = first_valid + 1; t < series_len; ++t) {
        const float r = high[t] - low[t];
        // y += (r - y) * alpha (fused)
        y = __fmaf_rn(r - y, alpha, y);
        out[base + t] = y;
    }

    // Second pass: convert EMA to CVI in-place
    for (int t = warm; t < series_len; ++t) {
        const float curr = out[base + t];
        const float old  = out[base + (t - period)];
        out[base + t] = 100.0f * (curr - old) / old;
    }
}

// Variant that takes precomputed range = high - low.
extern "C" __global__
void cvi_batch_from_range_f32(const float* __restrict__ range,
                              const int*   __restrict__ periods,
                              const float* __restrict__ alphas,
                              const int*   __restrict__ warm_indices,
                              int series_len,
                              int first_valid,
                              int n_combos,
                              float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int   period = periods[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || first_valid < 0 || warm >= series_len) return;

    const int base = combo * series_len;

    // Initialize entire row to NaN cooperatively
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    float y = range[first_valid];
    out[base + first_valid] = y;
    for (int t = first_valid + 1; t < series_len; ++t) {
        const float r = range[t];
        y = __fmaf_rn(r - y, alpha, y);
        out[base + t] = y;
    }
    for (int t = warm; t < series_len; ++t) {
        const float curr = out[base + t];
        const float old  = out[base + (t - period)];
        out[base + t] = 100.0f * (curr - old) / old;
    }
}

// Many-series × one-param (time-major)
// Inputs: high_tm[t * num_series + s], low_tm[t * num_series + s]
extern "C" __global__
void cvi_many_series_one_param_f32(const float* __restrict__ high_tm,
                                   const float* __restrict__ low_tm,
                                   const int*   __restrict__ first_valids,
                                   int period,
                                   float alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    const int stride = num_series;
    const int fv = first_valids[s];

    // Clear column to NaN
    for (int t = 0; t < series_len; ++t) {
        out_tm[t * stride + s] = NAN;
    }
    if (fv < 0 || fv >= series_len) return;

    const int warm = fv + (2 * period - 1);
    if (warm >= series_len) return;

    // First pass: EMA into out_tm
    float y = high_tm[fv * stride + s] - low_tm[fv * stride + s];
    out_tm[fv * stride + s] = y;
    for (int t = fv + 1; t < series_len; ++t) {
        const float r = high_tm[t * stride + s] - low_tm[t * stride + s];
        y = __fmaf_rn(r - y, alpha, y);
        out_tm[t * stride + s] = y;
    }

    // Second pass: convert to CVI in-place
    for (int t = warm; t < series_len; ++t) {
        const float curr = out_tm[t * stride + s];
        const float old  = out_tm[(t - period) * stride + s];
        out_tm[t * stride + s] = 100.0f * (curr - old) / old;
    }
}

