// CUDA kernels for ROCR (Rate of Change Ratio): out[t] = data[t] / data[t-period]
//
// Semantics (must match scalar Rust implementation):
// - Warmup prefix length = first_valid + period. Indices before warmup are NaN.
// - For t >= warmup:
//     - If denominator (data[t-period]) is 0.0 or NaN => write 0.0
//     - Else write data[t] / data[t-period] (propagates NaN from numerator)
//
// Batch kernel supports one series × many params (grid.y = combos).
// Many-series kernel consumes time-major input with a single period.

#include <cuda_runtime.h>

#ifndef ROCR_NAN
#define ROCR_NAN (__int_as_float(0x7fffffff))  // quiet NaN bit-pattern
#endif

// Inline NaN predicate via IEEE-754 property to avoid function call overhead.
static __device__ __forceinline__ bool rocr_isnan(float x) { return x != x; }

// ---------------- OPTIONAL: precompute 1/x once -----------------------------
// inv_out[j] = 0.0f when data[j] == 0 or NaN, else inv_out[j] = 1.0f / data[j].
extern "C" __global__ void rocr_prepare_inv_f32(
    const float* __restrict__ data,  // [len]
    int len,
    float* __restrict__ inv_out      // [len]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < len) {
        float x = data[i];
        inv_out[i] = (x == 0.0f || rocr_isnan(x)) ? 0.0f : (1.0f / x);
        i += stride;
    }
}

// ---------------- One-series × many-params (optimized) ----------------------
// If inv_opt is non-null, inv_opt[j] must be as above (0 or 1/x).
extern "C" __global__ void rocr_batch_f32(
    const float* __restrict__ data,     // [len]
    const float* __restrict__ inv_opt,  // [len] or nullptr
    int len,
    int first_valid,
    const int* __restrict__ periods,    // [n_combos]
    int n_combos,
    float* __restrict__ out             // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period;
    const int row_off = combo * len;

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Phase 1: prefix region -> NaN (branch-free)
    for (int i = tid; i < warm && i < len; i += stride) {
        out[row_off + i] = ROCR_NAN;
    }

    // Advance starting index to the first position this thread handles >= warm
    int start = tid;
    if (start < warm) {
        int delta = warm - start;
        int steps = (delta + stride - 1) / stride;  // once per thread
        start += steps * stride;
    }

    // Phase 2: compute region [warm, len)
    if (inv_opt) {
        int i = start;
        const int stride4 = stride << 2;
        const int end4 = len - 3 * stride;
        for (; i < end4; i += stride4) {
            // step 0
            {
                const int d = i - period;
                const float inv = inv_opt[d];
                out[row_off + i] = (inv == 0.0f || rocr_isnan(inv)) ? 0.0f : (data[i] * inv);
            }
            // step 1
            {
                const int i1 = i + stride;
                const int d1 = i1 - period;
                const float inv1 = inv_opt[d1];
                out[row_off + i1] = (inv1 == 0.0f || rocr_isnan(inv1)) ? 0.0f : (data[i1] * inv1);
            }
            // step 2
            {
                const int i2 = i + 2 * stride;
                const int d2 = i2 - period;
                const float inv2 = inv_opt[d2];
                out[row_off + i2] = (inv2 == 0.0f || rocr_isnan(inv2)) ? 0.0f : (data[i2] * inv2);
            }
            // step 3
            {
                const int i3 = i + 3 * stride;
                const int d3 = i3 - period;
                const float inv3 = inv_opt[d3];
                out[row_off + i3] = (inv3 == 0.0f || rocr_isnan(inv3)) ? 0.0f : (data[i3] * inv3);
            }
        }
        for (; i < len; i += stride) {
            const int d = i - period;
            const float inv = inv_opt[d];
            out[row_off + i] = (inv == 0.0f || rocr_isnan(inv)) ? 0.0f : (data[i] * inv);
        }
    } else {
        int i = start;
        const int stride4 = stride << 2;
        const int end4 = len - 3 * stride;
        for (; i < end4; i += stride4) {
            // step 0
            {
                const int d = i - period;
                const float denom = data[d];
                out[row_off + i] = (denom == 0.0f || rocr_isnan(denom)) ? 0.0f : (data[i] / denom);
            }
            // step 1
            {
                const int i1 = i + stride;
                const int d1 = i1 - period;
                const float denom1 = data[d1];
                out[row_off + i1] = (denom1 == 0.0f || rocr_isnan(denom1)) ? 0.0f : (data[i1] / denom1);
            }
            // step 2
            {
                const int i2 = i + 2 * stride;
                const int d2 = i2 - period;
                const float denom2 = data[d2];
                out[row_off + i2] = (denom2 == 0.0f || rocr_isnan(denom2)) ? 0.0f : (data[i2] / denom2);
            }
            // step 3
            {
                const int i3 = i + 3 * stride;
                const int d3 = i3 - period;
                const float denom3 = data[d3];
                out[row_off + i3] = (denom3 == 0.0f || rocr_isnan(denom3)) ? 0.0f : (data[i3] / denom3);
            }
        }
        for (; i < len; i += stride) {
            const int d = i - period;
            const float denom = data[d];
            out[row_off + i] = (denom == 0.0f || rocr_isnan(denom)) ? 0.0f : (data[i] / denom);
        }
    }
}

// ---------------- Many-series × one-param (time-major, coalesced-ready) -----
// This kernel supports both the existing 1D launch (block=(Bx,1,1), grid=(Gx,cols,1))
// and an optional 2D launch (block=(Sx,Ty,1), grid=(Gseries,Gtime,1)).
// - 1D compat path: threads iterate over time (t) for a fixed series (blockIdx.y).
// - 2D optimized path: threads vary across series (x) for coalesced accesses at fixed time (y).
extern "C" __global__ void rocr_many_series_one_param_f32(
    const float* __restrict__ data_tm,  // rows x cols (time-major)
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,  // [cols]
    float* __restrict__ out_tm             // rows x cols (time-major)
) {
    if (period <= 0) return;

    const int stride_series = num_series; // time-major stride

    if (blockDim.y == 1 && gridDim.y == (unsigned)num_series) {
        // 1D compatibility path: original mapping (series = blockIdx.y; t along x)
        const int series = blockIdx.y;
        if (series >= num_series) return;
        const int warm = first_valids[series] + period;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int step = gridDim.x * blockDim.x;

        // Phase 1: prefix -> NaN
        for (int tt = t; tt < series_len && tt < warm; tt += step) {
            out_tm[tt * stride_series + series] = ROCR_NAN;
        }
        // Advance start to >= warm
        int t_start = t;
        if (t_start < warm) {
            int delta = warm - t_start;
            int steps = (delta + step - 1) / step;
            t_start += steps * step;
        }
        for (int tt = t_start; tt < series_len; tt += step) {
            const float denom = data_tm[(tt - period) * stride_series + series];
            out_tm[tt * stride_series + series] =
                (denom == 0.0f || rocr_isnan(denom)) ? 0.0f : (data_tm[tt * stride_series + series] / denom);
        }
        return;
    }

    // 2D optimized path: threads span series (x) and time (y)
    const int s0 = blockIdx.x * blockDim.x + threadIdx.x; // series index
    const int t0 = blockIdx.y * blockDim.y + threadIdx.y; // time index
    const int s_step = blockDim.x * gridDim.x;
    const int t_step = blockDim.y * gridDim.y;

    for (int s = s0; s < num_series; s += s_step) {
        const int warm = first_valids[s] + period;
        // Phase 1: set NaNs for prefix per-series
        for (int t = t0; t < series_len && t < warm; t += t_step) {
            out_tm[t * stride_series + s] = ROCR_NAN;
        }
        // Phase 2: compute per-series
        int t_start = t0;
        if (t_start < warm) {
            int delta = warm - t_start;
            int steps = (delta + t_step - 1) / t_step; // once per thread
            t_start += steps * t_step;
        }
        for (int t = t_start; t < series_len; t += t_step) {
            const float denom = data_tm[(t - period) * stride_series + s];
            out_tm[t * stride_series + s] =
                (denom == 0.0f || rocr_isnan(denom)) ? 0.0f : (data_tm[t * stride_series + s] / denom);
        }
    }
}

