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
#include <math.h>

#ifndef ROCR_NAN
#define ROCR_NAN (__int_as_float(0x7fffffff))
#endif

// ---------------- One-series × many-params (plain) -------------------------
// If inv is non-null, it should contain inv[j] = 0.0f when data[j] == 0 or NaN,
// otherwise inv[j] = 1.0f / data[j]. We still guard with (inv==0) to enforce the
// scalar 0.0 policy even if current is NaN.
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

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float v = ROCR_NAN;
        if (t >= warm) {
            const int d_idx = t - period;
            if (inv_opt) {
                const float inv = inv_opt[d_idx];
                if (inv == 0.0f || isnan(inv)) {
                    v = 0.0f;
                } else {
                    v = data[t] * inv;
                }
            } else {
                const float denom = data[d_idx];
                if (denom == 0.0f || isnan(denom)) {
                    v = 0.0f;
                } else {
                    v = data[t] / denom;
                }
            }
        }
        out[row_off + t] = v;
        t += stride;
    }
}

// ---------------- Many-series × one-param (time-major) --------------------
// data_tm: rows x cols, time-major. first_valids: [cols]
extern "C" __global__ void rocr_many_series_one_param_f32(
    const float* __restrict__ data_tm,  // rows x cols (time-major)
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm          // rows x cols (time-major)
) {
    const int series = blockIdx.y;
    if (series >= num_series || period <= 0) return;

    const int stride = num_series; // time-major stride
    const int warm = first_valids[series] + period;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + series;
        if (t < warm) {
            out_tm[out_idx] = ROCR_NAN;
        } else {
            const float denom = data_tm[(t - period) * stride + series];
            if (denom == 0.0f || isnan(denom)) {
                out_tm[out_idx] = 0.0f;
            } else {
                const float curr = data_tm[t * stride + series];
                out_tm[out_idx] = curr / denom;
            }
        }
        t += step;
    }
}

