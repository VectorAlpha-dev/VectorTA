// CUDA kernels for Market Facilitation Index (marketefi)
//
// Formula per time step:
//   marketefi[t] = (high[t] - low[t]) / volume[t]
// Semantics:
//   - Write NaN for indices before `first_valid`.
//   - For t >= first_valid: if any input is NaN or volume == 0, write NaN.
//   - Otherwise write the ratio in FP32.

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void marketefi_kernel_f32(const float* __restrict__ high,
                                                 const float* __restrict__ low,
                                                 const float* __restrict__ volume,
                                                 int len,
                                                 int first_valid,
                                                 float* __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (len <= 0) return;
    const int first = first_valid < 0 ? 0 : first_valid;

    for (int i = tid; i < len; i += stride) {
        if (i < first) {
            out[i] = CUDART_NAN_F;
            continue;
        }

        const float h = high[i];
        const float l = low[i];
        const float v = volume[i];
        if (isnan(h) || isnan(l) || isnan(v) || v == 0.0f) {
            out[i] = CUDART_NAN_F;
        } else {
            out[i] = (h - l) / v;
        }
    }
}

// Many-series × one-param (time-major). Paramless indicator; we still separate entry
// for API parity. Each block handles one series; threads sweep the time dimension.
extern "C" __global__ void marketefi_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ volume_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x; // series index
    if (s >= num_series || series_len <= 0) return;
    const int stride_series = num_series; // time-major layout
    const int first = first_valids ? max(0, first_valids[s]) : 0;

    // Prefix NaNs up to first
    for (int t = threadIdx.x; t < min(first, series_len); t += blockDim.x) {
        out_tm[t * stride_series + s] = CUDART_NAN_F;
    }

    // Each thread processes every blockDim.x-th time index starting at its lane
    for (int t = threadIdx.x + first; t < series_len; t += blockDim.x) {
        const float h = high_tm[t * stride_series + s];
        const float l = low_tm[t * stride_series + s];
        const float v = volume_tm[t * stride_series + s];
        float outv;
        if (isnan(h) || isnan(l) || isnan(v) || v == 0.0f) {
            outv = CUDART_NAN_F;
        } else {
            outv = (h - l) / v;
        }
        out_tm[t * stride_series + s] = outv;
    }
}

