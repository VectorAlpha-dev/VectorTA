// CUDA kernel for MEDPRICE (Median Price).
//
// Each thread processes a subset of the price series using a grid-stride loop.
// The kernel expects FP32 inputs and writes FP32 outputs, mirroring the
// (high + low) * 0.5 calculation performed by the scalar Rust implementation.
// NaN handling matches the CPU path: any NaN in the inputs or indices before
// the first valid element produce NaN in the output.

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void medprice_kernel_f32(const float* __restrict__ high,
                                               const float* __restrict__ low,
                                               int len,
                                               int first_valid,
                                               float* __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (len <= 0) return;

    const int clamped_first = first_valid < 0 ? 0 : first_valid;
    for (int i = tid; i < len; i += stride) {
        if (i < clamped_first) {
            out[i] = CUDART_NAN_F;
            continue;
        }
        const float h = high[i];
        const float l = low[i];
        if (isnan(h) || isnan(l)) {
            out[i] = CUDART_NAN_F;
        } else {
            out[i] = 0.5f * (h + l);
        }
    }
}

// Batch: one series × many params (medprice has no params; rows=1 for parity)
extern "C" __global__ void medprice_batch_f32(const float* __restrict__ high,
                                              const float* __restrict__ low,
                                              int len,
                                              int rows,
                                              const int* __restrict__ first_valids,
                                              float* __restrict__ out) {
    const int row = blockIdx.y; // 0..rows-1
    if (row >= rows) return;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int fv = first_valids ? max(first_valids[row], 0) : 0;
    const int base = row * len;
    for (int i = tid; i < len; i += stride) {
        if (i < fv) { out[base + i] = CUDART_NAN_F; continue; }
        const float h = high[i];
        const float l = low[i];
        out[base + i] = (isnan(h) || isnan(l)) ? CUDART_NAN_F : 0.5f * (h + l);
    }
}

// Many-series × one-param (time-major layout)
// Input/output are time-major: index (t, s) -> t * cols + s
extern "C" __global__ void medprice_many_series_one_param_f32(const float* __restrict__ high_tm,
                                                              const float* __restrict__ low_tm,
                                                              int cols,
                                                              int rows,
                                                              const int* __restrict__ first_valids,
                                                              float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;
    const int fv = first_valids ? max(first_valids[s], 0) : 0;
    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;
        if (t < fv) { out_tm[idx] = CUDART_NAN_F; continue; }
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        out_tm[idx] = (isnan(h) || isnan(l)) ? CUDART_NAN_F : 0.5f * (h + l);
    }
}
