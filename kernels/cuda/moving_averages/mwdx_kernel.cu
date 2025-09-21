// CUDA kernels for the Midway Weighted Exponential (MWDX) indicator.
//
// The batch variant evaluates a single price series against multiple factor
// combinations. Each block owns one combination because the recurrence has a
// strict data dependency on the previous output value. Threads collaborate to
// initialise the output row, and lane 0 performs the sequential smoothing to
// mirror the scalar CPU implementation exactly. A second kernel covers the
// time-major "many series × one factor" entry point that the Rust wrapper and
// Python bindings expose.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void mwdx_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ facs,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    const float fac = facs[combo];
    const float beta = 1.0f - fac;
    const int row_offset = combo * series_len;

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
