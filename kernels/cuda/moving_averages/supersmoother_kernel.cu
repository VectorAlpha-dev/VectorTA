// CUDA kernels for the SuperSmoother filter.
//
// Each batch kernel thread evaluates one parameter combination sequentially.
// The many-series variant operates on time-major input where columns represent
// different price series sharing a single parameter set.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef SUPERSMOOTHER_NAN
#define SUPERSMOOTHER_NAN (__int_as_float(0x7fffffff))
#endif

static __device__ __forceinline__ void supersmoother_coeffs(float period, float* a, float* b, float* c) {
    const float PI = 3.14159265358979323846f;
    const float SQRT2 = 1.41421356237f;
    const float denom = period;
    const float factor = SQRT2 * PI / denom;
    const float a_val = expf(-factor);
    const float a_sq = a_val * a_val;
    const float b_val = 2.0f * a_val * cosf(factor);
    const float c_val = (1.0f + a_sq - b_val) * 0.5f;
    *a = a_val;
    *b = b_val;
    *c = c_val;
}

extern "C" __global__ void supersmoother_batch_f32(const float* __restrict__ prices,
                                                    const int* __restrict__ periods,
                                                    int series_len,
                                                    int n_combos,
                                                    int first_valid,
                                                    float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int base = combo * series_len;
    float* __restrict__ row_out = out + base;

    for (int idx = 0; idx < series_len; ++idx) {
        row_out[idx] = SUPERSMOOTHER_NAN;
    }

    if (period <= 0 || period > series_len) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        return;
    }

    const int warm = first_valid + period - 1;
    if (warm >= series_len) {
        return;
    }

    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    supersmoother_coeffs((float)period, &a, &b, &c);
    const float a_sq = a * a;

    row_out[warm] = prices[warm];
    if (warm + 1 < series_len) {
        row_out[warm + 1] = prices[warm + 1];
    }

    for (int idx = warm + 2; idx < series_len; ++idx) {
        const float prev1 = row_out[idx - 1];
        const float prev2 = row_out[idx - 2];
        const float cur = prices[idx];
        const float prev_price = prices[idx - 1];
        row_out[idx] = c * (cur + prev_price) + b * prev1 - a_sq * prev2;
    }
}

extern "C" __global__ void supersmoother_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    const int stride = num_series;
    auto idx_tm = [stride, series](int row) { return row * stride + series; };

    for (int row = 0; row < series_len; ++row) {
        out_tm[idx_tm(row)] = SUPERSMOOTHER_NAN;
    }

    if (period <= 0 || period > series_len) {
        return;
    }

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        return;
    }

    const int warm = first_valid + period - 1;
    if (warm >= series_len) {
        return;
    }

    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    supersmoother_coeffs((float)period, &a, &b, &c);
    const float a_sq = a * a;

    out_tm[idx_tm(warm)] = prices_tm[idx_tm(warm)];
    if (warm + 1 < series_len) {
        out_tm[idx_tm(warm + 1)] = prices_tm[idx_tm(warm + 1)];
    }

    for (int row = warm + 2; row < series_len; ++row) {
        const float prev1 = out_tm[idx_tm(row - 1)];
        const float prev2 = out_tm[idx_tm(row - 2)];
        const float cur = prices_tm[idx_tm(row)];
        const float prev_price = prices_tm[idx_tm(row - 1)];
        out_tm[idx_tm(row)] = c * (cur + prev_price) + b * prev1 - a_sq * prev2;
    }
}
