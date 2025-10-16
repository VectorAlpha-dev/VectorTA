// CUDA kernels for Accelerator Oscillator (ACOSC).
//
// ACOSC uses fixed periods (5 and 34). It computes:
//   median = (high + low)/2
//   AO     = SMA5(median) - SMA34(median)
//   ACOSC  = AO - SMA5(AO)
// and the momentum/change = diff(ACOSC).
//
// Semantics match the scalar Rust implementation:
//   - Warmup prefix is NaN up to index (first_valid + 38).
//   - NaNs in inputs are propagated by skipping output writes before warmup and
//     by writing NaN for any index whose med or AO cannot be formed.
//   - Single precision (f32) arithmetic with ring buffers for rolling sums.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// One-series × one-(fixed)-param batch entry. We keep the name *_batch_f32 to
// stay consistent with other indicators even though n_combos == 1.
extern "C" __global__
void acosc_batch_f32(const float* __restrict__ high,
                     const float* __restrict__ low,
                     int series_len,
                     int first_valid,
                     float* __restrict__ out_osc,
                     float* __restrict__ out_change) {
    if (series_len <= 0) return;

    // Initialize outputs to NaN; cheap and avoids uninitialized tails.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += blockDim.x * gridDim.x) {
        out_osc[i] = CUDART_NAN_F;
        out_change[i] = CUDART_NAN_F;
    }
    __syncthreads();

    // Single thread executes sequential scan (rolling sums).
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int clamp_first = first_valid < 0 ? 0 : first_valid;
    if (clamp_first >= series_len) return;

    // Constants and ring buffers.
    const int P5 = 5;
    const int P34 = 34;
    const float INV5 = 1.0f / 5.0f;
    const float INV34 = 1.0f / 34.0f;
    float q5[5];
    float q34[34];
    float q5ao[5];
    float sum5 = 0.0f, sum34 = 0.0f, sum5ao = 0.0f;
    int i5 = 0, i34 = 0, i5ao = 0;

    // Prefill 34 and first 5 with medians.
    int end34 = clamp_first + P34;
    if (end34 > series_len) end34 = series_len;
    // Not enough to ever produce output.
    if ((series_len - clamp_first) < (P34 + P5)) {
        return;
    }

    for (int i = clamp_first; i < clamp_first + P34; ++i) {
        const float h = high[i];
        const float l = low[i];
        const float med = (h + l) * 0.5f;
        sum34 += med;
        q34[i - clamp_first] = med;
        if (i - clamp_first < P5) {
            sum5 += med;
            q5[i - clamp_first] = med;
        }
    }
    // Warm phase to accumulate AO's SMA5 (indices clamp_first+34 .. clamp_first+38 exclusive of last)
    for (int i = clamp_first + P34; i < clamp_first + P34 + P5 - 1; ++i) {
        const float h = high[i];
        const float l = low[i];
        const float med = (h + l) * 0.5f;
        sum34 += med - q34[i34];
        q34[i34] = med;
        i34 = (i34 + 1) % P34;
        const float sma34 = sum34 * INV34;

        sum5 += med - q5[i5];
        q5[i5] = med;
        i5 = (i5 + 1) % P5;
        const float sma5 = sum5 * INV5;
        const float ao = sma5 - sma34;
        sum5ao += ao;
        q5ao[i5ao] = ao;
        ++i5ao;
    }
    if (i5ao == P5) i5ao = 0;

    float prev_res = 0.0f;
    // Main outputs begin at warm index (first_valid + 38)
    for (int i = clamp_first + P34 + P5 - 1; i < series_len; ++i) {
        const float h = high[i];
        const float l = low[i];
        const float med = (h + l) * 0.5f;
        sum34 += med - q34[i34];
        q34[i34] = med;
        i34 = (i34 + 1) % P34;
        const float sma34 = sum34 * INV34;

        sum5 += med - q5[i5];
        q5[i5] = med;
        i5 = (i5 + 1) % P5;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        const float old_ao = q5ao[i5ao];
        sum5ao += ao - old_ao;
        q5ao[i5ao] = ao;
        i5ao = (i5ao + 1) % P5;

        const float sma5ao = sum5ao * INV5;
        const float res = ao - sma5ao;
        const float mom = res - prev_res;
        prev_res = res;
        out_osc[i] = res;
        out_change[i] = mom;
    }
}

// Many-series × one-(fixed)-param, time-major layout: [t][series]
extern "C" __global__
void acosc_many_series_one_param_f32(const float* __restrict__ high_tm,
                                     const float* __restrict__ low_tm,
                                     const int* __restrict__ first_valids,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_osc_tm,
                                     float* __restrict__ out_change_tm) {
    const int s = blockIdx.x; // one block per series
    if (s >= num_series || series_len <= 0) return;

    const int stride = num_series; // time-major
    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];

    // Init outputs for this series to NaN in parallel
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_osc_tm[t * stride + s] = CUDART_NAN_F;
        out_change_tm[t * stride + s] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (fv >= series_len) return;
    if ((series_len - fv) < 39) return; // 34 + 5

    const int P5 = 5;
    const int P34 = 34;
    const float INV5 = 1.0f / 5.0f;
    const float INV34 = 1.0f / 34.0f;
    float q5[5];
    float q34[34];
    float q5ao[5];
    float sum5 = 0.0f, sum34 = 0.0f, sum5ao = 0.0f;
    int i5 = 0, i34 = 0, i5ao = 0;

    for (int t = fv; t < fv + P34; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;
        sum34 += med;
        q34[t - fv] = med;
        if (t - fv < P5) {
            sum5 += med;
            q5[t - fv] = med;
        }
    }

    for (int t = fv + P34; t < fv + P34 + P5 - 1; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;
        sum34 += med - q34[i34];
        q34[i34] = med;
        i34 = (i34 + 1) % P34;
        const float sma34 = sum34 * INV34;

        sum5 += med - q5[i5];
        q5[i5] = med;
        i5 = (i5 + 1) % P5;
        const float sma5 = sum5 * INV5;
        const float ao = sma5 - sma34;
        sum5ao += ao;
        q5ao[i5ao] = ao;
        ++i5ao;
    }
    if (i5ao == P5) i5ao = 0;

    float prev_res = 0.0f;
    for (int t = fv + P34 + P5 - 1; t < series_len; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;
        sum34 += med - q34[i34];
        q34[i34] = med;
        i34 = (i34 + 1) % P34;
        const float sma34 = sum34 * INV34;

        sum5 += med - q5[i5];
        q5[i5] = med;
        i5 = (i5 + 1) % P5;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        const float old_ao = q5ao[i5ao];
        sum5ao += ao - old_ao;
        q5ao[i5ao] = ao;
        i5ao = (i5ao + 1) % P5;

        const float sma5ao = sum5ao * INV5;
        const float res = ao - sma5ao;
        const float mom = res - prev_res;
        prev_res = res;
        out_osc_tm[t * stride + s] = res;
        out_change_tm[t * stride + s] = mom;
    }
}

