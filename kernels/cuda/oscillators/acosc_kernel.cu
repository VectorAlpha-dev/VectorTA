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
//   - change at the first valid output index (first_valid + 38) is computed as
//     res - 0.0 (i.e., not NaN) to match existing unit tests and CPU scalar.
//   - Single precision (f32) arithmetic with ring buffers for rolling sums.
//
// Implementation notes (performance/numerics):
//   - Rolling sums use Kahan-style compensation to reduce FP32 drift without FP64.
//   - Index wrap uses increment + compare instead of modulo.
//   - Prefill/warmup loops are fixed-trip and may be unrolled by nvcc.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ------------------------------
// Kahan helpers (FP32 only)
// ------------------------------
__device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
__device__ __forceinline__ void kahan_add_sub(float add, float sub, float &sum, float &c) {
    kahan_add(add, sum, c);
    kahan_add(-sub, sum, c);
}

// ------------------------------
// AO helpers for time-parallel path
// ------------------------------
static __device__ __forceinline__ float acosc_median(const float* __restrict__ high,
                                                     const float* __restrict__ low,
                                                     int idx) {
    return (high[idx] + low[idx]) * 0.5f;
}

static __device__ __forceinline__ float acosc_ao_at(const float* __restrict__ high,
                                                    const float* __restrict__ low,
                                                    int idx,
                                                    int first_valid) {
    float sum34 = 0.0f;
    float sum5 = 0.0f;
    const int rel = idx - first_valid;
#pragma unroll
    for (int k = 0; k < 34; ++k) {
        const float med = acosc_median(high, low, idx - k);
        sum34 += med;
        if (rel >= 38 && k < 5) sum5 += med;
    }
    if (rel < 38) {
        // Match the scalar warmup quirk: SMA5(median) is only updated starting at i=34, so the
        // first four AO samples (i=34..37) use a non-contiguous short window that still includes
        // medians from the very beginning of the valid slice.
        //
        // This affects ACOSC at i=38..41 because SMA5(AO) includes AO[34..37].
        if (rel == 34) {
            sum5 = acosc_median(high, low, first_valid + 1) +
                   acosc_median(high, low, first_valid + 2) +
                   acosc_median(high, low, first_valid + 3) +
                   acosc_median(high, low, first_valid + 4) +
                   acosc_median(high, low, idx);
        } else if (rel == 35) {
            sum5 = acosc_median(high, low, first_valid + 2) +
                   acosc_median(high, low, first_valid + 3) +
                   acosc_median(high, low, first_valid + 4) +
                   acosc_median(high, low, idx - 1) +
                   acosc_median(high, low, idx);
        } else if (rel == 36) {
            sum5 = acosc_median(high, low, first_valid + 3) +
                   acosc_median(high, low, first_valid + 4) +
                   acosc_median(high, low, idx - 2) +
                   acosc_median(high, low, idx - 1) +
                   acosc_median(high, low, idx);
        } else {
            // rel == 37 (the only remaining value we rely on)
            sum5 = acosc_median(high, low, first_valid + 4) +
                   acosc_median(high, low, idx - 3) +
                   acosc_median(high, low, idx - 2) +
                   acosc_median(high, low, idx - 1) +
                   acosc_median(high, low, idx);
        }
    }
    return sum5 * (1.0f / 5.0f) - sum34 * (1.0f / 34.0f);
}

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

    const int fv = first_valid < 0 ? 0 : first_valid;
    const int warm = fv + 34 + 5 - 1; // first_valid + 38
    const float nn = CUDART_NAN_F;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < series_len; i += stride) {
        if (i < warm) {
            out_osc[i] = nn;
            out_change[i] = nn;
            continue;
        }

        // Compute AO for i..i-5 so we can also compute the prior ACOSC in-thread.
        const float ao0 = acosc_ao_at(high, low, i, fv);
        const float ao1 = acosc_ao_at(high, low, i - 1, fv);
        const float ao2 = acosc_ao_at(high, low, i - 2, fv);
        const float ao3 = acosc_ao_at(high, low, i - 3, fv);
        const float ao4 = acosc_ao_at(high, low, i - 4, fv);

        const float sma5_ao = (ao0 + ao1 + ao2 + ao3 + ao4) * (1.0f / 5.0f);
        const float res = ao0 - sma5_ao;

        float prev_res = 0.0f;
        if (i != warm) {
            const float ao5 = acosc_ao_at(high, low, i - 5, fv);
            const float sma5_ao_prev = (ao1 + ao2 + ao3 + ao4 + ao5) * (1.0f / 5.0f);
            prev_res = ao1 - sma5_ao_prev;
        }

        out_osc[i] = res;
        out_change[i] = res - prev_res;
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
    float sum5 = 0.0f, c5 = 0.0f;
    float sum34 = 0.0f, c34 = 0.0f;
    float sum5ao = 0.0f, c5ao = 0.0f;
    int i5 = 0, i34 = 0, i5ao = 0;

    #pragma unroll
    for (int k = 0; k < P34; ++k) {
        const int t = fv + k;
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;
        kahan_add(med, sum34, c34);
        q34[k] = med;
        if (k < P5) {
            kahan_add(med, sum5, c5);
            q5[k] = med;
        }
    }

    for (int t = fv + P34; t < fv + P34 + P5 - 1; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;

        const float old34 = q34[i34];
        kahan_add_sub(med, old34, sum34, c34);
        q34[i34] = med;
        ++i34; if (i34 == P34) i34 = 0;
        const float sma34 = sum34 * INV34;

        const float old5 = q5[i5];
        kahan_add_sub(med, old5, sum5, c5);
        q5[i5] = med;
        ++i5; if (i5 == P5) i5 = 0;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        kahan_add(ao, sum5ao, c5ao);
        q5ao[i5ao] = ao;
        ++i5ao; if (i5ao == P5) i5ao = 0;
    }

    float prev_res = 0.0f;
    for (int t = fv + P34 + P5 - 1; t < series_len; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;

        const float old34 = q34[i34];
        kahan_add_sub(med, old34, sum34, c34);
        q34[i34] = med;
        ++i34; if (i34 == P34) i34 = 0;
        const float sma34 = sum34 * INV34;

        const float old5 = q5[i5];
        kahan_add_sub(med, old5, sum5, c5);
        q5[i5] = med;
        ++i5; if (i5 == P5) i5 = 0;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        const float old_ao = q5ao[i5ao];
        kahan_add_sub(ao, old_ao, sum5ao, c5ao);
        q5ao[i5ao] = ao;
        ++i5ao; if (i5ao == P5) i5ao = 0;

        const float sma5ao = sum5ao * INV5;
        const float res = ao - sma5ao;
        const float mom = res - prev_res; // keep change first output non-NaN
        prev_res = res;
        out_osc_tm[t * stride + s] = res;
        out_change_tm[t * stride + s] = mom;
    }
}

// ----------------------------------------
// OPTIONAL: Warp-striped many-series kernel (time-major, coalesced)
// One warp (32 threads) processes up to 32 series in lockstep.
// Shared-memory ring buffers laid out as [window][lane] to avoid bank conflicts.
// Signature matches the standard many-series kernel; wrappers may pick via heuristic.
// ----------------------------------------
extern "C" __global__
void acosc_many_series_one_param_f32_warp(const float* __restrict__ high_tm,
                                          const float* __restrict__ low_tm,
                                          const int* __restrict__ first_valids,
                                          int num_series,
                                          int series_len,
                                          float* __restrict__ out_osc_tm,
                                          float* __restrict__ out_change_tm) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_idx = blockIdx.x;        // one warp per block
    const int s = warp_idx * WARP_SIZE + lane;
    if (s >= num_series || series_len <= 0) return;

    const int stride = num_series;
    int fv = first_valids[s]; if (fv < 0) fv = 0;

    // NaN init: each lane initializes all time steps for its series.
    // Using the same t across the warp preserves coalescing across series.
    for (int t = 0; t < series_len; ++t) {
        out_osc_tm[t * stride + s] = CUDART_NAN_F;
        out_change_tm[t * stride + s] = CUDART_NAN_F;
    }

    if (fv >= series_len) return;
    const int P5 = 5;
    const int P34 = 34;
    if ((series_len - fv) < (P34 + P5)) return;

    const float INV5 = 1.0f / 5.0f;
    const float INV34 = 1.0f / 34.0f;

    extern __shared__ float smem[];
    float* q34  = smem;                    // P34 * WARP_SIZE
    float* q5   = q34  + P34 * WARP_SIZE;  // P5  * WARP_SIZE
    float* q5ao = q5   + P5  * WARP_SIZE;  // P5  * WARP_SIZE

    auto SM = [&](int k) { return k * WARP_SIZE + lane; };

    float sum34 = 0.0f, c34 = 0.0f;
    float sum5  = 0.0f, c5  = 0.0f;
    float sum5ao= 0.0f, c5ao= 0.0f;
    int i34 = 0, i5 = 0, i5ao = 0;

    // Prefill
    #pragma unroll
    for (int k = 0; k < P34; ++k) {
        const int t = fv + k;
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;
        kahan_add(med, sum34, c34);
        q34[SM(k)] = med;
        if (k < P5) {
            kahan_add(med, sum5, c5);
            q5[SM(k)] = med;
        }
    }

    // Warm phase (4 steps)
    for (int t = fv + P34; t < fv + P34 + P5 - 1; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;

        const float old34 = q34[SM(i34)];
        kahan_add_sub(med, old34, sum34, c34);
        q34[SM(i34)] = med;
        ++i34; if (i34 == P34) i34 = 0;
        const float sma34 = sum34 * INV34;

        const float old5 = q5[SM(i5)];
        kahan_add_sub(med, old5, sum5, c5);
        q5[SM(i5)] = med;
        ++i5; if (i5 == P5) i5 = 0;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        kahan_add(ao, sum5ao, c5ao);
        q5ao[SM(i5ao)] = ao;
        ++i5ao; if (i5ao == P5) i5ao = 0;
    }

    float prev_res = 0.0f;
    for (int t = fv + P34 + P5 - 1; t < series_len; ++t) {
        const float med = (high_tm[t * stride + s] + low_tm[t * stride + s]) * 0.5f;

        const float old34 = q34[SM(i34)];
        kahan_add_sub(med, old34, sum34, c34);
        q34[SM(i34)] = med;
        ++i34; if (i34 == P34) i34 = 0;
        const float sma34 = sum34 * INV34;

        const float old5 = q5[SM(i5)];
        kahan_add_sub(med, old5, sum5, c5);
        q5[SM(i5)] = med;
        ++i5; if (i5 == P5) i5 = 0;
        const float sma5 = sum5 * INV5;

        const float ao = sma5 - sma34;
        const float old_ao = q5ao[SM(i5ao)];
        kahan_add_sub(ao, old_ao, sum5ao, c5ao);
        q5ao[SM(i5ao)] = ao;
        ++i5ao; if (i5ao == P5) i5ao = 0;

        const float sma5ao = sum5ao * INV5;
        const float res = ao - sma5ao;
        const float mom = res - prev_res; // match scalar semantics
        prev_res = res;
        out_osc_tm[t * stride + s] = res;
        out_change_tm[t * stride + s] = mom;
    }
}
