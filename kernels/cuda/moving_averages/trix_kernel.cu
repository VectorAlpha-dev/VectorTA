// CUDA kernels for TRIX (Triple Exponential Average Oscillator)
//
// Math: triple EMA of ln(price), output is delta of EMA3 scaled by 10000.
// Category: recurrence/IIR per parameter or per series (no large shared memory).
//
// Semantics (must match scalar):
// - Warmup index for a given period p and first valid index fv is
//   warmup_end = fv + 3*(p-1) + 1. Indices < warmup_end are filled with NaN.
// - The first finite TRIX sample is written at index == warmup_end.
// - We use FP32 on device; upstream wrappers build CPU baselines on FP32-rounded
//   inputs for CUDA tests.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Quiet NaN (canonical single-precision)
#ifndef TRIX_QNAN_U32
#define TRIX_QNAN_U32 0x7fc00000u
#endif

static __device__ __forceinline__ float trix_qnan() {
    return __int_as_float((int)TRIX_QNAN_U32);
}

// FMA EMA update: prev += a * (x - prev)
static __device__ __forceinline__ float ema_step(float prev, float x, float a) {
    return fmaf(a, x - prev, prev);
}

// Double-precision EMA update (used to better match scalar reference on tricky series).
static __device__ __forceinline__ double ema_step_d(double prev, double x, double a) {
    return fma(a, x - prev, prev);
}

// -------------------------------------------
// 1) Param-sweep over periods (one price series)
//    Inputs are ln(price) precomputed on host for reuse across rows.
//    One block per combo, single thread sequential scan (simple+robust).
// -------------------------------------------
extern "C" __global__
void trix_batch_f32(const float* __restrict__ logs,
                    const int*   __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) return;

    const int period = periods[combo];
    if (period <= 0 || series_len <= 0) return;

    float* __restrict__ out_row = out + combo * series_len;

    const int warmup_end = first_valid + 3 * (period - 1) + 1; // first finite sample index
    const int nan_to = warmup_end < series_len ? warmup_end : series_len;
    const float qn = trix_qnan();
    for (int i = 0; i < nan_to; ++i) out_row[i] = qn;
    if (warmup_end >= series_len) return;

    const float a = 2.0f / (float(period) + 1.0f);
    const float inv_n = 1.0f / float(period);
    const float SCALE = 10000.0f;
    const double a_d = (double)a;
    const double inv_n_d = (double)inv_n;

    // Stage 1 seed: EMA1 via SMA of logs[first_valid .. first_valid+period)
    float sum1 = 0.0f;
    for (int i = first_valid; i < first_valid + period; ++i) {
        sum1 += logs[i];
    }
    float ema1 = sum1 * inv_n; // at idx = first_valid + period - 1

    // Build remaining EMA1 values (period-1) and accumulate for EMA2 seed
    float sum_ema1 = ema1;
    int end2 = first_valid + 2 * period - 1;
    for (int i = first_valid + period; i < end2; ++i) {
        ema1 = ema_step(ema1, logs[i], a);
        sum_ema1 += ema1;
    }

    // Stage 2 seed: EMA2 via SMA of first `period` EMA1s
    float ema2 = sum_ema1 * inv_n; // at idx = first_valid + 2*period - 2

    // Build remaining EMA2 values (period-1) and accumulate for EMA3 seed
    double sum_ema2 = (double)ema2;
    int end3 = first_valid + 3 * period - 2;
    for (int i = end2; i < end3; ++i) {
        ema1 = ema_step(ema1, logs[i], a);
        ema2 = ema_step(ema2, ema1, a);
        sum_ema2 += (double)ema2;
    }

    // Stage 3 seed: EMA3 via SMA of first `period` EMA2s
    double ema3_prev = sum_ema2 * inv_n_d; // at idx = first_valid + 3*period - 3

    // First TRIX sample at warmup_end (== first_valid + 3*(p-1) + 1)
    int t = warmup_end;
    double ema3 = ema3_prev;
    {
        const float lv = logs[t];
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        ema3 = ema_step_d(ema3_prev, (double)ema2, a_d);
        out_row[t] = (float)((ema3 - ema3_prev) * (double)SCALE);
        ema3_prev = ema3;
        ++t;
    }

    // Main time loop
    for (; t < series_len; ++t) {
        const float lv = logs[t];
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        ema3 = ema_step_d(ema3_prev, (double)ema2, a_d);
        out_row[t] = (float)((ema3 - ema3_prev) * (double)SCALE);
        ema3_prev = ema3;
    }
}

// -------------------------------------------
// 1b) Param-sweep over periods (one price series), warp-cooperative scan.
//
// - One warp per combo; emits 32 timesteps per iteration via an inclusive scan
//   over affine transforms for each EMA stage.
// - Warmup/seed is computed sequentially in lane0 (<= ~3*period steps).
//
// Launch guidance: blockDim.x should be a multiple of 32 (e.g., 256).
// -------------------------------------------
extern "C" __global__
void trix_batch_warp_scan_f32(const float* __restrict__ logs,
                              const int*   __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const unsigned mask = 0xffffffffu;
    const int lane = (int)(threadIdx.x & 31);
    const int warp_id = (int)(threadIdx.x >> 5);
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int combo = (int)blockIdx.x * warps_per_block + warp_id;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float qn = trix_qnan();
    float* __restrict__ out_row = out + (size_t)combo * (size_t)series_len;

    if (period <= 0 || series_len <= 0) {
        for (int i = lane; i < series_len; i += 32) out_row[i] = qn;
        return;
    }

    int fv = first_valid;
    if (fv < 0) fv = 0;

    const int warmup_end = fv + 3 * (period - 1) + 1; // first finite sample index
    const int nan_to = warmup_end < series_len ? warmup_end : series_len;
    for (int i = lane; i < nan_to; i += 32) out_row[i] = qn;
    if (warmup_end >= series_len) return;

    const float a = 2.0f / (float(period) + 1.0f);
    const float one_minus_a = 1.0f - a;
    const float inv_n = 1.0f / float(period);
    const float SCALE = 10000.0f;

    float ema1 = 0.0f;
    float ema2 = 0.0f;
    float ema3_prev = 0.0f;
    if (lane == 0) {
        float sum1 = 0.0f;
        for (int i = fv; i < fv + period; ++i) sum1 += logs[i];
        ema1 = sum1 * inv_n; // at idx = fv + period - 1

        float sum_ema1 = ema1;
        const int end2 = fv + 2 * period - 1;
        for (int i = fv + period; i < end2; ++i) {
            ema1 = ema_step(ema1, logs[i], a);
            sum_ema1 += ema1;
        }
        ema2 = sum_ema1 * inv_n; // at idx = fv + 2*period - 2

        float sum_ema2 = ema2;
        const int end3 = fv + 3 * period - 2;
        for (int i = end2; i < end3; ++i) {
            ema1 = ema_step(ema1, logs[i], a);
            ema2 = ema_step(ema2, ema1, a);
            sum_ema2 += ema2;
        }
        ema3_prev = sum_ema2 * inv_n; // at idx = fv + 3*period - 3 (== warmup_end - 1)
    }

    ema1 = __shfl_sync(mask, ema1, 0);
    ema2 = __shfl_sync(mask, ema2, 0);
    ema3_prev = __shfl_sync(mask, ema3_prev, 0);

    for (int t0 = warmup_end; t0 < series_len; t0 += 32) {
        const int t = t0 + lane;
        const float lv = (t < series_len) ? logs[t] : 0.0f;

        // EMA1 scan: y = one_minus_a*y_prev + a*lv
        float A1 = one_minus_a;
        float B1 = a * lv;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A1, offset);
            const float B_prev = __shfl_up_sync(mask, B1, offset);
            if (lane >= offset) {
                const float A_cur = A1;
                const float B_cur = B1;
                A1 = A_cur * A_prev;
                B1 = __fmaf_rn(A_cur, B_prev, B_cur);
            }
        }
        const float ema1_lane = __fmaf_rn(A1, ema1, B1);

        // EMA2 scan: y = one_minus_a*y_prev + a*ema1_lane
        float A2 = one_minus_a;
        float B2 = a * ema1_lane;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A2, offset);
            const float B_prev = __shfl_up_sync(mask, B2, offset);
            if (lane >= offset) {
                const float A_cur = A2;
                const float B_cur = B2;
                A2 = A_cur * A_prev;
                B2 = __fmaf_rn(A_cur, B_prev, B_cur);
            }
        }
        const float ema2_lane = __fmaf_rn(A2, ema2, B2);

        // EMA3 scan: y = one_minus_a*y_prev + a*ema2_lane
        float A3 = one_minus_a;
        float B3 = a * ema2_lane;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A3, offset);
            const float B_prev = __shfl_up_sync(mask, B3, offset);
            if (lane >= offset) {
                const float A_cur = A3;
                const float B_cur = B3;
                A3 = A_cur * A_prev;
                B3 = __fmaf_rn(A_cur, B_prev, B_cur);
            }
        }
        const float ema3_lane = __fmaf_rn(A3, ema3_prev, B3);

        const float ema3_prev_lane =
            (lane == 0) ? ema3_prev : __shfl_up_sync(mask, ema3_lane, 1);
        if (t < series_len) out_row[t] = (ema3_lane - ema3_prev_lane) * SCALE;

        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        ema1 = __shfl_sync(mask, ema1_lane, last_lane);
        ema2 = __shfl_sync(mask, ema2_lane, last_lane);
        ema3_prev = __shfl_sync(mask, ema3_lane, last_lane);
    }
}

// --------------------------------------------------------------
// 2) Multi-series, one period, time-major layout (prices_tm[t*N + s])
//    Compute ln(price) on device (series are independent).
// --------------------------------------------------------------
extern "C" __global__
void trix_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm)
{
    const int sidx = blockIdx.x;
    if (sidx >= num_series || threadIdx.x != 0) return;
    if (period <= 0 || series_len <= 0) return;

    int fv = first_valids[sidx];
    if (fv < 0) fv = 0;

    const int warmup_end = fv + 3 * (period - 1) + 1;
    const int nan_to = warmup_end < series_len ? warmup_end : series_len;
    const float qn = trix_qnan();
    for (int t = 0; t < nan_to; ++t) {
        out_tm[t * num_series + sidx] = qn;
    }
    if (warmup_end >= series_len) return;

    const float a = 2.0f / (float(period) + 1.0f);
    const float inv_n = 1.0f / float(period);
    const float SCALE = 10000.0f;

    // Stage 1 seed: EMA1 via SMA of ln(price)
    float sum1 = 0.0f;
    for (int i = fv; i < fv + period; ++i) {
        const float px = prices_tm[i * num_series + sidx];
        sum1 += logf(px);
    }
    float ema1 = sum1 * inv_n;

    // Build remaining EMA1 values (period-1) and accumulate for EMA2 seed
    float sum_ema1 = ema1;
    int end2 = fv + 2 * period - 1;
    for (int i = fv + period; i < end2; ++i) {
        const float lv = logf(prices_tm[i * num_series + sidx]);
        ema1 = ema_step(ema1, lv, a);
        sum_ema1 += ema1;
    }

    // Stage 2 seed: EMA2 via SMA of EMA1
    float ema2 = sum_ema1 * inv_n;

    // Build remaining EMA2 values (period-1) and accumulate for EMA3 seed
    float sum_ema2 = ema2;
    int end3 = fv + 3 * period - 2;
    for (int i = end2; i < end3; ++i) {
        const float lv = logf(prices_tm[i * num_series + sidx]);
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        sum_ema2 += ema2;
    }

    // Stage 3 seed
    float ema3_prev = sum_ema2 * inv_n;

    // First TRIX sample at warmup_end
    int t = warmup_end;
    float ema3 = ema3_prev;
    {
        const float lv = logf(prices_tm[t * num_series + sidx]);
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        ema3 = ema_step(ema3_prev, ema2, a);
        out_tm[t * num_series + sidx] = (ema3 - ema3_prev) * SCALE;
        ema3_prev = ema3;
        ++t;
    }

    for (; t < series_len; ++t) {
        const float lv = logf(prices_tm[t * num_series + sidx]);
        ema1 = ema_step(ema1, lv, a);
        ema2 = ema_step(ema2, ema1, a);
        ema3 = ema_step(ema3_prev, ema2, a);
        out_tm[t * num_series + sidx] = (ema3 - ema3_prev) * SCALE;
        ema3_prev = ema3;
    }
}

