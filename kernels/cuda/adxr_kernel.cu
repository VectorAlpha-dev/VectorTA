// CUDA kernels for ADXR (Average Directional Index Rating)
//
// Semantics mirror the scalar Rust implementation in src/indicators/adxr.rs:
// - Warmup: values before (first_valid + 2*period) are NaN
// - Wilder smoothing with alpha = 1/period for smoothed +DM/-DM
// - DX = 100 * |+DM_s - -DM_s| / (+DM_s + -DM_s), denominator 0 -> 0
// - ADX builds as mean of first `period` DX values, then Wilder recursion
// - ADXR[t] = 0.5 * (ADX[t] + ADX[t - period]) once the ring is full

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

static __forceinline__ __device__ float fmax3(float a, float b, float c) {
    return fmaxf(a, fmaxf(b, c));
}

extern "C" __global__
void adxr_batch_f32(const float* __restrict__ high,
                    const float* __restrict__ low,
                    const float* __restrict__ close,
                    const int* __restrict__ periods,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.y * gridDim.x + blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int base = combo * series_len;

    // Fill entire row with NaNs cooperatively
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    // Only one thread per row runs the sequential recurrence
    if (threadIdx.x != 0) return;

    // Bootstrap sums over first window: i in [first+1 .. first+period]
    int i = first_valid + 1;
    const int stop = min(first_valid + period, series_len - 1);
    // Accumulate in double for improved numeric stability.
    double atr_sum = 0.0;
    double pdm_sum = 0.0;
    double mdm_sum = 0.0;
    while (i <= stop) {
        const float pc = close[i - 1];
        const float ch = high[i];
        const float cl = low[i];
        const float ph = high[i - 1];
        const float pl = low[i - 1];

        const double tr = (double)fmax3(ch - cl, fabsf(ch - pc), fabsf(cl - pc));
        atr_sum += tr;

        const float up = ch - ph;
        const float down = pl - cl;
        if (up > down && up > 0.0f) pdm_sum += (double)up;
        if (down > up && down > 0.0f) mdm_sum += (double)down;
        ++i;
    }

    // Initial DX from bootstrap sums (ATR cancels)
    const double denom0 = pdm_sum + mdm_sum;
    const double initial_dx = (denom0 > 0.0) ? (100.0 * fabs(pdm_sum - mdm_sum) / denom0) : 0.0;

    const float p = (float)period;
    const float inv_p = 1.0f / p;
    const float one_minus = 1.0f - inv_p;
    const float pm1 = p - 1.0f;
    const int warmup_start = first_valid + 2 * period;

    // Wilder running state
    double atr = atr_sum;
    double pdm_s = pdm_sum;
    double mdm_s = mdm_sum;

    double dx_sum = initial_dx;
    int dx_count = 1;
    double adx_last = NAN;
    bool have_adx = false;

    // Use a local ring buffer for ADX history. Periods are small in practice.
    double ring_local[256];
    const bool use_local = (period <= 256);
    if (use_local) {
        for (int k = 0; k < period; ++k) ring_local[k] = NAN;
    }
    int head = 0;

    i = first_valid + period + 1;
    while (i < series_len) {
        const float pc = close[i - 1];
        const float ch = high[i];
        const float cl = low[i];
        const float ph = high[i - 1];
        const float pl = low[i - 1];

        const double tr = (double)fmax3(ch - cl, fabsf(ch - pc), fabsf(cl - pc));
        const float up = ch - ph;
        const float down = pl - cl;
        const float plus_dm = (up > down && up > 0.0f) ? up : 0.0f;
        const float minus_dm = (down > up && down > 0.0f) ? down : 0.0f;

        atr   = fma(atr,   (double)one_minus, tr);
        pdm_s = fma(pdm_s, (double)one_minus, (double)plus_dm);
        mdm_s = fma(mdm_s, (double)one_minus, (double)minus_dm);

        const double denom = pdm_s + mdm_s;
        const double dx = (denom > 0.0) ? (100.0 * fabs(pdm_s - mdm_s) / denom) : 0.0;

        if (dx_count < period) {
            dx_sum += dx;
            dx_count += 1;
            if (dx_count == period) {
                adx_last = dx_sum * inv_p;
                have_adx = true;
                // push into ring
                double prev = use_local ? ring_local[head] : NAN;
                if (use_local) ring_local[head] = adx_last;
                head += 1; if (head == period) head = 0;
                if (i >= warmup_start && !isnan(prev)) {
                    out[base + i] = (float)(0.5 * (adx_last + prev));
                }
            }
        } else if (have_adx) {
            const double adx_curr = (adx_last * (double)pm1 + dx) * (double)inv_p;
            adx_last = adx_curr;
            double prev = use_local ? ring_local[head] : NAN;
            if (use_local) ring_local[head] = adx_curr;
            head += 1; if (head == period) head = 0;
            if (i >= warmup_start && !isnan(prev)) {
                out[base + i] = (float)(0.5 * (adx_curr + prev));
            }
        }

        ++i;
    }
}

// Many-series × one-param (time-major). Each block handles one series with a single thread.
extern "C" __global__
void adxr_many_series_one_param_f32(const float* __restrict__ high_tm,
                                    const float* __restrict__ low_tm,
                                    const float* __restrict__ close_tm,
                                    const int* __restrict__ first_valids,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int stride = num_series;

    // Fill with NaNs
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    // Bootstrap over [first+1 .. first+period]
    int i = first_valid + 1;
    const int stop = min(first_valid + period, series_len - 1);
    double atr_sum = 0.0;
    double pdm_sum = 0.0;
    double mdm_sum = 0.0;
    while (i <= stop) {
        const float pc = close_tm[(i - 1) * stride + series];
        const float ch = high_tm[i * stride + series];
        const float cl = low_tm[i * stride + series];
        const float ph = high_tm[(i - 1) * stride + series];
        const float pl = low_tm[(i - 1) * stride + series];
        const double tr = (double)fmax3(ch - cl, fabsf(ch - pc), fabsf(cl - pc));
        atr_sum += tr;
        const float up = ch - ph;
        const float down = pl - cl;
        if (up > down && up > 0.0f) pdm_sum += (double)up;
        if (down > up && down > 0.0f) mdm_sum += (double)down;
        ++i;
    }

    const double denom0 = pdm_sum + mdm_sum;
    const double initial_dx = (denom0 > 0.0) ? (100.0 * fabs(pdm_sum - mdm_sum) / denom0) : 0.0;

    const float p = (float)period;
    const float inv_p = 1.0f / p;
    const float one_minus = 1.0f - inv_p;
    const float pm1 = p - 1.0f;
    const int warmup_start = first_valid + 2 * period;

    double atr = atr_sum;
    double pdm_s = pdm_sum;
    double mdm_s = mdm_sum;

    double dx_sum = initial_dx;
    int dx_count = 1;
    double adx_last = NAN;
    bool have_adx = false;

    int head = 0;
    // Use small local ring for typical periods; fall back to noop if too large
    const bool use_local = (period <= 256);
    double ring_local[256];
    if (use_local) {
        for (int k = 0; k < period; ++k) ring_local[k] = NAN;
    }

    i = first_valid + period + 1;
    while (i < series_len) {
        const float pc = close_tm[(i - 1) * stride + series];
        const float ch = high_tm[i * stride + series];
        const float cl = low_tm[i * stride + series];
        const float ph = high_tm[(i - 1) * stride + series];
        const float pl = low_tm[(i - 1) * stride + series];

        const double tr = (double)fmax3(ch - cl, fabsf(ch - pc), fabsf(cl - pc));
        const float up = ch - ph;
        const float down = pl - cl;
        const float plus_dm = (up > down && up > 0.0f) ? up : 0.0f;
        const float minus_dm = (down > up && down > 0.0f) ? down : 0.0f;

        atr   = fma(atr,   (double)one_minus, tr);
        pdm_s = fma(pdm_s, (double)one_minus, (double)plus_dm);
        mdm_s = fma(mdm_s, (double)one_minus, (double)minus_dm);

        const double denom = pdm_s + mdm_s;
        const double dx = (denom > 0.0) ? (100.0 * fabs(pdm_s - mdm_s) / denom) : 0.0;

        if (dx_count < period) {
            dx_sum += dx;
            dx_count += 1;
            if (dx_count == period) {
                adx_last = dx_sum * inv_p;
                have_adx = true;
                double prev = use_local ? ring_local[head] : NAN;
                if (use_local) ring_local[head] = adx_last;
                head += 1; if (head == period) head = 0;
                if (i >= warmup_start && !isnan(prev)) {
                    out_tm[i * stride + series] = (float)(0.5 * (adx_last + prev));
                }
            }
        } else if (have_adx) {
            const double adx_curr = (adx_last * (double)pm1 + dx) * (double)inv_p;
            adx_last = adx_curr;
            double prev = use_local ? ring_local[head] : NAN;
            if (use_local) ring_local[head] = adx_curr;
            head += 1; if (head == period) head = 0;
            if (i >= warmup_start && !isnan(prev)) {
                out_tm[i * stride + series] = (float)(0.5 * (adx_curr + prev));
            }
        }

        ++i;
    }
}
