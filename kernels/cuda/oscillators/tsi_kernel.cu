// CUDA kernels for True Strength Index (TSI)
//
// Matches scalar semantics in src/indicators/tsi.rs:
// - Warmup: first_valid + long + short elements are NaN
// - Use EMA on momentum and on abs(momentum) with periods long, short
// - NaN inputs do not advance state; output becomes NaN at those indices
// - Division by zero => NaN; outputs clamped to [-100, 100]
// - FP32 math; use FMA where beneficial

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

extern "C" __global__
void tsi_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ longs,
                   const int* __restrict__ shorts,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;

    // Initialize entire row with NaNs in parallel to guarantee warmup prefix
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-thread sequential scan per row
    if (first_valid < 0 || first_valid >= series_len) return;

    const int l = longs[combo];
    const int s = shorts[combo];
    if (l <= 0 || s <= 0) return;

    const double long_alpha  = 2.0 / (double(l) + 1.0);
    const double short_alpha = 2.0 / (double(s) + 1.0);
    const double long_1m  = 1.0 - long_alpha;
    const double short_1m = 1.0 - short_alpha;

    const int warm = first_valid + l + s;
    if (series_len <= first_valid + 1) return;

    double prev = (double)prices[first_valid];
    const double nextv = (double)prices[first_valid + 1];
    if (!isfinite(nextv)) return; // follow scalar early-exit behavior

    const double first_mom = nextv - prev;
    prev = nextv;

    // Seed EMAs with first momentum
    double ema_long_num  = first_mom;
    double ema_short_num = first_mom;
    double ema_long_den  = fabs(first_mom);
    double ema_short_den = fabs(first_mom);

    for (int i = first_valid + 2; i < series_len; ++i) {
        const double cur = (double)prices[i];
        if (!isfinite(cur)) {
            if (i >= warm) out[base + i] = NAN;
            continue;
        }

        const double m = cur - prev;
        prev = cur;

        const double am = fabs(m);
        // EMA updates in FP64 to reduce drift
        ema_long_num  = long_alpha * m + long_1m * ema_long_num;
        ema_short_num = short_alpha * ema_long_num + short_1m * ema_short_num;

        ema_long_den  = long_alpha * am + long_1m * ema_long_den;
        ema_short_den = short_alpha * ema_long_den + short_1m * ema_short_den;

        if (i >= warm) {
            const double den = ema_short_den;
            if (den == 0.0) {
                out[base + i] = NAN;
            } else {
                const double v = 100.0 * (ema_short_num / den);
                out[base + i] = clampf((float)v, -100.0f, 100.0f);
            }
        }
    }
}

// Many-series (time-major), one param. Output time-major [rows x cols].
extern "C" __global__
void tsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   int long_p,
                                   int short_p,
                                   int num_series,
                                   int series_len,
                                   const int* __restrict__ first_valids,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index (column)
    if (s >= num_series) return;
    if (long_p <= 0 || short_p <= 0) return;

    const double long_alpha  = 2.0 / (double(long_p) + 1.0);
    const double short_alpha = 2.0 / (double(short_p) + 1.0);
    const double long_1m  = 1.0 - long_alpha;
    const double short_1m = 1.0 - short_alpha;

    const int first = max(0, first_valids[s]);
    if (first >= series_len) return;
    const int warm = first + long_p + short_p;

    // Initialize prefix to NaN for this series
    for (int t = 0; t < min(warm, series_len); ++t) {
        out_tm[t * num_series + s] = NAN;
    }

    if (series_len <= first + 1) return;
    const int idx0 = first * num_series + s;
    double prev = (double)prices_tm[idx0];
    const double nextv = (double)prices_tm[idx0 + num_series];
    if (!isfinite(nextv)) return;

    const double first_mom = nextv - prev;
    prev = nextv;

    double ema_long_num  = first_mom;
    double ema_short_num = first_mom;
    double ema_long_den  = fabs(first_mom);
    double ema_short_den = fabs(first_mom);

    for (int t = first + 2; t < series_len; ++t) {
        const int idx = t * num_series + s;
        const double cur = (double)prices_tm[idx];
        if (!isfinite(cur)) {
            if (t >= warm) out_tm[idx] = NAN;
            continue;
        }

        const double m = cur - prev;
        prev = cur;
        const double am = fabs(m);

        ema_long_num  = long_alpha * m + long_1m * ema_long_num;
        ema_short_num = short_alpha * ema_long_num + short_1m * ema_short_num;
        ema_long_den  = long_alpha * am + long_1m * ema_long_den;
        ema_short_den = short_alpha * ema_long_den + short_1m * ema_short_den;

        if (t >= warm) {
            const double den = ema_short_den;
            if (den == 0.0) {
                out_tm[idx] = NAN;
            } else {
                const double v = 100.0 * (ema_short_num / den);
                out_tm[idx] = clampf((float)v, -100.0f, 100.0f);
            }
        }
    }
}
