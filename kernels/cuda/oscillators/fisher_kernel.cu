// CUDA kernels for Fisher Transform (Ehlers variant) in FP32.
//
// Behavior mirrors src/indicators/fisher.rs (scalar path):
// - Inputs are HL2 midpoints (0.5 * (high + low)) precomputed on host for batch
//   and many-series kernels to reduce memory traffic.
// - Warmup prefix: indices < (first_valid + period - 1) are NaN.
// - Recurrence:
//     val1 = 0.67 * val1 + 0.66 * ((hl - min) / max(max-min, 0.001) - 0.5)
//     val1 is clamped to [-0.999, 0.999]
//     signal[t] = prev_fish
//     fisher[t] = 0.5*ln((1+val1)/(1-val1)) + 0.5*prev_fish
// - Any NaN in the active window propagates via FP ops; host precompute ensures
//   first_valid respects NaN gating across high and low.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif
#ifndef CUDART_INF
#define CUDART_INF (__longlong_as_double(0x7ff0000000000000ULL))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// One-series × many-params (batch). Each block handles one period row.
extern "C" __global__ void fisher_batch_f32(const float* __restrict__ hl,
                                             const int*   __restrict__ periods,
                                             int series_len,
                                             int n_combos,
                                             int first_valid,
                                             float* __restrict__ out_fisher,
                                             float* __restrict__ out_signal) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    // Guard invalid inputs by writing NaNs
    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) {
            out_fisher[base + i] = NAN;
            out_signal[base + i] = NAN;
        }
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        for (int i = 0; i < series_len; ++i) { out_fisher[base + i] = NAN; out_signal[base + i] = NAN; }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) { out_fisher[base + i] = NAN; out_signal[base + i] = NAN; }

    // Sequential scan per row (thread 0), minimizes divergence and matches scalar recurrence.
    if (threadIdx.x != 0) return;

    double prev_fish = 0.0;
    double val1 = 0.0;

    for (int t = warm; t < series_len; ++t) {
        const int start = t + 1 - period;
        double minv = CUDART_INF;
        double maxv = -CUDART_INF;
        // Window scan over HL2 midpoints
        for (int k = 0; k < period; ++k) {
            const double x = static_cast<double>(hl[start + k]);
            if (x < minv) minv = x;
            if (x > maxv) maxv = x;
        }
        double range = maxv - minv;
        if (range < 0.001) range = 0.001;
        const double x = static_cast<double>(hl[t]);
        val1 = 0.67 * val1 + 0.66 * ((x - minv) / range - 0.5);
        if (val1 > 0.99) val1 = 0.999; else if (val1 < -0.99) val1 = -0.999;
        out_signal[base + t] = static_cast<float>(prev_fish);
        const double ln_term = log((1.0 + val1) / (1.0 - val1));
        const double new_fish = 0.5 * ln_term + 0.5 * prev_fish;
        out_fisher[base + t] = static_cast<float>(new_fish);
        prev_fish = new_fish;
    }
}

// Many-series × one-param (time-major). Each thread handles one series (column).
extern "C" __global__ void fisher_many_series_one_param_f32(
    const float* __restrict__ hl_tm,        // time-major: [row * num_series + series]
    const int*   __restrict__ first_valids, // per-series first valid index
    int num_series,
    int series_len,
    int period,
    float* __restrict__ fisher_tm,
    float* __restrict__ signal_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        for (int r = 0; r < series_len; ++r) {
            const int idx = r * num_series + series;
            fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
        }
        return;
    }

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (UNLIKELY(first_valid >= series_len || (series_len - first_valid) < period)) {
        for (int r = 0; r < series_len; ++r) {
            const int idx = r * num_series + series;
            fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int r = 0; r < warm; ++r) {
        const int idx = r * num_series + series;
        fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
    }

    double prev_fish = 0.0;
    double val1 = 0.0;
    for (int r = warm; r < series_len; ++r) {
        const int start = r + 1 - period;
        double minv = CUDART_INF;
        double maxv = -CUDART_INF;
        // Window scan with time-major stride
        for (int k = 0; k < period; ++k) {
            const int idx = (start + k) * num_series + series;
            const double x = static_cast<double>(hl_tm[idx]);
            if (x < minv) minv = x;
            if (x > maxv) maxv = x;
        }
        double range = maxv - minv;
        if (range < 0.001) range = 0.001;
        const double x = static_cast<double>(hl_tm[r * num_series + series]);
        val1 = 0.67 * val1 + 0.66 * ((x - minv) / range - 0.5);
        if (val1 > 0.99) val1 = 0.999; else if (val1 < -0.99) val1 = -0.999;
        const int idxo = r * num_series + series;
        signal_tm[idxo] = static_cast<float>(prev_fish);
        const double ln_term = log((1.0 + val1) / (1.0 - val1));
        const double new_fish = 0.5 * ln_term + 0.5 * prev_fish;
        fisher_tm[idxo] = static_cast<float>(new_fish);
        prev_fish = new_fish;
    }
}
