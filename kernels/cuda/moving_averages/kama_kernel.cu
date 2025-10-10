// CUDA kernels for the Kaufman Adaptive Moving Average (KAMA).
//
// The kernels follow the VRAM-first pattern used throughout the moving
// averages module: all arithmetic is performed in FP32 buffers exposed to the
// caller, while the recurrence itself is evaluated in FP64 to keep numerical
// drift aligned with the CPU reference implementation.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

namespace {
__device__ inline double kama_const_max() {
    return 2.0 / 31.0; // 2 / (30 + 1)
}

__device__ inline double kama_const_diff() {
    return (2.0 / 3.0) - kama_const_max();
}
} // namespace

extern "C" __global__
void kama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int base = combo * series_len;

    const float nan_f = NAN;
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = nan_f;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    if (period <= 0 || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int valid = series_len - first_valid;
    if (period >= valid) {
        return;
    }

    const int warm = first_valid + period;
    if (warm >= series_len) {
        return;
    }

    const int lookback = period - 1;
    double sum_roc1 = 0.0;
    const int today = first_valid;

    for (int i = 0; i <= lookback; ++i) {
        const int idx0 = today + i;
        const int idx1 = idx0 + 1;
        const double prev = static_cast<double>(prices[idx0]);
        const double next = static_cast<double>(prices[idx1]);
        sum_roc1 += fabs(next - prev);
    }

    const int initial_idx = today + lookback + 1;
    double prev_kama = static_cast<double>(prices[initial_idx]);
    out[base + initial_idx] = static_cast<float>(prev_kama);

    int trailing_idx = today;
    double trailing_value = static_cast<double>(prices[trailing_idx]);

    const double const_max = kama_const_max();
    const double const_diff = kama_const_diff();

    for (int i = initial_idx + 1; i < series_len; ++i) {
        const double price = static_cast<double>(prices[i]);
        const double prev_price = static_cast<double>(prices[i - 1]);
        const double next_trailing = static_cast<double>(prices[trailing_idx + 1]);

        sum_roc1 -= fabs(next_trailing - trailing_value);
        sum_roc1 += fabs(price - prev_price);

        trailing_value = next_trailing;
        trailing_idx += 1;

        const double anchor = static_cast<double>(prices[trailing_idx]);
        const double direction = fabs(price - anchor);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);
        double sc = er * const_diff + const_max;
        sc *= sc;

        prev_kama += (price - prev_kama) * sc;
        out[base + i] = static_cast<float>(prev_kama);
    }
}

// Prefix-optimized batch kernel: uses host-precomputed prefix of |p[t]-p[t-1]| to
// seed the initial sum_roc1 in O(1) per combo, matching scalar behavior.
//
// prefix_roc1 must be a length-(series_len+1) array where:
//   prefix_roc1[0] = 0
//   prefix_roc1[t] = sum_{k=1..t} |p[k]-p[k-1]| with NaN-insensitive accumulation (host)
extern "C" __global__
void kama_batch_prefix_f32(const float* __restrict__ prices,
                           const float* __restrict__ prefix_roc1,
                           const int* __restrict__ periods,
                           int series_len,
                           int n_combos,
                           int first_valid,
                           float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int base = combo * series_len;

    const float nan_f = NAN;
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = nan_f;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    if (period <= 0 || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int valid = series_len - first_valid;
    if (period >= valid) {
        return;
    }

    const int lookback = period - 1;
    const int initial_idx = first_valid + lookback + 1;
    if (initial_idx >= series_len) {
        return;
    }

    // Seed Σ|Δp| via prefix sums (host-precomputed)
    double sum_roc1 = static_cast<double>(
        prefix_roc1[first_valid + period] - prefix_roc1[first_valid]
    );

    // Seed first output
    double prev_kama = static_cast<double>(prices[initial_idx]);
    out[base + initial_idx] = static_cast<float>(prev_kama);

    int trailing_idx = first_valid;
    double trailing_value = static_cast<double>(prices[trailing_idx]);

    const double const_max = kama_const_max();
    const double const_diff = kama_const_diff();

    for (int i = initial_idx + 1; i < series_len; ++i) {
        const double price = static_cast<double>(prices[i]);
        const double prev_price = static_cast<double>(prices[i - 1]);
        const double next_trailing = static_cast<double>(prices[trailing_idx + 1]);

        sum_roc1 -= fabs(next_trailing - trailing_value);
        sum_roc1 += fabs(price - prev_price);

        trailing_value = next_trailing;
        trailing_idx += 1;

        const double anchor = static_cast<double>(prices[trailing_idx]);
        const double direction = fabs(price - anchor);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);
        double sc = er * const_diff + const_max;
        sc *= sc;

        prev_kama += (price - prev_kama) * sc;
        out[base + i] = static_cast<float>(prev_kama);
    }
}

extern "C" __global__
void kama_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) {
        return;
    }

    const float nan_f = NAN;
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * num_series + series] = nan_f;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    if (period <= 0) {
        return;
    }

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int valid = series_len - first_valid;
    if (period >= valid) {
        return;
    }

    const int warm = first_valid + period;
    if (warm >= series_len) {
        return;
    }

    const int lookback = period - 1;

    auto at = [num_series](const float* buf, int row, int col) {
        return buf[row * num_series + col];
    };

    double sum_roc1 = 0.0;
    for (int i = 0; i <= lookback; ++i) {
        const double prev = static_cast<double>(at(prices_tm, first_valid + i, series));
        const double next = static_cast<double>(at(prices_tm, first_valid + i + 1, series));
        sum_roc1 += fabs(next - prev);
    }

    const int initial_idx = first_valid + lookback + 1;
    double prev_kama = static_cast<double>(at(prices_tm, initial_idx, series));
    out_tm[initial_idx * num_series + series] = static_cast<float>(prev_kama);

    int trailing_idx = first_valid;
    double trailing_value = static_cast<double>(at(prices_tm, trailing_idx, series));

    const double const_max = kama_const_max();
    const double const_diff = kama_const_diff();

    for (int t = initial_idx + 1; t < series_len; ++t) {
        const double price = static_cast<double>(at(prices_tm, t, series));
        const double prev_price = static_cast<double>(at(prices_tm, t - 1, series));
        const double next_trailing = static_cast<double>(at(prices_tm, trailing_idx + 1, series));

        sum_roc1 -= fabs(next_trailing - trailing_value);
        sum_roc1 += fabs(price - prev_price);

        trailing_value = next_trailing;
        trailing_idx += 1;

        const double anchor = static_cast<double>(at(prices_tm, trailing_idx, series));
        const double direction = fabs(price - anchor);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);
        double sc = er * const_diff + const_max;
        sc *= sc;

        prev_kama += (price - prev_kama) * sc;
        out_tm[t * num_series + series] = static_cast<float>(prev_kama);
    }
}
