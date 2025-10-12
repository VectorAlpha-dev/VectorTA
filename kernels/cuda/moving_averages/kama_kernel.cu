// Optimized CUDA kernels for the Kaufman Adaptive Moving Average (KAMA).
//
// Drop-in replacements tuned for CUDA 13 / Ada (SM 8.9).
// Changes vs. previous version:
//  - Remove full-buffer NaN clear + __syncthreads(); only write NaNs where needed.
//  - Warp-parallel reduction to seed initial Σ|Δp|.
//  - Fewer global loads by carrying prev_price and reusing trailing_value.
//  - Use FMA for the recurrence.
//  - Add __launch_bounds__(32) hint (one warp per block recommended).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h> // CUDART_NAN_F

namespace {

constexpr int WARP = 32;

__device__ __forceinline__ double kama_const_max() {
    return 2.0 / 31.0; // 2 / (30 + 1)
}

__device__ __forceinline__ double kama_const_diff() {
    return (2.0 / 3.0) - kama_const_max();
}

__device__ __forceinline__ double warp_sum(double v) {
    unsigned m = __activemask();
    #pragma unroll
    for (int off = WARP >> 1; off > 0; off >>= 1) {
        v += __shfl_down_sync(m, v, off);
    }
    return v;
}

} // namespace

// ============================================================================
// 1) One series, many period combos (batch), prices are contiguous (row-major).
//    Each block handles one combo; recommend blockDim.x == 32.
// ============================================================================
extern "C" __global__ __launch_bounds__(32)
void kama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    // Validate once; if invalid, fill all NaNs and return.
    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid)) ||
        ((first_valid + period) >= series_len);

    const float nan_f = CUDART_NAN_F;

    if (invalid) {
        // Parallel full clear to NaN for this combo; no further writes follow.
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = nan_f;
        }
        return;
    }

    // First valid output index for KAMA (right after the warmup window).
    const int initial_idx = first_valid + period;

    // Clear only the prefix that must be NaN. No barrier needed.
    for (int i = threadIdx.x; i < initial_idx; i += blockDim.x) {
        out[base + i] = nan_f;
    }

    // --- Initialize Σ|Δp| over [first_valid .. first_valid + period - 1]
    // Warp-parallel reduction across the first warp.
    double sum_roc1 = 0.0;
    if (threadIdx.x < WARP) {
        const int lane = threadIdx.x;
        double local = 0.0;
        const int start = first_valid;
        const int end   = first_valid + period; // exclusive for the left idx
        for (int j = start + lane; j < end; j += WARP) {
            const double a = static_cast<double>(prices[j]);
            const double b = static_cast<double>(prices[j + 1]);
            local += fabs(b - a);
        }
        local = warp_sum(local);
        if (lane == 0) sum_roc1 = local;
    }

    // Only thread 0 proceeds to the recurrence (sequential by nature).
    if (threadIdx.x != 0) return;

    // Seed the first KAMA output.
    double prev_price = static_cast<double>(prices[initial_idx]);
    double prev_kama  = prev_price;
    out[base + initial_idx] = static_cast<float>(prev_kama);

    int    trailing_idx   = first_valid;
    double trailing_value = static_cast<double>(prices[trailing_idx]);

    const double cmax  = kama_const_max();
    const double cdiff = kama_const_diff();

    for (int i = initial_idx + 1; i < series_len; ++i) {
        const double price         = static_cast<double>(prices[i]);
        const double next_trailing = static_cast<double>(prices[trailing_idx + 1]);

        // Incremental update of Σ|Δp| with two abs terms (entering & leaving).
        sum_roc1 += fabs(price - prev_price) - fabs(next_trailing - trailing_value);

        // Slide the trailing window one step.
        trailing_value = next_trailing;
        trailing_idx  += 1;

        // Reuse trailing_value as the "anchor" (saves a global load).
        const double direction = fabs(price - trailing_value);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);

        double sc = er * cdiff + cmax;
        sc *= sc;

        // Fused multiply-add for the recurrence.
        prev_kama = fma(price - prev_kama, sc, prev_kama);
        out[base + i] = static_cast<float>(prev_kama);

        // Carry prev_price forward (saves a global load of prices[i-1]).
        prev_price = price;
    }
}

// Prefix-optimized batch kernel: uses host-precomputed prefix of |p[t]-p[t-1]| to
// seed the initial sum_roc1 in O(1) per combo, matching scalar behavior.
//
// prefix_roc1 must be a length-(series_len+1) array where:
//   prefix_roc1[0] = 0
//   prefix_roc1[t] = sum_{k=1..t} |p[k]-p[k-1]| with NaN-insensitive accumulation (host)
extern "C" __global__ __launch_bounds__(32)
void kama_batch_prefix_f32(const float* __restrict__ prices,
                           const float* __restrict__ prefix_roc1,
                           const int* __restrict__ periods,
                           int series_len,
                           int n_combos,
                           int first_valid,
                           float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    const int initial_idx = first_valid + period; // first KAMA output index
    const float nan_f = CUDART_NAN_F;

    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid)) ||
        (initial_idx >= series_len);

    if (invalid) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = nan_f;
        }
        return;
    }

    for (int i = threadIdx.x; i < initial_idx; i += blockDim.x) {
        out[base + i] = nan_f;
    }

    if (threadIdx.x != 0) return;

    // Seed Σ|Δp| via prefix (O(1)).
    double sum_roc1 = static_cast<double>(
        prefix_roc1[initial_idx] - prefix_roc1[first_valid]
    );

    // Seed first output.
    double prev_price = static_cast<double>(prices[initial_idx]);
    double prev_kama  = prev_price;
    out[base + initial_idx] = static_cast<float>(prev_kama);

    int    trailing_idx   = first_valid;
    double trailing_value = static_cast<double>(prices[trailing_idx]);

    const double cmax  = kama_const_max();
    const double cdiff = kama_const_diff();

    for (int i = initial_idx + 1; i < series_len; ++i) {
        const double price         = static_cast<double>(prices[i]);
        const double next_trailing = static_cast<double>(prices[trailing_idx + 1]);

        sum_roc1 += fabs(price - prev_price) - fabs(next_trailing - trailing_value);

        trailing_value = next_trailing;
        trailing_idx  += 1;

        const double direction = fabs(price - trailing_value);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);

        double sc = er * cdiff + cmax;
        sc *= sc;

        prev_kama = fma(price - prev_kama, sc, prev_kama);
        out[base + i] = static_cast<float>(prev_kama);

        prev_price = price;
    }
}

extern "C" __global__ __launch_bounds__(32)
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

    const int first_valid = first_valids[series];
    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid));

    const int initial_idx = first_valid + period;
    const float nan_f = CUDART_NAN_F;

    // Helper for time-major indexing
    auto at = [num_series](const float* buf, int row, int col) {
        return buf[row * num_series + col];
    };

    // If invalid, clear the entire column.
    if (invalid || initial_idx >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + series] = nan_f;
        }
        return;
    }

    // Clear only the prefix to NaN.
    for (int t = threadIdx.x; t < initial_idx; t += blockDim.x) {
        out_tm[t * num_series + series] = nan_f;
    }

    // Warp-parallel init of Σ|Δp|.
    double sum_roc1 = 0.0;
    if (threadIdx.x < WARP) {
        const int lane = threadIdx.x;
        double local = 0.0;
        const int start = first_valid;
        const int end   = first_valid + period; // exclusive for left idx
        for (int j = start + lane; j < end; j += WARP) {
            const double a = static_cast<double>(at(prices_tm, j,     series));
            const double b = static_cast<double>(at(prices_tm, j + 1, series));
            local += fabs(b - a);
        }
        local = warp_sum(local);
        if (lane == 0) sum_roc1 = local;
    }

    if (threadIdx.x != 0) return;

    double prev_price = static_cast<double>(at(prices_tm, initial_idx, series));
    double prev_kama  = prev_price;
    out_tm[initial_idx * num_series + series] = static_cast<float>(prev_kama);

    int    trailing_idx   = first_valid;
    double trailing_value = static_cast<double>(at(prices_tm, trailing_idx, series));

    const double cmax  = kama_const_max();
    const double cdiff = kama_const_diff();

    for (int t = initial_idx + 1; t < series_len; ++t) {
        const double price         = static_cast<double>(at(prices_tm, t, series));
        const double next_trailing = static_cast<double>(at(prices_tm, trailing_idx + 1, series));

        sum_roc1 += fabs(price - prev_price) - fabs(next_trailing - trailing_value);

        trailing_value = next_trailing;
        trailing_idx  += 1;

        const double direction = fabs(price - trailing_value);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);

        double sc = er * cdiff + cmax;
        sc *= sc;

        prev_kama = fma(price - prev_kama, sc, prev_kama);
        out_tm[t * num_series + series] = static_cast<float>(prev_kama);

        prev_price = price;
    }
}
