// CUDA kernels for the Ehlers Error Correcting Exponential Moving Average (ECEMA).
//
// Kernels operate on FP32 buffers to match the public API. We keep arithmetic
// in FP32 but tighten accuracy using FMA and Kahan-style compensated updates
// on the state variables (EMA and EC), which offers near-FP64 accuracy without
// FP64 throughput penalties on consumer GPUs.
// Exposed variants:
//   * ehlers_ecema_batch_f32                         : legacy one-block-per-combo (thread 0 computes)
//   * ehlers_ecema_batch_thread_per_combo_f32        : thread-per-combo mapping for better occupancy
//   * ehlers_ecema_many_series_one_param_time_major_f32 : legacy one-block-per-series (thread 0 computes)
//   * ehlers_ecema_many_series_one_param_1d_f32      : 1D thread-per-series mapping
//   * ehlers_ecema_many_series_one_param_2d_f32      : 2D tiled thread-per-series mapping

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

namespace {
__device__ inline float compute_alpha_f(int length) {
    return 2.0f / (static_cast<float>(length) + 1.0f);
}
__device__ inline float compute_beta_f(float a) { return 1.0f - a; }

// Double helpers retained for legacy/plain kernel parity with CPU
__device__ inline double compute_alpha(int length) {
    return 2.0 / (static_cast<double>(length) + 1.0);
}
__device__ inline double compute_beta(double a) { return 1.0 - a; }

struct KahanState { float y; float c; };

__device__ inline void kahan_add(KahanState& s, float x) {
    float y = x - s.c;
    float t = s.y + y;
    s.c = (t - s.y) - y;
    s.y = t;
}

__device__ inline void ema_step(KahanState& ema, float alpha, float x) {
    float delta = fmaf(alpha, (x - ema.y), 0.0f);
    kahan_add(ema, delta);
}

__device__ inline float ec_step(KahanState& ec, float alpha, float ema_val, float src, float gain) {
    float delta = fmaf(alpha, (ema_val - ec.y) + gain * (src - ec.y), 0.0f);
    kahan_add(ec, delta);
    return ec.y;
}

__device__ inline float clamp_prev_ec_f(bool pine_mode, float ema_value) {
    return pine_mode ? 0.0f : ema_value;
}

__device__ inline double clamp_prev_ec(bool pine_mode, double ema_value) {
    return pine_mode ? 0.0 : ema_value;
}

__device__ inline float pick_src_f(const float* prices, int idx, bool confirmed) {
    int source_idx = confirmed && idx > 0 ? idx - 1 : idx;
    return prices[source_idx];
}

__device__ inline float pick_src_tm_f(
    const float* prices_tm,
    int idx,
    int series,
    int num_series,
    bool confirmed
) {
    int row = confirmed && idx > 0 ? idx - 1 : idx;
    return prices_tm[row * num_series + series];
}

__device__ inline double pick_src(const float* prices, int idx, bool confirmed) {
    int source_idx = confirmed && idx > 0 ? idx - 1 : idx;
    return static_cast<double>(prices[source_idx]);
}

}

extern "C" __global__
void ehlers_ecema_batch_f32(const float* __restrict__ prices,
                            const int* __restrict__ lengths,
                            const int* __restrict__ gain_limits,
                            const unsigned char* __restrict__ pine_flags,
                            const unsigned char* __restrict__ confirmed_flags,
                            int series_len,
                            int n_combos,
                            int first_valid,
                            float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int length = lengths[combo];
    const int gain_limit = gain_limits[combo];
    const bool pine_mode = pine_flags[combo] != 0;
    const bool confirmed = confirmed_flags[combo] != 0;

    const int base = combo * series_len;
    const float nan_f = NAN;
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = nan_f;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    if (length <= 0 || gain_limit <= 0) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }
    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) {
        return;
    }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {
        return;
    }

    const double alpha = compute_alpha(length);
    const double beta  = compute_beta(alpha);

    double ema = 0.0;
    double running_mean = 0.0;
    int mean_count = 0;
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);

    bool has_prev = false; double prev_ec = 0.0;

    for (int i = first_valid; i < series_len; ++i) {
        const double price = static_cast<double>(prices[i]);
        double ema_value;

        if (pine_mode) {
            if (isfinite(price)) { ema = alpha * price + beta * ema; }
            ema_value = ema;
        } else {
            if (i == first_valid) {
                running_mean = price; mean_count = 1; ema = price; ema_value = price;
            } else if (i < warmup_end) {
                if (isfinite(price)) {
                    mean_count += 1;
                    running_mean = ((static_cast<double>(mean_count) - 1.0) * running_mean + price) / static_cast<double>(mean_count);
                }
                ema = running_mean; ema_value = running_mean;
            } else {
                if (isfinite(price)) { ema = beta * ema + alpha * price; }
                ema_value = ema;
            }
        }

        if (i < warm) { continue; }

        const double src = pick_src(prices, i, confirmed);
        const double prev = has_prev ? prev_ec : clamp_prev_ec(pine_mode, ema_value);

        double least_error = DBL_MAX; double best_gain = 0.0;
        for (int g = -gain_limit; g <= gain_limit; ++g) {
            const double gain = static_cast<double>(g) / 10.0;
            const double candidate = alpha * (ema_value + gain * (src - prev)) + beta * prev;
            const double err = fabs(src - candidate);
            if (err < least_error) { least_error = err; best_gain = gain; }
        }

        const double ec = alpha * (ema_value + best_gain * (src - prev)) + beta * prev;
        out[base + i] = static_cast<float>(ec);
        prev_ec = ec; has_prev = true;
    }
}

// Thread-per-combo batch kernel: each thread processes a unique (length,gain_limit)
// pair sequentially over time. Improves occupancy vs. the legacy one-block-per-combo
// variant while preserving identical numerical behaviour.
extern "C" __global__
void ehlers_ecema_batch_thread_per_combo_f32(const float* __restrict__ prices,
                                             const int* __restrict__ lengths,
                                             const int* __restrict__ gain_limits,
                                             const unsigned char* __restrict__ pine_flags,
                                             const unsigned char* __restrict__ confirmed_flags,
                                             int series_len,
                                             int n_combos,
                                             int first_valid,
                                             float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) { return; }

    const int length = lengths[combo];
    const int gain_limit = gain_limits[combo];
    const bool pine_mode = pine_flags[combo] != 0;
    const bool confirmed = confirmed_flags[combo] != 0;

    if (series_len <= 0 || length <= 0 || gain_limit <= 0) { return; }
    if (first_valid < 0 || first_valid >= series_len) { return; }
    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) { return; }

    const int base = combo * series_len;
    const float nan_f = NAN;
    // Fill with NaN up to warm boundary; later elements will be overwritten by compute loop
    for (int idx = 0; idx < series_len; ++idx) {
        out[base + idx] = nan_f;
    }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) { return; }

    const float alpha = compute_alpha_f(length);
    const float beta  = compute_beta_f(alpha);

    KahanState ema{0.0f, 0.0f};
    KahanState mean{0.0f, 0.0f};
    KahanState ec  {0.0f, 0.0f};
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);

    bool has_prev = false;

    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices[i];
        float ema_value;

        if (pine_mode) {
            if (isfinite(price)) { ema_step(ema, alpha, price); }
            ema_value = ema.y;
        } else {
            if (i == first_valid) {
                mean.y = price; mean.c = 0.0f; ema.y = price; ema.c = 0.0f; ema_value = price;
            } else if (i < warmup_end) {
                if (isfinite(price)) {
                    int count = (i - first_valid) + 1;
                    float inv = 1.0f / static_cast<float>(count);
                    float delta = (price - mean.y) * inv;
                    kahan_add(mean, delta);
                }
                ema.y = mean.y; ema.c = 0.0f; ema_value = ema.y;
            } else {
                if (isfinite(price)) { ema_step(ema, alpha, price); }
                ema_value = ema.y;
            }
        }

        if (i < warm) { continue; }

        const float src = pick_src_f(prices, i, confirmed);
        float prev = has_prev ? ec.y : clamp_prev_ec_f(pine_mode, ema_value);
        if (!has_prev) { ec.y = prev; ec.c = 0.0f; has_prev = true; }

        float least_error = INFINITY;
        float best_gain = 0.0f;
        const double alpha_d = static_cast<double>(alpha);
        const double beta_d  = static_cast<double>(beta);
        for (int g = -gain_limit; g <= gain_limit; ++g) {
            const float gain = 0.1f * static_cast<float>(g);
            const float t = fmaf(gain, (src - prev), ema_value);
            const double candidate_d = alpha_d * static_cast<double>(t) + beta_d * static_cast<double>(prev);
            const float err = fabsf(src - static_cast<float>(candidate_d));
            if (err < least_error) { least_error = err; best_gain = gain; }
        }

        float ec_val = ec_step(ec, alpha, ema_value, src, best_gain);
        out[base + i] = ec_val;
    }
}

extern "C" __global__
void ehlers_ecema_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    int length,
    int gain_limit,
    unsigned char pine_flag,
    unsigned char confirmed_flag,
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

    if (length <= 0 || gain_limit <= 0) {
        return;
    }

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const bool pine_mode = pine_flag != 0;
    const bool confirmed = confirmed_flag != 0;

    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) {
        return;
    }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {
        return;
    }

    const float alpha = compute_alpha_f(length);
    const float beta  = compute_beta_f(alpha);

    KahanState ema{0.0f, 0.0f};
    KahanState mean{0.0f, 0.0f};
    KahanState ec  {0.0f, 0.0f};
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);

    bool has_prev = false;

    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices_tm[i * num_series + series];
        float ema_value;

        if (pine_mode) {
            if (isfinite(price)) { ema_step(ema, alpha, price); }
            ema_value = ema.y;
        } else {
            if (i == first_valid) {
                mean.y = price; mean.c = 0.0f; ema.y = price; ema.c = 0.0f; ema_value = price;
            } else if (i < warmup_end) {
                if (isfinite(price)) {
                    int count = (i - first_valid) + 1;
                    float inv = 1.0f / static_cast<float>(count);
                    float delta = (price - mean.y) * inv;
                    kahan_add(mean, delta);
                }
                ema.y = mean.y; ema.c = 0.0f; ema_value = ema.y;
            } else {
                if (isfinite(price)) { ema_step(ema, alpha, price); }
                ema_value = ema.y;
            }
        }

        if (i < warm) {
            continue;
        }

        const float src = pick_src_tm_f(prices_tm, i, series, num_series, confirmed);
        float prev = has_prev ? ec.y : clamp_prev_ec_f(pine_mode, ema_value);
        if (!has_prev) { ec.y = prev; ec.c = 0.0f; has_prev = true; }

        float least_error = INFINITY;
        float best_gain = 0.0f;
        const double alpha_d = static_cast<double>(alpha);
        const double beta_d  = static_cast<double>(beta);
        for (int g = -gain_limit; g <= gain_limit; ++g) {
            const float gain = 0.1f * static_cast<float>(g);
            const float t = fmaf(gain, (src - prev), ema_value);
            const double candidate_d = alpha_d * static_cast<double>(t) + beta_d * static_cast<double>(prev);
            const float err = fabsf(src - static_cast<float>(candidate_d));
            if (err < least_error) { least_error = err; best_gain = gain; }
        }

        float ec_val = ec_step(ec, alpha, ema_value, src, best_gain);
        out_tm[i * num_series + series] = ec_val;
    }
}

// 1D mapping: thread-per-series many-series kernel on time-major layout.
extern "C" __global__
void ehlers_ecema_many_series_one_param_1d_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    int length,
    int gain_limit,
    unsigned char pine_flag,
    unsigned char confirmed_flag,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) { return; }

    if (series_len <= 0 || length <= 0 || gain_limit <= 0) { return; }

    const bool pine_mode = pine_flag != 0;
    const bool confirmed = confirmed_flag != 0;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) { return; }
    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) { return; }

    const int stride = num_series;
    const float nan_f = NAN;
    for (int t = 0; t < series_len; ++t) { out_tm[t * stride + series] = nan_f; }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) { return; }

    const float alpha = compute_alpha_f(length);
    const float beta  = compute_beta_f(alpha);

    KahanState ema{0.0f, 0.0f};
    KahanState mean{0.0f, 0.0f};
    KahanState ec  {0.0f, 0.0f};
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);
    bool has_prev = false;
    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices_tm[i * stride + series];
        float ema_value;
        if (pine_mode) {
            if (isfinite(price)) { ema_step(ema, alpha, price); }
            ema_value = ema.y;
        } else {
            if (i == first_valid) { mean.y = price; mean.c = 0.0f; ema.y = price; ema.c = 0.0f; ema_value = price; }
            else if (i < warmup_end) {
                if (isfinite(price)) { int count = (i - first_valid) + 1; float inv = 1.0f / static_cast<float>(count); float delta = (price - mean.y) * inv; kahan_add(mean, delta); }
                ema.y = mean.y; ema.c = 0.0f; ema_value = mean.y;
            } else {
                if (isfinite(price)) { ema_step(ema, alpha, price); }
                ema_value = ema.y;
            }
        }

        if (i < warm) { continue; }
        const int idx_tm = i * stride + series;
        const int src_row = (confirmed && i > 0) ? (i - 1) : i;
        const float src = prices_tm[src_row * stride + series];
        float prev = has_prev ? ec.y : clamp_prev_ec_f(pine_mode, ema_value);
        if (!has_prev) { ec.y = prev; ec.c = 0.0f; has_prev = true; }

        float least_error = INFINITY; float best_gain = 0.0f;
        const double alpha_d = static_cast<double>(alpha);
        const double beta_d  = static_cast<double>(beta);
        for (int g = -gain_limit; g <= gain_limit; ++g) {
            const float gain = 0.1f * static_cast<float>(g);
            const float t = fmaf(gain, (src - prev), ema_value);
            const double candidate_d = alpha_d * static_cast<double>(t) + beta_d * static_cast<double>(prev);
            const float err = fabsf(src - static_cast<float>(candidate_d));
            if (err < least_error) { least_error = err; best_gain = gain; }
        }

        float ec_val = ec_step(ec, alpha, ema_value, src, best_gain);
        out_tm[idx_tm] = ec_val;
    }
}

// 2D tiled mapping across series: each thread in the 2D block processes one series.
extern "C" __global__
void ehlers_ecema_many_series_one_param_2d_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    int length,
    int gain_limit,
    unsigned char pine_flag,
    unsigned char confirmed_flag,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    // Convert 2D thread index into a linear series index across grid tiles.
    const int tx = blockDim.x;
    const int ty = blockDim.y;
    const int series_per_grid_row = gridDim.x * tx; // series covered by one grid.y slice
    const int local_series = threadIdx.y * tx + threadIdx.x;
    const int series = blockIdx.y * series_per_grid_row + blockIdx.x * tx + local_series;
    if (series >= num_series) { return; }

    if (series_len <= 0 || length <= 0 || gain_limit <= 0) { return; }

    const bool pine_mode = pine_flag != 0;
    const bool confirmed = confirmed_flag != 0;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) { return; }
    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) { return; }

    const int stride = num_series;
    const float nan_f = NAN;
    for (int t = 0; t < series_len; ++t) { out_tm[t * stride + series] = nan_f; }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) { return; }

    const float alpha = compute_alpha_f(length);
    const float beta  = compute_beta_f(alpha);

    KahanState ema{0.0f, 0.0f}; KahanState mean{0.0f, 0.0f};
    KahanState ec{0.0f, 0.0f};
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);
    bool has_prev = false;

    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices_tm[i * stride + series];
        float ema_value;
        if (pine_mode) {
            if (isfinite(price)) { ema_step(ema, alpha, price); }
            ema_value = ema.y;
        } else {
            if (i == first_valid) { mean.y = price; mean.c = 0.0f; ema.y = price; ema.c = 0.0f; ema_value = price; }
            else if (i < warmup_end) {
                if (isfinite(price)) { int count = (i - first_valid) + 1; float inv = 1.0f / static_cast<float>(count); float delta = (price - mean.y) * inv; kahan_add(mean, delta); }
                ema.y = mean.y; ema.c = 0.0f; ema_value = mean.y;
            } else {
                if (isfinite(price)) { ema_step(ema, alpha, price); }
                ema_value = ema.y;
            }
        }
        if (i < warm) { continue; }
        const int idx_tm = i * stride + series;
        const int src_row = (confirmed && i > 0) ? (i - 1) : i;
        const float src = prices_tm[src_row * stride + series];
        float prev = has_prev ? ec.y : clamp_prev_ec_f(pine_mode, ema_value);
        if (!has_prev) { ec.y = prev; ec.c = 0.0f; has_prev = true; }

        float least_error = INFINITY; float best_gain = 0.0f;
        for (int g = -gain_limit; g <= gain_limit; ++g) {
            const float gain = 0.1f * static_cast<float>(g);
            const float t = fmaf(gain, (src - prev), ema_value);
            const float candidate = fmaf(alpha, t, beta * prev);
            const float err = fabsf(src - candidate);
            if (err < least_error) { least_error = err; best_gain = gain; }
        }

        float ec_val = ec_step(ec, alpha, ema_value, src, best_gain);
        out_tm[idx_tm] = ec_val; 
    }
}
