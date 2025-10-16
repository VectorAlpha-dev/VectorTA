// CUDA kernels for the WaveTrend indicator.
//
// Batch kernel: each thread processes one parameter combination (row) and
// streams through the time series sequentially to match the scalar CPU
// reference exactly. Outputs are laid out row-major with shape (rows, len).
//
// Many-series kernel (time-major): each thread processes one series (column)
// for a single parameter set and scans forward in time. Inputs are stored in
// time-major layout: index = t*cols + series.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline bool is_finite(float x) {
    return !isnan(x) && !isinf(x);
}
}

extern "C" __global__ void wavetrend_batch_f32(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int n_combos,
    const int* __restrict__ channel_lengths,
    const int* __restrict__ average_lengths,
    const int* __restrict__ ma_lengths,
    const float* __restrict__ factors,
    float* __restrict__ wt1_out,
    float* __restrict__ wt2_out,
    float* __restrict__ wt_diff_out
) {
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = row0; row < n_combos; row += stride) {
        const int ch = channel_lengths[row];
        const int avg = average_lengths[row];
        const int ma = ma_lengths[row];
        const double factor_d = static_cast<double>(factors[row]);

        float* wt1_row = wt1_out + static_cast<size_t>(row) * len;
        float* wt2_row = wt2_out + static_cast<size_t>(row) * len;
        float* diff_row = wt_diff_out + static_cast<size_t>(row) * len;

        // Pre-fill outputs with NaN to cover warmup and invalid regions
        for (int i = 0; i < len; ++i) {
            wt1_row[i] = CUDART_NAN_F;
            wt2_row[i] = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }

        const double alpha_ch = 2.0 / (static_cast<double>(ch) + 1.0);
        const double beta_ch = 1.0 - alpha_ch;
        const double alpha_avg = 2.0 / (static_cast<double>(avg) + 1.0);
        const double beta_avg = 1.0 - alpha_avg;
        const double inv_ma = ma > 0 ? 1.0 / static_cast<double>(ma) : 0.0;

        int warmup = first_valid + ch - 1 + avg - 1 + ma - 1;
        if (warmup < 0) {
            warmup = 0;
        }
        if (warmup > len) {
            warmup = len;
        }

        double esa_state = CUDART_NAN;
        double de_state = CUDART_NAN;
        double wt1_state = CUDART_NAN;

        double sum_wt1 = 0.0;
        int window_count = 0;

        for (int i = first_valid; i < len; ++i) {
            const double price = static_cast<double>(prices[i]);

            // Stage 1: ESA EMA(channel_lengths)
            double esa_val = CUDART_NAN;
            if (is_finite(price)) {
                if (!isfinite(esa_state)) {
                    esa_state = price;
                } else {
                    esa_state = fma(beta_ch, esa_state, alpha_ch * price);
                }
                esa_val = esa_state;
            }

            // Stage 2: DE EMA(channel_lengths) on |price - ESA|
            double absdiff = CUDART_NAN;
            if (is_finite(price) && isfinite(esa_val)) {
                absdiff = fabs(price - esa_val);
            }

            double de_val = CUDART_NAN;
            if (isfinite(absdiff)) {
                if (!isfinite(de_state)) {
                    de_state = absdiff;
                } else {
                    de_state = fma(beta_ch, de_state, alpha_ch * absdiff);
                }
                de_val = de_state;
            }

            // Stage 3: CI and WT1 = EMA(average_length)
            double ci_val = CUDART_NAN;
            if (is_finite(price) && isfinite(esa_val) && isfinite(de_val)) {
                const double denom = factor_d * de_val;
                if (denom != 0.0 && isfinite(denom)) {
                    ci_val = (price - esa_val) / denom;
                }
            }

            double wt1_val_d = CUDART_NAN;
            if (isfinite(ci_val)) {
                if (!isfinite(wt1_state)) {
                    wt1_state = ci_val;
                } else {
                    wt1_state = fma(beta_avg, wt1_state, alpha_avg * ci_val);
                }
                wt1_val_d = wt1_state;
            }

            wt1_row[i] = static_cast<float>(wt1_val_d);

            if (isfinite(wt1_val_d)) {
                sum_wt1 += wt1_val_d;
                window_count += 1;
            }

            if (ma > 0 && i >= ma) {
                const float old = wt1_row[i - ma];
                if (is_finite(old)) {
                    sum_wt1 -= old;
                    window_count -= 1;
                }
            }

            float wt2_val = CUDART_NAN_F;
            if (ma > 0 && window_count >= ma) {
                wt2_val = static_cast<float>(sum_wt1 * inv_ma);
            }
            wt2_row[i] = wt2_val;

            if (i >= warmup && isfinite(wt2_val) && isfinite(static_cast<float>(wt1_val_d))) {
                diff_row[i] = wt2_val - static_cast<float>(wt1_val_d);
            } else {
                diff_row[i] = CUDART_NAN_F;
            }
        }

        for (int i = 0; i < warmup; ++i) {
            wt1_row[i] = CUDART_NAN_F;
            wt2_row[i] = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }
    }
}

// Many-series × one-param (time-major layout)
//
// Arg order mirrors other wrappers (e.g., WTO):
//   prices_tm: time-major input of shape (rows, cols)
//   cols: number of series (columns)
//   rows: number of timesteps (rows)
//   channel_length, average_length, ma_length, factor: parameters
//   first_valids: per-series first finite index (len = cols)
//   wt1_tm, wt2_tm, wt_diff_tm: time-major outputs of shape (rows, cols)
extern "C" __global__ void wavetrend_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int channel_length,
    int average_length,
    int ma_length,
    float factor,
    const int* __restrict__ first_valids,
    float* __restrict__ wt1_tm,
    float* __restrict__ wt2_tm,
    float* __restrict__ wt_diff_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) {
        return;
    }

    float* wt1_col = wt1_tm + series;
    float* wt2_col = wt2_tm + series;
    float* diff_col = wt_diff_tm + series;

    // Prefill with NaNs
    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols;
        wt1_col[idx] = CUDART_NAN_F;
        wt2_col[idx] = CUDART_NAN_F;
        diff_col[idx] = CUDART_NAN_F;
    }

    if (channel_length <= 0 || average_length <= 0 || ma_length <= 0) {
        return;
    }

    const int first_valid = first_valids[series];
    const int start_ci = first_valid + channel_length - 1;
    const int warmup = first_valid + channel_length - 1 + average_length - 1 + ma_length - 1;

    const double alpha_ch = 2.0 / (static_cast<double>(channel_length) + 1.0);
    const double beta_ch = 1.0 - alpha_ch;
    const double alpha_avg = 2.0 / (static_cast<double>(average_length) + 1.0);
    const double beta_avg = 1.0 - alpha_avg;
    const double inv_ma = 1.0 / static_cast<double>(ma_length);
    const double factor_d = static_cast<double>(factor);

    bool esa_init = false;
    bool de_init = false;
    bool wt1_init = false;
    double esa = 0.0;
    double de = 0.0;
    double wt1_state = 0.0;

    double sum_wt1 = 0.0;
    int window_count = 0;

    for (int t = first_valid; t < rows; ++t) {
        const double price = static_cast<double>(prices_tm[t * cols + series]);
        const bool price_finite = isfinite(price);

        // ESA EMA(channel_length)
        if (!esa_init) {
            if (!price_finite) {
                continue;
            }
            esa = price;
            esa_init = true;
        } else if (price_finite) {
            esa = fma(alpha_ch, price, beta_ch * esa);
        }

        // DE EMA(channel_length) on |price - ESA|
        const double absdiff = fabs(price - esa);
        if (!de_init) {
            if (isfinite(absdiff)) {
                de = absdiff;
                de_init = true;
            } else {
                continue;
            }
        } else if (isfinite(absdiff)) {
            de = fma(alpha_ch, absdiff, beta_ch * de);
        }

        if (!de_init) {
            continue;
        }

        // CI and WT1 EMA(average_length)
        const double denom = factor_d * de;
        double ci_val = NAN;
        if (denom != 0.0 && isfinite(denom) && price_finite) {
            ci_val = (price - esa) / denom;
        }
        if (!wt1_init) {
            if (isfinite(ci_val)) {
                wt1_state = ci_val;
                wt1_init = true;
            }
        } else if (isfinite(ci_val)) {
            wt1_state = fma(alpha_avg, ci_val, beta_avg * wt1_state);
        }

        const int idx = t * cols;
        float wt1_val_f = CUDART_NAN_F;
        if (wt1_init) {
            wt1_val_f = static_cast<float>(wt1_state);
        }

        // Maintain rolling window counters regardless; commit outputs only after warmup
        if (isfinite(static_cast<double>(wt1_val_f))) {
            sum_wt1 += wt1_state;
            window_count += 1;
        }
        if (t >= ma_length) {
            const float old = wt1_col[(t - ma_length) * cols];
            if (isfinite(static_cast<double>(old))) {
                sum_wt1 -= static_cast<double>(old);
                window_count -= 1;
            }
        }

        // Write current step values; clean warmup prefix afterwards
        wt1_col[idx] = wt1_val_f;
        float wt2_val_f = CUDART_NAN_F;
        if (window_count >= ma_length) {
            wt2_val_f = static_cast<float>(sum_wt1 * inv_ma);
        }
        wt2_col[idx] = wt2_val_f;
        if (t >= warmup && isfinite(static_cast<double>(wt1_val_f)) && isfinite(static_cast<double>(wt2_val_f))) {
            diff_col[idx] = wt2_val_f - wt1_val_f;
        } else {
            diff_col[idx] = CUDART_NAN_F;
        }
    }

    // Explicitly clear warmup prefix to NaNs to mirror scalar semantics
    for (int t = 0; t < rows && t < warmup; ++t) {
        const int idx = t * cols;
        wt1_col[idx] = CUDART_NAN_F;
        wt2_col[idx] = CUDART_NAN_F;
        diff_col[idx] = CUDART_NAN_F;
    }
}
