// CUDA kernels for the WaveTrend Oscillator (WTO).
//
// The kernels operate in single precision and follow the PineScript-equivalent
// computation used by the scalar Rust implementation: ESA and D EMA recurrences
// remain sequential, while CUDA parallelism is exploited across parameter
// combinations or across independent series.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Utility to prefill output rows with NaNs.
__device__ inline void fill_nan(float* ptr, int len) {
    const float nan = nanf("");
    for (int i = 0; i < len; ++i) {
        ptr[i] = nan;
    }
}

extern "C" __global__
void wto_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ channel_lengths,
                   const int* __restrict__ average_lengths,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ wt1_out,
                   float* __restrict__ wt2_out,
                   float* __restrict__ hist_out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int chan = channel_lengths[combo];
    const int avg = average_lengths[combo];

    float* wt1_row = wt1_out + combo * series_len;
    float* wt2_row = wt2_out + combo * series_len;
    float* hist_row = hist_out + combo * series_len;

    fill_nan(wt1_row, series_len);
    fill_nan(wt2_row, series_len);
    fill_nan(hist_row, series_len);

    if (chan <= 0 || avg <= 0) {
        return;
    }

    const double alpha_ch = 2.0 / (static_cast<double>(chan) + 1.0);
    const double beta_ch = 1.0 - alpha_ch;
    const double alpha_avg = 2.0 / (static_cast<double>(avg) + 1.0);
    const double beta_avg = 1.0 - alpha_avg;

    const int start_ci = first_valid + chan - 1;
    const float nan = nanf("");

    bool esa_init = false;
    bool d_init = false;
    bool wt1_init = false;
    double esa = 0.0;
    double d = 0.0;
    double wt1_val = 0.0;

    // Maintain both a small ring buffer and streaming prefix-sum pair.
    // Using (ps - ps4) mirrors the CPU SMA implementation (prefix sums),
    // reducing tiny rounding deltas in the histogram.
    double window[4] = {0.0, 0.0, 0.0, 0.0};
    int window_count = 0;
    int window_idx = 0;
    double ps = 0.0;   // prefix sum of WT1
    double ps4 = 0.0;  // prefix sum delayed by 4 samples

    for (int t = 0; t < series_len; ++t) {
        const double price = static_cast<double>(prices[t]);
        const bool price_finite = isfinite(price);

        if (t < first_valid) {
            continue;
        }

        if (!esa_init) {
            if (!price_finite) {
                continue;
            }
            esa = price;
            esa_init = true;
        } else if (price_finite) {
            // Mirror scalar: tmp = alpha*x; esa = beta*esa + tmp (fused on the add path)
            double tmp = alpha_ch * price;
            esa = fma(beta_ch, esa, tmp);
        }

        const double diff = price - esa;
        const double abs_diff = fabs(diff);

        if (t >= start_ci) {
            if (!d_init) {
                if (isfinite(abs_diff)) {
                    d = abs_diff;
                    d_init = true;
                } else {
                    continue;
                }
            } else if (isfinite(abs_diff)) {
                double tmp = alpha_ch * abs_diff;
                d = fma(beta_ch, d, tmp);
            }

            if (!d_init) {
                continue;
            }

            const double denom = 0.015 * d;
            double ci_val = 0.0;
            bool ci_valid = false;
            if (denom != 0.0f && isfinite(denom) && price_finite) {
                ci_val = diff / denom;
                if (isfinite(ci_val)) {
                    ci_valid = true;
                }
            }

            if (!wt1_init) {
                if (!ci_valid) {
                    continue;
                }
                wt1_val = ci_val;
                wt1_init = true;
            } else if (ci_valid) {
                double tmp = alpha_avg * ci_val;
                wt1_val = fma(beta_avg, wt1_val, tmp);
            }

            if (!wt1_init) {
                continue;
            }

            wt1_row[t] = static_cast<float>(wt1_val);

            // Update ring + streaming prefix sums
            if (window_count == 4) {
                ps4 += window[window_idx];
            } else {
                window_count++;
            }
            ps += wt1_val;
            window[window_idx] = wt1_val;
            window_idx = (window_idx + 1) & 3;

            if (window_count == 4) {
                const double wt2_val = (ps - ps4) * 0.25;
                const float wt1_f = static_cast<float>(wt1_val);
                const float wt2_f = static_cast<float>(wt2_val);
                wt2_row[t] = wt2_f;
                hist_row[t] = wt1_f - wt2_f;
            }
        }
    }

    // Ensure we leave explicit NaNs where histogram never became valid
    for (int t = 0; t < series_len; ++t) {
        if (!isfinite(hist_row[t]) && !isfinite(wt1_row[t])) {
            hist_row[t] = nan;
        }
    }
}

extern "C" __global__
void wto_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int channel_length,
    int average_length,
    const int* __restrict__ first_valids,
    float* __restrict__ wt1_tm,
    float* __restrict__ wt2_tm,
    float* __restrict__ hist_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) {
        return;
    }

    float* wt1_col = wt1_tm + series;
    float* wt2_col = wt2_tm + series;
    float* hist_col = hist_tm + series;

    const float nan = nanf("");
    for (int t = 0; t < rows; ++t) {
        wt1_col[t * cols] = nan;
        wt2_col[t * cols] = nan;
        hist_col[t * cols] = nan;
    }

    if (channel_length <= 0 || average_length <= 0) {
        return;
    }

    const int first_valid = first_valids[series];
    const int start_ci = first_valid + channel_length - 1;

    const double alpha_ch = 2.0 / (static_cast<double>(channel_length) + 1.0);
    const double beta_ch = 1.0 - alpha_ch;
    const double alpha_avg = 2.0 / (static_cast<double>(average_length) + 1.0);
    const double beta_avg = 1.0 - alpha_avg;

    bool esa_init = false;
    bool d_init = false;
    bool wt1_init = false;
    double esa = 0.0;
    double d = 0.0;
    double wt1_val = 0.0;

    double window[4] = {0.0, 0.0, 0.0, 0.0};
    int window_count = 0;
    int window_idx = 0;
    double ps = 0.0;
    double ps4 = 0.0;

    for (int t = 0; t < rows; ++t) {
        const double price = static_cast<double>(prices_tm[t * cols + series]);
        const bool price_finite = isfinite(price);

        if (t < first_valid) {
            continue;
        }

        if (!esa_init) {
            if (!price_finite) {
                continue;
            }
            esa = price;
            esa_init = true;
        } else if (price_finite) {
            double tmp = alpha_ch * price;
            esa = fma(beta_ch, esa, tmp);
        }

        const double diff = price - esa;
        const double abs_diff = fabs(diff);

        if (t >= start_ci) {
            if (!d_init) {
                if (isfinite(abs_diff)) {
                    d = abs_diff;
                    d_init = true;
                } else {
                    continue;
                }
            } else if (isfinite(abs_diff)) {
                double tmp = alpha_ch * abs_diff;
                d = fma(beta_ch, d, tmp);
            }

            if (!d_init) {
                continue;
            }

            const double denom = 0.015 * d;
            double ci_val = 0.0;
            bool ci_valid = false;
            if (denom != 0.0f && isfinite(denom) && price_finite) {
                ci_val = diff / denom;
                if (isfinite(ci_val)) {
                    ci_valid = true;
                }
            }

            if (!wt1_init) {
                if (!ci_valid) {
                    continue;
                }
                wt1_val = ci_val;
                wt1_init = true;
            } else if (ci_valid) {
                double tmp = alpha_avg * ci_val;
                wt1_val = fma(beta_avg, wt1_val, tmp);
            }

            if (!wt1_init) {
                continue;
            }

            wt1_col[t * cols] = static_cast<float>(wt1_val);

            if (window_count == 4) {
                ps4 += window[window_idx];
            } else {
                window_count++;
            }
            ps += wt1_val;
            window[window_idx] = wt1_val;
            window_idx = (window_idx + 1) & 3;

            if (window_count == 4) {
                const double wt2_val = (ps - ps4) * 0.25;
                const float wt1_f = static_cast<float>(wt1_val);
                const float wt2_f = static_cast<float>(wt2_val);
                wt2_col[t * cols] = wt2_f;
                hist_col[t * cols] = wt1_f - wt2_f;
            }
        }
    }
}
