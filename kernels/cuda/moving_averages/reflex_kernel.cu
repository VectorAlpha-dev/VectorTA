// CUDA kernels for the Reflex indicator.
//
// Each parameter combination (period) is assigned to a dedicated block that
// processes the entire series sequentially. The recurrence only depends on the
// last `period` smoothed values, so we keep a circular buffer in shared memory
// to avoid additional global allocations while still matching the scalar
// implementation's warmup semantics (first `period` outputs = 0.0, remainder
// initialised to NaN).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

constexpr float REFLEX_PI_F = 3.14159265358979323846f;
constexpr float REFLEX_SQRT2_F = 1.4142135623730951f;

static __device__ __forceinline__ float reflex_compute_ssf(float price,
                                                           float prev_price,
                                                           float c,
                                                           float b,
                                                           float a_sq,
                                                           float prev_ssf1,
                                                           float prev_ssf2) {
    const double val = static_cast<double>(c) *
            (static_cast<double>(price) + static_cast<double>(prev_price)) +
        static_cast<double>(b) * static_cast<double>(prev_ssf1) -
        static_cast<double>(a_sq) * static_cast<double>(prev_ssf2);
    return static_cast<float>(val);
}

static __device__ __forceinline__ bool reflex_isfinite(float v) {
    return !(isnan(v) || isinf(v));
}

extern "C" __global__
void reflex_batch_f32(const float* __restrict__ prices,
                      const int* __restrict__ periods,
                      int series_len,
                      int n_combos,
                      int first_valid,
                      float* __restrict__ out) {
    int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    int period = periods[combo];
    if (period < 2 || series_len <= 0) {
        return;
    }

    float* out_row = out + combo * series_len;
    const float nan = nanf("");
    for (int i = 0; i < series_len; ++i) {
        out_row[i] = nan;
    }
    int warm = period < series_len ? period : series_len;
    for (int i = 0; i < warm; ++i) {
        out_row[i] = 0.0f;
    }

    int half_period_i = period / 2;
    if (half_period_i < 1) {
        half_period_i = 1;
    }
    const float half_period = static_cast<float>(half_period_i);
    const float a = __expf(-REFLEX_SQRT2_F * REFLEX_PI_F / half_period);
    const float a_sq = a * a;
    const float b = 2.0f * a * __cosf(REFLEX_SQRT2_F * REFLEX_PI_F / half_period);
    const float c = 0.5f * (1.0f + a_sq - b);
    int start = first_valid;
    if (start < 0) {
        start = 0;
    }
    if (start >= series_len) {
        return;
    }

    extern __shared__ float history[];
    for (int i = 0; i < period; ++i) {
        history[i] = nanf("");
    }

    if (start > period) {
        for (int i = period; i < start && i < series_len; ++i) {
            out_row[i] = 0.0f;
        }
    }

    double ms_prev = 0.0;
    float prev_ssf1 = 0.0f;
    float prev_ssf2 = 0.0f;
    int finite_ssf_count = 0; // saturates at 2; tracks usable history for recurrence

    int window_head = 0; // points to oldest element in `history`
    int window_count = 0; // number of elements stored (<= period)
    const double inv_period_d = 1.0 / static_cast<double>(period);

    for (int i = start; i < series_len; ++i) {
        const float price = prices[i];

        if (!reflex_isfinite(price)) {
            if (period > 0) {
                if (window_count < period) {
                    int tail = (window_head + window_count) % period;
                    history[tail] = nanf("");
                    window_count += 1;
                } else {
                    history[window_head] = nanf("");
                    window_head = (window_head + 1) % period;
                }
            }
            ms_prev = 0.0;
            continue;
        }

        float prev_price = (i == 0) ? price : prices[i - 1];

        float prev1 = (finite_ssf_count >= 1 && reflex_isfinite(prev_ssf1)) ? prev_ssf1 : price;
        float prev2 = (finite_ssf_count >= 2 && reflex_isfinite(prev_ssf2)) ? prev_ssf2 : prev1;

        float ssf_i = (finite_ssf_count < 2)
            ? price
            : reflex_compute_ssf(price, prev_price, c, b, a_sq, prev1, prev2);

        bool ssf_ok = reflex_isfinite(ssf_i);
        bool updated_ms = false;

        if (period > 0 && window_count == period && ssf_ok) {
            const float ssf_period = history[window_head];
            if (reflex_isfinite(ssf_period)) {
                double slope = (static_cast<double>(ssf_period) - static_cast<double>(ssf_i)) * inv_period_d;
                double my_sum = 0.0;
                bool valid = true;
                for (int t = 1; t <= period; ++t) {
                    int idx = window_head + period - t;
                    if (idx >= period) {
                        idx -= period;
                    }
                    const float past = history[idx];
                    if (!reflex_isfinite(past)) {
                        valid = false;
                        break;
                    }
                    const double pred = static_cast<double>(ssf_i) + slope * static_cast<double>(t);
                    my_sum += pred - static_cast<double>(past);
                }

                if (valid) {
                    my_sum *= inv_period_d;
                    double ms = 0.04 * my_sum * my_sum + 0.96 * ms_prev;
                    if (ms > 0.0 && isfinite(ms)) {
                        out_row[i] = static_cast<float>(my_sum / sqrt(ms));
                        ms_prev = ms;
                    } else {
                        ms_prev = 0.0;
                    }
                    updated_ms = true;
                }
            }
        }

        if (!updated_ms) {
            if (window_count < period || !ssf_ok) {
                ms_prev = 0.0;
            }
        }

        const float stored_value = ssf_ok ? ssf_i : nanf("");
        if (period > 0) {
            if (window_count < period) {
                int tail = (window_head + window_count) % period;
                history[tail] = stored_value;
                window_count += 1;
            } else {
                history[window_head] = stored_value;
                window_head = (window_head + 1) % period;
            }
        }

        if (ssf_ok) {
            if (finite_ssf_count >= 1) {
                prev_ssf2 = prev_ssf1;
            }
            prev_ssf1 = ssf_i;
            if (finite_ssf_count < 2) {
                finite_ssf_count += 1;
            }
        }
    }
}

extern "C" __global__
void reflex_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                      int period,
                                      int num_series,
                                      int series_len,
                                      const int* __restrict__ first_valids,
                                      float* __restrict__ out_tm) {
    int series = blockIdx.x;
    if (series >= num_series) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    if (period < 2 || series_len <= 0) {
        return;
    }

    const float nan = nanf("");
    for (int t = 0; t < series_len; ++t) {
        out_tm[t * num_series + series] = nan;
    }
    int warm = period < series_len ? period : series_len;
    for (int t = 0; t < warm; ++t) {
        out_tm[t * num_series + series] = 0.0f;
    }

    int half_period_i = period / 2;
    if (half_period_i < 1) {
        half_period_i = 1;
    }
    const float half_period = static_cast<float>(half_period_i);
    const float a = __expf(-REFLEX_SQRT2_F * REFLEX_PI_F / half_period);
    const float a_sq = a * a;
    const float b = 2.0f * a * __cosf(REFLEX_SQRT2_F * REFLEX_PI_F / half_period);
    const float c = 0.5f * (1.0f + a_sq - b);
    int start = first_valids[series];
    if (start < 0) {
        start = 0;
    }
    if (start >= series_len) {
        return;
    }

    extern __shared__ float history[];
    for (int i = 0; i < period; ++i) {
        history[i] = nanf("");
    }

    if (start > period) {
        for (int t = period; t < start && t < series_len; ++t) {
            out_tm[t * num_series + series] = 0.0f;
        }
    }

    double ms_prev = 0.0;
    float prev_ssf1 = 0.0f;
    float prev_ssf2 = 0.0f;
    int finite_ssf_count = 0;
    int window_head = 0;
    int window_count = 0;
    const double inv_period_d = 1.0 / static_cast<double>(period);

    for (int t = start; t < series_len; ++t) {
        const int idx = t * num_series + series;
        const float price = prices_tm[idx];

        if (!reflex_isfinite(price)) {
            if (period > 0) {
                if (window_count < period) {
                    int tail = (window_head + window_count) % period;
                    history[tail] = nanf("");
                    window_count += 1;
                } else {
                    history[window_head] = nanf("");
                    window_head = (window_head + 1) % period;
                }
            }
            ms_prev = 0.0;
            continue;
        }

        float prev_price = (t == 0) ? price : prices_tm[(t - 1) * num_series + series];

        float prev1 = (finite_ssf_count >= 1 && reflex_isfinite(prev_ssf1)) ? prev_ssf1 : price;
        float prev2 = (finite_ssf_count >= 2 && reflex_isfinite(prev_ssf2)) ? prev_ssf2 : prev1;

        float ssf_t = (finite_ssf_count < 2)
            ? price
            : reflex_compute_ssf(price, prev_price, c, b, a_sq, prev1, prev2);

        bool ssf_ok = reflex_isfinite(ssf_t);
        bool updated_ms = false;

        if (period > 0 && window_count == period && ssf_ok) {
            const float ssf_period = history[window_head];
            if (reflex_isfinite(ssf_period)) {
                double slope = (static_cast<double>(ssf_period) - static_cast<double>(ssf_t)) * inv_period_d;
                double my_sum = 0.0;
                bool valid = true;
                for (int k = 1; k <= period; ++k) {
                    int idx_hist = window_head + period - k;
                    if (idx_hist >= period) {
                        idx_hist -= period;
                    }
                    const float past = history[idx_hist];
                    if (!reflex_isfinite(past)) {
                        valid = false;
                        break;
                    }
                    const double pred = static_cast<double>(ssf_t) + slope * static_cast<double>(k);
                    my_sum += pred - static_cast<double>(past);
                }

                if (valid) {
                    my_sum *= inv_period_d;
                    double ms = 0.04 * my_sum * my_sum + 0.96 * ms_prev;
                    if (ms > 0.0 && isfinite(ms)) {
                        out_tm[idx] = static_cast<float>(my_sum / sqrt(ms));
                        ms_prev = ms;
                    } else {
                        ms_prev = 0.0;
                    }
                    updated_ms = true;
                }
            }
        }

        if (!updated_ms) {
            if (window_count < period || !ssf_ok) {
                ms_prev = 0.0;
            }
        }

        const float stored_value = ssf_ok ? ssf_t : nanf("");
        if (period > 0) {
            if (window_count < period) {
                int tail = (window_head + window_count) % period;
                history[tail] = stored_value;
                window_count += 1;
            } else {
                history[window_head] = stored_value;
                window_head = (window_head + 1) % period;
            }
        }

        if (ssf_ok) {
            if (finite_ssf_count >= 1) {
                prev_ssf2 = prev_ssf1;
            }
            prev_ssf1 = ssf_t;
            if (finite_ssf_count < 2) {
                finite_ssf_count += 1;
            }
        }
    }
}
