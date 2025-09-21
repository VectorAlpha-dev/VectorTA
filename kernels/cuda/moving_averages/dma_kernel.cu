// CUDA kernel for DMA (Dickson Moving Average) computations.
//
// This kernel mirrors the scalar DMA implementation but executes each parameter
// combination on the GPU. Each block processes one parameter set and walks the
// series sequentially, keeping fidelity with the adaptive EMA search while
// still benefiting from device-side residency for large batch sweeps.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

static __device__ __forceinline__ float weighted_sum_norm(int p) {
    return 0.5f * static_cast<float>(p) * static_cast<float>(p + 1);
}

static __device__ __forceinline__ void kahan_add(float value, float& sum, float& comp) {
    float y = value - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}

extern "C" __global__
void dma_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ hull_lengths,
                   const int* __restrict__ ema_lengths,
                   const int* __restrict__ ema_gain_limits,
                   const int* __restrict__ hull_types,  // 0 = WMA, 1 = EMA
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    // Single-thread execution per combo keeps the logic identical to the scalar path.
    if (threadIdx.x != 0) {
        return;
    }

    const int hull_length = hull_lengths[combo];
    const int ema_length = ema_lengths[combo];
    const int ema_gain_limit = ema_gain_limits[combo];
    const int hull_type = hull_types[combo];

    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0) {
        return;
    }

    const int half = hull_length / 2;
    const int sqrt_len = static_cast<int>(floorf(sqrtf(static_cast<float>(hull_length)) + 0.5f));

    const int base_out = combo * series_len;
    for (int i = 0; i < series_len; ++i) {
        out[base_out + i] = NAN;
    }

    if (first_valid >= series_len) {
        return;
    }

    // Shared memory scratch ring for diff smoothing.
    extern __shared__ float shared[];
    float* diff_ring = shared;

    const float alpha_e = 2.0f / (static_cast<float>(ema_length) + 1.0f);
    const int i0_e = first_valid + (ema_length > 0 ? ema_length - 1 : 0);

    float e0_prev = 0.0f;
    bool e0_init_done = false;
    float ec_prev = 0.0f;
    bool ec_init_done = false;

    const int i0_half = first_valid + (half > 0 ? half - 1 : 0);
    const int i0_full = first_valid + (hull_length > 0 ? hull_length - 1 : 0);

    float a_half = 0.0f;
    float s_half = 0.0f;
    float a_half_c = 0.0f;
    float s_half_c = 0.0f;
    bool half_ready = false;

    float a_full = 0.0f;
    float s_full = 0.0f;
    float a_full_c = 0.0f;
    float s_full_c = 0.0f;
    bool full_ready = false;

    int diff_filled = 0;
    int diff_pos = 0;
    float diff_sum_seed = 0.0f;
    float diff_sum_seed_c = 0.0f;

    float a_diff = 0.0f;
    float s_diff = 0.0f;
    float a_diff_c = 0.0f;
    float s_diff_c = 0.0f;
    bool diff_wma_init_done = false;

    float diff_ema = 0.0f;
    bool diff_ema_init_done = false;
    const float alpha_sqrt = (sqrt_len > 0)
        ? 2.0f / (static_cast<float>(sqrt_len) + 1.0f)
        : 0.0f;

    float e_half_prev = 0.0f;
    float e_full_prev = 0.0f;
    bool e_half_init_done = false;
    bool e_full_init_done = false;
    const float alpha_half = (half > 0)
        ? 2.0f / (static_cast<float>(half) + 1.0f)
        : 0.0f;
    const float alpha_full = (hull_length > 0)
        ? 2.0f / (static_cast<float>(hull_length) + 1.0f)
        : 0.0f;

    const bool is_wma = (hull_type == 0);
    float hull_val = NAN;

    for (int i = first_valid; i < series_len; ++i) {
        const float x = prices[i];

        if (!e0_init_done) {
            if (i >= i0_e) {
                int start = i + 1 - ema_length;
                float sum = 0.0f;
                for (int k = start; k <= i; ++k) {
                    sum += prices[k];
                }
                e0_prev = sum / static_cast<float>(ema_length);
                e0_init_done = true;
            }
        } else {
            e0_prev = alpha_e * x + (1.0f - alpha_e) * e0_prev;
        }

        float diff_now = NAN;

        if (is_wma) {
            if (half > 0) {
                if (!half_ready) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        float wsum_local = 0.0f;
                        float wsum_c = 0.0f;
                        for (int j = 0; j < half; ++j) {
                            const int idx = start + j;
                            const float w = static_cast<float>(j + 1);
                            const float v = prices[idx];
                            kahan_add(v, sum, sum_c);
                            kahan_add(w * v, wsum_local, wsum_c);
                        }
                        a_half = sum;
                        s_half = wsum_local;
                        a_half_c = sum_c;
                        s_half_c = wsum_c;
                        half_ready = true;
                    }
                } else {
                    const float a_prev = a_half;
                    const float old = prices[i - half];
                    kahan_add(x, a_half, a_half_c);
                    kahan_add(-old, a_half, a_half_c);
                    kahan_add(static_cast<float>(half) * x, s_half, s_half_c);
                    kahan_add(-a_prev, s_half, s_half_c);
                }
            }

            if (hull_length > 0) {
                if (!full_ready) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        float wsum_local = 0.0f;
                        float wsum_c = 0.0f;
                        for (int j = 0; j < hull_length; ++j) {
                            const int idx = start + j;
                            const float w = static_cast<float>(j + 1);
                            const float v = prices[idx];
                            kahan_add(v, sum, sum_c);
                            kahan_add(w * v, wsum_local, wsum_c);
                        }
                        a_full = sum;
                        s_full = wsum_local;
                        a_full_c = sum_c;
                        s_full_c = wsum_c;
                        full_ready = true;
                    }
                } else {
                    const float a_prev = a_full;
                    const float old = prices[i - hull_length];
                    kahan_add(x, a_full, a_full_c);
                    kahan_add(-old, a_full, a_full_c);
                    kahan_add(static_cast<float>(hull_length) * x, s_full, s_full_c);
                    kahan_add(-a_prev, s_full, s_full_c);
                }
            }

            if (half_ready && full_ready) {
                const float denom_half = fmaxf(weighted_sum_norm(half), 1.0f);
                const float denom_full = fmaxf(weighted_sum_norm(hull_length), 1.0f);
                const float w_half = s_half / denom_half;
                const float w_full = s_full / denom_full;
                diff_now = 2.0f * w_half - w_full;
            }
        } else {
            if (half > 0) {
                if (!e_half_init_done) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) {
                            kahan_add(prices[k], sum, sum_c);
                        }
                        e_half_prev = sum / static_cast<float>(half);
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = alpha_half * x + (1.0f - alpha_half) * e_half_prev;
                }
            }

            if (hull_length > 0) {
                if (!e_full_init_done) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) {
                            kahan_add(prices[k], sum, sum_c);
                        }
                        e_full_prev = sum / static_cast<float>(hull_length);
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = alpha_full * x + (1.0f - alpha_full) * e_full_prev;
                }
            }

            if (e_half_init_done && e_full_init_done) {
                diff_now = 2.0f * e_half_prev - e_full_prev;
            }
        }

        if (!isnan(diff_now) && sqrt_len > 0) {
            if (diff_filled < sqrt_len) {
                diff_ring[diff_filled] = diff_now;
                kahan_add(diff_now, diff_sum_seed, diff_sum_seed_c);
                diff_filled += 1;

                if (diff_filled == sqrt_len) {
                    if (is_wma) {
                        a_diff = 0.0f;
                        s_diff = 0.0f;
                        a_diff_c = 0.0f;
                        s_diff_c = 0.0f;
                        for (int j = 0; j < sqrt_len; ++j) {
                            const float w = static_cast<float>(j + 1);
                            const float v = diff_ring[j];
                            kahan_add(v, a_diff, a_diff_c);
                            kahan_add(w * v, s_diff, s_diff_c);
                        }
                        diff_wma_init_done = true;
                        const float denom = fmaxf(weighted_sum_norm(sqrt_len), 1.0f);
                        hull_val = s_diff / denom;
                    } else {
                        diff_ema = diff_sum_seed / static_cast<float>(sqrt_len);
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos = (diff_pos + 1) % sqrt_len;

                if (is_wma) {
                    if (!diff_wma_init_done) {
                        diff_wma_init_done = true;
                    }
                    const float a_prev = a_diff;
                    kahan_add(diff_now, a_diff, a_diff_c);
                    kahan_add(-old, a_diff, a_diff_c);
                    kahan_add(static_cast<float>(sqrt_len) * diff_now, s_diff, s_diff_c);
                    kahan_add(-a_prev, s_diff, s_diff_c);
                    const float denom = fmaxf(weighted_sum_norm(sqrt_len), 1.0f);
                    hull_val = s_diff / denom;
                } else {
                    if (!diff_ema_init_done) {
                        diff_ema = diff_now;
                        diff_ema_init_done = true;
                    } else {
                        diff_ema = alpha_sqrt * diff_now + (1.0f - alpha_sqrt) * diff_ema;
                    }
                    hull_val = diff_ema;
                }
            }
        }

        float ec_now = NAN;
        if (e0_init_done) {
            if (!ec_init_done) {
                ec_prev = e0_prev;
                ec_now = ec_prev;
                ec_init_done = true;
            } else {
                float least_error = FLT_MAX;
                float best_gain = 0.0f;
                for (int gain_i = 0; gain_i <= ema_gain_limit; ++gain_i) {
                    const float g = static_cast<float>(gain_i) / 10.0f;
                    const float pred = alpha_e * (e0_prev + g * (x - ec_prev))
                        + (1.0f - alpha_e) * ec_prev;
                    const float err = fabsf(x - pred);
                    if (err < least_error) {
                        least_error = err;
                        best_gain = g;
                    }
                }
                ec_now = alpha_e * (e0_prev + best_gain * (x - ec_prev))
                    + (1.0f - alpha_e) * ec_prev;
                ec_prev = ec_now;
            }
        }

        if (!isnan(hull_val) && !isnan(ec_now)) {
            out[base_out + i] = 0.5f * (hull_val + ec_now);
        }
    }
}

// Many-series × one-parameter variant. Each block handles one series with
// time-major layout (rows = time, cols = series). The implementation mirrors
// the single-series kernel but indexes into the matrix using the series stride.
extern "C" __global__
void dma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   int hull_length,
                                   int ema_length,
                                   int ema_gain_limit,
                                   int hull_type,  // 0 = WMA, 1 = EMA
                                   int series_len,
                                   int num_series,
                                   const int* __restrict__ first_valids,
                                   int sqrt_len,
                                   float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0) {
        return;
    }

    const int stride = num_series;
    const int base_out = series_idx;

    if (threadIdx.x == 0) {
        for (int i = 0; i < series_len; ++i) {
            out_tm[base_out + i * stride] = NAN;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    const int first_valid = first_valids[series_idx];
    if (first_valid >= series_len) {
        return;
    }

    extern __shared__ float diff_ring[];

    const int half = hull_length / 2;
    const int sqrt_len_clamped = (sqrt_len > 0) ? sqrt_len : 1;

    const float alpha_e = 2.0f / (static_cast<float>(ema_length) + 1.0f);
    const int i0_e = first_valid + (ema_length > 0 ? ema_length - 1 : 0);

    float e0_prev = 0.0f;
    bool e0_init_done = false;
    float ec_prev = 0.0f;
    bool ec_init_done = false;

    const int i0_half = first_valid + (half > 0 ? half - 1 : 0);
    const int i0_full = first_valid + (hull_length > 0 ? hull_length - 1 : 0);

    float a_half = 0.0f;
    float s_half = 0.0f;
    float a_half_c = 0.0f;
    float s_half_c = 0.0f;
    bool half_ready = false;

    float a_full = 0.0f;
    float s_full = 0.0f;
    float a_full_c = 0.0f;
    float s_full_c = 0.0f;
    bool full_ready = false;

    int diff_filled = 0;
    int diff_pos = 0;
    float diff_sum_seed = 0.0f;
    float diff_sum_seed_c = 0.0f;

    float a_diff = 0.0f;
    float s_diff = 0.0f;
    float a_diff_c = 0.0f;
    float s_diff_c = 0.0f;
    bool diff_wma_init_done = false;

    float diff_ema = 0.0f;
    bool diff_ema_init_done = false;
    const float alpha_sqrt = (sqrt_len_clamped > 0)
        ? 2.0f / (static_cast<float>(sqrt_len_clamped) + 1.0f)
        : 0.0f;

    float e_half_prev = 0.0f;
    float e_full_prev = 0.0f;
    bool e_half_init_done = false;
    bool e_full_init_done = false;
    const float alpha_half = (half > 0)
        ? 2.0f / (static_cast<float>(half) + 1.0f)
        : 0.0f;
    const float alpha_full = (hull_length > 0)
        ? 2.0f / (static_cast<float>(hull_length) + 1.0f)
        : 0.0f;

    const bool is_wma = (hull_type == 0);
    float hull_val = NAN;

    for (int i = first_valid; i < series_len; ++i) {
        const int idx = i * stride + series_idx;
        const float x = prices_tm[idx];

        if (!e0_init_done) {
            if (i >= i0_e) {
                int start = i + 1 - ema_length;
                float sum = 0.0f;
                float sum_c = 0.0f;
                for (int k = start; k <= i; ++k) {
                    kahan_add(prices_tm[k * stride + series_idx], sum, sum_c);
                }
                e0_prev = sum / static_cast<float>(ema_length);
                e0_init_done = true;
            }
        } else {
            e0_prev = alpha_e * x + (1.0f - alpha_e) * e0_prev;
        }

        float diff_now = NAN;

        if (is_wma) {
            if (half > 0) {
                if (!half_ready) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        float wsum_local = 0.0f;
                        float wsum_c = 0.0f;
                        for (int j = 0; j < half; ++j) {
                            const int sidx = start + j;
                            const float w = static_cast<float>(j + 1);
                            const float v = prices_tm[sidx * stride + series_idx];
                            kahan_add(v, sum, sum_c);
                            kahan_add(w * v, wsum_local, wsum_c);
                        }
                        a_half = sum;
                        s_half = wsum_local;
                        a_half_c = sum_c;
                        s_half_c = wsum_c;
                        half_ready = true;
                    }
                } else {
                    const float a_prev = a_half;
                    const float old = prices_tm[(i - half) * stride + series_idx];
                    kahan_add(x, a_half, a_half_c);
                    kahan_add(-old, a_half, a_half_c);
                    kahan_add(static_cast<float>(half) * x, s_half, s_half_c);
                    kahan_add(-a_prev, s_half, s_half_c);
                }
            }

            if (hull_length > 0) {
                if (!full_ready) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        float wsum_local = 0.0f;
                        float wsum_c = 0.0f;
                        for (int j = 0; j < hull_length; ++j) {
                            const int sidx = start + j;
                            const float w = static_cast<float>(j + 1);
                            const float v = prices_tm[sidx * stride + series_idx];
                            kahan_add(v, sum, sum_c);
                            kahan_add(w * v, wsum_local, wsum_c);
                        }
                        a_full = sum;
                        s_full = wsum_local;
                        a_full_c = sum_c;
                        s_full_c = wsum_c;
                        full_ready = true;
                    }
                } else {
                    const float a_prev = a_full;
                    const float old = prices_tm[(i - hull_length) * stride + series_idx];
                    kahan_add(x, a_full, a_full_c);
                    kahan_add(-old, a_full, a_full_c);
                    kahan_add(static_cast<float>(hull_length) * x, s_full, s_full_c);
                    kahan_add(-a_prev, s_full, s_full_c);
                }
            }

            if (half_ready && full_ready) {
                const float denom_half = fmaxf(weighted_sum_norm(half), 1.0f);
                const float denom_full = fmaxf(weighted_sum_norm(hull_length), 1.0f);
                const float w_half = s_half / denom_half;
                const float w_full = s_full / denom_full;
                diff_now = 2.0f * w_half - w_full;
            }
        } else {
            if (half > 0) {
                if (!e_half_init_done) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) {
                            kahan_add(prices_tm[k * stride + series_idx], sum, sum_c);
                        }
                        e_half_prev = sum / static_cast<float>(half);
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = alpha_half * x + (1.0f - alpha_half) * e_half_prev;
                }
            }

            if (hull_length > 0) {
                if (!e_full_init_done) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) {
                            kahan_add(prices_tm[k * stride + series_idx], sum, sum_c);
                        }
                        e_full_prev = sum / static_cast<float>(hull_length);
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = alpha_full * x + (1.0f - alpha_full) * e_full_prev;
                }
            }

            if (e_half_init_done && e_full_init_done) {
                diff_now = 2.0f * e_half_prev - e_full_prev;
            }
        }

        if (!isnan(diff_now) && sqrt_len_clamped > 0) {
            if (diff_filled < sqrt_len_clamped) {
                diff_ring[diff_filled] = diff_now;
                kahan_add(diff_now, diff_sum_seed, diff_sum_seed_c);
                diff_filled += 1;

                if (diff_filled == sqrt_len_clamped) {
                    if (is_wma) {
                        a_diff = 0.0f;
                        s_diff = 0.0f;
                        a_diff_c = 0.0f;
                        s_diff_c = 0.0f;
                        for (int j = 0; j < sqrt_len_clamped; ++j) {
                            const float w = static_cast<float>(j + 1);
                            const float v = diff_ring[j];
                            kahan_add(v, a_diff, a_diff_c);
                            kahan_add(w * v, s_diff, s_diff_c);
                        }
                        diff_wma_init_done = true;
                        const float denom = fmaxf(weighted_sum_norm(sqrt_len_clamped), 1.0f);
                        hull_val = s_diff / denom;
                    } else {
                        diff_ema = diff_sum_seed / static_cast<float>(sqrt_len_clamped);
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos = (diff_pos + 1) % sqrt_len_clamped;

                if (is_wma) {
                    if (!diff_wma_init_done) {
                        diff_wma_init_done = true;
                    }
                    const float a_prev = a_diff;
                    kahan_add(diff_now, a_diff, a_diff_c);
                    kahan_add(-old, a_diff, a_diff_c);
                    kahan_add(static_cast<float>(sqrt_len_clamped) * diff_now, s_diff, s_diff_c);
                    kahan_add(-a_prev, s_diff, s_diff_c);
                    const float denom = fmaxf(weighted_sum_norm(sqrt_len_clamped), 1.0f);
                    hull_val = s_diff / denom;
                } else {
                    if (!diff_ema_init_done) {
                        diff_ema = diff_now;
                        diff_ema_init_done = true;
                    } else {
                        diff_ema = alpha_sqrt * diff_now + (1.0f - alpha_sqrt) * diff_ema;
                    }
                    hull_val = diff_ema;
                }
            }
        }

        float ec_now = NAN;
        if (e0_init_done) {
            if (!ec_init_done) {
                ec_prev = e0_prev;
                ec_now = ec_prev;
                ec_init_done = true;
            } else {
                float least_error = FLT_MAX;
                float best_gain = 0.0f;
                for (int gain_i = 0; gain_i <= ema_gain_limit; ++gain_i) {
                    const float g = static_cast<float>(gain_i) / 10.0f;
                    const float pred = alpha_e * (e0_prev + g * (x - ec_prev))
                        + (1.0f - alpha_e) * ec_prev;
                    const float err = fabsf(x - pred);
                    if (err < least_error) {
                        least_error = err;
                        best_gain = g;
                    }
                }
                ec_now = alpha_e * (e0_prev + best_gain * (x - ec_prev))
                    + (1.0f - alpha_e) * ec_prev;
                ec_prev = ec_now;
            }
        }

        if (!isnan(hull_val) && !isnan(ec_now)) {
            out_tm[base_out + i * stride] = 0.5f * (hull_val + ec_now);
        }
    }
}
