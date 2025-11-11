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

static __device__ __forceinline__ float sum_read(float s, float c) { return s + c; }

static __device__ __forceinline__ float wsum_norm_i32(int p) {
    long long t = (long long)p * (p + 1);
    // exact divide-by-2 for integers; convert with round-to-nearest
    return __int2float_rn((int)(t >> 1));
}

// legacy weighted_sum_norm(fp32) removed; now using exact integer wsum_norm_i32

static __device__ __forceinline__ void kahan_add(float value, float& sum, float& comp) {
    float y = value - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}

// Quantized closed-form best-gain for the adaptive EMA.
// Matches argmin over {0, 0.1, 0.2, ..., ema_gain_limit*0.1} used in the loop.
static __device__ __forceinline__
float dma_quantized_best_gain(float x,
                              float e0_prev,
                              float ec_prev,
                              float alpha_e,
                              int   ema_gain_limit) {
    // CPU-equivalent two-candidate compare with deterministic tie-break
    // g in {0, 0.1, ..., ema_gain_limit*0.1}; tie -> lower gridpoint
    const float one_minus_alpha_e = 1.0f - alpha_e;
    const float base  = fmaf(alpha_e, e0_prev, one_minus_alpha_e * ec_prev);
    const float t     = alpha_e * (x - ec_prev);
    const float r     = x - base;

    const float EPS = 1e-20f;
    if (fabsf(t) <= EPS) return 0.0f;

    const float step = 0.1f;
    const int   limit = ema_gain_limit; // number of 0.1 steps
    float target = (r / t) / step;      // equals (r/t) * 10

    int i0 = (int)floorf(target);
    if (i0 < 0) i0 = 0; else if (i0 > limit) i0 = limit;
    int i1 = (i0 < limit) ? (i0 + 1) : i0;

    const float g0 = i0 * step;
    const float g1 = i1 * step;
    const float e0 = fabsf(r - t * g0);
    const float e1 = fabsf(r - t * g1);
    return (e0 <= e1) ? g0 : g1;
}

// Compute the updated ec value using the quantized optimal gain.
static __device__ __forceinline__
float dma_update_ec(float x,
                    float e0_prev,
                    float ec_prev,
                    float alpha_e,
                    int   ema_gain_limit) {
    const float g = dma_quantized_best_gain(x, e0_prev, ec_prev, alpha_e, ema_gain_limit);
    const float target = fmaf(g, x - ec_prev, e0_prev);
    return fmaf(alpha_e, target - ec_prev, ec_prev);
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
    const float denom_half_f = (half        > 0 ? wsum_norm_i32(half)        : 1.0f);
    const float denom_full_f = (hull_length > 0 ? wsum_norm_i32(hull_length) : 1.0f);
    const float denom_sqrt_f = (sqrt_len    > 0 ? wsum_norm_i32(sqrt_len)    : 1.0f);
    const float inv_w_half   = 1.0f / denom_half_f;
    const float inv_w_full   = 1.0f / denom_full_f;
    const float inv_w_sqrt   = 1.0f / denom_sqrt_f;

    const int base_out = combo * series_len;
    for (int i = 0; i < series_len; ++i) {
        out[base_out + i] = NAN;
    }

    if (first_valid >= series_len) {
        return;
    }

    // Shared memory scratch ring for diff smoothing.
    extern __shared__ __align__(16) float shared[];
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
                float sum_c = 0.0f;
                for (int k = start; k <= i; ++k) {
                    kahan_add(prices[k], sum, sum_c);
                }
                e0_prev = sum_read(sum, sum_c) / static_cast<float>(ema_length);
                e0_init_done = true;
            }
        } else {
            e0_prev = fmaf(alpha_e, x - e0_prev, e0_prev);
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
                    kahan_add(x - old, a_half, a_half_c);
                    // Rolling WMA weighted-sum update: s_new = s_prev + half * x - a_prev
                    kahan_add(fmaf(static_cast<float>(half), x, -a_prev), s_half, s_half_c);
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
                    kahan_add(x - old, a_full, a_full_c);
                    // Rolling WMA weighted-sum update: s_new = s_prev + hull_length * x - a_prev
                    kahan_add(fmaf(static_cast<float>(hull_length), x, -a_prev), s_full, s_full_c);
                }
            }

            if (half_ready && full_ready) {
                const float w_half = sum_read(s_half, s_half_c) * inv_w_half;
                const float w_full = sum_read(s_full, s_full_c) * inv_w_full;
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
                        e_half_prev = sum_read(sum, sum_c) / static_cast<float>(half);
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = fmaf(alpha_half, x - e_half_prev, e_half_prev);
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
                        e_full_prev = sum_read(sum, sum_c) / static_cast<float>(hull_length);
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = fmaf(alpha_full, x - e_full_prev, e_full_prev);
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
                        hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                    } else {
                        diff_ema = sum_read(diff_sum_seed, diff_sum_seed_c) / static_cast<float>(sqrt_len);
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos += 1; if (diff_pos == sqrt_len) diff_pos = 0;

                if (is_wma) {
                    if (!diff_wma_init_done) {
                        diff_wma_init_done = true;
                    }
                    const float a_prev = a_diff;
                    kahan_add(diff_now - old, a_diff, a_diff_c);
                    // Rolling WMA(diff) weighted-sum update: s_new = s_prev + sqrt_len * diff_now - a_prev
                    kahan_add(fmaf(static_cast<float>(sqrt_len), diff_now, -a_prev), s_diff, s_diff_c);
                    hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                } else {
                    if (!diff_ema_init_done) {
                        diff_ema = diff_now;
                        diff_ema_init_done = true;
                    } else {
                        diff_ema = fmaf(alpha_sqrt, diff_now - diff_ema, diff_ema);
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
                ec_now = dma_update_ec(x, e0_prev, ec_prev, alpha_e, ema_gain_limit);
                ec_prev = ec_now;
            }
        }

        if (!isnan(hull_val) && !isnan(ec_now)) {
            out[base_out + i] = 0.5f * (hull_val + ec_now);
        }
    }
}

// -----------------------------------------------------------------------------
// Tiled batch kernel: each thread computes one parameter combo over time.
// Shared memory is partitioned per-thread: [thread][max_sqrt_len].
// -----------------------------------------------------------------------------

template<int TX>
__device__ void dma_batch_tiled_f32_tx_core(const float* __restrict__ prices,
                                       const int* __restrict__ hull_lengths,
                                       const int* __restrict__ ema_lengths,
                                       const int* __restrict__ ema_gain_limits,
                                       const int* __restrict__ hull_types,
                                       int series_len,
                                       int n_combos,
                                       int first_valid,
                                       int combo_start,
                                       int sqrt_stride,
                                       float* __restrict__ out) {
    const int local = threadIdx.x;
    const int global_idx = combo_start + blockIdx.x * TX + local;
    if (global_idx >= n_combos) { return; }

    extern __shared__ __align__(16) float smem[];
    float* diff_ring = smem + local * sqrt_stride;

    const int hull_length = hull_lengths[global_idx];
    const int ema_length = ema_lengths[global_idx];
    const int ema_gain_limit = ema_gain_limits[global_idx];
    const int hull_type = hull_types[global_idx];

    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0) {
        return;
    }

    const int half = hull_length / 2;
    const int sqrt_len = max(1, (int)floorf(sqrtf((float)hull_length) + 0.5f));
    const float denom_half_f = (half        > 0 ? wsum_norm_i32(half)        : 1.0f);
    const float denom_full_f = (hull_length > 0 ? wsum_norm_i32(hull_length) : 1.0f);
    const float denom_sqrt_f = (sqrt_len    > 0 ? wsum_norm_i32(sqrt_len)    : 1.0f);
    const float inv_w_half   = 1.0f / denom_half_f;
    const float inv_w_full   = 1.0f / denom_full_f;
    const float inv_w_sqrt   = 1.0f / denom_sqrt_f;

    const int base_out = global_idx * series_len;
    // initialize outputs to NaN (one thread covers its series)
    for (int i = 0; i < series_len; ++i) {
        out[base_out + i] = NAN;
    }
    if (first_valid >= series_len) { return; }

    const float alpha_e = 2.0f / (float(ema_length) + 1.0f);
    const int i0_e = first_valid + (ema_length > 0 ? ema_length - 1 : 0);

    float e0_prev = 0.0f; bool e0_init_done = false;
    float ec_prev = 0.0f; bool ec_init_done = false;

    const int i0_half = first_valid + (half > 0 ? half - 1 : 0);
    const int i0_full = first_valid + (hull_length > 0 ? hull_length - 1 : 0);

    float a_half = 0.0f, s_half = 0.0f, a_half_c = 0.0f, s_half_c = 0.0f; bool half_ready = false;
    float a_full = 0.0f, s_full = 0.0f, a_full_c = 0.0f, s_full_c = 0.0f; bool full_ready = false;

    int diff_filled = 0, diff_pos = 0;
    float diff_sum_seed = 0.0f, diff_sum_seed_c = 0.0f;
    float a_diff = 0.0f, s_diff = 0.0f, a_diff_c = 0.0f, s_diff_c = 0.0f; bool diff_wma_init_done = false;
    float diff_ema = 0.0f; bool diff_ema_init_done = false;
    const float alpha_sqrt = (sqrt_len > 0) ? 2.0f / (float(sqrt_len) + 1.0f) : 0.0f;

    float e_half_prev = 0.0f, e_full_prev = 0.0f; bool e_half_init_done = false, e_full_init_done = false;
    const float alpha_half = (half > 0) ? 2.0f / (float(half) + 1.0f) : 0.0f;
    const float alpha_full = (hull_length > 0) ? 2.0f / (float(hull_length) + 1.0f) : 0.0f;

    const bool is_wma = (hull_type == 0);
    float hull_val = NAN;

    for (int i = first_valid; i < series_len; ++i) {
        const float x = prices[i];

        if (!e0_init_done) {
            if (i >= i0_e) {
                int start = i + 1 - ema_length;
                float sum = 0.0f, sum_c = 0.0f;
                for (int k = start; k <= i; ++k) { kahan_add(prices[k], sum, sum_c); }
                e0_prev = sum_read(sum, sum_c) / float(ema_length);
                e0_init_done = true;
            }
        } else {
            e0_prev = fmaf(alpha_e, x - e0_prev, e0_prev);
        }

        float diff_now = NAN;
        if (is_wma) {
            if (half > 0) {
                if (!half_ready) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f, sum_c = 0.0f, wsum = 0.0f, wsum_c = 0.0f;
                        for (int j = 0; j < half; ++j) {
                            const int idx = start + j; const float w = float(j + 1); const float v = prices[idx];
                            kahan_add(v, sum, sum_c); kahan_add(w * v, wsum, wsum_c);
                        }
                        a_half = sum; s_half = wsum; a_half_c = sum_c; s_half_c = wsum_c; half_ready = true;
                    }
                } else {
                    const float a_prev = a_half; const float old = prices[i - half];
                    kahan_add(x - old, a_half, a_half_c);
                    kahan_add(fmaf(float(half), x, -a_prev), s_half, s_half_c);
                }
            }
            if (hull_length > 0) {
                if (!full_ready) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f, sum_c = 0.0f, wsum = 0.0f, wsum_c = 0.0f;
                        for (int j = 0; j < hull_length; ++j) {
                            const int idx = start + j; const float w = float(j + 1); const float v = prices[idx];
                            kahan_add(v, sum, sum_c); kahan_add(w * v, wsum, wsum_c);
                        }
                        a_full = sum; s_full = wsum; a_full_c = sum_c; s_full_c = wsum_c; full_ready = true;
                    }
                } else {
                    const float a_prev = a_full; const float old = prices[i - hull_length];
                    kahan_add(x - old, a_full, a_full_c);
                    kahan_add(fmaf(float(hull_length), x, -a_prev), s_full, s_full_c);
                }
            }
            if (half_ready && full_ready) {
                const float w_half = sum_read(s_half, s_half_c) * inv_w_half;
                const float w_full = sum_read(s_full, s_full_c) * inv_w_full;
                diff_now = 2.0f * w_half - w_full;
            }
        } else {
            if (half > 0) {
                if (!e_half_init_done) {
                    if (i >= i0_half) {
                        int start = i + 1 - half; float sum = 0.0f, sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) { kahan_add(prices[k], sum, sum_c); }
                        e_half_prev = sum_read(sum, sum_c) / float(half); e_half_init_done = true;
                    }
                } else { e_half_prev = fmaf(alpha_half, x - e_half_prev, e_half_prev); }
            }
            if (hull_length > 0) {
                if (!e_full_init_done) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length; float sum = 0.0f, sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) { kahan_add(prices[k], sum, sum_c); }
                        e_full_prev = sum_read(sum, sum_c) / float(hull_length); e_full_init_done = true;
                    }
                } else { e_full_prev = fmaf(alpha_full, x - e_full_prev, e_full_prev); }
            }
            if (e_half_init_done && e_full_init_done) { diff_now = 2.0f * e_half_prev - e_full_prev; }
        }

        if (!isnan(diff_now) && sqrt_len > 0) {
            if (diff_filled < sqrt_len) {
                diff_ring[diff_filled] = diff_now; kahan_add(diff_now, diff_sum_seed, diff_sum_seed_c);
                diff_filled += 1;
                if (diff_filled == sqrt_len) {
                    if (is_wma) {
                        a_diff = 0.0f; s_diff = 0.0f; a_diff_c = 0.0f; s_diff_c = 0.0f;
                        for (int j = 0; j < sqrt_len; ++j) {
                            const float w = float(j + 1); const float v = diff_ring[j];
                            kahan_add(v, a_diff, a_diff_c); kahan_add(w * v, s_diff, s_diff_c);
                        }
                        diff_wma_init_done = true;
                        hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                    } else {
                        diff_ema = sum_read(diff_sum_seed, diff_sum_seed_c) / float(sqrt_len); diff_ema_init_done = true; hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos]; diff_ring[diff_pos] = diff_now; diff_pos += 1; if (diff_pos == sqrt_len) diff_pos = 0;
                if (is_wma) {
                    if (!diff_wma_init_done) { diff_wma_init_done = true; }
                    const float a_prev = a_diff;
                    kahan_add(diff_now - old, a_diff, a_diff_c);
                    kahan_add(fmaf(float(sqrt_len), diff_now, -a_prev), s_diff, s_diff_c);
                    hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                } else {
                    if (!diff_ema_init_done) { diff_ema = diff_now; diff_ema_init_done = true; }
                    else { diff_ema = fmaf(alpha_sqrt, diff_now - diff_ema, diff_ema); }
                    hull_val = diff_ema;
                }
            }
        }

        float ec_now = NAN;
        if (e0_init_done) {
            if (!ec_init_done) { ec_prev = e0_prev; ec_now = ec_prev; ec_init_done = true; }
            else { ec_now = dma_update_ec(x, e0_prev, ec_prev, alpha_e, ema_gain_limit); ec_prev = ec_now; }
        }

        if (!isnan(hull_val) && !isnan(ec_now)) { out[base_out + i] = 0.5f * (hull_val + ec_now); }
    }
}

extern "C" {
__global__ void dma_batch_tiled_f32_tx64(
    const float* __restrict__ prices,
    const int* __restrict__ hull_lengths,
    const int* __restrict__ ema_lengths,
    const int* __restrict__ ema_gain_limits,
    const int* __restrict__ hull_types,
    int series_len,
    int n_combos,
    int first_valid,
    int combo_start,
    int sqrt_stride,
    float* __restrict__ out) {
    dma_batch_tiled_f32_tx_core<64>(prices, hull_lengths, ema_lengths, ema_gain_limits, hull_types,
                                    series_len, n_combos, first_valid, combo_start, sqrt_stride, out);
}
__global__ void dma_batch_tiled_f32_tx128(
    const float* __restrict__ prices,
    const int* __restrict__ hull_lengths,
    const int* __restrict__ ema_lengths,
    const int* __restrict__ ema_gain_limits,
    const int* __restrict__ hull_types,
    int series_len,
    int n_combos,
    int first_valid,
    int combo_start,
    int sqrt_stride,
    float* __restrict__ out) {
    dma_batch_tiled_f32_tx_core<128>(prices, hull_lengths, ema_lengths, ema_gain_limits, hull_types,
                                     series_len, n_combos, first_valid, combo_start, sqrt_stride, out);
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

    extern __shared__ __align__(16) float diff_ring[];

    const int half = hull_length / 2;
    const int sqrt_len_clamped = (sqrt_len > 0) ? sqrt_len : 1;

    // Weighted moving average normalizers for half/full/sqrt windows.
    // Use exact integer normalization converted to fp32.
    const float denom_half_f = (half        > 0 ? wsum_norm_i32(half)        : 1.0f);
    const float denom_full_f = (hull_length > 0 ? wsum_norm_i32(hull_length) : 1.0f);
    const float denom_sqrt_f = (sqrt_len_clamped > 0 ? wsum_norm_i32(sqrt_len_clamped) : 1.0f);
    const float inv_w_half   = 1.0f / denom_half_f;
    const float inv_w_full   = 1.0f / denom_full_f;
    const float inv_w_sqrt   = 1.0f / denom_sqrt_f;

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
                e0_prev = sum_read(sum, sum_c) / static_cast<float>(ema_length);
                e0_init_done = true;
            }
        } else {
            e0_prev = fmaf(alpha_e, x - e0_prev, e0_prev);
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
                    kahan_add(x - old, a_half, a_half_c);
                    // Rolling WMA weighted-sum update: s_new = s_prev + half * x - a_prev
                    kahan_add(fmaf(static_cast<float>(half), x, -a_prev), s_half, s_half_c);
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
                    kahan_add(x - old, a_full, a_full_c);
                    // Rolling WMA weighted-sum update: s_new = s_prev + hull_length * x - a_prev
                    kahan_add(fmaf(static_cast<float>(hull_length), x, -a_prev), s_full, s_full_c);
                }
            }

            if (half_ready && full_ready) {
                const float w_half = sum_read(s_half, s_half_c) * inv_w_half;
                const float w_full = sum_read(s_full, s_full_c) * inv_w_full;
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
                        e_half_prev = sum_read(sum, sum_c) / static_cast<float>(half);
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = fmaf(alpha_half, x - e_half_prev, e_half_prev);
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
                        e_full_prev = sum_read(sum, sum_c) / static_cast<float>(hull_length);
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = fmaf(alpha_full, x - e_full_prev, e_full_prev);
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
                        hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                    } else {
                        diff_ema = sum_read(diff_sum_seed, diff_sum_seed_c) / static_cast<float>(sqrt_len_clamped);
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos += 1; if (diff_pos == sqrt_len_clamped) diff_pos = 0;

                if (is_wma) {
                    if (!diff_wma_init_done) {
                        diff_wma_init_done = true;
                    }
                    const float a_prev = a_diff;
                    kahan_add(diff_now - old, a_diff, a_diff_c);
                    // Rolling WMA(diff) weighted-sum update: s_new = s_prev + sqrt_len * diff_now - a_prev
                    kahan_add(fmaf(static_cast<float>(sqrt_len_clamped), diff_now, -a_prev), s_diff, s_diff_c);
                    hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                } else {
                    if (!diff_ema_init_done) {
                        diff_ema = diff_now;
                        diff_ema_init_done = true;
                    } else {
                        diff_ema = fmaf(alpha_sqrt, diff_now - diff_ema, diff_ema);
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
                ec_now = dma_update_ec(x, e0_prev, ec_prev, alpha_e, ema_gain_limit);
                ec_prev = ec_now;
            }
        }

        if (!isnan(hull_val) && !isnan(ec_now)) {
            out_tm[base_out + i * stride] = 0.5f * (hull_val + ec_now);
        }
    }
}

// -----------------------------------------------------------------------------
// Many-series tiled 2D kernel: each thread in Y processes one series
// sequentially; shared memory is partitioned by threadIdx.y.
// -----------------------------------------------------------------------------

template<int TY>
__device__ void dma_ms1p_tiled_f32_tx1_ty_core(const float* __restrict__ prices_tm,
                                          int hull_length,
                                          int ema_length,
                                          int ema_gain_limit,
                                          int hull_type,
                                          int series_len,
                                          int num_series,
                                          const int* __restrict__ first_valids,
                                          int sqrt_len,
                                          float* __restrict__ out_tm) {
    const int base_series = blockIdx.x * TY;
    const int series_idx = base_series + threadIdx.y;
    if (series_idx >= num_series) { return; }
    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0) { return; }

    extern __shared__ __align__(16) float smem[];
    float* diff_ring = smem + threadIdx.y * max(1, sqrt_len);

    const int stride = num_series;
    const int base_out = series_idx;

    if (threadIdx.x == 0) {
        for (int i = 0; i < series_len; ++i) { out_tm[base_out + i * stride] = NAN; }
    }
    if (threadIdx.x != 0) { return; }

    const int first_valid = first_valids[series_idx];
    if (first_valid >= series_len) { return; }

    const int half = hull_length / 2;
    const int sqrt_len_clamped = max(1, sqrt_len);
    const float denom_half_f = (half        > 0 ? wsum_norm_i32(half)              : 1.0f);
    const float denom_full_f = (hull_length > 0 ? wsum_norm_i32(hull_length)       : 1.0f);
    const float denom_sqrt_f = (sqrt_len_clamped > 0 ? wsum_norm_i32(sqrt_len_clamped) : 1.0f);
    const float inv_w_half   = 1.0f / denom_half_f;
    const float inv_w_full   = 1.0f / denom_full_f;
    const float inv_w_sqrt   = 1.0f / denom_sqrt_f;

    const float alpha_e = 2.0f / (float(ema_length) + 1.0f);
    const int i0_e = first_valid + (ema_length > 0 ? ema_length - 1 : 0);

    float e0_prev = 0.0f; bool e0_init_done = false;
    float ec_prev = 0.0f; bool ec_init_done = false;

    const int i0_half = first_valid + (half > 0 ? half - 1 : 0);
    const int i0_full = first_valid + (hull_length > 0 ? hull_length - 1 : 0);

    float a_half = 0.0f, s_half = 0.0f, a_half_c = 0.0f, s_half_c = 0.0f; bool half_ready = false;
    float a_full = 0.0f, s_full = 0.0f, a_full_c = 0.0f, s_full_c = 0.0f; bool full_ready = false;

    int diff_filled = 0, diff_pos = 0; float diff_sum_seed = 0.0f, diff_sum_seed_c = 0.0f;
    float a_diff = 0.0f, s_diff = 0.0f, a_diff_c = 0.0f, s_diff_c = 0.0f; bool diff_wma_init_done = false;
    float diff_ema = 0.0f; bool diff_ema_init_done = false;
    const float alpha_sqrt = (sqrt_len_clamped > 0) ? 2.0f / (float(sqrt_len_clamped) + 1.0f) : 0.0f;

    float e_half_prev = 0.0f, e_full_prev = 0.0f; bool e_half_init_done = false, e_full_init_done = false;
    const float alpha_half = (half > 0) ? 2.0f / (float(half) + 1.0f) : 0.0f;
    const float alpha_full = (hull_length > 0) ? 2.0f / (float(hull_length) + 1.0f) : 0.0f;

    const bool is_wma = (hull_type == 0);
    float hull_val = NAN;

    for (int i = first_valid; i < series_len; ++i) {
        const int idx = i * stride + series_idx; const float x = prices_tm[idx];

        if (!e0_init_done) {
            if (i >= i0_e) {
                int start = i + 1 - ema_length; float sum = 0.0f, sum_c = 0.0f;
                for (int k = start; k <= i; ++k) { kahan_add(prices_tm[k * stride + series_idx], sum, sum_c); }
                e0_prev = sum_read(sum, sum_c) / float(ema_length); e0_init_done = true;
            }
        } else { e0_prev = fmaf(alpha_e, x - e0_prev, e0_prev); }

        float diff_now = NAN;
        if (is_wma) {
            if (half > 0) {
                if (!half_ready) {
                    if (i >= i0_half) {
                        int start = i + 1 - half; float sum = 0.0f, sum_c = 0.0f, wsum = 0.0f, wsum_c = 0.0f;
                        for (int j = 0; j < half; ++j) {
                            const int sidx = start + j; const float w = float(j + 1);
                            const float v = prices_tm[sidx * stride + series_idx];
                            kahan_add(v, sum, sum_c); kahan_add(w * v, wsum, wsum_c);
                        }
                        a_half = sum; s_half = wsum; a_half_c = sum_c; s_half_c = wsum_c; half_ready = true;
                    }
                } else {
                    const float a_prev = a_half; const float old = prices_tm[(i - half) * stride + series_idx];
                    kahan_add(x - old, a_half, a_half_c);
                    kahan_add(float(half) * x, s_half, s_half_c); kahan_add(-a_prev, s_half, s_half_c);
                }
            }
            if (hull_length > 0) {
                if (!full_ready) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length; float sum = 0.0f, sum_c = 0.0f, wsum = 0.0f, wsum_c = 0.0f;
                        for (int j = 0; j < hull_length; ++j) {
                            const int sidx = start + j; const float w = float(j + 1);
                            const float v = prices_tm[sidx * stride + series_idx];
                            kahan_add(v, sum, sum_c); kahan_add(w * v, wsum, wsum_c);
                        }
                        a_full = sum; s_full = wsum; a_full_c = sum_c; s_full_c = wsum_c; full_ready = true;
                    }
                } else {
                    const float a_prev = a_full; const float old = prices_tm[(i - hull_length) * stride + series_idx];
                    kahan_add(x - old, a_full, a_full_c);
                    kahan_add(float(hull_length) * x, s_full, s_full_c); kahan_add(-a_prev, s_full, s_full_c);
                }
            }
            if (half_ready && full_ready) {
                const float w_half = sum_read(s_half, s_half_c) * inv_w_half; const float w_full = sum_read(s_full, s_full_c) * inv_w_full;
                diff_now = 2.0f * w_half - w_full;
            }
        } else {
            if (half > 0) {
                if (!e_half_init_done) {
                    if (i >= i0_half) {
                        int start = i + 1 - half; float sum = 0.0f, sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) { kahan_add(prices_tm[k * stride + series_idx], sum, sum_c); }
                        e_half_prev = sum_read(sum, sum_c) / float(half); e_half_init_done = true;
                    }
                } else { e_half_prev = fmaf(alpha_half, x - e_half_prev, e_half_prev); }
            }
            if (hull_length > 0) {
                if (!e_full_init_done) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length; float sum = 0.0f, sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) { kahan_add(prices_tm[k * stride + series_idx], sum, sum_c); }
                        e_full_prev = sum_read(sum, sum_c) / float(hull_length); e_full_init_done = true;
                    }
                } else { e_full_prev = fmaf(alpha_full, x - e_full_prev, e_full_prev); }
            }
            if (e_half_init_done && e_full_init_done) { diff_now = 2.0f * e_half_prev - e_full_prev; }
        }

        if (!isnan(diff_now) && sqrt_len_clamped > 0) {
            if (diff_filled < sqrt_len_clamped) {
                diff_ring[diff_filled] = diff_now; kahan_add(diff_now, diff_sum_seed, diff_sum_seed_c);
                diff_filled += 1;
                if (diff_filled == sqrt_len_clamped) {
                    if (is_wma) {
                        a_diff = 0.0f; s_diff = 0.0f; a_diff_c = 0.0f; s_diff_c = 0.0f;
                        for (int j = 0; j < sqrt_len_clamped; ++j) {
                            const float w = float(j + 1); const float v = diff_ring[j];
                            kahan_add(v, a_diff, a_diff_c); kahan_add(w * v, s_diff, s_diff_c);
                        }
                        diff_wma_init_done = true;
                        hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                    } else {
                        diff_ema = sum_read(diff_sum_seed, diff_sum_seed_c) / float(sqrt_len_clamped); diff_ema_init_done = true; hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos]; diff_ring[diff_pos] = diff_now; diff_pos += 1; if (diff_pos == sqrt_len_clamped) diff_pos = 0;
                if (is_wma) {
                    if (!diff_wma_init_done) { diff_wma_init_done = true; }
                    const float a_prev = a_diff;
                    kahan_add(diff_now - old, a_diff, a_diff_c);
                    kahan_add(fmaf(float(sqrt_len_clamped), diff_now, -a_prev), s_diff, s_diff_c);
                    hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                } else {
                    if (!diff_ema_init_done) { diff_ema = diff_now; diff_ema_init_done = true; }
                    else { diff_ema = fmaf(alpha_sqrt, diff_now - diff_ema, diff_ema); }
                    hull_val = diff_ema;
                }
            }
        }

        float ec_now = NAN;
        if (e0_init_done) {
            if (!ec_init_done) { ec_prev = e0_prev; ec_now = ec_prev; ec_init_done = true; }
            else { ec_now = dma_update_ec(x, e0_prev, ec_prev, alpha_e, ema_gain_limit); ec_prev = ec_now; }
        }

        if (!isnan(hull_val) && !isnan(ec_now)) { out_tm[base_out + i * stride] = 0.5f * (hull_val + ec_now); }
    }
}

extern "C" {
__global__ void dma_ms1p_tiled_f32_tx1_ty2(const float* __restrict__ prices_tm,
                                           int hull_length,
                                           int ema_length,
                                           int ema_gain_limit,
                                           int hull_type,
                                           int series_len,
                                           int num_series,
                                           const int* __restrict__ first_valids,
                                           int sqrt_len,
                                           float* __restrict__ out_tm) {
    dma_ms1p_tiled_f32_tx1_ty_core<2>(prices_tm, hull_length, ema_length, ema_gain_limit, hull_type,
                                      series_len, num_series, first_valids, sqrt_len, out_tm);
}
__global__ void dma_ms1p_tiled_f32_tx1_ty4(const float* __restrict__ prices_tm,
                                           int hull_length,
                                           int ema_length,
                                           int ema_gain_limit,
                                           int hull_type,
                                           int series_len,
                                           int num_series,
                                           const int* __restrict__ first_valids,
                                           int sqrt_len,
                                           float* __restrict__ out_tm) {
    dma_ms1p_tiled_f32_tx1_ty_core<4>(prices_tm, hull_length, ema_length, ema_gain_limit, hull_type,
                                      series_len, num_series, first_valids, sqrt_len, out_tm);
}
}
