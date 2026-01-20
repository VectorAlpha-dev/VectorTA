





#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>


#ifndef UMA_WARP_SIZE
#define UMA_WARP_SIZE 32
#endif



#ifndef UMA_SHARED_LOG2_MAX
#define UMA_SHARED_LOG2_MAX 512
#endif



#ifndef UMA_LOG2_TABLE_SIZE
#define UMA_LOG2_TABLE_SIZE 8192
#endif

#if defined(UMA_PRECOMPUTE_LOG2)
__constant__ float UMA_LOG2_TABLE[UMA_LOG2_TABLE_SIZE + 1];
#endif

#if defined(UMA_USE_LDG)
  #define UMA_RO_LOAD(ptr) __ldg((ptr))
#else
  #define UMA_RO_LOAD(ptr) (*(ptr))
#endif


static __device__ __forceinline__ bool is_nan(float v) { return isnan(v); }

static __device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

static __device__ __forceinline__ float load_tm(
    const float* __restrict__ data, int num_series, int series_idx, int t) {
    return UMA_RO_LOAD(&data[t * num_series + series_idx]);
}


static __device__ __forceinline__ float warp_reduce_sum(float v, unsigned mask) {
#pragma unroll
    for (int off = UMA_WARP_SIZE >> 1; off > 0; off >>= 1) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}



static __device__ __forceinline__ float uma_weight_pow(int base, float p) {





    const float l2 = __log2f((float)base);
    return exp2f(p * l2);
}


static __device__ __forceinline__ float compute_rsi(
    const float* __restrict__ data, int start, int end, int period) {

    if (period <= 1) return 50.0f;
    int len = end - start;
    if (len <= period) return 50.0f;

    float invP = 1.0f / (float)period;
    float beta = 1.0f - invP;
    float avg_gain = 0.0f, avg_loss = 0.0f;

    int warm_end = start + period;
    if (warm_end >= end) return 50.0f;

    for (int idx = start + 1; idx <= warm_end; ++idx) {
        float cur = UMA_RO_LOAD(&data[idx]);
        float prev = UMA_RO_LOAD(&data[idx - 1]);
        if (!isfinite(cur) || !isfinite(prev)) return 50.0f;
        float d = cur - prev;
        if (d > 0.0f) avg_gain += d;
        else if (d < 0.0f) avg_loss -= d;
    }
    avg_gain *= invP; avg_loss *= invP;
    float denom = avg_gain + avg_loss;
    float rsi = (denom == 0.0f) ? 50.0f : 100.0f * avg_gain / denom;

    for (int idx = warm_end + 1; idx < end; ++idx) {
        float cur = UMA_RO_LOAD(&data[idx]);
        float prev = UMA_RO_LOAD(&data[idx - 1]);
        if (!isfinite(cur) || !isfinite(prev)) return 50.0f;
        float d = cur - prev;
        float gain = d > 0.0f ? d : 0.0f;
        float loss = d < 0.0f ? -d : 0.0f;
        avg_gain = fmaf(invP, gain, beta * avg_gain);
        avg_loss = fmaf(invP, loss, beta * avg_loss);
        denom = avg_gain + avg_loss;
        rsi = (denom == 0.0f) ? 50.0f : 100.0f * avg_gain / denom;
    }
    return rsi;
}








static __device__ __forceinline__ float compute_rsi_last_warp(
    const float* __restrict__ data, int start, int end, int period,
    int lane, int lanes, unsigned mask) {

    if (period <= 1) return 50.0f;
    const int len = end - start;
    if (len <= period) return 50.0f;
    const int m = len - 1;

    const float alpha = 1.0f / (float)period;
    const float beta  = 1.0f - alpha;


    float up0_part = 0.0f;
    float dn0_part = 0.0f;
    int ok0 = 1;
    for (int t = lane + 1; t <= period; t += lanes) {
        const int idx = start + t;
        const float cur = UMA_RO_LOAD(&data[idx]);
        const float prev = UMA_RO_LOAD(&data[idx - 1]);
        if (!isfinite(cur) || !isfinite(prev)) {
            ok0 = 0;
            continue;
        }
        const float d = cur - prev;
        if (d > 0.0f) up0_part += d;
        else if (d < 0.0f) dn0_part -= d;
    }
    const int all_ok0 = __all_sync(mask, ok0);
    const float up0 = warp_reduce_sum(up0_part, mask);
    const float dn0 = warp_reduce_sum(dn0_part, mask);
    if (!all_ok0) return 50.0f;

    const float avg_up0 = up0 * alpha;
    const float avg_dn0 = dn0 * alpha;


    const int tail_len = m - period;
    if (tail_len <= 0) {
        const float denom0 = avg_up0 + avg_dn0;
        return (denom0 == 0.0f) ? 50.0f : (100.0f * avg_up0 / denom0);
    }



    const float log2_beta = __log2f(beta);
    const float beta_step = exp2f((float)lanes * log2_beta);
    float pow_b = exp2f((float)lane * log2_beta);

    float up_tail_part = 0.0f;
    float dn_tail_part = 0.0f;
    int ok1 = 1;
    for (int k = lane; k < tail_len; k += lanes) {
        const int t = m - k;
        const int idx = start + t;
        const float cur = UMA_RO_LOAD(&data[idx]);
        const float prev = UMA_RO_LOAD(&data[idx - 1]);
        if (!isfinite(cur) || !isfinite(prev)) {
            ok1 = 0;
        } else {
            const float d = cur - prev;
            const float gain = d > 0.0f ? d : 0.0f;
            const float loss = d < 0.0f ? -d : 0.0f;
            up_tail_part = fmaf(pow_b, gain, up_tail_part);
            dn_tail_part = fmaf(pow_b, loss, dn_tail_part);
        }
        pow_b *= beta_step;
    }

    const int all_ok1 = __all_sync(mask, ok1);
    const float up_tail = warp_reduce_sum(up_tail_part, mask);
    const float dn_tail = warp_reduce_sum(dn_tail_part, mask);
    if (!all_ok1) return 50.0f;

    const float beta_tail = exp2f((float)tail_len * log2_beta);
    const float avg_up = fmaf(beta_tail, avg_up0, alpha * up_tail);
    const float avg_dn = fmaf(beta_tail, avg_dn0, alpha * dn_tail);

    const float denom = avg_up + avg_dn;
    return (denom == 0.0f) ? 50.0f : (100.0f * avg_up / denom);
}

static __device__ __forceinline__ float compute_rsi_tm(
    const float* __restrict__ data_tm, int num_series, int series_idx,
    int start, int end, int period) {

    if (period <= 1) return 50.0f;
    int len = end - start;
    if (len <= period) return 50.0f;

    float invP = 1.0f / (float)period;
    float beta = 1.0f - invP;
    float avg_gain = 0.0f, avg_loss = 0.0f;

    int warm_end = start + period;
    if (warm_end >= end) return 50.0f;

    for (int idx = start + 1; idx <= warm_end; ++idx) {
        float cur = load_tm(data_tm, num_series, series_idx, idx);
        float prev = load_tm(data_tm, num_series, series_idx, idx - 1);
        if (!isfinite(cur) || !isfinite(prev)) return 50.0f;
        float d = cur - prev;
        if (d > 0.0f) avg_gain += d;
        else if (d < 0.0f) avg_loss -= d;
    }
    avg_gain *= invP; avg_loss *= invP;
    float denom = avg_gain + avg_loss;
    float rsi = (denom == 0.0f) ? 50.0f : 100.0f * avg_gain / denom;

    for (int idx = warm_end + 1; idx < end; ++idx) {
        float cur = load_tm(data_tm, num_series, series_idx, idx);
        float prev = load_tm(data_tm, num_series, series_idx, idx - 1);
        if (!isfinite(cur) || !isfinite(prev)) return 50.0f;
        float d = cur - prev;
        float gain = d > 0.0f ? d : 0.0f;
        float loss = d < 0.0f ? -d : 0.0f;
        avg_gain = fmaf(invP, gain, beta * avg_gain);
        avg_loss = fmaf(invP, loss, beta * avg_loss);
        denom = avg_gain + avg_loss;
        rsi = (denom == 0.0f) ? 50.0f : 100.0f * avg_gain / denom;
    }
    return rsi;
}


extern "C" __global__ __launch_bounds__(UMA_WARP_SIZE, 8)
void uma_batch_f32(const float* __restrict__ prices,
                   const float* __restrict__ volumes,
                   int has_volume,
                   const float* __restrict__ accelerators,
                   const int* __restrict__ min_lengths,
                   const int* __restrict__ max_lengths,
                   const int* __restrict__ smooth_lengths,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ raw_out,
                   float* __restrict__ final_out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int lane = threadIdx.x & (UMA_WARP_SIZE - 1);
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lanes = min(blockDim.x, UMA_WARP_SIZE);
    const unsigned mask = __ballot_sync(full_mask, threadIdx.x < lanes);

    const float accelerator = accelerators[combo];
    const int min_len = min_lengths[combo];
    const int max_len = max_lengths[combo];
    const int smooth_len = smooth_lengths[combo];

    if (series_len <= 0 || max_len <= 0 || min_len <= 0) return;

    const int base = combo * series_len;


    for (int i = lane; i < series_len; i += lanes) {
        raw_out[base + i] = NAN;
        final_out[base + i] = NAN;
    }

    if (first_valid >= series_len) return;
    __syncwarp(mask);


    float length_f = (float)max_len;
    float sum = 0.0f, sum_sq = 0.0f;
    int count = 0;

    const int warm_raw = first_valid + max_len - 1;

    for (int i = first_valid; i < series_len; ++i) {
        int proceed = 0;
        int len_r = 0;
        int window_start_weights = 0;
        float p = 0.0f;
        float price_now_lane0 = 0.0f;

        if (lane == 0) {
            const float price_now = UMA_RO_LOAD(&prices[i]);
            price_now_lane0 = price_now;

            if (!is_nan(price_now)) {
                sum = fmaf(price_now, 1.0f, sum);
                sum_sq = fmaf(price_now, price_now, sum_sq);
                ++count;
            }
            if (i >= first_valid + max_len) {
                const float price_old = UMA_RO_LOAD(&prices[i - max_len]);
                if (!is_nan(price_old)) {
                    sum -= price_old;
                    sum_sq -= price_old * price_old;
                    --count;
                }
            }
            if (i >= warm_raw && count == max_len && isfinite(price_now)) {
                const float mean = sum / (float)max_len;
                float var = sum_sq / (float)max_len - mean * mean;
                var = fmaxf(var, 0.0f);



                if (isfinite(var) && isfinite(mean)) {
                    const float diff = price_now - mean;
                    const float diff2 = diff * diff;
                    const float inner2 = 0.0625f * var;
                    const float outer2 = 3.0625f * var;
                    if (diff2 <= inner2)          length_f += 1.0f;
                    else if (diff2 > outer2)      length_f -= 1.0f;
                    length_f = clampf(length_f, (float)min_len, (float)max_len);

                    len_r = (int)floorf(length_f + 0.5f);
                    if (len_r < min_len) len_r = min_len;
                    if (len_r > max_len) len_r = max_len;
                    if (len_r < 1)       len_r = 1;

                    if (i + 1 >= len_r) {

                        bool try_volume = false;
                        if (has_volume && volumes != nullptr) {
                            const float vol_now = UMA_RO_LOAD(&volumes[i]);
                            if (!is_nan(vol_now) && vol_now != 0.0f) {
                                try_volume = true;
                            }
                        }
                        proceed = try_volume ? 2 : 1;
                    }
                }
            }

            if (proceed) {
                window_start_weights = i + 1 - len_r;
            }
        }


        proceed               = __shfl_sync(mask, proceed, 0);
        len_r                 = __shfl_sync(mask, len_r, 0);
        window_start_weights  = __shfl_sync(mask, window_start_weights, 0);
        const float price_now = __shfl_sync(mask, price_now_lane0, 0);

        if (!proceed) continue;


        float mf = 50.0f;
        if (proceed == 2) {

            int len_mf = len_r;
            int available = (i + 1) - first_valid;
            if (len_mf > available) len_mf = available;

            if (len_mf >= 2) {
                const int start_mf = (i + 1) - len_mf;

                float up_part = 0.0f, down_part = 0.0f;
                int ok = 1;

                const int items = len_mf - 1;
                for (int jj = lane; jj < items; jj += lanes) {
                    const int j = start_mf + 1 + jj;
                    const float px_cur  = UMA_RO_LOAD(&prices[j]);
                    const float px_prev = UMA_RO_LOAD(&prices[j - 1]);
                    const float vol_j   = UMA_RO_LOAD(&volumes[j]);
                    if (!isfinite(px_cur) || !isfinite(px_prev) || !isfinite(vol_j)) {
                        ok = 0;
                    } else {
                        const float delta = px_cur - px_prev;
                        if (delta > 0.0f)      up_part   += vol_j;
                        else if (delta < 0.0f) down_part += vol_j;
                    }
                }

                const int all_ok = __all_sync(mask, ok);
                float up_vol   = warp_reduce_sum(up_part, mask);
                float down_vol = warp_reduce_sum(down_part, mask);
                if (all_ok) {
                    const float denom_vol = up_vol + down_vol;
                    if (denom_vol > 0.0f) mf = 100.0f * up_vol / denom_vol;
                } else {

                    proceed = 1;
                }
            } else {

                proceed = 1;
            }
        }

        if (proceed == 1) {

            int window_start_rsi = (i + 1) - (len_r * 2);
            if (window_start_rsi < 0) window_start_rsi = 0;
            const int window_end = i + 1;
            mf = compute_rsi_last_warp(prices, window_start_rsi, window_end, len_r, lane, lanes, mask);
        }



        const float mf_scaled = fmaf(mf, 2.0f, -100.0f);
        p = fmaf(fabsf(mf_scaled), 0.04f, accelerator);


        float ws_part = 0.0f, wt_part = 0.0f;

        for (int j = lane; j < len_r; j += lanes) {
            const int idx = window_start_weights + j;
            const float v = UMA_RO_LOAD(&prices[idx]);
            if (!is_nan(v)) {
                const int base_pow = len_r - j;
                const float w = uma_weight_pow(base_pow, p);
                ws_part = fmaf(v, w, ws_part);
                wt_part += w;
            }
        }
        float ws = warp_reduce_sum(ws_part, mask);
        float wt = warp_reduce_sum(wt_part, mask);

        if (lane == 0) {
            float result = price_now;
            if (wt > 0.0f) result = ws / wt;
            raw_out[base + i] = result;
        }
        __syncwarp(mask);
    }


    if (smooth_len <= 1) {

        for (int i = lane; i < series_len; i += lanes) {
            final_out[base + i] = raw_out[base + i];
        }
        return;
    }

    const int warm_smooth = (first_valid + max_len - 1) + smooth_len - 1;
    if (warm_smooth >= series_len) return;

    if (lane == 0) {
        const int lookback = smooth_len - 1;
        const float denom = 0.5f * (float)smooth_len * (float)(smooth_len + 1);

        float weighted_sum = 0.0f;
        float plain_sum = 0.0f;
        const int warm_raw_idx = first_valid + max_len - 1;

        for (int j = 0; j < lookback; ++j) {
            const float val = raw_out[base + warm_raw_idx + j];
            weighted_sum += ((float)j + 1.0f) * val;
            plain_sum += val;
        }
        int first_idx = warm_raw_idx + lookback;
        const float first_val = raw_out[base + first_idx];
        weighted_sum += (float)smooth_len * first_val;
        plain_sum += first_val;
        final_out[base + first_idx] = weighted_sum / denom;

        weighted_sum -= plain_sum;
        plain_sum -= raw_out[base + warm_raw_idx];

        for (int i = first_idx + 1; i < series_len; ++i) {
            const float v_new = raw_out[base + i];
            weighted_sum += (float)smooth_len * v_new;
            plain_sum += v_new;
            final_out[base + i] = weighted_sum / denom;
            weighted_sum -= plain_sum;
            const float v_old = raw_out[base + i - lookback];
            plain_sum -= v_old;
        }
    }
}


extern "C" __global__ __launch_bounds__(UMA_WARP_SIZE, 8)
void uma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const float* __restrict__ volumes_tm,
                                   int has_volume,
                                   float accelerator,
                                   int min_length,
                                   int max_length,
                                   int smooth_length,
                                   int num_series,
                                   int series_len,
                                   const int* __restrict__ first_valids,
                                   float* __restrict__ raw_out_tm,
                                   float* __restrict__ final_out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) return;

    const int lane = threadIdx.x & (UMA_WARP_SIZE - 1);
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lanes = min(blockDim.x, UMA_WARP_SIZE);
    const unsigned mask = __ballot_sync(full_mask, threadIdx.x < lanes);

    int fv = first_valids[series_idx];
    if (fv < 0) fv = 0;
    if (fv >= series_len) return;


    for (int t = lane; t < series_len; t += lanes) {
        const int idx = t * num_series + series_idx;
        raw_out_tm[idx] = NAN;
        final_out_tm[idx] = NAN;
    }
    if (max_length <= 0 || min_length <= 0 || smooth_length < 0) return;
    __syncwarp(mask);


    float length_f = (float)max_length;
    float sum = 0.0f, sum_sq = 0.0f;
    int count = 0;

    const int warm_raw = fv + max_length - 1;

    for (int i = fv; i < series_len; ++i) {
        int proceed = 0;
        int len_r = 0;
        int window_start_weights = 0;
        float p = 0.0f;
        float price_now_lane0 = 0.0f;

        if (lane == 0) {
            const float price_now = load_tm(prices_tm, num_series, series_idx, i);
            price_now_lane0 = price_now;
            if (!is_nan(price_now)) {
                sum = fmaf(price_now, 1.0f, sum);
                sum_sq = fmaf(price_now, price_now, sum_sq);
                ++count;
            }
            if (i >= fv + max_length) {
                const float price_old = load_tm(prices_tm, num_series, series_idx, i - max_length);
                if (!is_nan(price_old)) {
                    sum -= price_old;
                    sum_sq -= price_old * price_old;
                    --count;
                }
            }
            if (i >= warm_raw && count == max_length && isfinite(price_now)) {
                const float mean = sum / (float)max_length;
                float var = sum_sq / (float)max_length - mean * mean;
                var = fmaxf(var, 0.0f);
                if (isfinite(var) && isfinite(mean)) {
                    const float diff = price_now - mean;
                    const float diff2 = diff * diff;
                    const float inner2 = 0.0625f * var;
                    const float outer2 = 3.0625f * var;
                    if (diff2 <= inner2)          length_f += 1.0f;
                    else if (diff2 > outer2)      length_f -= 1.0f;
                    length_f = clampf(length_f, (float)min_length, (float)max_length);

                    len_r = (int)floorf(length_f + 0.5f);
                    if (len_r < min_length) len_r = min_length;
                    if (len_r > max_length) len_r = max_length;
                    if (len_r < 1)          len_r = 1;

                    if (i + 1 >= len_r) {
                        bool try_volume = false;
                        if (has_volume && volumes_tm != nullptr) {
                            const float vol_now = load_tm(volumes_tm, num_series, series_idx, i);
                            if (!is_nan(vol_now) && vol_now != 0.0f) try_volume = true;
                        }
                        proceed = try_volume ? 2 : 1;
                    }
                }
            }
            if (proceed) window_start_weights = i + 1 - len_r;
        }

        proceed               = __shfl_sync(mask, proceed, 0);
        len_r                 = __shfl_sync(mask, len_r, 0);
        window_start_weights  = __shfl_sync(mask, window_start_weights, 0);
        const float price_now = __shfl_sync(mask, price_now_lane0, 0);

        if (!proceed) continue;

        float mf = 50.0f;
        if (proceed == 2) {
            int len_mf = len_r;
            int available = (i + 1) - fv;
            if (len_mf > available) len_mf = available;

            if (len_mf >= 2) {
                const int start_mf = (i + 1) - len_mf;
                float up_part = 0.0f, down_part = 0.0f;
                int ok = 1;
                const int items = len_mf - 1;
                for (int jj = lane; jj < items; jj += lanes) {
                    const int j = start_mf + 1 + jj;
                    const float px_cur  = load_tm(prices_tm,  num_series, series_idx, j);
                    const float px_prev = load_tm(prices_tm,  num_series, series_idx, j - 1);
                    const float vol_j   = load_tm(volumes_tm, num_series, series_idx, j);
                    if (!isfinite(px_cur) || !isfinite(px_prev) || !isfinite(vol_j)) {
                        ok = 0;
                    } else {
                        const float delta = px_cur - px_prev;
                        if (delta > 0.0f)      up_part   += vol_j;
                        else if (delta < 0.0f) down_part += vol_j;
                    }
                }
                const int all_ok = __all_sync(mask, ok);
                float up_vol   = warp_reduce_sum(up_part, mask);
                float down_vol = warp_reduce_sum(down_part, mask);
                if (all_ok) {
                    const float denom_vol = up_vol + down_vol;
                    if (denom_vol > 0.0f) mf = 100.0f * up_vol / denom_vol;
                } else {
                    proceed = 1;
                }
            } else {
                proceed = 1;
            }
        }
        if (proceed == 1) {
            int window_start_rsi = (i + 1) - (len_r * 2);
            if (window_start_rsi < 0) window_start_rsi = 0;
            const int window_end = i + 1;


            if (lane == 0) {
                mf = compute_rsi_tm(prices_tm, num_series, series_idx, window_start_rsi, window_end, len_r);
            }
            mf = __shfl_sync(mask, mf, 0);
        }


        const float mf_scaled = fmaf(mf, 2.0f, -100.0f);
        const float p_local = fmaf(fabsf(mf_scaled), 0.04f, accelerator);

        float ws_part = 0.0f, wt_part = 0.0f;
        for (int j = lane; j < len_r; j += lanes) {
            const int t = window_start_weights + j;
            const float v = load_tm(prices_tm, num_series, series_idx, t);
            if (!is_nan(v)) {
                const int base_pow = len_r - j;
                const float w = uma_weight_pow(base_pow, p_local);
                ws_part = fmaf(v, w, ws_part);
                wt_part += w;
            }
        }
        float ws = warp_reduce_sum(ws_part, mask);
        float wt = warp_reduce_sum(wt_part, mask);

        if (lane == 0) {
            const int out_idx = i * num_series + series_idx;
            float result = price_now;
            if (wt > 0.0f) result = ws / wt;
            raw_out_tm[out_idx] = result;
        }
        __syncwarp(mask);
    }


    if (smooth_length <= 1) {
        for (int t = fv + lane; t < series_len; t += lanes) {
            const int idx = t * num_series + series_idx;
            final_out_tm[idx] = raw_out_tm[idx];
        }
        return;
    }

    const int warm_smooth = (fv + max_length - 1) + smooth_length - 1;
    if (warm_smooth >= series_len) return;

    if (lane == 0) {
        const int lookback = smooth_length - 1;
        const float denom = 0.5f * (float)smooth_length * (float)(smooth_length + 1);

        float weighted_sum = 0.0f;
        float plain_sum = 0.0f;

        for (int j = 0; j < lookback; ++j) {
            const int t = (fv + max_length - 1) + j;
            const float v = raw_out_tm[t * num_series + series_idx];
            weighted_sum += ((float)j + 1.0f) * v;
            plain_sum += v;
        }
        int first_idx = (fv + max_length - 1) + lookback;
        const float first_val = raw_out_tm[first_idx * num_series + series_idx];
        weighted_sum += (float)smooth_length * first_val;
        plain_sum += first_val;
        final_out_tm[first_idx * num_series + series_idx] = weighted_sum / denom;

        weighted_sum -= plain_sum;
        plain_sum -= raw_out_tm[(fv + max_length - 1) * num_series + series_idx];

        for (int t = first_idx + 1; t < series_len; ++t) {
            const float v_new = raw_out_tm[t * num_series + series_idx];
            weighted_sum += (float)smooth_length * v_new;
            plain_sum += v_new;
            final_out_tm[t * num_series + series_idx] = weighted_sum / denom;
            weighted_sum -= plain_sum;
            const float v_old = raw_out_tm[(t - lookback) * num_series + series_idx];
            plain_sum -= v_old;
        }
    }
}

