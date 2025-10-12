// CUDA kernels for the New Adaptive Moving Average (NAMA).
//
// Kernels operate on FP32 buffers (matching the public interface) while
// promoting arithmetic that benefits from extended precision to FP64. We keep
// shared data such as the sliding-window deques in shared memory to avoid
// repeated global reads.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ inline double nama_true_range(
    int idx,
    int first_valid,
    int has_ohlc,
    const float* __restrict__ prices,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close
) {
    if (has_ohlc) {
        const double h = static_cast<double>(high[idx]);
        const double l = static_cast<double>(low[idx]);
        if (idx == first_valid) {
            return h - l;
        }
        const double prev_close = static_cast<double>(close[idx - 1]);
        const double hl = h - l;
        const double hc = fabs(h - prev_close);
        const double lc = fabs(l - prev_close);
        return fmax(hl, fmax(hc, lc));
    }

    if (idx == first_valid) {
        return 0.0;
    }
    const double cur = static_cast<double>(prices[idx]);
    const double prev = static_cast<double>(prices[idx - 1]);
    return fabs(cur - prev);
}

// --- helpers: wrap-around without modulo ------------------------------------
__device__ __forceinline__ int add_wrap(int a, int b, int cap) {
    int s = a + b;
    return (s >= cap) ? (s - cap) : s;
}

__device__ __forceinline__ int last_pos_from_front_and_size(int front, int size, int cap) {
    int lp = front + size - 1;
    return (lp >= cap) ? (lp - cap) : lp;
}

__device__ __forceinline__ void dq_push_max(
    int idx, int cap, int* dq, int& front, int& size,
    const float* __restrict__ prices)
{
    const double cur = static_cast<double>(prices[idx]);
    while (size > 0) {
        const int lp = last_pos_from_front_and_size(front, size, cap);
        const int last_idx = dq[lp];
        const double last_val = static_cast<double>(prices[last_idx]);
        if (last_val <= cur) { size -= 1; } else { break; }
    }
    const int ip = add_wrap(front, size, cap);
    dq[ip] = idx;
    size += 1;
}

__device__ __forceinline__ void dq_push_min(
    int idx, int cap, int* dq, int& front, int& size,
    const float* __restrict__ prices)
{
    const double cur = static_cast<double>(prices[idx]);
    while (size > 0) {
        const int lp = last_pos_from_front_and_size(front, size, cap);
        const int last_idx = dq[lp];
        const double last_val = static_cast<double>(prices[last_idx]);
        if (last_val >= cur) { size -= 1; } else { break; }
    }
    const int ip = add_wrap(front, size, cap);
    dq[ip] = idx;
    size += 1;
}

__device__ __forceinline__ void dq_pop_older(
    int win_start, int cap, int* dq, int& front, int& size)
{
    while (size > 0) {
        const int head_idx = dq[front];
        if (head_idx < win_start) {
            front = (front + 1 == cap) ? 0 : front + 1;
            size -= 1;
        } else break;
    }
}

extern "C" __global__
void nama_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ high,
                    const float* __restrict__ low,
                    const float* __restrict__ close,
                    int has_ohlc,
                    const int* __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > series_len) return;

    const int warm = first_valid + period - 1;
    const int base = combo * series_len;

    // Initialize outputs to NaN cooperatively
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) out[base + idx] = NAN;
    __syncthreads();

    if (threadIdx.x != 0 || warm >= series_len) return;

    // --- shared memory layout: [dq_max | dq_min | tr_ring]
    extern __shared__ int shared_i[];
    const int cap = period + 1;                    // deque capacity
    int* dq_max = shared_i;                        // int[cap]
    int* dq_min = shared_i + cap;                  // int[cap]
    float* tr_ring = reinterpret_cast<float*>(shared_i + 2*cap); // float[period]

    int max_front = 0, max_size = 0;
    int min_front = 0, min_size = 0;

    // --- Seed window [first_valid..warm] ---------------------------------
    double eff_sum = 0.0;
    int wr = 0; // write pointer into TR ring (points to oldest)

    if (has_ohlc) {
        float prev_c = 0.0f;
        for (int j = first_valid; j <= warm; ++j) {
            dq_push_max(j, cap, dq_max, max_front, max_size, prices);
            dq_push_min(j, cap, dq_min, min_front, min_size, prices);

            float tr;
            if (j == first_valid) {
                tr = high[j] - low[j];
            } else {
                const float h = high[j], l = low[j];
                const float hc = fabsf(h - prev_c);
                const float lc = fabsf(l - prev_c);
                const float hl = h - l;
                tr = fmaxf(hl, fmaxf(hc, lc));
            }
            tr_ring[wr] = tr;
            wr = (wr + 1 == period) ? 0 : wr + 1;
            eff_sum += static_cast<double>(tr);

            prev_c = close[j];
        }
    } else {
        float prev_p = 0.0f;
        for (int j = first_valid; j <= warm; ++j) {
            dq_push_max(j, cap, dq_max, max_front, max_size, prices);
            dq_push_min(j, cap, dq_min, min_front, min_size, prices);

            float tr;
            if (j == first_valid) {
                tr = 0.0f;
            } else {
                const float cur = prices[j];
                tr = fabsf(cur - prev_p);
            }
            tr_ring[wr] = tr;
            wr = (wr + 1 == period) ? 0 : wr + 1;
            eff_sum += static_cast<double>(tr);

            prev_p = prices[j];
        }
    }

    if (max_size == 0 || min_size == 0) return;

    const double hi0 = static_cast<double>(prices[dq_max[max_front]]);
    const double lo0 = static_cast<double>(prices[dq_min[min_front]]);
    double alpha = (eff_sum != 0.0) ? (hi0 - lo0) / eff_sum : 0.0;
    double prev = alpha * static_cast<double>(prices[warm]);
    out[base + warm] = static_cast<float>(prev);

    // --- Rolling loop -----------------------------------------------------
    if (has_ohlc) {
        float prev_c = close[warm];
        for (int i = warm + 1; i < series_len; ++i) {
            dq_push_max(i, cap, dq_max, max_front, max_size, prices);
            dq_push_min(i, cap, dq_min, min_front, min_size, prices);

            const int win_start = i + 1 - period;
            dq_pop_older(win_start, cap, dq_max, max_front, max_size);
            dq_pop_older(win_start, cap, dq_min, min_front, min_size);

            // TR ring update (OHLC)
            float tr_new;
            {
                const float h = high[i], l = low[i];
                const float hc = fabsf(h - prev_c);
                const float lc = fabsf(l - prev_c);
                const float hl = h - l;
                tr_new = fmaxf(hl, fmaxf(hc, lc));
            }
            const float tr_old = tr_ring[wr];
            tr_ring[wr] = tr_new;
            wr = (wr + 1 == period) ? 0 : wr + 1;
            eff_sum += static_cast<double>(tr_new) - static_cast<double>(tr_old);
            prev_c = close[i];

            if (max_size == 0 || min_size == 0) continue;
            const double hi = static_cast<double>(prices[dq_max[max_front]]);
            const double lo = static_cast<double>(prices[dq_min[min_front]]);
            alpha = (eff_sum != 0.0) ? (hi - lo) / eff_sum : 0.0;

            const double src = static_cast<double>(prices[i]);
            prev = alpha * src + (1.0 - alpha) * prev;
            out[base + i] = static_cast<float>(prev);
        }
    } else {
        float prev_p = prices[warm];
        for (int i = warm + 1; i < series_len; ++i) {
            dq_push_max(i, cap, dq_max, max_front, max_size, prices);
            dq_push_min(i, cap, dq_min, min_front, min_size, prices);

            const int win_start = i + 1 - period;
            dq_pop_older(win_start, cap, dq_max, max_front, max_size);
            dq_pop_older(win_start, cap, dq_min, min_front, min_size);

            // TR ring update (close-only)
            const float cur = prices[i];
            const float tr_new = fabsf(cur - prev_p);
            const float tr_old = tr_ring[wr];
            tr_ring[wr] = tr_new;
            wr = (wr + 1 == period) ? 0 : wr + 1;
            eff_sum += static_cast<double>(tr_new) - static_cast<double>(tr_old);
            prev_p = cur;

            if (max_size == 0 || min_size == 0) continue;
            const double hi = static_cast<double>(prices[dq_max[max_front]]);
            const double lo = static_cast<double>(prices[dq_min[min_front]]);
            alpha = (eff_sum != 0.0) ? (hi - lo) / eff_sum : 0.0;

            const double src = static_cast<double>(prices[i]);
            prev = alpha * src + (1.0 - alpha) * prev;
            out[base + i] = static_cast<float>(prev);
        }
    }
}

// Prefix-optimized batch kernel (degenerate TR only):
// Uses host-precomputed prefix of |p[t] - p[t-1]| (NaN-insensitive) to seed the
// initial eff_sum for the warmup window in O(1), then updates eff_sum using
// prefix differences. Mirrors the CPU batch optimization that precomputes TR
// once and reuses it across rows.
//
// prefix_tr must have length = series_len + 1 and follow:
//   prefix_tr[0] = 0
//   prefix_tr[t] = sum_{k=1..t} |p[k] - p[k-1]|  (with NaN-insensitive accumulation)
extern "C" __global__
void nama_batch_prefix_f32(const float* __restrict__ prices,
                           const float* __restrict__ prefix_tr,
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
    if (period <= 0 || period > series_len) {
        return;
    }
    const int warm = first_valid + period - 1;
    const int base = combo * series_len;

    // Initialize NaNs up-front (all threads cooperate)
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0 || warm >= series_len) {
        return;
    }

    extern __shared__ int shared_i[];
    const int capacity = period + 1;
    int* dq_max = shared_i;
    int* dq_min = shared_i + capacity;

    int max_front = 0;
    int max_size = 0;
    int min_front = 0;
    int min_size = 0;

    // Seed deques over [first_valid..warm]
    for (int j = first_valid; j <= warm; ++j) {
        dq_push_max(j, capacity, dq_max, max_front, max_size, prices);
        dq_push_min(j, capacity, dq_min, min_front, min_size, prices);
    }

    if (max_size == 0 || min_size == 0) {
        return;
    }

    // Seed eff_sum using prefix differences: sum_{j=first..warm} TR[j]
    // For degenerate TR, TR[first] = 0, so this equals prefix_tr[warm] - prefix_tr[first]
    double eff_sum = static_cast<double>(prefix_tr[warm] - prefix_tr[first_valid]);

    const double hi = static_cast<double>(prices[dq_max[max_front]]);
    const double lo = static_cast<double>(prices[dq_min[min_front]]);
    double alpha = 0.0;
    if (eff_sum != 0.0) {
        alpha = (hi - lo) / eff_sum;
    }
    double prev = alpha * static_cast<double>(prices[warm]);
    out[base + warm] = static_cast<float>(prev);

    for (int i = warm + 1; i < series_len; ++i) {
        // push current index
        dq_push_max(i, capacity, dq_max, max_front, max_size, prices);
        dq_push_min(i, capacity, dq_min, min_front, min_size, prices);

        // pop older indices
        const int win_start = i + 1 - period;
        dq_pop_older(win_start, capacity, dq_max, max_front, max_size);
        dq_pop_older(win_start, capacity, dq_min, min_front, min_size);

        // Update eff_sum using prefix differences (degenerate TR):
        // eff_sum += TR[i] - TR[i - period]
        // TR[t] = prefix_tr[t] - prefix_tr[t-1]
        const double tr_add = static_cast<double>(prefix_tr[i] - prefix_tr[i - 1]);
        const double tr_sub = static_cast<double>(prefix_tr[i - period] - prefix_tr[i - period - 1]);
        eff_sum = eff_sum + tr_add - tr_sub;

        if (max_size == 0 || min_size == 0) {
            continue;
        }
        const double hi_cur = static_cast<double>(prices[dq_max[max_front]]);
        const double lo_cur = static_cast<double>(prices[dq_min[min_front]]);
        alpha = 0.0;
        if (eff_sum != 0.0) {
            alpha = (hi_cur - lo_cur) / eff_sum;
        }

        const double src = static_cast<double>(prices[i]);
        prev = alpha * src + (1.0 - alpha) * prev;
        out[base + i] = static_cast<float>(prev);
    }
}

__device__ inline double nama_true_range_tm(
    int t,
    int first_valid_t,
    int has_ohlc,
    int series,
    int num_series,
    const float* __restrict__ prices_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm
) {
    const int idx = t * num_series + series;
    if (has_ohlc) {
        const double h = static_cast<double>(high_tm[idx]);
        const double l = static_cast<double>(low_tm[idx]);
        if (t == first_valid_t) {
            return h - l;
        }
        const double prev_close = static_cast<double>(close_tm[(t - 1) * num_series + series]);
        const double hl = h - l;
        const double hc = fabs(h - prev_close);
        const double lc = fabs(l - prev_close);
        return fmax(hl, fmax(hc, lc));
    }

    if (t == first_valid_t) {
        return 0.0;
    }
    const double cur = static_cast<double>(prices_tm[idx]);
    const double prev = static_cast<double>(prices_tm[(t - 1) * num_series + series]);
    return fabs(cur - prev);
}

extern "C" __global__
void nama_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int has_ohlc,
    int num_series,
    int series_len,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x;
    if (s >= num_series || period <= 0 || period > series_len) return;

    const int first_valid = first_valids[s];
    const int warm = first_valid + period - 1;

    for (int t = threadIdx.x; t < series_len; t += blockDim.x)
        out_tm[t * num_series + s] = NAN;
    __syncthreads();

    if (threadIdx.x != 0 || warm >= series_len) return;

    extern __shared__ int shared_i[];
    const int cap = period + 1;
    int* dq_max = shared_i;
    int* dq_min = shared_i + cap;
    float* tr_ring = reinterpret_cast<float*>(shared_i + 2*cap);

    int max_front = 0, max_size = 0;
    int min_front = 0, min_size = 0;

    auto price_at = [&](int t)->float { return prices_tm[t * num_series + s]; };
    auto push_max_tm = [&](int t_idx) {
        const double cur = static_cast<double>(price_at(t_idx));
        while (max_size > 0) {
            const int lp = last_pos_from_front_and_size(max_front, max_size, cap);
            const int last_idx = dq_max[lp];
            const double last_val = static_cast<double>(price_at(last_idx));
            if (last_val <= cur) { max_size -= 1; } else { break; }
        }
        const int ip = add_wrap(max_front, max_size, cap);
        dq_max[ip] = t_idx;
        max_size += 1;
    };
    auto push_min_tm = [&](int t_idx) {
        const double cur = static_cast<double>(price_at(t_idx));
        while (min_size > 0) {
            const int lp = last_pos_from_front_and_size(min_front, min_size, cap);
            const int last_idx = dq_min[lp];
            const double last_val = static_cast<double>(price_at(last_idx));
            if (last_val >= cur) { min_size -= 1; } else { break; }
        }
        const int ip = add_wrap(min_front, min_size, cap);
        dq_min[ip] = t_idx;
        min_size += 1;
    };

    // Seed window
    double eff_sum = 0.0;
    int wr = 0;

    if (has_ohlc) {
        float prev_c = 0.0f;
        for (int t = first_valid; t <= warm; ++t) {
            push_max_tm(t);
            push_min_tm(t);

            float tr;
            if (t == first_valid) {
                const int idx = t * num_series + s;
                tr = high_tm[idx] - low_tm[idx];
            } else {
                const int idx = t * num_series + s;
                const float h = high_tm[idx], l = low_tm[idx];
                const float hc = fabsf(h - prev_c);
                const float lc = fabsf(l - prev_c);
                const float hl = h - l;
                tr = fmaxf(hl, fmaxf(hc, lc));
            }
            tr_ring[wr] = tr; wr = (wr + 1 == period) ? 0 : wr + 1;
            eff_sum += static_cast<double>(tr);

            prev_c = close_tm[t * num_series + s];
        }
    } else {
        float prev_p = 0.0f;
        for (int t = first_valid; t <= warm; ++t) {
            push_max_tm(t);
            push_min_tm(t);

            float tr = (t == first_valid) ? 0.0f : fabsf(price_at(t) - prev_p);
            tr_ring[wr] = tr; wr = (wr + 1 == period) ? 0 : wr + 1;
            eff_sum += static_cast<double>(tr);

            prev_p = price_at(t);
        }
    }

    if (max_size == 0 || min_size == 0) return;

    const double hi0 = static_cast<double>(price_at(dq_max[max_front]));
    const double lo0 = static_cast<double>(price_at(dq_min[min_front]));
    double alpha = (eff_sum != 0.0) ? (hi0 - lo0) / eff_sum : 0.0;

    double prev = alpha * static_cast<double>(price_at(warm));
    out_tm[warm * num_series + s] = static_cast<float>(prev);

    if (has_ohlc) {
        float prev_c = close_tm[warm * num_series + s];
        for (int t = warm + 1; t < series_len; ++t) {
            push_max_tm(t);
            push_min_tm(t);

            const int win_start = t + 1 - period;
            dq_pop_older(win_start, cap, dq_max, max_front, max_size);
            dq_pop_older(win_start, cap, dq_min, min_front, min_size);

            float tr_new;
            {
                const int idx = t * num_series + s;
                const float h = high_tm[idx], l = low_tm[idx];
                const float hc = fabsf(h - prev_c);
                const float lc = fabsf(l - prev_c);
                const float hl = h - l;
                tr_new = fmaxf(hl, fmaxf(hc, lc));
            }
            const float tr_old = tr_ring[wr];
            tr_ring[wr] = tr_new;
            wr = (wr + 1 == period) ? 0 : wr + 1;
            eff_sum += static_cast<double>(tr_new) - static_cast<double>(tr_old);
            prev_c = close_tm[t * num_series + s];

            if (max_size == 0 || min_size == 0) continue;
            const double hi = static_cast<double>(price_at(dq_max[max_front]));
            const double lo = static_cast<double>(price_at(dq_min[min_front]));
            alpha = (eff_sum != 0.0) ? (hi - lo) / eff_sum : 0.0;

            const double src = static_cast<double>(price_at(t));
            prev = alpha * src + (1.0 - alpha) * prev;
            out_tm[t * num_series + s] = static_cast<float>(prev);
        }
    } else {
        float prev_p = price_at(warm);
        for (int t = warm + 1; t < series_len; ++t) {
            push_max_tm(t);
            push_min_tm(t);

            const int win_start = t + 1 - period;
            dq_pop_older(win_start, cap, dq_max, max_front, max_size);
            dq_pop_older(win_start, cap, dq_min, min_front, min_size);

            const float cur = price_at(t);
            const float tr_new = fabsf(cur - prev_p);
            const float tr_old = tr_ring[wr];
            tr_ring[wr] = tr_new;
            wr = (wr + 1 == period) ? 0 : wr + 1;
            eff_sum += static_cast<double>(tr_new) - static_cast<double>(tr_old);
            prev_p = cur;

            if (max_size == 0 || min_size == 0) continue;
            const double hi = static_cast<double>(price_at(dq_max[max_front]));
            const double lo = static_cast<double>(price_at(dq_min[min_front]));
            alpha = (eff_sum != 0.0) ? (hi - lo) / eff_sum : 0.0;

            const double src = static_cast<double>(cur);
            prev = alpha * src + (1.0 - alpha) * prev;
            out_tm[t * num_series + s] = static_cast<float>(prev);
        }
    }
}
