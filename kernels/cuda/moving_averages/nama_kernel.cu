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

__device__ inline void nama_push_max(
    int idx,
    int capacity,
    int* dq,
    int& front,
    int& size,
    const float* __restrict__ prices
) {
    const double cur = static_cast<double>(prices[idx]);
    while (size > 0) {
        const int last_pos = (front + size - 1) % capacity;
        const int last_idx = dq[last_pos];
        const double last_val = static_cast<double>(prices[last_idx]);
        if (last_val <= cur) {
            size -= 1;
        } else {
            break;
        }
    }
    const int insert_pos = (front + size) % capacity;
    dq[insert_pos] = idx;
    size += 1;
}

__device__ inline void nama_push_min(
    int idx,
    int capacity,
    int* dq,
    int& front,
    int& size,
    const float* __restrict__ prices
) {
    const double cur = static_cast<double>(prices[idx]);
    while (size > 0) {
        const int last_pos = (front + size - 1) % capacity;
        const int last_idx = dq[last_pos];
        const double last_val = static_cast<double>(prices[last_idx]);
        if (last_val >= cur) {
            size -= 1;
        } else {
            break;
        }
    }
    const int insert_pos = (front + size) % capacity;
    dq[insert_pos] = idx;
    size += 1;
}

__device__ inline void nama_pop_older(
    int win_start,
    int capacity,
    int* dq,
    int& front,
    int& size
) {
    while (size > 0) {
        const int idx = dq[front];
        if (idx < win_start) {
            front = (front + 1) % capacity;
            size -= 1;
        } else {
            break;
        }
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

    double eff_sum = 0.0;

    for (int j = first_valid; j <= warm; ++j) {
        nama_push_max(j, capacity, dq_max, max_front, max_size, prices);
        nama_push_min(j, capacity, dq_min, min_front, min_size, prices);
        eff_sum += nama_true_range(j, first_valid, has_ohlc, prices, high, low, close);
    }

    if (max_size == 0 || min_size == 0) {
        return;
    }

    const int hi_idx = dq_max[max_front];
    const int lo_idx = dq_min[min_front];
    const double hi = static_cast<double>(prices[hi_idx]);
    const double lo = static_cast<double>(prices[lo_idx]);

    double alpha = 0.0;
    if (eff_sum != 0.0) {
        alpha = (hi - lo) / eff_sum;
    }

    double prev = alpha * static_cast<double>(prices[warm]);
    out[base + warm] = static_cast<float>(prev);

    for (int i = warm + 1; i < series_len; ++i) {
        nama_push_max(i, capacity, dq_max, max_front, max_size, prices);
        nama_push_min(i, capacity, dq_min, min_front, min_size, prices);

        const int win_start = i + 1 - period;
        nama_pop_older(win_start, capacity, dq_max, max_front, max_size);
        nama_pop_older(win_start, capacity, dq_min, min_front, min_size);

        eff_sum += nama_true_range(i, first_valid, has_ohlc, prices, high, low, close);
        eff_sum -= nama_true_range(i - period, first_valid, has_ohlc, prices, high, low, close);

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
        // push_max
        const double cur = static_cast<double>(prices[j]);
        while (max_size > 0) {
            const int last_pos = (max_front + max_size - 1) % capacity;
            const int last_idx = dq_max[last_pos];
            const double last_val = static_cast<double>(prices[last_idx]);
            if (last_val <= cur) {
                max_size -= 1;
            } else {
                break;
            }
        }
        const int insert_pos_max = (max_front + max_size) % capacity;
        dq_max[insert_pos_max] = j;
        max_size += 1;

        // push_min
        while (min_size > 0) {
            const int last_pos = (min_front + min_size - 1) % capacity;
            const int last_idx = dq_min[last_pos];
            const double last_val = static_cast<double>(prices[last_idx]);
            if (last_val >= cur) {
                min_size -= 1;
            } else {
                break;
            }
        }
        const int insert_pos_min = (min_front + min_size) % capacity;
        dq_min[insert_pos_min] = j;
        min_size += 1;
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
        const double cur = static_cast<double>(prices[i]);
        while (max_size > 0) {
            const int last_pos = (max_front + max_size - 1) % capacity;
            const int last_idx = dq_max[last_pos];
            const double last_val = static_cast<double>(prices[last_idx]);
            if (last_val <= cur) { max_size -= 1; } else { break; }
        }
        int insert_pos_max = (max_front + max_size) % capacity;
        dq_max[insert_pos_max] = i;
        max_size += 1;

        while (min_size > 0) {
            const int last_pos = (min_front + min_size - 1) % capacity;
            const int last_idx = dq_min[last_pos];
            const double last_val = static_cast<double>(prices[last_idx]);
            if (last_val >= cur) { min_size -= 1; } else { break; }
        }
        int insert_pos_min = (min_front + min_size) % capacity;
        dq_min[insert_pos_min] = i;
        min_size += 1;

        // pop older indices
        const int win_start = i + 1 - period;
        while (max_size > 0) {
            const int head = dq_max[max_front];
            if (head < win_start) { max_front = (max_front + 1) % capacity; max_size -= 1; } else { break; }
        }
        while (min_size > 0) {
            const int head = dq_min[min_front];
            if (head < win_start) { min_front = (min_front + 1) % capacity; min_size -= 1; } else { break; }
        }

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
    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0 || period > series_len) {
        return;
    }

    const int first_valid = first_valids[series];
    const int warm = first_valid + period - 1;

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * num_series + series] = NAN;
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

    auto push_max = [&](int t_idx) {
        const double cur = static_cast<double>(prices_tm[t_idx * num_series + series]);
        while (max_size > 0) {
            const int last_pos = (max_front + max_size - 1) % capacity;
            const int prev_t = dq_max[last_pos];
            const double prev_val =
                static_cast<double>(prices_tm[prev_t * num_series + series]);
            if (prev_val <= cur) {
                max_size -= 1;
            } else {
                break;
            }
        }
        const int insert_pos = (max_front + max_size) % capacity;
        dq_max[insert_pos] = t_idx;
        max_size += 1;
    };

    auto push_min = [&](int t_idx) {
        const double cur = static_cast<double>(prices_tm[t_idx * num_series + series]);
        while (min_size > 0) {
            const int last_pos = (min_front + min_size - 1) % capacity;
            const int prev_t = dq_min[last_pos];
            const double prev_val =
                static_cast<double>(prices_tm[prev_t * num_series + series]);
            if (prev_val >= cur) {
                min_size -= 1;
            } else {
                break;
            }
        }
        const int insert_pos = (min_front + min_size) % capacity;
        dq_min[insert_pos] = t_idx;
        min_size += 1;
    };

    auto pop_old = [&](int win_start) {
        while (max_size > 0) {
            const int head_t = dq_max[max_front];
            if (head_t < win_start) {
                max_front = (max_front + 1) % capacity;
                max_size -= 1;
            } else {
                break;
            }
        }
        while (min_size > 0) {
            const int head_t = dq_min[min_front];
            if (head_t < win_start) {
                min_front = (min_front + 1) % capacity;
                min_size -= 1;
            } else {
                break;
            }
        }
    };

    double eff_sum = 0.0;
    for (int t = first_valid; t <= warm; ++t) {
        push_max(t);
        push_min(t);
        eff_sum += nama_true_range_tm(
            t,
            first_valid,
            has_ohlc,
            series,
            num_series,
            prices_tm,
            high_tm,
            low_tm,
            close_tm);
    }

    if (max_size == 0 || min_size == 0) {
        return;
    }

    const double hi =
        static_cast<double>(prices_tm[dq_max[max_front] * num_series + series]);
    const double lo =
        static_cast<double>(prices_tm[dq_min[min_front] * num_series + series]);

    double alpha = 0.0;
    if (eff_sum != 0.0) {
        alpha = (hi - lo) / eff_sum;
    }

    double prev = alpha * static_cast<double>(prices_tm[warm * num_series + series]);
    out_tm[warm * num_series + series] = static_cast<float>(prev);

    for (int t = warm + 1; t < series_len; ++t) {
        push_max(t);
        push_min(t);
        pop_old(t + 1 - period);

        eff_sum += nama_true_range_tm(
            t,
            first_valid,
            has_ohlc,
            series,
            num_series,
            prices_tm,
            high_tm,
            low_tm,
            close_tm);
        eff_sum -= nama_true_range_tm(
            t - period,
            first_valid,
            has_ohlc,
            series,
            num_series,
            prices_tm,
            high_tm,
            low_tm,
            close_tm);

        if (max_size == 0 || min_size == 0) {
            continue;
        }

        const double hi_cur =
            static_cast<double>(prices_tm[dq_max[max_front] * num_series + series]);
        const double lo_cur =
            static_cast<double>(prices_tm[dq_min[min_front] * num_series + series]);

        alpha = 0.0;
        if (eff_sum != 0.0) {
            alpha = (hi_cur - lo_cur) / eff_sum;
        }

        const double src = static_cast<double>(prices_tm[t * num_series + series]);
        prev = alpha * src + (1.0 - alpha) * prev;
        out_tm[t * num_series + series] = static_cast<float>(prev);
    }
}
