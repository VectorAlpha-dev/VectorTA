
#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

extern "C" __global__
void vama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ base_periods,
                    const int* __restrict__ vol_periods,
                    const float* __restrict__ alphas,
                    const float* __restrict__ betas,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ ema_buf,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    if (series_len <= 0) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int base_period = base_periods[combo];
    const int vol_period = vol_periods[combo];
    if (base_period <= 0 || vol_period <= 0) {
        return;
    }


    const double alpha = 2.0 / (static_cast<double>(base_period) + 1.0);
    const double beta  = 1.0 - alpha;
    const int base_offset = combo * series_len;

    if (threadIdx.x != 0) {
        return;
    }




    for (int t = 0; t < first_valid; ++t) {
        out[base_offset + t] = NAN;
    }

    const float first_price_f = prices[first_valid];
    double mean = static_cast<double>(first_price_f);
    int valid_count = 1;

    int warm_base_end = first_valid + base_period;
    if (warm_base_end > series_len) {
        warm_base_end = series_len;
    }

    const int max_period = (base_period > vol_period) ? base_period : vol_period;
    const int warm = first_valid + max_period - 1;
    if (warm >= series_len) {
        return;
    }







    extern __shared__ unsigned char smem_rb[];
    double* dq_max_vals = reinterpret_cast<double*>(smem_rb);
    int* dq_max_idx = reinterpret_cast<int*>(dq_max_vals + vol_period);
    uintptr_t p = reinterpret_cast<uintptr_t>(dq_max_idx + vol_period);
    p = (p + 7u) & ~uintptr_t(7u);
    double* dq_min_vals = reinterpret_cast<double*>(p);
    int* dq_min_idx = reinterpret_cast<int*>(dq_min_vals + vol_period);
    int headMax = 0, tailMax = 0;
    int headMin = 0, tailMin = 0;



    int i = first_valid;
    double ema_d = mean;
    double prev = mean;
    for (; i < warm_base_end; ++i) {
        const float price_f = prices[i];
        if (i != first_valid && isfinite(price_f)) {
            const double prev_total = mean * static_cast<double>(valid_count);
            ++valid_count;
            mean = (prev_total + static_cast<double>(price_f)) / static_cast<double>(valid_count);
            prev = mean;
        }
        ema_d = mean;

        const int available = i + 1 - first_valid;
        const int window_len = (available < vol_period) ? available : vol_period;
        const int start = i + 1 - window_len;

        while (headMax != tailMax) {
            int idx = dq_max_idx[headMax];
            if (idx >= start) break;
            headMax = (headMax + 1) % vol_period;
        }
        while (headMin != tailMin) {
            int idx = dq_min_idx[headMin];
            if (idx >= start) break;
            headMin = (headMin + 1) % vol_period;
        }

        if (isfinite(price_f) && isfinite(static_cast<float>(ema_d))) {
            const double d = static_cast<double>(price_f) - ema_d;

            while (headMax != tailMax) {
                int last = (tailMax == 0 ? vol_period - 1 : tailMax - 1);
                if (dq_max_vals[last] <= d) {
                    tailMax = last;
                } else break;
            }
            dq_max_vals[tailMax] = d;
            dq_max_idx[tailMax] = i;
            tailMax = (tailMax + 1) % vol_period;

            while (headMin != tailMin) {
                int last = (tailMin == 0 ? vol_period - 1 : tailMin - 1);
                if (dq_min_vals[last] >= d) {
                    tailMin = last;
                } else break;
            }
            dq_min_vals[tailMin] = d;
            dq_min_idx[tailMin] = i;
            tailMin = (tailMin + 1) % vol_period;
        }

        if (i < warm) {
            out[base_offset + i] = NAN;
        } else if (!isfinite(static_cast<float>(ema_d)) || headMax == tailMax || headMin == tailMin) {
            out[base_offset + i] = static_cast<float>(ema_d);
        } else {
            const double adj = 0.5 * (dq_max_vals[headMax] + dq_min_vals[headMin]);
            out[base_offset + i] = static_cast<float>(ema_d + adj);
        }
    }


    for (; i < series_len; ++i) {
        const float price_f = prices[i];
        if (isfinite(price_f)) {
            prev = fma(beta, prev, alpha * static_cast<double>(price_f));
        }
        ema_d = prev;

        const int available = i + 1 - first_valid;
        const int window_len = (available < vol_period) ? available : vol_period;
        const int start = i + 1 - window_len;

        while (headMax != tailMax) {
            int idx = dq_max_idx[headMax];
            if (idx >= start) break;
            headMax = (headMax + 1) % vol_period;
        }
        while (headMin != tailMin) {
            int idx = dq_min_idx[headMin];
            if (idx >= start) break;
            headMin = (headMin + 1) % vol_period;
        }

        if (isfinite(price_f) && isfinite(static_cast<float>(ema_d))) {
            const double d = static_cast<double>(price_f) - ema_d;

            while (headMax != tailMax) {
                int last = (tailMax == 0 ? vol_period - 1 : tailMax - 1);
                if (dq_max_vals[last] <= d) {
                    tailMax = last;
                } else break;
            }
            dq_max_vals[tailMax] = d;
            dq_max_idx[tailMax] = i;
            tailMax = (tailMax + 1) % vol_period;

            while (headMin != tailMin) {
                int last = (tailMin == 0 ? vol_period - 1 : tailMin - 1);
                if (dq_min_vals[last] >= d) {
                    tailMin = last;
                } else break;
            }
            dq_min_vals[tailMin] = d;
            dq_min_idx[tailMin] = i;
            tailMin = (tailMin + 1) % vol_period;
        }

        if (i < warm) {
            out[base_offset + i] = NAN;
        } else if (!isfinite(static_cast<float>(ema_d)) || headMax == tailMax || headMin == tailMin) {
            out[base_offset + i] = static_cast<float>(ema_d);
        } else {
            const double adj = 0.5 * (dq_max_vals[headMax] + dq_min_vals[headMin]);
            out[base_offset + i] = static_cast<float>(ema_d + adj);
        }
    }
}

extern "C" __global__
void vama_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    int base_period,
                                    int vol_period,
                                    float alpha,
                                    float beta,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ ema_tm,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        const int offset = t * num_series + series_idx;
        ema_tm[offset] = NAN;
        out_tm[offset] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    if (base_period <= 0 || vol_period <= 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int first_valid = first_valids[series_idx];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int stride = num_series;
    const int first_idx = first_valid * stride + series_idx;
    float first_price = prices_tm[first_idx];
    if (!isfinite(first_price)) {
        for (int t = first_valid + 1; t < series_len; ++t) {
            const float candidate = prices_tm[t * stride + series_idx];
            if (isfinite(candidate)) {
                first_price = candidate;
                break;
            }
        }
    }
    if (!isfinite(first_price)) {
        return;
    }

    double mean = static_cast<double>(first_price);
    int valid_count = 1;
    ema_tm[first_idx] = static_cast<float>(mean);

    int warm_base_end = first_valid + base_period;
    if (warm_base_end > series_len) {
        warm_base_end = series_len;
    }

    for (int t = first_valid + 1; t < warm_base_end; ++t) {
        const float price_f = prices_tm[t * stride + series_idx];
        if (isfinite(price_f)) {
            const double prev_total = mean * static_cast<double>(valid_count);
            ++valid_count;
            mean = (prev_total + static_cast<double>(price_f)) / static_cast<double>(valid_count);
        }
        ema_tm[t * stride + series_idx] = static_cast<float>(mean);
    }

    double prev = mean;
    for (int t = warm_base_end; t < series_len; ++t) {
        const float price_f = prices_tm[t * stride + series_idx];
        if (isfinite(price_f)) {
            prev = static_cast<double>(beta) * prev + static_cast<double>(alpha) * static_cast<double>(price_f);
        }
        ema_tm[t * stride + series_idx] = static_cast<float>(prev);
    }

    const int max_period = (base_period > vol_period) ? base_period : vol_period;
    const int warm = first_valid + max_period - 1;
    if (warm >= series_len) {
        return;
    }


    extern __shared__ unsigned char smem_rb2[];
    float* dq_max_vals = reinterpret_cast<float*>(smem_rb2);
    int*   dq_max_idx  = reinterpret_cast<int*>(dq_max_vals + vol_period);
    float* dq_min_vals = reinterpret_cast<float*>(dq_max_idx  + vol_period);
    int*   dq_min_idx  = reinterpret_cast<int*>(dq_min_vals + vol_period);
    int headMax = 0, tailMax = 0;
    int headMin = 0, tailMin = 0;

    for (int t = first_valid; t < series_len; ++t) {
        const int available = t + 1 - first_valid;
        const int window_len = (available < vol_period) ? available : vol_period;
        const int start = t + 1 - window_len;

        while (headMax != tailMax) {
            int idx = dq_max_idx[headMax];
            if (idx >= start) break;
            headMax = (headMax + 1) % vol_period;
        }
        while (headMin != tailMin) {
            int idx = dq_min_idx[headMin];
            if (idx >= start) break;
            headMin = (headMin + 1) % vol_period;
        }

        const int off = t * stride + series_idx;
        const float mid = ema_tm[off];
        const float p = prices_tm[off];
        if (isfinite(mid) && isfinite(p)) {
            const float d = p - mid;
            while (headMax != tailMax) {
                int last = (tailMax == 0 ? vol_period - 1 : tailMax - 1);
                if (dq_max_vals[last] <= d) {
                    tailMax = last;
                } else break;
            }
            dq_max_vals[tailMax] = d;
            dq_max_idx[tailMax]  = t;
            tailMax = (tailMax + 1) % vol_period;

            while (headMin != tailMin) {
                int last = (tailMin == 0 ? vol_period - 1 : tailMin - 1);
                if (dq_min_vals[last] >= d) {
                    tailMin = last;
                } else break;
            }
            dq_min_vals[tailMin] = d;
            dq_min_idx[tailMin]  = t;
            tailMin = (tailMin + 1) % vol_period;
        }

        if (t >= warm) {
            if (!isfinite(mid) || headMax == tailMax || headMin == tailMin) {
                out_tm[off] = mid;
            } else {
                out_tm[off] = mid + 0.5f * (dq_max_vals[headMax] + dq_min_vals[headMin]);
            }
        }
    }
}
