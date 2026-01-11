









#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h> 

namespace {

constexpr int WARP = 32;

__device__ __forceinline__ double kama_const_max() {
    return 2.0 / 31.0; 
}

__device__ __forceinline__ double kama_const_diff() {
    return (2.0 / 3.0) - kama_const_max();
}

__device__ __forceinline__ double warp_sum(double v) {
    unsigned m = __activemask();
    #pragma unroll
    for (int off = WARP >> 1; off > 0; off >>= 1) {
        v += __shfl_down_sync(m, v, off);
    }
    return v;
}

} 





extern "C" __global__ __launch_bounds__(32)
void kama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    
    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid)) ||
        ((first_valid + period) >= series_len);

    const float nan_f = CUDART_NAN_F;

    if (invalid) {
        
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = nan_f;
        }
        return;
    }

    
    const int initial_idx = first_valid + period;

    
    for (int i = threadIdx.x; i < initial_idx; i += blockDim.x) {
        out[base + i] = nan_f;
    }

    
    
    double sum_roc1 = 0.0;
    if (threadIdx.x < WARP) {
        const int lane = threadIdx.x;
        double local = 0.0;
        const int start = first_valid;
        const int end   = first_valid + period; 
        for (int j = start + lane; j < end; j += WARP) {
            const double a = static_cast<double>(prices[j]);
            const double b = static_cast<double>(prices[j + 1]);
            local += fabs(b - a);
        }
        local = warp_sum(local);
        if (lane == 0) sum_roc1 = local;
    }

    
    if (threadIdx.x != 0) return;

    
    double prev_price = static_cast<double>(prices[initial_idx]);
    double prev_kama  = prev_price;
    out[base + initial_idx] = static_cast<float>(prev_kama);

    int    trailing_idx   = first_valid;
    double trailing_value = static_cast<double>(prices[trailing_idx]);

    const double cmax  = kama_const_max();
    const double cdiff = kama_const_diff();

    for (int i = initial_idx + 1; i < series_len; ++i) {
        const double price         = static_cast<double>(prices[i]);
        const double next_trailing = static_cast<double>(prices[trailing_idx + 1]);

        
        sum_roc1 += fabs(price - prev_price) - fabs(next_trailing - trailing_value);

        
        trailing_value = next_trailing;
        trailing_idx  += 1;

        
        const double direction = fabs(price - trailing_value);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);

        double sc = er * cdiff + cmax;
        sc *= sc;

        
        prev_kama = fma(price - prev_kama, sc, prev_kama);
        out[base + i] = static_cast<float>(prev_kama);

        
        prev_price = price;
    }
}







extern "C" __global__ __launch_bounds__(32)
void kama_batch_prefix_f32(const float* __restrict__ prices,
                           const float* __restrict__ prefix_roc1,
                           const int* __restrict__ periods,
                           int series_len,
                           int n_combos,
                           int first_valid,
                           float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    const int initial_idx = first_valid + period; 
    const float nan_f = CUDART_NAN_F;

    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid)) ||
        (initial_idx >= series_len);

    if (invalid) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = nan_f;
        }
        return;
    }

    for (int i = threadIdx.x; i < initial_idx; i += blockDim.x) {
        out[base + i] = nan_f;
    }

    if (threadIdx.x == 0) {
        out[base + initial_idx] = prices[initial_idx];
    }

    
    
    
    const int lane = threadIdx.x;
    if (lane >= WARP) return;

    float prev_kama = prices[initial_idx];
    const float cmax = 2.0f / 31.0f;
    const float cdiff = (2.0f / 3.0f) - cmax;

    int chunk_start = initial_idx + 1;
    for (; (chunk_start + (WARP - 1)) < series_len; chunk_start += WARP) {
        const int t = chunk_start + lane;
        const float price = prices[t];
        const float sum_roc1 = prefix_roc1[t] - prefix_roc1[t - period];
        const float direction = fabsf(price - prices[t - period]);
        const float er = (sum_roc1 == 0.0f) ? 0.0f : (direction / sum_roc1);
        const float tmp = fmaf(er, cdiff, cmax);
        const float sc = tmp * tmp;

        float a = 1.0f - sc;
        float b = sc * price;

        const unsigned m = 0xFFFFFFFFu;
        #pragma unroll
        for (int off = 1; off < WARP; off <<= 1) {
            const float a_up = __shfl_up_sync(m, a, off);
            const float b_up = __shfl_up_sync(m, b, off);
            if (lane >= off) {
                b = fmaf(a, b_up, b);
                a = a * a_up;
            }
        }

        const float x = fmaf(a, prev_kama, b);
        out[base + t] = x;

        prev_kama = __shfl_sync(m, x, WARP - 1);
    }

    
    if (lane == 0) {
        float kama = prev_kama;
        for (int t = chunk_start; t < series_len; ++t) {
            const float price = prices[t];
            const float sum_roc1 = prefix_roc1[t] - prefix_roc1[t - period];
            const float direction = fabsf(price - prices[t - period]);
            const float er = (sum_roc1 == 0.0f) ? 0.0f : (direction / sum_roc1);
            const float tmp = fmaf(er, cdiff, cmax);
            const float sc = tmp * tmp;
            kama = fmaf(price - kama, sc, kama);
            out[base + t] = kama;
        }
    }
}

extern "C" __global__ __launch_bounds__(32)
void kama_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) {
        return;
    }

    const int first_valid = first_valids[series];
    const bool invalid =
        (period <= 0) ||
        (first_valid < 0 || first_valid >= series_len) ||
        (period >= (series_len - first_valid));

    const int initial_idx = first_valid + period;
    const float nan_f = CUDART_NAN_F;

    
    auto at = [num_series](const float* buf, int row, int col) {
        return buf[row * num_series + col];
    };

    
    if (invalid || initial_idx >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + series] = nan_f;
        }
        return;
    }

    
    for (int t = threadIdx.x; t < initial_idx; t += blockDim.x) {
        out_tm[t * num_series + series] = nan_f;
    }

    
    double sum_roc1 = 0.0;
    if (threadIdx.x < WARP) {
        const int lane = threadIdx.x;
        double local = 0.0;
        const int start = first_valid;
        const int end   = first_valid + period; 
        for (int j = start + lane; j < end; j += WARP) {
            const double a = static_cast<double>(at(prices_tm, j,     series));
            const double b = static_cast<double>(at(prices_tm, j + 1, series));
            local += fabs(b - a);
        }
        local = warp_sum(local);
        if (lane == 0) sum_roc1 = local;
    }

    if (threadIdx.x != 0) return;

    double prev_price = static_cast<double>(at(prices_tm, initial_idx, series));
    double prev_kama  = prev_price;
    out_tm[initial_idx * num_series + series] = static_cast<float>(prev_kama);

    int    trailing_idx   = first_valid;
    double trailing_value = static_cast<double>(at(prices_tm, trailing_idx, series));

    const double cmax  = kama_const_max();
    const double cdiff = kama_const_diff();

    for (int t = initial_idx + 1; t < series_len; ++t) {
        const double price         = static_cast<double>(at(prices_tm, t, series));
        const double next_trailing = static_cast<double>(at(prices_tm, trailing_idx + 1, series));

        sum_roc1 += fabs(price - prev_price) - fabs(next_trailing - trailing_value);

        trailing_value = next_trailing;
        trailing_idx  += 1;

        const double direction = fabs(price - trailing_value);
        const double er = (sum_roc1 == 0.0) ? 0.0 : (direction / sum_roc1);

        double sc = er * cdiff + cmax;
        sc *= sc;

        prev_kama = fma(price - prev_kama, sc, prev_kama);
        out_tm[t * num_series + series] = static_cast<float>(prev_kama);

        prev_price = price;
    }
}
