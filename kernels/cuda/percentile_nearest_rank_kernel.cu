





















#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}



static __device__ __forceinline__ bool is_nan(float x) { return !(x == x); }

static __device__ __forceinline__ int next_pow2(int x) {
    x = x - 1; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}



static __device__ __forceinline__ void cx_nan_last(float &a, float &b, bool up) {
    float aa = a, bb = b;
    const bool a_nan = is_nan(aa);
    const bool b_nan = is_nan(bb);

    float lo, hi;
    if (a_nan & b_nan) {
        lo = aa; hi = bb;
    } else if (a_nan) {
        lo = bb; hi = aa;
    } else if (b_nan) {
        lo = aa; hi = bb;
    } else {
        if (aa <= bb) { lo = aa; hi = bb; }
        else           { lo = bb; hi = aa; }
    }
    if (up) { a = lo; b = hi; } else { a = hi; b = lo; }
}




static __device__ void bitonic_sort_shared_nan_last(float* buf, int size) {
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;


    for (int k = 2; k <= size; k <<= 1) {

        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = tid; idx < size; idx += nthreads) {
                int ixj = idx ^ j;
                if (ixj > idx) {
                    bool up = ((idx & k) == 0);
                    float a = buf[idx];
                    float b = buf[ixj];
                    cx_nan_last(a, b, up);
                    buf[idx]  = a;
                    buf[ixj]  = b;
                }
            }
            __syncthreads();
        }
    }
}


static __device__ __forceinline__ int insert_sorted(float* sorted, int wl, float v) {

    int lo = 0, hi = wl;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        float mv = sorted[mid];
        if (!(v < mv)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    for (int i = wl; i > lo; --i) sorted[i] = sorted[i - 1];
    sorted[lo] = v;
    return wl + 1;
}


static __device__ __forceinline__ int erase_sorted(float* sorted, int wl, float v) {

    int lo = 0, hi = wl;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        float mv = sorted[mid];
        if (mv < v) {
            lo = mid + 1;
        } else if (v < mv) {
            hi = mid;
        } else {

            for (int i = mid; i + 1 < wl; ++i) sorted[i] = sorted[i + 1];
            return wl - 1;
        }
    }
    return wl;
}

static __device__ __forceinline__ int nearest_rank_index(float p_wl, int wl) {

    float raw_f = floorf(p_wl + 0.5f) - 1.0f;
    int   raw   = (int)raw_f;
    if (raw <= 0) return 0;
    if (raw >= wl) return wl - 1;
    return raw;
}

static __device__ __forceinline__ int nearest_rank_index_from_frac(float p_frac, int wl) {

    float raw_f = floorf(fmaf(p_frac, (float)wl, 0.5f)) - 1.0f;
    int   raw   = (int)raw_f;
    if (raw <= 0)     return 0;
    if (raw >= wl)    return wl - 1;
    return raw;
}

extern "C" __global__
void percentile_nearest_rank_batch_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ lengths,
    const float* __restrict__ percentages,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out,
    float* __restrict__ scratch,
    int max_length
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int length = lengths[combo];
    const float perc = percentages[combo];
    float* out_row = out + (size_t)combo * series_len;
    float* sorted = scratch + (size_t)combo * max_length;

    if (UNLIKELY(length <= 0 || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }

    const int warm = first_valid + length - 1;

    for (int i = 0; i < warm && i < series_len; ++i) out_row[i] = qnan32();
    if (warm >= series_len) return;

    const float p_frac = perc * 0.01f;
    const int window_start0 = warm + 1 - length;


    int wl = 0;
    for (int idx = window_start0; idx <= warm; ++idx) {
        float v = prices[idx];
        if (!isnan(v)) wl = insert_sorted(sorted, wl, v);
    }
    const int k_full = nearest_rank_index(p_frac * (float)length, length);

    int i = warm;
    while (true) {
        if (wl == 0) {
            out_row[i] = qnan32();
        } else {
            int k = (wl == length) ? k_full : nearest_rank_index(p_frac * (float)wl, wl);
            out_row[i] = sorted[k];
        }

        if (i + 1 >= series_len) break;

        int out_idx = i + 1 - length;
        float v_out = prices[out_idx];
        if (!isnan(v_out)) wl = erase_sorted(sorted, wl, v_out);
        float v_in = prices[i + 1];
        if (!isnan(v_in)) wl = insert_sorted(sorted, wl, v_in);
        i += 1;
    }
}


extern "C" __global__
void percentile_nearest_rank_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int length,
    float percentage,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm,
    float* __restrict__ scratch_cols,
    int max_length
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    const int fv = first_valids[series];
    if (UNLIKELY(length <= 0 || fv < 0 || fv >= rows)) {
        for (int t = 0; t < rows; ++t) out_tm[(size_t)t * cols + series] = qnan32();
        return;
    }
    const int warm = fv + length - 1;
    for (int t = 0; t < warm && t < rows; ++t) out_tm[(size_t)t * cols + series] = qnan32();
    if (warm >= rows) return;

    float* sorted = scratch_cols + (size_t)series * max_length;
    int wl = 0;
    const float p_frac = percentage * 0.01f;
    const int k_full = nearest_rank_index(p_frac * (float)length, length);

    auto load_tm = [&](int t) -> float { return prices_tm[(size_t)t * cols + series]; };
    auto store_tm = [&](int t, float v) { out_tm[(size_t)t * cols + series] = v; };


    const int w0 = warm + 1 - length;
    for (int idx = w0; idx <= warm; ++idx) {
        float v = load_tm(idx);
        if (!isnan(v)) wl = insert_sorted(sorted, wl, v);
    }

    int i = warm;
    while (true) {
        if (wl == 0) {
            store_tm(i, qnan32());
        } else {
            int k = (wl == length) ? k_full : nearest_rank_index(p_frac * (float)wl, wl);
            store_tm(i, sorted[k]);
        }
        if (i + 1 >= rows) break;
        float v_out = load_tm(i + 1 - length);
        if (!isnan(v_out)) wl = erase_sorted(sorted, wl, v_out);
        float v_in = load_tm(i + 1);
        if (!isnan(v_in)) wl = insert_sorted(sorted, wl, v_in);
        i += 1;
    }
}






extern "C" __global__
void percentile_nearest_rank_one_series_many_params_same_len_f32(
    const float* __restrict__ prices,
    int series_len,
    int length,
    const float* __restrict__ percentages,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {

    if (UNLIKELY(length <= 0 || first_valid < 0 || first_valid >= series_len)) {

        if (blockIdx.x == 0) {
            for (int c = threadIdx.x; c < n_combos; c += blockDim.x) {
                float* row = out + (size_t)c * series_len;
                for (int t = 0; t < series_len; ++t) row[t] = qnan32();
            }
        }
        return;
    }

    const int warm = first_valid + length - 1;
    if (blockIdx.x == 0) {

        for (int c = threadIdx.x; c < n_combos; c += blockDim.x) {
            float* row = out + (size_t)c * series_len;
            for (int t = 0; t < warm && t < series_len; ++t) row[t] = qnan32();
        }
    }
    __syncthreads();
    if (warm >= series_len) return;


    extern __shared__ float s_win[];
    const int S = next_pow2(length);


    for (int t = warm + blockIdx.x; t < series_len; t += gridDim.x) {

        const int start = t + 1 - length;


        int local_valid = 0;
        for (int i = threadIdx.x; i < length; i += blockDim.x) {
            float v = prices[start + i];
            s_win[i] = v;
            if (!is_nan(v)) local_valid += 1;
        }
        for (int i = threadIdx.x + length; i < S; i += blockDim.x) {
            s_win[i] = qnan32();
        }


        __shared__ int wl_shared;
        if (threadIdx.x == 0) wl_shared = 0;
        __syncthreads();

        int sum = local_valid;

        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if ((threadIdx.x & 31) == 0) atomicAdd(&wl_shared, sum);
        __syncthreads();

        const int wl = wl_shared;


        if (wl == 0) {
            for (int c = threadIdx.x; c < n_combos; c += blockDim.x) {
                out[(size_t)c * series_len + t] = qnan32();
            }
            __syncthreads();
            continue;
        }


        bitonic_sort_shared_nan_last(s_win, S);


        for (int c = threadIdx.x; c < n_combos; c += blockDim.x) {
            float p = percentages[c] * 0.01f;
            int   k = (wl == length)
                    ? nearest_rank_index_from_frac(p, length)
                    : nearest_rank_index_from_frac(p, wl);
            out[(size_t)c * series_len + t] = s_win[k];
        }
        __syncthreads();
    }
}
