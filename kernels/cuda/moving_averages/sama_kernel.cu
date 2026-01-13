







#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void sama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ lengths,
                    const float* __restrict__ min_alphas,
                    const float* __restrict__ maj_alphas,
                    const int* __restrict__ first_valids,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int length = lengths[combo];
    const float min_alpha = min_alphas[combo];
    const float maj_alpha = maj_alphas[combo];
    const int first_valid = first_valids[combo];

    if (length < 0 || first_valid >= series_len || series_len <= 0) {
        return;
    }

    const int row_offset = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float prev = NAN;

    for (int t = first_valid; t < series_len; ++t) {
        const float price = prices[t];
        if (!isfinite(price)) {
            out[row_offset + t] = NAN;
            continue;
        }

        int start = t - length;
        if (start < 0) {
            start = 0;
        }
        float hh = -CUDART_INF_F;
        float ll = CUDART_INF_F;
        for (int j = start; j <= t; ++j) {
            const float v = prices[j];
            if (!isfinite(v)) {
                continue;
            }
            if (v > hh) {
                hh = v;
            }
            if (v < ll) {
                ll = v;
            }
        }

        float mult = 0.0f;
        if (hh != ll) {
            const float numer = fabsf(2.0f * price - ll - hh);
            const float denom = hh - ll;
            if (denom != 0.0f) {
                mult = numer / denom;
            }
        }
        float alpha = (mult * (min_alpha - maj_alpha) + maj_alpha);
        alpha = alpha * alpha;

        if (!isfinite(prev)) {
            prev = price;
        } else {
            prev = __fmaf_rn(price - prev, alpha, prev);
        }

        out[row_offset + t] = prev;
    }
}

extern "C" __global__
void sama_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    int length,
                                    float min_alpha,
                                    float maj_alpha,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (length < 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float prev = NAN;

    for (int t = first_valid; t < series_len; ++t) {
        const int offset = t * stride + series_idx;
        const float price = prices_tm[offset];
        if (!isfinite(price)) {
            out_tm[offset] = NAN;
            continue;
        }

        int start = t - length;
        if (start < 0) {
            start = 0;
        }

        float hh = -CUDART_INF_F;
        float ll = CUDART_INF_F;
        for (int j = start; j <= t; ++j) {
            const float v = prices_tm[j * stride + series_idx];
            if (!isfinite(v)) {
                continue;
            }
            if (v > hh) {
                hh = v;
            }
            if (v < ll) {
                ll = v;
            }
        }

        float mult = 0.0f;
        if (hh != ll) {
            const float numer = fabsf(2.0f * price - ll - hh);
            const float denom = hh - ll;
            if (denom != 0.0f) {
                mult = numer / denom;
            }
        }
        float alpha = (mult * (min_alpha - maj_alpha) + maj_alpha);
        alpha = alpha * alpha;

        if (!isfinite(prev)) {
            prev = price;
        } else {
            prev = __fmaf_rn(price - prev, alpha, prev);
        }

        out_tm[offset] = prev;
    }
}











#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#if !defined(CUDA_HAS_LDG_WRAPPER)
#define CUDA_HAS_LDG_WRAPPER

static __device__ __forceinline__ float ldgf(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}
#endif



static __device__ __forceinline__ int clamp_start(int t, int length) {
    int s = t - length;
    return s < 0 ? 0 : s;
}


static __device__ __forceinline__
void pop_outdated_front(int*& q, int& front, int& size, const int cap, int start) {
    while (size > 0) {
        int idx = q[front];
        if (idx >= start) break;
        front = (front + 1);
        if (front == cap) front = 0;
        --size;
    }
}


static __device__ __forceinline__
void push_max_idx(const float* base, int*& q, int& back, int& size, const int cap, int k) {
    float vk = ldgf(base + k);
    if (!isfinite(vk)) return;
    while (size > 0) {
        int back_pos = (back == 0 ? cap - 1 : back - 1);
        float vb = ldgf(base + q[back_pos]);

        if (vb > vk) break;
        back = back_pos;
        --size;
    }
    q[back] = k;
    back = (back + 1);
    if (back == cap) back = 0;
    ++size;
}


static __device__ __forceinline__
void push_min_idx(const float* base, int*& q, int& back, int& size, const int cap, int k) {
    float vk = ldgf(base + k);
    if (!isfinite(vk)) return;
    while (size > 0) {
        int back_pos = (back == 0 ? cap - 1 : back - 1);
        float vb = ldgf(base + q[back_pos]);

        if (vb < vk) break;
        back = back_pos;
        --size;
    }
    q[back] = k;
    back = (back + 1);
    if (back == cap) back = 0;
    ++size;
}



extern "C" __global__
void sama_batch_f32_opt(const float* __restrict__ prices,
                        const int*   __restrict__ lengths,
                        const float* __restrict__ min_alphas,
                        const float* __restrict__ maj_alphas,
                        const int*   __restrict__ first_valids,
                        int series_len,
                        int n_combos,
                        int max_window,
                        float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int length      = lengths[combo];
    const float min_alpha = min_alphas[combo];
    const float maj_alpha = maj_alphas[combo];
    const int first_valid = first_valids[combo];

    if (length < 0 || first_valid >= series_len || series_len <= 0) return;

    const int row_offset = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;


    const float dalpha = min_alpha - maj_alpha;

    float prev = NAN;


    const bool use_deque = (max_window >= length);
    if (!use_deque) {
        for (int t = first_valid; t < series_len; ++t) {
            const float price_t = ldgf(prices + t);
            if (!isfinite(price_t)) {
                out[row_offset + t] = NAN;
                continue;
            }
            const int start = clamp_start(t, length);

            float hh = -CUDART_INF_F, ll = CUDART_INF_F;
            bool any = false;
            #pragma unroll 1
            for (int j = start; j <= t; ++j) {
                const float v = ldgf(prices + j);
                if (!isfinite(v)) continue;
                any = true;
                if (v > hh) hh = v;
                if (v < ll) ll = v;
            }
            float mult = 0.0f;
            if (any) {
                const float denom = hh - ll;
                if (denom != 0.0f) {
                    const float numer = fabsf(2.0f * price_t - ll - hh);
                    mult = numer / denom;
                }
            }
            float alpha = __fmaf_rn(mult, dalpha, maj_alpha);
            alpha = alpha * alpha;

            prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
            out[row_offset + t] = prev;
        }
        return;
    }


    extern __shared__ int shmem[];
    const int cap = max_window + 1;
    int* dq_max = shmem;
    int* dq_min = shmem + cap;

    int fmax = 0, bmax = 0, szmax = 0;
    int fmin = 0, bmin = 0, szmin = 0;

    for (int t = first_valid; t < series_len; ++t) {
        const int start = clamp_start(t, length);
        const float price_t = ldgf(prices + t);


        pop_outdated_front(dq_max, fmax, szmax, cap, start);
        pop_outdated_front(dq_min, fmin, szmin, cap, start);

        if (!isfinite(price_t)) {
            out[row_offset + t] = NAN;
            continue;
        }


        while (szmax > 0) {
            int back_pos = (bmax == 0 ? cap - 1 : bmax - 1);
            float vb = ldgf(prices + dq_max[back_pos]);
            if (vb > price_t) break;
            bmax = back_pos; --szmax;
        }
        dq_max[bmax] = t; bmax = (bmax + 1 == cap ? 0 : bmax + 1); ++szmax;

        while (szmin > 0) {
            int back_pos = (bmin == 0 ? cap - 1 : bmin - 1);
            float vb = ldgf(prices + dq_min[back_pos]);
            if (vb < price_t) break;
            bmin = back_pos; --szmin;
        }
        dq_min[bmin] = t; bmin = (bmin + 1 == cap ? 0 : bmin + 1); ++szmin;


        const float hh = ldgf(prices + dq_max[fmax]);
        const float ll = ldgf(prices + dq_min[fmin]);
        const float denom = hh - ll;
        float mult = 0.0f;
        if (denom != 0.0f) {
            const float numer = fabsf(2.0f * price_t - ll - hh);
            mult = numer / denom;
        }

        float alpha = __fmaf_rn(mult, (min_alpha - maj_alpha), maj_alpha);
        alpha = alpha * alpha;

        prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
        out[row_offset + t] = prev;
    }
}



extern "C" __global__
void sama_many_series_one_param_f32_opt(const float* __restrict__ prices_tm,
                                        const int*   __restrict__ first_valids,
                                        int length,
                                        float min_alpha,
                                        float maj_alpha,
                                        int num_series,
                                        int series_len,
                                        int max_window,
                                        float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) return;
    if (length < 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];


    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const float dalpha = min_alpha - maj_alpha;

    float prev = NAN;


    const bool use_deque = (max_window >= length);
    if (!use_deque) {
        for (int t = first_valid; t < series_len; ++t) {
            const int off = t * stride + series_idx;
            const float price_t = ldgf(prices_tm + off);
            if (!isfinite(price_t)) {
                out_tm[off] = NAN;
                continue;
            }
            const int start = clamp_start(t, length);
            float hh = -CUDART_INF_F, ll = CUDART_INF_F;
            bool any = false;
            #pragma unroll 1
            for (int j = start; j <= t; ++j) {
                const float v = ldgf(prices_tm + j * stride + series_idx);
                if (!isfinite(v)) continue;
                any = true;
                if (v > hh) hh = v;
                if (v < ll) ll = v;
            }
            float mult = 0.0f;
            if (any) {
                const float denom = hh - ll;
                if (denom != 0.0f) {
                    const float numer = fabsf(2.0f * price_t - ll - hh);
                    mult = numer / denom;
                }
            }
            float alpha = __fmaf_rn(mult, dalpha, maj_alpha);
            alpha = alpha * alpha;

            prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
            out_tm[off] = prev;
        }
        return;
    }


    extern __shared__ int shmem[];
    const int cap = max_window + 1;
    int* dq_max = shmem;
    int* dq_min = shmem + cap;

    int fmax = 0, bmax = 0, szmax = 0;
    int fmin = 0, bmin = 0, szmin = 0;

    auto load_tm = [&](int t)->float {
        return ldgf(prices_tm + t * stride + series_idx);
    };

    for (int t = first_valid; t < series_len; ++t) {
        const int start = clamp_start(t, length);
        const int off   = t * stride + series_idx;
        const float price_t = load_tm(t);


        pop_outdated_front(dq_max, fmax, szmax, cap, start);
        pop_outdated_front(dq_min, fmin, szmin, cap, start);

        if (!isfinite(price_t)) {
            out_tm[off] = NAN;
            continue;
        }


        while (szmax > 0) {
            int back_pos = (bmax == 0 ? cap - 1 : bmax - 1);
            float vb = load_tm(dq_max[back_pos]);
            if (vb > price_t) break;
            bmax = back_pos; --szmax;
        }
        dq_max[bmax] = t; bmax = (bmax + 1 == cap ? 0 : bmax + 1); ++szmax;

        while (szmin > 0) {
            int back_pos = (bmin == 0 ? cap - 1 : bmin - 1);
            float vb = load_tm(dq_min[back_pos]);
            if (vb < price_t) break;
            bmin = back_pos; --szmin;
        }
        dq_min[bmin] = t; bmin = (bmin + 1 == cap ? 0 : bmin + 1); ++szmin;

        const float hh = load_tm(dq_max[fmax]);
        const float ll = load_tm(dq_min[fmin]);
        const float denom = hh - ll;
        float mult = 0.0f;
        if (denom != 0.0f) {
            const float numer = fabsf(2.0f * price_t - ll - hh);
            mult = numer / denom;
        }

        float alpha = __fmaf_rn(mult, (min_alpha - maj_alpha), maj_alpha);
        alpha = alpha * alpha;

        prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
        out_tm[off] = prev;
    }
}
