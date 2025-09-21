// CUDA kernels for the Slope Adaptive Moving Average (SAMA).
//
// Each block covers a single parameter combination because the SAMA recurrence
// is inherently sequential. Threads cooperate to initialise the output row, and
// lane 0 evaluates the adaptive smoothing in-order to mirror the scalar CPU
// semantics exactly. A companion kernel handles the time-major many-series
// entry point that the Rust and Python bindings expose.

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
