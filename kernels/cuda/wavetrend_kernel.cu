// CUDA kernel for the WaveTrend indicator batch computation.
//
// Each thread processes one parameter combination (row) and streams through
// the time series sequentially to match the scalar CPU reference exactly.
// Outputs are laid out row-major with shape (rows, len).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline bool is_finite(float x) {
    return !isnan(x) && !isinf(x);
}
}

extern "C" __global__ void wavetrend_batch_f32(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int n_combos,
    const int* __restrict__ channel_lengths,
    const int* __restrict__ average_lengths,
    const int* __restrict__ ma_lengths,
    const float* __restrict__ factors,
    float* __restrict__ wt1_out,
    float* __restrict__ wt2_out,
    float* __restrict__ wt_diff_out
) {
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = row0; row < n_combos; row += stride) {
        const int ch = channel_lengths[row];
        const int avg = average_lengths[row];
        const int ma = ma_lengths[row];
        const float factor = factors[row];

        float* wt1_row = wt1_out + static_cast<size_t>(row) * len;
        float* wt2_row = wt2_out + static_cast<size_t>(row) * len;
        float* diff_row = wt_diff_out + static_cast<size_t>(row) * len;

        // Pre-fill outputs with NaN to cover warmup and invalid regions
        for (int i = 0; i < len; ++i) {
            wt1_row[i] = CUDART_NAN_F;
            wt2_row[i] = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }

        const float alpha_ch = 2.0f / (static_cast<float>(ch) + 1.0f);
        const float beta_ch = 1.0f - alpha_ch;
        const float alpha_avg = 2.0f / (static_cast<float>(avg) + 1.0f);
        const float beta_avg = 1.0f - alpha_avg;
        const float inv_ma = ma > 0 ? 1.0f / static_cast<float>(ma) : 0.0f;

        int warmup = first_valid + ch - 1 + avg - 1 + ma - 1;
        if (warmup < 0) {
            warmup = 0;
        }
        if (warmup > len) {
            warmup = len;
        }

        float esa_state = CUDART_NAN_F;
        float de_state = CUDART_NAN_F;
        float wt1_state = CUDART_NAN_F;

        float sum_wt1 = 0.0f;
        int window_count = 0;

        for (int i = first_valid; i < len; ++i) {
            const float price = prices[i];

            // Stage 1: ESA EMA(channel_lengths)
            float esa_val = CUDART_NAN_F;
            if (is_finite(price)) {
                if (!is_finite(esa_state)) {
                    esa_state = price;
                } else {
                    esa_state = fmaf(beta_ch, esa_state, alpha_ch * price);
                }
                esa_val = esa_state;
            }

            // Stage 2: DE EMA(channel_lengths) on |price - ESA|
            float absdiff = CUDART_NAN_F;
            if (is_finite(price) && is_finite(esa_val)) {
                absdiff = fabsf(price - esa_val);
            }

            float de_val = CUDART_NAN_F;
            if (is_finite(absdiff)) {
                if (!is_finite(de_state)) {
                    de_state = absdiff;
                } else {
                    de_state = fmaf(beta_ch, de_state, alpha_ch * absdiff);
                }
                de_val = de_state;
            }

            // Stage 3: CI and WT1 = EMA(average_length)
            float ci_val = CUDART_NAN_F;
            if (is_finite(price) && is_finite(esa_val) && is_finite(de_val)) {
                const float denom = factor * de_val;
                if (denom != 0.0f && isfinite(denom)) {
                    ci_val = (price - esa_val) / denom;
                }
            }

            float wt1_val = CUDART_NAN_F;
            if (is_finite(ci_val)) {
                if (!is_finite(wt1_state)) {
                    wt1_state = ci_val;
                } else {
                    wt1_state = fmaf(beta_avg, wt1_state, alpha_avg * ci_val);
                }
                wt1_val = wt1_state;
            }

            wt1_row[i] = wt1_val;

            if (is_finite(wt1_val)) {
                sum_wt1 += wt1_val;
                window_count += 1;
            }

            if (ma > 0 && i >= ma) {
                const float old = wt1_row[i - ma];
                if (is_finite(old)) {
                    sum_wt1 -= old;
                    window_count -= 1;
                }
            }

            float wt2_val = CUDART_NAN_F;
            if (ma > 0 && window_count >= ma) {
                wt2_val = sum_wt1 * inv_ma;
            }
            wt2_row[i] = wt2_val;

            if (i >= warmup && is_finite(wt2_val) && is_finite(wt1_val)) {
                diff_row[i] = wt2_val - wt1_val;
            } else {
                diff_row[i] = CUDART_NAN_F;
            }
        }

        for (int i = 0; i < warmup; ++i) {
            wt1_row[i] = CUDART_NAN_F;
            wt2_row[i] = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }
    }
}
