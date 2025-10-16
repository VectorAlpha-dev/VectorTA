// CUDA kernels for AlphaTrend (AT)
//
// Batch kernel (one series × many params):
//  - Inputs: high, low, close, precomputed TR, momentum table (rows=unique periods),
//            mapping from combo->momentum row, per-combo coeffs and periods
//  - Each thread processes one parameter combo sequentially over time.
//
// Many-series × one-param kernel (time-major):
//  - Inputs: precomputed TR_tm and momentum_tm for a single (coeff, period)
//  - Each thread handles one series (column), scanning sequentially over rows (time).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline bool is_finite(float x) { return !isnan(x) && !isinf(x); }
}

extern "C" __global__ void alphatrend_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ tr,                 // [len]
    const float* __restrict__ momentum_table,     // [n_mrows * len]
    const int*   __restrict__ mrow_for_combo,     // [n_combos]
    const float* __restrict__ coeffs,             // [n_combos]
    const int*   __restrict__ periods,            // [n_combos]
    int len,
    int first_valid,
    int n_combos,
    int n_mrows,
    float* __restrict__ k1_out,                   // [n_combos * len]
    float* __restrict__ k2_out)                   // [n_combos * len]
{
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = tid0; row < n_combos; row += stride) {
        const int period = periods[row];
        const float coeff = coeffs[row];
        float* k1_row = k1_out + (size_t)row * len;
        float* k2_row = k2_out + (size_t)row * len;

        // Pre-fill row with NaN to cover warmup and invalid regions
        for (int i = 0; i < len; ++i) {
            k1_row[i] = CUDART_NAN_F;
            k2_row[i] = CUDART_NAN_F;
        }

        if (period <= 0 || period > len) {
            continue;
        }

        int warm = first_valid + period - 1;
        if (warm >= len) {
            continue; // nothing to compute
        }

        const int mrow = mrow_for_combo[row];
        if (mrow < 0 || mrow >= n_mrows) {
            continue;
        }
        const float* mom = momentum_table + (size_t)mrow * len;

        // Initialize ATR window sum over TR[first_valid..warm]
        double sum = 0.0;
        for (int j = first_valid; j <= warm; ++j) {
            sum += (double)tr[j];
        }

        float prev_alpha = CUDART_NAN_F;
        float prev1 = CUDART_NAN_F;
        float prev2 = CUDART_NAN_F;

        const float p_inv = 1.0f / (float)period;
        for (int i = warm; i < len; ++i) {
            const float a = (float)(sum * (double)p_inv);
            const float up = fmaf(-coeff, a, low[i]);   // low - coeff*ATR
            const float dn = fmaf( coeff, a, high[i]);  // high + coeff*ATR
            const bool m_ge_50 = is_finite(mom[i]) ? (mom[i] >= 50.0f) : false;

            float cur;
            if (i == warm) {
                cur = m_ge_50 ? up : dn;
            } else if (m_ge_50) {
                cur = (up < prev_alpha) ? prev_alpha : up;
            } else {
                cur = (dn > prev_alpha) ? prev_alpha : dn;
            }
            k1_row[i] = cur;
            if (i >= warm + 2) {
                k2_row[i] = prev2;
            }
            prev2 = prev1;
            prev1 = cur;
            prev_alpha = cur;

            const int nxt = i + 1;
            if (nxt < len) {
                sum += (double)tr[nxt] - (double)tr[nxt - period];
            }
        }
    }
}

// Many-series × one-param (time-major)
// Data layout: time-major matrices with shape [series_len][num_series]
extern "C" __global__ void alphatrend_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ tr_tm,
    const float* __restrict__ momentum_tm,
    const int*   __restrict__ first_valids, // [num_series]
    int num_series,
    int series_len,
    float coeff,
    int period,
    float* __restrict__ k1_tm_out, // [series_len * num_series]
    float* __restrict__ k2_tm_out) // [series_len * num_series]
{
    int s = blockIdx.x * blockDim.x + threadIdx.x; // series index (column)
    if (s >= num_series) return;
    const int fv = first_valids[s];
    if (period <= 0 || fv >= series_len) {
        // Prefill column with NaN
        for (int t = 0; t < series_len; ++t) {
            const int idx = t * num_series + s;
            k1_tm_out[idx] = CUDART_NAN_F;
            k2_tm_out[idx] = CUDART_NAN_F;
        }
        return;
    }

    const int warm = fv + period - 1;
    const float p_inv = 1.0f / (float)period;

    // Prefill column with NaN
    for (int t = 0; t < series_len; ++t) {
        const int idx = t * num_series + s;
        k1_tm_out[idx] = CUDART_NAN_F;
        k2_tm_out[idx] = CUDART_NAN_F;
    }
    if (warm >= series_len) return;

    // Initialize sum over TR column window
    double sum = 0.0;
    for (int t = fv; t <= warm; ++t) {
        sum += (double)tr_tm[t * num_series + s];
    }

    float prev_alpha = CUDART_NAN_F;
    float prev1 = CUDART_NAN_F;
    float prev2 = CUDART_NAN_F;

    for (int t = warm; t < series_len; ++t) {
        const int idx = t * num_series + s;
        const float a = (float)(sum * (double)p_inv);
        const float up = fmaf(-coeff, a, low_tm[idx]);
        const float dn = fmaf( coeff, a, high_tm[idx]);
        const float m = momentum_tm[idx];
        const bool m_ge_50 = is_finite(m) ? (m >= 50.0f) : false;

        float cur;
        if (t == warm) {
            cur = m_ge_50 ? up : dn;
        } else if (m_ge_50) {
            cur = (up < prev_alpha) ? prev_alpha : up;
        } else {
            cur = (dn > prev_alpha) ? prev_alpha : dn;
        }
        k1_tm_out[idx] = cur;
        if (t >= warm + 2) {
            k2_tm_out[idx] = prev2;
        }
        prev2 = prev1;
        prev1 = cur;
        prev_alpha = cur;

        const int nxt = t + 1;
        if (nxt < series_len) {
            sum += (double)tr_tm[nxt * num_series + s] -
                   (double)tr_tm[(nxt - period) * num_series + s];
        }
    }
}

