// CUDA kernels for the WaveTrend indicator (optimized FP32 hot path).
//
// Changes vs previous version:
// - Eliminate FP64 in hot loops; use FMA-based EMA updates in float.
// - Kahan-style compensated rolling sum for WT2 SMA to retain precision.
// - Minimal prefill: only [0, first_valid) prefix; warmup cleared at end.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// ---------- helpers ----------
namespace {

__device__ __forceinline__ bool is_finite_f(float x) {
    return !(isnan(x) || isinf(x));
}

// FMA-based EMA update: state <- state + alpha * (x - state)
__device__ __forceinline__ float ema_update(float state, float x, float alpha) {
    return fmaf(alpha, x - state, state);
}

// Simple Kahan-style compensated accumulator for float
struct KahanSumF {
    float s;  // running sum
    float c;  // compensation
    __device__ KahanSumF() : s(0.0f), c(0.0f) {}
    __device__ void add(float x) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ void sub(float x) { add(-x); }
};

} // namespace

extern "C" __global__ void wavetrend_batch_f32(
    const float* __restrict__ prices,   // len
    int len,
    int first_valid,
    int n_combos,
    const int* __restrict__ channel_lengths,
    const int* __restrict__ average_lengths,
    const int* __restrict__ ma_lengths,
    const float* __restrict__ factors,
    float* __restrict__ wt1_out,        // (rows, len) row-major
    float* __restrict__ wt2_out,        // (rows, len) row-major
    float* __restrict__ wt_diff_out     // (rows, len) row-major
){
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride  = blockDim.x * gridDim.x;

    for (int row = tid; row < n_combos; row += stride) {
        const int ch  = channel_lengths[row];
        const int avg = average_lengths[row];
        const int ma  = ma_lengths[row];
        const float factor = factors[row];

        float* __restrict__ wt1_row  = wt1_out     + (size_t)row * (size_t)len;
        float* __restrict__ wt2_row  = wt2_out     + (size_t)row * (size_t)len;
        float* __restrict__ diff_row = wt_diff_out + (size_t)row * (size_t)len;

        // Early reject: invalid params -> write NaNs and continue
        if (len <= 0 || ch <= 0 || avg <= 0 || ma <= 0) {
            for (int i = 0; i < len; ++i) {
                wt1_row[i]  = CUDART_NAN_F;
                wt2_row[i]  = CUDART_NAN_F;
                diff_row[i] = CUDART_NAN_F;
            }
            continue;
        }

        const float alpha_ch  = 2.0f / (float(ch) + 1.0f);
        const float alpha_avg = 2.0f / (float(avg) + 1.0f);
        const float inv_ma    = 1.0f / (float)ma;

        // Warmup = first_valid + (ch-1) + (avg-1) + (ma-1)
        int warmup = first_valid + (ch - 1) + (avg - 1) + (ma - 1);
        if (warmup < 0)       warmup = 0;
        if (warmup > len)     warmup = len;

        // Prefill only the prefix before first_valid with NaNs
        int prefill = first_valid;
        if (prefill < 0) prefill = 0;
        if (prefill > len) prefill = len;
        for (int i = 0; i < prefill; ++i) {
            wt1_row[i]  = CUDART_NAN_F;
            wt2_row[i]  = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }

        // EMA states (float, initialized lazily when first finite arrives)
        bool esa_init = false, de_init = false, wt1_init = false;
        float esa = 0.0f, de = 0.0f, wt1_state = 0.0f;

        // Rolling SMA for WT2 (compensated)
        KahanSumF acc;
        int window_count = 0;

        int start = first_valid > 0 ? first_valid : 0;
        for (int i = start; i < len; ++i) {
            const float price = prices[i];
            const bool price_ok = is_finite_f(price);

            // ESA = EMA(price, ch)
            if (!esa_init) {
                if (price_ok) {
                    esa = price;
                    esa_init = true;
                }
            } else if (price_ok) {
                esa = ema_update(esa, price, alpha_ch);
            }

            // DE = EMA(|price - ESA|, ch)
            if (esa_init && price_ok) {
                const float absdiff = fabsf(price - esa);
                if (!de_init) {
                    de = absdiff;
                    de_init = true;
                } else {
                    de = ema_update(de, absdiff, alpha_ch);
                }
            }

            // CI and WT1 = EMA(CI, avg)
            float wt1_val = CUDART_NAN_F;
            if (esa_init && de_init && price_ok) {
                const float denom = factor * de;
                if (denom != 0.0f && is_finite_f(denom)) {
                    const float ci = (price - esa) / denom;
                    if (!wt1_init) {
                        if (is_finite_f(ci)) {
                            wt1_state = ci;
                            wt1_init = true;
                        }
                    } else if (is_finite_f(ci)) {
                        wt1_state = ema_update(wt1_state, ci, alpha_avg);
                    }
                }
            }
            if (wt1_init) wt1_val = wt1_state;

            // Store WT1 (always write, warmup cleared later)
            wt1_row[i] = wt1_val;

            // Maintain rolling WT2 window (compensated sum over valid WT1)
            if (is_finite_f(wt1_val)) { acc.add(wt1_val); ++window_count; }

            if (i >= ma) {
                const float old = wt1_row[i - ma];
                if (is_finite_f(old)) { acc.sub(old); --window_count; }
            }

            // WT2 (mean of last 'ma' valid WT1 samples in window)
            float wt2_val = CUDART_NAN_F;
            if (window_count >= ma) {
                wt2_val = acc.s * inv_ma;
            }
            wt2_row[i] = wt2_val;

            // DIFF (after warmup)
            if (i >= warmup && is_finite_f(wt2_val) && is_finite_f(wt1_val)) {
                diff_row[i] = wt2_val - wt1_val;
            } else {
                diff_row[i] = CUDART_NAN_F;
            }
        }

        // Clear warmup prefix to NaNs to mirror scalar semantics
        for (int i = 0; i < warmup; ++i) {
            wt1_row[i]  = CUDART_NAN_F;
            wt2_row[i]  = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }
    }
}

// Many-series × one-param (time-major layout), grid-stride over series
extern "C" __global__ void wavetrend_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm, // (rows, cols), time-major
    int cols,
    int rows,
    int channel_length,
    int average_length,
    int ma_length,
    float factor,
    const int* __restrict__ first_valids, // [cols]
    float* __restrict__ wt1_tm,           // (rows, cols), time-major
    float* __restrict__ wt2_tm,           // (rows, cols), time-major
    float* __restrict__ wt_diff_tm        // (rows, cols), time-major
){
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (rows <= 0 || cols <= 0 || channel_length <= 0 || average_length <= 0 || ma_length <= 0) return;

    const float alpha_ch  = 2.0f / (float(channel_length) + 1.0f);
    const float alpha_avg = 2.0f / (float(average_length) + 1.0f);
    const float inv_ma    = 1.0f / (float)ma_length;

    for (int series = tid; series < cols; series += stride) {
        float* __restrict__ wt1_col  = wt1_tm     + series;
        float* __restrict__ wt2_col  = wt2_tm     + series;
        float* __restrict__ diff_col = wt_diff_tm + series;

        const int first_valid = first_valids[series];
        int warmup = first_valid + (channel_length - 1) + (average_length - 1) + (ma_length - 1);
        if (warmup < 0) warmup = 0;
        if (warmup > rows) warmup = rows;

        // Prefill only prefix before first_valid
        int pre = first_valid;
        if (pre < 0) pre = 0;
        if (pre > rows) pre = rows;
        for (int t = 0; t < pre; ++t) {
            const int idx = t * cols;
            wt1_col[idx]  = CUDART_NAN_F;
            wt2_col[idx]  = CUDART_NAN_F;
            diff_col[idx] = CUDART_NAN_F;
        }

        // EMA states in FP64 to match CPU/previous GPU semantics more closely
        bool esa_init = false, de_init = false, wt1_init = false;
        double esa = 0.0, de = 0.0, wt1_state = 0.0;

        // Rolling WT2 window as plain FP64 sum (close to CPU)
        double sum_wt1 = 0.0;
        int window_count = 0;

        int start = first_valid > 0 ? first_valid : 0;
        for (int t = start; t < rows; ++t) {
            const int idx = t * cols;
            const double price = static_cast<double>(prices_tm[idx + series]);
            const bool price_ok = isfinite(price);

            // ESA
            if (!esa_init) {
                if (price_ok) { esa = price; esa_init = true; }
            } else if (price_ok) {
                const double alpha_ch_d = static_cast<double>(alpha_ch);
                const double beta_ch_d  = 1.0 - alpha_ch_d;
                esa = fma(alpha_ch_d, price, beta_ch_d * esa);
            }

            // DE
            if (esa_init && price_ok) {
                const double absdiff = fabs(price - esa);
                if (!de_init) { de = absdiff; de_init = isfinite(de); }
                else if (isfinite(absdiff)) {
                    const double alpha_ch_d = static_cast<double>(alpha_ch);
                    const double beta_ch_d  = 1.0 - alpha_ch_d;
                    de = fma(alpha_ch_d, absdiff, beta_ch_d * de);
                }
            }

            // CI & WT1
            float wt1_val = CUDART_NAN_F;
            if (esa_init && de_init && price_ok) {
                const double denom = static_cast<double>(factor) * de;
                if (denom != 0.0 && isfinite(denom)) {
                    const double ci = (price - esa) / denom;
                    if (!wt1_init) {
                        if (isfinite(ci)) { wt1_state = ci; wt1_init = true; }
                    } else if (isfinite(ci)) {
                        const double alpha_avg_d = static_cast<double>(alpha_avg);
                        const double beta_avg_d  = 1.0 - alpha_avg_d;
                        wt1_state = fma(alpha_avg_d, ci, beta_avg_d * wt1_state);
                    }
                }
            }
            if (wt1_init) wt1_val = static_cast<float>(wt1_state);
            wt1_col[idx] = wt1_val;

            // Rolling WT2 window
            if (isfinite(static_cast<double>(wt1_val))) { sum_wt1 += wt1_state; ++window_count; }
            if (t >= ma_length) {
                const float old = wt1_col[(t - ma_length) * cols];
                if (isfinite(static_cast<double>(old))) { sum_wt1 -= static_cast<double>(old); --window_count; }
            }

            float wt2_val = CUDART_NAN_F;
            if (window_count >= ma_length) wt2_val = static_cast<float>(sum_wt1 * inv_ma);
            wt2_col[idx] = wt2_val;

            // DIFF
            if (t >= warmup && isfinite(static_cast<double>(wt1_val)) && isfinite(static_cast<double>(wt2_val))) {
                diff_col[idx] = wt2_val - wt1_val;
            } else {
                diff_col[idx] = CUDART_NAN_F;
            }
        }

        // Clear warmup prefix
        for (int t = 0; t < rows && t < warmup; ++t) {
            const int idx = t * cols;
            wt1_col[idx]  = CUDART_NAN_F;
            wt2_col[idx]  = CUDART_NAN_F;
            diff_col[idx] = CUDART_NAN_F;
        }
    }
}
