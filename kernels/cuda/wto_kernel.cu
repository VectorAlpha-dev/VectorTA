// CUDA kernels for the WaveTrend Oscillator (WTO).
//
// This file provides FP32-optimized kernels with FMA usage and compensated
// accumulation for the SMA(4) tail (wt2). For the one-series-many-params path,
// we also broadcast the price per time step across the warp to reduce redundant
// global loads.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// --- Accuracy + memory helpers ------------------------------------------------

#ifndef WTO_UTILS_H
#define WTO_UTILS_H

// Canonical quiet NaN bit-pattern; cheaper than nanf("") inside tight loops.
static __device__ __forceinline__ float wto_nan() {
    return __int_as_float(0x7fc00000u);
}

// Compensated FP32 accumulator (Kahan/Neumaier style) suitable for streaming sums.
// Keeps hi+lo representing an extended-precision FP32 sum at FP32 throughput.
struct fpair {
    float hi, lo;
    __device__ __forceinline__ void init() { hi = 0.f; lo = 0.f; }
    __device__ __forceinline__ void add(float x) {
        float s  = hi + x;
        float bb = s - hi;
        float e  = (hi - (s - bb)) + (x - bb);  // round-off
        float t  = lo + e;
        float st = s + t;
        lo = t - (st - s);
        hi = st;
    }
    __device__ __forceinline__ float value() const { return hi + lo; }
};

// Broadcast prices[t] across the active lanes of a warp.
// This avoids 32 identical loads per warp (one series, many param combos).
static __device__ __forceinline__ float bcast_price(const float* __restrict__ prices, int t) {
    unsigned mask   = __activemask();
    int      leader = __ffs(mask) - 1;
    float    v      = 0.f;
    int      lane   = threadIdx.x & 31;
    if (lane == leader) { v = __ldg(prices + t); } // RO cache hint; safe for immutable input
    return __shfl_sync(mask, v, leader);
}

#endif // WTO_UTILS_H

// Utility to prefill output rows with NaNs (thread-local row clear; preserves
// existing host API that does not launch a separate fill kernel).
__device__ inline void fill_nan(float* ptr, int len) {
    const float nanv = wto_nan();
    for (int i = 0; i < len; ++i) {
        ptr[i] = nanv;
    }
}

// Optional coalesced NaN prefill for all three outputs in a single pass.
// Not wired from host in current integration but provided for future use.
extern "C" __global__
void wto_fill_nan3_f32(float* __restrict__ out_wt1,
                       float* __restrict__ out_wt2,
                       float* __restrict__ out_hist,
                       size_t total_elems) {
    const float qnan = wto_nan();
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elems;
         idx += (size_t)blockDim.x * gridDim.x) {
        out_wt1[idx] = qnan;
        out_wt2[idx] = qnan;
        out_hist[idx] = qnan;
    }
}

extern "C" __global__
void wto_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ channel_lengths,
                   const int*   __restrict__ average_lengths,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ wt1_out,
                   float* __restrict__ wt2_out,
                   float* __restrict__ hist_out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int chan = channel_lengths[combo];
    const int avg = average_lengths[combo];

    float* wt1_row  = wt1_out  + (size_t)combo * series_len;
    float* wt2_row  = wt2_out  + (size_t)combo * series_len;
    float* hist_row = hist_out + (size_t)combo * series_len;

    // Backward-compat prefill: keep per-row NaN clear until host adopts wto_fill_nan3_f32
    fill_nan(wt1_row, series_len);
    fill_nan(wt2_row, series_len);
    fill_nan(hist_row, series_len);

    if (chan <= 0 || avg <= 0) {
        return;
    }
    const int start_ci = first_valid + chan - 1;

    // EMA coefficients in FP64 for closer parity with CPU
    const double alpha_ch = 2.0 / (double(chan) + 1.0);
    const double beta_ch  = 1.0 - alpha_ch;
    const double alpha_av = 2.0 / (double(avg) + 1.0);
    const double beta_av  = 1.0 - alpha_av;

    // Recurrence state (FP64 to match CPU scalar tolerance)
    bool   esa_init = false, d_init = false, wt1_init = false;
    double esa = 0.0, d = 0.0, wt1 = 0.0;

    // WT2 via prefix-sum pair: (ps - ps4)/4 in FP64 (matches CPU)
    double window[4] = {0.0, 0.0, 0.0, 0.0};
    int    window_count = 0, window_idx = 0;
    double ps = 0.0, ps4 = 0.0;

    for (int t = 0; t < series_len; ++t) {
        // One load per warp; broadcast to all active lanes.
        const float  price_f32  = bcast_price(prices, t);
        const bool   priceFinite = isfinite(price_f32);
        const double price      = static_cast<double>(price_f32);

        if (t < first_valid) { continue; }

        if (!esa_init) {
            if (!priceFinite) { continue; }
            esa = price; esa_init = true;
        } else if (priceFinite) {
            esa = fma(beta_ch, esa, alpha_ch * price);
        }

        const double diff     = price - esa;
        const double abs_diff = fabs(diff);

        if (t == start_ci) {
            // Seed ESA already updated; seed d, compute ci (0.015*d) and set wt1=ci unconditionally
            const double absdiff0 = priceFinite ? fabs(price - esa) : __longlong_as_double(0x7ff8000000000000ULL);
            d = absdiff0; d_init = true;
            const double denom0 = 0.015 * d;
            double ci0 = 0.0;
            if (denom0 != 0.0 && isfinite(denom0)) {
                if (priceFinite) { ci0 = (price - esa) / denom0; }
                else { ci0 = __longlong_as_double(0x7ff8000000000000ULL); }
            } else {
                ci0 = 0.0;
            }
            wt1 = ci0; wt1_init = true;

            wt1_row[t] = static_cast<float>(wt1);
            if (window_count == 4) { ps4 += window[window_idx]; } else { ++window_count; }
            ps += wt1; window[window_idx] = wt1; window_idx = (window_idx + 1) & 3;
            if (window_count == 4) {
                const double wt2d = 0.25 * (ps - ps4);
                wt2_row[t] = static_cast<float>(wt2d);
                hist_row[t] = static_cast<float>(wt1 - wt2d);
            }
        } else if (t > start_ci) {
            if (isfinite(abs_diff)) { d = fma(beta_ch, d, alpha_ch * abs_diff); }
            double ci;
            const double denom = 0.015 * d;
            if (priceFinite) {
                if (denom != 0.0 && isfinite(denom)) { ci = diff / denom; }
                else { ci = 0.0; }
            } else {
                ci = __longlong_as_double(0x7ff8000000000000ULL);
            }
            if (isfinite(ci)) { wt1 = fma(beta_av, wt1, alpha_av * ci); }
            wt1_row[t] = static_cast<float>(wt1);
            if (window_count == 4) { ps4 += window[window_idx]; } else { ++window_count; }
            ps += wt1; window[window_idx] = wt1; window_idx = (window_idx + 1) & 3;
            if (window_count == 4) {
                const double wt2d  = 0.25 * (ps - ps4);
                wt2_row[t]  = static_cast<float>(wt2d);
                hist_row[t] = static_cast<float>(wt1 - wt2d);
            }
        }
    }
    // Ensure we leave explicit NaNs where histogram never became valid
    const float qnan = wto_nan();
    for (int t = 0; t < series_len; ++t) {
        if (!isfinite(hist_row[t]) && !isfinite(wt1_row[t])) {
            hist_row[t] = qnan;
        }
    }
}

extern "C" __global__
void wto_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int channel_length,
    int average_length,
    const int* __restrict__ first_valids,
    float* __restrict__ wt1_tm,
    float* __restrict__ wt2_tm,
    float* __restrict__ hist_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) {
        return;
    }

    float* wt1_col  = wt1_tm + series;
    float* wt2_col  = wt2_tm + series;
    float* hist_col = hist_tm + series;

    // Backward-compat prefill: keep per-column NaN clear
    const float qnan = wto_nan();
    for (int t = 0; t < rows; ++t) {
        wt1_col[t * cols]  = qnan;
        wt2_col[t * cols]  = qnan;
        hist_col[t * cols] = qnan;
    }

    if (channel_length <= 0 || average_length <= 0) {
        return;
    }

    const int first_valid = first_valids[series];
    const int start_ci = first_valid + channel_length - 1;

    const double alpha_ch = 2.0 / (double(channel_length) + 1.0);
    const double beta_ch  = 1.0 - alpha_ch;
    const double alpha_av = 2.0 / (double(average_length) + 1.0);
    const double beta_av  = 1.0 - alpha_av;

    bool   esa_init = false, d_init = false, wt1_init = false;
    double esa = 0.0, d = 0.0, wt1 = 0.0;

    // WT2 via 4-sample running sum updated by subtracting the outgoing sample (read from output)
    int    wt2_count = 0;
    double sum_wt1 = 0.0;

    for (int t = 0; t < rows; ++t) {
        const float  price_f32 = __ldg(prices_tm + (size_t)t * cols + series);
        const bool   priceFinite = isfinite(price_f32);
        const double price = static_cast<double>(price_f32);

        if (t < first_valid) { continue; }

        if (!esa_init) { if (!priceFinite) continue; esa = price; esa_init = true; }
        else if (priceFinite) { esa = fma(beta_ch, esa, alpha_ch * price); }

        const double diff     = price - esa;
        const double abs_diff = fabs(diff);

        if (t == start_ci) {
            const double absdiff0 = priceFinite ? fabs(price - esa) : __longlong_as_double(0x7ff8000000000000ULL);
            d = absdiff0; d_init = true;
            const double denom0 = 0.015 * d;
            double ci0 = 0.0;
            if (denom0 != 0.0 && isfinite(denom0)) {
                if (priceFinite) { ci0 = (price - esa) / denom0; }
                else { ci0 = __longlong_as_double(0x7ff8000000000000ULL); }
            } else { ci0 = 0.0; }
            wt1 = ci0; wt1_init = true;
            wt1_col[t * cols] = static_cast<float>(wt1);

            if (isfinite(wt1)) { sum_wt1 += wt1; }
            if (wt2_count < 4) { ++wt2_count; }
            if (wt2_count == 4) {
                const double wt2d  = 0.25 * sum_wt1;
                wt2_col[t * cols]  = static_cast<float>(wt2d);
                hist_col[t * cols] = static_cast<float>(wt1 - wt2d);
            }
        } else if (t > start_ci) {
            if (isfinite(abs_diff)) { d = fma(beta_ch, d, alpha_ch * abs_diff); }
            const double denom = 0.015 * d;
            double ci;
            if (priceFinite) {
                if (denom != 0.0 && isfinite(denom)) { ci = diff / denom; }
                else { ci = 0.0; }
            } else { ci = __longlong_as_double(0x7ff8000000000000ULL); }
            if (isfinite(ci)) { wt1 = fma(beta_av, wt1, alpha_av * ci); }
            wt1_col[t * cols] = static_cast<float>(wt1);

            if (isfinite(wt1)) { sum_wt1 += wt1; }
            if (wt2_count < 4) { ++wt2_count; }
            else {
                const float old_f = wt1_col[(t - 4) * cols];
                if (isfinite(static_cast<double>(old_f))) { sum_wt1 -= static_cast<double>(old_f); }
            }
            if (wt2_count == 4) {
                const double wt2d  = 0.25 * sum_wt1;
                wt2_col[t * cols]  = static_cast<float>(wt2d);
                hist_col[t * cols] = static_cast<float>(wt1 - wt2d);
            }
        }
    }
}
