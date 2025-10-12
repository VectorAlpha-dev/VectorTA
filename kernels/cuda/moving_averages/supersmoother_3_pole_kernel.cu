// CUDA kernels for the 3-pole SuperSmoother filter.
// Optimized for Ada+ (e.g., RTX 4090), CUDA 13.
// - One thread processes one series/period (no thread-0-only blocks)
// - Coalesced reads in the time-major kernel
// - FMA in the recurrence (FP64) for better accuracy
// - Pointer-increment inner loops for strided variant

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {

struct SupersmootherCoefs {
    double coef_source;
    double coef_prev1;
    double coef_prev2;
    double coef_prev3;
};

__device__ __forceinline__ SupersmootherCoefs make_coefs(int period) {
    const double inv_period = 1.0 / static_cast<double>(period);
    const double a = exp(-CUDART_PI * inv_period);
    const double b = 2.0 * a * cos(1.738 * CUDART_PI * inv_period);
    const double c = a * a;
    SupersmootherCoefs coefs;
    // Same algebra as the original implementation
    coefs.coef_source = 1.0 - c * c - b + b * c;
    coefs.coef_prev1 = b + c;
    coefs.coef_prev2 = -c - b * c;
    coefs.coef_prev3 = c * c;
    return coefs;
}

// --- Core row kernels --------------------------------------------------------

__device__ __forceinline__ void supersmoother_3_pole_row_with_coefs(
    const float* __restrict__ prices,
    int series_len,
    int first_valid,
    const SupersmootherCoefs& coefs,
    float* __restrict__ out)
{
    if (series_len <= 0) return;

    const int start = (first_valid < 0) ? 0 : first_valid;

    // Pre-fill NaN up to first valid (bounded by series_len)
    for (int t = 0; t < start && t < series_len; ++t) {
        out[t] = CUDART_NAN_F;
    }
    if (start >= series_len) return;

    // Seed first 3 samples with the input stream as in original code
    int t = start;

    double y0 = static_cast<double>(prices[t]);
    out[t] = static_cast<float>(y0);
    if (++t >= series_len) return;

    double y1 = static_cast<double>(prices[t]);
    out[t] = static_cast<float>(y1);
    if (++t >= series_len) return;

    double y2 = static_cast<double>(prices[t]);
    out[t] = static_cast<float>(y2);
    ++t;

    // Main recurrence with fused multiply-add in FP64
    #pragma unroll 4
    for (; t < series_len; ++t) {
        const double x = static_cast<double>(prices[t]);
        // y_next = cS*x + c1*y2 + c2*y1 + c3*y0  (FMA chained)
        const double y_next =
            fma(coefs.coef_prev3, y0,
            fma(coefs.coef_prev2, y1,
            fma(coefs.coef_prev1, y2, coefs.coef_source * x)));

        out[t] = static_cast<float>(y_next);
        y0 = y1; y1 = y2; y2 = y_next;
    }
}

__device__ __forceinline__ void supersmoother_3_pole_row(
    const float* __restrict__ prices,
    int series_len,
    int first_valid,
    int period,
    float* __restrict__ out)
{
    if (period <= 0 || series_len <= 0) return;
    const SupersmootherCoefs coefs = make_coefs(period);
    supersmoother_3_pole_row_with_coefs(prices, series_len, first_valid, coefs, out);
}

__device__ __forceinline__ void supersmoother_3_pole_row_strided_with_coefs(
    const float* __restrict__ prices,
    int series_len,
    int stride,
    int first_valid,
    const SupersmootherCoefs& coefs,
    float* __restrict__ out)
{
    if (series_len <= 0) return;

    const int start = (first_valid < 0) ? 0 : first_valid;

    // Write NaN prefix
    float* o = out;
    for (int t = 0; t < start && t < series_len; ++t) {
        *o = CUDART_NAN_F;
        o += stride;
    }
    if (start >= series_len) return;

    // Position pointers at first valid
    const float* p = prices + static_cast<size_t>(start) * static_cast<size_t>(stride);
    o = out + static_cast<size_t>(start) * static_cast<size_t>(stride);
    int t = start;

    // 3 seeds
    double y0 = static_cast<double>(*p); *o = static_cast<float>(y0);
    p += stride; o += stride; ++t; if (t >= series_len) return;

    double y1 = static_cast<double>(*p); *o = static_cast<float>(y1);
    p += stride; o += stride; ++t; if (t >= series_len) return;

    double y2 = static_cast<double>(*p); *o = static_cast<float>(y2);
    p += stride; o += stride; ++t;

    // Main loop
    #pragma unroll 4
    for (; t < series_len; ++t) {
        const double x = static_cast<double>(*p);
        const double y_next =
            fma(coefs.coef_prev3, y0,
            fma(coefs.coef_prev2, y1,
            fma(coefs.coef_prev1, y2, coefs.coef_source * x)));

        *o = static_cast<float>(y_next);
        y0 = y1; y1 = y2; y2 = y_next;

        p += stride; o += stride;
    }
}

__device__ __forceinline__ void supersmoother_3_pole_row_strided(
    const float* __restrict__ prices,
    int series_len,
    int stride,
    int first_valid,
    int period,
    float* __restrict__ out)
{
    if (period <= 0 || series_len <= 0) return;
    const SupersmootherCoefs coefs = make_coefs(period);
    supersmoother_3_pole_row_strided_with_coefs(prices, series_len, stride, first_valid, coefs, out);
}

}  // namespace

// --- Public kernels ----------------------------------------------------------
// Notes:
//  * __launch_bounds__(256) is a hint; use 128–256 threads per block at launch.
//  * Each thread processes exactly 1 combo/series. No thread-0-only blocks.

extern "C" __global__ __launch_bounds__(256)
void supersmoother_3_pole_batch_f32(
    const float* __restrict__ prices,   // single price series
    const int*   __restrict__ periods,  // [n_combos]
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out)            // [n_combos, series_len] row-major
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + static_cast<size_t>(combo) * static_cast<size_t>(series_len);

    supersmoother_3_pole_row(prices, series_len, first_valid, period, out_row);
}

// Same as above, but read precomputed coefficients per combo.
// (Use this variant when you can precompute coefs on the host.)
extern "C" __global__ __launch_bounds__(256)
void supersmoother_3_pole_batch_f32_precomp(
    const float* __restrict__ prices,                 // single price series
    const SupersmootherCoefs* __restrict__ coefs_arr, // [n_combos]
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const SupersmootherCoefs coefs = coefs_arr[combo];
    float* out_row = out + static_cast<size_t>(combo) * static_cast<size_t>(series_len);
    supersmoother_3_pole_row_with_coefs(prices, series_len, first_valid, coefs, out_row);
}

// Time-major: many series share the same period.
// Use threads across series for coalesced loads at each timestep.
extern "C" __global__ __launch_bounds__(256)
void supersmoother_3_pole_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,  // [series_len, num_series], time-major
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids, // [num_series]
    float* __restrict__ out_tm)           // [series_len, num_series], time-major
{
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    const int stride = num_series; // time-major stride
    const int first_valid = first_valids[series_idx];

    const float* series_prices = prices_tm + series_idx;
    float*       series_out    = out_tm    + series_idx;

    const SupersmootherCoefs coefs = make_coefs(period);

    supersmoother_3_pole_row_strided_with_coefs(
        series_prices, series_len, stride, first_valid, coefs, series_out);
}
