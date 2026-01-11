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

// Batch warp-scan kernel: one warp computes one combo (row) and emits 32 timesteps
// per iteration via an inclusive scan over the 3x3 affine state update:
//
//   y[t] = coef_source*x[t] + coef_prev1*y[t-1] + coef_prev2*y[t-2] + coef_prev3*y[t-3]
//
// State is s[t] = [y[t], y[t-1], y[t-2]] and:
//   s[t] = M*s[t-1] + [coef_source*x[t], 0, 0]
//   M = [[coef_prev1, coef_prev2, coef_prev3],
//        [    1     ,     0     ,     0     ],
//        [    0     ,     1     ,     0     ]]
//
// - blockDim.x must be exactly 32
// - warmup: [0, first_valid) = NaN; then 3 seeds = raw input stream at first_valid..first_valid+2
extern "C" __global__
void supersmoother_3_pole_batch_warp_scan_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0) return;
    if (threadIdx.x >= 32) return;

    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;

    float* out_row = out + static_cast<size_t>(combo) * static_cast<size_t>(series_len);

    if (first_valid < 0 || first_valid >= series_len) {
        for (int t = lane; t < series_len; t += 32) out_row[t] = CUDART_NAN_F;
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > series_len) {
        for (int t = lane; t < series_len; t += 32) out_row[t] = CUDART_NAN_F;
        return;
    }

    // Prefix NaNs only where needed
    for (int t = lane; t < first_valid; t += 32) out_row[t] = CUDART_NAN_F;

    // Seeds: first 3 samples are raw inputs (matches scalar)
    if (lane == 0) {
        out_row[first_valid] = prices[first_valid];
        if (first_valid + 1 < series_len) out_row[first_valid + 1] = prices[first_valid + 1];
        if (first_valid + 2 < series_len) out_row[first_valid + 2] = prices[first_valid + 2];
    }
    if (first_valid + 2 >= series_len) return;

    const SupersmootherCoefs coefs = make_coefs(period);

    // Previous state at t0-1 = first_valid+2
    double s0_prev = 0.0;
    double s1_prev = 0.0;
    double s2_prev = 0.0;
    if (lane == 0) {
        s2_prev = static_cast<double>(prices[first_valid]);     // y[t0-3]
        s1_prev = static_cast<double>(prices[first_valid + 1]); // y[t0-2]
        s0_prev = static_cast<double>(prices[first_valid + 2]); // y[t0-1]
    }
    s0_prev = __shfl_sync(mask, s0_prev, 0);
    s1_prev = __shfl_sync(mask, s1_prev, 0);
    s2_prev = __shfl_sync(mask, s2_prev, 0);

    // Constant state-transition matrix M
    const double m00 = coefs.coef_prev1;
    const double m01 = coefs.coef_prev2;
    const double m02 = coefs.coef_prev3;
    const double m10 = 1.0;
    const double m11 = 0.0;
    const double m12 = 0.0;
    const double m20 = 0.0;
    const double m21 = 1.0;
    const double m22 = 0.0;

    const int t0 = first_valid + 3;
    if (t0 >= series_len) return;

    struct Mat3 {
        double m00, m01, m02;
        double m10, m11, m12;
        double m20, m21, m22;
    };
    auto mat_mul = [] __device__ (const Mat3& A, const Mat3& B) -> Mat3 {
        Mat3 R;
        R.m00 = fma(A.m00, B.m00, fma(A.m01, B.m10, A.m02 * B.m20));
        R.m01 = fma(A.m00, B.m01, fma(A.m01, B.m11, A.m02 * B.m21));
        R.m02 = fma(A.m00, B.m02, fma(A.m01, B.m12, A.m02 * B.m22));

        R.m10 = fma(A.m10, B.m00, fma(A.m11, B.m10, A.m12 * B.m20));
        R.m11 = fma(A.m10, B.m01, fma(A.m11, B.m11, A.m12 * B.m21));
        R.m12 = fma(A.m10, B.m02, fma(A.m11, B.m12, A.m12 * B.m22));

        R.m20 = fma(A.m20, B.m00, fma(A.m21, B.m10, A.m22 * B.m20));
        R.m21 = fma(A.m20, B.m01, fma(A.m21, B.m11, A.m22 * B.m21));
        R.m22 = fma(A.m20, B.m02, fma(A.m21, B.m12, A.m22 * B.m22));
        return R;
    };
    auto mat_apply = [] __device__ (const Mat3& A, double x0, double x1, double x2,
                                    double& y0, double& y1, double& y2) {
        y0 = fma(A.m00, x0, fma(A.m01, x1, A.m02 * x2));
        y1 = fma(A.m10, x0, fma(A.m11, x1, A.m12 * x2));
        y2 = fma(A.m20, x0, fma(A.m21, x1, A.m22 * x2));
    };

    const Mat3 M { m00, m01, m02, m10, m11, m12, m20, m21, m22 };
    const Mat3 I { 1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0 };

    for (int tile = t0; tile < series_len; tile += 32) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        Mat3 P = valid ? M : I;

        // Per-step u term: v = [coef_source * x[t], 0, 0]
        double v0 = valid ? (coefs.coef_source * static_cast<double>(prices[t])) : 0.0;
        double v1 = 0.0;
        double v2 = 0.0;

        // Inclusive scan over composed transforms: (P_cur, v_cur) ∘ (P_prev, v_prev)
        // where s' = P*s + v and composition is:
        //   P = P_cur * P_prev
        //   v = P_cur * v_prev + v_cur
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const double p00_prev = __shfl_up_sync(mask, P.m00, offset);
            const double p01_prev = __shfl_up_sync(mask, P.m01, offset);
            const double p02_prev = __shfl_up_sync(mask, P.m02, offset);
            const double p10_prev = __shfl_up_sync(mask, P.m10, offset);
            const double p11_prev = __shfl_up_sync(mask, P.m11, offset);
            const double p12_prev = __shfl_up_sync(mask, P.m12, offset);
            const double p20_prev = __shfl_up_sync(mask, P.m20, offset);
            const double p21_prev = __shfl_up_sync(mask, P.m21, offset);
            const double p22_prev = __shfl_up_sync(mask, P.m22, offset);
            const double v0_prev  = __shfl_up_sync(mask, v0,  offset);
            const double v1_prev  = __shfl_up_sync(mask, v1,  offset);
            const double v2_prev  = __shfl_up_sync(mask, v2,  offset);
            if (lane >= offset) {
                const Mat3 P_prev { p00_prev, p01_prev, p02_prev, p10_prev, p11_prev, p12_prev, p20_prev, p21_prev, p22_prev };
                const Mat3 P_cur = P;
                P = mat_mul(P_cur, P_prev);

                double tv0, tv1, tv2;
                mat_apply(P_cur, v0_prev, v1_prev, v2_prev, tv0, tv1, tv2);
                v0 += tv0;
                v1 += tv1;
                v2 += tv2;
            }
        }

        // Apply prefix transform to previous state s_prev
        double h0, h1, h2;
        mat_apply(P, s0_prev, s1_prev, s2_prev, h0, h1, h2);

        const double y0 = h0 + v0;
        const double y1 = h1 + v1;
        const double y2 = h2 + v2;

        if (valid) {
            out_row[t] = static_cast<float>(y0);
        }

        const int remaining = series_len - tile;
        const int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        s0_prev = __shfl_sync(mask, y0, last_lane);
        s1_prev = __shfl_sync(mask, y1, last_lane);
        s2_prev = __shfl_sync(mask, y2, last_lane);
    }
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
