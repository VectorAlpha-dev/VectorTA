// CUDA kernels for the single-pole High-Pass filter (optimized).
//
// Exported symbols (referenced by Rust wrapper):
//   - highpass_batch_f32
//   - highpass_many_series_one_param_time_major_f32
// Keep these names in sync with cust::Module::get_function lookups.
//
// - FP64 internal math preserved to match the CPU reference path.
// - Uses sincospi() to compute sin/cos together for theta = 2*pi/period.
// - Grid-stride loops so one launch scales across SMs and problem sizes.
// - Batch kernel broadcasts prices[t] within a warp via __shfl_sync to avoid
//   redundant global loads (no shared memory, no __syncthreads()).
// - Uses fused multiply-add (fma) in FP64 for accuracy and throughput.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <math_functions.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Helpers: lane id and first-active-lane in a warp
static __forceinline__ __device__ int lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}
static __forceinline__ __device__ int first_active_lane(unsigned mask) {
    // __ffs() returns 1-based index of least-significant set bit
    return __ffs(mask) - 1;
}

// Compute HPF coefficients from period using sincospi() in FP64.
static __forceinline__ __device__
void hpf_coeffs_from_period(int period, double& c, double& oma, bool& ok) {
    ok = false;
    if (period <= 0) return;

    // sin(theta) and cos(theta) with theta = 2*pi/period ->
    // sincospi(2/period) computes both in one call (CUDA math API).
    double s, co;
    sincospi(2.0 / static_cast<double>(period), &s, &co);
    if (fabs(co) < 1e-12) return;

    const double alpha = 1.0 + ((s - 1.0) / co);
    c   = 1.0 - 0.5 * alpha;   // (1 - α/2)
    oma = 1.0 - alpha;         // (1 - α)
    ok = true;
}

// FP32 coefficient helper for the warp-scan batch kernel.
static __forceinline__ __device__
void hpf_coeffs_from_period_f32(int period, float& c, float& oma, bool& ok) {
    ok = false;
    if (period <= 0) return;

    float s, co;
    // theta = 2*pi/period => sincos(pi * (2/period))
    sincospif(2.0f / static_cast<float>(period), &s, &co);
    if (fabsf(co) < 1e-6f) return;

    const float alpha = 1.0f + ((s - 1.0f) / co);
    c   = 1.0f - 0.5f * alpha;
    oma = 1.0f - alpha;
    ok = true;
}

// Batch warp-scan kernel: one warp computes one combo (row) and emits 32 timesteps
// per iteration via an inclusive scan over the affine high-pass transform:
//   y_t = oma * y_{t-1} + c * (x_t - x_{t-1})
//
// - blockDim.x must be exactly 32
// - output is written once: prefix NaNs up to first_valid-1, then all t>=first_valid are computed
extern "C" __global__
void highpass_batch_warp_scan_f32(const float* __restrict__ prices,
                                  int first_valid,
                                  const int* __restrict__ periods,
                                  int series_len,
                                  int n_combos,
                                  float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0) return;
    if (threadIdx.x >= 32) return;

    const int period = periods[combo];
    float c, oma; bool ok;
    hpf_coeffs_from_period_f32(period, c, oma, ok);
    if (!ok || period > series_len) return;

    int fv = first_valid;
    if (fv < 0) fv = 0;
    if (fv > series_len) fv = series_len;

    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;
    const size_t base = (size_t)combo * (size_t)series_len;

    // NaN prefix up to fv-1
    for (int t = lane; t < fv; t += 32) {
        out[base + (size_t)t] = CUDART_NAN_F;
    }
    if (fv >= series_len) return;

    float y_prev = 0.0f;
    if (lane == 0) {
        const float x0 = prices[fv];
        y_prev = x0;
        out[base + (size_t)fv] = y_prev;
    }
    y_prev = __shfl_sync(mask, y_prev, 0);

    int t0 = fv + 1;
    if (t0 >= series_len) return;

    for (int tile = t0; tile < series_len; tile += 32) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        float A = valid ? oma : 1.0f;
        float B = 0.0f;
        if (valid) {
            const float x = prices[t];
            const float xm1 = prices[t - 1];
            B = c * (x - xm1);
        }

        // Inclusive scan of composed affine transforms (A, B)
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }

        const float y = fmaf(A, y_prev, B);
        if (valid) {
            out[base + (size_t)t] = y;
        }

        const int remaining = series_len - tile;
        const int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        y_prev = __shfl_sync(mask, y, last_lane);
    }
}

extern "C" __global__
void highpass_batch_f32(const float* __restrict__ prices,
                        int first_valid,
                        const int*   __restrict__ periods,
                        int series_len,
                        int n_combos,
                        float* __restrict__ out) {

    if (series_len <= 0 || n_combos <= 0) return;

    // Grid-stride loop over combos
    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += blockDim.x * gridDim.x)
    {
        const int period = periods[combo];
        double c, oma; bool ok;
        hpf_coeffs_from_period(period, c, oma, ok);
        if (!ok || period > series_len) {
            // Skip invalid; leave output untouched.
            continue;
        }

        const int base = combo * series_len;

        int fv = first_valid;
        if (fv < 0) fv = 0;
        if (fv > series_len) fv = series_len;

        // NaN prefix for missing data
        for (int t = 0; t < fv; ++t) {
            out[base + t] = CUDART_NAN_F;
        }
        if (fv >= series_len) continue;

        // Broadcast prices[fv] once per warp
        unsigned mask  = __activemask();
        int leader     = first_active_lane(mask);
        float p0_f     = (lane_id() == leader) ? prices[fv] : 0.0f;
        p0_f           = __shfl_sync(mask, p0_f, leader);
        double prev_x  = static_cast<double>(p0_f);
        double prev_y  = prev_x;
        out[base + fv] = static_cast<float>(prev_y);

        // Time recursion; broadcast prices[t] once/warp/step
        for (int t = fv + 1; t < series_len; ++t) {
            float xf = (lane_id() == leader) ? prices[t] : 0.0f;
            xf       = __shfl_sync(mask, xf, leader);
            const double x    = static_cast<double>(xf);
            const double diff = x - prev_x;
            const double y    = fma(oma, prev_y, c * diff); // fused multiply-add
            out[base + t]     = static_cast<float>(y);
            prev_x = x;
            prev_y = y;
        }
    }
}

extern "C" __global__
void highpass_many_series_one_param_time_major_f32(const float* __restrict__ prices_tm,
                                                   const int*   __restrict__ first_valids,
                                                   int period,
                                                   int num_series,
                                                   int series_len,
                                                   float* __restrict__ out_tm) {
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    double c, oma; bool ok;
    hpf_coeffs_from_period(period, c, oma, ok);
    if (!ok) return;

    const int stride = num_series;

    // Grid-stride over series
    for (int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
         series_idx < num_series;
         series_idx += blockDim.x * gridDim.x)
    {
        int fv = first_valids ? first_valids[series_idx] : 0;
        if (fv < 0) fv = 0;
        if (fv > series_len) fv = series_len;

        // Fill NaN prefix
        int idx = series_idx;
        for (int t = 0; t < fv; ++t) {
            out_tm[idx] = CUDART_NAN_F;
            idx += stride;
        }
        if (fv >= series_len) continue;

        // Seed at first valid
        idx = fv * stride + series_idx;
        double prev_x = static_cast<double>(prices_tm[idx]);
        double prev_y = prev_x;
        out_tm[idx]   = static_cast<float>(prev_y);

        // advance in time-major layout by 'stride'
        for (int t = fv + 1; t < series_len; ++t) {
            idx += stride;
            const double x    = static_cast<double>(prices_tm[idx]);
            const double diff = x - prev_x;
            const double y    = fma(oma, prev_y, c * diff);
            out_tm[idx]       = static_cast<float>(y);
            prev_x = x;
            prev_y = y;
        }
    }
}
