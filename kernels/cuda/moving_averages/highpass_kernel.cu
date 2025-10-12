// CUDA kernels for the single-pole High-Pass filter (optimized).
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

extern "C" __global__
void highpass_batch_f32(const float* __restrict__ prices,
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

        // Broadcast prices[0] once per warp
        unsigned mask  = __activemask();
        int leader     = first_active_lane(mask);
        float p0_f     = (lane_id() == leader) ? prices[0] : 0.0f;
        p0_f           = __shfl_sync(mask, p0_f, leader);
        double prev_x  = static_cast<double>(p0_f);
        double prev_y  = prev_x;
        out[base]      = static_cast<float>(prev_y);

        // Time recursion; broadcast prices[t] once/warp/step
        for (int t = 1; t < series_len; ++t) {
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
        int idx       = series_idx;
        double prev_x = static_cast<double>(prices_tm[idx]);
        double prev_y = prev_x;
        out_tm[idx]   = static_cast<float>(prev_y);

        // advance in time-major layout by 'stride'
        for (int t = 1; t < series_len; ++t) {
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
