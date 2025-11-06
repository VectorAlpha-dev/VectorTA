// CUDA kernels for the Volume Weighted Moving Average (VWMA).
//
// Each parameter combination (period) is assigned to a block in the Y dimension
// while the X dimension iterates over time indices. The kernel operates on
// precomputed prefix sums of price*volume and volume so that every thread only
// performs two subtractions and one division per output value.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Optional read-only load helper. On modern GPUs default loads are typically cached,
// but __ldg() can still help some patterns. Safe no-op fallback if unavailable.
#ifndef LDG
#  if __CUDA_ARCH__ >= 350
#    define LDG(p) __ldg(p)
#  else
#    define LDG(p) (*(p))
#  endif
#endif

// Quiet NaN helper to avoid any libm edge cases in device code.
__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }

extern "C" __global__
void vwma_batch_f32(const double* __restrict__ pv_prefix,
                    const double* __restrict__ vol_prefix,
                    const int*    __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int warm   = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        float value;
        if (t < warm) {
            value = nan_f32();
        } else {
            // prev = t - period
            const int prev = t - period;

            // Double math for accuracy
            double sum_pv  = LDG(&pv_prefix[t]);
            double sum_vol = LDG(&vol_prefix[t]);

            if (prev >= 0) {
                sum_pv  -= LDG(&pv_prefix[prev]);
                sum_vol -= LDG(&vol_prefix[prev]);
            }

            value = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                     : nan_f32();
        }
        out[base_out + t] = value;
        t += stride;
    }
}

extern "C" __global__
void vwma_multi_series_one_param_f32(const double* __restrict__ pv_prefix_tm,
                                     const double* __restrict__ vol_prefix_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const int warm = first_valids[series_idx] + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = nan_f32();
        } else {
            const int prev = t - period;
            const int idx  = out_idx;

            double sum_pv  = LDG(&pv_prefix_tm[idx]);
            double sum_vol = LDG(&vol_prefix_tm[idx]);

            if (prev >= 0) {
                const int prev_idx = prev * num_series + series_idx;
                sum_pv  -= LDG(&pv_prefix_tm[prev_idx]);
                sum_vol -= LDG(&vol_prefix_tm[prev_idx]);
            }

            out_tm[out_idx] = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                               : nan_f32();
        }
        t += stride;
    }
}

// Multi-series, one-period (time-major, coalesced across series)
// Threads in X -> series (contiguous), threads in Y -> a small tile of time steps.
extern "C" __global__
void vwma_multi_series_one_param_tm_coalesced_f32(const double* __restrict__ pv_prefix_tm,
                                                  const double* __restrict__ vol_prefix_tm,
                                                  int period,
                                                  int num_series,
                                                  int series_len,
                                                  const int* __restrict__ first_valids,
                                                  float* __restrict__ out_tm)
{
    // Map threads across series for coalesced accesses:
    const int series_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    // Each thread handles one series across a grid-stride in time (tile in blockDim.y).
    // Precompute warm-up per series (avoids re-reading first_valids inside the loop).
    const int warm = first_valids[series_idx] + period - 1;

    // 2D grid-stride over time, with threadIdx.y as an intra-block time tile
    for (int t = blockIdx.x * blockDim.y + threadIdx.y;
         t < series_len;
         t += gridDim.x * blockDim.y)
    {
        const int out_idx = t * num_series + series_idx;

        if (t < warm) {
            out_tm[out_idx] = nan_f32();
            continue;
        }

        const int prev = t - period;

        // Coalesced loads across the warp for both "current" and "prev"
        double sum_pv  = LDG(&pv_prefix_tm[out_idx]);
        double sum_vol = LDG(&vol_prefix_tm[out_idx]);

        if (prev >= 0) {
            const int prev_idx = prev * num_series + series_idx;
            sum_pv  -= LDG(&pv_prefix_tm[prev_idx]);
            sum_vol -= LDG(&vol_prefix_tm[prev_idx]);
        }

        out_tm[out_idx] = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                           : nan_f32();
    }
}
