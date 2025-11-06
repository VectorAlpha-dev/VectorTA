// CUDA kernels for Absolute Price Oscillator (APO).
//
// APO = EMA(short_period) - EMA(long_period)
//
// Semantics (must match scalar):
// - Prefix [0..first_valid) is NaN
// - At t = first_valid: output 0.0 (both EMAs seeded to the same first valid price)
// - Thereafter, sequential EMA updates per timestep; no extended warmup NaNs
// - Mid-stream NaNs in input propagate through the recurrence (no special handling)

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void apo_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ short_periods,
                   const float* __restrict__ short_alphas,
                   const int*   __restrict__ long_periods,
                   const float* __restrict__ long_alphas,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int  sp    = short_periods[combo];
    const int  lp    = long_periods[combo];
    if (sp <= 0 || lp <= 0 || sp >= lp) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const float a_s  = short_alphas[combo];
    const float a_l  = long_alphas[combo];
    const float oma_s= 1.0f - a_s;
    const float oma_l= 1.0f - a_l;

    const int base = combo * series_len;

    // Initialize only the NaN prefix; remaining cells will be overwritten
    for (int i = threadIdx.x; i < first_valid; i += blockDim.x) {
        out[base + i] = NAN;
    }

    // Single-threaded sequential scan per combo
    if (threadIdx.x != 0) return;

    // Seed EMAs at first_valid
    float se = prices[first_valid];
    float le = se;
    out[base + first_valid] = 0.0f;

    // Advance
    for (int i = first_valid + 1; i < series_len; ++i) {
        const float x = prices[i];
        // Propagate NaNs naturally via arithmetic
        se = a_s * x + oma_s * se;
        le = a_l * x + oma_l * le;
        out[base + i] = se - le;
    }
}

// Many-series, one (short,long) param. Time-major layout.
extern "C" __global__
void apo_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int short_period,
                                   float short_alpha,
                                   int long_period,
                                   float long_alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.x; // one block per series; thread 0 scans
    if (series_idx >= num_series || series_len <= 0) return;
    if (short_period <= 0 || long_period <= 0 || short_period >= long_period) return;

    const int stride = num_series; // time-major stride
    int fv = first_valids[series_idx];
    if (fv < 0) fv = 0;
    if (fv >= series_len) return;

    const float a_s   = short_alpha;
    const float a_l   = long_alpha;
    const float oma_s = 1.0f - a_s;
    const float oma_l = 1.0f - a_l;

    // Initialize only the NaN prefix for this series
    for (int t = threadIdx.x; t < fv; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    if (threadIdx.x != 0) return;

    // Seed at first_valid
    float se = prices_tm[fv * stride + series_idx];
    float le = se;
    out_tm[fv * stride + series_idx] = 0.0f;

    for (int t = fv + 1; t < series_len; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        se = a_s * x + oma_s * se;
        le = a_l * x + oma_l * le;
        out_tm[t * stride + series_idx] = se - le;
    }
}

