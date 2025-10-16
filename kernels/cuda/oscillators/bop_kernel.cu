// CUDA kernels for Balance of Power (BOP)
//
// Math per time index t:
//   bop = (close[t] - open[t]) / (high[t] - low[t])
//   if (high[t] - low[t]) <= 0 -> 0.0
//   warmup: write NaN for t < first_valid (first index where OHLC are all non-NaN)

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef BOP_NAN_F
#define BOP_NAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// One series × (degenerate) many-params (n_combos=1). We keep the signature similar
// to other batch kernels for API parity on the Rust side. Grid-Y should be 1.
extern "C" __global__ void bop_batch_f32(const float* __restrict__ open,
                                          const float* __restrict__ high,
                                          const float* __restrict__ low,
                                          const float* __restrict__ close,
                                          int len,
                                          int first_valid,
                                          float* __restrict__ out)
{
    const int combo = blockIdx.y; // kept for parity; expected 0
    if (UNLIKELY(combo > 0)) return; // no params -> only one row

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = BOP_NAN_F;
        if (t >= first_valid) {
            const float den = high[t] - low[t];
            if (den <= 0.0f) {
                out_val = 0.0f;
            } else {
                out_val = (close[t] - open[t]) / den;
            }
        }
        out[t] = out_val; // single row
        t += stride;
    }
}

// Many-series × one-param (time-major). No params for BOP; we compute per series.
// Inputs are time-major: [row=t][col=series] => idx = t*num_series + series
extern "C" __global__ void bop_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series id
    if (s >= num_series) return;

    const int fv = first_valids[s];
    if (UNLIKELY(fv < 0 || fv >= series_len)) {
        // All-NaN series or invalid first_valid -> output NaNs
        float* o = out_tm + s;
        for (int t = 0; t < series_len; ++t, o += num_series) *o = BOP_NAN_F;
        return;
    }

    // Fill NaNs before first_valid
    {
        float* o = out_tm + s;
        for (int t = 0; t < fv; ++t, o += num_series) *o = BOP_NAN_F;
    }

    // Compute from first_valid onward
    const float* po = open_tm  + (size_t)fv * num_series + s;
    const float* ph = high_tm  + (size_t)fv * num_series + s;
    const float* pl = low_tm   + (size_t)fv * num_series + s;
    const float* pc = close_tm + (size_t)fv * num_series + s;
    float*       pd = out_tm   + (size_t)fv * num_series + s;

    for (int t = fv; t < series_len; ++t) {
        const float den = *ph - *pl;
        float v;
        if (den <= 0.0f) {
            v = 0.0f;
        } else {
            v = (*pc - *po) / den;
        }
        *pd = v;
        po += num_series; ph += num_series; pl += num_series; pc += num_series; pd += num_series;
    }
}

