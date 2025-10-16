// CUDA kernels for Elder Ray Index (ERI)
//
// Math mirrors src/indicators/eri.rs exactly:
// bull[i] = high[i] - MA[i]
// bear[i] = low[i]  - MA[i]
// Warmup/NaN semantics:
//   - Outputs before warmup = first_valid + period - 1 are NaN regardless of MA availability.
//   - If any input at i is NaN, result at i becomes NaN by normal FP rules.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void eri_batch_f32(
    const float* __restrict__ high,   // [series_len]
    const float* __restrict__ low,    // [series_len]
    const float* __restrict__ ma,     // [series_len] (for this row)
    int series_len,
    int first_valid,
    int period,
    float* __restrict__ bull,         // [series_len] (this row)
    float* __restrict__ bear          // [series_len] (this row)
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= series_len) return;

    const int warm = first_valid + period - 1;
    if (i < warm) {
        const float nanv = nanf("");
        if (bull) bull[i] = nanv;
        if (bear) bear[i] = nanv;
        return;
    }

    const float h = high[i];
    const float l = low[i];
    const float m = ma[i];
    // Normal FP rules make any op with NaN yield NaN, matching scalar behavior
    if (bull) bull[i] = h - m;
    if (bear) bear[i] = l - m;
}

// Many-series, one param (time-major):
// Layout: matrices are time-major with shape [rows x cols]; index = t*cols + s
extern "C" __global__ void eri_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ ma_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids, // per-series first valid (triple-validity)
    int period,
    float* __restrict__ bull_tm,
    float* __restrict__ bear_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    const int fv = first_valids[s];
    const int warm = fv + period - 1;

    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;
        if (t < warm) {
            const float nanv = nanf("");
            if (bull_tm) bull_tm[idx] = nanv;
            if (bear_tm) bear_tm[idx] = nanv;
            continue;
        }
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float m = ma_tm[idx];
        if (bull_tm) bull_tm[idx] = h - m;
        if (bear_tm) bear_tm[idx] = l - m;
    }
}

