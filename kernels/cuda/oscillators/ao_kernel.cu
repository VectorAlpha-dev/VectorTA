// CUDA kernels for Awesome Oscillator (AO)
//
// Math: AO = SMA_short(hl2) - SMA_long(hl2)
//
// Batch kernel (one series × many params): prefix-sum based — O(1) per output.
// Many-series × one-param (time-major): per-series rolling sums with strided loads.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef AO_NAN_F
#define AO_NAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Batch kernel using prefix sums (double precision accumulation on host).
// Arguments:
//  - prefix: exclusive prefix sum of hl2 in float64, length = len+1, prefix[0]=0
//  - len: series length
//  - first_valid: index of first non-NaN element in hl2
//  - shorts, longs: arrays (n_combos) with short/long periods per row
//  - out: row-major [n_combos][len]
extern "C" __global__ void ao_batch_f32(const double* __restrict__ prefix,
                                         int len,
                                         int first_valid,
                                         const int* __restrict__ shorts,
                                         const int* __restrict__ longs,
                                         int n_combos,
                                         float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int s = shorts[combo];
    const int l = longs[combo];
    if (UNLIKELY(s <= 0 || l <= 0 || s >= l)) {
        // Invalid params: fill row with NaN to mirror CPU safeguards
        const int base = combo * len;
        for (int t = 0; t < len; ++t) out[base + t] = AO_NAN_F;
        return;
    }

    const int warm = first_valid + l - 1; // first valid AO index
    const int row_off = combo * len;

    // Parallelize across time (grid.x * blockDim.x)
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double inv_s = 1.0 / (double)s;
    const double inv_l = 1.0 / (double)l;

    while (t < len) {
        float out_val = AO_NAN_F;
        if (t >= warm) {
            int start_s = t + 1 - s;
            int start_l = t + 1 - l;
            if (start_s < 0) start_s = 0;
            if (start_l < 0) start_l = 0;
            const double sum_s = prefix[t + 1] - prefix[start_s];
            const double sum_l = prefix[t + 1] - prefix[start_l];
            const double ao = sum_s * inv_s - sum_l * inv_l;
            out_val = (float)ao;
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

extern "C" __global__ void ao_many_series_one_param_f32(
    const float* __restrict__ prices_tm, // time-major: [t][series]
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int short_p,
    int long_p,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    // Invalid: write NaNs for this series and return
    if (UNLIKELY(short_p <= 0 || long_p <= 0 || short_p >= long_p)) {
        float* o = out_tm + series;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = AO_NAN_F;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* o = out_tm + series;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = AO_NAN_F;
        return;
    }

    const int warm = first_valid + long_p - 1;

    // Prefill NaNs up to warm-1
    {
        float* o = out_tm + series;
        for (int row = 0; row < warm; ++row, o += num_series) *o = AO_NAN_F;
    }

    // Build initial window sums for short and long (strided by num_series)
    double sum_s = 0.0; // double for stability
    double sum_l = 0.0;
    const float* ps = prices_tm + (first_valid) * (size_t)num_series + series;
    const float* pl = ps;
    for (int k = 0; k < long_p; ++k) {
        const float v = *pl;
        sum_l += (double)v;
        if (k >= long_p - short_p) sum_s += (double)v;
        pl += num_series;
    }

    // First AO value at warm
    *(out_tm + (size_t)warm * num_series + series) = (float)(sum_s / (double)short_p - sum_l / (double)long_p);

    // Rolling updates
    const float* cur = prices_tm + ((size_t)warm + 1) * num_series + series;
    const float* old_s = prices_tm + ((size_t)first_valid + (long_p - short_p)) * num_series + series;
    const float* old_l = prices_tm + ((size_t)first_valid) * num_series + series;
    float*       dst = out_tm + ((size_t)warm + 1) * num_series + series;

    for (int row = warm + 1; row < series_len; ++row) {
        sum_s += (double)(*cur) - (double)(*old_s);
        sum_l += (double)(*cur) - (double)(*old_l);
        *dst = (float)(sum_s / (double)short_p - sum_l / (double)long_p);
        cur += num_series;
        old_s += num_series;
        old_l += num_series;
        dst += num_series;
    }
}
