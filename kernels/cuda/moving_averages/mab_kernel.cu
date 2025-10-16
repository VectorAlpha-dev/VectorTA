// CUDA kernels for Moving Average Bands (MAB)
//
// Behavior mirrors scalar MAB:
// - middle = fastMA
// - upper  = slowMA + devup * RMS(fastMA - slowMA, window=fast_period)
// - lower  = slowMA - devdn * RMS(fastMA - slowMA, window=fast_period)
// Warmup: first_output = first_valid + max(fast,slow) + fast - 1
// Before first_output: write NaN

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}

// Compute dev[t] = sqrt(mean((fast-slow)^2 over last fast_period)) for a single series
// Inputs are 1D fast and slow arrays of length len.
extern "C" __global__ void mab_dev_from_ma_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    int fast_period,
    int first_valid,
    int len,
    float* __restrict__ dev_out // length=len; dev_out[t] valid only for t>=first_output
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return; // single-thread sequential
    if (len <= 0 || fast_period <= 0) return;

    const int first_output = first_valid + max(fast_period, 0) + fast_period - 1; // max(slow,fast) handled by caller when used; here we only use fast
    // Note: in shared-weights fast-path, slow period equals that used to build slow[];
    // first_output used by the "apply" kernel for NaN policy; here we safely fill entries >= first_output

    // Prefill leading with NaN to be robust if read directly (optional)
    for (int t = 0; t < min(first_output, len); ++t) {
        dev_out[t] = qnan32();
    }
    if (first_output >= len) return;

    const int start0 = first_output + 1 - fast_period;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start0 + k;
        const double d = (double)fast[idx] - (double)slow[idx];
        sumsq += d * d;
    }
    dev_out[first_output] = (float)sqrt(sumsq / (double)fast_period);

    for (int i = first_output + 1; i < len; ++i) {
        const int old_idx = i - fast_period;
        const double oldd = (double)fast[old_idx] - (double)slow[old_idx];
        const double newd = (double)fast[i] - (double)slow[i];
        sumsq += newd * newd - oldd * oldd;
        dev_out[i] = (float)sqrt(sumsq / (double)fast_period);
    }
}

// Apply shared dev/MA to produce outputs for N rows (combos) where only devup/devdn vary.
// fast_period is used only to compute first_output and match warmup/NaN behavior.
extern "C" __global__ void mab_apply_dev_shared_ma_batch_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    const float* __restrict__ dev, // length=len
    int fast_period,
    int slow_period,
    int first_valid,
    int len,
    const float* __restrict__ devups, // length=rows
    const float* __restrict__ devdns, // length=rows
    int rows,
    float* __restrict__ out_upper,  // rows x len
    float* __restrict__ out_middle, // rows x len
    float* __restrict__ out_lower   // rows x len
) {
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const int row_off = row * len;
    const float devup = devups[row];
    const float devdn = devdns[row];

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nanf = qnan32();
    while (t < len) {
        float u = nanf, m = nanf, l = nanf;
        if (t >= warm) {
            const float d = dev[t];
            const float sm = slow[t];
            m = fast[t];
            u = sm + devup * d;
            l = sm - devdn * d;
        }
        out_upper[row_off + t]  = u;
        out_middle[row_off + t] = m;
        out_lower[row_off + t]  = l;
        t += stride;
    }
}

// Generic single-row kernel: compute dev and outputs from row-own fast/slow arrays.
extern "C" __global__ void mab_single_row_from_ma_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    int fast_period,
    int slow_period,
    int first_valid,
    int len,
    float devup,
    float devdn,
    float* __restrict__ out_upper,  // length=len (target row section)
    float* __restrict__ out_middle,
    float* __restrict__ out_lower
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return; // sequential per-row
    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const float nanf = qnan32();

    for (int t = 0; t < min(warm, len); ++t) {
        out_upper[t] = nanf;
        out_middle[t] = nanf;
        out_lower[t] = nanf;
    }
    if (warm >= len) return;

    // First valid output
    int start = (warm + 1) - fast_period;
    if (start < 0) start = 0;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start + k;
        const double d = (double)fast[idx] - (double)slow[idx];
        sumsq += d * d;
    }
    const float dev0 = (float)sqrt(sumsq / (double)fast_period);
    out_middle[warm] = fast[warm];
    out_upper[warm] = slow[warm] + devup * dev0;
    out_lower[warm] = slow[warm] - devdn * dev0;

    for (int i = warm + 1; i < len; ++i) {
        const int old_idx = i - fast_period;
        const double oldd = (double)fast[old_idx] - (double)slow[old_idx];
        const double newd = (double)fast[i] - (double)slow[i];
        sumsq += newd * newd - oldd * oldd;
        const float dev = (float)sqrt(sumsq / (double)fast_period);
        out_middle[i] = fast[i];
        out_upper[i] = slow[i] + devup * dev;
        out_lower[i] = slow[i] - devdn * dev;
    }
}

// Many-series (time-major), one param. fast and slow are rows x cols (time-major).
extern "C" __global__ void mab_many_series_one_param_time_major_f32(
    const float* __restrict__ fast_tm,
    const float* __restrict__ slow_tm,
    const int* __restrict__ first_valids, // length=cols
    int cols,
    int rows,
    int fast_period,
    int slow_period,
    float devup,
    float devdn,
    float* __restrict__ out_upper_tm,
    float* __restrict__ out_middle_tm,
    float* __restrict__ out_lower_tm
) {
    const int s = blockIdx.y;
    if (s >= cols) return;
    const int fv = first_valids[s];
    const int warm = fv + max(fast_period, slow_period) + fast_period - 1;

    if (threadIdx.x != 0 || blockIdx.x != 0) return; // one thread per series
    const int stride = cols;
    const float nanf = qnan32();

    for (int t = 0; t < min(warm, rows); ++t) {
        const int idx = t * stride + s;
        out_upper_tm[idx] = nanf;
        out_middle_tm[idx] = nanf;
        out_lower_tm[idx] = nanf;
    }
    if (warm >= rows) return;

    int start = (warm + 1) - fast_period;
    if (start < 0) start = 0;
    double sumsq = 0.0;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = (start + k) * stride + s;
        const double d = (double)fast_tm[idx] - (double)slow_tm[idx];
        sumsq += d * d;
    }
    {
        const int i = warm;
        const int idx = i * stride + s;
        const float dev = (float)sqrt(sumsq / (double)fast_period);
        out_middle_tm[idx] = fast_tm[idx];
        out_upper_tm[idx] = slow_tm[idx] + devup * dev;
        out_lower_tm[idx] = slow_tm[idx] - devdn * dev;
    }

    for (int i = warm + 1; i < rows; ++i) {
        const int old_idx = (i - fast_period) * stride + s;
        const int new_idx = i * stride + s;
        const double oldd = (double)fast_tm[old_idx] - (double)slow_tm[old_idx];
        const double newd = (double)fast_tm[new_idx] - (double)slow_tm[new_idx];
        sumsq += newd * newd - oldd * oldd;
        const float dev = (float)sqrt(sumsq / (double)fast_period);
        out_middle_tm[new_idx] = fast_tm[new_idx];
        out_upper_tm[new_idx] = slow_tm[new_idx] + devup * dev;
        out_lower_tm[new_idx] = slow_tm[new_idx] - devdn * dev;
    }
}

