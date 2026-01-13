








#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}

__device__ __forceinline__ float sma_from_prefix_f32(
    const double* __restrict__ pref_sum,
    const int* __restrict__ pref_nan,
    int t,
    int period
) {
    const int t1 = t + 1;
    const int t0 = t + 1 - period;
    if ((pref_nan[t1] - pref_nan[t0]) != 0) return qnan32();
    const double sum = pref_sum[t1] - pref_sum[t0];
    return (float)(sum / (double)period);
}






extern "C" __global__ void mab_batch_from_prefix_sma_f32(
    const double* __restrict__ pref_close_sum,
    const int* __restrict__ pref_close_nan,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const float* __restrict__ devups,
    const float* __restrict__ devdns,
    int len,
    int first_valid,
    int rows,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    const int fast_period = fast_periods[row];
    const int slow_period = slow_periods[row];
    const float devup = devups[row];
    const float devdn = devdns[row];

    if (fast_period <= 0 || slow_period <= 0 || len <= 0) return;

    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const int row_off = row * len;
    const float nanf = qnan32();

    for (int t = 0; t < min(warm, len); ++t) {
        out_upper[row_off + t] = nanf;
        out_middle[row_off + t] = nanf;
        out_lower[row_off + t] = nanf;
    }
    if (warm >= len) return;

    const float inv_fast = 1.0f / (float)fast_period;

    float sumsq = 0.0f;
    const int start0 = (warm + 1) - fast_period;
    for (int k = 0; k < fast_period; ++k) {
        const int idx = start0 + k;
        const float fm = sma_from_prefix_f32(pref_close_sum, pref_close_nan, idx, fast_period);
        const float sm = sma_from_prefix_f32(pref_close_sum, pref_close_nan, idx, slow_period);
        const float d = fm - sm;
        sumsq = fmaf(d, d, sumsq);
    }

    float dev = sqrtf(sumsq * inv_fast);
    float fm = sma_from_prefix_f32(pref_close_sum, pref_close_nan, warm, fast_period);
    float sm = sma_from_prefix_f32(pref_close_sum, pref_close_nan, warm, slow_period);
    out_middle[row_off + warm] = fm;
    out_upper[row_off + warm] = sm + devup * dev;
    out_lower[row_off + warm] = sm - devdn * dev;

    for (int i = warm + 1; i < len; ++i) {
        const int old_idx = i - fast_period;

        const float fn = sma_from_prefix_f32(pref_close_sum, pref_close_nan, i, fast_period);
        const float sn = sma_from_prefix_f32(pref_close_sum, pref_close_nan, i, slow_period);
        const float fo = sma_from_prefix_f32(pref_close_sum, pref_close_nan, old_idx, fast_period);
        const float so = sma_from_prefix_f32(pref_close_sum, pref_close_nan, old_idx, slow_period);

        const float newd = fn - sn;
        const float oldd = fo - so;
        sumsq = (sumsq + newd * newd) - oldd * oldd;
        if (!isnan(sumsq) && sumsq < 0.0f) sumsq = 0.0f;
        dev = sqrtf(sumsq * inv_fast);

        out_middle[row_off + i] = fn;
        out_upper[row_off + i] = sn + devup * dev;
        out_lower[row_off + i] = sn - devdn * dev;
    }
}



extern "C" __global__ void mab_dev_from_ma_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    int fast_period,
    int first_valid,
    int len,
    float* __restrict__ dev_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0 || fast_period <= 0) return;

    const int first_output = first_valid + max(fast_period, 0) + fast_period - 1;




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



extern "C" __global__ void mab_apply_dev_shared_ma_batch_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    const float* __restrict__ dev,
    int fast_period,
    int slow_period,
    int first_valid,
    int len,
    const float* __restrict__ devups,
    const float* __restrict__ devdns,
    int rows,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower
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


extern "C" __global__ void mab_single_row_from_ma_f32(
    const float* __restrict__ fast,
    const float* __restrict__ slow,
    int fast_period,
    int slow_period,
    int first_valid,
    int len,
    float devup,
    float devdn,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int warm = first_valid + max(fast_period, slow_period) + fast_period - 1;
    const float nanf = qnan32();

    for (int t = 0; t < min(warm, len); ++t) {
        out_upper[t] = nanf;
        out_middle[t] = nanf;
        out_lower[t] = nanf;
    }
    if (warm >= len) return;


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


extern "C" __global__ void mab_many_series_one_param_time_major_f32(
    const float* __restrict__ fast_tm,
    const float* __restrict__ slow_tm,
    const int* __restrict__ first_valids,
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

    if (threadIdx.x != 0 || blockIdx.x != 0) return;
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
