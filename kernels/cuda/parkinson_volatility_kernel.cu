#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ bool valid_high_low(float high, float low) {
    return isfinite(high) && isfinite(low) && high > 0.0f && low > 0.0f;
}

extern "C" __global__ void parkinson_volatility_build_prefix_f64(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    double* __restrict__ prefix_sum,
    int* __restrict__ prefix_invalid
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    prefix_sum[0] = 0.0;
    prefix_invalid[0] = 0;

    double sum = 0.0;
    int invalid = 0;
    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const float h = high[i];
            const float l = low[i];
            if (valid_high_low(h, l)) {
                const double x = log((double)h / (double)l);
                sum += x * x;
            } else {
                invalid += 1;
            }
        }
        prefix_sum[i + 1] = sum;
        prefix_invalid[i + 1] = invalid;
    }
}

extern "C" __global__ void parkinson_volatility_batch_f32(
    const double* __restrict__ prefix_sum,
    const int* __restrict__ prefix_invalid,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out_volatility,
    float* __restrict__ out_variance
) {
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > len) {
        return;
    }

    const int warmup = first_valid + period - 1;
    const int base = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);
    const double denom = ((double)period) * (4.0 * 0.69314718055994530942);

    for (int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         t < len;
         t += (int)gridDim.x * (int)blockDim.x) {
        float vol_out = nan_f;
        float var_out = nan_f;

        if (t >= warmup) {
            const int end = t + 1;
            const int start = end - period;
            const int invalid = prefix_invalid[end] - prefix_invalid[start];
            if (invalid == 0) {
                double variance = (prefix_sum[end] - prefix_sum[start]) / denom;
                if (variance < 0.0) {
                    variance = 0.0;
                }
                var_out = (float)variance;
                vol_out = sqrtf((float)variance);
            }
        }

        out_volatility[base + t] = vol_out;
        out_variance[base + t] = var_out;
    }
}

extern "C" __global__ void parkinson_volatility_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int* __restrict__ first_valids,
    int period,
    int cols,
    int rows,
    float* __restrict__ out_volatility_tm,
    float* __restrict__ out_variance_tm
) {
    const int s = (int)blockIdx.x;
    if (s >= cols) {
        return;
    }

    const float nan_f = __int_as_float(0x7fffffff);
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        const int idx = t * cols + s;
        out_volatility_tm[idx] = nan_f;
        out_variance_tm[idx] = nan_f;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }
    if (period <= 0 || period > rows) {
        return;
    }

    const int first_valid = first_valids[s];
    if (first_valid < 0 || first_valid >= rows) {
        return;
    }

    const int warmup = first_valid + period - 1;
    if (warmup >= rows) {
        return;
    }

    const double denom = ((double)period) * (4.0 * 0.69314718055994530942);
    double sum = 0.0;
    int invalid = 0;

    for (int t = first_valid; t <= warmup; ++t) {
        const int idx = t * cols + s;
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        if (valid_high_low(h, l)) {
            const double x = log((double)h / (double)l);
            sum += x * x;
        } else {
            invalid += 1;
        }
    }

    if (invalid == 0) {
        double variance = sum / denom;
        if (variance < 0.0) {
            variance = 0.0;
        }
        const int idx = warmup * cols + s;
        out_variance_tm[idx] = (float)variance;
        out_volatility_tm[idx] = sqrtf((float)variance);
    }

    for (int t = warmup + 1; t < rows; ++t) {
        const int old_idx = (t - period) * cols + s;
        const float old_h = high_tm[old_idx];
        const float old_l = low_tm[old_idx];
        if (valid_high_low(old_h, old_l)) {
            const double x = log((double)old_h / (double)old_l);
            sum -= x * x;
        } else {
            invalid -= 1;
        }

        const int idx = t * cols + s;
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        if (valid_high_low(h, l)) {
            const double x = log((double)h / (double)l);
            sum += x * x;
        } else {
            invalid += 1;
        }

        if (invalid == 0) {
            double variance = sum / denom;
            if (variance < 0.0) {
                variance = 0.0;
            }
            out_variance_tm[idx] = (float)variance;
            out_volatility_tm[idx] = sqrtf((float)variance);
        }
    }
}
