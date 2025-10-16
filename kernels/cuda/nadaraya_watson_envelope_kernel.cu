// CUDA kernels for Nadaraya–Watson Envelope (Gaussian kernel regression + MAE bands).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LDG
#  if __CUDA_ARCH__ >= 350
#    define LDG(p) __ldg(p)
#  else
#    define LDG(p) (*(p))
#  endif
#endif

__device__ __forceinline__ float qnan_f32() { return __int_as_float(0x7fffffff); }

// One-series × many-params. Each combo (row) is processed sequentially by thread 0 of its block.
// Weights are pre-scaled by 1/den on host. Lookbacks vary per row; weights are laid out with stride = max_lookback.
extern "C" __global__
void nadaraya_watson_envelope_batch_f32(const float* __restrict__ data,
                                        const float* __restrict__ weights_flat,
                                        const int*   __restrict__ lookbacks,
                                        const float* __restrict__ multipliers,
                                        int series_len,
                                        int n_combos,
                                        int first_valid,
                                        int max_lookback,
                                        float* __restrict__ out_upper,
                                        float* __restrict__ out_lower)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    // Single-thread per combo for sequential MAE window maintenance.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int L = lookbacks[combo];
    const float mult = multipliers[combo];
    const int warm_out = first_valid + L - 1;
    const int MAE_LEN = 499;
    const int warm_total = warm_out + MAE_LEN - 1;

    const int base = combo * series_len;
    const int wbase = combo * max_lookback;

    // Fill warmup with NaNs
    const int prefix = min(warm_total, series_len);
    for (int t = 0; t < prefix; ++t) {
        out_upper[base + t] = qnan_f32();
        out_lower[base + t] = qnan_f32();
    }
    if (warm_total >= series_len) return;

    // Rolling MAE state
    float ring[499];
    #pragma unroll
    for (int i = 0; i < MAE_LEN; ++i) ring[i] = qnan_f32();
    int head = 0;
    int filled = 0;
    float sum = 0.0f;
    int nan_count = 0;

    // Iterate forward in time
    for (int t = warm_out; t < series_len; ++t) {
        // Dot-product regression at endpoint t over L samples
        bool any_nan = false;
        double acc = 0.0;
        #pragma unroll 1
        for (int k = 0; k < L; ++k) {
            const float x = LDG(&data[t - k]);
            if (isnan(x)) { any_nan = true; break; }
            const float wk = LDG(&weights_flat[wbase + k]);
            acc += (double)x * (double)wk; // weights pre-scaled by inv_den
        }
        float y = any_nan ? qnan_f32() : (float)acc;

        // Residual for MAE
        float resid = (!isnan(LDG(&data[t])) && !isnan(y)) ? fabsf(LDG(&data[t]) - y) : qnan_f32();

        // Pop old if full
        if (filled == MAE_LEN) {
            float old = ring[head];
            if (isnan(old)) {
                if (nan_count > 0) nan_count -= 1;
            } else {
                sum -= old;
            }
        } else {
            filled += 1;
        }

        // Push new
        ring[head] = resid;
        if (isnan(resid)) nan_count += 1; else sum += resid;
        head += 1; if (head == MAE_LEN) head = 0;

        if (t >= warm_total) {
            float upper = qnan_f32();
            float lower = qnan_f32();
            if (nan_count == 0 && !isnan(y)) {
                float mae = (sum / (float)MAE_LEN) * mult;
                upper = y + mae;
                lower = y - mae;
            }
            out_upper[base + t] = upper;
            out_lower[base + t] = lower;
        }
    }
}

// Many-series × one-param, time-major layout (cols=series_len, rows=num_series).
extern "C" __global__
void nadaraya_watson_envelope_many_series_one_param_f32(const float* __restrict__ data_tm,
                                                        const float* __restrict__ weights,
                                                        int lookback,
                                                        float multiplier,
                                                        int num_series,
                                                        int series_len,
                                                        const int* __restrict__ first_valids,
                                                        float* __restrict__ out_upper_tm,
                                                        float* __restrict__ out_lower_tm)
{
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int L = lookback;
    const int warm_out = first_valids[series] + L - 1;
    const int MAE_LEN = 499;
    const int warm_total = warm_out + MAE_LEN - 1;

    // Warm prefix
    for (int t = 0; t < min(warm_total, series_len); ++t) {
        const int idx = t * num_series + series;
        out_upper_tm[idx] = qnan_f32();
        out_lower_tm[idx] = qnan_f32();
    }
    if (warm_total >= series_len) return;

    float ring[499];
    #pragma unroll
    for (int i = 0; i < MAE_LEN; ++i) ring[i] = qnan_f32();
    int head = 0;
    int filled = 0;
    float sum = 0.0f;
    int nan_count = 0;

    for (int t = warm_out; t < series_len; ++t) {
        bool any_nan = false;
        double acc = 0.0;
        #pragma unroll 1
        for (int k = 0; k < L; ++k) {
            int idx = (t - k) * num_series + series;
            float x = LDG(&data_tm[idx]);
            if (isnan(x)) { any_nan = true; break; }
            float wk = LDG(&weights[k]);
            acc += (double)x * (double)wk;
        }
        float y = any_nan ? qnan_f32() : (float)acc;

        const int idx_t = t * num_series + series;
        float x_t = LDG(&data_tm[idx_t]);
        float resid = (!isnan(x_t) && !isnan(y)) ? fabsf(x_t - y) : qnan_f32();

        if (filled == MAE_LEN) {
            float old = ring[head];
            if (isnan(old)) { if (nan_count > 0) nan_count -= 1; } else { sum -= old; }
        } else {
            filled += 1;
        }
        ring[head] = resid;
        if (isnan(resid)) nan_count += 1; else sum += resid;
        head += 1; if (head == MAE_LEN) head = 0;

        if (t >= warm_total) {
            float upper = qnan_f32();
            float lower = qnan_f32();
            if (nan_count == 0 && !isnan(y)) {
                float mae = (sum / (float)MAE_LEN) * multiplier;
                upper = y + mae;
                lower = y - mae;
            }
            out_upper_tm[idx_t] = upper;
            out_lower_tm[idx_t] = lower;
        }
    }
}

