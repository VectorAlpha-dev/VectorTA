// CUDA kernels for VOSC (Volume Oscillator)
//
// Category: Prefix-sum/rational.
// Given prefix sums P where P[0]=0 and P[t+1]=P[t]+x[t], each output is:
//   long_avg = (P[t+1] - P[t+1-L]) / L
//   short_avg = (P[t+1] - P[t+1-S]) / S
//   VOSC = 100 * (short_avg - long_avg) / long_avg
// Warmup: indices < first_valid + L - 1 are NaN.

#include <cuda_runtime.h>

#ifndef VOSC_NAN
#define VOSC_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// One-series × many-params (plain), grid.y enumerates parameter combos
extern "C" __global__ void vosc_batch_prefix_f32(
    const double* __restrict__ prefix_sum, // length = len + 1
    int len,
    int first_valid,
    const int* __restrict__ short_periods, // [n_combos]
    const int* __restrict__ long_periods,  // [n_combos]
    int n_combos,
    float* __restrict__ out                // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int S = short_periods[combo];
    const int L = long_periods[combo];
    if (UNLIKELY(S <= 0 || L <= 0)) return;

    const int warm = first_valid + L - 1;
    const int row_off = combo * len;
    const double inv_S = 1.0 / static_cast<double>(S);
    const double inv_L = 1.0 / static_cast<double>(L);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        float out_val = VOSC_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            int sS = t1 - S; if (sS < 0) sS = 0;
            int sL = t1 - L; if (sL < 0) sL = 0;
            const double short_sum = prefix_sum[t1] - prefix_sum[sS];
            const double long_sum  = prefix_sum[t1] - prefix_sum[sL];
            const double lavg = long_sum * inv_L;
            const double savg = short_sum * inv_S;
            const double v = 100.0 * (savg - lavg) / lavg;
            out_val = static_cast<float>(v);
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// Many-series × one-param (time-major):
// prefix_tm: (rows+1) x cols, time-major; out_tm: rows x cols, time-major
extern "C" __global__ void vosc_many_series_one_param_f32(
    const double* __restrict__ prefix_tm, // (rows+1) x cols in time-major order
    int short_period,
    int long_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids, // [num_series]
    float* __restrict__ out_tm            // rows x cols, time-major
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (UNLIKELY(short_period <= 0 || long_period <= 0)) return;

    const int warm = first_valids[series] + long_period - 1;
    const int stride = num_series;
    const double inv_S = 1.0 / static_cast<double>(short_period);
    const double inv_L = 1.0 / static_cast<double>(long_period);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = t * stride + series;
        float out_val = VOSC_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            int sS = t1 - short_period; if (sS < 0) sS = 0;
            int sL = t1 - long_period;  if (sL < 0) sL = 0;
            const int p_idx_t  = t1 * stride + series;
            const int p_idx_sS = sS * stride + series;
            const int p_idx_sL = sL * stride + series;
            const double short_sum = prefix_tm[p_idx_t] - prefix_tm[p_idx_sS];
            const double long_sum  = prefix_tm[p_idx_t] - prefix_tm[p_idx_sL];
            const double lavg = long_sum * inv_L;
            const double savg = short_sum * inv_S;
            const double v = 100.0 * (savg - lavg) / lavg;
            out_val = static_cast<float>(v);
        }
        out_tm[out_idx] = out_val;
        t += step;
    }
}

