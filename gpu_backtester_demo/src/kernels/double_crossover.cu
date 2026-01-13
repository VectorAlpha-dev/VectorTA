


#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>




static __device__ __forceinline__ int sgn_eps(float f, float s, float eps) {
    return (f > s + eps) - (f < s - eps);
}

static __device__ __forceinline__ float ld_tm(const float* __restrict__ a, int dim, int t, int i) {
#if __CUDA_ARCH__ >= 350
    return __ldg(a + (size_t)t * (size_t)dim + (size_t)i);
#else
    return a[(size_t)t * (size_t)dim + (size_t)i];
#endif
}






extern "C" __global__
void transpose_row_to_tm(const float* __restrict__ in,
                         int rows, int cols,
                         float* __restrict__ out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    for (int k = tid; k < total; k += blockDim.x * gridDim.x) {
        const int r = k / cols;
        const int c = k % cols;
        out[(size_t)c * (size_t)rows + (size_t)r] = in[(size_t)r * (size_t)cols + (size_t)c];
    }
}





extern "C" __global__
void compute_log_returns_f32(const float* __restrict__ prices,
                             int T,
                             float* __restrict__ lr_out)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < T; t += blockDim.x * gridDim.x) {
        if (t == 0) { lr_out[0] = 0.0f; continue; }
        const float p  = prices[t];
        const float pm = prices[t-1];
        lr_out[t] = (p > 0.0f && pm > 0.0f && isfinite(p) && isfinite(pm)) ? logf(p) - logf(pm) : 0.0f;
    }
}




#define STRAT_LONG_ONLY              (1u<<0)
#define STRAT_NO_FLIP                (1u<<1)
#define STRAT_TRADE_ON_NEXT_BAR      (1u<<2)
#define STRAT_ENFORCE_FAST_LT_SLOW   (1u<<3)
#define STRAT_SIGNED_EXPOSURE        (1u<<4)





extern "C" __global__
void double_cross_backtest_tm_flex_f32(
    const float* __restrict__ fast_ma_T,
    const float* __restrict__ slow_ma_T,
    const float* __restrict__ lr,

    int T,
    int Pf_tile, int Ps_tile,
    int Pf_total, int Ps_total,
    int f_offset, int s_offset,

    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int first_valid,

    float commission,
    float eps_rel,
    unsigned int flags,
    int M,
    float* __restrict__ metrics_out
){
    const float log_comm = (commission > 0.0f) ? log1pf(-commission) : 0.0f;

    const int total_pairs = Pf_tile * Ps_tile;
    for (int pair_local = blockIdx.x * blockDim.x + threadIdx.x;
         pair_local < total_pairs;
         pair_local += blockDim.x * gridDim.x)
    {
        const int pf_local  = pair_local / Ps_tile;
        const int ps_local  = pair_local % Ps_tile;
        const int pf_global = f_offset + pf_local;
        const int ps_global = s_offset + ps_local;

        const int period_f = fast_periods[pf_global];
        const int period_s = slow_periods[ps_global];

        const size_t pair_global = (size_t)pf_global * (size_t)Ps_total + (size_t)ps_global;
        const size_t base = pair_global * (size_t)M;

        if ((flags & STRAT_ENFORCE_FAST_LT_SLOW) && (period_f >= period_s)) {
            if (M > 0) metrics_out[base + 0] = 0.0f;
            if (M > 1) metrics_out[base + 1] = 0.0f;
            if (M > 2) metrics_out[base + 2] = 0.0f;
            if (M > 3) metrics_out[base + 3] = 0.0f;
            if (M > 4) metrics_out[base + 4] = 0.0f;
            if (M > 5) metrics_out[base + 5] = 0.0f;
            if (M > 6) metrics_out[base + 6] = 0.0f;
            continue;
        }

        const int t_valid = first_valid + max(period_f, period_s) - 1;
        const int start_t = t_valid + ((flags & STRAT_TRADE_ON_NEXT_BAR) ? 1 : 0);

        if (start_t >= T) {
            if (M > 0) metrics_out[base + 0] = 0.0f;
            if (M > 1) metrics_out[base + 1] = 0.0f;
            if (M > 2) metrics_out[base + 2] = 0.0f;
            if (M > 3) metrics_out[base + 3] = 0.0f;
            if (M > 4) metrics_out[base + 4] = 0.0f;
            if (M > 5) metrics_out[base + 5] = 0.0f;
            if (M > 6) metrics_out[base + 6] = 0.0f;
            continue;
        }

        int   pos         = 0;
        float log_eq      = 0.0f;
        float log_peak    = 0.0f;
        float max_dd      = 0.0f;
        int   trades      = 0;
        long long sum_abs_pos = 0;
        long long sum_pos     = 0;

        float mean = 0.0f, m2 = 0.0f;
        long long n = 0;

        for (int t = start_t; t < T; ++t) {
            const int t_sig = t - ((flags & STRAT_TRADE_ON_NEXT_BAR) ? 1 : 0);

            const float f = ld_tm(fast_ma_T, Pf_tile, t_sig, pf_local);
            const float s = ld_tm(slow_ma_T, Ps_tile, t_sig, ps_local);
            const float eps = (eps_rel > 0.0f) ? (eps_rel * fmaxf(1.0f, fabsf(s))) : 0.0f;

            int sign = sgn_eps(f, s, eps);
            if (flags & STRAT_LONG_ONLY) sign = max(sign, 0);

            if (sign != pos) {
                if (pos == 0 || sign == 0) {
                    if (commission > 0.0f) log_eq += log_comm;
                    trades += 1;
                    pos = (sign == 0) ? 0 : sign;
                } else {
                    if (flags & STRAT_NO_FLIP) {
                        if (commission > 0.0f) log_eq += log_comm;
                        trades += 1;
                        pos = 0;
                    } else {
                        if (commission > 0.0f) log_eq += 2.0f * log_comm;
                        trades += 2;
                        pos = sign;
                    }
                }
            }

            const float lr_t = lr[t];
            log_eq += (float)pos * lr_t;

            if (log_eq > log_peak) log_peak = log_eq;
            const float dd = 1.0f - expf(log_eq - log_peak);
            if (dd > max_dd) max_dd = dd;

            const float step_r = (pos == 0) ? 0.0f : (float)pos * expm1f(lr_t);
            n += 1;
            const float delta = step_r - mean;
            mean += delta / (float)n;
            m2   += delta * (step_r - mean);

            sum_abs_pos += llabs((long long)pos);
            sum_pos     += (long long)pos;
        }

        const float variance = (n > 1) ? (m2 / (float)(n - 1)) : 0.0f;
        const float exposure = (n > 0) ? (float)((double)sum_abs_pos / (double)n) : 0.0f;
        const float net_expo = (n > 0) ? (float)((double)sum_pos / (double)n)     : 0.0f;

        if (M > 0) metrics_out[base + 0] = expf(log_eq) - 1.0f;
        if (M > 1) metrics_out[base + 1] = (float)trades;
        if (M > 2) metrics_out[base + 2] = max_dd;
        if (M > 3) metrics_out[base + 3] = mean;
        if (M > 4) metrics_out[base + 4] = sqrtf(variance);
        if (M > 5) metrics_out[base + 5] = exposure;
        if (M > 6) metrics_out[base + 6] = (flags & STRAT_SIGNED_EXPOSURE) ? net_expo : 0.0f;
    }
}
