








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef TRENDFLEX_NAN
#define TRENDFLEX_NAN (__int_as_float(0x7fffffff))
#endif




#ifndef TRENDFLEX_ASSUME_OUT_PREFILLED
#define TRENDFLEX_ASSUME_OUT_PREFILLED 0
#endif


#ifndef TRENDFLEX_USE_RSQRT_NR
#define TRENDFLEX_USE_RSQRT_NR 0
#endif


static __device__ __forceinline__ float trendflex_round_half(float v) {
    return roundf(v);
}

static __device__ __forceinline__ float inv_sqrt_pos(float x) {
#if TRENDFLEX_USE_RSQRT_NR


    float y = rsqrtf(x);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
#else
    return 1.0f / sqrtf(x);
#endif
}


extern "C" __global__ void trendflex_batch_f32(const float* __restrict__ prices,
                                               const int*   __restrict__ periods,
                                               int series_len,
                                               int n_combos,
                                               int first_valid,
                                               int max_period,
                                               float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];

    if (series_len <= 0 || period <= 0 || period >= series_len) return;
    if (first_valid < 0 || first_valid >= series_len) return;
    if (max_period <= 0 || period > max_period) return;

    const int tail_len = series_len - first_valid;
    if (tail_len < period) return;

    int ss_period = (int)trendflex_round_half(0.5f * (float)period);
    if (ss_period < 1) ss_period = 1;
    if (tail_len < ss_period) return;


    const double PI    = 3.1415926535897932384626433832795;
    const double ROOT2 = 1.4142135623730951;

    const double inv_ss = 1.0 / (double)ss_period;
    const double k      = ROOT2 * PI * inv_ss;
    const double a      = exp(-k);
    const double a_sq   = a * a;
    const double b      = 2.0 * a * cos(k);
    const double c      = 0.5 * (1.0 + a_sq - b);

    const int warm = first_valid + period;
    const int warm_clamped = (warm < series_len) ? warm : series_len;

    const size_t base = (size_t)combo * (size_t)series_len;
    float* __restrict__ row_out = out + base;

#if !TRENDFLEX_ASSUME_OUT_PREFILLED
    for (int i = 0; i < warm_clamped; ++i) {
        row_out[i] = TRENDFLEX_NAN;
    }
#endif
    if (warm >= series_len) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    float* __restrict__ sh = reinterpret_cast<float*>(shraw);
    float* __restrict__ ring = sh + (size_t)threadIdx.x * (size_t)max_period;

    const int fidx = first_valid;


    double prev2 = (double)prices[fidx];
    ring[0] = (float)prev2;
    double rolling_sum = (double)ring[0];

    double prev1 = prev2;
    double prev_price = prev2;
    if (period >= 2) {
        const double p1 = (double)prices[fidx + 1];
        prev1 = p1;
        prev_price = p1;
        ring[1] = (float)p1;
        rolling_sum += (double)ring[1];
    }

    for (int t = 2; t < period; ++t) {
        const double cur_price = (double)prices[fidx + t];
        const double ss = fma(c, (cur_price + prev_price),
                              fma(b, prev1, -a_sq * prev2));
        const float ss_f = (float)ss;
        ring[t] = ss_f;
        rolling_sum += (double)ss_f;
        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }

    const double tp_f   = (double)period;
    const double inv_tp = 1.0 / tp_f;
    double ms_prev = 0.0;

    for (int row = warm; row < series_len; ++row) {
        const double cur_price = (double)prices[row];
        const double ss = fma(c, (cur_price + prev_price),
                              fma(b, prev1, -a_sq * prev2));

        const float ss_f = (float)ss;
        const double ss_q = (double)ss_f;
        const double my_sum  = (tp_f * ss_q - rolling_sum) * inv_tp;
        const double ms_current = fma(0.04, my_sum * my_sum, 0.96 * ms_prev);
        ms_prev = ms_current;

        float out_val = 0.0f;
        if (ms_current > 0.0) {
            out_val = (float)(my_sum / sqrt(ms_current));
        }
        row_out[row] = out_val;

        const int pos = (row - fidx) % period;
        const double ss_old = (double)ring[pos];
        ring[pos] = ss_f;
        rolling_sum += ss_q - ss_old;

        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }
}


extern "C" __global__ void trendflex_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ ssf_tm,
    float* __restrict__ out_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;
    if (series_len <= 0 || period <= 0 || period >= series_len) return;

    const int stride = num_series;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int tail_len = series_len - first_valid;
    if (tail_len < period) return;

    const double PI    = 3.1415926535897932384626433832795;
    const double ROOT2 = 1.4142135623730951;

    int ss_period = (int)trendflex_round_half(0.5f * (float)period);
    if (ss_period < 1) ss_period = 1;
    if (tail_len < ss_period) return;

    const double inv_ss = 1.0 / (double)ss_period;
    const double k = ROOT2 * PI * inv_ss;
    const double a_d    = exp(-k);
    const double a_sq_d = a_d * a_d;
    const double b_d    = 2.0 * a_d * cos(k);
    const double c_d    = 0.5 * (1.0 + a_sq_d - b_d);
    const float  a      = (float)a_d;
    const float  a_sq   = (float)a_sq_d;
    const float  b      = (float)b_d;
    const float  c      = (float)c_d;


    auto at = [stride, series](int row) { return row * stride + series; };

    const int warm = first_valid + period;
#if !TRENDFLEX_ASSUME_OUT_PREFILLED
    const int nan_end = warm < series_len ? warm : series_len;
    for (int row = 0; row < nan_end; ++row) {
        out_tm[at(row)] = TRENDFLEX_NAN;
    }
#endif
    if (warm >= series_len) return;


    const int fidx = first_valid;


    float prev2 = prices_tm[at(fidx)];
    ssf_tm[at(fidx)] = prev2;
    float rolling_sum = prev2;


    float prev1, prev_price;
    if (tail_len > 1) {
        const float p1 = prices_tm[at(fidx + 1)];
        prev1 = p1;
        ssf_tm[at(fidx + 1)] = prev1;
        rolling_sum += prev1;
        prev_price = p1;
    } else {
        return;
    }


    for (int t = 2; t < period; ++t) {
        const float cur_price = prices_tm[at(fidx + t)];
        const float ss = fmaf(c, (cur_price + prev_price),
                              fmaf(b, prev1, -a_sq * prev2));
        ssf_tm[at(fidx + t)] = ss;
        rolling_sum += ss;
        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }


    const float tp_f   = (float)period;
    const float inv_tp = 1.0f / tp_f;
    float ms_prev = 0.0f;

    for (int row = warm; row < series_len; ++row) {
        const float cur_price = prices_tm[at(row)];
        const float ss = fmaf(c, (cur_price + prev_price),
                              fmaf(b, prev1, -a_sq * prev2));

        const float my_sum  = (tp_f * ss - rolling_sum) * inv_tp;
        const float my_sum2 = my_sum * my_sum;
        const float ms_current = fmaf(0.04f, my_sum2, 0.96f * ms_prev);
        ms_prev = ms_current;

        float out_val = 0.0f;
        if (ms_current > 0.0f) {
            out_val = my_sum * inv_sqrt_pos(ms_current);
        }
        out_tm[at(row)] = out_val;

        const float ss_old = ssf_tm[at(row - period)];
        rolling_sum += ss - ss_old;
        ssf_tm[at(row)] = ss;

        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }
}
