









#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>





static __device__ __forceinline__ float ld_ro(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}


static __device__ __forceinline__ float qNaNf() {
    return __uint_as_float(0x7fc00000u);
}




struct fpair { float hi, lo; };

static __device__ __forceinline__ fpair make_fpair(float x) { return {x, 0.0f}; }

static __device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bp = s - a;
    e = (a - (s - bp)) + (b - bp);
}

static __device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}

static __device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p);
}


static __device__ __forceinline__ fpair ds_madd(float beta, const fpair &x, float alpha, float u) {
    float p1, e1; two_prod(beta, x.hi, p1, e1);
    float p2, e2; two_prod(beta, x.lo, p2, e2);
    float p3, e3; two_prod(alpha,    u, p3, e3);

    float s1, r1; two_sum(p1, p2, s1, r1);
    float s2, r2; two_sum(s1, p3, s2, r2);

    float lo = (e1 + e2 + e3) + (r1 + r2);
    float hi, lo2; quick_two_sum(s2, lo, hi, lo2);
    return {hi, lo2};
}

static __device__ __forceinline__ float ds_val(const fpair &x) {
    return x.hi + x.lo;
}






static __device__ __forceinline__ void qqe_compute_series_f32(
    const float* __restrict__ prices,
    int N,
    int first_valid,
    int rsi_p,
    int ema_p,
    float fast_k,
    float* __restrict__ out_fast,
    float* __restrict__ out_slow)
{
    if (N <= 0) return;
    if (first_valid >= N) return;
    if (rsi_p <= 0 || ema_p <= 0) return;

    const int rsi_start = first_valid + rsi_p;
    if (rsi_start >= N) return;
    const int warm = first_valid + rsi_p + ema_p - 2;


    float avg_gain = 0.0f, avg_loss = 0.0f;
    bool bad = false;
    const int init_end = min(first_valid + rsi_p, N - 1);
    for (int i = first_valid + 1; i <= init_end; ++i) {
        float di   = ld_ro(&prices[i]);
        float dim1 = ld_ro(&prices[i - 1]);
        float delta = di - dim1;
        if (!isfinite(delta)) { bad = true; break; }
        if (delta > 0.0f) avg_gain += delta;
        else if (delta < 0.0f) avg_loss -= delta;
    }
    if (bad) return;

    const float inv_rsi  = 1.0f / (float)rsi_p;
    const float beta_rsi = 1.0f - inv_rsi;

    avg_gain *= inv_rsi;
    avg_loss *= inv_rsi;


    float rsi;
    {
        float denom = avg_gain + avg_loss;
        rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_gain / denom);
    }

    out_fast[rsi_start] = rsi;

    if (warm <= rsi_start) out_slow[rsi_start] = rsi;


    float running_mean = rsi;
    const float ema_alpha = 2.0f / ((float)ema_p + 1.0f);
    const float ema_beta  = 1.0f - ema_alpha;
    float prev_ema = rsi;


    const float atr_alpha = 1.0f / 14.0f;
    const float atr_beta  = 1.0f - atr_alpha;
    float wwma = 0.0f;
    float atrrsi = 0.0f;
    float prev_fast_val = rsi;


#pragma unroll 1
    for (int i = rsi_start + 1; i < N; ++i) {

        float di   = ld_ro(&prices[i]);
        float dim1 = ld_ro(&prices[i - 1]);
        float delta = di - dim1;
        float gain = (delta > 0.0f) ? delta : 0.0f;
        float loss = (delta < 0.0f) ? -delta : 0.0f;

        avg_gain = fmaf(beta_rsi, avg_gain, inv_rsi * gain);
        avg_loss = fmaf(beta_rsi, avg_loss, inv_rsi * loss);

        float denom = avg_gain + avg_loss;
        rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_gain / denom);


        float fast_i;
        if (i < rsi_start + ema_p) {
            float n = (float)(i - rsi_start + 1);
            running_mean = ((n - 1.0f) * running_mean + rsi) / n;
            prev_ema = running_mean;
            fast_i = running_mean;
        } else {
            prev_ema = fmaf(ema_beta, prev_ema, ema_alpha * rsi);
            fast_i = prev_ema;
        }
        out_fast[i] = fast_i;

        if (i == warm) {
            out_slow[i] = fast_i;
            prev_fast_val = fast_i;
        } else if (i > warm) {

            float tr = fabsf(fast_i - prev_fast_val);
            wwma   = fmaf(atr_beta,  wwma,  atr_alpha * tr);
            atrrsi = fmaf(atr_beta, atrrsi, atr_alpha * wwma);
            float qup = fast_i + atrrsi * fast_k;
            float qdn = fast_i - atrrsi * fast_k;

            float prev = out_slow[i - 1];
            float slow;
            if (qup < prev) slow = qup;
            else if (fast_i > prev && prev_fast_val < prev) slow = qdn;
            else if (qdn > prev) slow = qdn;
            else if (fast_i < prev && prev_fast_val > prev) slow = qup;
            else slow = prev;

            out_slow[i] = slow;
            prev_fast_val = fast_i;
        }
    }
}








extern "C" __global__ void qqe_batch_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ rsi_periods,
    const int*   __restrict__ ema_periods,
    const float* __restrict__ fast_factors,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int rsi_p = rsi_periods[combo];
    const int ema_p = ema_periods[combo];
    const float fast_k = fast_factors[combo];
    if (rsi_p <= 0 || ema_p <= 0) return;

    const int row_fast = 2 * combo;
    const int row_slow = row_fast + 1;
    float* __restrict__ out_fast = out + row_fast * series_len;
    float* __restrict__ out_slow = out + row_slow * series_len;


    int warm = first_valid + rsi_p + ema_p - 2;
    if (warm > series_len) warm = series_len;
    const float nanv = qNaNf();
    for (int idx = threadIdx.x; idx < warm; idx += blockDim.x) {
        out_fast[idx] = nanv;
        out_slow[idx] = nanv;
    }
    __syncthreads();


    if (threadIdx.x == 0 && blockIdx.x == 0) {
        qqe_compute_series_f32(prices, series_len, first_valid, rsi_p, ema_p, fast_k, out_fast, out_slow);
    }
}







extern "C" __global__ void qqe_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int rsi_period,
    int ema_period,
    float fast_factor,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y;
    if (s >= num_series) return;
    if (rsi_period <= 0 || ema_period <= 0) return;

    const int fv = first_valids[s];


    int warm = fv + rsi_period + ema_period - 2;
    if (warm > series_len) warm = series_len;
    const int pitch = 2 * num_series;
    const float nanv = qNaNf();
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out_tm[t * pitch + s] = nanv;
        out_tm[t * pitch + (s + num_series)] = nanv;
    }
    __syncthreads();

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const int rsi_start = fv + rsi_period;
    if (rsi_start >= series_len) return;


    float avg_gain_f = 0.0f, avg_loss_f = 0.0f;
    bool bad = false;
    const int init_end = min(fv + rsi_period, series_len - 1);
    for (int i = fv + 1; i <= init_end; ++i) {
        float di   = ld_ro(&prices_tm[i * num_series + s]);
        float dim1 = ld_ro(&prices_tm[(i - 1) * num_series + s]);
        float delta = di - dim1;
        if (!isfinite(delta)) { bad = true; break; }
        if (delta > 0.0f) avg_gain_f += delta; else if (delta < 0.0f) avg_loss_f -= delta;
    }
    if (bad) return;

    const float inv_rsi  = 1.0f / (float)rsi_period;
    const float beta_rsi = 1.0f - inv_rsi;
    avg_gain_f *= inv_rsi; avg_loss_f *= inv_rsi;
    fpair avg_gain = make_fpair(avg_gain_f);
    fpair avg_loss = make_fpair(avg_loss_f);

    float rsi;
    {
        float denom = ds_val(avg_gain) + ds_val(avg_loss);
        rsi = (denom == 0.0f) ? 50.0f : (100.0f * ds_val(avg_gain) / denom);
    }
    out_tm[rsi_start * pitch + s] = rsi;
    if (warm <= rsi_start) out_tm[rsi_start * pitch + (s + num_series)] = rsi;

    float running_mean = rsi;
    const float ema_alpha = 2.0f / ((float)ema_period + 1.0f);
    const float ema_beta  = 1.0f - ema_alpha;
    float prev_ema = rsi;
    const float atr_alpha = 1.0f / 14.0f;
    const float atr_beta  = 1.0f - atr_alpha;
    float wwma = 0.0f, atrrsi = 0.0f;
    float prev_fast_val = rsi;

#pragma unroll 1
    for (int i = rsi_start + 1; i < series_len; ++i) {
        float di   = ld_ro(&prices_tm[i * num_series + s]);
        float dim1 = ld_ro(&prices_tm[(i - 1) * num_series + s]);
        float delta = di - dim1;
        float gain = (delta > 0.0f) ? delta : 0.0f;
        float loss = (delta < 0.0f) ? -delta : 0.0f;

        avg_gain = ds_madd(beta_rsi, avg_gain, inv_rsi, gain);
        avg_loss = ds_madd(beta_rsi, avg_loss, inv_rsi, loss);
        float denom = ds_val(avg_gain) + ds_val(avg_loss);
        rsi = (denom == 0.0f) ? 50.0f : (100.0f * ds_val(avg_gain) / denom);

        float fast_i;
        if (i < rsi_start + ema_period) {
            float n = (float)(i - rsi_start + 1);
            running_mean = ((n - 1.0f) * running_mean + rsi) / n;
            prev_ema = running_mean;
            fast_i = running_mean;
        } else {
            prev_ema = fmaf(ema_beta, prev_ema, ema_alpha * rsi);
            fast_i = prev_ema;
        }
        out_tm[i * pitch + s] = fast_i;

        if (i == warm) {
            out_tm[i * pitch + (s + num_series)] = fast_i;
            prev_fast_val = fast_i;
        } else if (i > warm) {
            float tr = fabsf(fast_i - prev_fast_val);
            wwma   = fmaf(atr_beta,  wwma,  atr_alpha * tr);
            atrrsi = fmaf(atr_beta, atrrsi, atr_alpha * wwma);
            float qup = fast_i + atrrsi * fast_factor;
            float qdn = fast_i - atrrsi * fast_factor;

            float prev = out_tm[(i - 1) * pitch + (s + num_series)];
            float slow;
            if (qup < prev) slow = qup;
            else if (fast_i > prev && prev_fast_val < prev) slow = qdn;
            else if (qdn > prev) slow = qdn;
            else if (fast_i < prev && prev_fast_val > prev) slow = qup;
            else slow = prev;

            out_tm[i * pitch + (s + num_series)] = slow;
            prev_fast_val = fast_i;
        }
    }
}
