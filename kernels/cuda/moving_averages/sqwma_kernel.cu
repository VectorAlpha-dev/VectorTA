




#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef OUT_PER_THREAD


#define OUT_PER_THREAD 8
#endif

static __device__ __forceinline__ float sqwma_weight_sum(int period) {

    double p = static_cast<double>(period);
    double sum = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0 - 1.0;
    return static_cast<float>(sum);
}

static __device__ __forceinline__ float sqwma_eval_f32(
    float P_f, float S0, float S1, float S2, float inv_weight_sum)
{

    float P2 = P_f * P_f;
    float acc = fmaf(P2, S0, fmaf(-2.0f * P_f, S1, S2));
    return acc * inv_weight_sum;
}

static __device__ __forceinline__ void sqwma_init_sums_series_major(
    const float* __restrict__ x, int t, int P, float& S0, float& S1, float& S2)
{

    S0 = 0.0f; S1 = 0.0f; S2 = 0.0f;
    const int last = P - 2;
#pragma unroll 4
    for (int r = 0; r <= last; ++r) {
        float xr = x[t - r];
        float rf = static_cast<float>(r);
        S0 = S0 + xr;
        S1 = fmaf(rf, xr, S1);
        S2 = fmaf(rf * rf, xr, S2);
    }
}

static __device__ __forceinline__ void sqwma_advance_sums_series_major(
    const float* __restrict__ x, int t, int P, float& S0, float& S1, float& S2)
{

    const int oldest_idx = t - (P - 2);
    const float x_old = x[oldest_idx];
    const float x_new = x[t + 1];

    float S1_old = S1;

    S0 = (S0 - x_old) + x_new;

    S1 = fmaf(static_cast<float>(-(P - 1)), x_old, S1_old + (S0 - x_new + x_old));

    const float Pm1 = static_cast<float>(P - 1);
    const float S0_old = S0 - x_new + x_old;
    S2 = S2 + 2.0f * S1_old + S0_old - (Pm1 * Pm1) * x_old;
}

extern "C" __global__
void sqwma_batch_f32(const float* __restrict__ prices,
                     const int*   __restrict__ periods,
                     int series_len,
                     int n_combos,
                     int first_valid,
                     float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    const int base_out = combo * series_len;


    if (period <= 1) {
        for (int t = blockIdx.x * blockDim.x + threadIdx.x;
             t < series_len;
             t += gridDim.x * blockDim.x)
        {
            out[base_out + t] = NAN;
        }
        return;
    }

    const int warm = first_valid + period + 1;
    const float inv_ws = 1.0f / sqwma_weight_sum(period);
    const float P_f    = static_cast<float>(period);


    const int warm_cap = warm < series_len ? warm : series_len;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x;
         t < warm_cap;
         t += gridDim.x * blockDim.x)
    {
        out[base_out + t] = NAN;
    }
    if (warm >= series_len) return;



    const int tile_size = blockDim.x * OUT_PER_THREAD;
    for (int tile = warm + blockIdx.x * tile_size;
         tile < series_len;
         tile += gridDim.x * tile_size)
    {
        int t0 = tile + threadIdx.x * OUT_PER_THREAD;
        if (t0 >= series_len) continue;


        float S0, S1, S2;
        sqwma_init_sums_series_major(prices, t0, period, S0, S1, S2);


#pragma unroll
        for (int i = 0; i < OUT_PER_THREAD; ++i) {
            int t = t0 + i;
            if (t >= series_len) break;

            float value = sqwma_eval_f32(P_f, S0, S1, S2, inv_ws);
            out[base_out + t] = value;


            if ((i + 1) < OUT_PER_THREAD && (t + 1) < series_len) {
                sqwma_advance_sums_series_major(prices, t, period, S0, S1, S2);
            }
        }
    }
}

extern "C" __global__
void sqwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series || series_len <= 0) return;

    if (period <= 1) {
        for (int t = blockIdx.x * blockDim.x + threadIdx.x;
             t < series_len;
             t += gridDim.x * blockDim.x)
        {
            out_tm[t * num_series + series_idx] = NAN;
        }
        return;
    }

    const int warm = first_valids[series_idx] + period + 1;
    const float inv_ws = 1.0f / sqwma_weight_sum(period);
    const float P_f    = static_cast<float>(period);


    const int warm_cap = warm < series_len ? warm : series_len;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x;
         t < warm_cap;
         t += gridDim.x * blockDim.x)
    {
        out_tm[t * num_series + series_idx] = NAN;
    }
    if (warm >= series_len) return;


    const int tile_size = blockDim.x * OUT_PER_THREAD;
    for (int tile = warm + blockIdx.x * tile_size;
         tile < series_len;
         tile += gridDim.x * tile_size)
    {
        int t0 = tile + threadIdx.x * OUT_PER_THREAD;
        if (t0 >= series_len) continue;


        auto load_tm = [&](int t) {
            return prices_tm[t * num_series + series_idx];
        };
        auto store_tm = [&](int t, float v) {
            out_tm[t * num_series + series_idx] = v;
        };


        float S0 = 0.f, S1 = 0.f, S2 = 0.f;
        const int last = period - 2;
#pragma unroll 4
        for (int r = 0; r <= last; ++r) {
            float xr = load_tm(t0 - r);
            float rf = static_cast<float>(r);
            S0 = S0 + xr;
            S1 = fmaf(rf, xr, S1);
            S2 = fmaf(rf * rf, xr, S2);
        }

#pragma unroll
        for (int i = 0; i < OUT_PER_THREAD; ++i) {
            int t = t0 + i;
            if (t >= series_len) break;

            float value = sqwma_eval_f32(P_f, S0, S1, S2, inv_ws);
            store_tm(t, value);

            if ((i + 1) < OUT_PER_THREAD && (t + 1) < series_len) {

                const int oldest_t = t - (period - 2);
                const float x_old = load_tm(oldest_t);
                const float x_new = load_tm(t + 1);

                float S1_old = S1;
                const float Pm1 = static_cast<float>(period - 1);
                const float S0_old = S0;


                S0 = (S0 - x_old) + x_new;

                S1 = fmaf(-Pm1, x_old, S1_old + S0_old);

                S2 = S2 + 2.0f * S1_old + S0_old - (Pm1 * Pm1) * x_old;
            }
        }
    }
}
