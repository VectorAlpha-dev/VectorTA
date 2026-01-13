







#ifndef CUDA_COPPOCK_F32_H_
#define CUDA_COPPOCK_F32_H_

#include <cuda.h>
#include <cuda_runtime.h>

#define XNAN __int_as_float(0x7fffffff)



__device__ __forceinline__ bool any_nan3(float a, float b, float c) {

    return __isnanf(a) | __isnanf(b) | __isnanf(c);
}

__device__ __forceinline__ float roc_sum_times100(float c, float inv_s, float inv_l) {


    float inv_sum = inv_s + inv_l;
    return fmaf(c, inv_sum, -2.0f) * 100.0f;
}


__device__ __forceinline__ void comp_add(float x, float &sum, float &comp) {
    float t = sum + x;
    if (fabsf(sum) >= fabsf(x)) comp += (sum - t) + x;
    else                       comp += (x   - t) + sum;
    sum = t;
}

__device__ __forceinline__ void comp_sub(float x, float &sum, float &comp) {

    comp_add(-x, sum, comp);
}




extern "C" __global__ void coppock_batch_f32(
    const float* __restrict__ price,
    const float* __restrict__ inv,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    const int* __restrict__ ma_periods,
    int n_combos,
    float* __restrict__ out
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos) return;

    const int s = shorts[row];
    const int l = longs[row];
    const int m = ma_periods[row];
    const int largest = s > l ? s : l;
    const int warm = first_valid + largest + (m - 1);

    float* row_out = out + (size_t)row * (size_t)len;


    const int pre = warm < len ? warm : len;
    for (int t = 0; t < pre; ++t) row_out[t] = XNAN;
    if (warm >= len) return;


    const float denom_w = 0.5f * (float)m * (float)(m + 1);


    float sum = 0.0f, sum_c = 0.0f;
    float wsum = 0.0f, wsum_c = 0.0f;
    int bad_count = 0;

    int w = 1;
    const int start = warm - m + 1;
    for (int j = start; j <= warm; ++j, ++w) {
        const int js = j - s;
        const int jl = j - l;


        const float c  = price[j];
        const float ps = price[js];
        const float pl = price[jl];

        const bool invalid = any_nan3(c, ps, pl);
        if (invalid) { ++bad_count; continue; }

        const float v = roc_sum_times100(c, inv[js], inv[jl]);


        comp_add(v, sum, sum_c);
        comp_add(v * (float)w, wsum, wsum_c);
    }

    if (bad_count > 0) {
        row_out[warm] = XNAN;
    } else {

        const float sum_eff  = sum + sum_c;
        const float wsum_eff = wsum + wsum_c;
        (void)sum_eff;
        row_out[warm] = wsum_eff / denom_w;
    }


    bool state_valid = (bad_count == 0);


    for (int t = warm + 1; t < len; ++t) {

        const int jn  = t;
        const int jns = jn - s;
        const int jnl = jn - l;
        const float cn  = price[jn];
        const float pns = price[jns];
        const float pnl = price[jnl];
        const bool inv_new = any_nan3(cn, pns, pnl);

        float v_new = 0.0f;
        if (!inv_new) v_new = roc_sum_times100(cn, inv[jns], inv[jnl]);


        const int jo  = t - m;
        const int jos = jo - s;
        const int jol = jo - l;
        const float co  = price[jo];
        const float pos = price[jos];
        const float pol = price[jol];
        const bool inv_old = any_nan3(co, pos, pol);

        float v_old = 0.0f;
        if (!inv_old) v_old = roc_sum_times100(co, inv[jos], inv[jol]);


        bad_count += (int)inv_new - (int)inv_old;

        if (bad_count == 0) {
            if (!state_valid) {

                sum = 0.0f; sum_c = 0.0f;
                wsum = 0.0f; wsum_c = 0.0f;
                int ww = 1;
                const int rst = t - m + 1;
                for (int j = rst; j <= t; ++j, ++ww) {
                    const int js2 = j - s;
                    const int jl2 = j - l;
                    const float c2  = price[j];
                    const float ps2 = price[js2];
                    const float pl2 = price[jl2];
                    (void)ps2; (void)pl2;

                    const float v2 = roc_sum_times100(c2, inv[js2], inv[jl2]);
                    comp_add(v2, sum, sum_c);
                    comp_add(v2 * (float)ww, wsum, wsum_c);
                }
                state_valid = true;
            } else {


                const float sum_prev = sum + sum_c;

                comp_add((float)m * v_new, wsum, wsum_c);
                comp_sub(sum_prev,            wsum, wsum_c);

                comp_add(v_new, sum, sum_c);
                comp_sub(v_old, sum, sum_c);
            }
            const float wsum_eff = wsum + wsum_c;
            row_out[t] = wsum_eff / denom_w;
        } else {
            row_out[t] = XNAN;
            state_valid = false;
        }
    }
}







extern "C" __global__ void coppock_batch_time_parallel_f32(
    const float* __restrict__ price,
    const float* __restrict__ inv,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    const int* __restrict__ ma_periods,
    int n_combos,
    float* __restrict__ out
)
{
    const int row = (int)blockIdx.y;
    if (row >= n_combos) return;

    const int s = shorts[row];
    const int l = longs[row];
    const int m = ma_periods[row];
    if (s <= 0 || l <= 0 || m <= 0 || len <= 0) return;

    const int largest = s > l ? s : l;
    const int warm = first_valid + largest + (m - 1);

    float* row_out = out + (size_t)row * (size_t)len;


    const float denom_w = 0.5f * (float)m * (float)(m + 1);
    const float inv_denom = __fdividef(1.0f, denom_w);

    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    while (t < len) {
        float out_val = XNAN;
        if (t >= warm) {
            const int start = t - m + 1;
            float wsum = 0.0f;
            bool bad = false;


            int w = 1;
            for (int j = start; j <= t; ++j, ++w) {
                const int js = j - s;
                const int jl = j - l;

                const float c  = price[j];
                const float ps = price[js];
                const float pl = price[jl];
                if (any_nan3(c, ps, pl)) { bad = true; break; }

                const float v = roc_sum_times100(c, inv[js], inv[jl]);
                wsum = fmaf(v, (float)w, wsum);
            }

            if (!bad) out_val = wsum * inv_denom;
        }
        row_out[t] = out_val;
        t += stride;
    }
}




extern "C" __global__ void coppock_many_series_one_param_f32(
    const float* __restrict__ price_tm,
    const float* __restrict__ inv_tm,
    const int* __restrict__ first_valids,
    int cols, int rows,
    int short_p, int long_p, int ma_period,
    float* __restrict__ out_tm
)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int first_valid = first_valids[s];
    const int largest = short_p > long_p ? short_p : long_p;
    const int m = ma_period;
    const int warm = first_valid + largest + (m - 1);
    const float denom_w = 0.5f * (float)m * (float)(m + 1);


    const int pre = warm < rows ? warm : rows;
    for (int t = 0; t < pre; ++t) {
        out_tm[(size_t)t * (size_t)cols + s] = XNAN;
    }
    if (warm >= rows) return;


    float sum = 0.0f, sum_c = 0.0f;
    float wsum = 0.0f, wsum_c = 0.0f;
    int bad_count = 0;

    int w = 1;
    const int start = warm - m + 1;
    for (int j = start; j <= warm; ++j, ++w) {
        const int js = j - short_p;
        const int jl = j - long_p;

        const size_t idxj  = (size_t)j  * (size_t)cols + s;
        const size_t idxjs = (size_t)js * (size_t)cols + s;
        const size_t idxjl = (size_t)jl * (size_t)cols + s;

        const float c  = price_tm[idxj];
        const float ps = price_tm[idxjs];
        const float pl = price_tm[idxjl];

        const bool invalid = any_nan3(c, ps, pl);
        if (invalid) { ++bad_count; continue; }

        const float v = roc_sum_times100(c, inv_tm[idxjs], inv_tm[idxjl]);
        comp_add(v, sum, sum_c);
        comp_add(v * (float)w, wsum, wsum_c);
    }

    {
        float* dst = out_tm + (size_t)warm * (size_t)cols + s;
        if (bad_count > 0) *dst = XNAN;
        else               *dst = (wsum + wsum_c) / denom_w;
    }

    bool state_valid = (bad_count == 0);


    for (int t = warm + 1; t < rows; ++t) {

        const int jn = t;
        const int jns = jn - short_p;
        const int jnl = jn - long_p;

        const size_t idxjn  = (size_t)jn  * (size_t)cols + s;
        const size_t idxjns = (size_t)jns * (size_t)cols + s;
        const size_t idxjnl = (size_t)jnl * (size_t)cols + s;

        const float cn  = price_tm[idxjn];
        const float pns = price_tm[idxjns];
        const float pnl = price_tm[idxjnl];
        const bool inv_new = any_nan3(cn, pns, pnl);

        float v_new = 0.0f;
        if (!inv_new) v_new = roc_sum_times100(cn, inv_tm[idxjns], inv_tm[idxjnl]);


        const int jo = t - m;
        const int jos = jo - short_p;
        const int jol = jo - long_p;

        const size_t idxjo  = (size_t)jo  * (size_t)cols + s;
        const size_t idxjos = (size_t)jos * (size_t)cols + s;
        const size_t idxjol = (size_t)jol * (size_t)cols + s;

        const float co  = price_tm[idxjo];
        const float pos = price_tm[idxjos];
        const float pol = price_tm[idxjol];
        const bool inv_old = any_nan3(co, pos, pol);

        float v_old = 0.0f;
        if (!inv_old) v_old = roc_sum_times100(co, inv_tm[idxjos], inv_tm[idxjol]);

        bad_count += (int)inv_new - (int)inv_old;

        float* dst = out_tm + (size_t)t * (size_t)cols + s;
        if (bad_count == 0) {
            if (!state_valid) {

                sum = 0.0f; sum_c = 0.0f;
                wsum = 0.0f; wsum_c = 0.0f;
                int ww = 1;
                const int rst = t - m + 1;
                for (int j = rst; j <= t; ++j, ++ww) {
                    const int js2 = j - short_p;
                    const int jl2 = j - long_p;

                    const size_t idxj2  = (size_t)j   * (size_t)cols + s;
                    const size_t idxjs2 = (size_t)js2 * (size_t)cols + s;
                    const size_t idxjl2 = (size_t)jl2 * (size_t)cols + s;

                    const float c2  = price_tm[idxj2];
                    const float v2  = roc_sum_times100(c2, inv_tm[idxjs2], inv_tm[idxjl2]);
                    comp_add(v2, sum, sum_c);
                    comp_add(v2 * (float)ww, wsum, wsum_c);
                }
                state_valid = true;
            } else {
                const float sum_prev = sum + sum_c;
                comp_add((float)m * v_new, wsum, wsum_c);
                comp_sub(sum_prev,            wsum, wsum_c);
                comp_add(v_new, sum, sum_c);
                comp_sub(v_old, sum, sum_c);
            }
            *dst = (wsum + wsum_c) / denom_w;
        } else {
            *dst = XNAN;
            state_valid = false;
        }
    }
}

#endif
