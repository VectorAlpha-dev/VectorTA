
















#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }




extern "C" __global__ void dpo_batch_f32(
    const float*  __restrict__ data,
    const float2* __restrict__ prefix_sum_ds,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const int back = period / 2 + 1;
    const int warm = max(first_valid + period - 1, back);
    const int row_off = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nanf = f32_nan();

    const float inv_p = 1.0f / (float)period;

    const float* __restrict__ price_base = data - back;
    while (t < len) {
        float out_val = nanf;
        if (t >= warm) {
            const int wr = t + 1;
            const int wl = wr - period;


            const float2 r = prefix_sum_ds[wr];
            const float2 l = prefix_sum_ds[wl];
            const float sum_hi = r.x - l.x;
            const float sum_lo = r.y - l.y;

            const float price = price_base[t];

            float tmp = fmaf(-inv_p, sum_hi, price);
            out_val    = fmaf(-inv_p, sum_lo, tmp);
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}




extern "C" __global__ void dpo_many_series_one_param_time_major_f32(
    const float*  __restrict__ data_tm,
    const float2* __restrict__ prefix_sum_tm_ds,
    const int*    __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int back = period / 2 + 1;
    const int warm = max(fv + period - 1, back);

    const int stride = gridDim.x * blockDim.x;
    const float nanf = f32_nan();
    const float inv_p = 1.0f / (float)period;

    for (int t = tx; t < rows; t += stride) {
        float out_val = nanf;
        if (t >= warm) {
            const int wr = (t * cols + s) + 1;
            const int wl = (t >= period) ? ((t - period) * cols + s) + 1 : 0;

            const float2 r = prefix_sum_tm_ds[wr];
            const float2 l = prefix_sum_tm_ds[wl];
            const float sum_hi = r.x - l.x;
            const float sum_lo = r.y - l.y;

            const float price = data_tm[(t - back) * cols + s];
            float tmp = fmaf(-inv_p, sum_hi, price);
            out_val    = fmaf(-inv_p, sum_lo, tmp);
        }
        out_tm[t * cols + s] = out_val;
    }
}
