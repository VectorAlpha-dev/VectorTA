











#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }





extern "C" __global__ void cfo_batch_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_weighted,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ scalars,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float scalar = scalars[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;


    const double n = (double)period;
    const double sx = (double)(period * (period + 1)) * 0.5;
    const double sx2 = (double)(period * (period + 1) * (2 * period + 1)) / 6.0;
    const double inv_denom = 1.0 / (n * sx2 - sx * sx);
    const double half_nm1 = 0.5 * (n - 1.0);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const float nanf = f32_nan();
    while (t < len) {
        float out_val = nanf;
        if (t >= warm) {

            const int idx = t - first_valid;
            const int r1 = idx + 1;
            const int l1_minus1 = r1 - period;


            const double sum_y = prefix_sum[first_valid + r1] - prefix_sum[first_valid + l1_minus1];
            const double sum_xy_raw = prefix_weighted[first_valid + r1] - prefix_weighted[first_valid + l1_minus1];
            const double sum_xy = sum_xy_raw - ((double)l1_minus1) * sum_y;

            const double b = (-sx) * sum_y + n * sum_xy;
            const double b_scaled = b * inv_denom;
            const double f = b_scaled * half_nm1 + sum_y / n;
            const float cur = data[t];
            if (!isnan(cur) && cur != 0.0f) {

                out_val = scalar * (1.0f - (float)(f / (double)cur));
            } else {
                out_val = nanf;
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}





extern "C" __global__ void cfo_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,
    const double* __restrict__ prefix_sum_tm,
    const double* __restrict__ prefix_weighted_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float scalar,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int warm = fv + period - 1;


    const double n = (double)period;
    const double sx = (double)(period * (period + 1)) * 0.5;
    const double sx2 = (double)(period * (period + 1) * (2 * period + 1)) / 6.0;
    const double inv_denom = 1.0 / (n * sx2 - sx * sx);
    const double half_nm1 = 0.5 * (n - 1.0);



    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float nanf = f32_nan();

        int t = 0;
        for (; t < fv && t < rows; ++t) {
            out_tm[t * cols + s] = nanf;
        }
        if (t >= rows) return;


        double sum_y = 0.0;
        double sum_xy = 0.0;
        int warm_needed = period - 1;
        int k = 0;
        for (; k < warm_needed && t < rows; ++k, ++t) {
            float v = data_tm[t * cols + s];
            double vd = (double)v;
            double w = (double)(k + 1);
            sum_y += vd;
            sum_xy += vd * w;
            out_tm[t * cols + s] = nanf;
        }
        if (t >= rows) return;


        {
            float v = data_tm[t * cols + s];
            double vd = (double)v;
            sum_y += vd;
            sum_xy += vd * n;
            double b = (-sx) * sum_y + n * sum_xy;
            double f = (b * inv_denom) * half_nm1 + sum_y / n;
            out_tm[t * cols + s] = (!isnan(v) && v != 0.0f)
                ? (float)(scalar * (1.0 - f / (double)v))
                : nanf;
            ++t;
        }


        for (; t < rows; ++t) {
            float v_new = data_tm[t * cols + s];
            float v_old = data_tm[(t - period) * cols + s];
            double vd_new = (double)v_new;
            double vd_old = (double)v_old;
            double new_sum_xy = (n * vd_new) + (sum_xy - sum_y);
            double new_sum_y = sum_y - vd_old + vd_new;
            sum_xy = new_sum_xy;
            sum_y = new_sum_y;
            double b = (-sx) * sum_y + n * sum_xy;
            double f = (b * inv_denom) * half_nm1 + sum_y / n;
            out_tm[t * cols + s] = (!isnan(v_new) && v_new != 0.0f)
                ? (float)(scalar * (1.0 - f / (double)v_new))
                : nanf;
        }
    }
}
