







#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {


__global__ void kaufmanstop_axpy_row_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ ma_row,
    int len,
    float signed_mult,
    int warm,
    int base_is_low,
    float* __restrict__ out_row
) {
    const float* __restrict__ base = base_is_low ? low : high;


    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        float out;
        if (i < warm) {
            out = CUDART_NAN_F;
        } else {

            out = fmaf(ma_row[i], signed_mult, base[i]);
        }
        out_row[i] = out;
    }
}


__global__ void kaufmanstop_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ ma_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    float signed_mult,
    int base_is_low,
    int period,
    float* __restrict__ out_tm
){
    const float* __restrict__ base_tm = base_is_low ? low_tm : high_tm;




    if (gridDim.y == 1 && blockDim.y == 1) {

        const int total = rows * cols;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
            const int s = i % cols;
            const int t = i / cols;
            const int warm = first_valids[s] + period - 1;
            float out;
            if (t < warm) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_tm[i], signed_mult, base_tm[i]);
            }
            out_tm[i] = out;
        }
    } else {

        int s = blockIdx.x * blockDim.x + threadIdx.x;
        int t0 = blockIdx.y * blockDim.y + threadIdx.y;
        int t_stride = blockDim.y * gridDim.y;

        if (s >= cols) return;
        const int warm = first_valids[s] + period - 1;

        for (int t = t0; t < rows; t += t_stride) {
            const int idx = t * cols + s;
            float out;
            if (t < warm) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_tm[idx], signed_mult, base_tm[idx]);
            }
            out_tm[idx] = out;
        }
    }
}




__global__ void kaufmanstop_one_series_many_params_time_major_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ ma_pm,
    const int*   __restrict__ warm_ps,
    const float* __restrict__ signed_mults,
    int rows,
    int params,
    int base_is_low,
    float* __restrict__ out_pm
) {
    extern __shared__ float s_base[];
    const float* __restrict__ base = base_is_low ? low : high;


    int p  = blockIdx.y * blockDim.y + threadIdx.y;
    int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    int t_stride = blockDim.x * gridDim.x;

    for (int t = t0; t < rows; t += t_stride) {

        if (threadIdx.y == 0) {
            s_base[threadIdx.x] = base[t];
        }
        __syncthreads();

        if (p < params) {
            const int idx = p * rows + t;
            float out;
            if (t < warm_ps[p]) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_pm[idx], signed_mults[p], s_base[threadIdx.x]);
            }
            out_pm[idx] = out;
        }
        __syncthreads();
    }
}

}

