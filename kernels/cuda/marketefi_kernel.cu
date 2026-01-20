








#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>



__device__ __forceinline__ float mfi_elem(float h, float l, float v, bool ok) {
    if (!ok) return CUDART_NAN_F;
    if (isnan(h) || isnan(l) || isnan(v) || v == 0.0f) return CUDART_NAN_F;
    return (h - l) / v;
}


extern "C" __global__ void marketefi_kernel_f32(const float* __restrict__ high,
                                                 const float* __restrict__ low,
                                                 const float* __restrict__ volume,
                                                 int len,
                                                 int first_valid,
                                                 float* __restrict__ out) {
    if (len <= 0) return;

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int first  = first_valid < 0 ? 0 : first_valid;


    constexpr int ILP = 4;

    for (int base = tid; base < len; base += stride * ILP) {
#pragma unroll
        for (int k = 0; k < ILP; ++k) {
            int i = base + k * blockDim.x;
            if (i < len) {
                const bool ok = (i >= first);
                const float h = high[i];
                const float l = low[i];
                const float v = volume[i];
                out[i] = mfi_elem(h, l, v, ok);
            }
        }
    }
}







#ifndef MKT_T_TILE
#define MKT_T_TILE 128
#endif

extern "C" __global__ void marketefi_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ volume_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {

    if (num_series <= 0 || series_len <= 0) return;


    const bool legacy_1d = (gridDim.y == 1) && (gridDim.x == num_series);
    if (legacy_1d) {

        const int s = blockIdx.x;
        if (s >= num_series) return;
        const int first = first_valids ? (first_valids[s] < 0 ? 0 : first_valids[s]) : 0;
        const int stride_series = num_series;


        for (int t = threadIdx.x; t < min(first, series_len); t += blockDim.x) {
            out_tm[t * stride_series + s] = CUDART_NAN_F;
        }

        for (int t = threadIdx.x + first; t < series_len; t += blockDim.x) {
            const int idx = t * stride_series + s;
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            const float v = volume_tm[idx];
            out_tm[idx] = mfi_elem(h, l, v, true);
        }
        return;
    }


    const uintptr_t mask16 = 0xF;
    const bool aligned16 =
        (((uintptr_t)high_tm   | (uintptr_t)low_tm |
          (uintptr_t)volume_tm | (uintptr_t)out_tm |
          (uintptr_t)first_valids) & mask16) == 0;
    const bool vec_ok = aligned16 && ((num_series & 3) == 0);

    if (vec_ok) {

        const int series4 = num_series >> 2;
        const int s4 = blockIdx.y * blockDim.x + threadIdx.x;
        if (s4 >= series4) return;


        int4 fv4 = make_int4(0, 0, 0, 0);
        if (first_valids) {
            const int4* __restrict__ fv_ptr = reinterpret_cast<const int4*>(first_valids);
            fv4 = fv_ptr[s4];
            fv4.x = fv4.x < 0 ? 0 : fv4.x;
            fv4.y = fv4.y < 0 ? 0 : fv4.y;
            fv4.z = fv4.z < 0 ? 0 : fv4.z;
            fv4.w = fv4.w < 0 ? 0 : fv4.w;
        }

        const float4* __restrict__ H = reinterpret_cast<const float4*>(high_tm);
        const float4* __restrict__ L = reinterpret_cast<const float4*>(low_tm);
        const float4* __restrict__ V = reinterpret_cast<const float4*>(volume_tm);
        float4* __restrict__ O       = reinterpret_cast<float4*>(out_tm);

        const int stride4_t = series4;

        for (int t0 = blockIdx.x * MKT_T_TILE; t0 < series_len; t0 += gridDim.x * MKT_T_TILE) {
            const int t_end = min(series_len, t0 + MKT_T_TILE);

#pragma unroll 4
            for (int t = t0; t < t_end; ++t) {
                const int idx4 = t * stride4_t + s4;

                const float4 h4 = H[idx4];
                const float4 l4 = L[idx4];
                const float4 v4 = V[idx4];

                float4 out4;
                out4.x = mfi_elem(h4.x, l4.x, v4.x, t >= fv4.x);
                out4.y = mfi_elem(h4.y, l4.y, v4.y, t >= fv4.y);
                out4.z = mfi_elem(h4.z, l4.z, v4.z, t >= fv4.z);
                out4.w = mfi_elem(h4.w, l4.w, v4.w, t >= fv4.w);

                O[idx4] = out4;
            }
        }
    } else {

        const int s = blockIdx.y * blockDim.x + threadIdx.x;
        if (s >= num_series) return;

        const int first = first_valids ? (first_valids[s] < 0 ? 0 : first_valids[s]) : 0;
        const int stride_series = num_series;

        for (int t0 = blockIdx.x * MKT_T_TILE; t0 < series_len; t0 += gridDim.x * MKT_T_TILE) {
            const int t_end = min(series_len, t0 + MKT_T_TILE);

#pragma unroll 4
            for (int t = t0; t < t_end; ++t) {
                const int idx = t * stride_series + s;
                const float h = high_tm[idx];
                const float l = low_tm[idx];
                const float v = volume_tm[idx];
                out_tm[idx] = mfi_elem(h, l, v, t >= first);
            }
        }
    }
}

