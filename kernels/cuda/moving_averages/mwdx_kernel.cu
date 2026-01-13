








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


#if !defined(__CUDACC_VER_MAJOR__)
#define __CUDACC_VER_MAJOR__ 0
#endif
#if __CUDACC_VER_MAJOR__ >= 12
#include <cuda/annotated_ptr>
#endif


static __device__ __forceinline__ float qnan() {

    return __int_as_float(0x7fffffff);
}



static __device__ __forceinline__ void prefetch_L2(const void* p) {
#if __CUDA_ARCH__ >= 800
    asm volatile ("prefetch.global.L2 [%0];" :: "l"(p));
#endif
}


extern "C" __global__
void mwdx_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ facs,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) {
        return;
    }


#if __CUDACC_VER_MAJOR__ >= 12
    const float* __restrict__ prices_persist =
        cuda::associate_access_property(prices, cuda::access_property::persisting{});
#else
    const float* __restrict__ prices_persist = prices;
#endif

    const float fac = facs[combo];
    const float beta = 1.0f - fac;
    const int row_offset = combo * series_len;


    if (first_valid < 0 || first_valid >= series_len) {
        for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
            out[row_offset + idx] = qnan();
        }
        return;
    }


    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[row_offset + idx] = qnan();
    }



    if (threadIdx.x == 0) {
        float prev = prices_persist[first_valid];
        out[row_offset + first_valid] = prev;


        const int PDIST = 64;
        for (int t = first_valid + 1; t < series_len; ++t) {
#if __CUDA_ARCH__ >= 800
            int pf = t + PDIST;
            if (pf < series_len) prefetch_L2(prices_persist + pf);
#endif
            const float price = prices_persist[t];
            prev = __fmaf_rn(price, fac, beta * prev);
            out[row_offset + t] = prev;
        }
    }
}

extern "C" __global__
void mwdx_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    float fac,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) {
        return;
    }

    const float beta = 1.0f - fac;
    const int stride = num_series;
    const int first_valid = first_valids[series_idx];


    if (first_valid < 0 || first_valid >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * stride + series_idx] = qnan();
        }
        return;
    }


    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * stride + series_idx] = qnan();
    }


    if (threadIdx.x == 0) {
        int offset = first_valid * stride + series_idx;
        float prev = prices_tm[offset];
        out_tm[offset] = prev;
        for (int t = first_valid + 1; t < series_len; ++t) {
            offset = t * stride + series_idx;
            const float price = prices_tm[offset];
            prev = __fmaf_rn(price, fac, beta * prev);
            out_tm[offset] = prev;
        }
    }
}




template<int TX, int TY>
__device__ void mwdx_many_series_one_param_tiled2d_f32_core(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    const int s_base = blockIdx.y * TY;
    const int s_local = s_base + threadIdx.y;
    if (s_local >= num_series || series_len <= 0) return;

    const float beta = 1.0f - fac;
    const int stride = num_series;
    const int first_valid = first_valids[s_local];

    if (first_valid < 0 || first_valid >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += TX) {
            out_tm[t * stride + s_local] = qnan();
        }
        return;
    }


    for (int t = threadIdx.x; t < first_valid; t += TX) {
        out_tm[t * stride + s_local] = qnan();
    }

    if (threadIdx.x == 0) {
        int off0 = first_valid * stride + s_local;
        float prev = prices_tm[off0];
        out_tm[off0] = prev;
        for (int t = first_valid + 1; t < series_len; ++t) {
            const int off = t * stride + s_local;
            const float price = prices_tm[off];
            prev = __fmaf_rn(price, fac, beta * prev);
            out_tm[off] = prev;
        }
    }
}

extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f32_tx128_ty2(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f32_core<128, 2>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}

extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f32_tx128_ty4(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f32_core<128, 4>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}
