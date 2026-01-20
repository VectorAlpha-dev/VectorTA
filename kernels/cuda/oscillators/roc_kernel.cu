







#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#if __CUDACC_VER_MAJOR__ >= 12

#endif


__device__ __forceinline__ float qnanf() { return nanf(""); }





extern "C" __global__
void roc_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;


    float* __restrict__ out_row = out + combo * series_len;


    const int period = periods[combo];
    if (period <= 0) {

        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_row[t] = qnanf();
        }
        return;
    }


    const int warm = first_valid + period;


    if (warm >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_row[t] = qnanf();
        }
        return;
    }




    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out_row[t] = qnanf();
    }






    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {

        float cur  =
#if __CUDA_ARCH__ >= 350
            __ldg(&prices[t]);
#else
            prices[t];
#endif
        float prev =
#if __CUDA_ARCH__ >= 350
            __ldg(&prices[t - period]);
#else
            prices[t - period];
#endif



        if (prev == 0.0f || isnan(prev)) {
            out_row[t] = 0.0f;
        } else {


            const float inv_prev = 1.0f / prev;
            const float rel      = fmaf(cur, inv_prev, -1.0f);
            out_row[t] = 100.0f * rel;
        }
    }
}





extern "C" __global__
void roc_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    if (period <= 0) {

        for (int t = 0; t < rows; ++t) {
            out_tm[t * cols + s] = qnanf();
        }
        return;
    }

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) {
            out_tm[t * cols + s] = qnanf();
        }
        return;
    }

    const int warm = fv + period;


    for (int t = 0; t < warm && t < rows; ++t) {
        out_tm[t * cols + s] = qnanf();
    }


    for (int t = max(0, warm); t < rows; ++t) {
        const int idx  = t * cols + s;
        const float cur =
#if __CUDA_ARCH__ >= 350
            __ldg(&prices_tm[idx]);
#else
            prices_tm[idx];
#endif
        const float prev =
#if __CUDA_ARCH__ >= 350
            __ldg(&prices_tm[(t - period) * cols + s]);
#else
            prices_tm[(t - period) * cols + s];
#endif

        if (prev == 0.0f || isnan(prev)) {
            out_tm[idx] = 0.0f;
        } else {
            const float inv_prev = 1.0f / prev;
            const float rel      = fmaf(cur, inv_prev, -1.0f);
            out_tm[idx] = 100.0f * rel;
        }
    }
}

