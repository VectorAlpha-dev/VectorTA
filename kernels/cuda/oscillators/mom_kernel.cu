






#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void mom_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base   = combo * series_len;
    const int period = periods[combo];


    if (series_len <= 0 || first_valid >= series_len || period <= 0) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }

    int warm = first_valid + period;
    if (warm >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }


    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }



    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const float cur  = __ldg(&prices[t]);
        const float prev = __ldg(&prices[t - period]);
        out[base + t] = cur - prev;
    }
}



extern "C" __global__
void mom_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm)
{
    if (cols <= 0 || rows <= 0) return;
    if (period <= 0) return;


    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv   = first_valids[s];
        const int warm = fv + period;


        if (fv < 0 || fv >= rows || warm >= rows) {
            for (int t = 0; t < rows; ++t) {
                out_tm[t * cols + s] = NAN;
            }
            continue;
        }


        for (int t = 0; t < warm; ++t) {
            out_tm[t * cols + s] = NAN;
        }


        for (int t = warm; t < rows; ++t) {
            const int idx = t * cols + s;
            out_tm[idx] = __ldg(&prices_tm[idx]) - __ldg(&prices_tm[(t - period) * cols + s]);
        }
    }
}
