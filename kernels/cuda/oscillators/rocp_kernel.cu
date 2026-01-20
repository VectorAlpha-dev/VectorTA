







#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>




#ifndef ROCP_QNAN

__device__ __forceinline__ float rocp_qnan() {
    return __int_as_float(0x7fc00000);
}
#define ROCP_QNAN rocp_qnan()
#endif











extern "C" __global__
void rocp_batch_f32(const float* __restrict__ data,
                    const float* __restrict__ inv,
                    const int* __restrict__ periods,
                    int len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= n_combos) return;
    const int period = periods[row];
    if (period <= 0) return;

    const int base = row * len;

    const int start = first_valid + period;


    const int warm = (start < len) ? start : len;
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = ROCP_QNAN;
    }

    if (start >= len) return;


    int t = start + threadIdx.x;
    const int stride = blockDim.x;


    for (; t + 3*stride < len; t += 4*stride) {
        const float c0  = data[t];
        const float ip0 = inv[t - period];
        out[base + t] = fmaf(c0, ip0, -1.0f);

        const int t1 = t + stride;
        const float c1  = data[t1];
        const float ip1 = inv[t1 - period];
        out[base + t1] = fmaf(c1, ip1, -1.0f);

        const int t2 = t + 2*stride;
        const float c2  = data[t2];
        const float ip2 = inv[t2 - period];
        out[base + t2] = fmaf(c2, ip2, -1.0f);

        const int t3 = t + 3*stride;
        const float c3  = data[t3];
        const float ip3 = inv[t3 - period];
        out[base + t3] = fmaf(c3, ip3, -1.0f);
    }


    for (; t < len; t += stride) {
        const float c  = data[t];
        const float ip = inv[t - period];
        out[base + t] = fmaf(c, ip, -1.0f);
    }
}









extern "C" __global__
void rocp_many_series_one_param_f32(const float* __restrict__ data_tm,
                                    const int* __restrict__ firsts,
                                    int cols,
                                    int rows,
                                    int period,
                                    float* __restrict__ out) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || period <= 0) return;

    const int first = firsts[s];
    if (first >= rows) {

        for (int t = 0; t < rows; ++t) {
            out[t * cols + s] = ROCP_QNAN;
        }
        return;
    }

    const int warm = first + period;

    const int limit = (warm < rows) ? warm : rows;
    for (int t = 0; t < limit; ++t) {
        out[t * cols + s] = ROCP_QNAN;
    }
    if (warm >= rows) return;


    for (int t = warm; t < rows; ++t) {
        const float c = data_tm[t * cols + s];
        const float p = data_tm[(t - period) * cols + s];
        out[t * cols + s] = (c - p) / p;
    }
}

