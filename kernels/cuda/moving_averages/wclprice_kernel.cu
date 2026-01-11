




#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


extern "C" __global__ void wclprice_batch_f32(const float* __restrict__ high,
                                              const float* __restrict__ low,
                                              const float* __restrict__ close,
                                              int series_len,
                                              int first_valid,
                                              float* __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    if (series_len <= 0) return;
    const int fv = first_valid < 0 ? 0 : first_valid;
    for (int i = tid; i < series_len; i += stride) {
        if (i < fv) {
            out[i] = CUDART_NAN_F; continue;
        }
        const float h = high[i];
        const float l = low[i];
        const float c = close[i];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            out[i] = CUDART_NAN_F; continue;
        }
        
        out[i] = c * 0.5f + (h + l) * 0.25f;
    }
}



extern "C" __global__ void wclprice_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x; 
    if (s >= cols || cols <= 0 || rows <= 0) return;
    const int fv = max(0, first_valids[s]);
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    for (int t0 = tid; t0 < rows; t0 += stride) {
        const int idx = t0 * cols + s;
        if (t0 < fv) { out_tm[idx] = CUDART_NAN_F; continue; }
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            out_tm[idx] = CUDART_NAN_F; continue;
        }
        out_tm[idx] = c * 0.5f + (h + l) * 0.25f;
    }
}

