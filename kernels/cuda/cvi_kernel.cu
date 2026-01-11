












#include <cuda_runtime.h>
#include <math.h>

//////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

//////////////////////////////////////////////////////////////////////////////

extern "C" __global__
void cvi_batch_f32(const float* __restrict__ high,
                   const float* __restrict__ low,
                   const int*   __restrict__ periods,
                   const float* __restrict__ alphas,
                   const int*   __restrict__ warm_indices,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int total_threads = blockDim.x * gridDim.x;
    int combo = blockIdx.x * blockDim.x + threadIdx.x;

    for (; combo < n_combos; combo += total_threads) {
        const int   period = periods[combo];
        const float alpha  = alphas[combo];
        const int   warm   = warm_indices[combo];
        if (period <= 0 || warm >= series_len) continue;

        const int base = combo * series_len;

        
        float y = high[first_valid] - low[first_valid];
        out[base + first_valid] = y;

        
        for (int t = first_valid + 1; t < series_len; ++t) {
            const float r = high[t] - low[t];
            y = __fmaf_rn((r - y), alpha, y);
            out[base + t] = y;
        }

        
        for (int t = series_len - 1; t >= warm; --t) {
            const float curr = out[base + t];
            const float old  = out[base + (t - period)];
            out[base + t] = 100.0f * (curr - old) / old;
        }

        
        for (int t = 0; t < warm; ++t) {
            out[base + t] = NAN;
        }
    }
}


extern "C" __global__
void cvi_batch_from_range_f32(const float* __restrict__ range,
                              const int*   __restrict__ periods,
                              const float* __restrict__ alphas,
                              const int*   __restrict__ warm_indices,
                              int series_len,
                              int first_valid,
                              int n_combos,
                              float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int total_threads = blockDim.x * gridDim.x;
    int combo = blockIdx.x * blockDim.x + threadIdx.x;

    for (; combo < n_combos; combo += total_threads) {
        const int   period = periods[combo];
        const float alpha  = alphas[combo];
        const int   warm   = warm_indices[combo];
        if (period <= 0 || warm >= series_len) continue;

        const int base = combo * series_len;

        
        float y = range[first_valid];
        out[base + first_valid] = y;
        for (int t = first_valid + 1; t < series_len; ++t) {
            const float r = range[t];
            y = __fmaf_rn((r - y), alpha, y);
            out[base + t] = y;
        }

        
        for (int t = series_len - 1; t >= warm; --t) {
            const float curr = out[base + t];
            const float old  = out[base + (t - period)];
            out[base + t] = 100.0f * (curr - old) / old;
        }

        
        for (int t = 0; t < warm; ++t) {
            out[base + t] = NAN;
        }
    }
}



extern "C" __global__
void cvi_many_series_one_param_f32(const float* __restrict__ high_tm,
                                   const float* __restrict__ low_tm,
                                   const int*   __restrict__ first_valids,
                                   int period,
                                   float alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;

    
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < num_series;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s];
        if (fv < 0 || fv >= series_len) {
            continue;
        }

        const int warm = fv + (2 * period - 1);
        if (warm >= series_len) {
            continue;
        }

        
        float y = high_tm[fv * stride + s] - low_tm[fv * stride + s];
        out_tm[fv * stride + s] = y;

        for (int t = fv + 1; t < series_len; ++t) {
            const float r = high_tm[t * stride + s] - low_tm[t * stride + s];
            y = __fmaf_rn((r - y), alpha, y);
            out_tm[t * stride + s] = y;
        }

        
        for (int t = series_len - 1; t >= warm; --t) {
            const float curr = out_tm[t * stride + s];
            const float old  = out_tm[(t - period) * stride + s];
            out_tm[t * stride + s] = 100.0f * (curr - old) / old;
        }

        
        for (int t = 0; t < warm; ++t) {
            out_tm[t * stride + s] = NAN;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

extern "C" __global__
void range_from_high_low_f32(const float* __restrict__ high,
                             const float* __restrict__ low,
                             int series_len,
                             float* __restrict__ range)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x;
         t < series_len;
         t += blockDim.x * gridDim.x)
    {
        range[t] = high[t] - low[t];
    }
}

