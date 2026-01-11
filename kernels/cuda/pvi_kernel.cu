

















#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ bool finite_f(float x) { return isfinite(x); }


extern "C" __global__ void pvi_build_scale_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ scale_out)
{
    if (len <= 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return; 

    const int fv = first_valid < 0 ? 0 : first_valid;
    for (int i = 0; i < len; ++i) scale_out[i] = nanf("");
    if (fv >= len) return;

    scale_out[fv] = 1.0f; 

    
    double prev_close = (double)close[fv];
    double prev_vol = (double)volume[fv];
    double accum = 1.0; 

    for (int i = fv + 1; i < len; ++i) {
        const float cf = close[i];
        const float vf = volume[i];
        if (finite_f(cf) && finite_f(vf) && isfinite(prev_close) && isfinite(prev_vol)) {
            if ((double)vf > prev_vol) {
                const double c = (double)cf;
                const double r = (c - prev_close) / prev_close;
                
                accum += r * accum;
            }
            scale_out[i] = (float)accum;
            prev_close = (double)cf;
            prev_vol = (double)vf;
        } else {
            scale_out[i] = nanf("");
            if (finite_f(cf) && finite_f(vf)) {
                prev_close = (double)cf;
                prev_vol = (double)vf;
            }
        }
    }
}







extern "C" __global__ void pvi_build_scale_warp16_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ scale_out)
{
    if (len <= 0) return;
    if (blockIdx.x != 0) return;

    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 16) return;
    const unsigned mask = 0x0000ffffu;

    const int fv = first_valid < 0 ? 0 : first_valid;
    const float nan_f = nanf("");

    
    for (int i = lane; i < fv && i < len; i += 16) scale_out[i] = nan_f;
    if (fv >= len) return;

    if (lane == 0) scale_out[fv] = 1.0f;
    if (fv + 1 >= len) return;

    double accum0 = 1.0; 

    for (int t0 = fv + 1; t0 < len; t0 += 16) {
        const int i = t0 + lane;
        double f = 1.0;
        if (i < len) {
            const float cf = close[i];
            const float c0 = close[i - 1];
            const float vf = volume[i];
            const float v0 = volume[i - 1];
            if ((double)vf > (double)v0) {
                const double c = (double)cf;
                const double prev = (double)c0;
                const double r = (c - prev) / prev;
                f = 1.0 + r;
            }
        }

        
        double prefix = f;
        for (int offset = 1; offset < 16; offset <<= 1) {
            double other = __shfl_up_sync(mask, prefix, offset, 16);
            if (lane >= offset) prefix *= other;
        }

        const double base = __shfl_sync(mask, accum0, 0, 16);
        if (i < len) scale_out[i] = (float)(base * prefix);

        const double tile_prod = __shfl_sync(mask, prefix, 15, 16);
        if (lane == 0) accum0 *= tile_prod;
    }
}



extern "C" __global__ void pvi_apply_scale_batch_f32(
    const float* __restrict__ scale,
    int len,
    int first_valid,
    const float* __restrict__ initial_values, 
    int rows,
    float* __restrict__ out)                 
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x; 
    const int r = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y; 
    if (t >= len || r >= rows || rows <= 0) return;

    const float nan_f = nanf("");
    const size_t out_idx = (size_t)r * (size_t)len + (size_t)t;

    if (t < first_valid) {
        out[out_idx] = nan_f;
        return;
    }
    const float ivf = initial_values[r];
    if (t == first_valid) {
        out[out_idx] = ivf;
        return;
    }

    const float s = scale[t];
    if (!isfinite(s)) {
        out[out_idx] = nan_f;
        return;
    }

    const double iv = (double)ivf;
    const double sd = (double)s;
    out[out_idx] = (float)(iv * sd);
}


extern "C" __global__ void pvi_apply_batch_direct_f32(
    const float* __restrict__ close,   
    const float* __restrict__ volume,  
    int len,
    int first_valid,
    const float* __restrict__ initial_values, 
    int rows,
    float* __restrict__ out) 
{
    if (rows <= 0 || len <= 0) return;
    const int fv = first_valid < 0 ? 0 : first_valid;
    const float nan_f = nanf("");

    const int stride = blockDim.x * gridDim.x;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += stride) {
        
        for (int t = 0; t < min(fv, len); ++t) out[(size_t)r * len + t] = nan_f;
        if (fv >= len) continue;

        double pvi = (double)initial_values[r];
        out[(size_t)r * len + fv] = (float)pvi;
        if (fv + 1 >= len) continue;

        double prev_close = (double)close[fv];
        double prev_vol   = (double)volume[fv];
        for (int t = fv + 1; t < len; ++t) {
            const float cf = close[t];
            const float vf = volume[t];
            if (isfinite(cf) && isfinite(vf) && isfinite(prev_close) && isfinite(prev_vol)) {
                if ((double)vf > prev_vol) {
                    const double c = (double)cf;
                    
                    pvi *= c / prev_close;
                }
                out[(size_t)r * len + t] = (float)pvi;
                prev_close = (double)cf;
                prev_vol   = (double)vf;
            } else {
                out[(size_t)r * len + t] = nan_f;
                if (isfinite(cf) && isfinite(vf)) {
                    prev_close = (double)cf;
                    prev_vol   = (double)vf;
                }
            }
        }
    }
}


extern "C" __global__ void pvi_many_series_one_param_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids, 
    float initial_value,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; 
    if (s >= cols || rows <= 0) return;

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const float nan_f = nanf("");

    
    for (int t = 0; t < fv && t < rows; ++t) {
        out_tm[t * cols + s] = nan_f;
    }
    if (fv >= rows) return;

    double pvi = (double)initial_value;
    out_tm[fv * cols + s] = (float)pvi;
    if (fv + 1 >= rows) return;

    double prev_close = (double)close_tm[fv * cols + s];
    double prev_vol = (double)volume_tm[fv * cols + s];

    for (int t = fv + 1; t < rows; ++t) {
        const float cf = close_tm[t * cols + s];
        const float vf = volume_tm[t * cols + s];
        if (isfinite(cf) && isfinite(vf) && isfinite(prev_close) && isfinite(prev_vol)) {
            if ((double)vf > prev_vol) {
                const double c = (double)cf;
                const double r = (c - prev_close) / prev_close;
                pvi += r * pvi; 
            }
            out_tm[t * cols + s] = (float)pvi;
            prev_close = (double)cf;
            prev_vol = (double)vf;
        } else {
            out_tm[t * cols + s] = nan_f;
            if (isfinite(cf) && isfinite(vf)) {
                prev_close = (double)cf;
                prev_vol = (double)vf;
            }
        }
    }
}

