









#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h> 



static __device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;   
    sum = t;
}




extern "C" __global__ void vpt_batch_f32(
    const float* __restrict__ price,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{
    
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0) return;

    const float nan_f = CUDART_NAN_F;

    
    if (first_valid < 0) first_valid = 0;

    
    const int warm_end = (first_valid < len) ? first_valid : (len - 1);
    for (int i = 0; i <= warm_end; ++i) out[i] = nan_f;

    
    if (first_valid + 1 >= len) return;

    
    if (first_valid < 1) {
        for (int t = first_valid + 1; t < len; ++t) out[t] = nan_f;
        return;
    }

    
    float p0 = price[first_valid - 1];
    float p1 = price[first_valid];
    float v1 = volume[first_valid];

    
    bool ok = isfinite(p0) && isfinite(p1) && isfinite(v1) && (p0 != 0.0f);
    if (!ok) {
        for (int t = first_valid + 1; t < len; ++t) out[t] = nan_f;
        return;
    }

    float prev_p = p1;

    
    float sum = v1 * ((p1 - p0) / p0);
    float c = 0.0f;

    
    for (int t = first_valid + 1; t < len; ++t) {
        float pt = price[t];
        float vt = volume[t];

        bool good = isfinite(prev_p) && isfinite(pt) && isfinite(vt) && (prev_p != 0.0f);
        if (!good) {
            
            for (int j = t; j < len; ++j) out[j] = nan_f;
            return;
        }

        float cur = vt * ((pt - prev_p) / prev_p);
        kahan_add(cur, sum, c);
        out[t] = sum;

        prev_p = pt;
    }
}





extern "C" __global__ void vpt_many_series_one_param_f32(
    const float* __restrict__ price_tm,   
    const float* __restrict__ volume_tm,  
    int cols,
    int rows,
    const int* __restrict__ first_valids, 
    float* __restrict__ out_tm)           
{
    
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const float nan_f = CUDART_NAN_F;

        int fv = first_valids[s];
        if (fv < 0) fv = 0; 

        
        float sum = 0.0f;
        float c = 0.0f;
        float prev_p = nan_f;           
        bool sticky_nan = false;        

        
        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            const float pt = price_tm[idx];
            const float vt = volume_tm[idx];

            if (t <= fv) {
                
                out_tm[idx] = nan_f;

                
                if (t == fv) {
                    if (fv < 1) { 
                        sticky_nan = true;
                    } else {
                        const float p0 = price_tm[(t - 1) * cols + s];
                        const float v1 = vt;
                        const bool ok = isfinite(p0) && isfinite(pt) && isfinite(v1) && (p0 != 0.0f);
                        if (ok) {
                            sum = v1 * ((pt - p0) / p0); 
                            c = 0.0f;
                            prev_p = pt;
                        } else {
                            sticky_nan = true;
                        }
                    }
                }
                continue; 
            }

            
            if (sticky_nan) {
                out_tm[idx] = nan_f;
                continue;
            }

            const bool good = isfinite(prev_p) && isfinite(pt) && isfinite(vt) && (prev_p != 0.0f);
            if (!good) {
                sticky_nan = true;
                out_tm[idx] = nan_f;
                continue;
            }

            const float cur = vt * ((pt - prev_p) / prev_p);
            kahan_add(cur, sum, c);
            out_tm[idx] = sum;
            prev_p = pt;
        }
    }
}

