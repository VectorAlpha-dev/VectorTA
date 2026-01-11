








#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef ERI_TIME_TILE

#define ERI_TIME_TILE 16
#endif


__device__ __forceinline__ float eri_qnan() {
    
    return nanf("");
}





extern "C" __global__ void eri_batch_f32(
    const float* __restrict__ high,   
    const float* __restrict__ low,    
    const float* __restrict__ ma,     
    int series_len,
    int first_valid,
    int period,
    float* __restrict__ bull,         
    float* __restrict__ bear          
) {
    const int stride = blockDim.x * gridDim.x;
    const int warm   = first_valid + period - 1;
    const float nanv = eri_qnan();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += stride) {
        if (i < warm) {
            if (bull) bull[i] = nanv;
            if (bear) bear[i] = nanv;
        } else {
            const float m = ma[i];
            if (bull) bull[i] = high[i] - m;
            if (bear) bear[i] = low[i]  - m;
        }
    }
}






extern "C" __global__ void eri_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ ma_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids, 
    int period,
    float* __restrict__ bull_tm,
    float* __restrict__ bear_tm
) {
    const int s  = blockIdx.x * blockDim.x + threadIdx.x; 
    if (s >= cols) return;

    const int warm   = first_valids[s] + period - 1;
    const float nanv = eri_qnan();

    
    for (int t0 = blockIdx.y * ERI_TIME_TILE; t0 < rows; t0 += gridDim.y * ERI_TIME_TILE) {
        const int tlimit = (rows - t0 > ERI_TIME_TILE) ? ERI_TIME_TILE : (rows - t0);

        
        int prefix = warm - t0;
        if (prefix < 0) prefix = 0;
        if (prefix > tlimit) prefix = tlimit;
        if (prefix > 0) {
            for (int tt = 0; tt < prefix; ++tt) {
                const int idx = (t0 + tt) * cols + s;
                if (bull_tm) bull_tm[idx] = nanv;
                if (bear_tm) bear_tm[idx] = nanv;
            }
        }

        
        if (prefix < tlimit) {
            for (int tt = prefix; tt < tlimit; ++tt) {
                const int idx = (t0 + tt) * cols + s;
                const float m = ma_tm[idx];
                if (bull_tm) bull_tm[idx] = high_tm[idx] - m;
                if (bear_tm) bear_tm[idx] = low_tm[idx]  - m;
            }
        }
    }
}







extern "C" __global__ void eri_one_series_many_params_time_major_f32(
    const float* __restrict__ high,      
    const float* __restrict__ low,       
    const float* __restrict__ ma_tm,     
    int P,                               
    int rows,
    int first_valid,
    const int* __restrict__ periods,     
    int period,                           
    float* __restrict__ bull_out,        
    float* __restrict__ bear_out,        
    int out_row_major                    
) {
    __shared__ float sh_high[ERI_TIME_TILE];
    __shared__ float sh_low [ERI_TIME_TILE];

    const float nanv = eri_qnan();

    const int p0      = blockIdx.x * blockDim.x + threadIdx.x;
    const int pstride = gridDim.x  * blockDim.x;

    
    for (int t0 = blockIdx.y * ERI_TIME_TILE; t0 < rows; t0 += gridDim.y * ERI_TIME_TILE) {
        const int tlimit = (rows - t0 > ERI_TIME_TILE) ? ERI_TIME_TILE : (rows - t0);

        
        if (threadIdx.x < tlimit) {
            sh_high[threadIdx.x] = high[t0 + threadIdx.x];
            sh_low [threadIdx.x] = low [t0 + threadIdx.x];
        }
        __syncthreads();

        
        for (int p = p0; p < P; p += pstride) {
            const int per   = (periods ? periods[p] : period);
            const int warm  = first_valid + per - 1;
            const int base  = t0 * P + p;

            
            int prefix = warm - t0;
            if (prefix < 0) prefix = 0;
            if (prefix > tlimit) prefix = tlimit;
            if (prefix > 0) {
                if (out_row_major) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        const int t = t0 + tt;
                        if (bull_out) bull_out[p*rows + t] = nanv;
                        if (bear_out) bear_out[p*rows + t] = nanv;
                    }
                } else if (bull_out && bear_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        const int idx = base + tt * P;
                        bull_out[idx] = nanv;
                        bear_out[idx] = nanv;
                    }
                } else if (bull_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        bull_out[base + tt * P] = nanv;
                    }
                } else if (bear_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        bear_out[base + tt * P] = nanv;
                    }
                }
            }

            
            if (prefix < tlimit) {
                if (out_row_major) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int t = t0 + tt;
                        const float m = ma_tm[base + tt * P];
                        if (bull_out) bull_out[p*rows + t] = sh_high[tt] - m;
                        if (bear_out) bear_out[p*rows + t] = sh_low [tt] - m;
                    }
                } else if (bull_out && bear_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        const float m = ma_tm[idx];
                        bull_out[idx] = sh_high[tt] - m;
                        bear_out[idx] = sh_low [tt] - m;
                    }
                } else if (bull_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        bull_out[idx] = sh_high[tt] - ma_tm[idx];
                    }
                } else if (bear_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        bear_out[idx] = sh_low[tt] - ma_tm[idx];
                    }
                }
            }
        }
        __syncthreads();
    }
}






extern "C" __global__ void transpose_rm_to_tm_32x32_pad_f32(
    const float* __restrict__ in,
    int R, int C,
    float* __restrict__ out
){
    __shared__ float tile[32][32+1]; 

    int c0 = blockIdx.x * 32 + threadIdx.x; 
    int r0 = blockIdx.y * 32 + threadIdx.y; 

    if (r0 < R && c0 < C) {
        tile[threadIdx.y][threadIdx.x] = in[r0 * C + c0];
    } else {
        tile[threadIdx.y][threadIdx.x] = eri_qnan();
    }
    __syncthreads();

    int r1 = blockIdx.y * 32 + threadIdx.x; 
    int c1 = blockIdx.x * 32 + threadIdx.y; 
    if (r1 < R && c1 < C) {
        out[c1 * R + r1] = tile[threadIdx.x][threadIdx.y];
    }
}

