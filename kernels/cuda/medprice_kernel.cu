


#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <math_constants.h>
#include <math_functions.h>

#ifndef ROW_TILE
#define ROW_TILE 8
#endif


static __device__ __forceinline__ bool is_aligned_16(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0);
}

static __device__ __forceinline__ float medprice_scalar(float h, float l) {

    const float s = h + l;
    return __isnanf(s) ? CUDART_NAN_F : 0.5f * s;
}




extern "C" __global__ __launch_bounds__(256, 2)
void medprice_kernel_f32(const float* __restrict__ high,
                         const float* __restrict__ low,
                         int len,
                         int first_valid,
                         float* __restrict__ out)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    if (len <= 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;


    for (int i = tid; i < len && i < fv; i += stride) {
        out[i] = CUDART_NAN_F;
    }


    const bool do_vec = is_aligned_16(high) && is_aligned_16(low) && is_aligned_16(out);
    if (do_vec) {

        const int vecStart = ((fv + 3) & ~3);

        for (int i = fv + tid; i < vecStart && i < len; i += stride) {
            out[i] = medprice_scalar(high[i], low[i]);
        }

        if (vecStart < len) {
            const int totalVec = (len - vecStart) >> 2;
            const float4* __restrict__ h4 = reinterpret_cast<const float4*>(high + vecStart);
            const float4* __restrict__ l4 = reinterpret_cast<const float4*>(low  + vecStart);
            float4* __restrict__ o4       = reinterpret_cast<float4*>(out + vecStart);

            for (int v = tid; v < totalVec; v += stride) {
                const float4 ah = h4[v];
                const float4 al = l4[v];
                float4 r;
                float sx = ah.x + al.x; r.x = __isnanf(sx) ? CUDART_NAN_F : 0.5f * sx;
                float sy = ah.y + al.y; r.y = __isnanf(sy) ? CUDART_NAN_F : 0.5f * sy;
                float sz = ah.z + al.z; r.z = __isnanf(sz) ? CUDART_NAN_F : 0.5f * sz;
                float sw = ah.w + al.w; r.w = __isnanf(sw) ? CUDART_NAN_F : 0.5f * sw;
                o4[v] = r;
            }


            const int tailStart = vecStart + (totalVec << 2);
            for (int i = tailStart + tid; i < len; i += stride) {
                out[i] = medprice_scalar(high[i], low[i]);
            }
        }
    } else {

        for (int i = max(fv, 0) + tid; i < len; i += stride) {
            out[i] = medprice_scalar(high[i], low[i]);
        }
    }
}





extern "C" __global__ __launch_bounds__(256, 2)
void medprice_batch_f32(const float* __restrict__ high,
                        const float* __restrict__ low,
                        int len,
                        int rows,
                        const int* __restrict__ first_valids,
                        float* __restrict__ out)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int rowTileBase = blockIdx.y * ROW_TILE;
    if (rowTileBase >= rows) return;

    const bool do_vec = is_aligned_16(high) && is_aligned_16(low) && is_aligned_16(out);
    if (do_vec) {
        const int totalVec = len >> 2;
        const float4* __restrict__ h4 = reinterpret_cast<const float4*>(high);
        const float4* __restrict__ l4 = reinterpret_cast<const float4*>(low);

        for (int v = tid; v < totalVec; v += stride) {
            const int i = v << 2;
            const float4 ah = h4[v];
            const float4 al = l4[v];


            const float sx = ah.x + al.x; const float r0 = __isnanf(sx) ? CUDART_NAN_F : 0.5f * sx;
            const float sy = ah.y + al.y; const float r1 = __isnanf(sy) ? CUDART_NAN_F : 0.5f * sy;
            const float sz = ah.z + al.z; const float r2 = __isnanf(sz) ? CUDART_NAN_F : 0.5f * sz;
            const float sw = ah.w + al.w; const float r3 = __isnanf(sw) ? CUDART_NAN_F : 0.5f * sw;

            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                float* __restrict__ o = out + row * len + i;
                o[0] = (i + 0 < fv) ? CUDART_NAN_F : r0;
                o[1] = (i + 1 < fv) ? CUDART_NAN_F : r1;
                o[2] = (i + 2 < fv) ? CUDART_NAN_F : r2;
                o[3] = (i + 3 < fv) ? CUDART_NAN_F : r3;
            }
        }


        const int tailStart = (totalVec << 2);
        for (int i = tailStart + tid; i < len; i += stride) {
            const float r = medprice_scalar(high[i], low[i]);
            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                out[row * len + i] = (i < fv) ? CUDART_NAN_F : r;
            }
        }
    } else {

        for (int i = tid; i < len; i += stride) {
            const float r = medprice_scalar(high[i], low[i]);
            #pragma unroll
            for (int ry = 0; ry < ROW_TILE; ++ry) {
                const int row = rowTileBase + ry;
                if (row >= rows) break;
                const int fv = first_valids ? max(first_valids[row], 0) : 0;
                out[row * len + i] = (i < fv) ? CUDART_NAN_F : r;
            }
        }
    }
}





extern "C" __global__ __launch_bounds__(256, 2)
void medprice_many_series_one_param_f32(const float* __restrict__ high_tm,
                                        const float* __restrict__ low_tm,
                                        int cols,
                                        int rows,
                                        const int* __restrict__ first_valids,
                                        float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids ? max(first_valids[s], 0) : 0;


    #pragma unroll 4
    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;
        if (t < fv) {
            out_tm[idx] = CUDART_NAN_F;
        } else {
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            const float ssum = h + l;
            out_tm[idx] = __isnanf(ssum) ? CUDART_NAN_F : 0.5f * ssum;
        }
    }
}
