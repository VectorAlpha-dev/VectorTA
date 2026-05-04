#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef VPT_SCAN_BLOCK_X
#define VPT_SCAN_BLOCK_X 256
#endif

#ifndef VPT_SCAN_ITEMS_PER_THREAD
#define VPT_SCAN_ITEMS_PER_THREAD 8
#endif

#define VPT_SCAN_TILE (VPT_SCAN_BLOCK_X * VPT_SCAN_ITEMS_PER_THREAD)


static __device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}


extern "C" __global__ void vpt_scan_blocks_f32(
    const float* __restrict__ price,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out,
    double* __restrict__ block_sums)
{
    __shared__ double scan[VPT_SCAN_TILE];
    __shared__ double temp[VPT_SCAN_TILE];

    const int base = blockIdx.x * VPT_SCAN_TILE;
    const int tid = threadIdx.x;
    const float nan_f = CUDART_NAN_F;

    if (first_valid < 0) first_valid = 0;

    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        const int idx = base + lane;
        double inc = 0.0;
        if (idx >= first_valid && idx < len) {
            if (idx < 1) {
                inc = (double)nan_f;
            } else {
                const float p0 = price[idx - 1];
                const float p1 = price[idx];
                const float v1 = volume[idx];
                inc = (isfinite(p0) && p0 != 0.0f && isfinite(p1) && isfinite(v1))
                    ? (double)v1 * ((double)p1 - (double)p0) / (double)p0
                    : (double)nan_f;
            }
        }
        scan[lane] = inc;
    }
    __syncthreads();

    for (int offset = 1; offset < VPT_SCAN_TILE; offset <<= 1) {
        #pragma unroll
        for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * VPT_SCAN_BLOCK_X;
            temp[lane] = scan[lane] + (lane >= offset ? scan[lane - offset] : 0.0);
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * VPT_SCAN_BLOCK_X;
            scan[lane] = temp[lane];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        const int idx = base + lane;
        if (idx < len) out[idx] = idx <= first_valid ? nan_f : (float)scan[lane];
    }

    if (tid == 0) {
        int remaining = len - base;
        int count = remaining > VPT_SCAN_TILE ? VPT_SCAN_TILE : remaining;
        block_sums[blockIdx.x] = count > 0 ? scan[count - 1] : 0.0;
    }
}


extern "C" __global__ void vpt_scan_block_sums_f64(
    double* __restrict__ block_sums,
    int num_blocks)
{
    __shared__ double scan[VPT_SCAN_TILE];
    __shared__ double temp[VPT_SCAN_TILE];

    const int tid = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        scan[lane] = lane < num_blocks ? block_sums[lane] : 0.0;
    }
    __syncthreads();

    for (int offset = 1; offset < VPT_SCAN_TILE; offset <<= 1) {
        #pragma unroll
        for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * VPT_SCAN_BLOCK_X;
            temp[lane] = scan[lane] + (lane >= offset ? scan[lane - offset] : 0.0);
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * VPT_SCAN_BLOCK_X;
            scan[lane] = temp[lane];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        if (lane < num_blocks) block_sums[lane] = scan[lane];
    }
}


extern "C" __global__ void vpt_add_block_offsets_f32(
    float* __restrict__ out,
    int len,
    int first_valid,
    const double* __restrict__ block_sums)
{
    const int base = blockIdx.x * VPT_SCAN_TILE;
    if (blockIdx.x == 0) return;

    if (first_valid < 0) first_valid = 0;
    const double offset = block_sums[blockIdx.x - 1];
    const int tid = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        const int idx = base + lane;
        if (idx < len && idx > first_valid) out[idx] = (float)((double)out[idx] + offset);
    }
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
