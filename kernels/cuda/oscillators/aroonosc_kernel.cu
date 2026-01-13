










#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif




__device__ __forceinline__
void max_earliest_update(float v, int i, float &best_v, int &best_i) {
    if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
}


__device__ __forceinline__
void min_earliest_update(float v, int i, float &best_v, int &best_i) {
    if (v < best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
}


__device__ __forceinline__
void warp_argmaxmin_earliest(float &max_v, int &max_i, float &min_v, int &min_i, unsigned mask) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float mv = __shfl_down_sync(mask, max_v, offset);
        int   mi = __shfl_down_sync(mask, max_i, offset);
        if (mv > max_v || (mv == max_v && mi < max_i)) { max_v = mv; max_i = mi; }

        float nv = __shfl_down_sync(mask, min_v, offset);
        int   ni = __shfl_down_sync(mask, min_i, offset);
        if (nv < min_v || (nv == min_v && ni < min_i)) { min_v = nv; min_i = ni; }
    }
}



extern "C" __global__
void aroonosc_batch_f32(const float* __restrict__ high,
                        const float* __restrict__ low,
                        const int*   __restrict__ lengths,
                        int series_len,
                        int first_valid,
                        int n_combos,
                        float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int base = combo * series_len;

    const int L = lengths[combo];
    if (L <= 0 || first_valid < 0 || first_valid >= series_len) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = CUDART_NAN_F;
        }
        return;
    }

    const int warm = first_valid + L;
    if (warm >= series_len) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = CUDART_NAN_F;
        }
        return;
    }


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out[base + i] = CUDART_NAN_F;
    }

    const float scale = 100.0f / (float)L;


    const unsigned mask = __activemask();
    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x / WARP_SIZE;


    for (int t = warm + warp_id; t < series_len; t += warps_per_block) {
        const int start = t - L;


        float max_v = high[start];
        int   max_i = start;
        float min_v = low[start];
        int   min_i = start;



        for (int j = start + lane; j <= t; j += WARP_SIZE) {
            const float h = high[j];
            const float l = low[j];
            max_earliest_update(h, j, max_v, max_i);
            min_earliest_update(l, j, min_v, min_i);
        }


        warp_argmaxmin_earliest(max_v, max_i, min_v, min_i, mask);

        if (lane == 0) {
            float v = (float)(max_i - min_i) * scale;

            v = fminf(100.0f, fmaxf(-100.0f, v));
            out[base + t] = v;
        }
    }
}




extern "C" __global__
void aroonosc_many_series_one_param_f32(const float* __restrict__ high_tm,
                                        const float* __restrict__ low_tm,
                                        const int*   __restrict__ first_valids,
                                        int num_series,
                                        int series_len,
                                        int length,
                                        float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || series_len <= 0) return;

    if (length <= 0) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + s] = CUDART_NAN_F;
        }
        return;
    }

    const int fv   = first_valids[s] < 0 ? 0 : first_valids[s];
    const int warm = fv + length;
    if (warm >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + s] = CUDART_NAN_F;
        }
        return;
    }


    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out_tm[t * num_series + s] = CUDART_NAN_F;
    }

    const float scale  = 100.0f / (float)length;
    const int   stride = num_series;

    if (threadIdx.x != 0) return;

    for (int t = warm; t < series_len; ++t) {
        const int start = t - length;
        int   hi_idx = start,  lo_idx = start;
        float hi_val = high_tm[start * stride + s];
        float lo_val =  low_tm[start * stride + s];

        for (int j = start + 1; j <= t; ++j) {
            const float h = high_tm[j * stride + s];
            if (h > hi_val) { hi_val = h; hi_idx = j; }
            const float l = low_tm[j * stride + s];
            if (l < lo_val) { lo_val = l; lo_idx = j; }
        }
        float v = (float)(hi_idx - lo_idx) * scale;
        v = fminf(100.0f, fmaxf(-100.0f, v));
        out_tm[t * stride + s] = v;
    }
}

