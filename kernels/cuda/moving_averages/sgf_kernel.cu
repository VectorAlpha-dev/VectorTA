#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef SGF_BLOCK_X
#define SGF_BLOCK_X 128
#endif
#ifndef SGF_SERIES_PER_BLOCK
#define SGF_SERIES_PER_BLOCK 4
#endif
#ifndef SGF_MAX_PERIOD
#define SGF_MAX_PERIOD 4096
#endif
#ifndef SGF_USE_CONST_WEIGHTS
#define SGF_USE_CONST_WEIGHTS 1
#endif

#if SGF_USE_CONST_WEIGHTS
__constant__ float c_sgf_weights[SGF_MAX_PERIOD];
#endif

extern "C" __global__
void sgf_batch_f32(const float* __restrict__ prices,
                   const float* __restrict__ weights_flat,
                   const int* __restrict__ periods,
                   const int* __restrict__ warm_indices,
                   int series_len,
                   int n_combos,
                   int max_period,
                   float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int t = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= series_len) return;

    const int warm = warm_indices[combo];
    const int out_base = combo * series_len;

    extern __shared__ float smem[];
    float* w_sh = smem;
    float* tile = smem + max_period;

    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        w_sh[k] = weights_flat[combo * max_period + k];
    }
    __syncthreads();

    const int tile_start = int(blockIdx.x * blockDim.x);
    const int load_begin = max(tile_start - (period - 1), 0);
    const int load_end = min(tile_start + int(blockDim.x) - 1, series_len - 1);
    const int load_len = max(0, load_end - load_begin + 1);
    for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
        tile[i] = prices[load_begin + i];
    }
    __syncthreads();

    if (t < warm) {
        out[out_base + t] = NAN;
        return;
    }

    const int start = t - period + 1;
    const int tile_off = start - load_begin;
    float acc = 0.0f;
    for (int k = 0; k < period; ++k) {
        acc = fmaf(tile[tile_off + k], w_sh[k], acc);
    }
    out[out_base + t] = acc;
}

extern "C" __global__
void sgf_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const float* __restrict__ weights,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {
    const int series_block_base = int(blockIdx.y * blockDim.y);
    const int s_local = int(threadIdx.y);
    const int s = series_block_base + s_local;
    if (s >= num_series) return;

    extern __shared__ float smem[];
    float* tile = smem;
#if !SGF_USE_CONST_WEIGHTS
    float* w_sh = smem;
    tile = w_sh + period;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        w_sh[k] = weights[k];
    }
    __syncthreads();
#endif

    const int tile_t0 = int(blockIdx.x * blockDim.x);
    const int local_t = int(threadIdx.x);
    const int t = tile_t0 + local_t;
    const int warm = first_valids[s] + period - 1;

    const int in_begin = max(tile_t0 - (period - 1), 0);
    const int in_end = min(tile_t0 + int(blockDim.x) - 1, series_len - 1);
    const int load_len = max(0, in_end - in_begin + 1);
    const int tile_span = load_len * int(blockDim.y);

    int linear = local_t * int(blockDim.y) + s_local;
    for (int idx = linear; idx < tile_span; idx += int(blockDim.x * blockDim.y)) {
        int dt = idx / int(blockDim.y);
        int ss = idx % int(blockDim.y);
        int gs = series_block_base + ss;
        if (gs < num_series) {
            tile[idx] = prices_tm[(in_begin + dt) * num_series + gs];
        }
    }
    __syncthreads();

    if (t >= series_len) return;
    if (t < warm) {
        out_tm[t * num_series + s] = NAN;
        return;
    }

    const int start_t = t - period + 1;
    const int base = (start_t - in_begin) * int(blockDim.y) + s_local;
    float acc = 0.0f;
    for (int k = 0; k < period; ++k) {
#if SGF_USE_CONST_WEIGHTS
        acc = fmaf(tile[base + k * int(blockDim.y)], c_sgf_weights[k], acc);
#else
        acc = fmaf(tile[base + k * int(blockDim.y)], w_sh[k], acc);
#endif
    }
    out_tm[t * num_series + s] = acc;
}
