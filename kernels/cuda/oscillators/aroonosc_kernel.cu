// CUDA kernels for Aroon Oscillator (AROONOSC).
//
// Semantics mirror the scalar Rust implementation in src/indicators/aroonosc.rs:
// - Warmup index per series/row: warm = first_valid + length
// - Before warm: outputs remain NaN
// - After warm: compute indices of highest high and lowest low over the last
//   (length + 1) bars and emit 100/length * (idx_high - idx_low), clamped to [-100, 100].
// - Inputs are FP32; outputs are FP32.
// - We keep a straightforward O(window) rescan per t, but parallelize the
//   inner reduction across warp lanes for the one-series × many-params path.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ---- helpers (inline, header-only) -----------------------------------------

// Earliest-arg comparator for max: prefer larger value; on equal, smaller index.
__device__ __forceinline__
void max_earliest_update(float v, int i, float &best_v, int &best_i) {
    if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
}

// Earliest-arg comparator for min: prefer smaller value; on equal, smaller index.
__device__ __forceinline__
void min_earliest_update(float v, int i, float &best_v, int &best_i) {
    if (v < best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
}

// Warp-wide reduce (value,index) to lane 0 using earliest-index tiebreak.
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

// ---- one-series × many-params (primary) ------------------------------------
// One CUDA block per parameter row. Warps cooperatively rescan windows.
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

    const int warm = first_valid + L;        // window [t-L .. t] has L+1 elems
    if (warm >= series_len) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = CUDART_NAN_F;
        }
        return;
    }

    // Warmup region only; everything >= warm is overwritten by the main loop.
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out[base + i] = CUDART_NAN_F;
    }

    const float scale = 100.0f / (float)L;

    // Warp-parallel rescan setup.
    const unsigned mask = __activemask();
    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // Each warp handles t = warm + warp_id, warm + warp_id + warps_per_block, ...
    for (int t = warm + warp_id; t < series_len; t += warps_per_block) {
        const int start = t - L;

        // Seed with the first element to preserve scalar NaN/tie semantics.
        float max_v = high[start];
        int   max_i = start;
        float min_v = low[start];
        int   min_i = start;

        // Lane-strided, coalesced scan over [start..t]
        // (lane 0 also revisits 'start' but won’t change best due to tie rule)
        for (int j = start + lane; j <= t; j += WARP_SIZE) {
            const float h = high[j];
            const float l = low[j];
            max_earliest_update(h, j, max_v, max_i);
            min_earliest_update(l, j, min_v, min_i);
        }

        // Reduce to lane 0
        warp_argmaxmin_earliest(max_v, max_i, min_v, min_i, mask);

        if (lane == 0) {
            float v = (float)(max_i - min_i) * scale;
            // Clamp to [-100, 100] branchlessly
            v = fminf(100.0f, fmaxf(-100.0f, v));
            out[base + t] = v;
        }
    }
}

// ---- many-series × one-param (time-major) ----------------------------------
// Kept simple; small cleanups + branchless clamp. (The time-major layout
// prevents coalescing when a block owns one series; deeper changes optional.)
extern "C" __global__
void aroonosc_many_series_one_param_f32(const float* __restrict__ high_tm,
                                        const float* __restrict__ low_tm,
                                        const int*   __restrict__ first_valids,
                                        int num_series,
                                        int series_len,
                                        int length,
                                        float* __restrict__ out_tm) {
    const int s = blockIdx.x; // one block per series
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

    // Warmup region only; everything >= warm is overwritten by the main loop.
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out_tm[t * num_series + s] = CUDART_NAN_F;
    }

    const float scale  = 100.0f / (float)length;
    const int   stride = num_series;

    if (threadIdx.x != 0) return; // single lane (keep simple)

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

