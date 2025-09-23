// CUDA kernels for Ehlers Distance Coefficient Filter (EDCF).
//
// Design goals (mirrors ALMA CUDA scaffolding):
// - Plain batch path: one series × many-params using a scratch distance buffer
//   per parameter (avoids Ncombo×Ntime temporary matrices).
// - Tiled apply path: shared-memory tiling along time for long series to
//   improve locality when reducing the (dist,price) window per output.
// - Many-series path: supports 1D (per-series sequential time walk) and a 2D
//   tiled variant that computes local distance tiles on-the-fly with halos to
//   avoid global dist matrices.
// - FP32 arithmetic with FMA where appropriate; NaN boundary semantics match
//   the scalar implementation: outputs prior to warmup are NaN, where
//   warm = first_valid + 2*period.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void edcf_compute_dist_f32(const float* __restrict__ prices,
                           int len,
                           int period,
                           int first_valid,
                           float* __restrict__ dist) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int start = first_valid + period;

    for (int k = idx; k < len; k += stride) {
        if (k < start) {
            dist[k] = 0.0f;
            continue;
        }
        const float xk = prices[k];
        float sum_h = 0.0f, sum_c = 0.0f;
        for (int lb = 1; lb < period; ++lb) {
            const float d  = xk - prices[k - lb];
            const float q  = d * d;
            const float qe = __fmaf_rn(d, d, -q);
            const float t  = sum_h + q;
            const float z  = (fabsf(sum_h) >= fabsf(q)) ? (sum_h - t) + q : (q - t) + sum_h;
            sum_c += z + qe;
            sum_h  = t;
        }
        dist[k] = sum_h + sum_c;
    }
}

extern "C" __global__
void edcf_apply_weights_f32(const float* __restrict__ prices,
                            const float* __restrict__ dist,
                            int len,
                            int period,
                            int first_valid,
                            float* __restrict__ out_row) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int warm = first_valid + 2 * period;
    if (warm > len) {
        warm = len;
    }

    for (int j = idx; j < len; j += stride) {
        if (j < warm) {
            out_row[j] = NAN;
            continue;
        }

        float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        int i = 0;
        for (; i + 3 < period; i += 4) {
            const int k0 = j - (i + 0);
            const int k1 = j - (i + 1);
            const int k2 = j - (i + 2);
            const int k3 = j - (i + 3);
            const float w0 = dist[k0];
            const float w1 = dist[k1];
            const float w2 = dist[k2];
            const float w3 = dist[k3];
            const float v0 = prices[k0];
            const float v1 = prices[k1];
            const float v2 = prices[k2];
            const float v3 = prices[k3];
            n0 = __fmaf_rn(w0, v0, n0);
            n1 = __fmaf_rn(w1, v1, n1);
            n2 = __fmaf_rn(w2, v2, n2);
            n3 = __fmaf_rn(w3, v3, n3);
            d0 += w0; d1 += w1; d2 += w2; d3 += w3;
        }
        for (; i < period; ++i) {
            const int k = j - i;
            const float w = dist[k];
            const float v = prices[k];
            n0 = __fmaf_rn(w, v, n0);
            d0 += w;
        }
        const float num = (n0 + n1) + (n2 + n3);
        const float den = (d0 + d1) + (d2 + d3);
        out_row[j] = (den != 0.0f) ? (num / den) : NAN;
    }
}

// ------------------------ Tiled apply kernel ------------------------
// Each block computes `TILE` outputs for a single row (parameter combo).
// Shared memory stages the contiguous region [base-(P-1) .. base+TILE-1]
// for both `prices` and `dist`, enabling coalesced global loads and reuse.
// The distance buffer is expected to be already computed for this period
// into `dist` (scratch for the series), avoiding huge Ncombo×Ntime temporaries.

template<int TILE>
__device__ __forceinline__ void edcf_apply_weights_tiled_f32_impl(const float* __restrict__ prices,
                                                                  const float* __restrict__ dist,
                                                                  int len,
                                                                  int period,
                                                                  int first_valid,
                                                                  float* __restrict__ out_row) {
    const int block_outputs = TILE;
    const int base = blockIdx.x * block_outputs;
    if (base >= len) { return; }

    // Dynamic shared: [prices | dist], 16-byte aligned (match ALMA convention)
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    const int tile_prices_elems = (block_outputs + period - 1); // for [base-(P-1) .. base+TILE-1]
    float* sh_prices = smem;
    // Round up to 16B boundary (multiple of 4 floats) for second region
    const int sh_prices_aligned_elems = ((tile_prices_elems + 3) / 4) * 4;
    float* sh_dist   = sh_prices + sh_prices_aligned_elems;

    // Load contiguous windows from global to shared (zero fill OOB)
    // Start index in global memory
    const int start = base - (period - 1);
    const int end_incl = min(base + block_outputs - 1, len - 1);
    const int tile_elems = (end_incl - start + 1);

    // Vectorized load when aligned and length >= 4
    const int vec_elems = (tile_elems / 4) * 4; // floor to multiple of 4
    for (int i = threadIdx.x * 4; i < vec_elems; i += blockDim.x * 4) {
        int gidx = start + i;
        float4 pv = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 dv = make_float4(0.f, 0.f, 0.f, 0.f);
        // Require 16B alignment of global pointers for vectorized path
        if (gidx >= 0 && gidx + 3 < len && ((gidx & 3) == 0)) {
            const float4* __restrict__ p4 = reinterpret_cast<const float4*>(prices + gidx);
            const float4* __restrict__ d4 = reinterpret_cast<const float4*>(dist + gidx);
            pv = *p4;
            dv = *d4;
        } else {
            // Partial overlap: load scalars with bounds checks
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int idx = gidx + k;
                float p = 0.f, d = 0.f;
                if (idx >= 0 && idx < len) { p = prices[idx]; d = dist[idx]; }
                ((float*)&pv)[k] = p;
                ((float*)&dv)[k] = d;
            }
        }
        // sh_prices is 16B-aligned, and i is multiple of 4 floats here
        reinterpret_cast<float4*>(sh_prices + i)[0] = pv;
        reinterpret_cast<float4*>(sh_dist   + i)[0] = dv;
    }
    // Tail scalars
    for (int t = vec_elems + threadIdx.x; t < tile_elems; t += blockDim.x) {
        int gidx = start + t;
        float p = 0.f, d = 0.f;
        if (gidx >= 0 && gidx < len) { p = prices[gidx]; d = dist[gidx]; }
        sh_prices[t] = p;
        sh_dist[t] = d;
    }
    __syncthreads();

    const int warm = first_valid + 2 * period;

    // Each thread computes one output (optionally more via striding)
    for (int off = threadIdx.x; off < block_outputs && (base + off) < len; off += blockDim.x) {
        const int j = base + off;
        if (j < warm) {
            out_row[j] = CUDART_NAN_F;
            continue;
        }
        // Local window in shared memory starts at (off)
        const int sh_off = (off + (period - 1)) - (period - 1); // = off
        float n0 = 0.f, n1 = 0.f, n2 = 0.f, n3 = 0.f;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        int i2 = 0;
        for (; i2 + 3 < period; i2 += 4) {
            const float w0 = sh_dist[sh_off + i2 + 0];
            const float w1 = sh_dist[sh_off + i2 + 1];
            const float w2 = sh_dist[sh_off + i2 + 2];
            const float w3 = sh_dist[sh_off + i2 + 3];
            const float v0 = sh_prices[sh_off + i2 + 0];
            const float v1 = sh_prices[sh_off + i2 + 1];
            const float v2 = sh_prices[sh_off + i2 + 2];
            const float v3 = sh_prices[sh_off + i2 + 3];
            n0 = __fmaf_rn(w0, v0, n0);
            n1 = __fmaf_rn(w1, v1, n1);
            n2 = __fmaf_rn(w2, v2, n2);
            n3 = __fmaf_rn(w3, v3, n3);
            d0 += w0; d1 += w1; d2 += w2; d3 += w3;
        }
        for (; i2 < period; ++i2) {
            const float w = sh_dist[sh_off + i2];
            const float v = sh_prices[sh_off + i2];
            n0 = __fmaf_rn(w, v, n0);
            d0 += w;
        }
        const float num = (n0 + n1) + (n2 + n3);
        const float den = (d0 + d1) + (d2 + d3);
        out_row[j] = (den != 0.f) ? (num / den) : CUDART_NAN_F;
    }
}

extern "C" __global__ void edcf_apply_weights_tiled_f32_tile128(const float* __restrict__ prices,
                                                                 const float* __restrict__ dist,
                                                                 int len,
                                                                 int period,
                                                                 int first_valid,
                                                                 float* __restrict__ out_row) {
    edcf_apply_weights_tiled_f32_impl<128>(prices, dist, len, period, first_valid, out_row);
}
extern "C" __global__ void edcf_apply_weights_tiled_f32_tile256(const float* __restrict__ prices,
                                                                 const float* __restrict__ dist,
                                                                 int len,
                                                                 int period,
                                                                 int first_valid,
                                                                 float* __restrict__ out_row) {
    edcf_apply_weights_tiled_f32_impl<256>(prices, dist, len, period, first_valid, out_row);
}
extern "C" __global__ void edcf_apply_weights_tiled_f32_tile512(const float* __restrict__ prices,
                                                                 const float* __restrict__ dist,
                                                                 int len,
                                                                 int period,
                                                                 int first_valid,
                                                                 float* __restrict__ out_row) {
    edcf_apply_weights_tiled_f32_impl<512>(prices, dist, len, period, first_valid, out_row);
}

// ------------------------ Many-series (1D) -------------------------
// Each block owns one series (grid.x = num_series). A single thread (lane 0)
// walks time sequentially using small ring buffers for the last `period`
// distances and prices. Other lanes zero-init the output then exit.

extern "C" __global__
void edcf_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) { return; }
    const int stride = num_series;

    // Initialize output row with NaN in parallel
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) { return; }

    const int first_valid = first_valids[series_idx];
    if (first_valid < 0 || first_valid >= series_len) { return; }

    const int warm = first_valid + 2 * period;
    if (warm >= series_len) { return; }

    // Small local rings for last-period prices and distances (align shared)
    extern __shared__ __align__(16) unsigned char local_raw[];
    float* local = reinterpret_cast<float*>(local_raw);
    float* ring_p = local;            // [period]
    float* ring_d = local + period;   // [period]
    for (int i = 0; i < period; ++i) { ring_p[i] = 0.f; ring_d[i] = 0.f; }
    int head = 0;

    // Prime with values from [first_valid .. first_valid + period - 1]
    for (int t = first_valid; t < first_valid + period && t < series_len; ++t) {
        ring_p[head] = prices_tm[t * stride + series_idx];
        head = (head + 1) % period;
    }

    // Compute distances for t in [first_valid + period .. series_len)
    for (int t = first_valid + period; t < series_len; ++t) {
        const float xk = prices_tm[t * stride + series_idx];
        float sum_h = 0.f, sum_c = 0.f;
        int pos = (head + period - 1) % period;
        for (int lb = 1; lb < period; ++lb) {
            float prev = ring_p[pos];
            float d = xk - prev;
            float q = d * d;
            float qe = __fmaf_rn(d, d, -q);
            float tsum = sum_h + q;
            float z = (fabsf(sum_h) >= fabsf(q)) ? (sum_h - tsum) + q : (q - tsum) + sum_h;
            sum_c += z + qe;
            sum_h = tsum;
            pos = (pos + period - 1) % period;
        }
        ring_p[head] = xk;
        ring_d[head] = sum_h + sum_c;
        head = (head + 1) % period;

        // Once we pass warm, emit output
        if (t >= warm) {
            // Accumulate over last `period` (aligned so ring head points to next slot)
            float n0 = 0.f, n1 = 0.f, n2 = 0.f, n3 = 0.f;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            int ppos = (head + period - 1) % period;
            int i = 0;
            for (; i + 3 < period; i += 4) {
                float w0 = ring_d[ppos]; float v0 = ring_p[ppos]; ppos = (ppos + period - 1) % period;
                float w1 = ring_d[ppos]; float v1 = ring_p[ppos]; ppos = (ppos + period - 1) % period;
                float w2 = ring_d[ppos]; float v2 = ring_p[ppos]; ppos = (ppos + period - 1) % period;
                float w3 = ring_d[ppos]; float v3 = ring_p[ppos]; ppos = (ppos + period - 1) % period;
                n0 = __fmaf_rn(w0, v0, n0);
                n1 = __fmaf_rn(w1, v1, n1);
                n2 = __fmaf_rn(w2, v2, n2);
                n3 = __fmaf_rn(w3, v3, n3);
                d0 += w0; d1 += w1; d2 += w2; d3 += w3;
            }
            for (; i < period; ++i) {
                float w = ring_d[ppos]; float v = ring_p[ppos]; ppos = (ppos + period - 1) % period;
                n0 = __fmaf_rn(w, v, n0);
                d0 += w;
            }
            float num = (n0 + n1) + (n2 + n3);
            float den = (d0 + d1) + (d2 + d3);
            out_tm[t * stride + series_idx] = (den != 0.f) ? (num / den) : CUDART_NAN_F;
        }
    }
}

// ------------------------ Many-series (2D tiled) -------------------
// Grid.x sweeps time in tiles of TX; grid.y sweeps series in tiles of TY.
// For each (tile_t, tile_s) block, we load a price halo large enough to
// compute distances for [t0-(P-1) .. t0+TX-1] so that outputs for
// [t0 .. t0+TX-1] can be formed entirely from shared memory without global
// dist storage. For early times, outputs remain NaN by warmup check.

template<int TX, int TY>
__device__ __forceinline__ void edcf_ms1p_tiled_f32_impl(const float* __restrict__ prices_tm,
                                                         const int* __restrict__ first_valids,
                                                         int period,
                                                         int cols,
                                                         int rows,
                                                         float* __restrict__ out_tm) {
    const int tile_t0 = blockIdx.x * TX;
    const int tile_s0 = blockIdx.y * TY;
    if (tile_t0 >= rows || tile_s0 >= cols) { return; }
    const int stride = cols;

    // Shared layout per series lane: [prices tile | dist tile]
    const int prices_elems = TX + 2 * (period - 1); // [t0-2(P-1) .. t0+TX-1]
    const int dist_elems   = TX + (period - 1);     // [t0-(P-1) .. t0+TX-1]
    extern __shared__ __align__(16) unsigned char smem2_raw[];
    float* smem2 = reinterpret_cast<float*>(smem2_raw);
    // Partition by TY
    const int per_series = prices_elems + dist_elems;
    float* base_ptr = smem2 + threadIdx.y * per_series;
    float* sh_prices = base_ptr;
    float* sh_dist   = base_ptr + prices_elems;

    // For each active series lane (<= TY and within cols)
    const int s = tile_s0 + threadIdx.y;
    if (s >= cols) { return; }

    // Compute first_valid and warm for this series
    const int first_valid = first_valids[s];
    const int warm = first_valid + 2 * period;

    // Load prices window
    const int p_start = tile_t0 - 2 * (period - 1);
    const int p_end = min(tile_t0 + TX - 1, rows - 1);
    const int p_len = (p_end - p_start + 1);

    // Vectorized load across time for this series (threadIdx.x provides lanes)
    for (int t = threadIdx.x; t < p_len; t += blockDim.x) {
        int ti = p_start + t;
        float v = 0.f;
        if (ti >= 0 && ti < rows) { v = prices_tm[ti * stride + s]; }
        sh_prices[t] = v;
    }
    __syncthreads();

    // Compute dist for k in [t0-(P-1) .. t0+TX-1]
    const int d_start = tile_t0 - (period - 1);
    const int d_end = min(tile_t0 + TX - 1, rows - 1);
    const int d_len = (d_end - d_start + 1);

    for (int u = threadIdx.x; u < d_len; u += blockDim.x) {
        int k = d_start + u;
        float xk;
        if (k >= 0 && (k - (p_start)) >= 0 && (k - p_start) < p_len) {
            xk = sh_prices[(k - p_start)];
        } else {
            xk = 0.f;
        }
        float sum_h = 0.f, sum_c = 0.f;
        // Gather from sh_prices with index (k-lb - p_start)
        #pragma unroll 4
        for (int lb = 1; lb < period; ++lb) {
            int idx = (k - lb) - p_start;
            float prev = 0.f;
            if (idx >= 0 && idx < p_len) { prev = sh_prices[idx]; }
            float d = xk - prev;
            float q = d * d;
            float qe = __fmaf_rn(d, d, -q);
            float t = sum_h + q;
            float z = (fabsf(sum_h) >= fabsf(q)) ? (sum_h - t) + q : (q - t) + sum_h;
            sum_c += z + qe;
            sum_h = t;
        }
        sh_dist[u] = sum_h + sum_c;
    }
    __syncthreads();

    // Emit outputs for j in [t0 .. t0+TX-1]
    for (int off = threadIdx.x; off < TX && (tile_t0 + off) < rows; off += blockDim.x) {
        int j = tile_t0 + off;
        float y = CUDART_NAN_F;
        if (j >= warm) {
            float n0 = 0.f, n1 = 0.f, n2 = 0.f, n3 = 0.f;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            // indexes relative to sh arrays
            int sh_off = (j - (d_start)); // index into sh_dist where k=j
            int sh_p_off0 = (j - p_start); // index into sh_prices where t=j
            int i = 0;
            for (; i + 3 < period; i += 4) {
                float w0 = 0.f, v0 = 0.f;
                float w1 = 0.f, v1 = 0.f;
                float w2 = 0.f, v2 = 0.f;
                float w3 = 0.f, v3 = 0.f;
                int k0 = sh_off - (i + 0); int p0 = sh_p_off0 - (i + 0);
                int k1 = sh_off - (i + 1); int p1 = sh_p_off0 - (i + 1);
                int k2 = sh_off - (i + 2); int p2 = sh_p_off0 - (i + 2);
                int k3 = sh_off - (i + 3); int p3 = sh_p_off0 - (i + 3);
                if (k0 >= 0 && k0 < d_len) w0 = sh_dist[k0]; if (p0 >= 0 && p0 < p_len) v0 = sh_prices[p0];
                if (k1 >= 0 && k1 < d_len) w1 = sh_dist[k1]; if (p1 >= 0 && p1 < p_len) v1 = sh_prices[p1];
                if (k2 >= 0 && k2 < d_len) w2 = sh_dist[k2]; if (p2 >= 0 && p2 < p_len) v2 = sh_prices[p2];
                if (k3 >= 0 && k3 < d_len) w3 = sh_dist[k3]; if (p3 >= 0 && p3 < p_len) v3 = sh_prices[p3];
                n0 = __fmaf_rn(w0, v0, n0); d0 += w0;
                n1 = __fmaf_rn(w1, v1, n1); d1 += w1;
                n2 = __fmaf_rn(w2, v2, n2); d2 += w2;
                n3 = __fmaf_rn(w3, v3, n3); d3 += w3;
            }
            for (; i < period; ++i) {
                float w = 0.f, v = 0.f;
                int k_rel = sh_off - i;
                int p_rel = sh_p_off0 - i;
                if (k_rel >= 0 && k_rel < d_len) w = sh_dist[k_rel];
                if (p_rel >= 0 && p_rel < p_len) v = sh_prices[p_rel];
                n0 = __fmaf_rn(w, v, n0);
                d0 += w;
            }
            float num = (n0 + n1) + (n2 + n3);
            float den = (d0 + d1) + (d2 + d3);
            y = (den != 0.f) ? (num / den) : CUDART_NAN_F;
        }
        out_tm[j * stride + s] = y;
    }
}

extern "C" __global__ void edcf_ms1p_tiled_f32_tx128_ty2(const float* __restrict__ prices_tm,
                                                          const int* __restrict__ first_valids,
                                                          int period,
                                                          int cols,
                                                          int rows,
                                                          float* __restrict__ out_tm) {
    edcf_ms1p_tiled_f32_impl<128, 2>(prices_tm, first_valids, period, cols, rows, out_tm);
}
extern "C" __global__ void edcf_ms1p_tiled_f32_tx128_ty4(const float* __restrict__ prices_tm,
                                                          const int* __restrict__ first_valids,
                                                          int period,
                                                          int cols,
                                                          int rows,
                                                          float* __restrict__ out_tm) {
    edcf_ms1p_tiled_f32_impl<128, 4>(prices_tm, first_valids, period, cols, rows, out_tm);
}
