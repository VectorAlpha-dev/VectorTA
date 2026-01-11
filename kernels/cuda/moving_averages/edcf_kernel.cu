













#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include "../ds_float2.cuh"


__device__ __forceinline__ dsf ds_renorm_full(float s, float e) {
    float t  = s + e;
    float z  = t - s;
    float lo = (s - (t - z)) + (e - z);
    return {t, lo};
}

__device__ __forceinline__ dsf ds_add_full(dsf a, dsf b) {
    
    float s  = a.hi + b.hi;
    float z  = s - a.hi;
    float e  = (a.hi - (s - z)) + (b.hi - z);
    e += a.lo + b.lo;
    return ds_renorm_full(s, e);
}

__device__ __forceinline__ dsf ds_scale_full(dsf a, float s) {
    float p  = a.hi * s;
    float e  = fmaf(a.hi, s, -p) + a.lo * s;
    return ds_renorm_full(p, e);
}

__device__ __forceinline__ dsf ds_mul_full(dsf a, dsf b) {
    float p  = a.hi * b.hi;
    float e  = fmaf(a.hi, b.hi, -p);
    e += a.hi * b.lo + a.lo * b.hi;
    e += a.lo * b.lo;
    return ds_renorm_full(p, e);
}

__device__ __forceinline__ dsf ds_neg_full(dsf a) { return {-a.hi, -a.lo}; }
__device__ __forceinline__ dsf ds_sub_full(dsf a, dsf b) { return ds_add_full(a, ds_neg_full(b)); }

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




template<int TILE>
__device__ __forceinline__ void edcf_compute_dist_rolling_tiled_f32(const float* __restrict__ prices,
                                                    int len,
                                                    int period,
                                                    int first_valid,
                                                    float* __restrict__ dist)
{
    const int m = period - 1;
    if (m <= 0 || len <= 0) return;

    const int start = first_valid + period;               
    const int base  = blockIdx.x * TILE;
    const int j1    = min(base + TILE - 1, len - 1);

    
    if (j1 < start) {
        for (int k = base + threadIdx.x; k <= j1; k += blockDim.x) dist[k] = 0.0f;
        return;
    }

    const int j0 = max(start, base);
    const int p_start = j0 - m;                           
    const int p_len   = j1 - p_start + 1;                 

    extern __shared__ float sh_prices[];                  
    
    for (int t = threadIdx.x; t < p_len; t += blockDim.x) {
        sh_prices[t] = prices[p_start + t];
    }
    __syncthreads();

    
    if (threadIdx.x == 0) {
        
        
        
        dsf sum1 = ds_set(0.f);
        dsf sum2 = ds_set(0.f);
        #pragma unroll 1
        for (int i = 0; i < m; ++i) {
            const float v  = sh_prices[i];
            const dsf dv = ds_set(v);
            sum1 = ds_add_full(sum1, dv);
            sum2 = ds_add_full(sum2, ds_mul_full(dv, dv));
        }

        
        for (int k = base; k < j0; ++k) if (k <= j1) dist[k] = 0.0f;

        
        const int out_cnt = j1 - j0 + 1;
        #pragma unroll 1
        for (int u = 0; u < out_cnt; ++u) {
            const float xk   = sh_prices[m + u];
            const dsf xk_ds  = ds_set(xk);
            const dsf xk2_ds = ds_mul_full(xk_ds, xk_ds);
            dsf w_ds = ds_add_full(ds_scale_full(xk2_ds, (float)m), sum2);
            w_ds = ds_add_full(w_ds, ds_scale_full(ds_mul_full(xk_ds, sum1), -2.f));
            float w = w_ds.hi + w_ds.lo;
            
            w = fmaxf(w, 0.0f);
            dist[j0 + u] = w;

            
            const float v_out  = sh_prices[u];
            sum1 = ds_add_full(sum1, ds_set(xk - v_out));
            const dsf v_out2_ds = ds_mul_full(ds_set(v_out), ds_set(v_out));
            sum2 = ds_add_full(sum2, ds_sub_full(xk2_ds, v_out2_ds));
        }
    }
}

extern "C" __global__
void edcf_compute_dist_rolling_f32_tile128(const float* __restrict__ prices,
                                           int len, int period, int first_valid,
                                           float* __restrict__ dist) {
    edcf_compute_dist_rolling_tiled_f32<128>(prices, len, period, first_valid, dist);
}
extern "C" __global__
void edcf_compute_dist_rolling_f32_tile256(const float* __restrict__ prices,
                                           int len, int period, int first_valid,
                                           float* __restrict__ dist) {
    edcf_compute_dist_rolling_tiled_f32<256>(prices, len, period, first_valid, dist);
}
extern "C" __global__
void edcf_compute_dist_rolling_f32_tile512(const float* __restrict__ prices,
                                           int len, int period, int first_valid,
                                           float* __restrict__ dist) {
    edcf_compute_dist_rolling_tiled_f32<512>(prices, len, period, first_valid, dist);
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








template<int TILE>
__device__ __forceinline__ void edcf_apply_weights_tiled_f32_impl(const float* __restrict__ prices,
                                                                  const float* __restrict__ dist,
                                                                  int len,
                                                                  int period,
                                                                  int first_valid,
                                                                  float* __restrict__ out_row) {
    const int P = period;
    const int m = P - 1;
    const int base = blockIdx.x * TILE;
    if (base >= len) { return; }

    
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    const int tile_prices_elems = (TILE + m); 
    float* sh_prices = smem;
    const int sh_prices_aligned_elems = ((tile_prices_elems + 3) / 4) * 4; 
    float* sh_dist   = sh_prices + sh_prices_aligned_elems;

    
    const int start = base - m;
    const int end_incl = min(base + TILE - 1, len - 1);
    const int tile_elems = (end_incl - start + 1);

    
    const int vec_elems = (tile_elems / 4) * 4;
    for (int i = threadIdx.x * 4; i < vec_elems; i += blockDim.x * 4) {
        int gidx = start + i;
        float4 pv = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 dv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gidx >= first_valid && gidx + 3 < len && ((gidx & 3) == 0)) {
            const float4* __restrict__ p4 = reinterpret_cast<const float4*>(prices + gidx);
            const float4* __restrict__ d4 = reinterpret_cast<const float4*>(dist + gidx);
            pv = *p4; dv = *d4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int idx = gidx + k;
                float p = 0.f, w = 0.f;
                if (idx >= 0 && idx < len) {
                    if (idx >= first_valid) {
                        p = prices[idx];
                        w = dist[idx];
                    } else {
                        p = 0.f;
                        w = 0.f;
                    }
                }
                ((float*)&pv)[k] = p; ((float*)&dv)[k] = w;
            }
        }
        reinterpret_cast<float4*>(sh_prices + i)[0] = pv;
        reinterpret_cast<float4*>(sh_dist   + i)[0] = dv;
    }
    for (int t = vec_elems + threadIdx.x; t < tile_elems; t += blockDim.x) {
        int gidx = start + t;
        float p = 0.f, w = 0.f;
        if (gidx >= 0 && gidx < len) {
            if (gidx >= first_valid) {
                p = prices[gidx];
                w = dist[gidx];
            } else {
                p = 0.f;
                w = 0.f;
            }
        }
        sh_prices[t] = p;
        sh_dist[t]   = w;
    }
    __syncthreads();

    
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f;
        #pragma unroll 1
        for (int i = 0; i < tile_elems; ++i) {
            const float w = sh_dist[i];
            const float p = sh_prices[i];
            a += w;
            b = __fmaf_rn(w, p, b);
            sh_dist[i]   = a; 
            sh_prices[i] = b; 
        }
    }
    __syncthreads();

    const int warm = first_valid + 2 * P;

    for (int off = threadIdx.x; off < TILE && (base + off) < len; off += blockDim.x) {
        const int j = base + off;
        if (j < warm) { out_row[j] = CUDART_NAN_F; continue; }
        const int pos_j    = j - start; 
        const int pos_prev = pos_j - P;
        const float pw  = sh_dist[pos_j]  - ((pos_prev >= 0) ? sh_dist[pos_prev]  : 0.f);
        const float pwv = sh_prices[pos_j]- ((pos_prev >= 0) ? sh_prices[pos_prev]: 0.f);
        out_row[j] = (pw != 0.f) ? (pwv / pw) : CUDART_NAN_F;
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

    
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) { return; }

    const int first_valid = first_valids[series_idx];
    if (first_valid < 0 || first_valid >= series_len) { return; }

    const int warm = first_valid + 2 * period;
    if (warm >= series_len) { return; }

    
    extern __shared__ __align__(16) unsigned char local_raw[];
    float* local = reinterpret_cast<float*>(local_raw);
    float* ring_p = local;            
    float* ring_d = local + period;   
    for (int i = 0; i < period; ++i) { ring_p[i] = 0.f; ring_d[i] = 0.f; }
    int head = 0;

    
    for (int t = first_valid; t < first_valid + period && t < series_len; ++t) {
        ring_p[head] = prices_tm[t * stride + series_idx];
        head = (head + 1) % period;
    }

    
    float w_sum = 0.f;
    float wv_sum = 0.f;

    
    for (int t = first_valid + period; t < series_len; ++t) {
        const float xk = prices_tm[t * stride + series_idx];

        
        const float w_out = ring_d[head];
        const float p_out = ring_p[head];
        w_sum  -= w_out;
        wv_sum -= w_out * p_out;

        
        float sum_h = 0.f, sum_c = 0.f;
        int pos = (head + period - 1) % period;
        for (int lb = 1; lb < period; ++lb) {
            const float prev = ring_p[pos];
            const float d  = xk - prev;
            const float q  = d * d;
            const float qe = __fmaf_rn(d, d, -q);
            const float tsum = sum_h + q;
            const float z  = (fabsf(sum_h) >= fabsf(q)) ? (sum_h - tsum) + q : (q - tsum) + sum_h;
            sum_c += z + qe;
            sum_h  = tsum;
            pos = (pos + period - 1) % period;
        }
        const float w_new = sum_h + sum_c;

        
        ring_p[head] = xk;
        ring_d[head] = w_new;

        
        w_sum  += w_new;
        wv_sum = __fmaf_rn(w_new, xk, wv_sum);

        head = (head + 1) % period;

        
        if (t >= warm) {
            out_tm[t * stride + series_idx] = (w_sum != 0.f) ? (wv_sum / w_sum) : CUDART_NAN_F;
        }
    }
}








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

    
    const int prices_elems = TX + 2 * (period - 1); 
    const int dist_elems   = TX + (period - 1);     
    extern __shared__ __align__(16) unsigned char smem2_raw[];
    float* smem2 = reinterpret_cast<float*>(smem2_raw);
    
    const int per_series = prices_elems + dist_elems;
    float* base_ptr = smem2 + threadIdx.y * per_series;
    float* sh_prices = base_ptr;
    float* sh_dist   = base_ptr + prices_elems;

    
    const int s = tile_s0 + threadIdx.y;
    if (s >= cols) { return; }

    
    const int first_valid = first_valids[s];
    const int warm = first_valid + 2 * period;

    
    const int p_start = tile_t0 - 2 * (period - 1);
    const int p_end = min(tile_t0 + TX - 1, rows - 1);
    const int p_len = (p_end - p_start + 1);

    
    for (int t = threadIdx.x; t < p_len; t += blockDim.x) {
        int ti = p_start + t;
        float v = 0.f;
        if (ti >= 0 && ti < rows) {
            
            
            v = (ti >= first_valid) ? prices_tm[ti * stride + s] : 0.f;
        }
        sh_prices[t] = v;
    }
    __syncthreads();

    
    const int d_start = tile_t0 - (period - 1);
    const int d_end = min(tile_t0 + TX - 1, rows - 1);
    const int d_len = (d_end - d_start + 1);

    for (int u = threadIdx.x; u < d_len; u += blockDim.x) {
        int k = d_start + u;
        
        
        
        const int start = first_valid + period;
        if (k < start) {
            sh_dist[u] = 0.f;
            continue;
        }
        float xk;
        if (k >= 0 && (k - (p_start)) >= 0 && (k - p_start) < p_len) {
            xk = sh_prices[(k - p_start)];
        } else {
            xk = 0.f;
        }
        float sum_h = 0.f, sum_c = 0.f;
        
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

    
    
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f;
        #pragma unroll 1
        for (int i = 0; i < d_len; ++i) {
            const int xp = i + (d_start - p_start);
            const float w = sh_dist[i];
            const float p = (xp >= 0 && xp < p_len) ? sh_prices[xp] : 0.f;
            a += w;
            b = __fmaf_rn(w, p, b);
            sh_dist[i]   = a;     
            sh_prices[i] = b;     
        }
    }
    __syncthreads();

    
    for (int off = threadIdx.x; off < TX && (tile_t0 + off) < rows; off += blockDim.x) {
        const int j = tile_t0 + off;
        float y = CUDART_NAN_F;
        if (j >= warm) {
            const int pos_j    = (j - d_start);      
            const int pos_prev = pos_j - period;
            const float pw  = sh_dist[pos_j]   - ((pos_prev >= 0) ? sh_dist[pos_prev]   : 0.f);
            const float pwv = sh_prices[pos_j] - ((pos_prev >= 0) ? sh_prices[pos_prev] : 0.f);
            y = (pw != 0.f) ? (pwv / pw) : CUDART_NAN_F;
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
