





#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    float block_sum = 0.0f;
    if (wid == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum; 
}









struct ds {
    float hi;
    float lo;
};

static __forceinline__ __device__ ds ds_make(float x) {
    ds r; r.hi = x; r.lo = 0.0f; return r;
}

static __forceinline__ __device__ float ds_value(const ds& v) {
    return v.hi + v.lo;
}


static __forceinline__ __device__ void twoProductFMA(float a, float b, float &x, float &y) {
    x = a * b;
    y = fmaf(a, b, -x);
}


static __forceinline__ __device__ void twoSum(float a, float b, float &s, float &e) {
    s = a + b;
    float z = s - a;
    e = (a - (s - z)) + (b - z);
}


static __forceinline__ __device__ void ds_rma_update(ds &s, float keep, float inc) {
    
    float p_hi, p_err;
    twoProductFMA(s.hi, keep, p_hi, p_err);
    float t_lo = s.lo * keep;               

    
    float sh, e_sum;
    twoSum(p_hi, inc, sh, e_sum);

    
    float slo = e_sum + (p_err + t_lo);
    float new_hi = sh + slo;
    s.lo = slo - (new_hi - sh);
    s.hi = new_hi;
}









extern "C" __global__
void di_batch_from_precomputed_f32(const float* __restrict__ up,
                                   const float* __restrict__ dn,
                                   const float* __restrict__ tr,
                                   const int* __restrict__ periods,
                                   const int* __restrict__ warm_indices,
                                   int series_len,
                                   int first_valid,
                                   int n_combos,
                                   float* __restrict__ plus_out,
                                   float* __restrict__ minus_out)
{
    for (int combo = blockIdx.x; combo < n_combos; combo += gridDim.x) {
        const int period = periods[combo];
        const int warm   = warm_indices[combo];
        if (period <= 0 || warm < 0 || warm >= series_len) continue;

        const float invp = 1.0f / (float)period;
        const float keep = 1.0f - invp; 

        const int base = combo * series_len;

        
        const int start = first_valid + 1;
        const int stop  = first_valid + period; 
        if (stop > series_len) {
            
            for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
                plus_out [base + i] = NAN;
                minus_out[base + i] = NAN;
            }
            __syncthreads();
            continue;
        }

        
        for (int i = threadIdx.x; i < warm; i += blockDim.x) {
            plus_out [base + i] = NAN;
            minus_out[base + i] = NAN;
        }
        __syncthreads();

        
        float lp = 0.0f, lm = 0.0f, lt = 0.0f;
        for (int t = start + threadIdx.x; t < stop; t += blockDim.x) {
            lp += up[t];
            lm += dn[t];
            lt += tr[t];
        }
        float sp = block_reduce_sum(lp);
        float sm = block_reduce_sum(lm);
        float st = block_reduce_sum(lt);

        if (threadIdx.x == 0) {
            
            ds cur_p = ds_make(sp);
            ds cur_m = ds_make(sm);
            ds cur_t = ds_make(st);

            float denom = ds_value(cur_t);
            float scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
            plus_out [base + warm] = ds_value(cur_p) * scale;
            minus_out[base + warm] = ds_value(cur_m) * scale;

            
            for (int t = warm + 1; t < series_len; ++t) {
                ds_rma_update(cur_p, keep, up[t]);
                ds_rma_update(cur_m, keep, dn[t]);
                ds_rma_update(cur_t, keep, tr[t]);

                denom = ds_value(cur_t);
                scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
                plus_out [base + t] = ds_value(cur_p) * scale;
                minus_out[base + t] = ds_value(cur_m) * scale;
            }
        }
        __syncthreads();
    }
}








extern "C" __global__
void di_many_series_one_param_f32(const float* __restrict__ high_tm,
                                  const float* __restrict__ low_tm,
                                  const float* __restrict__ close_tm,
                                  const int* __restrict__ first_valids,
                                  int period,
                                  int num_series,
                                  int series_len,
                                  float* __restrict__ plus_tm,
                                  float* __restrict__ minus_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;
    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    const float invp = 1.0f / (float)period;
    const float keep = 1.0f - invp;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];
        if (first_valid < 0 || first_valid >= series_len) {
            
            for (int t = lane; t < series_len; t += warpSize) {
                plus_tm [t * stride + s] = NAN;
                minus_tm[t * stride + s] = NAN;
            }
            continue;
        }

        const int start = first_valid + 1;
        const int stop  = first_valid + period; 
        if (stop > series_len) {
            for (int t = lane; t < series_len; t += warpSize) {
                plus_tm [t * stride + s] = NAN;
                minus_tm[t * stride + s] = NAN;
            }
            continue;
        }
        const int warm = stop - 1;

        
        for (int t = lane; t < warm; t += warpSize) {
            plus_tm [t * stride + s] = NAN;
            minus_tm[t * stride + s] = NAN;
        }

        
        float lp = 0.0f, lm = 0.0f, lt = 0.0f;
        for (int t = start + lane; t < stop; t += warpSize) {
            const float ch = high_tm[t * stride + s];
            const float cl = low_tm [t * stride + s];
            const float ph = high_tm[(t - 1) * stride + s];
            const float pl = low_tm [(t - 1) * stride + s];
            const float pc = close_tm[(t - 1) * stride + s];

            const float dp = ch - ph;
            const float dm = pl - cl;
            if (dp > dm && dp > 0.0f) lp += dp;
            if (dm > dp && dm > 0.0f) lm += dm;

            float tr = ch - cl;
            float hc = fabsf(ch - pc);
            if (hc > tr) tr = hc;
            float lc = fabsf(cl - pc);
            if (lc > tr) tr = lc;
            lt += tr;
        }
        
        lp = warp_reduce_sum(lp);
        lm = warp_reduce_sum(lm);
        lt = warp_reduce_sum(lt);

        if (lane == 0) {
            ds cur_p = ds_make(lp);
            ds cur_m = ds_make(lm);
            ds cur_t = ds_make(lt);

            float denom = ds_value(cur_t);
            float scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
            plus_tm [warm * stride + s] = ds_value(cur_p) * scale;
            minus_tm[warm * stride + s] = ds_value(cur_m) * scale;

            
            int t  = warm + 1;
            const float* h_ptr  = high_tm  + t * stride + s;
            const float* l_ptr  = low_tm   + t * stride + s;
            const float* ph_ptr = high_tm  + (t - 1) * stride + s;
            const float* pl_ptr = low_tm   + (t - 1) * stride + s;
            const float* pc_ptr = close_tm + (t - 1) * stride + s;

            for (; t < series_len; ++t) {
                const float ch = *h_ptr;  const float cl = *l_ptr;
                const float ph = *ph_ptr; const float pl = *pl_ptr;
                const float pc = *pc_ptr;

                const float dp = ch - ph;
                const float dm = pl - cl;
                const float inc_p = (dp > dm && dp > 0.0f) ? dp : 0.0f;
                const float inc_m = (dm > dp && dm > 0.0f) ? dm : 0.0f;

                float tr = ch - cl;
                float hc = fabsf(ch - pc);
                if (hc > tr) tr = hc;
                float lc = fabsf(cl - pc);
                if (lc > tr) tr = lc;

                ds_rma_update(cur_p, keep, inc_p);
                ds_rma_update(cur_m, keep, inc_m);
                ds_rma_update(cur_t, keep, tr);

                denom = ds_value(cur_t);
                scale = (denom == 0.0f) ? 0.0f : 100.0f / denom;
                plus_tm [t * stride + s] = ds_value(cur_p) * scale;
                minus_tm[t * stride + s] = ds_value(cur_m) * scale;

                h_ptr  += stride;  l_ptr  += stride;
                ph_ptr += stride;  pl_ptr += stride;  pc_ptr += stride;
            }
        }
    }
}
