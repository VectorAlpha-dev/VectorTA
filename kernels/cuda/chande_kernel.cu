












#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

static __forceinline__ __device__ float tr_at(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int t,
    int first_valid)
{
    const float hi = high[t];
    const float lo = low[t];
    if (t == first_valid) {
        return hi - lo;
    }
    const float pc = close[t - 1];
    float tr = hi - lo;
    float hc = fabsf(hi - pc);
    if (hc > tr) tr = hc;
    float lc = fabsf(lo - pc);
    if (lc > tr) tr = lc;
    return tr;
}

extern "C" __global__
void chande_batch_f32(const float* __restrict__ high,
                      const float* __restrict__ low,
                      const float* __restrict__ close,
                      const int* __restrict__ periods,
                      const float* __restrict__ mults,
                      const int* __restrict__ dirs,
                      const float* __restrict__ alphas,
                      const int* __restrict__ warm_indices,
                      int series_len,
                      int first_valid,
                      int n_combos,
                      float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int   period = periods[combo];
    const float mult   = mults[combo];
    const int   dir    = dirs[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;


    double sum_tr = 0.0;
    for (int t = first_valid; t < first_valid + period; ++t) {
        sum_tr += (double)tr_at(high, low, close, t, first_valid);
    }
    double atr = sum_tr / (double)period;


    {
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        const int wstart = warm + 1 - period;
        for (int t = wstart; t <= warm; ++t) {
            const float v = (dir != 0) ? high[t] : low[t];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + warm] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }


    for (int t = warm + 1; t < series_len; ++t) {
        const float tri = tr_at(high, low, close, t, first_valid);
        atr = fma((double)tri - atr, (double)alpha, atr);
        const int wstart = t + 1 - period;
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        for (int k = wstart; k <= t; ++k) {
            const float v = (dir != 0) ? high[k] : low[k];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + t] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }
}


extern "C" __global__
void chande_batch_from_tr_f32(const float* __restrict__ high,
                              const float* __restrict__ low,
                              const float* __restrict__ tr,
                              const int* __restrict__ periods,
                              const float* __restrict__ mults,
                              const int* __restrict__ dirs,
                              const float* __restrict__ alphas,
                              const int* __restrict__ warm_indices,
                              int series_len,
                              int first_valid,
                              int n_combos,
                              float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int   period = periods[combo];
    const float mult   = mults[combo];
    const int   dir    = dirs[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) { out[base + idx] = NAN; }
    __syncthreads();
    if (threadIdx.x != 0) return;


    double sum_tr = 0.0;
    for (int t = first_valid; t < first_valid + period; ++t) { sum_tr += (double)tr[t]; }
    double atr = sum_tr / (double)period;

    {
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        const int wstart = warm + 1 - period;
        for (int t = wstart; t <= warm; ++t) {
            const float v = (dir != 0) ? high[t] : low[t];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + warm] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }

    for (int t = warm + 1; t < series_len; ++t) {
        const float tri = tr[t];
        atr = fma((double)tri - atr, (double)alpha, atr);
        const int wstart = t + 1 - period;
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        for (int k = wstart; k <= t; ++k) {
            const float v = (dir != 0) ? high[k] : low[k];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + t] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }
}


extern "C" __global__
void chande_many_series_one_param_f32(const float* __restrict__ high_tm,
                                      const float* __restrict__ low_tm,
                                      const float* __restrict__ close_tm,
                                      const int* __restrict__ first_valids,
                                      int period,
                                      float mult,
                                      int dir,
                                      float alpha,
                                      int num_series,
                                      int series_len,
                                      float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;
    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];

        for (int t = lane; t < series_len; t += warpSize) {
            out_tm[t * stride + s] = NAN;
        }
        if (first_valid < 0 || first_valid >= series_len) continue;
        const int warm = first_valid + period - 1;
        if (warm >= series_len) continue;

        if (lane == 0) {

            double sum_tr = 0.0;
            for (int t = first_valid; t < first_valid + period; ++t) {
                const float hi = high_tm[t * stride + s];
                const float lo = low_tm[t * stride + s];
                float tri;
                if (t == first_valid) {
                    tri = hi - lo;
                } else {
                    const float pc = close_tm[(t - 1) * stride + s];
                    float tr = hi - lo;
                    float hc = fabsf(hi - pc);
                    if (hc > tr) tr = hc;
                    float lc = fabsf(lo - pc);
                    if (lc > tr) tr = lc;
                    tri = tr;
                }
                sum_tr += (double)tri;
            }
            double atr = sum_tr / (double)period;

            {
                float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
                const int wstart = warm + 1 - period;
                for (int t = wstart; t <= warm; ++t) {
                    const float v = (dir != 0) ? high_tm[t * stride + s] : low_tm[t * stride + s];
                    if (dir != 0) { if (v > extrema) extrema = v; }
                    else          { if (v < extrema) extrema = v; }
                }
                out_tm[warm * stride + s] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
            }

            for (int t = warm + 1; t < series_len; ++t) {
                const float hi = high_tm[t * stride + s];
                const float lo = low_tm[t * stride + s];
                const float pc = close_tm[(t - 1) * stride + s];
                float tr = hi - lo;
                float hc = fabsf(hi - pc);
                if (hc > tr) tr = hc;
                float lc = fabsf(lo - pc);
                if (lc > tr) tr = lc;
                atr = fma((double)tr - atr, (double)alpha, atr);
                const int wstart = t + 1 - period;
                float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
                for (int k = wstart; k <= t; ++k) {
                    const float v = (dir != 0) ? high_tm[k * stride + s] : low_tm[k * stride + s];
                    if (dir != 0) { if (v > extrema) extrema = v; }
                    else          { if (v < extrema) extrema = v; }
                }
                out_tm[t * stride + s] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
            }
        }
    }
}












static __forceinline__ __device__ float tr_from_hlpc(
    float hi, float lo, float pc, int t, int first_valid)
{
    if (t == first_valid) return hi - lo;
    float tr  = hi - lo;
    float hc  = fabsf(hi - pc);
    float lc  = fabsf(lo - pc);
    if (hc > tr) tr = hc;
    if (lc > tr) tr = lc;
    return tr;
}



static __forceinline__ __device__ void dq_push_monotone(
    int* __restrict__ idx_buf,
    float* __restrict__ val_buf,
    unsigned int mask,
    int& head, int& tail,
    int idx_new, float val_new, bool keep_max)
{

    while (head != tail) {
        unsigned int last = (static_cast<unsigned int>(tail - 1)) & mask;
        float back_val = val_buf[last];
        if (keep_max ? (back_val >= val_new) : (back_val <= val_new)) break;
        tail = static_cast<int>(last);
    }
    val_buf[tail] = val_new;
    idx_buf[tail] = idx_new;
    tail = static_cast<int>((static_cast<unsigned int>(tail) + 1u) & mask);
}


static __forceinline__ __device__ void dq_pop_expired(
    const int* __restrict__ idx_buf,
    unsigned int mask,
    int& head, int tail, int window_start)
{
    while (head != tail) {
        if (idx_buf[head] >= window_start) break;
        head = static_cast<int>((static_cast<unsigned int>(head) + 1u) & mask);
    }
}


static __forceinline__ __device__ float dq_front_value(
    const float* __restrict__ val_buf, unsigned int mask, int head)
{
    return val_buf[head & mask];
}








extern "C" __global__
void chande_one_series_many_params_f32(const float* __restrict__ high,
                                       const float* __restrict__ low,
                                       const float* __restrict__ close,
                                       const int*   __restrict__ periods,
                                       const float* __restrict__ mults,
                                       const int*   __restrict__ dirs,
                                       const float* __restrict__ alphas,
                                       int first_valid,
                                       int series_len,
                                       int n_combos,
                                       int queue_cap,
                                       int*   __restrict__ dq_idx,
                                       float* __restrict__ dq_val,
                                       float* __restrict__ out)
{
    const int lane            = threadIdx.x & 31;
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;

    const unsigned full_mask = 0xFFFFFFFFu;
    const unsigned int qmask = static_cast<unsigned int>(queue_cap - 1);

    for (int w = warp_idx; w < (n_combos + 31) / 32; w += total_warps) {
        const int combo = (w << 5) + lane;
        if (combo >= n_combos) continue;

        const int   period = periods[combo];
        const float mult   = mults[combo];
        const int   dir    = dirs[combo];
        const float alpha  = alphas[combo];

        const int warm = first_valid + period - 1;
        const int base = combo * series_len;

        if (period <= 0 || warm >= series_len || first_valid >= series_len) {

            for (int t0 = 0; t0 < series_len; ++t0) {
                out[base + t0] = NAN;
            }
            continue;
        }


        for (int t0 = 0; t0 < warm; ++t0) {
            out[base + t0] = NAN;
        }


        int*   ring_idx = dq_idx + combo * queue_cap;
        float* ring_val = dq_val + combo * queue_cap;
        int head = 0, tail = 0;


        float seed_sum = 0.0f, c = 0.0f;
        float atr = 0.0f;
        bool  atr_seeded = false;


        float prev_close_b = 0.0f;
        for (int t = 0; t < series_len; ++t) {

            float hi = 0.0f, lo = 0.0f, pc = 0.0f;
            if (lane == 0) {
                hi = high[t];
                lo = low[t];
                if (t > 0) pc = close[t - 1];
            }
            hi = __shfl_sync(full_mask, hi, 0);
            lo = __shfl_sync(full_mask, lo, 0);
            if (t > 0) prev_close_b = __shfl_sync(full_mask, pc, 0);


            if (t >= first_valid) {
                const float v = (dir != 0) ? hi : lo;
                dq_push_monotone(ring_idx, ring_val, qmask, head, tail, t, v, (dir != 0));

                const int wstart = t + 1 - period;
                dq_pop_expired(ring_idx, qmask, head, tail, wstart);
            }


            if (t >= first_valid && !atr_seeded) {
                const float tri = tr_from_hlpc(hi, lo, prev_close_b, t, first_valid);

                const float y = tri - c;
                const float tmp = seed_sum + y;
                c = (tmp - seed_sum) - y;
                seed_sum = tmp;

                if (t == warm) {
                    atr = seed_sum / static_cast<float>(period);
                    atr_seeded = true;


                    const float ext = dq_front_value(ring_val, qmask, head);
                    out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
                }
            } else if (atr_seeded && t > warm) {

                const float tri = tr_from_hlpc(hi, lo, prev_close_b, t, first_valid);
                atr = __fmaf_rn(alpha, (tri - atr), atr);


                const float ext = dq_front_value(ring_val, qmask, head);
                out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
            }
        }
    }
}


extern "C" __global__
void chande_one_series_many_params_from_tr_f32(const float* __restrict__ high,
                                               const float* __restrict__ low,
                                               const float* __restrict__ tr,
                                               const int*   __restrict__ periods,
                                               const float* __restrict__ mults,
                                               const int*   __restrict__ dirs,
                                               const float* __restrict__ alphas,
                                               int first_valid,
                                               int series_len,
                                               int n_combos,
                                               int queue_cap,
                                               int*   __restrict__ dq_idx,
                                               float* __restrict__ dq_val,
                                               float* __restrict__ out)
{
    const int lane            = threadIdx.x & 31;
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;
    const unsigned full_mask = 0xFFFFFFFFu;
    const unsigned int qmask = static_cast<unsigned int>(queue_cap - 1);

    for (int w = warp_idx; w < (n_combos + 31) / 32; w += total_warps) {
        const int combo = (w << 5) + lane;
        if (combo >= n_combos) continue;

        const int   period = periods[combo];
        const float mult   = mults[combo];
        const int   dir    = dirs[combo];
        const float alpha  = alphas[combo];

        const int warm = first_valid + period - 1;
        const int base = combo * series_len;

        if (period <= 0 || warm >= series_len || first_valid >= series_len) {
            for (int t0 = 0; t0 < series_len; ++t0) out[base + t0] = NAN;
            continue;
        }
        for (int t0 = 0; t0 < warm; ++t0) out[base + t0] = NAN;

        int*   ring_idx = dq_idx + combo * queue_cap;
        float* ring_val = dq_val + combo * queue_cap;
        int head = 0, tail = 0;

        float seed_sum = 0.0f, c = 0.0f;
        float atr = 0.0f;
        bool  atr_seeded = false;

        for (int t = 0; t < series_len; ++t) {
            float hi = 0.0f, lo = 0.0f, tri = 0.0f;
            if (lane == 0) {
                hi  = high[t];
                lo  = low[t];
                tri = tr[t];
            }
            hi  = __shfl_sync(full_mask, hi, 0);
            lo  = __shfl_sync(full_mask, lo, 0);
            tri = __shfl_sync(full_mask, tri, 0);

            if (t >= first_valid) {
                const float v = (dir != 0) ? hi : lo;
                dq_push_monotone(ring_idx, ring_val, qmask, head, tail, t, v, (dir != 0));
                const int wstart = t + 1 - period;
                dq_pop_expired(ring_idx, qmask, head, tail, wstart);
            }

            if (t >= first_valid && !atr_seeded) {
                const float y = tri - c;
                const float tmp = seed_sum + y;
                c = (tmp - seed_sum) - y;
                seed_sum = tmp;

                if (t == warm) {
                    atr = seed_sum / static_cast<float>(period);
                    atr_seeded = true;
                    const float ext = dq_front_value(ring_val, qmask, head);
                    out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
                }
            } else if (atr_seeded && t > warm) {
                atr = __fmaf_rn(alpha, (tri - atr), atr);
                const float ext = dq_front_value(ring_val, qmask, head);
                out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
            }
        }
    }
}
