// CUDA kernels for Directional Indicator (DI): computes +DI and -DI using
// Wilder's smoothing. Matches scalar warmup/NaN rules exactly:
// - Warm index = first_valid + period - 1; prior outputs are NaN.
// - Initial window sums loop over t in [first_valid+1 .. first_valid+period-1].
// - Thereafter, sequential RMA update with keep = 1 - 1/period.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ---- Block-wide reduction helpers (float) ----------------------------------
static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
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
    return block_sum; // valid in lane 0 of warp 0
}

// ---- Batch kernel using host-precomputed up/dn/tr arrays --------------------
// up, dn, tr are length = series_len arrays with zeros prior to first_valid+1.
// periods and warm_indices have length n_combos. plus_out/minus_out are
// [n_combos * series_len] row-major (row = combo).
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
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int warm   = warm_indices[combo];
    if (period <= 0 || warm < 0 || warm >= series_len) return;

    const float invp = 1.0f / (float)period;
    const float keep = 1.0f - invp; // Wilder keep factor

    const int base = combo * series_len;

    // Initialize entire row to NaN cooperatively
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        plus_out[base + i]  = NAN;
        minus_out[base + i] = NAN;
    }
    __syncthreads();

    // Accumulate initial window sums over t in [first_valid+1 .. first_valid+period-1]
    const int start = first_valid + 1;
    const int stop  = first_valid + period; // exclusive
    if (stop > series_len) return;

    float lp = 0.0f;
    float lm = 0.0f;
    float lt = 0.0f;
    for (int t = start + threadIdx.x; t < stop; t += blockDim.x) {
        lp += up[t];
        lm += dn[t];
        lt += tr[t];
    }
    float sp = block_reduce_sum(lp);
    float sm = block_reduce_sum(lm);
    float st = block_reduce_sum(lt);

    if (threadIdx.x == 0) {
        // cur_* kept in FP64 to mirror scalar stability
        double cur_p = (double)sp;
        double cur_m = (double)sm;
        double cur_t = (double)st;
        float scale = (cur_t == 0.0) ? 0.0f : (float)(100.0 / cur_t);
        plus_out[base + warm]  = (float)(cur_p) * scale;
        minus_out[base + warm] = (float)(cur_m) * scale;

        const double k = (double)keep;
        for (int t = warm + 1; t < series_len; ++t) {
            cur_p = fma(cur_p, k, (double)up[t]);
            cur_m = fma(cur_m, k, (double)dn[t]);
            cur_t = fma(cur_t, k, (double)tr[t]);
            scale = (cur_t == 0.0) ? 0.0f : (float)(100.0 / cur_t);
            plus_out[base + t]  = (float)(cur_p) * scale;
            minus_out[base + t] = (float)(cur_m) * scale;
        }
    }
}

// ---- Many-series × one-param (time-major) ----------------------------------
// Inputs are time-major: X_tm[t * num_series + s]. Output buffers are also
// time-major (rows x cols laid out with stride = num_series). Each warp
// advances one series sequentially.
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
    const int warp_in_block   = threadIdx.x >> 5; // / warpSize
    const int warps_per_block = blockDim.x >> 5;  // / warpSize
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    const float invp = 1.0f / (float)period;
    const double keep = (double)(1.0f - invp);

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];
        // Initialize entire column to NaN cooperatively by lanes
        for (int t = lane; t < series_len; t += warpSize) {
            plus_tm[t * stride + s]  = NAN;
            minus_tm[t * stride + s] = NAN;
        }

        if (first_valid < 0 || first_valid >= series_len) continue;
        const int start = first_valid + 1;
        const int stop  = first_valid + period; // exclusive
        if (stop > series_len) continue; // insufficient samples
        const int warm = stop - 1;

        // Seed sums across initial window using lane-parallel accumulation
        double sp = 0.0, sm = 0.0, st = 0.0;
        double lp = 0.0, lm = 0.0, lt = 0.0;
        for (int t = start + lane; t < stop; t += warpSize) {
            const float ch = high_tm[t * stride + s];
            const float cl = low_tm[t * stride + s];
            const float pc = close_tm[(t - 1) * stride + s];
            const float dp = ch - high_tm[(t - 1) * stride + s];
            const float dm = low_tm[(t - 1) * stride + s] - cl;
            if (dp > dm && dp > 0.0f) lp += (double)dp;
            if (dm > dp && dm > 0.0f) lm += (double)dm;
            float tr = ch - cl;
            float hc = fabsf(ch - pc);
            if (hc > tr) tr = hc;
            float lc = fabsf(cl - pc);
            if (lc > tr) tr = lc;
            lt += (double)tr;
        }
        // Warp reduce doubles via two-step (cast to float for shuffle then widen)
        float rp = (float)lp; rp = warp_reduce_sum(rp); sp = (double)rp;
        float rm = (float)lm; rm = warp_reduce_sum(rm); sm = (double)rm;
        float rt = (float)lt; rt = warp_reduce_sum(rt); st = (double)rt;

        if (lane == 0) {
            double cur_p = sp;
            double cur_m = sm;
            double cur_t = st;
            float scale = (cur_t == 0.0) ? 0.0f : (float)(100.0 / cur_t);
            plus_tm[warm * stride + s]  = (float)cur_p * scale;
            minus_tm[warm * stride + s] = (float)cur_m * scale;

            for (int t = warm + 1; t < series_len; ++t) {
                const float ch = high_tm[t * stride + s];
                const float cl = low_tm[t * stride + s];
                const float ph = high_tm[(t - 1) * stride + s];
                const float pl = low_tm[(t - 1) * stride + s];
                const float pc = close_tm[(t - 1) * stride + s];
                const float dp = ch - ph;
                const float dm = pl - cl;
                const float inc_p = (dp > dm && dp > 0.0f) ? dp : 0.0f;
                const float inc_m = (dm > dp && dm > 0.0f) ? dm : 0.0f;
                float tr = ch - cl;
                float hc = fabsf(ch - pc);
                if (hc > tr) tr = hc;
                float lc = fabsf(cl - pc);
                if (lc > tr) tr = lc;
                cur_p = fma(cur_p, keep, (double)inc_p);
                cur_m = fma(cur_m, keep, (double)inc_m);
                cur_t = fma(cur_t, keep, (double)tr);
                scale = (cur_t == 0.0) ? 0.0f : (float)(100.0 / cur_t);
                plus_tm[t * stride + s]  = (float)cur_p * scale;
                minus_tm[t * stride + s] = (float)cur_m * scale;
            }
        }
    }
}

