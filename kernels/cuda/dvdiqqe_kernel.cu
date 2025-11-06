// CUDA kernels for DVDIQQE (Dual Volume Divergence Index QQE-style)
// Optimized FP32-only path with FMA, Kahan summation, and reduced global stores.
// Warmup/NaN semantics preserved.
//
// Kernels:
// - dvdiqqe_batch_f32: one series × many params (each block handles 1 combo)
// - dvdiqqe_many_series_one_param_f32: many series × one param (time‑major)
//
// Warmup/NaN semantics:
//   warm = first_valid + (2*period - 1);
//   outputs before warm are NaN; at warm index TLs are seeded to dvdi.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

// ----------------------------- helpers ---------------------------------

static __forceinline__ __device__ float dvdiqqe_tick_volume(
    const float o, const float c, const float tick, float &tickrng_prev)
{
    const float rng = c - o;
    const float arng = fabsf(rng);
    const float tickrng = (arng < tick) ? tickrng_prev : rng;
    tickrng_prev = tickrng;
    const float tv = (tick > 0.0f) ? fabsf(tickrng) / tick : 0.0f;
    return tv > 0.0f ? tv : 0.0f;
}

// Pine-like selection: prefer provided volume if finite, else tick-volume.
// use_tick_only is kept at 0 in our kernels per original logic.
static __forceinline__ __device__ float select_volume(
    const float vol_opt, const float tick_vol, const int use_tick_only)
{
    if (use_tick_only) return tick_vol;
    if (isfinite(vol_opt)) return vol_opt;
    return tick_vol;
}

// Kahan compensated addition for long-running sums (PVI/NVI, etc.)
static __forceinline__ __device__ void kahan_add(float &sum, float &comp, const float x)
{
    float y = x - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}

// EMA in stable FMA form: y += a*(x - y)
static __forceinline__ __device__ float ema_update(const float y, const float x, const float a)
{
    return __fmaf_rn(a, (x - y), y);
}

// Safe quiet NaN
static __forceinline__ __device__ float qnan_f32() { return nanf(""); }

extern "C" __global__
void dvdiqqe_batch_f32(
    const float* __restrict__ open,
    const float* __restrict__ close,
    const float* __restrict__ volume, // may be null; use has_volume flag
    const int   has_volume,
    const int*  __restrict__ periods,
    const int*  __restrict__ smoothings,
    const float* __restrict__ fast_mults,
    const float* __restrict__ slow_mults,
    const int   n_combos,
    const int   series_len,
    const int   first_valid,
    const float tick_size,
    const int   center_dynamic, // 1 dynamic, 0 static
    float* __restrict__ out_dvdi,
    float* __restrict__ out_fast,
    float* __restrict__ out_slow,
    float* __restrict__ out_center)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0 || first_valid >= series_len) return;

    const int   period    = periods[combo];
    const int   smoothing = smoothings[combo];
    const float fast_mult = fast_mults[combo];
    const float slow_mult = slow_mults[combo];
    if (period <= 0 || smoothing <= 0) return;

    const int   wper = period * 2 - 1;
    const int   warm = first_valid + wper;

    float* dvdi_row = out_dvdi  + combo * series_len;
    float* fast_row = out_fast  + combo * series_len;
    float* slow_row = out_slow  + combo * series_len;
    float* cent_row = out_center+ combo * series_len;

    // FP32 coefficients
    const float a_p = 2.0f / (float)(period + 1);
    const float a_s = 2.0f / (float)(smoothing + 1);
    const float a_r = 2.0f / (float)(wper + 1);

    // Rolling state (FP32 + compensation where needed)
    float pvi_sum = 0.0f, pvi_c = 0.0f;
    float nvi_sum = 0.0f, nvi_c = 0.0f;

    float pvi_ema = 0.0f, nvi_ema = 0.0f; int pvi_cnt = 0;
    float pdiv_ema = 0.0f, ndiv_ema = 0.0f; int div_cnt = 0;

    float dvdi_prev = 0.0f; bool dvdi_inited = false;

    float rng_ema1 = 0.0f, rng_ema2 = 0.0f; int rng1_cnt = 0, rng2_cnt = 0; bool rng2_ready = false;

    float center_mean = 0.0f; int center_cnt = 0;

    float prev_vol = 0.0f;
    float prev_close = 0.0f;
    float tickrng_prev = tick_size;

    // Single-thread sequential per row to avoid multi-warp duplicates with large blocks
    if (threadIdx.x == 0)
    {
        #pragma unroll 1
        for (int t = 0; t < series_len; ++t) {
            const float oi = open[t];
            const float ci = close[t];
            if (!isfinite(ci)) { continue; }

            // Tick/real volume selection
            const float tick_vol = dvdiqqe_tick_volume(oi, ci, tick_size, tickrng_prev);
            const float real_vol = has_volume ? volume[t] : NAN;
            const float sel_vol  = select_volume(real_vol, tick_vol, /*use_tick_only*/ 0);

            if (t == 0) { prev_close = ci; prev_vol = sel_vol; }

            const float dpc = ci - prev_close;
            if (sel_vol > prev_vol) { kahan_add(pvi_sum, pvi_c,  dpc); }
            if (sel_vol < prev_vol) { kahan_add(nvi_sum, nvi_c, -dpc); }
            prev_close = ci; prev_vol = sel_vol;

            // Warmup-adjusted EMA(pvi/nvi)
            if (t >= first_valid) {
                const float pvi_val = pvi_sum + pvi_c;
                const float nvi_val = nvi_sum + nvi_c;
                if (pvi_cnt < period) {
                    // running mean via 1/n step
                    pvi_cnt += 1;
                    const float inv = 1.0f / (float)pvi_cnt;
                    pvi_ema = __fmaf_rn((pvi_val - pvi_ema), inv, pvi_ema);
                    nvi_ema = __fmaf_rn((nvi_val - nvi_ema), inv, nvi_ema);
                } else {
                    pvi_ema = ema_update(pvi_ema, pvi_val, a_p);
                    nvi_ema = ema_update(nvi_ema, nvi_val, a_p);
                }
            }

            const float pdiv = (pvi_sum + pvi_c) - pvi_ema;
            const float ndiv = (nvi_sum + nvi_c) - nvi_ema;

            // EMA(divergences) with warmup
            if (t >= first_valid) {
                if (div_cnt < smoothing) {
                    div_cnt += 1;
                    const float inv = 1.0f / (float)div_cnt;
                    pdiv_ema = __fmaf_rn((pdiv - pdiv_ema), inv, pdiv_ema);
                    ndiv_ema = __fmaf_rn((ndiv - ndiv_ema), inv, ndiv_ema);
                } else {
                    pdiv_ema = ema_update(pdiv_ema, pdiv, a_s);
                    ndiv_ema = ema_update(ndiv_ema, ndiv, a_s);
                }
            }

            const float dv = pdiv_ema - ndiv_ema;
            dvdi_row[t] = dv;  // warmup NaN masked later

            // |Δdvdi| range double-EMA with warmups
            if (!dvdi_inited) { dvdi_prev = dv; dvdi_inited = true; }
            const float abs_delta = fabsf(dv - dvdi_prev);
            if (t >= first_valid + 1) {
                if (rng1_cnt < wper) {
                    rng1_cnt += 1;
                    const float inv = 1.0f / (float)rng1_cnt;
                    rng_ema1 = __fmaf_rn((abs_delta - rng_ema1), inv, rng_ema1);
                } else {
                    rng_ema1 = ema_update(rng_ema1, abs_delta, a_r);
                }
                if (!rng2_ready) { rng2_ready = true; rng2_cnt = 0; }
                if (rng2_cnt < wper) {
                    rng2_cnt += 1;
                    const float inv2 = 1.0f / (float)rng2_cnt;
                    rng_ema2 = __fmaf_rn((rng_ema1 - rng_ema2), inv2, rng_ema2);
                } else {
                    rng_ema2 = ema_update(rng_ema2, rng_ema1, a_r);
                }
            }

            // Trailing levels (seed at warm)
            if (t == warm && rng2_cnt >= 1) {
                fast_row[t] = dv; slow_row[t] = dv;
            } else if (t > warm && rng2_cnt >= wper) {
                const float fr = rng_ema2 * fast_mult;
                const float sr = rng_ema2 * slow_mult;

                const float prev_fast = fast_row[t - 1];
                if (dv > prev_fast) {
                    const float nv = dv - fr;
                    fast_row[t] = (nv < prev_fast) ? prev_fast : nv;
                } else {
                    const float nv = dv + fr;
                    fast_row[t] = (nv > prev_fast) ? prev_fast : nv;
                }

                const float prev_slow = slow_row[t - 1];
                if (dv > prev_slow) {
                    const float nv = dv - sr;
                    slow_row[t] = (nv < prev_slow) ? prev_slow : nv;
                } else {
                    const float nv = dv + sr;
                    slow_row[t] = (nv > prev_slow) ? prev_slow : nv;
                }
            }

            // Center line
            if (t >= warm) {
                if (center_dynamic) {
                    // running mean of dv
                    center_cnt += 1;
                    const float invc = 1.0f / (float)center_cnt;
                    center_mean = __fmaf_rn((dv - center_mean), invc, center_mean);
                    cent_row[t] = center_mean;
                } else {
                    cent_row[t] = 0.0f;
                }
            }

            dvdi_prev = dv;
        }
    }

    __syncthreads(); // let thread 0 finish

    // Finalize warmup NaNs cooperatively (avoid whole-row prefill)
    const float qnan = qnan_f32();
    for (int i = threadIdx.x; i < series_len && i < warm; i += blockDim.x) {
        dvdi_row[i] = qnan;
        fast_row[i] = qnan;
        slow_row[i] = qnan;
        cent_row[i] = qnan;
    }
}

// ---------------- many series × one param (time-major) ------------------

extern "C" __global__
void dvdiqqe_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int   has_volume,
    const int*  __restrict__ first_valids, // per series (column)
    const int   period,
    const int   smoothing,
    const float fast_mult,
    const float slow_mult,
    const float tick_size,
    const int   center_dynamic,
    const int   num_series,
    const int   series_len,
    float* __restrict__ dvdi_tm,
    float* __restrict__ fast_tm,
    float* __restrict__ slow_tm,
    float* __restrict__ center_tm)
{
    if (period <= 0 || smoothing <= 0 || num_series <= 0 || series_len <= 0) return;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    const int wper = period * 2 - 1;
    const float a_p = 2.0f / (float)(period + 1);
    const float a_s = 2.0f / (float)(smoothing + 1);
    const float a_r = 2.0f / (float)(wper + 1);

    const int warp_idx = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep    = gridDim.x  * warps_per_block;
    const int cols     = num_series;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];
        if (first_valid < 0 || first_valid >= series_len) {
            // still need to NaN warmup region cooperatively
            const float qnan = qnan_f32();
            for (int t = lane; t < series_len; t += warpSize) {
                const int idx = t * cols + s;
                dvdi_tm[idx] = qnan; fast_tm[idx] = qnan; slow_tm[idx] = qnan; center_tm[idx] = qnan;
            }
            continue;
        }

        const int warm = first_valid + wper;

        // Rolling state per series (lane 0 does the scan)
        float pvi_sum = 0.0f, pvi_c = 0.0f;
        float nvi_sum = 0.0f, nvi_c = 0.0f;

        float pvi_ema = 0.0f, nvi_ema = 0.0f; int pvi_cnt = 0;
        float pdiv_ema = 0.0f, ndiv_ema = 0.0f; int div_cnt = 0;

        float dvdi_prev = 0.0f; bool dvdi_inited = false;

        float rng_ema1 = 0.0f, rng_ema2 = 0.0f; int rng1_cnt = 0, rng2_cnt = 0; bool rng2_ready = false;

        float center_mean = 0.0f; int center_cnt = 0;

        float prev_vol = 0.0f;
        float prev_close = 0.0f;
        float tickrng_prev = tick_size;

        if (lane == 0) {
            #pragma unroll 1
            for (int t = 0; t < series_len; ++t) {
                const int idx = t * cols + s;
                const float oi = open_tm[idx];
                const float ci = close_tm[idx];
                if (!isfinite(ci)) continue;

                const float tick_vol = dvdiqqe_tick_volume(oi, ci, tick_size, tickrng_prev);
                const float vol_tm   = has_volume ? volume_tm[idx] : NAN;
                const float sel_vol  = select_volume(vol_tm, tick_vol, /*use_tick_only*/ 0);

                if (t == 0) { prev_close = ci; prev_vol = sel_vol; }

                const float dpc = ci - prev_close;
                if (sel_vol > prev_vol) { kahan_add(pvi_sum, pvi_c,  dpc); }
                if (sel_vol < prev_vol) { kahan_add(nvi_sum, nvi_c, -dpc); }
                prev_close = ci; prev_vol = sel_vol;

                if (t >= first_valid) {
                    const float pvi_val = pvi_sum + pvi_c;
                    const float nvi_val = nvi_sum + nvi_c;
                    if (pvi_cnt < period) {
                        pvi_cnt += 1;
                        const float inv = 1.0f / (float)pvi_cnt;
                        pvi_ema = __fmaf_rn((pvi_val - pvi_ema), inv, pvi_ema);
                        nvi_ema = __fmaf_rn((nvi_val - nvi_ema), inv, nvi_ema);
                    } else {
                        pvi_ema = ema_update(pvi_ema, pvi_val, a_p);
                        nvi_ema = ema_update(nvi_ema, nvi_val, a_p);
                    }
                }

                const float pdiv = (pvi_sum + pvi_c) - pvi_ema;
                const float ndiv = (nvi_sum + nvi_c) - nvi_ema;

                if (t >= first_valid) {
                    if (div_cnt < smoothing) {
                        div_cnt += 1;
                        const float inv = 1.0f / (float)div_cnt;
                        pdiv_ema = __fmaf_rn((pdiv - pdiv_ema), inv, pdiv_ema);
                        ndiv_ema = __fmaf_rn((ndiv - ndiv_ema), inv, ndiv_ema);
                    } else {
                        pdiv_ema = ema_update(pdiv_ema, pdiv, a_s);
                        ndiv_ema = ema_update(ndiv_ema, ndiv, a_s);
                    }
                }

                const float dv = pdiv_ema - ndiv_ema;
                dvdi_tm[idx] = dv;

                if (!dvdi_inited) { dvdi_prev = dv; dvdi_inited = true; }
                const float abs_delta = fabsf(dv - dvdi_prev);
                if (t >= first_valid + 1) {
                    if (rng1_cnt < wper) {
                        rng1_cnt += 1;
                        const float inv = 1.0f / (float)rng1_cnt;
                        rng_ema1 = __fmaf_rn((abs_delta - rng_ema1), inv, rng_ema1);
                    } else {
                        rng_ema1 = ema_update(rng_ema1, abs_delta, a_r);
                    }
                    if (!rng2_ready) { rng2_ready = true; rng2_cnt = 0; }
                    if (rng2_cnt < wper) {
                        rng2_cnt += 1;
                        const float inv2 = 1.0f / (float)rng2_cnt;
                        rng_ema2 = __fmaf_rn((rng_ema1 - rng_ema2), inv2, rng_ema2);
                    } else {
                        rng_ema2 = ema_update(rng_ema2, rng_ema1, a_r);
                    }
                }

                if (t == warm && rng2_cnt >= 1) {
                    fast_tm[idx] = dv; slow_tm[idx] = dv;
                } else if (t > warm && rng2_cnt >= wper) {
                    const float fr = rng_ema2 * fast_mult;
                    const float sr = rng_ema2 * slow_mult;
                    const float pf = fast_tm[(t - 1) * cols + s];
                    const float ps = slow_tm[(t - 1) * cols + s];

                    if (dv > pf) { const float nv = dv - fr; fast_tm[idx] = (nv < pf) ? pf : nv; }
                    else          { const float nv = dv + fr; fast_tm[idx] = (nv > pf) ? pf : nv; }

                    if (dv > ps) { const float nv = dv - sr; slow_tm[idx] = (nv < ps) ? ps : nv; }
                    else          { const float nv = dv + sr; slow_tm[idx] = (nv > ps) ? ps : nv; }
                }

                if (t >= warm) {
                    if (center_dynamic) {
                        center_cnt += 1;
                        const float invc = 1.0f / (float)center_cnt;
                        center_mean = __fmaf_rn((dv - center_mean), invc, center_mean);
                        center_tm[idx] = center_mean;
                    } else {
                        center_tm[idx] = 0.0f;
                    }
                }

                dvdi_prev = dv;
            }
        }

        // Warmup NaNs for this column (warp-cooperative)
        const float qnan = qnan_f32();
        for (int t = lane; t < series_len && t < warm; t += warpSize) {
            const int idx = t * cols + s;
            dvdi_tm[idx] = qnan; fast_tm[idx] = qnan; slow_tm[idx] = qnan; center_tm[idx] = qnan;
        }
    }
}
