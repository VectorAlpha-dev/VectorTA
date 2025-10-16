// CUDA kernels for DVDIQQE (Dual Volume Divergence Index QQE-style)
//
// Math pattern: sequential IIR recurrences per parameter row or series.
// We implement:
// - dvdiqqe_batch_f32: one series × many params (each block handles 1 combo)
// - dvdiqqe_many_series_one_param_f32: many series × one param (time-major)
//
// Warmup/NaN semantics match the scalar implementation:
//   warm = first_valid + (2*period - 1);
//   outputs before warm are NaN; at warm index TLs are seeded to dvdi.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

static __forceinline__ __device__ float fin_or(const float v, const float alt) {
    return isfinite(v) ? v : alt;
}

static __forceinline__ __device__ float dvdiqqe_tick_volume(
    const float o, const float c, const float tick, float &tickrng_prev)
{
    const float rng = c - o;
    const float arng = fabsf(rng);
    const float tickrng = (arng < tick) ? tickrng_prev : rng;
    tickrng_prev = tickrng;
    const float tv = fabsf(tickrng) / tick;
    return tv > 0.0f ? tv : 0.0f;
}

// Build selected volume according to Pine-like policy
static __forceinline__ __device__ float select_volume(
    const float vol_opt, const float tick_vol, const int use_tick_only)
{
    if (use_tick_only) return tick_vol;
    if (isfinite(vol_opt)) return vol_opt;
    return tick_vol;
}

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

    const int   period = periods[combo];
    const int   smoothing = smoothings[combo];
    const float fast_mult = fast_mults[combo];
    const float slow_mult = slow_mults[combo];
    if (period <= 0 || smoothing <= 0 || first_valid >= series_len) {
        return;
    }
    const int wper = period * 2 - 1;
    const int warm = first_valid + wper;

    float* dvdi_row  = out_dvdi  + combo * series_len;
    float* fast_row  = out_fast  + combo * series_len;
    float* slow_row  = out_slow  + combo * series_len;
    float* cent_row  = out_center+ combo * series_len;

    // Prefill with NaNs
    const float qnan = nanf("");
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        dvdi_row[i] = qnan; fast_row[i] = qnan; slow_row[i] = qnan; cent_row[i] = qnan;
    }
    __syncthreads();
    if (threadIdx.x != 0) return; // single-lane sequential per row

    // Constants
    const double a_p = 2.0 / (double)(period + 1);
    const double b_p = 1.0 - a_p;
    const double a_s = 2.0 / (double)(smoothing + 1);
    const double b_s = 1.0 - a_s;
    const double a_r = 2.0 / (double)(wper + 1);
    const double b_r = 1.0 - a_r;

    // Rolling vars
    double pvi = 0.0, nvi = 0.0;
    // EMA with running-mean warmup
    double pvi_ema = 0.0, nvi_ema = 0.0; int pvi_cnt = 0;
    double pdiv_ema = 0.0, ndiv_ema = 0.0; int div_cnt = 0;
    double dvdi_prev = 0.0; bool dvdi_inited = false;
    // Double EMA over ranges with running-mean warmups
    double rng_ema1 = 0.0, rng_ema2 = 0.0; int rng1_cnt = 0, rng2_cnt = 0; bool rng2_ready = false;
    double center_sum = 0.0; double center_cnt = 0.0;

    float prev_vol = 0.0f;
    float prev_close = 0.0f;
    float tickrng_prev = tick_size;

    // Main scan
    for (int t = 0; t < series_len; ++t) {
        const float oi = open[t];
        const float ci = close[t];
        if (!isfinite(ci)) {
            continue; // keep NaNs; EMA update waits for finite values
        }

        // Build selected volume and PVI/NVI deltas
        const float tick_vol = dvdiqqe_tick_volume(oi, ci, tick_size, tickrng_prev);
        const float real_vol = has_volume ? volume[t] : NAN;
        const float sel_vol = select_volume(real_vol, tick_vol, /*use_tick_only*/ 0);

        if (t == 0) {
            prev_close = ci; prev_vol = sel_vol;
        }

        if (sel_vol > prev_vol) { pvi += (double)(ci - prev_close); }
        if (sel_vol < prev_vol) { nvi -= (double)(ci - prev_close); }
        prev_close = ci; prev_vol = sel_vol;

        // EMA(pvi), EMA(nvi) with running-mean warmup across `period`
        if (t >= first_valid) {
            if (pvi_cnt < period) {
                // running mean
                pvi_cnt += 1;
                pvi_ema = ((double)(pvi_cnt - 1) * pvi_ema + pvi) / (double)pvi_cnt;
                nvi_ema = ((double)(pvi_cnt - 1) * nvi_ema + nvi) / (double)pvi_cnt;
            } else {
                pvi_ema = a_p * pvi + b_p * pvi_ema;
                nvi_ema = a_p * nvi + b_p * nvi_ema;
            }
        }

        // divergences
        const double pdiv = pvi - pvi_ema;
        const double ndiv = nvi - nvi_ema;

        // EMA(divergences) with running-mean warmup across `smoothing`
        if (t >= first_valid) {
            if (div_cnt < smoothing) {
                div_cnt += 1;
                pdiv_ema = ((double)(div_cnt - 1) * pdiv_ema + pdiv) / (double)div_cnt;
                ndiv_ema = ((double)(div_cnt - 1) * ndiv_ema + ndiv) / (double)div_cnt;
            } else {
                pdiv_ema = a_s * pdiv + b_s * pdiv_ema;
                ndiv_ema = a_s * ndiv + b_s * ndiv_ema;
            }
        }

        const double dv = pdiv_ema - ndiv_ema;
        dvdi_row[t] = (float)dv; // will be NaN-overwritten for warmup later

        // Range EMAs over |Δdvdi|
        if (!dvdi_inited) { dvdi_prev = dv; dvdi_inited = true; }
        const double abs_delta = fabs(dv - dvdi_prev);
        if (t >= first_valid + 1) {
            if (rng1_cnt < wper) {
                rng1_cnt += 1; // running mean warmup for first EMA
                rng_ema1 = ((double)(rng1_cnt - 1) * rng_ema1 + abs_delta) / (double)rng1_cnt;
            } else {
                rng_ema1 = a_r * abs_delta + b_r * rng_ema1;
            }
            // feed into second EMA with its own running-mean warmup
            if (!rng2_ready) { rng2_cnt = 0; rng2_ready = true; }
            if (rng2_cnt < wper) {
                rng2_cnt += 1;
                rng_ema2 = ((double)(rng2_cnt - 1) * rng_ema2 + rng_ema1) / (double)rng2_cnt;
            } else {
                rng_ema2 = a_r * rng_ema1 + b_r * rng_ema2;
            }
        }

        // Trailing levels
        if (t == warm && rng2_cnt >= 1) {
            fast_row[t] = (float)dv; slow_row[t] = (float)dv;
        } else if (t > warm && rng2_cnt >= wper) {
            const double fr = rng_ema2 * (double)fast_mult;
            const double sr = rng_ema2 * (double)slow_mult;
            // fast TL
            const double prev_fast = (double)fast_row[t - 1];
            if (dv > prev_fast) {
                double nv = dv - fr; fast_row[t] = (float)((nv < prev_fast) ? prev_fast : nv);
            } else {
                double nv = dv + fr; fast_row[t] = (float)((nv > prev_fast) ? prev_fast : nv);
            }
            // slow TL
            const double prev_slow = (double)slow_row[t - 1];
            if (dv > prev_slow) {
                double nv = dv - sr; slow_row[t] = (float)((nv < prev_slow) ? prev_slow : nv);
            } else {
                double nv = dv + sr; slow_row[t] = (float)((nv > prev_slow) ? prev_slow : nv);
            }
        }

        // Center line
        if (t >= warm) {
            if (center_dynamic) {
                if (isfinite((float)dv)) { center_sum += dv; center_cnt += 1.0; }
                cent_row[t] = (center_cnt > 0.0) ? (float)(center_sum / center_cnt) : qnan;
            } else {
                cent_row[t] = 0.0f;
            }
        }

        dvdi_prev = dv;
    }

    // Enforce warmup NaNs on dvdi and TLs
    for (int i = 0; i < series_len && i < warm; ++i) {
        dvdi_row[i] = qnan; fast_row[i] = qnan; slow_row[i] = qnan;
    }
}

// Many-series × one param kernel (time-major)
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
    const double a_p = 2.0 / (double)(period + 1);
    const double b_p = 1.0 - a_p;
    const double a_s = 2.0 / (double)(smoothing + 1);
    const double b_s = 1.0 - a_s;
    const double a_r = 2.0 / (double)(wper + 1);
    const double b_r = 1.0 - a_r;

    const int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep       = gridDim.x * warps_per_block;

    const int cols = num_series;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];
        const int warm = first_valid + wper;
        // plane pointers (time-major)
        float* dvdi_plane  = dvdi_tm;
        float* fast_plane  = fast_tm;
        float* slow_plane  = slow_tm;
        float* cent_plane  = center_tm;

        // Initialize column to NaN cooperatively
        const float qnan = nanf("");
        for (int t = lane; t < series_len; t += warpSize) {
            const int idx = t * cols + s;
            dvdi_plane[idx] = qnan; fast_plane[idx] = qnan; slow_plane[idx] = qnan; cent_plane[idx] = qnan;
        }
        if (lane != 0) continue;
        if (first_valid < 0 || first_valid >= series_len) continue;

        // Rolling state
        double pvi = 0.0, nvi = 0.0;
        double pvi_ema = 0.0, nvi_ema = 0.0; int pvi_cnt = 0;
        double pdiv_ema = 0.0, ndiv_ema = 0.0; int div_cnt = 0;
        double dvdi_prev = 0.0; bool dvdi_inited = false;
        double rng_ema1 = 0.0, rng_ema2 = 0.0; int rng1_cnt = 0, rng2_cnt = 0; bool rng2_ready = false;
        double center_sum = 0.0; double center_cnt = 0.0;

        float prev_vol = 0.0f;
        float prev_close = 0.0f;
        float tickrng_prev = tick_size;

        for (int t = 0; t < series_len; ++t) {
            const int idx = t * cols + s;
            const float oi = open_tm[idx];
            const float ci = close_tm[idx];
            if (!isfinite(ci)) continue;

            const float tick_vol = dvdiqqe_tick_volume(oi, ci, tick_size, tickrng_prev);
            const float vol_tm = has_volume ? volume_tm[idx] : NAN;
            const float sel_vol = select_volume(vol_tm, tick_vol, /*use_tick_only*/ 0);
            if (t == 0) { prev_close = ci; prev_vol = sel_vol; }

            if (sel_vol > prev_vol) { pvi += (double)(ci - prev_close); }
            if (sel_vol < prev_vol) { nvi -= (double)(ci - prev_close); }
            prev_close = ci; prev_vol = sel_vol;

            if (t >= first_valid) {
                if (pvi_cnt < period) { pvi_cnt += 1; pvi_ema = ((double)(pvi_cnt - 1) * pvi_ema + pvi) / (double)pvi_cnt; nvi_ema = ((double)(pvi_cnt - 1) * nvi_ema + nvi) / (double)pvi_cnt; }
                else { pvi_ema = a_p * pvi + b_p * pvi_ema; nvi_ema = a_p * nvi + b_p * nvi_ema; }
            }
            const double pdiv = pvi - pvi_ema;
            const double ndiv = nvi - nvi_ema;
            if (t >= first_valid) {
                if (div_cnt < smoothing) { div_cnt += 1; pdiv_ema = ((double)(div_cnt - 1) * pdiv_ema + pdiv) / (double)div_cnt; ndiv_ema = ((double)(div_cnt - 1) * ndiv_ema + ndiv) / (double)div_cnt; }
                else { pdiv_ema = a_s * pdiv + b_s * pdiv_ema; ndiv_ema = a_s * ndiv + b_s * ndiv_ema; }
            }

            const double dv = pdiv_ema - ndiv_ema;
            dvdi_plane[idx] = (float)dv;

            if (!dvdi_inited) { dvdi_prev = dv; dvdi_inited = true; }
            const double abs_delta = fabs(dv - dvdi_prev);
            if (t >= first_valid + 1) {
                if (rng1_cnt < wper) { rng1_cnt += 1; rng_ema1 = ((double)(rng1_cnt - 1) * rng_ema1 + abs_delta) / (double)rng1_cnt; }
                else { rng_ema1 = a_r * abs_delta + b_r * rng_ema1; }
                if (!rng2_ready) { rng2_cnt = 0; rng2_ready = true; }
                if (rng2_cnt < wper) { rng2_cnt += 1; rng_ema2 = ((double)(rng2_cnt - 1) * rng_ema2 + rng_ema1) / (double)rng2_cnt; }
                else { rng_ema2 = a_r * rng_ema1 + b_r * rng_ema2; }
            }

            if (t == warm && rng2_cnt >= 1) {
                fast_plane[idx] = (float)dv; slow_plane[idx] = (float)dv;
            } else if (t > warm && rng2_cnt >= wper) {
                const double fr = rng_ema2 * (double)fast_mult;
                const double sr = rng_ema2 * (double)slow_mult;
                const double prev_fast = (double)fast_plane[(t - 1) * cols + s];
                if (dv > prev_fast) {
                    double nv = dv - fr; fast_plane[idx] = (float)((nv < prev_fast) ? prev_fast : nv);
                } else {
                    double nv = dv + fr; fast_plane[idx] = (float)((nv > prev_fast) ? prev_fast : nv);
                }
                const double prev_slow = (double)slow_plane[(t - 1) * cols + s];
                if (dv > prev_slow) {
                    double nv = dv - sr; slow_plane[idx] = (float)((nv < prev_slow) ? prev_slow : nv);
                } else {
                    double nv = dv + sr; slow_plane[idx] = (float)((nv > prev_slow) ? prev_slow : nv);
                }
            }

            if (t >= warm) {
                if (center_dynamic) {
                    if (isfinite((float)dv)) { center_sum += dv; center_cnt += 1.0; }
                    cent_plane[idx] = (center_cnt > 0.0) ? (float)(center_sum / center_cnt) : qnan;
                } else {
                    cent_plane[idx] = 0.0f;
                }
            }

            dvdi_prev = dv;
        }
        // Enforce warmup NaNs
        for (int t = 0; t < series_len && t < warm; ++t) {
            const int idx = t * cols + s; dvdi_plane[idx] = qnan; fast_plane[idx] = qnan; slow_plane[idx] = qnan;
        }
    }
}
