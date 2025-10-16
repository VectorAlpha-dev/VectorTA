// CUDA kernel for the CCI Cycle indicator (one-series × many-params and many-series × one-param).
//
// Design: Recurrence/IIR per thread (row). We intentionally use the naive O(L)
// scans for the two stochastic windows and for the CCI mean absolute deviation
// to match the scalar semantics exactly without complicated shared structures.
// Given typical small L (<= ~64), this maps well to GPU throughput when many
// rows (parameter combinations) run in parallel.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace { __device__ inline bool is_finitef(float x) { return !isnan(x) && !isinf(x); } }

// One series × many params. Row-major output (rows=n_combos, cols=len)
extern "C" __global__ void cci_cycle_batch_f32(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int n_combos,
    const int* __restrict__ lengths,
    const float* __restrict__ factors,
    float* __restrict__ out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int row = tid; row < n_combos; row += stride) {
        const int L = lengths[row];
        const float factor = factors[row];
        float* row_out = out + static_cast<size_t>(row) * len;

        // Prefill with NaNs
        for (int i = 0; i < len; ++i) {
            row_out[i] = CUDART_NAN_F;
        }

        if (L <= 0 || L > len) continue;
        const int needed = L * 2; // match scalar guard
        if (len - first_valid < needed) continue;

        const float invL = 1.0f / (float)L;
        const int half = (L + 1) / 2;
        const float alpha_s = 2.0f / (half + 1.0f);
        const float beta_s = 1.0f - alpha_s;
        const float alpha_l = 2.0f / (L + 1.0f);
        const float beta_l = 1.0f - alpha_l;
        const int smma_p = max(1, (int)llroundf(sqrtf((float)L)));

        // ---- Initial SMA and MAD for CCI at index first_valid + L - 1 ----
        int i0 = first_valid;
        int i1 = first_valid + L; // exclusive
        float sum = 0.0f;
        for (int i = i0; i < i1; ++i) sum += prices[i];
        float sma = sum * invL;
        float sum_abs = 0.0f;
        for (int i = i0; i < i1; ++i) sum_abs += fabsf(prices[i] - sma);
        int out_start = first_valid + L - 1;
        float denom = 0.015f * (sum_abs * invL);
        float cci = (denom == 0.0f) ? 0.0f : ((prices[out_start] - sma) / denom);

        // EMA(short/long) states over CCI
        float ema_s = cci;
        float ema_l = cci;

        // SMMA state over double-EMA
        float smma = CUDART_NAN_F;
        float smma_sum = 0.0f;
        int smma_count = 0;
        bool smma_inited = false;

        // First stochastic stage smoothing state
        float prev_f1 = CUDART_NAN_F;
        float prev_pf = CUDART_NAN_F; // smoothed f1

        // Second stochastic stage smoothing state
        float prev_out = CUDART_NAN_F;

        // Local rings for ccis and pf
        const int MAX_L = 2048; // generous upper bound for L
        if (L > MAX_L) continue; // safety: unsupported oversized length
        float ccis_ring[2048];
        float pf_ring[2048];
        int ccis_valid = 0;
        int pf_valid = 0;

        // Prime indices < out_start remain NaN in row_out.
        for (int i = out_start; i < len; ++i) {
            // Rolling SMA for CCI
            const float entering = prices[i];
            const float exiting = prices[i - L];
            sum = sum - exiting + entering;
            sma = sum * invL;
            // Re-scan window for MAD
            float sabs = 0.0f;
            const int wstart = i + 1 - L;
            for (int k = wstart; k <= i; ++k) {
                sabs += fabsf(prices[k] - sma);
            }
            denom = 0.015f * (sabs * invL);
            cci = (denom == 0.0f) ? 0.0f : ((entering - sma) / denom);

            // EMA short/long on CCI
            ema_s = fmaf(beta_s, ema_s, alpha_s * cci);
            ema_l = fmaf(beta_l, ema_l, alpha_l * cci);
            const float de = ema_s + ema_s - ema_l; // double-EMA

            // SMMA (Wilder RMA) on double-EMA
            if (!smma_inited) {
                if (is_finitef(de)) {
                    smma_sum += de;
                    smma_count += 1;
                    if (smma_count >= smma_p) {
                        smma = smma_sum / (float)smma_p;
                        smma_inited = true;
                    }
                }
            } else {
                smma = (smma * (smma_p - 1) + de) / (float)smma_p;
            }

            // Maintain ccis ring buffer
            int pos = i % L;
            ccis_ring[pos] = smma; // may be NaN before smma_inited
            if (ccis_valid < L) ccis_valid++;

            // First stochastic on ccis (smma) over last L
            float pf = CUDART_NAN_F;
            if (i >= first_valid + L - 1) {
                // compute window min/max over available ccis
                int have = min(ccis_valid, L);
                float mn1 = CUDART_INF_F;
                float mx1 = -CUDART_INF_F;
                for (int k = 0; k < have; ++k) {
                    int idx = (i - k) % L;
                    float v = ccis_ring[idx];
                    if (is_finitef(v)) {
                        if (v < mn1) mn1 = v;
                        if (v > mx1) mx1 = v;
                    }
                }
                if (is_finitef(mn1) && is_finitef(mx1)) {
                    float range = mx1 - mn1;
                    float cur_f1 = 50.0f;
                    if (range > 0.0f && is_finitef(smma)) {
                        cur_f1 = ((smma - mn1) / range) * 100.0f;
                    } else {
                        cur_f1 = isnan(prev_f1) ? 50.0f : prev_f1;
                    }
                    pf = (isnan(prev_pf) || factor == 0.0f) ? cur_f1
                                                             : fmaf((cur_f1 - prev_pf), factor, prev_pf);
                    prev_f1 = cur_f1;
                    prev_pf = pf;
                }
            }

            // Maintain pf ring buffer
            pf_ring[pos] = pf; // may be NaN
            if (pf_valid < L) pf_valid++;

            // Second stochastic on pf over last L -> final out
            float out_i = CUDART_NAN_F;
            if (i >= first_valid + L - 1) {
                int have = min(pf_valid, L);
                float mn2 = CUDART_INF_F;
                float mx2 = -CUDART_INF_F;
                for (int k = 0; k < have; ++k) {
                    int idx = (i - k) % L;
                    float v = pf_ring[idx];
                    if (is_finitef(v)) {
                        if (v < mn2) mn2 = v;
                        if (v > mx2) mx2 = v;
                    }
                }
                if (is_finitef(mn2) && is_finitef(mx2)) {
                    float range = mx2 - mn2;
                    if (range > 0.0f && is_finitef(pf)) {
                        float f2 = ((pf - mn2) / range) * 100.0f;
                        out_i = (isnan(prev_out) || factor == 0.0f)
                                    ? f2
                                    : fmaf((f2 - prev_out), factor, prev_out);
                        prev_out = out_i;
                    } else {
                        out_i = isnan(prev_out) ? 50.0f : prev_out;
                        prev_out = out_i;
                    }
                }
            }

            row_out[i] = out_i;
        }
    }
}

// Many-series × one-param (time-major). One thread per series (row), sequential in time.
extern "C" __global__ void cci_cycle_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    int length,
    float factor,
    float* __restrict__ out_tm
) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x; // series id
    if (rid >= rows) return;

    const int L = length;
    const float invL = 1.0f / (float)L;
    const int half = (L + 1) / 2;
    const float alpha_s = 2.0f / (half + 1.0f);
    const float beta_s = 1.0f - alpha_s;
    const float alpha_l = 2.0f / (L + 1.0f);
    const float beta_l = 1.0f - alpha_l;
    const int smma_p = max(1, (int)llroundf(sqrtf((float)L)));

    const float* prices = data_tm + (size_t)rid * cols;
    float* out_row = out_tm + (size_t)rid * cols;
    // Prefill with NaNs
    for (int i = 0; i < cols; ++i) out_row[i] = CUDART_NAN_F;

    int first_valid = first_valids[rid];
    if (first_valid < 0) first_valid = 0;
    if (L <= 0 || L > cols) return;
    if (cols - first_valid < L * 2) return;

    // Initial window
    int i0 = first_valid;
    int i1 = first_valid + L;
    float sum = 0.0f;
    for (int i = i0; i < i1; ++i) sum += prices[i];
    float sma = sum * invL;
    float sum_abs = 0.0f;
    for (int i = i0; i < i1; ++i) sum_abs += fabsf(prices[i] - sma);
    int out_start = first_valid + L - 1;
    float denom = 0.015f * (sum_abs * invL);
    float cci = (denom == 0.0f) ? 0.0f : ((prices[out_start] - sma) / denom);

    float ema_s = cci, ema_l = cci;
    float smma = CUDART_NAN_F, smma_sum = 0.0f; int smma_count = 0; bool smma_inited = false;
    float prev_f1 = CUDART_NAN_F, prev_pf = CUDART_NAN_F, prev_out = CUDART_NAN_F;

    const int MAX_L = 2048; if (L > MAX_L) return;
    float ccis_ring[2048]; int ccis_valid = 0;
    float pf_ring[2048]; int pf_valid = 0;

    for (int i = out_start; i < cols; ++i) {
        // rolling SMA + MAD
        float entering = prices[i];
        float exiting = prices[i - L];
        sum = sum - exiting + entering;
        sma = sum * invL;
        float sabs = 0.0f; int wstart = i + 1 - L;
        for (int k = wstart; k <= i; ++k) sabs += fabsf(prices[k] - sma);
        denom = 0.015f * (sabs * invL);
        cci = (denom == 0.0f) ? 0.0f : ((entering - sma) / denom);

        // ema short/long
        ema_s = fmaf(beta_s, ema_s, alpha_s * cci);
        ema_l = fmaf(beta_l, ema_l, alpha_l * cci);
        float de = ema_s + ema_s - ema_l;

        if (!smma_inited) {
            if (is_finitef(de)) { smma_sum += de; smma_count += 1; if (smma_count >= smma_p) { smma = smma_sum / (float)smma_p; smma_inited = true; } }
        } else { smma = (smma * (smma_p - 1) + de) / (float)smma_p; }

        // maintain ccis ring
        int pos = i % L; ccis_ring[pos] = smma; if (ccis_valid < L) ccis_valid++;

        // first stochastic
        float pf = CUDART_NAN_F;
        float mn1 = CUDART_INF_F, mx1 = -CUDART_INF_F; int have1 = min(ccis_valid, L);
        for (int k = 0; k < have1; ++k) { int idx = (i - k) % L; float v = ccis_ring[idx]; if (is_finitef(v)) { if (v < mn1) mn1 = v; if (v > mx1) mx1 = v; } }
        if (is_finitef(mn1) && is_finitef(mx1)) {
            float range = mx1 - mn1;
            float cur_f1 = 50.0f;
            if (range > 0.0f && is_finitef(smma)) cur_f1 = ((smma - mn1) / range) * 100.0f; else cur_f1 = isnan(prev_f1) ? 50.0f : prev_f1;
            pf = (isnan(prev_pf) || factor == 0.0f) ? cur_f1 : fmaf((cur_f1 - prev_pf), factor, prev_pf);
            prev_f1 = cur_f1; prev_pf = pf;
        }

        // maintain pf ring
        pf_ring[pos] = pf; if (pf_valid < L) pf_valid++;

        // second stochastic
        float out_i = CUDART_NAN_F; int have2 = min(pf_valid, L); float mn2 = CUDART_INF_F, mx2 = -CUDART_INF_F;
        for (int k = 0; k < have2; ++k) { int idx = (i - k) % L; float v = pf_ring[idx]; if (is_finitef(v)) { if (v < mn2) mn2 = v; if (v > mx2) mx2 = v; } }
        if (is_finitef(mn2) && is_finitef(mx2)) {
            float range = mx2 - mn2;
            if (range > 0.0f && is_finitef(pf)) {
                float f2 = ((pf - mn2) / range) * 100.0f;
                out_i = (isnan(prev_out) || factor == 0.0f) ? f2 : fmaf((f2 - prev_out), factor, prev_out);
                prev_out = out_i;
            } else {
                out_i = isnan(prev_out) ? 50.0f : prev_out;
                prev_out = out_i;
            }
        }
        out_row[i] = out_i;
    }
}
