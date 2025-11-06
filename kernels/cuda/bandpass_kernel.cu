// CUDA kernels for the Band-Pass indicator (Ehlers-style two-stage filter).
//
// This rewrite removes FP64 from hot paths, fuses normalization/trigger/signal
// into a single streaming pass, and seeds the IIR from the first finite
// triplet of HP samples. It keeps time-major accesses coalesced.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>

// Utility: check finite (float)
static __forceinline__ __device__ bool is_finite_f32(float x) {
    return !isnan(x) && !isinf(x);
}

// High-pass coefficients helper (FP32) using sincospif (sin/cos of π·x).
// theta = 2π/period  ==>  sincospi(2/period)
static __forceinline__ __device__ void hpf_coeffs_from_period_f32(int period,
                                                                  float &c_out,
                                                                  float &oma_out,
                                                                  bool  &ok) {
    ok = false;
    if (period <= 0) return;
    float s, co;
    // computes sinf(π*x) and cosf(π*x) accurately in single precision
    sincospif(2.0f / static_cast<float>(period), &s, &co);  // π * (2/period)
    if (fabsf(co) < 1e-7f) return; // avoid division blow-up for tiny cos
    const float alpha = 1.0f + ((s - 1.0f) / co);
    c_out   = 1.0f - 0.5f * alpha;   // (1 - α/2)
    oma_out = 1.0f - alpha;          // (1 - α)
    ok = true;
}

// -----------------------------------------
// Batch kernel: one-series × many-params
// Consumes precomputed HP rows (row-major).
// -----------------------------------------
extern "C" __global__ __launch_bounds__(256, 2)
void bandpass_batch_from_hp_f32(
    const float* __restrict__ hp,          // shape: hp_rows × len
    int hp_rows,
    int len,
    const int*   __restrict__ hp_row_idx,  // per-combo index into hp rows
    const float* __restrict__ alphas,      // per-combo alpha
    const float* __restrict__ betas,       // per-combo beta (cos term)
    const int*   __restrict__ trig_periods,// per-combo trigger HP period
    int n_combos,
    float* __restrict__ out_bp,            // shape: n_combos × len
    float* __restrict__ out_bpn,           // shape: n_combos × len
    float* __restrict__ out_sig,           // shape: n_combos × len
    float* __restrict__ out_trg            // shape: n_combos × len
) {
    const int row0   = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = row0; row < n_combos; row += stride) {
        const int hp_idx = hp_row_idx[row];
        if (hp_idx < 0 || hp_idx >= hp_rows) continue;

        const float* __restrict__ hp_row = hp + static_cast<size_t>(hp_idx) * len;
        float* __restrict__ bp_row   = out_bp  ? out_bp  + static_cast<size_t>(row) * len : nullptr;
        float* __restrict__ bpn_row  = out_bpn ? out_bpn + static_cast<size_t>(row) * len : nullptr;
        float* __restrict__ sig_row  = out_sig ? out_sig + static_cast<size_t>(row) * len : nullptr;
        float* __restrict__ trg_row  = out_trg ? out_trg + static_cast<size_t>(row) * len : nullptr;

        // Band-pass recurrence coefficients (FP32).
        const float alpha = alphas[row];
        const float beta  = betas[row];
        const float a = 0.5f * (1.0f - alpha);
        const float c = beta * (1.0f + alpha);
        const float d = -alpha;

        // Trigger HPF coefficients (FP32).
        float hc = 0.0f, homa = 0.0f; bool ok_hp;
        hpf_coeffs_from_period_f32(trig_periods[row], hc, homa, ok_hp);

        // Find start index where hp[i],hp[i-1],hp[i-2] are all finite (robust IIR seeding).
        int start = 2;
        for (; start < len; ++start) {
            const float x2 = hp_row[start];
            const float x1 = hp_row[start - 1];
            const float x0 = hp_row[start - 2];
            if (is_finite_f32(x2) && is_finite_f32(x1) && is_finite_f32(x0)) break;
        }
        const int warm_bp = min(start, len);

        // Fill warm-up region with NaNs
        for (int i = 0; i < warm_bp; ++i) {
            if (bp_row)  bp_row[i]  = CUDART_NAN_F;
            if (bpn_row) bpn_row[i] = CUDART_NAN_F;
            if (trg_row) trg_row[i] = CUDART_NAN_F;
            if (sig_row) sig_row[i] = CUDART_NAN_F;
        }
        if (warm_bp >= len) continue;

        // Initialize recurrence state from hp[start-2], hp[start-1].
        float y_im2 = hp_row[start - 2];
        float y_im1 = hp_row[start - 1];

        // Peak-normalization and trigger state.
        constexpr float K = 0.991f;
        float peak   = 0.0f;
        float prev_x = 0.0f, prev_y = 0.0f;
        bool  trig_init = false;

        // Main pass: compute bp, bpn, trg, sig in a single streaming loop.
        #pragma unroll 4
        for (int i = start; i < len; ++i) {
            const float hi   = hp_row[i];
            const float him2 = hp_row[i - 2];

            // 2nd-order band-pass IIR:
            // y = d*y[i-2] + c*y[i-1] + a*(hp[i] - hp[i-2])
            float y = __fmaf_rn(d, y_im2, __fmaf_rn(c, y_im1, a * (hi - him2)));

            if (bp_row) bp_row[i] = y;

            // Normalization (decaying peak envelope)
            peak = K * peak;
            const float av = fabsf(y);
            if (av > peak) peak = av;
            const float inv_peak = (peak > 0.0f) ? (1.0f / peak) : 0.0f;
            const float bn = y * inv_peak;
            if (bpn_row) bpn_row[i] = bn;

            // Trigger HPF on normalized series
            float tr_val = CUDART_NAN_F;
            if (ok_hp) {
                if (!trig_init) {
                    prev_x = bn;
                    prev_y = bn;
                    trig_init = true;
                    tr_val = bn;
                } else {
                    // y = (1-α)*y_prev + c*(x - x_prev)
                    prev_y = __fmaf_rn(homa, prev_y, hc * (bn - prev_x));
                    prev_x = bn;
                    tr_val = prev_y;
                }
            }
            if (trg_row) trg_row[i] = tr_val;

            // Signal: compare bpn vs trigger
            if (sig_row) {
                float s = 0.0f;
                if (is_finite_f32(tr_val)) {
                    s = (bn < tr_val) ? 1.0f : ((bn > tr_val) ? -1.0f : 0.0f);
                }
                sig_row[i] = s;
            }

            // Advance IIR state
            y_im2 = y_im1;
            y_im1 = y;
        }
    }
}

// ------------------------------------------------------
// Many-series × one-param, time-major HP input.
// hp_tm is time-major: idx = t*cols + s
// ------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 2)
void bandpass_many_series_one_param_time_major_from_hp_f32(
    const float* __restrict__ hp_tm, // shape: rows × cols, time-major
    int cols,                        // number of series (columns)
    int rows,                        // series length (rows)
    float alpha_f,
    float beta_f,
    int trig_period,
    float* __restrict__ out_bp_tm,
    float* __restrict__ out_bpn_tm,
    float* __restrict__ out_sig_tm,
    float* __restrict__ out_trg_tm
) {
    if (cols <= 0 || rows <= 0) return;

    const float a = 0.5f * (1.0f - alpha_f);
    const float c = beta_f * (1.0f + alpha_f);
    const float d = -alpha_f;

    // Trigger coefficients (FP32)
    float hc = 0.0f, homa = 0.0f; bool ok_hp;
    hpf_coeffs_from_period_f32(trig_period, hc, homa, ok_hp);

    const int tpb = blockDim.x * gridDim.x;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += tpb) {
        auto at      = [&](const float* base, int t) -> float { return base[static_cast<size_t>(t) * cols + s]; };
        auto out_ref = [&](float* base, int t) -> float& { return base[static_cast<size_t>(t) * cols + s]; };

        // Find first finite triplet for this column (robust seed).
        int start = 2;
        for (; start < rows; ++start) {
            if (is_finite_f32(at(hp_tm, start)) &&
                is_finite_f32(at(hp_tm, start - 1)) &&
                is_finite_f32(at(hp_tm, start - 2))) break;
        }
        const int warm_bp = min(start, rows);

        // Warm-up region -> NaN
        for (int t = 0; t < warm_bp; ++t) {
            if (out_bp_tm)   out_ref(out_bp_tm,  t) = CUDART_NAN_F;
            if (out_bpn_tm)  out_ref(out_bpn_tm, t) = CUDART_NAN_F;
            if (out_trg_tm)  out_ref(out_trg_tm, t) = CUDART_NAN_F;
            if (out_sig_tm)  out_ref(out_sig_tm, t) = CUDART_NAN_F;
        }
        if (warm_bp >= rows) continue;

        // IIR seeds
        float y_im2 = at(hp_tm, warm_bp - 2);
        float y_im1 = at(hp_tm, warm_bp - 1);

        // Envelope + trigger state
        constexpr float K = 0.991f;
        float peak   = 0.0f;
        float prev_x = 0.0f, prev_y = 0.0f;
        bool  trig_init = false;

        // Main pass over time (coalesced across the warp at each t)
        #pragma unroll 4
        for (int t = warm_bp; t < rows; ++t) {
            const float hi   = at(hp_tm, t);
            const float him2 = at(hp_tm, t - 2);

            float y = __fmaf_rn(d, y_im2, __fmaf_rn(c, y_im1, a * (hi - him2)));
            if (out_bp_tm) out_ref(out_bp_tm, t) = y;

            // Normalize
            peak = K * peak;
            const float av = fabsf(y);
            if (av > peak) peak = av;
            const float inv_peak = (peak > 0.0f) ? (1.0f / peak) : 0.0f;
            const float bn = y * inv_peak;
            if (out_bpn_tm) out_ref(out_bpn_tm, t) = bn;

            // Trigger
            float tr_val = CUDART_NAN_F;
            if (ok_hp) {
                if (!trig_init) { prev_x = bn; prev_y = bn; trig_init = true; tr_val = bn; }
                else { prev_y = __fmaf_rn(homa, prev_y, hc * (bn - prev_x)); prev_x = bn; tr_val = prev_y; }
            }
            if (out_trg_tm) out_ref(out_trg_tm, t) = tr_val;

            // Signal
            if (out_sig_tm) {
                float sgn = 0.0f;
                if (is_finite_f32(tr_val)) {
                    sgn = (bn < tr_val) ? 1.0f : ((bn > tr_val) ? -1.0f : 0.0f);
                }
                out_ref(out_sig_tm, t) = sgn;
            }

            y_im2 = y_im1;
            y_im1 = y;
        }
    }
}

