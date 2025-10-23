// CUDA kernels for Low Pass Channel (LPC) — FP32 + optional alpha LUT + coalesced stores
#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

static __forceinline__ __device__ bool finite_f(float x) { return isfinite(x); }

// ---- FP32 alpha from period (device) ----
static __forceinline__ __device__ float alpha_from_period_iir_f(int p) {
    if (p < 1) p = 1;
    const float omega = 2.0f * CUDART_PI_F / (float)p;
    float s, c;
    // CUDA provides fast, accurate sincosf()
    sincosf(omega, &s, &c);
    return (1.0f - s) / c;
}

static __forceinline__ __device__ float lut_or_formula_alpha(
    int p, const float* __restrict__ alpha_lut, int lut_len, int lut_pmin)
{
    if (p < lut_pmin) p = lut_pmin;
    if (alpha_lut) {
        int idx = p - lut_pmin;
        if (idx < 0) idx = 0;
        if (idx >= lut_len) idx = lut_len - 1;
        return alpha_lut[idx];
    }
    return alpha_from_period_iir_f(p);
}

// =============================
// One-series × many-params (v2)
// =============================
extern "C" __global__ void lpc_batch_f32_v2(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ src,
    int len,
    const float* __restrict__ tr_opt,      // optional (len) or nullptr
    const int*   __restrict__ fixed_periods,
    const float* __restrict__ cycle_mults,
    const float* __restrict__ tr_mults,
    int n_combos,
    int first_valid,
    int cutoff_mode,                        // 0=fixed, 1=adaptive (uses dom)
    int max_cycle_limit,                    // <=0: no cap
    const float* __restrict__ dom,          // len when adaptive; may be nullptr for fixed
    // ---- new (optional) alpha LUT ----
    const float* __restrict__ alpha_lut,    // may be nullptr (falls back to sincosf)
    int alpha_lut_len,                      // number of entries in LUT
    int alpha_lut_pmin,                     // period corresponding to lut[0]
    // ---- new: output layout selector ----
    int out_time_major,                     // 0=row-major [combo,len], 1=time-major [len,combo]
    float* __restrict__ out_filter,
    float* __restrict__ out_high,
    float* __restrict__ out_low
){
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const uint32_t qnan_bits = 0x7fc00000u;
    const float qnan = __int_as_float(qnan_bits);

    for (int combo = tid; combo < n_combos; combo += stride) {
        // Accessors for the two output layouts (branchless indexing)
        auto store_triplet = [&](int i, float f, float hi, float lo) {
            size_t idx = out_time_major ? (size_t)i * (size_t)n_combos + (size_t)combo
                                        : (size_t)combo * (size_t)len    + (size_t)i;
            out_filter[idx] = f;
            out_high[idx]   = hi;
            out_low[idx]    = lo;
        };

        // Warm-up NaNs for [0, first_valid)
        if (first_valid > 0) {
            const int upto = first_valid < len ? first_valid : len;
            for (int i = 0; i < upto; ++i) store_triplet(i, qnan, qnan, qnan);
            if (first_valid >= len) continue;
        }

        // Per-combo scalars
        const float tm       = tr_mults[combo];
        const int   p_fixed  = fixed_periods[combo];
        const float cm       = cycle_mults[combo];
        const bool  adaptive = (cutoff_mode != 0) && (dom != nullptr);

        // Seed at first_valid
        const int i0 = first_valid;
        float s_prev = src[i0];
        float f_prev = s_prev;

        // TR seed (precomputed or raw Wilder TR)
        float tr_prev = tr_opt ? tr_opt[i0] : (high[i0] - low[i0]);
        float ftr_prev = tr_prev;

        // Initial alpha (cached)
        int last_p = adaptive ? 0 : p_fixed;
        float alpha = lut_or_formula_alpha(p_fixed, alpha_lut, alpha_lut_len, alpha_lut_pmin);

        // Emit seed
        store_triplet(i0, f_prev, f_prev + tm * tr_prev, f_prev - tm * tr_prev);

        // Main loop (time-sequential)
        #pragma unroll 1
        for (int i = i0 + 1; i < len; ++i) {
            // --- period & alpha ---
            int p_i = p_fixed;
            if (adaptive) {
                float base = dom[i];
                if (!finite_f(base)) {
                    p_i = p_fixed;
                } else {
                    float pd = nearbyintf(base * cm);
                    if (pd < 3.0f) pd = 3.0f;
                    if (max_cycle_limit > 0 && pd > (float)max_cycle_limit) pd = (float)max_cycle_limit;
                    p_i = (int)pd;
                }
            }
            if (p_i != last_p) {
                alpha  = lut_or_formula_alpha(p_i, alpha_lut, alpha_lut_len, alpha_lut_pmin);
                last_p = p_i;
            }
            const float one_m_a = 1.0f - alpha;
            const float w = 0.5f * one_m_a;

            // --- filtered source ---
            const float s_i = src[i];
            // f_i = alpha * f_prev + 0.5*(1-alpha)*(s_i + s_prev)
            const float f_i = fmaf(alpha, f_prev, w * (s_i + s_prev));
            s_prev = s_i;
            f_prev = f_i;

            // --- Wilder TR (raw or precomputed) ---
            float tr_i;
            if (tr_opt) {
                tr_i = tr_opt[i];
            } else {
                const float hl  = high[i] - low[i];
                const float c_l = fabsf(close[i] - low[i - 1]);
                const float c_h = fabsf(close[i] - high[i - 1]);
                tr_i = fmaxf(hl, fmaxf(c_l, c_h));
            }
            const float ftr_i = fmaf(alpha, ftr_prev, w * (tr_i + tr_prev));
            tr_prev  = tr_i;
            ftr_prev = ftr_i;

            // --- channel ---
            const float hi = f_i + tm * ftr_i;
            const float lo = f_i - tm * ftr_i;
            store_triplet(i, f_i, hi, lo);
        }
    }
}

// Back-compat batch kernel: preserve original symbol and signature.
// Keep original FP64 accumulator semantics to satisfy existing unit tests,
// while the optimized FP32 variant is exposed via lpc_batch_f32_v2 above.
extern "C" __global__ void lpc_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ src,
    int len,
    const float* __restrict__ tr_opt, // optional precomputed TR (len), may be null
    const int* __restrict__ fixed_periods,
    const float* __restrict__ cycle_mults,
    const float* __restrict__ tr_mults,
    int n_combos,
    int first_valid,
    int cutoff_mode,           // 0=fixed, 1=adaptive(using dom)
    int max_cycle_limit,       // cap for adaptive; <=0 means no cap
    const float* __restrict__ dom, // length=len when adaptive; may be nullptr for fixed
    float* __restrict__ out_filter, // shape: n_combos × len (row-major)
    float* __restrict__ out_high,
    float* __restrict__ out_low)
{
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int combo = row0; combo < n_combos; combo += stride) {
        float* f_row  = out_filter + (size_t)combo * (size_t)len;
        float* hi_row = out_high   + (size_t)combo * (size_t)len;
        float* lo_row = out_low    + (size_t)combo * (size_t)len;

        const float tm_f = tr_mults[combo];
        const double tm = (double)tm_f;
        const int p_fixed = fixed_periods[combo];
        const float cm_f = cycle_mults[combo];
        const double cm = (double)cm_f;

        // Warmup prefix NaNs
        const uint32_t qnan_bits = 0x7fc00000u;
        const float qnan = __int_as_float(qnan_bits);
        const int warm = first_valid < len ? first_valid : len;
        for (int i = 0; i < warm; ++i) {
            f_row[i]  = qnan;
            hi_row[i] = qnan;
            lo_row[i] = qnan;
        }
        if (first_valid >= len) continue;

        // Seed at first_valid
        const int i0 = first_valid;
        const double s0 = (double)src[i0];
        f_row[i0] = (float)s0;

        double tr_prev = (double)(tr_opt ? tr_opt[i0] : (high[i0] - low[i0]));
        double ftr_prev = tr_prev;
        hi_row[i0] = (float)(s0 + tm * tr_prev);
        lo_row[i0] = (float)(s0 - tm * tr_prev);

        // Alpha cache
        int last_p = (cutoff_mode == 0 ? p_fixed : 0);
        // Use FP64 math for alpha for parity with previous kernel
        auto alpha_from_period_iir = [](int p)->double {
            if (p < 1) p = 1;
            const double omega = 2.0 * CUDART_PI / (double)p;
            double s = sin(omega), c = cos(omega);
            return (1.0 - s) / c;
        };
        double alpha = (cutoff_mode == 0 ? alpha_from_period_iir(p_fixed) : 0.0);

        for (int i = i0 + 1; i < len; ++i) {
            // Determine period
            int p_i = p_fixed;
            if (cutoff_mode != 0 && dom != nullptr) {
                double base = (double)dom[i];
                if (!isfinite(base)) {
                    p_i = p_fixed;
                } else {
                    double pd = nearbyint(base * cm);
                    if (pd < 3.0) pd = 3.0;
                    if (max_cycle_limit > 0 && pd > (double)max_cycle_limit) pd = (double)max_cycle_limit;
                    p_i = (int)pd;
                }
            }

            if (p_i != last_p) {
                last_p = p_i;
                alpha = alpha_from_period_iir(p_i);
            }
            const double one_m_a = 1.0 - alpha;

            // Filtered source (uses s[i] and s[i-1])
            const double s_im1 = (double)src[i - 1];
            const double s_i   = (double)src[i];
            const double prev_f = (double)f_row[i - 1];
            const double f_i = fma(alpha, prev_f, 0.5 * one_m_a * (s_i + s_im1));
            f_row[i] = (float)f_i;

            // Wilder TR
            double tr_i;
            if (tr_opt) {
                tr_i = (double)tr_opt[i];
            } else {
                const double hl  = (double)(high[i] - low[i]);
                const double c_l = fabs((double)close[i] - (double)low[i - 1]);
                const double c_h = fabs((double)close[i] - (double)high[i - 1]);
                tr_i = fmax(hl, fmax(c_l, c_h));
            }
            const double ftr_i = fma(alpha, ftr_prev, 0.5 * one_m_a * (tr_i + tr_prev));
            tr_prev = tr_i;
            ftr_prev = ftr_i;

            hi_row[i] = (float)(f_i + tm * ftr_i);
            lo_row[i] = (float)(f_i - tm * ftr_i);
        }
    }
}

// ============================================================
// Many-series × one-param, time-major input, fixed cutoff only
// (kept API; converted to FP32 & FMA)
// ============================================================
extern "C" __global__ void lpc_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ src_tm,
    int cols,                    // number of series
    int rows,                    // length per series
    int fixed_period,
    float cycle_mult,            // unused; reserved for parity
    float tr_mult,
    int cutoff_mode,             // must be 0 (fixed)
    int max_cycle_limit,         // reserved
    const int* __restrict__ first_valids, // per-series
    float* __restrict__ out_filter_tm,
    float* __restrict__ out_high_tm,
    float* __restrict__ out_low_tm
) {
    const int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s0 >= cols) return;

    const uint32_t qnan_bits = 0x7fc00000u;
    const float qnan = __int_as_float(qnan_bits);

    const int first = first_valids[s0];
    for (int t = 0; t < (first < rows ? first : rows); ++t) {
        const size_t idx = (size_t)t * (size_t)cols + (size_t)s0;
        out_filter_tm[idx] = qnan;
        out_high_tm[idx]   = qnan;
        out_low_tm[idx]    = qnan;
    }
    if (first >= rows) return;

    const float tm = tr_mult;
    float alpha = alpha_from_period_iir_f(fixed_period);

    auto AT = [&](const float* a, int t) -> float { return a[(size_t)t * (size_t)cols + (size_t)s0]; };
    auto W  = [&](float* a, int t, float v)       { a[(size_t)t * (size_t)cols + (size_t)s0] = v;  };

    // Seed
    float s_prev = AT(src_tm, first);
    float f_prev = s_prev;
    float tr_prev = AT(high_tm, first) - AT(low_tm, first);
    float ftr_prev = tr_prev;

    W(out_filter_tm, first, f_prev);
    W(out_high_tm,   first, f_prev + tm * tr_prev);
    W(out_low_tm,    first, f_prev - tm * tr_prev);

    // Loop
    #pragma unroll 1
    for (int t = first + 1; t < rows; ++t) {
        const float one_m_a = 1.0f - alpha;
        const float w = 0.5f * one_m_a;

        const float s_i = AT(src_tm, t);
        const float f_i = fmaf(alpha, f_prev, w * (s_i + s_prev));
        s_prev = s_i;
        f_prev = f_i;

        const float hl  = AT(high_tm, t) - AT(low_tm, t);
        const float c_l = fabsf(AT(close_tm, t) - AT(low_tm, t - 1));
        const float c_h = fabsf(AT(close_tm, t) - AT(high_tm, t - 1));
        const float tr_i = fmaxf(hl, fmaxf(c_l, c_h));

        const float ftr_i = fmaf(alpha, ftr_prev, w * (tr_i + tr_prev));
        tr_prev = tr_i;
        ftr_prev = ftr_i;

        W(out_filter_tm, t, f_i);
        W(out_high_tm,   t, f_i + tm * ftr_i);
        W(out_low_tm,    t, f_i - tm * ftr_i);
    }
}
