// CUDA kernels for Low Pass Channel (LPC)
//
// Math pattern: single-pole low-pass IIR on the source and on Wilder TR,
// with optional adaptive per-bar period derived from a host-precomputed
// dominant cycle (IFM). Kernels are FP32 with double accumulators.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

static __forceinline__ __device__ bool finite_f(float x) { return !isnan(x) && !isinf(x); }

// Helper: compute alpha from period in double precision.
static __forceinline__ __device__ double alpha_from_period_iir(int p) {
    // Guard minimal sensible period
    if (p < 1) p = 1;
    const double omega = 2.0 * CUDART_PI / static_cast<double>(p);
    double s = sin(omega), c = cos(omega);
    // Match scalar formula exactly (no extra guards here to keep parity)
    return (1.0 - s) / c;
}

// Batch: one-series × many-params. Optional `dom` (adaptive) shared across rows.
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
    float* __restrict__ out_low
) {
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int combo = row0; combo < n_combos; combo += stride) {
        float* f_row  = out_filter + (size_t)combo * len;
        float* hi_row = out_high   + (size_t)combo * len;
        float* lo_row = out_low    + (size_t)combo * len;

        const float tm_f = tr_mults[combo];
        const double tm = (double)tm_f;
        const int p_fixed = fixed_periods[combo];
        const float cm_f = cycle_mults[combo];
        const double cm = (double)cm_f;

        // Warmup prefix NaNs
        const float qnan = nanf("");
        for (int i = 0; i < min(first_valid, len); ++i) {
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
        double alpha = (cutoff_mode == 0 ? alpha_from_period_iir(p_fixed) : 0.0);

        for (int i = i0 + 1; i < len; ++i) {
            // Determine period
            int p_i = p_fixed;
            if (cutoff_mode != 0 && dom != nullptr) {
                double base = (double)dom[i];
                if (isnan(base)) {
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

// Many-series × one-param (time-major input). Supports fixed cutoff only.
extern "C" __global__ void lpc_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ src_tm,
    int cols,
    int rows,
    int fixed_period,
    float cycle_mult,   // unused in fixed mode; reserved for parity
    float tr_mult,
    int cutoff_mode,    // must be 0 (fixed)
    int max_cycle_limit, // reserved
    const int* __restrict__ first_valids, // per-series
    float* __restrict__ out_filter_tm,
    float* __restrict__ out_high_tm,
    float* __restrict__ out_low_tm
) {
    const int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s0 >= cols) return;

    const int first = first_valids[s0];
    const float qnan = nanf("");
    for (int t = 0; t < min(first, rows); ++t) {
        out_filter_tm[t * cols + s0] = qnan;
        out_high_tm[t * cols + s0]   = qnan;
        out_low_tm[t * cols + s0]    = qnan;
    }
    if (first >= rows) return;

    const double tm = (double)tr_mult;
    // Fixed period only (parity with wrapper policy)
    int last_p = fixed_period;
    double alpha = alpha_from_period_iir(fixed_period);

    auto AT = [&](const float* a, int t) -> float { return a[(size_t)t * cols + s0]; };
    auto W  = [&](float* a, int t, float v) { a[(size_t)t * cols + s0] = v; };

    // Seed
    const double s0d = (double)AT(src_tm, first);
    W(out_filter_tm, first, (float)s0d);
    double tr_prev = (double)(AT(high_tm, first) - AT(low_tm, first));
    double ftr_prev = tr_prev;
    W(out_high_tm, first, (float)(s0d + tm * tr_prev));
    W(out_low_tm,  first, (float)(s0d - tm * tr_prev));

    for (int t = first + 1; t < rows; ++t) {
        const double one_m_a = 1.0 - alpha;
        const double s_im1 = (double)AT(src_tm, t - 1);
        const double s_i   = (double)AT(src_tm, t);
        const double prev_f = (double)AT(out_filter_tm, t - 1);
        const double f_i = fma(alpha, prev_f, 0.5 * one_m_a * (s_i + s_im1));
        W(out_filter_tm, t, (float)f_i);

        const double hl  = (double)(AT(high_tm, t) - AT(low_tm, t));
        const double c_l = fabs((double)AT(close_tm, t) - (double)AT(low_tm, t - 1));
        const double c_h = fabs((double)AT(close_tm, t) - (double)AT(high_tm, t - 1));
        const double tr_i = fmax(hl, fmax(c_l, c_h));
        const double ftr_i = fma(alpha, ftr_prev, 0.5 * one_m_a * (tr_i + tr_prev));
        tr_prev = tr_i;
        ftr_prev = ftr_i;

        W(out_high_tm, t, (float)(f_i + tm * ftr_i));
        W(out_low_tm,  t, (float)(f_i - tm * ftr_i));
    }
}
