#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

static __forceinline__ __device__ bool finite_f(float x) { return isfinite(x); }


static __forceinline__ __device__ float alpha_from_period_iir_f(int p) {
    if (p < 1) p = 1;
    const float omega = 2.0f * CUDART_PI_F / (float)p;
    float s, c;

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

extern "C" __global__ void lpc_build_true_range_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    float* __restrict__ tr_out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    if (i == 0) {
        tr_out[0] = high[0] - low[0];
        return;
    }

    const float hl  = high[i] - low[i];
    const float c_l = fabsf(close[i] - low[i - 1]);
    const float c_h = fabsf(close[i] - high[i - 1]);
    tr_out[i] = fmaxf(hl, fmaxf(c_l, c_h));
}

extern "C" __global__ void lpc_build_dom_cycle_f32_serial(
    const float* __restrict__ src,
    int len,
    int max_cycle_limit,
    double* __restrict__ delta_phase_ring,
    int delta_phase_ring_len,
    float* __restrict__ dom_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const uint32_t qnan_bits = 0x7fc00000u;
    const float qnan = __int_as_float(qnan_bits);
    for (int i = 0; i < len; ++i) {
        dom_out[i] = qnan;
    }
    if (len < 8 || delta_phase_ring_len <= 0) return;

    for (int i = 0; i < delta_phase_ring_len; ++i) {
        delta_phase_ring[i] = 0.0;
    }

    double in_phase_hist[4] = {0.0, 0.0, 0.0, 0.0};
    double quadrature_hist[4] = {0.0, 0.0, 0.0, 0.0};
    double real_prev = 0.0;
    double imag_prev = 0.0;
    double inst_prev = 0.0;
    double dom_prev = CUDART_NAN;

    for (int i = 7; i < len; ++i) {
        const int src_m11 = (i >= 11) ? (i - 11) : 0;
        const int src_m9  = (i >= 9) ? (i - 9) : 0;

        const double val1 = (double)src[i] - (double)src[i - 7];
        const double val1_4 = (double)src[i - 4] - (double)src[src_m11];
        const double val1_2 = (double)src[i - 2] - (double)src[src_m9];

        const double in_phase_i =
            1.25 * (val1_4 - 0.635 * val1_2) + 0.635 * in_phase_hist[(i - 3) & 3];
        const double quadrature_i =
            val1_2 - 0.338 * val1 + 0.338 * quadrature_hist[(i - 2) & 3];

        const double in_phase_prev = in_phase_hist[(i - 1) & 3];
        const double quadrature_prev = quadrature_hist[(i - 1) & 3];

        const double real_i =
            0.2 * (in_phase_i * in_phase_prev + quadrature_i * quadrature_prev) + 0.8 * real_prev;
        const double imag_i =
            0.2 * (in_phase_i * quadrature_prev - in_phase_prev * quadrature_i) + 0.8 * imag_prev;

        double delta_i = 0.0;
        if (real_i != 0.0) {
            delta_i = atan(imag_i / real_i);
        }
        delta_phase_ring[i % delta_phase_ring_len] = delta_i;

        double val2 = 0.0;
        bool found_period = false;
        double inst_i = inst_prev;
        const int limit = max_cycle_limit < i ? max_cycle_limit : i;
        for (int j = 0; j <= limit; ++j) {
            val2 += delta_phase_ring[(i - j) % delta_phase_ring_len];
            if (val2 > 2.0 * CUDART_PI && !found_period) {
                inst_i = (double)j;
                found_period = true;
                break;
            }
        }

        if (!found_period) {
            inst_i = (i > 0) ? inst_prev : 20.0;
        }

        const double dom_i = !isnan(dom_prev) ? (0.25 * inst_i + 0.75 * dom_prev) : inst_i;
        dom_out[i] = (float)dom_i;

        in_phase_hist[i & 3] = in_phase_i;
        quadrature_hist[i & 3] = quadrature_i;
        real_prev = real_i;
        imag_prev = imag_i;
        inst_prev = inst_i;
        dom_prev = dom_i;
    }
}


extern "C" __global__ void lpc_batch_f32_v2(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ src,
    int len,
    const float* __restrict__ tr_opt,
    const int*   __restrict__ fixed_periods,
    const float* __restrict__ cycle_mults,
    const float* __restrict__ tr_mults,
    int n_combos,
    int first_valid,
    int cutoff_mode,
    int max_cycle_limit,
    const float* __restrict__ dom,

    const float* __restrict__ alpha_lut,
    int alpha_lut_len,
    int alpha_lut_pmin,

    int out_time_major,
    float* __restrict__ out_filter,
    float* __restrict__ out_high,
    float* __restrict__ out_low
){
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const uint32_t qnan_bits = 0x7fc00000u;
    const float qnan = __int_as_float(qnan_bits);

    for (int combo = tid; combo < n_combos; combo += stride) {

        auto store_triplet = [&](int i, float f, float hi, float lo) {
            size_t idx = out_time_major ? (size_t)i * (size_t)n_combos + (size_t)combo
                                        : (size_t)combo * (size_t)len    + (size_t)i;
            out_filter[idx] = f;
            out_high[idx]   = hi;
            out_low[idx]    = lo;
        };


        if (first_valid > 0) {
            const int upto = first_valid < len ? first_valid : len;
            for (int i = 0; i < upto; ++i) store_triplet(i, qnan, qnan, qnan);
            if (first_valid >= len) continue;
        }


        const float tm       = tr_mults[combo];
        const int   p_fixed  = fixed_periods[combo];
        const float cm       = cycle_mults[combo];
        const bool  adaptive = (cutoff_mode != 0) && (dom != nullptr);


        const int i0 = first_valid;
        float s_prev = src[i0];
        float f_prev = s_prev;


        float tr_prev = tr_opt ? tr_opt[i0] : (high[i0] - low[i0]);
        float ftr_prev = tr_prev;


        int last_p = adaptive ? 0 : p_fixed;
        float alpha = lut_or_formula_alpha(p_fixed, alpha_lut, alpha_lut_len, alpha_lut_pmin);


        store_triplet(i0, f_prev, f_prev + tm * tr_prev, f_prev - tm * tr_prev);


        #pragma unroll 1
        for (int i = i0 + 1; i < len; ++i) {

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


            const float s_i = src[i];

            const float f_i = fmaf(alpha, f_prev, w * (s_i + s_prev));
            s_prev = s_i;
            f_prev = f_i;


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


            const float hi = f_i + tm * ftr_i;
            const float lo = f_i - tm * ftr_i;
            store_triplet(i, f_i, hi, lo);
        }
    }
}


extern "C" __global__ void lpc_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ src,
    int len,
    const float* __restrict__ tr_opt,
    const int* __restrict__ fixed_periods,
    const float* __restrict__ cycle_mults,
    const float* __restrict__ tr_mults,
    int n_combos,
    int first_valid,
    int cutoff_mode,
    int max_cycle_limit,
    const float* __restrict__ dom,
    float* __restrict__ out_filter,
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


        const uint32_t qnan_bits = 0x7fc00000u;
        const float qnan = __int_as_float(qnan_bits);
        const int warm = first_valid < len ? first_valid : len;
        for (int i = 0; i < warm; ++i) {
            f_row[i]  = qnan;
            hi_row[i] = qnan;
            lo_row[i] = qnan;
        }
        if (first_valid >= len) continue;


        const int i0 = first_valid;
        const double s0 = (double)src[i0];
        f_row[i0] = (float)s0;

        double tr_prev = (double)(tr_opt ? tr_opt[i0] : (high[i0] - low[i0]));
        double ftr_prev = tr_prev;
        hi_row[i0] = (float)(s0 + tm * tr_prev);
        lo_row[i0] = (float)(s0 - tm * tr_prev);


        int last_p = (cutoff_mode == 0 ? p_fixed : 0);

        auto alpha_from_period_iir = [](int p)->double {
            if (p < 1) p = 1;
            const double omega = 2.0 * CUDART_PI / (double)p;
            double s = sin(omega), c = cos(omega);
            return (1.0 - s) / c;
        };
        double alpha = (cutoff_mode == 0 ? alpha_from_period_iir(p_fixed) : 0.0);

        for (int i = i0 + 1; i < len; ++i) {

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


            const double s_im1 = (double)src[i - 1];
            const double s_i   = (double)src[i];
            const double prev_f = (double)f_row[i - 1];
            const double f_i = fma(alpha, prev_f, 0.5 * one_m_a * (s_i + s_im1));
            f_row[i] = (float)f_i;


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


extern "C" __global__ void lpc_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ src_tm,
    int cols,
    int rows,
    int fixed_period,
    float cycle_mult,
    float tr_mult,
    int cutoff_mode,
    int max_cycle_limit,
    const int* __restrict__ first_valids,
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


    float s_prev = AT(src_tm, first);
    float f_prev = s_prev;
    float tr_prev = AT(high_tm, first) - AT(low_tm, first);
    float ftr_prev = tr_prev;

    W(out_filter_tm, first, f_prev);
    W(out_high_tm,   first, f_prev + tm * tr_prev);
    W(out_low_tm,    first, f_prev - tm * tr_prev);


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
