// CUDA kernels for the MESA Adaptive Moving Average (MAMA).
//
// Each parameter combination (fast/slow limit pair) is evaluated sequentially
// while keeping the price series resident on device. The recurrence matches the
// scalar Rust implementation, including the 10-sample warmup window that remains
// NaN in the outputs. FP32 storage is used for device buffers while the core
// arithmetic promotes to FP64 to preserve numerical stability.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr double PI_D = 3.14159265358979323846264338327950288;
constexpr double RAD2DEG_D = 180.0 / PI_D;

// --- helper: 8-deep shift register kept entirely in registers ---
struct Shift8d {
    double r0, r1, r2, r3, r4, r5, r6, r7;

    __device__ __forceinline__ void seed(double v) {
        r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = v;
    }
    __device__ __forceinline__ void push(double v) {
        r7 = r6; r6 = r5; r5 = r4; r4 = r3;
        r3 = r2; r2 = r1; r1 = r0; r0 = v;
    }
    __device__ __forceinline__ void taps(double& x0, double& x2, double& x4, double& x6) const {
        x0 = r0; x2 = r2; x4 = r4; x6 = r6;
    }
    __device__ __forceinline__ double lag3() const { return r3; }
};

static __device__ __forceinline__ double hilbert_fma(double x0, double x2, double x4, double x6) {
    const double H0 = 0.0962;
    const double H1 = 0.5769;
    const double H2 = -0.5769;
    const double H3 = -0.0962;
    double t = fma(H2, x4, H3 * x6);
    t = fma(H1, x2, t);
    return fma(H0, x0, t);
}

// A non-FMA variant to better match CPU scalar code when fused ops are not used
static __device__ __forceinline__ double hilbert_nfma(double x0, double x2, double x4, double x6) {
    const double H0 = 0.0962;
    const double H1 = 0.5769;
    const double H2 = -0.5769;
    const double H3 = -0.0962;
    return H0 * x0 + H1 * x2 + H2 * x4 + H3 * x6;
}

static __device__ __forceinline__ double atan_fast_f64(double z) {
    // Match Rust atan_fast exactly (FMA sequencing and no extra z* on the +t term).
    const double C0 = 0.2447;
    const double C1 = 0.0663;
    const double PIO4 = PI_D * 0.25; // FRAC_PI_4
    const double PIO2 = PI_D * 0.5;  // FRAC_PI_2

    double a = fabs(z);
    if (a <= 1.0) {
        // PIO4.mul_add(z, z.mul_add(a - 1.0, t))
        double t = fma(C1, a, C0);                 // t = C1*a + C0
        double inner = fma(z, (a - 1.0), t);       // z*(a-1) + t
        return fma(PIO4, z, inner);                // PIO4*z + inner
    } else {
        // inv = 1/z; base = PIO4.mul_add(inv, inv.mul_add(|inv|-1.0, t))
        double inv = 1.0 / z;
        double ai = fabs(inv);
        double t = fma(C1, ai, C0);                // C1*|inv| + C0
        double inner = fma(inv, (ai - 1.0), t);    // inv*(|inv|-1) + t
        double base = fma(PIO4, inv, inner);       // PIO4*inv + inner
        return (z > 0.0) ? (PIO2 - base) : (-PIO2 - base);
    }
}

static __device__ __forceinline__ double clamp_double(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

} // namespace

extern "C" __global__ __launch_bounds__(256, 2)
void mama_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ fast_limits,
                    const float* __restrict__ slow_limits,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out_mama,
                    float* __restrict__ out_fama) {
    if (series_len <= 0) return;

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const float nanf32 = nanf("");

    // Numerically strict path when there's exactly one combo: reproduce the original
    // ring-buffer algorithm byte-for-byte to minimize divergence from CPU apply_slice.
    if (n_combos == 1) {
        if (tid != 0) return; // single logical worker for strict path
        const int combo = 0;

        float* out_m_row = out_mama + combo * series_len;
        float* out_f_row = out_fama + combo * series_len;

        int fv = first_valid;
        if (fv < 0) fv = 0;
        if (fv >= series_len || series_len <= 0) {
            const float nn = nanf("");
            for (int i = 0; i < series_len; ++i) { out_m_row[i] = nn; out_f_row[i] = nn; }
            return;
        }

        double fast = static_cast<double>(fast_limits[combo]);
        double slow = static_cast<double>(slow_limits[combo]);
        const float nn = nanf("");
        // Initialize entire row to NaN (as original); we will overwrite valid tail
        for (int i = 0; i < series_len; ++i) { out_m_row[i] = nn; out_f_row[i] = nn; }
        if (!(fast > 0.0) || !(slow > 0.0)) {
            return;
        }

        const int warm = fv + 10;

        // Original power-of-two ring buffers (may spill, acceptable for combos==1)
        constexpr int RING = 8;
        constexpr int MASK = RING - 1;
        double smooth_buf[RING];
        double detrender_buf[RING];
        double i1_buf[RING];
        double q1_buf[RING];

        double seed_price = static_cast<double>(prices[fv]);
        for (int k = 0; k < RING; ++k) {
            smooth_buf[k] = seed_price;
            detrender_buf[k] = seed_price;
            i1_buf[k] = seed_price;
            q1_buf[k] = seed_price;
        }

        double prev_mesa_period = 0.0;
        double prev_mama = seed_price;
        double prev_fama = seed_price;
        double prev_i2_sm = 0.0;
        double prev_q2_sm = 0.0;
        double prev_re = 0.0;
        double prev_im = 0.0;
        double prev_phase = 0.0;

        for (int i = fv; i < series_len; ++i) {
            double price = static_cast<double>(prices[i]);
            double s1 = (i >= fv + 1) ? static_cast<double>(prices[i - 1]) : price;
            double s2 = (i >= fv + 2) ? static_cast<double>(prices[i - 2]) : price;
            double s3 = (i >= fv + 3) ? static_cast<double>(prices[i - 3]) : price;
            double smooth_val = (4.0 * price + 3.0 * s1 + 2.0 * s2 + s3) * 0.1;

            int idx = (i - fv) & MASK;
            smooth_buf[idx] = smooth_val;

            double x0 = smooth_buf[idx];
            double x2 = smooth_buf[(idx - 2) & MASK];
            double x4 = smooth_buf[(idx - 4) & MASK];
            double x6 = smooth_buf[(idx - 6) & MASK];

            double mesa_mult = fma(0.075, prev_mesa_period, 0.54);
            double dt_val = hilbert_fma(x0, x2, x4, x6) * mesa_mult;
            detrender_buf[idx] = dt_val;

            double i1_val = detrender_buf[(idx - 3) & MASK];
            i1_buf[idx] = i1_val;

            double d0 = detrender_buf[idx];
            double d2 = detrender_buf[(idx - 2) & MASK];
            double d4 = detrender_buf[(idx - 4) & MASK];
            double d6 = detrender_buf[(idx - 6) & MASK];
            double q1_val = hilbert_fma(d0, d2, d4, d6) * mesa_mult;
            q1_buf[idx] = q1_val;

            double j_i = hilbert_fma(i1_buf[idx],
                                     i1_buf[(idx - 2) & MASK],
                                     i1_buf[(idx - 4) & MASK],
                                     i1_buf[(idx - 6) & MASK]) * mesa_mult;
            double j_q = hilbert_fma(q1_buf[idx],
                                     q1_buf[(idx - 2) & MASK],
                                     q1_buf[(idx - 4) & MASK],
                                     q1_buf[(idx - 6) & MASK]) * mesa_mult;

            double i2 = i1_val - j_q;
            double q2 = q1_val + j_i;
            double i2_sm = 0.2 * i2 + 0.8 * prev_i2_sm;
            double q2_sm = 0.2 * q2 + 0.8 * prev_q2_sm;
            double re    = 0.2 * (i2_sm * prev_i2_sm + q2_sm * prev_q2_sm) + 0.8 * prev_re;
            double im    = 0.2 * (i2_sm * prev_q2_sm - q2_sm * prev_i2_sm) + 0.8 * prev_im;
            prev_i2_sm = i2_sm; prev_q2_sm = q2_sm; prev_re = re; prev_im = im;

            double mesa_period = prev_mesa_period;
            if (re != 0.0 && im != 0.0) {
                double ratio = im / re;
                double ang = atan_fast_f64(ratio);
                double candidate = (2.0 * PI_D) / ang;
                mesa_period = candidate;
            }
            double upper = 1.5 * prev_mesa_period;
            double lower = 0.67 * prev_mesa_period;
            if (mesa_period > upper) mesa_period = upper;
            if (mesa_period < lower) mesa_period = lower;
            if (mesa_period < 6.0)   mesa_period = 6.0;
            if (mesa_period > 50.0)  mesa_period = 50.0;
            mesa_period = 0.2 * mesa_period + 0.8 * prev_mesa_period;
            prev_mesa_period = mesa_period;

            double phase = prev_phase;
            if (i1_val != 0.0) {
                double ratio = q1_val / i1_val;
                double ang = atan_fast_f64(ratio);
                phase = ang * RAD2DEG_D;
            }
            double dp = prev_phase - phase;
            if (dp < 1.0) dp = 1.0;
            prev_phase = phase;

            double alpha = fast / dp;
            double lo = slow < fast ? slow : fast;
            double hi = slow < fast ? fast : slow;
            alpha = clamp_double(alpha, lo, hi);

            double cur_mama = alpha * price + (1.0 - alpha) * prev_mama;
            double cur_fama = 0.5 * alpha * cur_mama + (1.0 - 0.5 * alpha) * prev_fama;
            prev_mama = cur_mama; prev_fama = cur_fama;

            if (i >= warm) {
                out_m_row[i] = static_cast<float>(cur_mama);
                out_f_row[i] = static_cast<float>(cur_fama);
            }
        }
        return;
    }

    for (int combo = tid; combo < n_combos; combo += stride) {
        float* out_m_row = out_mama + combo * series_len;
        float* out_f_row = out_fama + combo * series_len;

        int fv = first_valid;
        if (fv < 0) fv = 0;
        if (fv >= series_len) {
            // Entire row NaN if fv beyond series
            for (int i = 0; i < series_len; ++i) { out_m_row[i] = nanf32; out_f_row[i] = nanf32; }
            continue;
        }

        double fast = static_cast<double>(fast_limits[combo]);
        double slow = static_cast<double>(slow_limits[combo]);
        if (!(fast > 0.0) || !(slow > 0.0)) {
            for (int i = 0; i < series_len; ++i) { out_m_row[i] = nanf32; out_f_row[i] = nanf32; }
            continue;
        }

        const int warm = fv + 10;

        double seed_price = static_cast<double>(prices[fv]);
        double p1 = seed_price, p2 = seed_price, p3 = seed_price; // legacy state (unused after switch)

        Shift8d smooth, detrender, i1r, q1r;
        smooth.seed(seed_price);
        detrender.seed(seed_price);
        i1r.seed(seed_price);
        q1r.seed(seed_price);

        double prev_mesa_period = 0.0;
        double prev_mama = seed_price;
        double prev_fama = seed_price;
        double prev_i2_sm = 0.0;
        double prev_q2_sm = 0.0;
        double prev_re = 0.0;
        double prev_im = 0.0;
        double prev_phase = 0.0;

        // Fill [0, warm) with NaN once
        const int nan_end = (warm < series_len ? warm : series_len);
        for (int i = 0; i < nan_end; ++i) { out_m_row[i] = nanf32; out_f_row[i] = nanf32; }

        const bool use_nfma = (n_combos == 1);
        for (int i = fv; i < series_len; ++i) {
            double price = static_cast<double>(prices[i]);
            double s1 = (i >= fv + 1) ? p1 : price;
            double s2 = (i >= fv + 2) ? p2 : price;
            double s3 = (i >= fv + 3) ? p3 : price;
            double smooth_val = 0.1 * fma(4.0, price, fma(3.0, s1, fma(2.0, s2, s3)));
            p3 = p2; p2 = p1; p1 = price;

            smooth.push(smooth_val);
            double x0, x2, x4, x6; smooth.taps(x0, x2, x4, x6);

            double mesa_mult = fma(0.075, prev_mesa_period, 0.54);
            double dt_val = (use_nfma ? hilbert_nfma(x0, x2, x4, x6)
                                      : hilbert_fma(x0, x2, x4, x6)) * mesa_mult;
            detrender.push(dt_val);

            double i1_val = detrender.lag3();
            i1r.push(i1_val);

            double d0, d2, d4, d6; detrender.taps(d0, d2, d4, d6);
            double q1_val = (use_nfma ? hilbert_nfma(d0, d2, d4, d6)
                                      : hilbert_fma(d0, d2, d4, d6)) * mesa_mult;
            q1r.push(q1_val);

            double ii0, ii2, ii4, ii6; i1r.taps(ii0, ii2, ii4, ii6);
            double qq0, qq2, qq4, qq6; q1r.taps(qq0, qq2, qq4, qq6);
            double j_i = (use_nfma ? hilbert_nfma(ii0, ii2, ii4, ii6)
                                   : hilbert_fma(ii0, ii2, ii4, ii6)) * mesa_mult;
            double j_q = (use_nfma ? hilbert_nfma(qq0, qq2, qq4, qq6)
                                   : hilbert_fma(qq0, qq2, qq4, qq6)) * mesa_mult;

            double i2 = i1_val - j_q;
            double q2 = q1_val + j_i;

            double i2_sm = fma(0.2, i2, 0.8 * prev_i2_sm);
            double q2_sm = fma(0.2, q2, 0.8 * prev_q2_sm);
            double re    = fma(0.2, i2_sm * prev_i2_sm + q2_sm * prev_q2_sm, 0.8 * prev_re);
            double im    = fma(0.2, i2_sm * prev_q2_sm - q2_sm * prev_i2_sm, 0.8 * prev_im);
            prev_i2_sm = i2_sm; prev_q2_sm = q2_sm; prev_re = re; prev_im = im;

            double mesa_period = prev_mesa_period;
            if (re != 0.0 && im != 0.0) {
                double ratio = im / re;
                double ang = atan_fast_f64(ratio);
                double candidate = (2.0 * PI_D) / ang;
                mesa_period = candidate;
            }
            double upper = 1.5 * prev_mesa_period;
            double lower = 0.67 * prev_mesa_period;
            if (mesa_period > upper) mesa_period = upper;
            if (mesa_period < lower) mesa_period = lower;
            if (mesa_period < 6.0)   mesa_period = 6.0;
            if (mesa_period > 50.0)  mesa_period = 50.0;
            mesa_period = fma(0.2, mesa_period, 0.8 * prev_mesa_period);
            prev_mesa_period = mesa_period;

            double phase = prev_phase;
            if (i1_val != 0.0) {
                double ratio = q1_val / i1_val;
                double ang = atan_fast_f64(ratio);
                phase = ang * RAD2DEG_D;
            }
            double dp = prev_phase - phase;
            if (dp < 1.0) dp = 1.0;
            prev_phase = phase;

            double alpha = fast / dp;
            double lo = slow < fast ? slow : fast;
            double hi = slow < fast ? fast : slow;
            alpha = clamp_double(alpha, lo, hi);

            double cur_mama = fma(alpha, price, (1.0 - alpha) * prev_mama);
            double cur_fama = fma(0.5 * alpha, cur_mama, (1.0 - 0.5 * alpha) * prev_fama);
            prev_mama = cur_mama; prev_fama = cur_fama;

            if (i >= warm) {
                out_m_row[i] = static_cast<float>(cur_mama);
                out_f_row[i] = static_cast<float>(cur_fama);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void mama_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    float fast_limit,
                                    float slow_limit,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_mama_tm,
                                    float* __restrict__ out_fama_tm) {
    if (series_len <= 0 || num_series <= 0) return;

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const float nanf32 = nanf("");

    double fast = static_cast<double>(fast_limit);
    double slow = static_cast<double>(slow_limit);

    for (int series_idx = tid; series_idx < num_series; series_idx += stride) {
        if (!(fast > 0.0) || !(slow > 0.0)) {
            for (int t = 0; t < series_len; ++t) {
                int idx = t * num_series + series_idx;
                out_mama_tm[idx] = nanf32; out_fama_tm[idx] = nanf32;
            }
            continue;
        }

        int fv = first_valids[series_idx];
        if (fv < 0) fv = 0;
        if (fv >= series_len) {
            for (int t = 0; t < series_len; ++t) {
                int idx = t * num_series + series_idx;
                out_mama_tm[idx] = nanf32; out_fama_tm[idx] = nanf32;
            }
            continue;
        }

        const int warm = fv + 10;

        int base = fv * num_series + series_idx;
        double seed_price = static_cast<double>(prices_tm[base]);
        double p1 = seed_price, p2 = seed_price, p3 = seed_price;

        Shift8d smooth, detrender, i1r, q1r;
        smooth.seed(seed_price);
        detrender.seed(seed_price);
        i1r.seed(seed_price);
        q1r.seed(seed_price);

        double prev_mesa_period = 0.0;
        double prev_mama = seed_price;
        double prev_fama = seed_price;
        double prev_i2_sm = 0.0;
        double prev_q2_sm = 0.0;
        double prev_re = 0.0;
        double prev_im = 0.0;
        double prev_phase = 0.0;

        // Fill [0, warm) with NaN prefix
        const int nan_end = (warm < series_len ? warm : series_len);
        for (int t = 0; t < nan_end; ++t) {
            int idx = t * num_series + series_idx;
            out_mama_tm[idx] = nanf32; out_fama_tm[idx] = nanf32;
        }

        for (int t = fv; t < series_len; ++t) {
            int idx_tm = t * num_series + series_idx;
            double price = static_cast<double>(prices_tm[idx_tm]);
            double s1 = (t >= fv + 1)
                ? static_cast<double>(prices_tm[(t - 1) * num_series + series_idx])
                : price;
            double s2 = (t >= fv + 2)
                ? static_cast<double>(prices_tm[(t - 2) * num_series + series_idx])
                : price;
            double s3 = (t >= fv + 3)
                ? static_cast<double>(prices_tm[(t - 3) * num_series + series_idx])
                : price;
            double smooth_val = (4.0 * price + 3.0 * s1 + 2.0 * s2 + s3) / 10.0;

            smooth.push(smooth_val);
            double x0, x2, x4, x6; smooth.taps(x0, x2, x4, x6);

            double mesa_mult = 0.075 * prev_mesa_period + 0.54;
            // Use FMA path to mirror scalar mul_add ordering more closely
            double dt_val = hilbert_fma(x0, x2, x4, x6) * mesa_mult;
            detrender.push(dt_val);

            double i1_val = detrender.lag3();
            i1r.push(i1_val);

            double d0, d2, d4, d6; detrender.taps(d0, d2, d4, d6);
            double q1_val = hilbert_fma(d0, d2, d4, d6) * mesa_mult;
            q1r.push(q1_val);

            double ii0, ii2, ii4, ii6; i1r.taps(ii0, ii2, ii4, ii6);
            double qq0, qq2, qq4, qq6; q1r.taps(qq0, qq2, qq4, qq6);
            double j_i = hilbert_fma(ii0, ii2, ii4, ii6) * mesa_mult;
            double j_q = hilbert_fma(qq0, qq2, qq4, qq6) * mesa_mult;

            double i2 = i1_val - j_q;
            double q2 = q1_val + j_i;

            double i2_sm = 0.2 * i2 + 0.8 * prev_i2_sm;
            double q2_sm = 0.2 * q2 + 0.8 * prev_q2_sm;
            double re    = 0.2 * (i2_sm * prev_i2_sm + q2_sm * prev_q2_sm) + 0.8 * prev_re;
            double im    = 0.2 * (i2_sm * prev_q2_sm - q2_sm * prev_i2_sm) + 0.8 * prev_im;
            prev_i2_sm = i2_sm; prev_q2_sm = q2_sm; prev_re = re; prev_im = im;

            double mesa_period = prev_mesa_period;
            if (re != 0.0 && im != 0.0) {
                double ratio = im / re;
                double ang = atan_fast_f64(ratio);
                double candidate = (2.0 * PI_D) / ang;
                mesa_period = candidate;
            }
            double upper = 1.5 * prev_mesa_period;
            double lower = 0.67 * prev_mesa_period;
            if (mesa_period > upper) mesa_period = upper;
            if (mesa_period < lower) mesa_period = lower;
            if (mesa_period < 6.0)   mesa_period = 6.0;
            if (mesa_period > 50.0)  mesa_period = 50.0;
            mesa_period = 0.2 * mesa_period + 0.8 * prev_mesa_period;
            prev_mesa_period = mesa_period;

            double phase = prev_phase;
            if (i1_val != 0.0) {
                double ratio = q1_val / i1_val;
                double ang = atan_fast_f64(ratio);
                phase = ang * RAD2DEG_D;
            }
            double dp = prev_phase - phase;
            if (dp < 1.0) dp = 1.0;
            prev_phase = phase;

            double alpha = fast / dp;
            double lo = slow < fast ? slow : fast;
            double hi = slow < fast ? fast : slow;
            alpha = clamp_double(alpha, lo, hi);

            double cur_mama = alpha * price + (1.0 - alpha) * prev_mama;
            double cur_fama = 0.5 * alpha * cur_mama + (1.0 - 0.5 * alpha) * prev_fama;
            prev_mama = cur_mama; prev_fama = cur_fama;

            if (t >= warm) {
                out_mama_tm[idx_tm] = static_cast<float>(cur_mama);
                out_fama_tm[idx_tm] = static_cast<float>(cur_fama);
            }
        }
    }
}
