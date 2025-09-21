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

static __device__ __forceinline__ double hilbert(double x0, double x2, double x4, double x6) {
    return 0.0962 * x0 + 0.5769 * x2 - 0.5769 * x4 - 0.0962 * x6;
}

static __device__ __forceinline__ double atan_fast_f64(double z) {
    const double C0 = 0.2447;
    const double C1 = 0.0663;
    const double PIO4 = PI_D * 0.25;
    const double PIO2 = PI_D * 0.5;

    double a = fabs(z);
    if (a <= 1.0) {
        double t = fma(C1, a, C0); // C0 + C1 * a
        return fma(PIO4, z, fma(z, a - 1.0, t) * z);
    } else {
        double inv = 1.0 / z;
        double t = fma(C1, fabs(inv), C0);
        double base = fma(PIO4, inv, fma(inv, fabs(inv) - 1.0, t) * inv);
        return (z > 0.0) ? (PIO2 - base) : (-PIO2 - base);
    }
}

static __device__ __forceinline__ double clamp_double(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

} // namespace

extern "C" __global__
void mama_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ fast_limits,
                    const float* __restrict__ slow_limits,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out_mama,
                    float* __restrict__ out_fama) {
    int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    const float nanf32 = nanf("");
    float* out_m_row = out_mama + combo * series_len;
    float* out_f_row = out_fama + combo * series_len;
    for (int i = 0; i < series_len; ++i) {
        out_m_row[i] = nanf32;
        out_f_row[i] = nanf32;
    }

    if (series_len <= 0) {
        return;
    }
    if (first_valid < 0) {
        first_valid = 0;
    }
    if (first_valid >= series_len) {
        return;
    }

    double fast = static_cast<double>(fast_limits[combo]);
    double slow = static_cast<double>(slow_limits[combo]);
    if (!(fast > 0.0) || !(slow > 0.0)) {
        return;
    }

    const int warm = first_valid + 10;

    double smooth_buf[7];
    double detrender_buf[7];
    double i1_buf[7];
    double q1_buf[7];

    double seed_price = static_cast<double>(prices[first_valid]);
    for (int k = 0; k < 7; ++k) {
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

    for (int i = first_valid; i < series_len; ++i) {
        double price = static_cast<double>(prices[i]);
        double s1 = (i >= first_valid + 1) ? static_cast<double>(prices[i - 1]) : price;
        double s2 = (i >= first_valid + 2) ? static_cast<double>(prices[i - 2]) : price;
        double s3 = (i >= first_valid + 3) ? static_cast<double>(prices[i - 3]) : price;
        double smooth_val = (4.0 * price + 3.0 * s1 + 2.0 * s2 + s3) / 10.0;

        int idx = (i - first_valid) % 7;
        smooth_buf[idx] = smooth_val;

        double x0 = smooth_buf[idx];
        double x2 = smooth_buf[(idx + 5) % 7];
        double x4 = smooth_buf[(idx + 3) % 7];
        double x6 = smooth_buf[(idx + 1) % 7];

        double mesa_mult = 0.075 * prev_mesa_period + 0.54;
        double dt_val = hilbert(x0, x2, x4, x6) * mesa_mult;
        detrender_buf[idx] = dt_val;

        double i1_val = (i >= first_valid + 3) ? detrender_buf[(idx + 4) % 7] : dt_val;
        i1_buf[idx] = i1_val;

        double d0 = detrender_buf[idx];
        double d2 = detrender_buf[(idx + 5) % 7];
        double d4 = detrender_buf[(idx + 3) % 7];
        double d6 = detrender_buf[(idx + 1) % 7];
        double q1_val = hilbert(d0, d2, d4, d6) * mesa_mult;
        q1_buf[idx] = q1_val;

        double j_i = hilbert(i1_buf[idx],
                             i1_buf[(idx + 5) % 7],
                             i1_buf[(idx + 3) % 7],
                             i1_buf[(idx + 1) % 7]) * mesa_mult;
        double j_q = hilbert(q1_buf[idx],
                             q1_buf[(idx + 5) % 7],
                             q1_buf[(idx + 3) % 7],
                             q1_buf[(idx + 1) % 7]) * mesa_mult;

        double i2 = i1_val - j_q;
        double q2 = q1_val + j_i;
        double i2_sm = 0.2 * i2 + 0.8 * prev_i2_sm;
        double q2_sm = 0.2 * q2 + 0.8 * prev_q2_sm;
        double re = 0.2 * (i2_sm * prev_i2_sm + q2_sm * prev_q2_sm) + 0.8 * prev_re;
        double im = 0.2 * (i2_sm * prev_q2_sm - q2_sm * prev_i2_sm) + 0.8 * prev_im;
        prev_i2_sm = i2_sm;
        prev_q2_sm = q2_sm;
        prev_re = re;
        prev_im = im;

        double mesa_period = prev_mesa_period;
        if (re != 0.0 && im != 0.0) {
            double ratio = im / re;
            double ang = atan_fast_f64(ratio);
            if (ang != 0.0 && isfinite(ang)) {
                double candidate = (2.0 * PI_D) / ang;
                if (isfinite(candidate)) {
                    mesa_period = candidate;
                }
            }
        }

        double upper = 1.5 * prev_mesa_period;
        double lower = 0.67 * prev_mesa_period;
        mesa_period = clamp_double(mesa_period, lower, upper);
        mesa_period = clamp_double(mesa_period, 6.0, 50.0);
        mesa_period = 0.2 * mesa_period + 0.8 * prev_mesa_period;
        prev_mesa_period = mesa_period;

        double phase = prev_phase;
        if (i1_val != 0.0) {
            double ratio = q1_val / i1_val;
            double ang = atan(ratio);
            if (isfinite(ang)) {
                phase = ang * RAD2DEG_D;
            }
        }
        double dp = prev_phase - phase;
        if (dp < 1.0) {
            dp = 1.0;
        }
        prev_phase = phase;

        double alpha = fast / dp;
        double lo = slow < fast ? slow : fast;
        double hi = slow < fast ? fast : slow;
        alpha = clamp_double(alpha, lo, hi);

        double cur_mama = alpha * price + (1.0 - alpha) * prev_mama;
        double cur_fama = 0.5 * alpha * cur_mama + (1.0 - 0.5 * alpha) * prev_fama;
        prev_mama = cur_mama;
        prev_fama = cur_fama;

        if (i >= warm) {
            out_m_row[i] = static_cast<float>(cur_mama);
            out_f_row[i] = static_cast<float>(cur_fama);
        }
    }
}

extern "C" __global__
void mama_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    float fast_limit,
                                    float slow_limit,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_mama_tm,
                                    float* __restrict__ out_fama_tm) {
    int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    const float nanf32 = nanf("");
    for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + series_idx;
        out_mama_tm[idx] = nanf32;
        out_fama_tm[idx] = nanf32;
    }

    if (series_len <= 0) {
        return;
    }

    int first_valid = first_valids[series_idx];
    if (first_valid < 0) {
        first_valid = 0;
    }
    if (first_valid >= series_len) {
        return;
    }

    double fast = static_cast<double>(fast_limit);
    double slow = static_cast<double>(slow_limit);
    if (!(fast > 0.0) || !(slow > 0.0)) {
        return;
    }

    const int warm = first_valid + 10;

    double smooth_buf[7];
    double detrender_buf[7];
    double i1_buf[7];
    double q1_buf[7];

    double seed_price = static_cast<double>(prices_tm[first_valid * num_series + series_idx]);
    for (int k = 0; k < 7; ++k) {
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

    for (int t = first_valid; t < series_len; ++t) {
        int idx_tm = t * num_series + series_idx;
        double price = static_cast<double>(prices_tm[idx_tm]);
        double s1 = (t >= first_valid + 1)
            ? static_cast<double>(prices_tm[(t - 1) * num_series + series_idx])
            : price;
        double s2 = (t >= first_valid + 2)
            ? static_cast<double>(prices_tm[(t - 2) * num_series + series_idx])
            : price;
        double s3 = (t >= first_valid + 3)
            ? static_cast<double>(prices_tm[(t - 3) * num_series + series_idx])
            : price;
        double smooth_val = (4.0 * price + 3.0 * s1 + 2.0 * s2 + s3) / 10.0;

        int idx = (t - first_valid) % 7;
        smooth_buf[idx] = smooth_val;

        double x0 = smooth_buf[idx];
        double x2 = smooth_buf[(idx + 5) % 7];
        double x4 = smooth_buf[(idx + 3) % 7];
        double x6 = smooth_buf[(idx + 1) % 7];

        double mesa_mult = 0.075 * prev_mesa_period + 0.54;
        double dt_val = hilbert(x0, x2, x4, x6) * mesa_mult;
        detrender_buf[idx] = dt_val;

        double i1_val = (t >= first_valid + 3) ? detrender_buf[(idx + 4) % 7] : dt_val;
        i1_buf[idx] = i1_val;

        double d0 = detrender_buf[idx];
        double d2 = detrender_buf[(idx + 5) % 7];
        double d4 = detrender_buf[(idx + 3) % 7];
        double d6 = detrender_buf[(idx + 1) % 7];
        double q1_val = hilbert(d0, d2, d4, d6) * mesa_mult;
        q1_buf[idx] = q1_val;

        double j_i = hilbert(i1_buf[idx],
                             i1_buf[(idx + 5) % 7],
                             i1_buf[(idx + 3) % 7],
                             i1_buf[(idx + 1) % 7]) * mesa_mult;
        double j_q = hilbert(q1_buf[idx],
                             q1_buf[(idx + 5) % 7],
                             q1_buf[(idx + 3) % 7],
                             q1_buf[(idx + 1) % 7]) * mesa_mult;

        double i2 = i1_val - j_q;
        double q2 = q1_val + j_i;
        double i2_sm = 0.2 * i2 + 0.8 * prev_i2_sm;
        double q2_sm = 0.2 * q2 + 0.8 * prev_q2_sm;
        double re = 0.2 * (i2_sm * prev_i2_sm + q2_sm * prev_q2_sm) + 0.8 * prev_re;
        double im = 0.2 * (i2_sm * prev_q2_sm - q2_sm * prev_i2_sm) + 0.8 * prev_im;
        prev_i2_sm = i2_sm;
        prev_q2_sm = q2_sm;
        prev_re = re;
        prev_im = im;

        double mesa_period = prev_mesa_period;
        if (re != 0.0 && im != 0.0) {
            double ratio = im / re;
            double ang = atan_fast_f64(ratio);
            if (ang != 0.0 && isfinite(ang)) {
                double candidate = (2.0 * PI_D) / ang;
                if (isfinite(candidate)) {
                    mesa_period = candidate;
                }
            }
        }

        double upper = 1.5 * prev_mesa_period;
        double lower = 0.67 * prev_mesa_period;
        mesa_period = clamp_double(mesa_period, lower, upper);
        mesa_period = clamp_double(mesa_period, 6.0, 50.0);
        mesa_period = 0.2 * mesa_period + 0.8 * prev_mesa_period;
        prev_mesa_period = mesa_period;

        double phase = prev_phase;
        if (i1_val != 0.0) {
            double ratio = q1_val / i1_val;
            double ang = atan(ratio);
            if (isfinite(ang)) {
                phase = ang * RAD2DEG_D;
            }
        }
        double dp = prev_phase - phase;
        if (dp < 1.0) {
            dp = 1.0;
        }
        prev_phase = phase;

        double alpha = fast / dp;
        double lo = slow < fast ? slow : fast;
        double hi = slow < fast ? fast : slow;
        alpha = clamp_double(alpha, lo, hi);

        double cur_mama = alpha * price + (1.0 - alpha) * prev_mama;
        double cur_fama = 0.5 * alpha * cur_mama + (1.0 - 0.5 * alpha) * prev_fama;
        prev_mama = cur_mama;
        prev_fama = cur_fama;

        if (t >= warm) {
            out_mama_tm[idx_tm] = static_cast<float>(cur_mama);
            out_fama_tm[idx_tm] = static_cast<float>(cur_fama);
        }
    }
}
