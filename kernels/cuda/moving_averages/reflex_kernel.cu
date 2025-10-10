// CUDA kernels for the Reflex indicator.
//
// Each parameter combination (period) is assigned to a dedicated block that
// processes the entire series sequentially. The recurrence only depends on the
// last `period` smoothed values, so we keep a circular buffer in shared memory
// to avoid additional global allocations while still matching the scalar
// implementation's warmup semantics (first `period` outputs = 0.0, remainder
// initialised to NaN).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Match scalar implementation constants: uses 1.414 (not exact sqrt(2)).
constexpr double REFLEX_PI_D = 3.14159265358979323846264338327950288;
constexpr double REFLEX_SQRT2_APPROX_D = 1.414; // matches scalar's 1.414_f64

static __device__ __forceinline__ float reflex_compute_ssf(float price,
                                                           float prev_price,
                                                           float c,
                                                           float b,
                                                           float a_sq,
                                                           float prev_ssf1,
                                                           float prev_ssf2) {
    const double val = static_cast<double>(c) *
            (static_cast<double>(price) + static_cast<double>(prev_price)) +
        static_cast<double>(b) * static_cast<double>(prev_ssf1) -
        static_cast<double>(a_sq) * static_cast<double>(prev_ssf2);
    return static_cast<float>(val);
}

static __device__ __forceinline__ bool reflex_isfinite(float v) {
    return !(isnan(v) || isinf(v));
}

extern "C" __global__
void reflex_batch_f32(const float* __restrict__ prices,
                      const int* __restrict__ periods,
                      int series_len,
                      int n_combos,
                      int /*first_valid (unused, kept for ABI)*/,
                      float* __restrict__ out) {
    int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) { return; }

    const int period = periods[combo];
    if (period < 2 || series_len <= 0) { return; }

    float* out_row = out + combo * series_len;
    // Initialize outputs to 0.0 to better match CPU debug warm behavior
    // in regions that are not computed due to leading NaNs.
    for (int i = 0; i < series_len; ++i) { out_row[i] = 0.0f; }
    const int warm = period < series_len ? period : series_len;
    for (int i = 0; i < warm; ++i) { out_row[i] = 0.0f; }

    // Coefficients in double to reduce FP error vs CPU f64
    int half_period_i = period / 2; if (half_period_i < 1) half_period_i = 1;
    const double half_p = static_cast<double>(half_period_i);
    const double a = exp(-REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double a2 = a * a;
    const double b = 2.0 * a * cos(REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double c = 0.5 * (1.0 + a2 - b);

    // Ring buffer (period+1) in dynamic shared memory (double for accuracy)
    extern __shared__ double sdata[];
    double* ring = sdata; // size >= period+1 per wrapper
    const int ring_len = period + 1;
    if (series_len > 0) ring[0] = static_cast<double>(prices[0]);
    if (series_len > 1) ring[1] = static_cast<double>(prices[1]);

    // Rolling sum of last `period` ssf values before including ssf[i]
    double ssf_sum = 0.0;
    double ssf_c = 0.0; // Kahan compensator for ssf_sum
    if (period == 1) {
        ssf_sum = (series_len > 0) ? ring[0] : 0.0;
    } else {
        ssf_sum = ((series_len > 0) ? ring[0] : 0.0)
                + ((series_len > 1) ? ring[1] : 0.0);
    }
    const double inv_p = 1.0 / static_cast<double>(period);
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta  = 1.0 - alpha;
    double ms = 0.0;

    for (int i = 2; i < series_len; ++i) {
        const int idx      = i % ring_len;
        const int idx_im1  = (i - 1) % ring_len;
        const int idx_im2  = (i - 2) % ring_len;

        const double di   = static_cast<double>(prices[i]);
        const double dim1 = static_cast<double>(prices[i - 1]);
        const double ssf_im1 = ring[idx_im1];
        const double ssf_im2 = ring[idx_im2];

        const double t0 = c * (di + dim1);
        const double t1 = fma(-a2, ssf_im2, t0);
        const double ssf_i = fma(b, ssf_im1, t1);

        ring[idx] = ssf_i;

        if (i < period) {
            ssf_sum += ssf_i;
            continue;
        }

        const int idx_ip = (i - period) % ring_len;
        const double ssf_ip = ring[idx_ip];
        const double mean_lp = ssf_sum * inv_p;
        const double my_sum = beta * ssf_i + alpha * ssf_ip - mean_lp;

        ms = fma(0.96, ms, 0.04 * my_sum * my_sum);
        if (ms > 0.0 && isfinite(ms)) {
            out_row[i] = static_cast<float>(my_sum / sqrt(ms));
        }

        // Kahan-compensated update: ssf_sum += (ssf_i - ssf_ip)
        {
            double y = (ssf_i - ssf_ip) - ssf_c;
            double t2 = ssf_sum + y;
            ssf_c = (t2 - ssf_sum) - y;
            ssf_sum = t2;
        }
    }
}

extern "C" __global__
void reflex_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                      int period,
                                      int num_series,
                                      int series_len,
                                      const int* __restrict__ /*first_valids (unused)*/,
                                      float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || threadIdx.x != 0) { return; }
    if (period < 2 || series_len <= 0) { return; }

    // Initialize outputs to 0.0 for consistency with batch behavior above.
    for (int t = 0; t < series_len; ++t) { out_tm[t * num_series + series] = 0.0f; }
    const int warm = period < series_len ? period : series_len;
    for (int t = 0; t < warm; ++t) { out_tm[t * num_series + series] = 0.0f; }

    int half_period_i = period / 2; if (half_period_i < 1) half_period_i = 1;
    const double half_p = static_cast<double>(half_period_i);
    const double a = exp(-REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double a2 = a * a;
    const double b = 2.0 * a * cos(REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double c = 0.5 * (1.0 + a2 - b);

    extern __shared__ double sdata[];
    double* ring = sdata; // period+1
    const int ring_len = period + 1;
    if (series_len > 0) ring[0] = static_cast<double>(prices_tm[0 * num_series + series]);
    if (series_len > 1) ring[1] = static_cast<double>(prices_tm[1 * num_series + series]);

    double ssf_sum = 0.0;
    double ssf_c = 0.0; // Kahan compensator for ssf_sum
    if (period == 1) {
        ssf_sum = (series_len > 0) ? ring[0] : 0.0;
    } else {
        ssf_sum = ((series_len > 0) ? ring[0] : 0.0)
                + ((series_len > 1) ? ring[1] : 0.0);
    }
    const double inv_p = 1.0 / static_cast<double>(period);
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta  = 1.0 - alpha;
    double ms = 0.0;

    for (int t = 2; t < series_len; ++t) {
        const int idx     = t % ring_len;
        const int idx_im1 = (t - 1) % ring_len;
        const int idx_im2 = (t - 2) % ring_len;

        const double di   = static_cast<double>(prices_tm[t * num_series + series]);
        const double dim1 = static_cast<double>(prices_tm[(t - 1) * num_series + series]);
        const double ssf_im1 = ring[idx_im1];
        const double ssf_im2 = ring[idx_im2];

        const double t0 = c * (di + dim1);
        const double t1 = fma(-a2, ssf_im2, t0);
        const double ssf_t = fma(b, ssf_im1, t1);
        ring[idx] = ssf_t;

        if (t < period) { ssf_sum += ssf_t; continue; }

        const int idx_ip = (t - period) % ring_len;
        const double ssf_ip = ring[idx_ip];
        const double mean_lp = ssf_sum * inv_p;
        const double my_sum = beta * ssf_t + alpha * ssf_ip - mean_lp;

        ms = fma(0.96, ms, 0.04 * my_sum * my_sum);
        if (ms > 0.0 && isfinite(ms)) {
            out_tm[t * num_series + series] = static_cast<float>(my_sum / sqrt(ms));
        }

        // Kahan-compensated update: ssf_sum += (ssf_t - ssf_ip)
        {
            double y = (ssf_t - ssf_ip) - ssf_c;
            double t2 = ssf_sum + y;
            ssf_c = (t2 - ssf_sum) - y;
            ssf_sum = t2;
        }
    }
}
