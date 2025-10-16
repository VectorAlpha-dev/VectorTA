// CUDA kernels for Decycler Oscillator (DEC_OSC)
//
// Semantics match src/indicators/dec_osc.rs (scalar path):
// - Two cascaded 2‑pole high‑pass sections (Q≈0.707) with half‑period for the second stage
// - Warmup prefix: indices [0 .. first_valid+2) = NaN
// - Output is 100*k * (osc / price)
// - FP32 I/O, FP64 internal math for stability; uses sincos() and fma

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Compute 2-pole HPF section coefficients for Ehlers Q≈0.707 at a given period.
// angle = 2*pi*0.707/period
static __forceinline__ __device__ void hp2_coeffs(double period, double &c, double &two_oma, double &oma_sq) {
    const double angle = (2.0 * CUDART_PI) * 0.707 / period;
    double s, co;
    sincos(angle, &s, &co);
    const double alpha = 1.0 + ((s - 1.0) / co);
    const double t = 1.0 - 0.5 * alpha;
    c = t * t;
    const double oma = 1.0 - alpha;
    two_oma = 2.0 * oma;
    oma_sq = oma * oma;
}

// One-series × many-params (batch). Grid-stride over combos so each thread
// processes one row (sequential over time) for good occupancy.
extern "C" __global__ void dec_osc_batch_f32(
    const float* __restrict__ prices,   // [series_len]
    const int*   __restrict__ periods,  // [n_combos]
    const float* __restrict__ ks,       // [n_combos]
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out             // [n_combos * series_len]
) {
    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += blockDim.x * gridDim.x) {
        const int base   = combo * series_len;
        const int period = periods[combo];
        const float kf   = ks[combo];

        // Fill warmup prefix with NaNs; we don't need to nuke entire row
        for (int i = 0; i < min(series_len, first_valid + 2); ++i) {
            out[base + i] = NAN;
        }

        if (UNLIKELY(period < 2 || period > series_len)) continue;
        if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) continue;
        const int tail = series_len - first_valid;
        if (UNLIKELY(tail < 2)) continue;

        const double p  = static_cast<double>(period);
        const double hp = 0.5 * p;

        double c1, two_oma1, oma1_sq;
        double c2, two_oma2, oma2_sq;
        hp2_coeffs(p,  c1, two_oma1, oma1_sq);
        hp2_coeffs(hp, c2, two_oma2, oma2_sq);

        const double scale = 100.0 * static_cast<double>(kf);

        // Seed states
        const int i0 = first_valid;
        const int i1 = first_valid + 1;
        if (i1 >= series_len) continue;
        double x2 = static_cast<double>(prices[i0]);
        double x1 = static_cast<double>(prices[i1]);
        double hp_prev_2 = x2;
        double hp_prev_1 = x1;
        double decosc_prev_2 = 0.0;
        double decosc_prev_1 = 0.0;

        for (int i = first_valid + 2; i < series_len; ++i) {
            const double d0 = static_cast<double>(prices[i]);
            const double dx  = d0 - 2.0 * x1 + x2;
            const double hp0 = fma(c1, dx, fma(two_oma1, hp_prev_1, -oma1_sq * hp_prev_2));

            const double dec    = d0 - hp0;
            const double d_dec1 = x1 - hp_prev_1;
            const double d_dec2 = x2 - hp_prev_2;
            const double decdx  = dec - 2.0 * d_dec1 + d_dec2;
            const double osc0   = fma(c2, decdx, fma(two_oma2, decosc_prev_1, -oma2_sq * decosc_prev_2));

            out[base + i] = static_cast<float>(scale * (osc0 / d0));

            hp_prev_2 = hp_prev_1;
            hp_prev_1 = hp0;
            decosc_prev_2 = decosc_prev_1;
            decosc_prev_1 = osc0;
            x2 = x1;
            x1 = d0;
        }
    }
}

// Many-series × one-param (time-major). Each thread handles one series (column).
extern "C" __global__ void dec_osc_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,  // [t * num_series + s]
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float k,
    float* __restrict__ out_tm           // [t * num_series + s]
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    // Prefill with NaN
    for (int t = 0; t < series_len; ++t) {
        out_tm[t * num_series + s] = NAN;
    }

    if (UNLIKELY(period < 2 || period > series_len)) return;
    const int first = first_valids[s];
    if (UNLIKELY(first < 0 || first >= series_len)) return;
    const int tail = series_len - first;
    if (UNLIKELY(tail < 2)) return;

    const double p  = static_cast<double>(period);
    const double hp = 0.5 * p;

    double c1, two_oma1, oma1_sq;
    double c2, two_oma2, oma2_sq;
    hp2_coeffs(p,  c1, two_oma1, oma1_sq);
    hp2_coeffs(hp, c2, two_oma2, oma2_sq);

    const double scale = 100.0 * static_cast<double>(k);

    const int i0 = first;
    const int i1 = first + 1;
    if (i1 >= series_len) return;
    auto load_tm = [&](int t) { return static_cast<double>(prices_tm[t * num_series + s]); };
    auto store_tm = [&](int t, float v) { out_tm[t * num_series + s] = v; };

    double x2 = load_tm(i0);
    double x1 = load_tm(i1);
    double hp_prev_2 = x2;
    double hp_prev_1 = x1;
    double decosc_prev_2 = 0.0;
    double decosc_prev_1 = 0.0;

    for (int t = first + 2; t < series_len; ++t) {
        const double d0 = load_tm(t);
        const double dx  = d0 - 2.0 * x1 + x2;
        const double hp0 = fma(c1, dx, fma(two_oma1, hp_prev_1, -oma1_sq * hp_prev_2));

        const double dec    = d0 - hp0;
        const double d_dec1 = x1 - hp_prev_1;
        const double d_dec2 = x2 - hp_prev_2;
        const double decdx  = dec - 2.0 * d_dec1 + d_dec2;
        const double osc0   = fma(c2, decdx, fma(two_oma2, decosc_prev_1, -oma2_sq * decosc_prev_2));

        store_tm(t, static_cast<float>(scale * (osc0 / d0)));

        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp0;
        decosc_prev_2 = decosc_prev_1;
        decosc_prev_1 = osc0;
        x2 = x1;
        x1 = d0;
    }
}
