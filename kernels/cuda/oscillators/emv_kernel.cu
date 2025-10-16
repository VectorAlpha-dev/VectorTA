// CUDA kernels for Ease of Movement (EMV)
//
// Semantics mirror the scalar Rust implementation in src/indicators/emv.rs:
// - EMV = ( (mid_t - mid_{t-1}) ) / ( volume_t / 10000 / range_t )
//       = (mid delta) * range_t * 10000 / volume_t
//   where mid_t = 0.5 * (high_t + low_t) and range_t = high_t - low_t
// - Warmup: outputs [0 .. first_valid] are NaN, first defined output is at
//   index warm = first_valid + 1.
// - NaN handling: if any of high/low/volume at t is NaN, output NaN and do not
//   update the carried state (last_mid). On zero range, output NaN and advance
//   last_mid = mid_t (matches scalar behavior).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef EMV_NAN
#define EMV_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// One-series × many-params (EMV has no params; n_combos will be 1)
extern "C" __global__ void emv_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ volume,
    int series_len,
    int n_combos,      // unused for EMV (no parameters); kept for parity
    int first_valid,
    float* __restrict__ out // length = n_combos * series_len
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* row = out + combo * series_len;

    if (UNLIKELY(series_len <= 0 || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) row[i] = EMV_NAN;
        return;
    }

    const int warm = first_valid + 1; // first defined output index

    // Prefill warmup region with NaNs
    for (int i = 0; i < warm && i < series_len; ++i) row[i] = EMV_NAN;

    // Seed last_mid from first_valid
    double last_mid = 0.5 * (static_cast<double>(high[first_valid]) + static_cast<double>(low[first_valid]));

    for (int i = warm; i < series_len; ++i) {
        const float hf = high[i];
        const float lf = low[i];
        const float vf = volume[i];

        if (UNLIKELY(isnan(hf) || isnan(lf) || isnan(vf))) {
            row[i] = EMV_NAN;
            continue; // do not update last_mid
        }

        const double h = static_cast<double>(hf);
        const double l = static_cast<double>(lf);
        const double v = static_cast<double>(vf);
        const double current_mid = 0.5 * (h + l);
        const double range = h - l;
        if (UNLIKELY(range == 0.0)) {
            row[i] = EMV_NAN;
            last_mid = current_mid; // advance state on zero-range
            continue;
        }
        const double br = v / 10000.0 / range;
        row[i] = static_cast<float>((current_mid - last_mid) / br);
        last_mid = current_mid;
    }
}

// Many-series × one-param (time-major): EMV has no params; compute per series.
// Layout: input and output are time-major: buf[t * num_series + s]
extern "C" __global__ void emv_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ volume_tm,
    const int*   __restrict__ first_valids, // per-series
    int num_series,
    int series_len,
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    const int fv = first_valids[s];
    if (UNLIKELY(series_len <= 0 || fv < 0 || fv >= series_len)) {
        float* o = out_tm + s;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = EMV_NAN;
        return;
    }

    const int warm = fv + 1;
    // Prefill [0 .. warm-1] with NaN
    {
        float* o = out_tm + s;
        for (int r = 0; r < warm && r < series_len; ++r, o += num_series) *o = EMV_NAN;
    }

    // Seed last_mid from row=fv
    double h0 = static_cast<double>(*(high_tm   + static_cast<size_t>(fv) * num_series + s));
    double l0 = static_cast<double>(*(low_tm    + static_cast<size_t>(fv) * num_series + s));
    double last_mid = 0.5 * (h0 + l0);

    for (int r = warm; r < series_len; ++r) {
        const float hf = *(high_tm   + static_cast<size_t>(r) * num_series + s);
        const float lf = *(low_tm    + static_cast<size_t>(r) * num_series + s);
        const float vf = *(volume_tm + static_cast<size_t>(r) * num_series + s);
        float* out_elem = out_tm + static_cast<size_t>(r) * num_series + s;

        if (UNLIKELY(isnan(hf) || isnan(lf) || isnan(vf))) {
            *out_elem = EMV_NAN;
            continue; // keep last_mid
        }
        const double h = static_cast<double>(hf);
        const double l = static_cast<double>(lf);
        const double v = static_cast<double>(vf);
        const double current_mid = 0.5 * (h + l);
        const double range = h - l;
        if (UNLIKELY(range == 0.0)) {
            *out_elem = EMV_NAN;
            last_mid = current_mid;
            continue;
        }
        const double br = v / 10000.0 / range;
        *out_elem = static_cast<float>((current_mid - last_mid) / br);
        last_mid = current_mid;
    }
}

