// CUDA kernels for Chande Momentum Oscillator (CMO)
//
// Semantics mirror the scalar implementation:
// - Warmup index: warm = first_valid + period
// - Outputs before warm are NaN
// - First valid output at warm computed from average gains/losses over the
//   initial window
// - Subsequent outputs use Wilder-style rolling update:
//     avg = ((avg * (period - 1)) + x) / period
// - Division-by-zero in (avg_gain + avg_loss) yields 0.0 (matches scalar)

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef CMO_NAN
#define CMO_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

extern "C" __global__ void cmo_batch_f32(
    const float*  __restrict__ prices,    // one series (FP32)
    const int*    __restrict__ periods,   // length = n_combos
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out              // length = n_combos * series_len
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base = combo * series_len;
    float* out_row = out + base;

    // Basic validation mirroring wrapper guards
    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = CMO_NAN;
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail <= period)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = CMO_NAN;
        return;
    }

    const int warm = first_valid + period; // first index with a defined output
    const double inv_p = 1.0 / static_cast<double>(period);
    const double pm1 = static_cast<double>(period - 1);

    // Prefill NaN up to (warm-1)
    for (int i = 0; i < warm; ++i) out_row[i] = CMO_NAN;

    // Initial averages from prices over (first_valid+1 ..= warm)
    double sum_g = 0.0;
    double sum_l = 0.0;
    double prev = static_cast<double>(prices[first_valid]);
    for (int i = first_valid + 1; i <= warm; ++i) {
        const double curr = static_cast<double>(prices[i]);
        const double diff = curr - prev;
        prev = curr;
        const double ad = fabs(diff);
        sum_g += 0.5 * (ad + diff);
        sum_l += 0.5 * (ad - diff);
    }
    double avg_g = sum_g * inv_p;
    double avg_l = sum_l * inv_p;
    {
        const double denom = avg_g + avg_l;
        out_row[warm] = (denom != 0.0) ? static_cast<float>(100.0 * ((avg_g - avg_l) / denom)) : 0.0f;
    }
    // Rolling update using Wilder formula
    for (int i = warm + 1; i < series_len; ++i) {
        const double curr = static_cast<double>(prices[i]);
        const double diff = curr - prev;
        prev = curr;
        const double ad = fabs(diff);
        const double g = 0.5 * (ad + diff);
        const double l = 0.5 * (ad - diff);
        avg_g = (avg_g * pm1 + g) * inv_p;
        avg_l = (avg_l * pm1 + l) * inv_p;
        const double denom = avg_g + avg_l;
        out_row[i] = (denom != 0.0) ? static_cast<float>(100.0 * ((avg_g - avg_l) / denom)) : 0.0f;
    }
}

// Many-series × one-param, time-major layout
// prices_tm: [row * num_series + series]
extern "C" __global__ void cmo_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm // time-major
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(period <= 0 || period > series_len || fv < 0 || fv >= series_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = CMO_NAN;
        return;
    }
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= period)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = CMO_NAN;
        return;
    }

    const int warm = fv + period;
    const double inv_p = 1.0 / static_cast<double>(period);
    const double pm1 = static_cast<double>(period - 1);

    // Prefill NaN up to warm-1
    {
        float* o = out_tm + series;
        for (int r = 0; r < warm; ++r, o += num_series) *o = CMO_NAN;
    }

    // Seed with initial averages across (fv+1 ..= warm)
    double prev = static_cast<double>(*(prices_tm + static_cast<size_t>(fv) * num_series + series));
    double sum_g = 0.0;
    double sum_l = 0.0;
    for (int r = fv + 1; r <= warm; ++r) {
        const double curr = static_cast<double>(*(prices_tm + static_cast<size_t>(r) * num_series + series));
        const double diff = curr - prev;
        prev = curr;
        const double ad = fabs(diff);
        const double g = 0.5 * (ad + diff);
        const double l = 0.5 * (ad - diff);
        sum_g += g;
        sum_l += l;
    }
    double avg_g = sum_g * inv_p;
    double avg_l = sum_l * inv_p;
    *(out_tm + static_cast<size_t>(warm) * num_series + series) = [&]() {
        const double denom = avg_g + avg_l;
        return (denom != 0.0) ? static_cast<float>(100.0 * ((avg_g - avg_l) / denom)) : 0.0f;
    }();

    // Rolling update across remaining rows
    for (int r = warm + 1; r < series_len; ++r) {
        const double curr = static_cast<double>(*(prices_tm + static_cast<size_t>(r) * num_series + series));
        const double diff = curr - prev;
        prev = curr;
        const double ad = fabs(diff);
        const double g = 0.5 * (ad + diff);
        const double l = 0.5 * (ad - diff);
        avg_g = (avg_g * pm1 + g) * inv_p;
        avg_l = (avg_l * pm1 + l) * inv_p;
        const double denom = avg_g + avg_l;
        *(out_tm + static_cast<size_t>(r) * num_series + series) =
            (denom != 0.0) ? static_cast<float>(100.0 * ((avg_g - avg_l) / denom)) : 0.0f;
    }
}
