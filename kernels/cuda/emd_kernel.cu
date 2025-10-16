// CUDA kernels for the Empirical Mode Decomposition (EMD) indicator.
//
// Math category: Recurrence/IIR with short fixed-window (50) and a variable
// window (2*period) moving average. We compute the band-pass filtered series
// sequentially per thread and use the output arrays themselves as scratch to
// maintain rolling sums in O(1): we write the raw series value first, and once
// the window is full we overwrite the same location with the averaged value,
// subtracting the element at t-window which still holds the raw value.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef EMD_NAN
#define EMD_NAN (__int_as_float(0x7fffffff))
#endif

// Prefill a row (length n) with NaNs.
__device__ inline void fill_nan_row(float* __restrict__ row, int n) {
    for (int i = 0; i < n; ++i) row[i] = EMD_NAN;
}

// ---------- One series × many params (batch) ----------
extern "C" __global__ void emd_batch_f32(
    const float* __restrict__ prices,   // midpoint price series, length = series_len
    const int*   __restrict__ periods,  // len = n_combos
    const float* __restrict__ deltas,   // len = n_combos
    const float* __restrict__ fractions,// len = n_combos
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ ub_out,         // flattened [combo * series_len + t]
    float* __restrict__ mb_out,
    float* __restrict__ lb_out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    float* __restrict__ ub_row = ub_out + base;
    float* __restrict__ mb_row = mb_out + base;
    float* __restrict__ lb_row = lb_out + base;

    // Prefill NaN for warmup semantics
    fill_nan_row(ub_row, series_len);
    fill_nan_row(mb_row, series_len);
    fill_nan_row(lb_row, series_len);

    const int   period   = periods[combo];
    const float delta    = deltas[combo];
    const float fraction = fractions[combo];
    if (period <= 0 || first_valid < 0 || first_valid >= series_len) return;

    // Constants for the band-pass filter
    const double two_pi = 6.28318530717958647692; // 2*pi
    const double beta   = cos(two_pi / (double)period);
    const double gamma  = 1.0 / cos(two_pi * 2.0 * (double)delta / (double)period);
    const double alpha  = gamma - sqrt(gamma * gamma - 1.0);
    const double half_one_minus_alpha = 0.5 * (1.0 - alpha);
    const double beta_times_one_plus_alpha = beta * (1.0 + alpha);

    // Rolling sums for 50-window (upper/lower) and 2*period (middle)
    const int per_up_low = 50;
    const int per_mid    = 2 * period;
    const double inv_up_low = 1.0 / (double)per_up_low;
    const double inv_mid    = 1.0 / (double)per_mid;

    double sum_up = 0.0; // sum of scaled peaks
    double sum_low = 0.0; // sum of scaled valleys
    double sum_mid = 0.0; // sum of band-pass values

    // Prior state for band-pass and peak/valley tracking
    double bp_prev1 = 0.0;
    double bp_prev2 = 0.0;
    double peak_prev = 0.0;
    double valley_prev = 0.0;
    double price_prev1 = 0.0;
    double price_prev2 = 0.0;

    int i = first_valid;
    if (i < series_len) {
        const double p0 = (double)prices[i];
        bp_prev1 = p0;
        bp_prev2 = p0;
        peak_prev = p0;
        valley_prev = p0;
        price_prev1 = p0;
        price_prev2 = p0;
    }

    int count = 0; // number of processed points since first_valid
    for (; i < series_len; ++i) {
        const double price = (double)prices[i];

        const double bp_curr = (count >= 2)
            ? (half_one_minus_alpha * (price - price_prev2)
               + beta_times_one_plus_alpha * bp_prev1
               - alpha * bp_prev2)
            : price;

        double peak_curr = peak_prev;
        double valley_curr = valley_prev;
        if (count >= 2) {
            if (bp_prev1 > bp_curr && bp_prev1 > bp_prev2) peak_curr = bp_prev1;
            if (bp_prev1 < bp_curr && bp_prev1 < bp_prev2) valley_curr = bp_prev1;
        }

        const double sp = peak_curr * (double)fraction;
        const double sv = valley_curr * (double)fraction;

        // Use output rows as scratch to keep prior raw values available
        mb_row[i] = (float)bp_curr;
        ub_row[i] = (float)sp;
        lb_row[i] = (float)sv;

        sum_mid += bp_curr;
        sum_up  += sp;
        sum_low += sv;

        const int filled = count + 1;
        if (filled > per_mid) {
            // subtract raw value from t - per_mid (still stored in mb_row)
            const double old_bp = (double)mb_row[i - per_mid];
            sum_mid -= old_bp;
        }
        if (filled > per_up_low) {
            const double old_sp = (double)ub_row[i - per_up_low];
            const double old_sv = (double)lb_row[i - per_up_low];
            sum_up  -= old_sp;
            sum_low -= old_sv;
        }

        if (filled >= per_up_low) {
            ub_row[i] = (float)(sum_up * inv_up_low);
            lb_row[i] = (float)(sum_low * inv_up_low);
        }
        if (filled >= per_mid) {
            mb_row[i] = (float)(sum_mid * inv_mid);
        }

        bp_prev2 = bp_prev1;
        bp_prev1 = bp_curr;
        peak_prev = peak_curr;
        valley_prev = valley_curr;
        price_prev2 = price_prev1;
        price_prev1 = price;
        ++count;
    }
}

// ---------- Many series × one param (time-major) ----------
extern "C" __global__ void emd_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm, // time-major: [t * cols + s]
    int cols,                            // number of series
    int rows,                            // series_len (time)
    int period,
    float delta,
    float fraction,
    const int* __restrict__ first_valids, // len = cols
    float* __restrict__ ub_tm,
    float* __restrict__ mb_tm,
    float* __restrict__ lb_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    // Column pointers for time-major layout
    float* __restrict__ ub_col = ub_tm + series;
    float* __restrict__ mb_col = mb_tm + series;
    float* __restrict__ lb_col = lb_tm + series;

    // Prefill NaN
    for (int t = 0; t < rows; ++t) {
        ub_col[t * cols] = EMD_NAN;
        mb_col[t * cols] = EMD_NAN;
        lb_col[t * cols] = EMD_NAN;
    }

    const int first_valid = first_valids[series];
    if (period <= 0 || first_valid < 0 || first_valid >= rows) return;

    const double two_pi = 6.28318530717958647692;
    const double beta   = cos(two_pi / (double)period);
    const double gamma  = 1.0 / cos(two_pi * 2.0 * (double)delta / (double)period);
    const double alpha  = gamma - sqrt(gamma * gamma - 1.0);
    const double half_one_minus_alpha = 0.5 * (1.0 - alpha);
    const double beta_times_one_plus_alpha = beta * (1.0 + alpha);

    const int per_up_low = 50;
    const int per_mid    = 2 * period;
    const double inv_up_low = 1.0 / (double)per_up_low;
    const double inv_mid    = 1.0 / (double)per_mid;

    double sum_up = 0.0, sum_low = 0.0, sum_mid = 0.0;
    double bp_prev1 = 0.0, bp_prev2 = 0.0;
    double peak_prev = 0.0, valley_prev = 0.0;
    double price_prev1 = 0.0, price_prev2 = 0.0;

    int t = first_valid;
    if (t < rows) {
        const double p0 = (double)prices_tm[(size_t)t * cols + series];
        bp_prev1 = p0; bp_prev2 = p0; peak_prev = p0; valley_prev = p0;
        price_prev1 = p0; price_prev2 = p0;
    }

    int count = 0;
    for (; t < rows; ++t) {
        const double price = (double)prices_tm[(size_t)t * cols + series];

        const double bp_curr = (count >= 2)
            ? (half_one_minus_alpha * (price - price_prev2)
               + beta_times_one_plus_alpha * bp_prev1
               - alpha * bp_prev2)
            : price;

        double peak_curr = peak_prev;
        double valley_curr = valley_prev;
        if (count >= 2) {
            if (bp_prev1 > bp_curr && bp_prev1 > bp_prev2) peak_curr = bp_prev1;
            if (bp_prev1 < bp_curr && bp_prev1 < bp_prev2) valley_curr = bp_prev1;
        }

        const double sp = peak_curr * (double)fraction;
        const double sv = valley_curr * (double)fraction;

        // scratch writes
        mb_col[(size_t)t * cols] = (float)bp_curr;
        ub_col[(size_t)t * cols] = (float)sp;
        lb_col[(size_t)t * cols] = (float)sv;

        sum_mid += bp_curr;
        sum_up  += sp;
        sum_low += sv;

        const int filled = count + 1;
        if (filled > per_mid) {
            const double old_bp = (double)mb_col[(size_t)(t - per_mid) * cols];
            sum_mid -= old_bp;
        }
        if (filled > per_up_low) {
            const double old_sp = (double)ub_col[(size_t)(t - per_up_low) * cols];
            const double old_sv = (double)lb_col[(size_t)(t - per_up_low) * cols];
            sum_up  -= old_sp;
            sum_low -= old_sv;
        }

        if (filled >= per_up_low) {
            ub_col[(size_t)t * cols] = (float)(sum_up * inv_up_low);
            lb_col[(size_t)t * cols] = (float)(sum_low * inv_up_low);
        }
        if (filled >= per_mid) {
            mb_col[(size_t)t * cols] = (float)(sum_mid * inv_mid);
        }

        bp_prev2 = bp_prev1;
        bp_prev1 = bp_curr;
        peak_prev = peak_curr;
        valley_prev = valley_curr;
        price_prev2 = price_prev1;
        price_prev1 = price;
        ++count;
    }
}

