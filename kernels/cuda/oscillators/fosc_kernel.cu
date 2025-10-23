// CUDA kernels for Forecast Oscillator (FOSC)
//
// FP32-first, compensated-accumulation rewrite for Ada/Lovelace (e.g., RTX 4090).
// Key changes vs previous version:
// - One thread per combo/series (1D launch) – no idle lanes, no __syncthreads().
// - Rolling accumulators (Σy, Σxy) use Kahan/Neumaier compensated FP32.
// - Use FMA (fmaf) in critical expressions to reduce rounding.
// - Warmup/NaN semantics preserved.
// - No FP64 in hot loops to avoid 1/64 FP64 throughput on GA10x/AD10x.

#include <cuda_runtime.h>
#include <math.h>

// IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// Using FP64 for running sums to match scalar parity

// === One-series × many-params (batch). Row-major output [combo][t] ===
extern "C" __global__ void fosc_batch_f32(
    const float* __restrict__ data,   // [len]
    int len,
    int first_valid,
    const int* __restrict__ periods,  // [n_combos]
    int n_combos,
    float* __restrict__ out)          // [n_combos][len] row-major
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;

    // Fill warmup prefix with NaN (single thread per combo)
    const int warm_end = (warm < len) ? warm : len;
    for (int t = 0; t < warm_end; ++t) {
        out[row_off + t] = f32_nan();
    }
    if (warm >= len) return;

    // OLS constants in FP64 for robust solve
    const double p   = (double)period;
    const double p1  = p + 1.0;
    const double inv_p = 1.0 / p;
    const double sx  = 0.5 * p * p1;                              // Σx
    const double sx2 = (p * p1 * (2.0 * p + 1.0)) / 6.0;          // Σx^2
    const double den = p * sx2 - sx * sx;                         // p*Σx^2 - (Σx)^2
    const double inv_den = (fabs(den) < 1e-18) ? 0.0 : (1.0 / den);

    // Initialize running sums over [first_valid .. first_valid + period - 2]
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double w = 1.0;
    for (int k = 0; k < period - 1; ++k, w += 1.0f) {
        const double d = (double)data[first_valid + k];
        sum_y += d;
        sum_xy = fma(d, w, sum_xy);
    }

    // TSF forecast for "next bar" carried to next loop iteration
    double tsf_prev = 0.0;

    for (int t = warm; t < len; ++t) {
        const float cur = data[t];          // new value for this time
        const double y_plus = sum_y + (double)cur;
        const double xy_plus = sum_xy + (double)cur * p;

        // OLS slope & intercept in FP64
        const double b = (p * xy_plus - sx * y_plus) * inv_den;
        const double a = (y_plus - b * sx) * inv_p;

        // FOSC: 100 * (1 - forecast/close). Use previous forecast for this bar.
        float out_val;
        if ((cur == cur) && cur != 0.0f) {
            // Compute in FP64 then cast for improved parity
            const double cd = (double)cur;
            const double ov = 100.0 * ((cd - tsf_prev) / cd);
            out_val = (float)ov;
        } else {
            out_val = f32_nan();
        }
        out[row_off + t] = out_val;

        // Forecast for "next" bar (used in the next iteration)
        tsf_prev = b * p1 + a;

        // Slide window by 1 using compensated updates
        const int old_idx = t + 1 - period;
        const float oldv = data[old_idx];

        // Maintain Σxy over the new window:
        //   xy_plus = sum_xy + newv*p
        //   sum_xy_next = xy_plus - y_plus
        sum_xy = xy_plus - y_plus;

        // Maintain Σy over the new window
        sum_y = y_plus - oldv;
    }
}

// === Many-series × one-param (time-major). Data is time-major [t][series] ===
extern "C" __global__ void fosc_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,     // [rows][cols] time-major
    const int* __restrict__ first_valids,  // [cols]
    int cols,
    int rows,
    int period,
    float* __restrict__ out_tm)            // [rows][cols] time-major
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // one thread per series
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int warm = fv + period - 1;

    // Fill warmup prefix with NaN for this series
    const int warm_end = (warm < rows) ? warm : rows;
    for (int t = 0; t < warm_end; ++t) {
        out_tm[t * cols + s] = f32_nan();
    }
    if (warm >= rows) return;

    // OLS constants in FP64 for robust solve
    const double p   = (double)period;
    const double p1  = p + 1.0;
    const double inv_p = 1.0 / p;
    const double sx  = 0.5 * p * p1;
    const double sx2 = (p * p1 * (2.0 * p + 1.0)) / 6.0;
    const double den = p * sx2 - sx * sx;
    const double inv_den = (fabs(den) < 1e-18) ? 0.0 : (1.0 / den);

    // Initialize sums for the first full window (except newest)
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double w = 1.0;
    for (int k = 0; k < period - 1; ++k, w += 1.0f) {
        const double d = (double)data_tm[(fv + k) * cols + s];
        sum_y += d;
        sum_xy = fma(d, w, sum_xy);
    }

    double tsf_prev = 0.0;
    for (int t = warm; t < rows; ++t) {
        const float cur = data_tm[t * cols + s];
        const double y_plus = sum_y + (double)cur;
        const double xy_plus = sum_xy + (double)cur * p;

        const double b = (p * xy_plus - sx * y_plus) * inv_den;
        const double a = (y_plus - b * sx) * inv_p;

        float out_val;
        if ((cur == cur) && cur != 0.0f) {
            const double cd = (double)cur;
            const double ov = 100.0 * ((cd - tsf_prev) / cd);
            out_val = (float)ov;
        } else {
            out_val = f32_nan();
        }
        out_tm[t * cols + s] = out_val;

        tsf_prev = b * p1 + a;

        const int old_idx = t + 1 - period;
        const float oldv = data_tm[old_idx * cols + s];

        sum_xy = xy_plus - y_plus;

        sum_y = y_plus - oldv;
    }
}

