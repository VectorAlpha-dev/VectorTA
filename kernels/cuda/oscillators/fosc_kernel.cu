// CUDA kernels for Forecast Oscillator (FOSC)
//
// Math pattern: recurrence/IIR per row/series.
// - Batch (one series × many params): each row (period) is processed by a
//   single thread that performs a sequential time scan using O(1) rolling
//   updates of the OLS accumulators. No global prefixes are required.
// - Many-series × one-param (time-major): each series/column is processed by a
//   single thread with the same sequential scan.
//
// Semantics (must match scalar):
// - Warmup per row/series: warm = first_valid + period - 1
// - Warmup prefix is filled with NaN
// - If current value is NaN or 0.0, output is NaN
// - Accumulations use float64; outputs are float32

#include <cuda_runtime.h>
#include <math.h>

// Helper to write an IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// One-series × many-params (batch). Row-major output [combo][t]
extern "C" __global__ void fosc_batch_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;

    // Fill warmup prefix with NaN
    for (int t = threadIdx.x; t < min(warm, len); t += blockDim.x) {
        out[row_off + t] = f32_nan();
    }
    __syncthreads();

    // Only a single thread performs the sequential scan to respect the
    // recurrence dependency across time.
    if (threadIdx.x != 0) return;

    if (warm >= len) return;

    // OLS constants for x = 1..=p
    const double p = (double)period;
    const double sx = 0.5 * p * (p + 1.0);                          // Σx
    const double sx2 = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0;      // Σx^2
    const double den = p * sx2 - sx * sx;
    const double inv_den = (fabs(den) < 1e-18) ? 0.0 : (1.0 / den);
    const double inv_p = 1.0 / p;
    const double p1 = p + 1.0;

    // Initialize running sums over [first_valid .. first_valid + period - 2]
    double sum_y = 0.0;
    double sum_xy = 0.0;
    for (int k = 0; k < period - 1; ++k) {
        const double d = (double)data[first_valid + k];
        sum_y += d;
        sum_xy = fma(d, (double)(k + 1), sum_xy);
    }

    double tsf_prev = 0.0; // matches scalar initialization
    for (int t = warm; t < len; ++t) {
        const double newv = (double)data[t];
        const double y_plus = sum_y + newv;
        const double xy_plus = sum_xy + newv * p;
        const double b = (p * xy_plus - sx * y_plus) * inv_den;
        const double a = (y_plus - b * sx) * inv_p;

        float out_val;
        const float cur = data[t];
        if (!isnan(cur) && cur != 0.0f) {
            // 100 * (1 - tsf_prev / cur)
            out_val = (float)(100.0 * (1.0 - tsf_prev / (double)cur));
        } else {
            out_val = f32_nan();
        }
        out[row_off + t] = out_val;

        tsf_prev = b * p1 + a; // forecast for next bar (used on next iteration)

        const int old_idx = t + 1 - period;
        const double oldv = (double)data[old_idx];
        // Maintain sums for the window [t - (p-1) .. t]
        sum_xy = xy_plus - y_plus; // subtract Σy to remove shift in weights
        sum_y = y_plus - oldv;
    }
}

// Many-series × one-param (time-major). Data/layout is time-major [t][series]
extern "C" __global__ void fosc_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series/column index
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int warm = fv + period - 1;

    // Fill warmup prefix with NaN for this series
    for (int t = threadIdx.x; t < min(warm, rows); t += blockDim.x) {
        out_tm[t * cols + s] = f32_nan();
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single thread scans time for this series
    if (warm >= rows) return;

    // OLS constants for x = 1..=p
    const double p = (double)period;
    const double sx = 0.5 * p * (p + 1.0);
    const double sx2 = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0;
    const double den = p * sx2 - sx * sx;
    const double inv_den = (fabs(den) < 1e-18) ? 0.0 : (1.0 / den);
    const double inv_p = 1.0 / p;
    const double p1 = p + 1.0;

    // Initialize sums for the first full window (except newest)
    double sum_y = 0.0;
    double sum_xy = 0.0;
    for (int k = 0; k < period - 1; ++k) {
        const double d = (double)data_tm[(fv + k) * cols + s];
        sum_y += d;
        sum_xy = fma(d, (double)(k + 1), sum_xy);
    }

    double tsf_prev = 0.0;
    for (int t = warm; t < rows; ++t) {
        const double newv = (double)data_tm[t * cols + s];
        const double y_plus = sum_y + newv;
        const double xy_plus = sum_xy + newv * p;
        const double b = (p * xy_plus - sx * y_plus) * inv_den;
        const double a = (y_plus - b * sx) * inv_p;

        float out_val;
        const float cur = (float)newv;
        if (!isnan(cur) && cur != 0.0f) {
            out_val = (float)(100.0 * (1.0 - tsf_prev / (double)cur));
        } else {
            out_val = f32_nan();
        }
        out_tm[t * cols + s] = out_val;

        tsf_prev = b * p1 + a;

        const int old_idx = t + 1 - period;
        const double oldv = (double)data_tm[old_idx * cols + s];
        sum_xy = xy_plus - y_plus;
        sum_y = y_plus - oldv;
    }
}

