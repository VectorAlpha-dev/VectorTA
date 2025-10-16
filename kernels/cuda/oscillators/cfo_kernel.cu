// CUDA kernels for Chande Forecast Oscillator (CFO)
//
// Math pattern: prefix-sum/rational.
// Host builds prefix sums over the valid segment and passes them to the kernel.
// Kernel computes O(1) per-output window statistics for each param row.
//
// Semantics:
// - Warmup per row: warm = first_valid + period - 1
// - Warmup prefix filled with NaN
// - If current value is NaN or zero: write NaN
// - Accumulations in float64; outputs in float32

#include <cuda_runtime.h>
#include <math.h>

// Helper to write an IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// One-series × many-params (batch). Uses host-provided prefix sums P (sum_y) and
// Q (sum_{k=1..j} k*y_k) over the valid segment that starts at first_valid.
// Periods and scalars arrays are length n_combos. Output is row-major
// [combo][t] (n_combos x len).
extern "C" __global__ void cfo_batch_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,      // length = len - first_valid + 1 (but passed as linear over whole len; kernel indexes with absolute t)
    const double* __restrict__ prefix_weighted, // same layout as prefix_sum
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ scalars,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float scalar = scalars[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;

    // OLS constants for x = 1..n
    const double n = (double)period;
    const double sx = (double)(period * (period + 1)) * 0.5; // Σx
    const double sx2 = (double)(period * (period + 1) * (2 * period + 1)) / 6.0; // Σx^2
    const double inv_denom = 1.0 / (n * sx2 - sx * sx);
    const double half_nm1 = 0.5 * (n - 1.0);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const float nanf = f32_nan();
    while (t < len) {
        float out_val = nanf;
        if (t >= warm) {
            // Map absolute index t (in full series) to valid-segment indices
            const int idx = t - first_valid;       // 0-based within valid segment
            const int r1 = idx + 1;                // 1-based right endpoint for prefix
            const int l1_minus1 = r1 - period;     // (left1 - 1)

            // prefix arrays are stored starting at absolute 0 with zeros up to first_valid
            const double sum_y = prefix_sum[first_valid + r1] - prefix_sum[first_valid + l1_minus1];
            const double sum_xy_raw = prefix_weighted[first_valid + r1] - prefix_weighted[first_valid + l1_minus1];
            const double sum_xy = sum_xy_raw - ((double)l1_minus1) * sum_y;

            const double b = (-sx) * sum_y + n * sum_xy;
            const double b_scaled = b * inv_denom;
            const double f = b_scaled * half_nm1 + sum_y / n;
            const float cur = data[t];
            if (!isnan(cur) && cur != 0.0f) {
                // CFO = scalar * (1 - f/cur)
                out_val = scalar * (1.0f - (float)(f / (double)cur));
            } else {
                out_val = nanf;
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// Many-series × one-param (time-major). Data layout: time-major with shape
// rows x cols. Prefix arrays P and Q are time-major with +1 length per element
// (i.e., stored at linear index (t*cols + s) + 1). first_valids has one entry
// per series/column.
extern "C" __global__ void cfo_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,
    const double* __restrict__ prefix_sum_tm,
    const double* __restrict__ prefix_weighted_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float scalar,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series/column
    const int tx = blockIdx.x * blockDim.x + threadIdx.x; // time index iterator
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int warm = fv + period - 1;

    // OLS constants
    const double n = (double)period;
    const double sx = (double)(period * (period + 1)) * 0.5;
    const double sx2 = (double)(period * (period + 1) * (2 * period + 1)) / 6.0;
    const double inv_denom = 1.0 / (n * sx2 - sx * sx);
    const double half_nm1 = 0.5 * (n - 1.0);

    const int stride = gridDim.x * blockDim.x; // stride over time dimension
    const float nanf = f32_nan();
    for (int t = tx; t < rows; t += stride) {
        float out_val = nanf;
        if (t >= warm) {
            const int idx_valid = t - fv;       // 0-based within valid segment
            const int r1 = idx_valid + 1;       // 1-based
            const int l1_minus1 = r1 - period;  // (left1 - 1)

            const int wr = (t * cols + s) + 1;
            const int wl = ((t - period) * cols + s) + 1; // since t >= warm, (t - period) >= (fv - 1)

            const double sum_y = prefix_sum_tm[wr] - prefix_sum_tm[wl];
            const double sum_xy_raw = prefix_weighted_tm[wr] - prefix_weighted_tm[wl];
            const double sum_xy = sum_xy_raw - ((double)l1_minus1) * sum_y;

            const double b = (-sx) * sum_y + n * sum_xy;
            const double b_scaled = b * inv_denom;
            const double f = b_scaled * half_nm1 + sum_y / n;
            const float cur = data_tm[t * cols + s];
            if (!isnan(cur) && cur != 0.0f) {
                out_val = scalar * (1.0f - (float)(f / (double)cur));
            } else {
                out_val = nanf;
            }
        }
        out_tm[t * cols + s] = out_val;
    }
}

