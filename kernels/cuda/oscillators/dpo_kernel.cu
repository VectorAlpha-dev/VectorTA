// CUDA kernels for Detrended Price Oscillator (DPO)
//
// Math pattern: prefix-sum/rational (simple average via prefix sums) with a lagged
// price subtraction. Host builds prefix sums over the valid segment and passes them
// to the kernel. Kernel computes O(1) per-output window ops.
//
// Semantics:
// - Warmup per row: warm = max(first_valid + period - 1, back) where back = period/2 + 1
// - Warmup prefix filled with NaN
// - Accumulations in float64; outputs in float32

#include <cuda_runtime.h>
#include <math.h>

// Helper to write an IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// One-series × many-params (batch). Uses host-provided prefix sums P (sum_y)
// over the valid segment that starts at first_valid.
// Periods array is length n_combos. Output is row-major [combo][t] (n_combos x len).
extern "C" __global__ void dpo_batch_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum, // length = len + 1; prefix_sum[w] holds sum over [first_valid..(w-1)]
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y; // row index
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const int back = period / 2 + 1;
    const int warm = max(first_valid + period - 1, back);
    const int row_off = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nanf = f32_nan();

    const double inv_p = 1.0 / (double)period;
    while (t < len) {
        float out_val = nanf;
        if (t >= warm) {
            const int wr = t + 1; // prefix right endpoint (1-based)
            const int wl = wr - period; // left-1
            const double sum = prefix_sum[wr] - prefix_sum[wl];
            const double avg = sum * inv_p;
            const float price = data[t - back];
            out_val = price - (float)avg;
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// Many-series × one-param (time-major). Data layout: time-major with shape rows x cols.
// Prefix array P is time-major with +1 length per element (linear index (t*cols + s) + 1).
// first_valids has one entry per series/column.
extern "C" __global__ void dpo_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,
    const double* __restrict__ prefix_sum_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series/column
    const int tx = blockIdx.x * blockDim.x + threadIdx.x; // time index iterator
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) return;

    const int back = period / 2 + 1;
    const int warm = max(fv + period - 1, back);

    const int stride = gridDim.x * blockDim.x; // stride over time dimension
    const float nanf = f32_nan();
    const double inv_p = 1.0 / (double)period;

    for (int t = tx; t < rows; t += stride) {
        float out_val = nanf;
        if (t >= warm) {
            const int wr = (t * cols + s) + 1;
            const int wl = ((t - period) * cols + s) + 1;
            const double sum = prefix_sum_tm[wr] - prefix_sum_tm[wl];
            const double avg = sum * inv_p;
            const float price = data_tm[(t - back) * cols + s];
            out_val = price - (float)avg;
        }
        out_tm[t * cols + s] = out_val;
    }
}

