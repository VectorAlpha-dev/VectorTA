// CUDA kernels for Detrended Price Oscillator (DPO)
//
// Math pattern: prefix-sum/rational (simple average via prefix sums) with a lagged
// price subtraction. Host builds prefix sums over the valid segment and passes them
// to the kernel. Kernel computes O(1) per-output window ops.
//
// Semantics:
// - Warmup per row: warm = max(first_valid + period - 1, back) where back = period/2 + 1
// - Warmup prefix filled with NaN
//
// Implementation note (Ada/RTX 40xx friendly):
// - Eliminate FP64 math from device hot path. We keep the host-provided prefix array
//   as f64 for ABI compatibility but convert each endpoint to f32 on load and do all
//   window math in f32. This removes slow FP64 ALU ops while preserving the 8-byte
//   prefix footprint and existing host code. The output is evaluated with fused
//   multiply-add (fmaf) for better FP32 accuracy.

#include <cuda_runtime.h>
#include <math.h>

// Helper to write an IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// One-series × many-params (batch). Uses host-provided prefix sums P (sum_y)
// over the valid segment that starts at first_valid.
// Periods array is length n_combos. Output is row-major [combo][t] (n_combos x len).
extern "C" __global__ void dpo_batch_f32(
    const float*  __restrict__ data,
    const float2* __restrict__ prefix_sum_ds, // len+1; {hi,lo} Kahan prefix (host-built)
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

    const float inv_p = 1.0f / (float)period;
    // Pointer to price lag base so that price = price_base[t]
    const float* __restrict__ price_base = data - back;
    while (t < len) {
        float out_val = nanf;
        if (t >= warm) {
            const int wr = t + 1;        // prefix right endpoint (1-based)
            const int wl = wr - period;  // left-1

            // Double-single prefix subtraction: (hi,lo) pair
            const float2 r = prefix_sum_ds[wr];
            const float2 l = prefix_sum_ds[wl];
            const float sum_hi = r.x - l.x;
            const float sum_lo = r.y - l.y;

            const float price = price_base[t];
            // out = price - inv_p * (sum_hi + sum_lo), evaluated by two fused FMAs
            float tmp = fmaf(-inv_p, sum_hi, price);
            out_val    = fmaf(-inv_p, sum_lo, tmp);
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// Many-series × one-param (time-major). Data layout: time-major with shape rows x cols.
// Prefix array P is time-major with +1 length per element (linear index (t*cols + s) + 1).
// first_valids has one entry per series/column.
extern "C" __global__ void dpo_many_series_one_param_time_major_f32(
    const float*  __restrict__ data_tm,
    const float2* __restrict__ prefix_sum_tm_ds,
    const int*    __restrict__ first_valids,
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
    const float inv_p = 1.0f / (float)period;

    for (int t = tx; t < rows; t += stride) {
        float out_val = nanf;
        if (t >= warm) {
            const int wr = (t * cols + s) + 1;
            const int wl = (t >= period) ? ((t - period) * cols + s) + 1 : 0;

            const float2 r = prefix_sum_tm_ds[wr];
            const float2 l = prefix_sum_tm_ds[wl];
            const float sum_hi = r.x - l.x;
            const float sum_lo = r.y - l.y;

            const float price = data_tm[(t - back) * cols + s];
            float tmp = fmaf(-inv_p, sum_hi, price);
            out_val    = fmaf(-inv_p, sum_lo, tmp);
        }
        out_tm[t * cols + s] = out_val;
    }
}
