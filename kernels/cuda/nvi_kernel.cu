// CUDA kernels for Negative Volume Index (NVI)
//
// Semantics follow src/indicators/nvi.rs exactly:
// - Warmup: find first valid index on host; indices before it are NaN
// - At first valid index: NVI = 1000.0
// - Thereafter: if volume[t] < volume[t-1] then
//       pct = (close[t] - close[t-1]) / close[t-1]
//       nvi += nvi * pct;
//   else nvi stays unchanged
// - No special NaN handling beyond initial first_valid detection; NaNs in
//   close/volume after warmup propagate via IEEE rules (matching CPU path).
// - Computation uses float64 accumulators for parity; outputs are float32.

#include <cuda_runtime.h>
#include <math.h>

// One-series kernel (row-major contiguous arrays)
extern "C" __global__ void nvi_batch_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{
    if (len <= 0) return;

    // Use a single thread to preserve strict sequential update order.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;

    // Warmup prefix
    for (int i = 0; i < fv && i < len; ++i) {
        out[i] = nanf("");
    }
    if (fv >= len) return;

    // Initial value
    double nvi = 1000.0;
    out[fv] = (float)nvi;
    if (fv + 1 >= len) return;

    double prev_close = (double)close[fv];
    double prev_volume = (double)volume[fv];

    for (int i = fv + 1; i < len; ++i) {
        const double c = (double)close[i];
        const double v = (double)volume[i];
        if (v < prev_volume) {
            const double pct = (c - prev_close) / prev_close;
            nvi += nvi * pct; // keep op ordering identical to CPU
        }
        out[i] = (float)nvi;
        prev_close = c;
        prev_volume = v;
    }
}

// Many-series × one-param, time-major: close_tm[t * cols + s]
extern "C" __global__ void nvi_many_series_one_param_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids, // [cols]
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols || rows <= 0) return;

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const float nan_f = nanf("");

    // Prefill column with NaN for warmup and potential invalids
    for (int t = 0; t < rows; ++t) {
        out_tm[t * cols + s] = nan_f;
    }
    if (fv >= rows) return;

    // Initial value at first valid
    double nvi = 1000.0;
    out_tm[fv * cols + s] = (float)nvi;
    if (fv + 1 >= rows) return;

    double prev_close = (double)close_tm[fv * cols + s];
    double prev_volume = (double)volume_tm[fv * cols + s];

    for (int t = fv + 1; t < rows; ++t) {
        const double c = (double)close_tm[t * cols + s];
        const double v = (double)volume_tm[t * cols + s];
        if (v < prev_volume) {
            const double pct = (c - prev_close) / prev_close;
            nvi += nvi * pct;
        }
        out_tm[t * cols + s] = (float)nvi;
        prev_close = c;
        prev_volume = v;
    }
}

