// CUDA kernels for On-Balance Volume (OBV).
//
// Math: OBV[i] = OBV[i-1] + sign(close[i] - close[i-1]) * volume[i]
// - sign in {-1, 0, +1}
// - Warmup: write NaN for indices < first_valid; write 0.0 at first_valid
// - Inputs are FP32; accumulations use FP64 for numerical parity; outputs FP32

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// Batch kernel: one price series × many params (OBV has no params).
// We keep the same shape as other batch kernels: grid.y indexes the
// parameter row (here, n_combos is typically 1).
extern "C" __global__ void obv_batch_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out)
{
    const int combo = blockIdx.y; // row index
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Clamp first_valid
    const int fv = first_valid < 0 ? 0 : first_valid;

    // Each thread writes warmup prefix NaNs for its slice
    for (int i = tid; i < fv && i < series_len; i += stride) {
        out[combo * series_len + i] = CUDART_NAN_F;
    }

    // Single-thread serial scan for the recurrence (carry dependency)
    if (tid == 0) {
        const int base = combo * series_len;
        if (fv < series_len) {
            out[base + fv] = 0.0f; // OBV starts at 0 at first valid
            double prev_obv = 0.0;
            double prev_close = static_cast<double>(close[fv]);

            for (int i = fv + 1; i < series_len; ++i) {
                const double c = static_cast<double>(close[i]);
                const double v = static_cast<double>(volume[i]);
                // Branchless sign via comparisons
                const int gt = (c > prev_close);
                const int lt = (c < prev_close);
                const double s = static_cast<double>(gt - lt);
                prev_obv = fma(v, s, prev_obv);
                out[base + i] = static_cast<float>(prev_obv);
                prev_close = c;
            }
        }
    }
}

// Many-series × one-param (time-major):
// - close_tm, volume_tm: [rows][cols] laid out as index = t*cols + s
// - first_valids: per-series first valid index (time index)
extern "C" __global__ void obv_many_series_one_param_time_major_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int*   __restrict__ first_valids,
    int cols, // number of series
    int rows, // series length
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series id
    if (s >= cols || rows <= 0) {
        return;
    }

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];

    // Warmup prefix
    for (int t = 0; t < rows && t < fv; ++t) {
        out_tm[t * cols + s] = CUDART_NAN_F;
    }

    if (fv < rows) {
        const int idx0 = fv * cols + s;
        out_tm[idx0] = 0.0f; // OBV starts at 0 at first valid
        double prev_obv = 0.0;
        double prev_close = static_cast<double>(close_tm[idx0]);
        for (int t = fv + 1; t < rows; ++t) {
            const int idx = t * cols + s;
            const double c = static_cast<double>(close_tm[idx]);
            const double v = static_cast<double>(volume_tm[idx]);
            const int gt = (c > prev_close);
            const int lt = (c < prev_close);
            const double sgn = static_cast<double>(gt - lt);
            prev_obv = fma(v, sgn, prev_obv);
            out_tm[idx] = static_cast<float>(prev_obv);
            prev_close = c;
        }
    }
}

