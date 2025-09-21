// CUDA kernel for Williams Accumulation/Distribution (WAD).
// Each thread processes one price series (row-major layout). The kernel matches
// the scalar CPU logic exactly, including the warm-up prefix and running sum.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void wad_series_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    int n_series,
    float* __restrict__ out) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= n_series || len <= 0) {
        return;
    }

    const int offset = series * len;
    const float* high_row = high + offset;
    const float* low_row = low + offset;
    const float* close_row = close + offset;
    float* out_row = out + offset;

    out_row[0] = 0.0f;
    double running = 0.0;
    double prev_close = static_cast<double>(close_row[0]);

    for (int i = 1; i < len; ++i) {
        const double hi = static_cast<double>(high_row[i]);
        const double lo = static_cast<double>(low_row[i]);
        const double c = static_cast<double>(close_row[i]);

        const double trh = prev_close > hi ? prev_close : hi;
        const double trl = prev_close < lo ? prev_close : lo;

        double ad = 0.0;
        if (c > prev_close) {
            ad = c - trl;
        } else if (c < prev_close) {
            ad = c - trh;
        }

        running += ad;
        out_row[i] = static_cast<float>(running);
        prev_close = c;
    }
}
