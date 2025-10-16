// CUDA kernels for Chaikin Accumulation/Distribution (AD)
//
// Two entry points:
//  - ad_series_f32: row-major layout, N independent series each of length `len`.
//  - ad_many_series_one_param_time_major_f32: time-major layout, many series sharing
//    no parameters (AD has none). Each block handles one series; thread 0 scans.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void ad_series_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int n_series,
    float* __restrict__ out)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= n_series || len <= 0) {
        return;
    }

    const int offset = series * len;
    const float* high_row = high + offset;
    const float* low_row = low + offset;
    const float* close_row = close + offset;
    const float* vol_row = volume + offset;
    float* out_row = out + offset;

    double sum = 0.0;
    // Match scalar CPU path: start at i=0, update sum if denom != 0.
    for (int i = 0; i < len; ++i) {
        const double h = static_cast<double>(high_row[i]);
        const double l = static_cast<double>(low_row[i]);
        const double c = static_cast<double>(close_row[i]);
        const double v = static_cast<double>(vol_row[i]);
        const double hl = h - l;
        if (hl != 0.0) {
            const double num = (c - l) - (h - c);
            sum += (num / hl) * v;
        }
        out_row[i] = static_cast<float>(sum);
    }
}

// Many-series kernel (time-major). Each block computes one series sequentially.
extern "C" __global__ void ad_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,   // [time][series]
    const float* __restrict__ low_tm,    // [time][series]
    const float* __restrict__ close_tm,  // [time][series]
    const float* __restrict__ volume_tm, // [time][series]
    int num_series,
    int series_len,
    float* __restrict__ out_tm)          // [time][series]
{
    const int series = blockIdx.x;
    if (series >= num_series || series_len <= 0) {
        return;
    }

    // Single-thread scan to preserve exact recurrence and order
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int t = 0; t < series_len; ++t) {
            const int idx = t * num_series + series;
            const double h = static_cast<double>(high_tm[idx]);
            const double l = static_cast<double>(low_tm[idx]);
            const double c = static_cast<double>(close_tm[idx]);
            const double v = static_cast<double>(volume_tm[idx]);
            const double hl = h - l;
            if (hl != 0.0) {
                const double num = (c - l) - (h - c);
                sum += (num / hl) * v;
            }
            out_tm[idx] = static_cast<float>(sum);
        }
    }
}

