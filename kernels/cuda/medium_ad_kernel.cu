// CUDA kernels for Median Absolute Deviation (MEDIUM_AD)
//
// Semantics match the scalar Rust implementation in src/indicators/medium_ad.rs:
// - Warmup: write NaN until index warm = first_valid + period - 1
// - If any NaN in the window, output is NaN
// - Period == 1: output 0.0 for finite input, else NaN
// - Even-length median = average of the two middle elements after sorting
//
// Numeric notes:
// - All window computations use float64 (double) for parity with CPU path
// - Outputs are float32 for bandwidth/VRAM efficiency (DeviceArrayF32)

#include <cuda_runtime.h>
#include <math.h>

// Upper bound on supported period in device local buffers. The host wrapper
// validates periods against this limit before launch.
#ifndef MEDIUM_AD_MAX_PERIOD
#define MEDIUM_AD_MAX_PERIOD 512
#endif

// Simple insertion sort for small windows (period <= 512)
__device__ __forceinline__ void insertion_sort_double(double* a, int n) {
    for (int i = 1; i < n; ++i) {
        double key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

__device__ __forceinline__ double median_from_sorted(const double* a, int n) {
    const int mid = n >> 1;
    if ((n & 1) == 1) {
        return a[mid];
    } else {
        // Even length: average of max(lower half) and a[mid]
        return 0.5 * (a[mid - 1] + a[mid]);
    }
}

extern "C" __global__ void medium_ad_batch_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) {
        // Host validates, but double-guard
        return;
    }
    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = nanf("");

    // Time index stride over X dimension
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Local stack buffer for window and |x - med|
    double buf[MEDIUM_AD_MAX_PERIOD];

    while (t < len) {
        float out_val = nan_f;

        if (t >= warm) {
            if (period == 1) {
                const float v = data[t];
                out_val = isfinite(v) ? 0.0f : nan_f;
            } else {
                const int start = t + 1 - period;
                bool has_nan = false;
                // Copy window to local buffer as f64 and check NaNs
                for (int k = 0; k < period; ++k) {
                    const float v = data[start + k];
                    if (!isfinite(v)) {
                        has_nan = true;
                    }
                    buf[k] = (double)v;
                }
                if (!has_nan) {
                    // Median of window
                    insertion_sort_double(buf, period);
                    const double med = median_from_sorted(buf, period);
                    // Transform to |x - med|
                    for (int k = 0; k < period; ++k) {
                        const double d = buf[k] - med;
                        buf[k] = fabs(d);
                    }
                    // Median of absolute deviations
                    insertion_sort_double(buf, period);
                    const double mad = median_from_sorted(buf, period);
                    out_val = (float)mad;
                }
            }
        }

        out[row_off + t] = out_val;
        t += stride;
    }
}

// Many-series × one-param, time-major layout
extern "C" __global__ void medium_ad_many_series_one_param_f32(
    const float* __restrict__ data_tm, // [rows * cols], time-major: data[t * cols + s]
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids, // [cols]
    float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    const float nan_f = nanf("");
    // Prefill column with NaNs for warmup/invalids
    for (int t = 0; t < rows; ++t) {
        out_tm[t * cols + s] = nan_f;
    }

    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) return;
    int first_valid = first_valids[s];
    if (first_valid < 0) first_valid = 0;
    const int warm = first_valid + period - 1;
    if (warm >= rows) return;

    double buf[MEDIUM_AD_MAX_PERIOD];

    for (int t = warm; t < rows; ++t) {
        const int start = t + 1 - period;
        bool has_nan = false;
        // Load window and check NaN
        for (int k = 0; k < period; ++k) {
            const float v = data_tm[(start + k) * cols + s];
            if (!isfinite(v)) has_nan = true;
            buf[k] = (double)v;
        }
        if (has_nan) {
            out_tm[t * cols + s] = nan_f;
            continue;
        }
        // Median
        insertion_sort_double(buf, period);
        const double med = median_from_sorted(buf, period);
        for (int k = 0; k < period; ++k) {
            const double d = buf[k] - med;
            buf[k] = fabs(d);
        }
        insertion_sort_double(buf, period);
        const double mad = median_from_sorted(buf, period);
        out_tm[t * cols + s] = (float)mad;
    }
}

