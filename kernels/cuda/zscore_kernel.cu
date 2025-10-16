// CUDA kernels for Zscore (SMA mean + standard deviation).
//
// Each parameter combination (period, nbdev) is assigned to blockIdx.y. Threads
// in the x-dimension iterate over time indices and compute z-scores using
// precomputed prefix sums of the input data and squared data, along with a
// prefix count of NaNs to preserve CPU parity. All accumulation happens in
// float64 to minimise drift; the final results are written as float32 values.

#include <cuda_runtime.h>
#include <math.h>

// ----------------- One-series × many-params (prefix-sum based) -----------------
extern "C" __global__ void zscore_sma_prefix_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int* __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ nbdevs,
    int n_combos,
    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const float nbdev = nbdevs[combo];
    if (period <= 0) {
        return;
    }

    const int warm = first_valid + period - 1;
    const int row_offset = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f;

        if (t >= warm) {
            if (nbdev != 0.0f) {
                int start = t + 1 - period;
                if (start < 0) {
                    start = 0;
                }

                const int nan_count = prefix_nan[t + 1] - prefix_nan[start];
                if (nan_count == 0) {
                    const double sum = prefix_sum[t + 1] - prefix_sum[start];
                    const double sum2 = prefix_sum_sq[t + 1] - prefix_sum_sq[start];
                    const double mean = sum / static_cast<double>(period);
                    double variance = (sum2 / static_cast<double>(period)) - (mean * mean);
                    if (variance > 0.0) {
                        const double std_base = sqrt(variance);
                        const double denom = std_base * static_cast<double>(nbdev);
                        if (denom != 0.0 && !isnan(denom)) {
                            const double val = static_cast<double>(data[t]);
                            const double z = (val - mean) / denom;
                            out_val = static_cast<float>(z);
                        }
                    }
                }
            }
        }

        out[row_offset + t] = out_val;
        t += stride;
    }
}

// ----------------- Many-series × one-param (time-major) -----------------
// Time-major layout [t][series]. Each block handles one series (column).
// Thread 0 performs the sequential sliding-window scan; other threads help
// initialize the column with NaNs. Mean is SMA; deviation is population stddev.
// Output is the z-score: (x - mean) / (stddev * nbdev). If nbdev == 0 or
// any NaN in the window, result is NaN. Warmup and NaN semantics match scalar.
extern "C" __global__ void zscore_many_series_one_param_f32(
    const float* __restrict__ data_tm,    // [rows * cols], time-major
    const int* __restrict__ first_valids, // [cols]
    int period,
    float nbdev,
    int cols,
    int rows,
    float* __restrict__ out_tm            // [rows * cols], time-major
) {
    const int series = blockIdx.x;
    if (series >= cols || period <= 0) return;
    const int stride = cols;

    // Fill column with NaN cooperatively
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= rows) return;

    const int warm = first_valid + period - 1;
    if (nbdev == 0.0f) {
        // All outputs remain NaN by contract when denominator is zero
        return;
    }

    const double inv_n = 1.0 / (double)period;

    // Bootstrap raw sums over initial window [first_valid .. warm]
    double s1 = 0.0, s2 = 0.0;
    int nan_in_win = 0;
    const int init_end = min(warm + 1, rows);
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) {
            nan_in_win++;
        } else {
            const double d = (double)v;
            s1 += d;
            s2 += d * d;
        }
    }

    if (warm < rows && nan_in_win == 0) {
        const double mean = s1 * inv_n;
        const double var = (s2 * inv_n) - (mean * mean);
        if (var > 1e-30) {
            const double sd_nb = sqrt(var) * (double)nbdev;
            const double x = (double)data_tm[warm * stride + series];
            out_tm[warm * stride + series] = (float)((x - mean) / sd_nb);
        }
        // else stays NaN
    }

    // Slide window forward
    for (int t = warm + 1; t < rows; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];

        if (isnan(old_v) || isnan(new_v)) {
            // Rebuild over the current window
            s1 = 0.0; s2 = 0.0; nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float vv = data_tm[k * stride + series];
                if (isnan(vv)) { nan_in_win++; }
                else { const double d = (double)vv; s1 += d; s2 += d * d; }
            }
        } else {
            // O(1) update
            const double od = (double)old_v;
            const double nd = (double)new_v;
            s1 += nd - od;
            s2 += (nd * nd) - (od * od);
        }

        if (nan_in_win == 0) {
            const double mean = s1 * inv_n;
            const double var  = (s2 * inv_n) - (mean * mean);
            if (var > 1e-30) {
                const double sd_nb = sqrt(var) * (double)nbdev;
                const double x = (double)new_v;
                out_tm[t * stride + series] = (float)((x - mean) / sd_nb);
            } else {
                // leave NaN
            }
        } else {
            // leave NaN
        }
    }
}

