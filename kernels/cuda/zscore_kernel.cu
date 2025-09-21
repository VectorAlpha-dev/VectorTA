// CUDA kernel for Zscore SMA + standard deviation batches.
//
// Each parameter combination (period, nbdev) is assigned to blockIdx.y. Threads
// in the x-dimension iterate over time indices and compute z-scores using
// precomputed prefix sums of the input data and squared data, along with a
// prefix count of NaNs to preserve CPU parity. All accumulation happens in
// float64 to minimise drift; the final results are written as float32 values.

#include <cuda_runtime.h>
#include <math.h>

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

