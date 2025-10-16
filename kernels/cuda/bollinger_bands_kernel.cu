// CUDA kernels for Bollinger Bands (SMA + standard deviation path).
//
// Batch kernel (one series × many params):
//   - grid.y indexes parameter combinations (period, devup, devdn)
//   - grid.x × blockDim.x covers time indices
//   - Uses host-precomputed prefix sums (sum, sum of squares) and prefix NaN counts
//   - Accumulations use float64 for numerical stability; outputs are float32
//
// Many-series kernel (time-major, one param):
//   - Inputs are (rows+1)×cols prefix arrays (sum, sumsq, nans)
//   - Each block.y is a series; block.x × blockDim.x covers time
//   - Writes three outputs: upper, middle, lower

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}

extern "C" __global__ void bollinger_bands_sma_prefix_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int* __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ devups,
    const float* __restrict__ devdns,
    int n_combos,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const float devup = devups[combo];
    const float devdn = devdns[combo];

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nanf = qnan32();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float u = nanf, m = nanf, l = nanf;
        if (t >= warm) {
            // Stickiness to mirror scalar rolling sums: once any NaN has been seen
            // since first_valid, results remain NaN even if the current window has none.
            const int nan_since_first = prefix_nan[t + 1] - prefix_nan[first_valid];
            if (nan_since_first == 0) {
                const int start = (t + 1) - period;
                const int s = start < 0 ? 0 : start;
                const double sum = prefix_sum[t + 1] - prefix_sum[s];
                const double sum2 = prefix_sum_sq[t + 1] - prefix_sum_sq[s];
                const double mean = sum / (double)period;
                double var = (sum2 / (double)period) - mean * mean;
                if (var < 0.0) var = 0.0; // guard tiny negative from FP error
                const double sd = sqrt(var);
                const double md = mean;
                m = (float)md;
                u = (float)(md + (double)devup * sd);
                l = (float)(md - (double)devdn * sd);
            }
        }
        out_upper[row_off + t]  = u;
        out_middle[row_off + t] = m;
        out_lower[row_off + t]  = l;
        t += stride;
    }
}

// Many-series (time-major) SMA + stddev with one parameter set (period, devup, devdn)
extern "C" __global__ void bollinger_bands_many_series_one_param_f32(
    const double* __restrict__ prefix_sum_tm,   // (rows+1) x cols
    const double* __restrict__ prefix_sum_sq_tm,// (rows+1) x cols
    const int* __restrict__ prefix_nan_tm,      // (rows+1) x cols
    int period,
    float devup,
    float devdn,
    int num_series,  // cols
    int series_len,  // rows
    const int* __restrict__ first_valids,       // cols
    float* __restrict__ out_upper_tm,           // rows x cols
    float* __restrict__ out_middle_tm,          // rows x cols
    float* __restrict__ out_lower_tm) {         // rows x cols
    const int s = blockIdx.y;
    if (s >= num_series) return;
    if (period <= 0) return;

    const int warm = first_valids[s] + period - 1;
    const int stride = num_series; // time-major indexing stride

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + s;
        float u = qnan32(), m = qnan32(), l = qnan32();
        if (t >= warm) {
            // Sticky NaN since first_valid for this series
            const int p_idx_t1 = (t + 1) * stride + s;
            const int p_idx_first = first_valids[s];
            const int nan_since_first = prefix_nan_tm[p_idx_t1] - prefix_nan_tm[p_idx_first * stride + s];
            if (nan_since_first == 0) {
                const int t1 = t + 1;
                int start = t1 - period; if (start < 0) start = 0;
                const int p_idx = t1 * stride + s;
                const int s_idx = start * stride + s;
                const double sum  = prefix_sum_tm[p_idx]    - prefix_sum_tm[s_idx];
                const double sum2 = prefix_sum_sq_tm[p_idx] - prefix_sum_sq_tm[s_idx];
                const double mean = sum / (double)period;
                double var = (sum2 / (double)period) - mean * mean;
                if (var < 0.0) var = 0.0;
                const double sd = sqrt(var);
                m = (float)mean;
                u = (float)(mean + (double)devup * sd);
                l = (float)(mean - (double)devdn * sd);
            }
        }
        out_upper_tm[out_idx]  = u;
        out_middle_tm[out_idx] = m;
        out_lower_tm[out_idx]  = l;
        t += step;
    }
}
