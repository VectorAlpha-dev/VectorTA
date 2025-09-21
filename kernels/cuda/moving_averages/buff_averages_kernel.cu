// CUDA kernel for Buff Averages batch computation using prefix sums.
//
// Each parameter combo (fast_period, slow_period) maps to blockIdx.y. Threads
// iterate over time indices (blockIdx.x/threadIdx.x) and use precomputed prefix
// sums of masked price*volume (pv) and volume-only (vv) arrays to evaluate the
// weighted averages in O(1) per output. Outputs prior to the warmup index are
// filled with NaNs to match the CPU indicator behaviour.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void buff_averages_batch_prefix_f32(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int fast_period = fast_periods[combo];
    const int slow_period = slow_periods[combo];
    if (fast_period <= 0 || slow_period <= 0) {
        return;
    }

    const int warm = first_valid + slow_period - 1;
    const int row_offset = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float fast_val = nan_f;
        float slow_val = nan_f;

        if (t >= warm) {
            int fast_start = t + 1 - fast_period;
            if (fast_start < 0) {
                fast_start = 0;
            }
            int slow_start = t + 1 - slow_period;
            if (slow_start < 0) {
                slow_start = 0;
            }

            const double slow_num = static_cast<double>(prefix_pv[t + 1]) -
                                    static_cast<double>(prefix_pv[slow_start]);
            const double slow_den = static_cast<double>(prefix_vv[t + 1]) -
                                    static_cast<double>(prefix_vv[slow_start]);
            slow_val = (slow_den != 0.0)
                ? static_cast<float>(slow_num / slow_den)
                : 0.0f;

            const double fast_num = static_cast<double>(prefix_pv[t + 1]) -
                                    static_cast<double>(prefix_pv[fast_start]);
            const double fast_den = static_cast<double>(prefix_vv[t + 1]) -
                                    static_cast<double>(prefix_vv[fast_start]);
            fast_val = (fast_den != 0.0)
                ? static_cast<float>(fast_num / fast_den)
                : 0.0f;
        }

        fast_out[row_offset + t] = fast_val;
        slow_out[row_offset + t] = slow_val;
        t += stride;
    }
}
