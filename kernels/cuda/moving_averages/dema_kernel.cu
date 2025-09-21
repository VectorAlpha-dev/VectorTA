// CUDA kernel for DEMA (Double Exponential Moving Average).
//
// Each CUDA block processes a single parameter combination (period) and walks
// the input series sequentially. The implementation keeps the recurrence
// identical to the scalar Rust path and only writes outputs once the warm-up
// window (first_valid + period - 1) has been reached, leaving earlier samples
// as NaN to match CPU semantics.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void dema_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int base = combo * series_len;

    // Initialise the entire output row with NaNs in parallel.
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    // Only lane 0 performs the sequential EMA recurrence for this combo.
    if (threadIdx.x != 0) {
        return;
    }

    if (first_valid >= series_len) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0) {
        return;
    }

    const float alpha = 2.0f / (static_cast<float>(period) + 1.0f);
    const float alpha1 = 1.0f - alpha;

    const int warm = first_valid + period - 1;
    const int start = first_valid;

    float ema = prices[start];
    float ema2 = ema;

    if (start >= warm && start < series_len) {
        out[base + start] = 2.0f * ema - ema2;
    }

    for (int t = start + 1; t < series_len; ++t) {
        const float price = prices[t];
        ema = alpha1 * ema + alpha * price;
        ema2 = alpha1 * ema2 + alpha * ema;

        if (t >= warm) {
            out[base + t] = 2.0f * ema - ema2;
        }
    }
}
