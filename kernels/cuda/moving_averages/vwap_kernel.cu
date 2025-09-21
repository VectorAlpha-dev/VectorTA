#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <limits.h>

// CUDA kernel for VWAP batch processing. Each thread handles one parameter
// combination and walks the time series sequentially (dependencies across time
// prevent straightforward intra-row parallelism). The kernel expects column-major
// output layout: rows correspond to parameter combinations and columns to time
// indices.
extern "C" __global__
void vwap_batch_f32(const long long* __restrict__ timestamps,
                    const float* __restrict__ volumes,
                    const float* __restrict__ prices,
                    const int* __restrict__ counts,
                    const int* __restrict__ unit_codes,
                    const long long* __restrict__ divisors,
                    const int* __restrict__ first_valids,
                    const int* __restrict__ month_ids,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int count = counts[combo];
    const int unit = unit_codes[combo];
    long long divisor = divisors[combo];
    const int warm_raw = first_valids[combo];

    const int base = combo * series_len;
    const float nan = __int_as_float(0x7fffffff);

    if (count <= 0 || series_len <= 0) {
        for (int t = 0; t < series_len; ++t) {
            out[base + t] = nan;
        }
        return;
    }

    if (unit != 3 && divisor <= 0) {
        divisor = 1;  // defensive: avoid division by zero
    }

    int warm = warm_raw;
    if (warm < 0) warm = 0;
    if (warm > series_len) warm = series_len;

    for (int t = 0; t < warm; ++t) {
        out[base + t] = nan;
    }

    long long current_gid = LLONG_MIN;
    float volume_sum = 0.0f;
    float vol_price_sum = 0.0f;

    const int month_div = (unit == 3 && divisor > 0) ? static_cast<int>(divisor) : 1;

    for (int t = warm; t < series_len; ++t) {
        long long gid;
        if (unit == 3) {
            const int month_val = month_ids ? month_ids[t] : 0;
            gid = static_cast<long long>(month_val / (month_div > 0 ? month_div : 1));
        } else {
            const long long ts = timestamps[t];
            gid = ts / divisor;
        }

        if (gid != current_gid) {
            current_gid = gid;
            volume_sum = 0.0f;
            vol_price_sum = 0.0f;
        }

        const float vol = volumes[t];
        const float price = prices[t];
        volume_sum += vol;
        vol_price_sum += vol * price;

        if (volume_sum > 0.0f) {
            out[base + t] = vol_price_sum / volume_sum;
        } else {
            out[base + t] = nan;
        }
    }
}
