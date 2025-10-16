// CUDA kernels for NET MyRSI (Ehlers' MyRSI + Noise Elimination Technique)
//
// Math pattern: Recurrence/streaming per time step, with small rings:
// - MyRSI: maintain rolling sums of up/down diffs using a ring of last `period` diffs
// - NET:    maintain a ring of last `period` MyRSI values and update the
//           pairwise "lt-minus-gt" count incrementally.
//
// Warmup/NaN parity with scalar:
// - warm = first_valid + period - 1
// - out[warm] = 0.0f (if period > 1) else NaN
// - [0, warm) are NaN
// - Division by zero in MyRSI denominator yields 0.0

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Quiets NaN constant
static __device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}

// Conservative bound for typical usage; host wrapper validates periods.
#ifndef NET_MYRSI_MAX_PERIOD
#define NET_MYRSI_MAX_PERIOD 2048
#endif

// Compute (#(v < s) - #(v > s)) over a contiguous slice [ptr, ptr+len)
static __device__ __forceinline__ int lt_minus_gt_slice(const double* ptr, int len, double s) {
    int lt = 0, gt = 0;
    for (int i = 0; i < len; ++i) {
        const double v = ptr[i];
        lt += (v < s);
        gt += (v > s);
    }
    return lt - gt;
}

extern "C" __global__
void net_myrsi_batch_f32(const float* __restrict__ prices,
                         const int*   __restrict__ periods,
                         int series_len,
                         int n_combos,
                         int first_valid,
                         float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + static_cast<size_t>(combo) * series_len;

    // Validate inputs and prefill with NaNs when invalid
    if (UNLIKELY(period <= 0 || period > NET_MYRSI_MAX_PERIOD ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < (period + 1))) { // need period+1 values
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) out_row[i] = qnan32();
    out_row[warm] = (period > 1) ? 0.0f : qnan32();

    // Two-pass exact parity with scalar helper functions using out_row as temporary for r
    const double denom = 0.5 * static_cast<double>(period) * static_cast<double>(period - 1);

    // Pass 1: compute MyRSI r[t] into out_row[t]
    for (int t = first_valid + period; t < series_len; ++t) {
        double cu = 0.0, cd = 0.0;
        #pragma unroll 1
        for (int j = 0; j < period; ++j) {
            const double newer = static_cast<double>(prices[t - j]);
            const double older = static_cast<double>(prices[t - j - 1]);
            const double diff  = newer - older;
            if (diff > 0.0)       cu += diff;
            else if (diff < 0.0)  cd += -diff;
        }
        const double sum = cu + cd;
        const double r = (sum != 0.0) ? ((cu - cd) / sum) : 0.0;
        out_row[t] = static_cast<float>(r);
    }

    // Pass 2: compute NET over r (in-place overwrite)
    for (int idx = warm + 1; idx < series_len; ++idx) {
        int num = 0;
        for (int i = 1; i < period; ++i) {
            const double vi = static_cast<double>(out_row[idx - i]);
            for (int k = 0; k < i; ++k) {
                const double vk = static_cast<double>(out_row[idx - k]);
                const double d = vi - vk;
                if (d > 0.0)      num -= 1;
                else if (d < 0.0) num += 1;
            }
        }
        out_row[idx] = (denom != 0.0) ? static_cast<float>(static_cast<double>(num) / denom) : 0.0f;
    }
}

// Many-series × one-param (time-major): prices_tm[t * cols + series]
extern "C" __global__
void net_myrsi_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    // Validate inputs for this series
    const int fv = first_valids[series];
    float* col_out = out_tm + series;
    if (UNLIKELY(period <= 0 || period > NET_MYRSI_MAX_PERIOD || fv < 0 || fv >= rows)) {
        for (int t = 0; t < rows; ++t) col_out[t * cols] = qnan32();
        return;
    }
    const int tail = rows - fv;
    if (UNLIKELY(tail < (period + 1))) {
        for (int t = 0; t < rows; ++t) col_out[t * cols] = qnan32();
        return;
    }

    const int warm = fv + period - 1;
    for (int t = 0; t < warm; ++t) col_out[t * cols] = qnan32();
    col_out[warm * cols] = (period > 1) ? 0.0f : qnan32();

    const double denom = 0.5 * static_cast<double>(period) * static_cast<double>(period - 1);
    auto load_tm = [&](int t) -> double { return static_cast<double>(prices_tm[static_cast<size_t>(t) * cols + series]); };

    // Pass 1: compute MyRSI r[t] into col_out[t*cols]
    for (int t = fv + period; t < rows; ++t) {
        double cu = 0.0, cd = 0.0;
        #pragma unroll 1
        for (int j = 0; j < period; ++j) {
            const double newer = load_tm(t - j);
            const double older = load_tm(t - j - 1);
            const double diff  = newer - older;
            if (diff > 0.0)       cu += diff;
            else if (diff < 0.0)  cd += -diff;
        }
        const double sum = cu + cd;
        const double r = (sum != 0.0) ? ((cu - cd) / sum) : 0.0;
        col_out[t * cols] = static_cast<float>(r);
    }

    // Pass 2: compute NET using stored r
    for (int idx = warm + 1; idx < rows; ++idx) {
        int num = 0;
        for (int i = 1; i < period; ++i) {
            const double vi = static_cast<double>(col_out[(idx - i) * cols]);
            for (int k = 0; k < i; ++k) {
                const double vk = static_cast<double>(col_out[(idx - k) * cols]);
                const double d = vi - vk;
                if (d > 0.0)      num -= 1;
                else if (d < 0.0) num += 1;
            }
        }
        col_out[idx * cols] = (denom != 0.0) ? static_cast<float>(static_cast<double>(num) / denom) : 0.0f;
    }
}
