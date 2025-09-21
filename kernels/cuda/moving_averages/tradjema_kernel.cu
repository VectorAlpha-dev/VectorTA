// CUDA kernels for the Trend Adjusted EMA (TrAdjEMA).
//
// Kernels operate on FP32 buffers (to match the public API) but promote all
// arithmetic to FP64 to stay numerically aligned with the CPU reference. Final
// results are cast back to FP32 when written to device memory.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ inline double compute_true_range(
    double high,
    double low,
    double prev_close,
    bool first_bar
) {
    if (first_bar) {
        return high - low;
    }
    double hl = high - low;
    double hc = fabs(high - prev_close);
    double lc = fabs(low - prev_close);
    return fmax(hl, fmax(hc, lc));
}

extern "C" __global__
void tradjema_batch_f32(const float* __restrict__ high,
                        const float* __restrict__ low,
                        const float* __restrict__ close,
                        const int* __restrict__ lengths,
                        const float* __restrict__ mults,
                        int series_len,
                        int n_combos,
                        int first_valid,
                        float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int length = lengths[combo];
    const float mult_f32 = mults[combo];
    const double mult = static_cast<double>(mult_f32);

    const int base = combo * series_len;

    if (length <= 1 || length > series_len || !isfinite(mult_f32) || mult_f32 <= 0.0f) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = NAN;
        }
        return;
    }

    const int warm = first_valid + length - 1;
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out[base + t] = NAN;
    }
    __syncthreads();

    if (warm >= series_len || threadIdx.x != 0) {
        return;
    }

    extern __shared__ double tr_buf[];
    const double alpha = 2.0 / (static_cast<double>(length) + 1.0);

    for (int k = 0; k < length; ++k) {
        const int idx = first_valid + k;
        const double prev_close = (idx == 0) ? 0.0 : static_cast<double>(close[idx - 1]);
        const double high_d = static_cast<double>(high[idx]);
        const double low_d = static_cast<double>(low[idx]);
        tr_buf[k] = compute_true_range(high_d, low_d, prev_close, idx == first_valid);
    }

    int head = length - 1;
    double tr_low = tr_buf[0];
    double tr_high = tr_buf[0];
    for (int k = 1; k < length; ++k) {
        const double v = tr_buf[k];
        tr_low = fmin(tr_low, v);
        tr_high = fmax(tr_high, v);
    }

    double current_tr = tr_buf[head];
    double tr_adj = (tr_high != tr_low) ? ((current_tr - tr_low) / (tr_high - tr_low)) : 0.0;
    const double src0 = static_cast<double>(close[warm - 1]);
    double y = alpha * (1.0 + tr_adj * mult) * (src0 - 0.0);
    out[base + warm] = static_cast<float>(y);

    for (int i = warm + 1; i < series_len; ++i) {
        const double prev_close = static_cast<double>(close[i - 1]);
        const double tr_new = compute_true_range(
            static_cast<double>(high[i]),
            static_cast<double>(low[i]),
            prev_close,
            false
        );

        head = (head + 1) % length;
        const double tr_old = tr_buf[head];
        tr_buf[head] = tr_new;

        if (tr_old <= tr_low || tr_old >= tr_high) {
            tr_low = tr_buf[0];
            tr_high = tr_buf[0];
            for (int k = 1; k < length; ++k) {
                const double v = tr_buf[k];
                tr_low = fmin(tr_low, v);
                tr_high = fmax(tr_high, v);
            }
        } else {
            tr_low = fmin(tr_low, tr_new);
            tr_high = fmax(tr_high, tr_new);
        }

        tr_adj = (tr_high != tr_low) ? ((tr_new - tr_low) / (tr_high - tr_low)) : 0.0;
        const double a = alpha * (1.0 + tr_adj * mult);
        const double src = static_cast<double>(close[i - 1]);
        y += a * (src - y);
        out[base + i] = static_cast<float>(y);
    }
}

extern "C" __global__
void tradjema_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int num_series,
    int series_len,
    int length,
    float mult_f32,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) {
        return;
    }

    if (length <= 1 || length > series_len || !isfinite(mult_f32) || mult_f32 <= 0.0f) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + series] = NAN;
        }
        return;
    }

    const int first_valid = first_valids[series];
    const int warm = first_valid + length - 1;

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * num_series + series] = NAN;
    }
    __syncthreads();

    if (warm >= series_len || threadIdx.x != 0) {
        return;
    }

    extern __shared__ double tr_buf[];
    const double mult = static_cast<double>(mult_f32);
    const double alpha = 2.0 / (static_cast<double>(length) + 1.0);

    auto at = [num_series](const float* buf, int row, int col) {
        return buf[row * num_series + col];
    };

    for (int k = 0; k < length; ++k) {
        const int idx = first_valid + k;
        const double prev_close = (idx == 0) ? 0.0 : static_cast<double>(at(close_tm, idx - 1, series));
        const double high_d = static_cast<double>(at(high_tm, idx, series));
        const double low_d = static_cast<double>(at(low_tm, idx, series));
        tr_buf[k] = compute_true_range(high_d, low_d, prev_close, idx == first_valid);
    }

    int head = length - 1;
    double tr_low = tr_buf[0];
    double tr_high = tr_buf[0];
    for (int k = 1; k < length; ++k) {
        const double v = tr_buf[k];
        tr_low = fmin(tr_low, v);
        tr_high = fmax(tr_high, v);
    }

    double current_tr = tr_buf[head];
    double tr_adj = (tr_high != tr_low) ? ((current_tr - tr_low) / (tr_high - tr_low)) : 0.0;
    const double src0 = static_cast<double>(at(close_tm, warm - 1, series));
    double y = alpha * (1.0 + tr_adj * mult) * (src0 - 0.0);
    out_tm[warm * num_series + series] = static_cast<float>(y);

    for (int i = warm + 1; i < series_len; ++i) {
        const double prev_close = static_cast<double>(at(close_tm, i - 1, series));
        const double tr_new = compute_true_range(
            static_cast<double>(at(high_tm, i, series)),
            static_cast<double>(at(low_tm, i, series)),
            prev_close,
            false
        );

        head = (head + 1) % length;
        const double tr_old = tr_buf[head];
        tr_buf[head] = tr_new;

        if (tr_old <= tr_low || tr_old >= tr_high) {
            tr_low = tr_buf[0];
            tr_high = tr_buf[0];
            for (int k = 1; k < length; ++k) {
                const double v = tr_buf[k];
                tr_low = fmin(tr_low, v);
                tr_high = fmax(tr_high, v);
            }
        } else {
            tr_low = fmin(tr_low, tr_new);
            tr_high = fmax(tr_high, tr_new);
        }

        tr_adj = (tr_high != tr_low) ? ((tr_new - tr_low) / (tr_high - tr_low)) : 0.0;
        const double a = alpha * (1.0 + tr_adj * mult);
        const double src = static_cast<double>(at(close_tm, i - 1, series));
        y += a * (src - y);
        out_tm[i * num_series + series] = static_cast<float>(y);
    }
}
