// CUDA kernels for Money Flow Index (MFI)
//
// Semantics mirror src/indicators/mfi.rs (scalar path):
// - Warmup: first non-NaN index + period - 1; values before are NaN
// - Division by zero -> 0.0
// - Uses host-built prefix sums of positive/negative money flow for batch
// - Many-series one-param scans per series sequentially (ring buffer), lane 0 per block

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float qnan() {
    return __int_as_float(0x7fffffff);
}

// Batch (one series × many params): sequential per-combo scan with O(1) updates.
// Avoids ring buffers by recomputing the evicted flow on-the-fly using tp[i] and tp[i-1].
extern "C" __global__
void mfi_batch_f32(const float* __restrict__ typical,
                   const float* __restrict__ volume,
                   int series_len,
                   int first_valid,
                   const int* __restrict__ periods,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;
    const int period = periods[combo];
    const int row_off = combo * series_len;
    const int warm = first_valid + period - 1;

    // Fill warmup with NaNs cooperatively
    for (int t = threadIdx.x; t < min(warm, series_len); t += blockDim.x) {
        out[row_off + t] = qnan();
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;
    if (warm >= series_len) return;

    // Seed sums across (first_valid+1 ..= warm)
    double pos_sum = 0.0, neg_sum = 0.0;
    float prev = typical[first_valid];
    for (int i = first_valid + 1; i <= warm; ++i) {
        const float tp = typical[i];
        const float vol = volume[i];
        const float diff = tp - prev;
        prev = tp;
        const float flow = tp * vol;
        if (diff > 0.0f) pos_sum += (double)flow;
        else if (diff < 0.0f) neg_sum += (double)flow;
    }
    // First value at warm index
    double tot = pos_sum + neg_sum;
    out[row_off + warm] = (tot <= 1e-14) ? 0.0f : (float)(100.0 * (pos_sum / tot));

    // Rolling updates for t > warm
    for (int t = warm + 1; t < series_len; ++t) {
        // add new flow at t
        const float tp_new = typical[t];
        const float vol_new = volume[t];
        const float diff_new = tp_new - typical[t - 1];
        const float flow_new = tp_new * vol_new;
        if (diff_new > 0.0f) pos_sum += (double)flow_new;
        else if (diff_new < 0.0f) neg_sum += (double)flow_new;

        // remove old flow starting once the window is full
        {
            const int i = t - period;
            const float tp_old = typical[i];
            const float diff_old = tp_old - typical[i - 1];
            const float flow_old = tp_old * volume[i];
            if (diff_old > 0.0f) pos_sum -= (double)flow_old;
            else if (diff_old < 0.0f) neg_sum -= (double)flow_old;
        }

        tot = pos_sum + neg_sum;
        out[row_off + t] = (tot <= 1e-14) ? 0.0f : (float)(100.0 * (pos_sum / tot));
    }
}

// Many-series × one-param (time-major). Each block handles one series; lane 0
// scans sequentially using a f64 ring buffer of size `period` for accuracy.
extern "C" __global__
void mfi_many_series_one_param_f32(const float* __restrict__ typical_tm,
                                   const float* __restrict__ volume_tm,
                                   const int* __restrict__ first_valids,
                                   int period,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || series_len <= 0 || period <= 0) return;
    const int first = first_valids[s];
    const int stride = num_series;

    // Initialize warmup region to NaN cooperatively
    if (first < 0 || first >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * stride + s] = qnan();
        }
        return;
    }
    const int warm = first + period - 1;
    for (int t = threadIdx.x; t < min(warm, series_len); t += blockDim.x) {
        out_tm[t * stride + s] = qnan();
    }

    if (threadIdx.x != 0) return; // lane 0 performs the scan

    // f64 ring buffers in dynamic shared memory: 2 * period entries
    extern __shared__ double shared[];
    double* pos_buf = shared;
    double* neg_buf = shared + period;
    // Zero-init ring
    for (int i = 0; i < period; ++i) { pos_buf[i] = 0.0; neg_buf[i] = 0.0; }

    // Seed using flows from (first+1 ..= warm)
    double pos_sum = 0.0, neg_sum = 0.0;
    float prev = typical_tm[first * stride + s];
    int ring = 0;
    for (int t = first + 1; t <= warm && t < series_len; ++t) {
        const float tp = typical_tm[t * stride + s];
        const float vol = volume_tm[t * stride + s];
        const float diff = tp - prev;
        prev = tp;
        const float flow = tp * vol;
        const double pos = (diff > 0.0f) ? (double)flow : 0.0;
        const double neg = (diff < 0.0f) ? (double)flow : 0.0;
        pos_buf[ring] = pos; neg_buf[ring] = neg;
        pos_sum += pos; neg_sum += neg;
        ring = (ring + 1) % period;
    }

    if (warm < series_len) {
        // First value at warm index
        const double tot0 = pos_sum + neg_sum;
        out_tm[warm * stride + s] = (tot0 <= 1e-14) ? 0.0f : (float)(100.0 * (pos_sum / tot0));
    }

    // Continue rolling
    for (int t = warm + 1; t < series_len; ++t) {
        const float tp = typical_tm[t * stride + s];
        const float vol = volume_tm[t * stride + s];
        const float diff = tp - prev;
        prev = tp;
        const float flow = tp * vol;

        // Evict
        const double old_pos = pos_buf[ring];
        const double old_neg = neg_buf[ring];
        pos_sum -= old_pos; neg_sum -= old_neg;

        // Insert
        const double pos = (diff > 0.0f) ? (double)flow : 0.0;
        const double neg = (diff < 0.0f) ? (double)flow : 0.0;
        pos_buf[ring] = pos; neg_buf[ring] = neg;
        pos_sum += pos; neg_sum += neg;
        ring = (ring + 1) % period;

        const double tot = pos_sum + neg_sum;
        out_tm[t * stride + s] = (tot <= 1e-14) ? 0.0f : (float)(100.0 * (pos_sum / tot));
    }
}
