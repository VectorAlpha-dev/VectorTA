// CUDA kernels for IFT RSI (Inverse Fisher Transform of RSI) in FP32.
//
// Behavior mirrors src/indicators/ift_rsi.rs (scalar path):
// - Warmup prefix: first_valid + rsi_period + wma_period - 1 written as NaN
// - RSI uses Wilder SMMA recurrence on gains/losses (positive/negative diffs)
// - Smoothed RSI: x = 0.1 * (RSI - 50)
// - LWMA with weights 1..wp maintained via O(1) rolling recurrence
// - Output is tanh(LWMA(x)) per timestep

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

static __device__ __forceinline__ float f32_qnan() {
    return __int_as_float(0x7fffffff);
}

// One-series × many-params (batch). Each block handles one combo sequentially.
// Inputs are original price series; gains/losses computed on the fly per row
// to minimize host-side buffers (acceptable because seed is O(rsi_period)).
extern "C" __global__ void ift_rsi_batch_f32(
    const float* __restrict__ data,
    int series_len,
    int n_combos,
    int first_valid,
    const int* __restrict__ rsi_periods,
    const int* __restrict__ wma_periods,
    float* __restrict__ out_values) {

    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int rp = rsi_periods[combo];
    const int wp = wma_periods[combo];
    if (UNLIKELY(rp <= 0 || wp <= 0 || rp > series_len || wp > series_len)) {
        // Write NaNs for robustness
        const int base = combo * series_len;
        for (int t = 0; t < series_len; ++t) out_values[base + t] = f32_qnan();
        return;
    }
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        const int base = combo * series_len;
        for (int t = 0; t < series_len; ++t) out_values[base + t] = f32_qnan();
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < max(rp, wp))) {
        const int base = combo * series_len;
        for (int t = 0; t < series_len; ++t) out_values[base + t] = f32_qnan();
        return;
    }

    const int warm = first_valid + rp + wp - 1;
    const int base = combo * series_len;
    // Fill NaNs up to warmup
    for (int t = 0; t < min(warm, series_len); ++t) out_values[base + t] = f32_qnan();

    // Single thread per block performs the scan (branchless for other threads)
    if (threadIdx.x != 0) return;

    // Wilder RSI seeding over first rp diffs after first_valid
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    const int start_seed = first_valid + 1;
    const int seed_end = start_seed + rp - 1;
    for (int i = start_seed; i <= seed_end; ++i) {
        const double d = (double)data[i] - (double)data[i - 1];
        if (d > 0.0) avg_gain += d; else avg_loss -= d;
    }
    const double rp_f = (double)rp;
    avg_gain /= rp_f;
    avg_loss /= rp_f;
    const double alpha = 1.0 / rp_f;
    const double beta  = 1.0 - alpha;

    // LWMA state over x = 0.1*(RSI-50)
    const double wp_f = (double)wp;
    const double denom_rcp = 1.0 / (0.5 * wp_f * (wp_f + 1.0));
    // Small ring buffer in registers/local memory
    extern __shared__ float shmem[];
    int head = 0;
    int filled = 0;
    double S1 = 0.0; // sum(x)
    double S2 = 0.0; // sum(k*x)

    // For wp up to CAP, we keep the last wp samples to update S1 efficiently
    // Otherwise, we still update S1 with a running sum by subtracting x_old read from data path (fallback not supported without ring),
    // so we constrain to CAP to keep memory bounded; typical wp << CAP.
    float* ring = shmem; // per-block ring of size >= wp floats

    // Iterate timesteps in sliced index space i = rp..(tail-1), corresponding to absolute t = first_valid + i
    for (int i = rp; i < tail; ++i) {
        // Update Wilder averages (use gain/loss from diff at absolute index (first_valid + i))
        if (i > rp) {
            const int abs_idx = first_valid + i;
            const double d = (double)data[abs_idx] - (double)data[abs_idx - 1];
            const double g = (d > 0.0) ? d : 0.0;
            const double l = (d > 0.0) ? 0.0 : -d;
            avg_gain = fma(avg_gain, beta, alpha * g);
            avg_loss = fma(avg_loss, beta, alpha * l);
        }

        const double rs = (avg_loss != 0.0) ? (avg_gain / avg_loss) : 100.0;
        const double rsi = 100.0 - 100.0 / (1.0 + rs);
        const float  x_f = (float)(0.1 * (rsi - 50.0));

        if (filled < wp) {
            // Build phase
            S1 += (double)x_f;
            S2 = fma((double)(filled + 1), (double)x_f, S2);
            if (ring) ring[head] = x_f;
            head = (head + 1 == wp) ? 0 : head + 1;
            filled += 1;
            if (filled == wp) {
                const double wma = S2 * denom_rcp;
                const int abs_t = first_valid + i;
                out_values[base + abs_t] = tanhf((float)wma);
            }
        } else {
            const float x_old = ring ? ring[head] : 0.0f; // ring always valid for wp<=CAP
            if (ring) ring[head] = x_f;
            head = (head + 1 == wp) ? 0 : head + 1;
            const double S1_prev = S1;
            S1 = S1_prev + (double)x_f - (double)x_old;
            S2 = (S2 - S1_prev) + wp_f * (double)x_f;
            const double wma = S2 * denom_rcp;
            const int abs_t = first_valid + i;
            out_values[base + abs_t] = tanhf((float)wma);
        }
    }
}

// Many-series × one-param (time-major). Each thread handles one series (column).
extern "C" __global__ void ift_rsi_many_series_one_param_f32(
    const float* __restrict__ data_tm,     // time-major [row * num_series + series]
    const int*   __restrict__ first_valids,// per-series first valid index
    int num_series,
    int series_len,
    int rsi_period,
    int wma_period,
    float* __restrict__ out_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;
    if (UNLIKELY(rsi_period <= 0 || wma_period <= 0 || rsi_period > series_len || wma_period > series_len)) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }
    int first = first_valids ? first_valids[series] : 0;
    if (first < 0) first = 0;
    if (UNLIKELY(first >= series_len)) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }
    const int tail = series_len - first;
    if (UNLIKELY(tail < max(rsi_period, wma_period))) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }

    const int warm = first + rsi_period + wma_period - 1;
    for (int r = 0; r < min(warm, series_len); ++r) out_tm[r * num_series + series] = f32_qnan();

    // Seed Wilder averages over first rsi_period diffs after first
    double avg_gain = 0.0, avg_loss = 0.0;
    const int seed_start = first + 1;
    const int seed_end = seed_start + rsi_period - 1;
    for (int i = seed_start; i <= seed_end; ++i) {
        const int idx = i * num_series + series;
        const int idx_prev = (i - 1) * num_series + series;
        const double d = (double)data_tm[idx] - (double)data_tm[idx_prev];
        if (d > 0.0) avg_gain += d; else avg_loss -= d;
    }
    const double rp_f = (double)rsi_period;
    avg_gain /= rp_f; avg_loss /= rp_f;
    const double alpha = 1.0 / rp_f;
    const double beta  = 1.0 - alpha;

    // LWMA state
    const int wp = wma_period;
    const double wp_f = (double)wp;
    const double denom_rcp = 1.0 / (0.5 * wp_f * (wp_f + 1.0));
    int head = 0, filled = 0;
    double S1 = 0.0, S2 = 0.0;
    extern __shared__ float shbuf[];
    // Use per-thread slice of shared memory as ring buffer: [threadIdx.x * wp .. +wp)
    float* ring = shbuf ? (shbuf + threadIdx.x * wma_period) : (float*)0;

    for (int r = first + rsi_period; r < series_len; ++r) {
        if (r > first + rsi_period) {
            const int idx = r * num_series + series;
            const int idx_prev = (r - 1) * num_series + series;
            const double d = (double)data_tm[idx] - (double)data_tm[idx_prev];
            const double g = (d > 0.0) ? d : 0.0;
            const double l = (d > 0.0) ? 0.0 : -d;
            avg_gain = fma(avg_gain, beta, alpha * g);
            avg_loss = fma(avg_loss, beta, alpha * l);
        }
        const double rs = (avg_loss != 0.0) ? (avg_gain / avg_loss) : 100.0;
        const double rsi = 100.0 - 100.0 / (1.0 + rs);
        const float  x_f = (float)(0.1 * (rsi - 50.0));

        if (filled < wp) {
            S1 += (double)x_f;
            S2 = fma((double)(filled + 1), (double)x_f, S2);
            if (ring) ring[head] = x_f;
            head = (head + 1 == wp) ? 0 : head + 1;
            filled += 1;
            if (filled == wp) {
                const double wma = S2 * denom_rcp;
                out_tm[r * num_series + series] = tanhf((float)wma);
            }
        } else {
            const float x_old = ring ? ring[head] : 0.0f;
            if (ring) ring[head] = x_f;
            head = (head + 1 == wp) ? 0 : head + 1;
            const double S1_prev = S1;
            S1 = S1_prev + (double)x_f - (double)x_old;
            S2 = (S2 - S1_prev) + wp_f * (double)x_f;
            const double wma = S2 * denom_rcp;
            out_tm[r * num_series + series] = tanhf((float)wma);
        }
    }
}
