// CUDA kernels for QQE (Quantitative Qualitative Estimation)
//
// Behavior rules:
// - Warmup/NaN semantics mirror the scalar Rust implementation.
// - Outputs are FP32; internal accumulations use double for stability.
// - Batch kernel: one block per parameter combo; thread 0 scans time sequentially
//   (recurrence). Other threads cooperatively prefill warmup NaNs.
// - Many-series kernel (time-major): one block per series; thread 0 scans time sequentially.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Compute QQE for a single series with given parameters.
// prices: input series (length N)
// N: series length
// first_valid: index of first finite element in prices
// rsi_p: RSI period (Wilder)
// ema_p: smoothing period for EMA of RSI
// fast_k: multiplier for ATR-of-RSI bands
// out_fast, out_slow: pointers to per-row outputs (length N)
static __device__ __forceinline__ void qqe_compute_series(
    const float* __restrict__ prices,
    int N,
    int first_valid,
    int rsi_p,
    int ema_p,
    float fast_k,
    float* __restrict__ out_fast,
    float* __restrict__ out_slow)
{
    if (N <= 0) return;
    if (first_valid >= N) return;
    if (rsi_p <= 0 || ema_p <= 0) return;

    const int rsi_start = first_valid + rsi_p;
    if (rsi_start >= N) return;
    const int warm = first_valid + rsi_p + ema_p - 2;

    // Wilder RSI initial averages over first rsi_p deltas
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    bool bad = false;
    const int init_end = min(first_valid + rsi_p, N - 1);
    for (int i = first_valid + 1; i <= init_end; ++i) {
        double di = (double)prices[i];
        double dim1 = (double)prices[i - 1];
        double delta = di - dim1;
        if (!isfinite(delta)) { bad = true; break; }
        if (delta > 0.0) avg_gain += delta; else if (delta < 0.0) avg_loss -= delta;
    }
    if (bad) return; // leave warmup NaNs in place

    const double inv_rsi = 1.0 / (double)rsi_p;
    const double beta_rsi = 1.0 - inv_rsi;
    avg_gain *= inv_rsi;
    avg_loss *= inv_rsi;

    // first RSI value at rsi_start
    double rsi;
    {
        const double denom = avg_gain + avg_loss;
        rsi = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);
    }
    // Initialize FAST at rsi_start
    if (rsi_start < N) {
        out_fast[rsi_start] = (float)rsi;
        // If warm <= rsi_start, also seed SLOW here
        if (warm <= rsi_start) out_slow[rsi_start] = (float)rsi;
    }

    // EMA-of-RSI warmup via running mean then recursive EMA
    double running_mean = rsi;
    const double ema_alpha = 2.0 / ((double)ema_p + 1.0);
    const double ema_beta = 1.0 - ema_alpha;
    double prev_ema = rsi;

    // QQE Slow line parameters (ATR-of-RSI style)
    const double atr_alpha = 1.0 / 14.0;
    const double atr_beta  = 1.0 - atr_alpha;
    double wwma = 0.0;
    double atrrsi = 0.0;
    double prev_fast_val = rsi;

    // Main scan from rsi_start+1 .. N-1
    for (int i = rsi_start + 1; i < N; ++i) {
        // Wilder RSI update
        double di = (double)prices[i];
        double dim1 = (double)prices[i - 1];
        double delta = di - dim1;
        double gain = (delta > 0.0) ? delta : 0.0;
        double loss = (delta < 0.0) ? -delta : 0.0;
        avg_gain = fma(beta_rsi, avg_gain, inv_rsi * gain);
        avg_loss = fma(beta_rsi, avg_loss, inv_rsi * loss);
        const double denom = avg_gain + avg_loss;
        rsi = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);

        // FAST
        double fast_i;
        if (i < rsi_start + ema_p) {
            // running mean during seed
            const double n = (double)(i - rsi_start + 1);
            running_mean = ((n - 1.0) * running_mean + rsi) / n;
            prev_ema = running_mean;
            fast_i = running_mean;
        } else {
            prev_ema = fma(ema_beta, prev_ema, ema_alpha * rsi);
            fast_i = prev_ema;
        }
        out_fast[i] = (float)fast_i;

        if (i == warm) {
            out_slow[i] = (float)fast_i; // anchor slow at warm
            prev_fast_val = fast_i;
        } else if (i > warm) {
            // QQE slow update
            const double tr = fabs(fast_i - prev_fast_val);
            wwma  = fma(atr_beta,  wwma,  atr_alpha * tr);
            atrrsi = fma(atr_beta, atrrsi, atr_alpha * wwma);
            const double qup = fast_i + atrrsi * (double)fast_k;
            const double qdn = fast_i - atrrsi * (double)fast_k;

            const double prev = (double)out_slow[i - 1];
            double slow;
            if (qup < prev) slow = qup;
            else if (fast_i > prev && prev_fast_val < prev) slow = qdn;
            else if (qdn > prev) slow = qdn;
            else if (fast_i < prev && prev_fast_val > prev) slow = qup;
            else slow = prev;
            out_slow[i] = (float)slow;
            prev_fast_val = fast_i;
        }
    }
}

// ----------------------------
// Batch: one series × many params
// Output layout: rows = 2 * n_combos, cols = series_len
//   row 2*c     = FAST
//   row 2*c + 1 = SLOW
// ----------------------------
extern "C" __global__ void qqe_batch_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ rsi_periods,
    const int*   __restrict__ ema_periods,
    const float* __restrict__ fast_factors,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int rsi_p = rsi_periods[combo];
    const int ema_p = ema_periods[combo];
    const float fast_k = fast_factors[combo];
    if (rsi_p <= 0 || ema_p <= 0) return;

    const int row_fast = 2 * combo;
    const int row_slow = row_fast + 1;
    float* __restrict__ out_fast = out + row_fast * series_len;
    float* __restrict__ out_slow = out + row_slow * series_len;

    // Warmup NaNs (only up to warm index)
    int warm = first_valid + rsi_p + ema_p - 2;
    if (warm > series_len) warm = series_len;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < warm; idx += gridDim.x * blockDim.x) {
        out_fast[idx] = NAN;
        out_slow[idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-thread sequential scan
    qqe_compute_series(prices, series_len, first_valid, rsi_p, ema_p, fast_k, out_fast, out_slow);
}

// ------------------------------------------------------------
// Many-series × one-param (time-major)
// prices_tm: [rows=series_len][cols=num_series]
// out_tm:    [rows=series_len][cols=2*num_series]  (col s = FAST, col s+num_series = SLOW)
// ------------------------------------------------------------
extern "C" __global__ void qqe_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int rsi_period,
    int ema_period,
    float fast_factor,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y;
    if (s >= num_series) return;
    if (rsi_period <= 0 || ema_period <= 0) return;

    const int fv = first_valids[s];
    // Prefill warmup NaNs for both outputs
    int warm = fv + rsi_period + ema_period - 2;
    if (warm > series_len) warm = series_len;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < warm; t += gridDim.x * blockDim.x) {
        out_tm[t * (2 * num_series) + s] = NAN;
        out_tm[t * (2 * num_series) + (s + num_series)] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;
    // Compute with a view into time-major layout
    // Wraps qqe_compute_series with strided accesses
    // Build contiguous views into temporary pointers via lambdas is not possible in C, so copy into
    // local helpers that index time-major buffers.

    // We reuse the same recurrence using direct indices.
    const int rsi_start = fv + rsi_period;
    if (rsi_start >= series_len) return;

    // Wilder RSI init
    double avg_gain = 0.0, avg_loss = 0.0;
    bool bad = false;
    const int init_end = min(fv + rsi_period, series_len - 1);
    for (int i = fv + 1; i <= init_end; ++i) {
        double di   = (double)prices_tm[i * num_series + s];
        double dim1 = (double)prices_tm[(i - 1) * num_series + s];
        double delta = di - dim1;
        if (!isfinite(delta)) { bad = true; break; }
        if (delta > 0.0) avg_gain += delta; else if (delta < 0.0) avg_loss -= delta;
    }
    if (bad) return;
    const double inv_rsi = 1.0 / (double)rsi_period;
    const double beta_rsi = 1.0 - inv_rsi;
    avg_gain *= inv_rsi; avg_loss *= inv_rsi;
    double rsi;
    {
        const double denom = avg_gain + avg_loss;
        rsi = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);
    }
    out_tm[rsi_start * (2 * num_series) + s] = (float)rsi;
    if (warm <= rsi_start) out_tm[rsi_start * (2 * num_series) + (s + num_series)] = (float)rsi;

    double running_mean = rsi;
    const double ema_alpha = 2.0 / ((double)ema_period + 1.0);
    const double ema_beta  = 1.0 - ema_alpha;
    double prev_ema = rsi;
    const double atr_alpha = 1.0 / 14.0;
    const double atr_beta  = 1.0 - atr_alpha;
    double wwma = 0.0, atrrsi = 0.0;
    double prev_fast_val = rsi;

    for (int i = rsi_start + 1; i < series_len; ++i) {
        double di   = (double)prices_tm[i * num_series + s];
        double dim1 = (double)prices_tm[(i - 1) * num_series + s];
        double delta = di - dim1;
        double gain = (delta > 0.0) ? delta : 0.0;
        double loss = (delta < 0.0) ? -delta : 0.0;
        avg_gain = fma(beta_rsi, avg_gain, inv_rsi * gain);
        avg_loss = fma(beta_rsi, avg_loss, inv_rsi * loss);
        const double denom = avg_gain + avg_loss;
        rsi = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);

        double fast_i;
        if (i < rsi_start + ema_period) {
            const double n = (double)(i - rsi_start + 1);
            running_mean = ((n - 1.0) * running_mean + rsi) / n;
            prev_ema = running_mean;
            fast_i = running_mean;
        } else {
            prev_ema = fma(ema_beta, prev_ema, ema_alpha * rsi);
            fast_i = prev_ema;
        }
        out_tm[i * (2 * num_series) + s] = (float)fast_i;

        if (i == warm) {
            out_tm[i * (2 * num_series) + (s + num_series)] = (float)fast_i;
            prev_fast_val = fast_i;
        } else if (i > warm) {
            const double tr = fabs(fast_i - prev_fast_val);
            wwma  = fma(atr_beta,  wwma,  atr_alpha * tr);
            atrrsi = fma(atr_beta, atrrsi, atr_alpha * wwma);
            const double qup = fast_i + atrrsi * (double)fast_factor;
            const double qdn = fast_i - atrrsi * (double)fast_factor;
            const double prev = (double)out_tm[(i - 1) * (2 * num_series) + (s + num_series)];
            double slow;
            if (qup < prev) slow = qup;
            else if (fast_i > prev && prev_fast_val < prev) slow = qdn;
            else if (qdn > prev) slow = qdn;
            else if (fast_i < prev && prev_fast_val > prev) slow = qup;
            else slow = prev;
            out_tm[i * (2 * num_series) + (s + num_series)] = (float)slow;
            prev_fast_val = fast_i;
        }
    }
}

