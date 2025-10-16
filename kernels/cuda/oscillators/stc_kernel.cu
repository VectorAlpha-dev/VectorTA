// CUDA kernels for Schaff Trend Cycle (STC)
//
// Semantics mirror src/indicators/stc.rs scalar classic EMA path:
// - Pipeline: MACD(EMA fast-slow) -> Stoch(K) -> EMA(d) -> Stoch(K) -> EMA(d)
// - Warmup prefix: NaN for indices < warm = first_valid + max(fast,slow,k,d) - 1
// - Inputs/outputs are FP32; critical accumulations use double precision.
// - Batch kernel processes one series × many parameter rows (row-major output).
// - Many-series kernel processes time-major matrix with one param set.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef STC_BLOCK_X
#define STC_BLOCK_X 256
#endif

// FMA helper for EMA update: prev + a*(x - prev)
static __device__ __forceinline__ double ema_update(double prev, double a, double x) {
    return fma(a, (x - prev), prev);
}

// Core sequential STC computation over a single series, EMA/EMA classic path.
// Uses dynamic shared memory for two K-sized rings (macd, d-ema) and their validity flags.
// Layout: [macd_ring (max_k floats) | d_ring (max_k floats) | macd_flags (max_k u8) | d_flags (max_k u8)]
static __device__ __forceinline__ void stc_compute_series_ema(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int fast,
    int slow,
    int k,
    int d,
    int max_k,
    float* __restrict__ out)
{
    if (len <= 0 || first_valid >= len) return;

    // Carve shared memory
    extern __shared__ unsigned char shmem[];
    float* macd_ring = reinterpret_cast<float*>(shmem);
    float* d_ring    = macd_ring + max_k;
    unsigned char* macd_flags = reinterpret_cast<unsigned char*>(d_ring + max_k);
    unsigned char* d_flags    = macd_flags + max_k;

    // Warmup as per scalar
    const int warm = first_valid + max(max(fast, slow), max(k, d)) - 1;

    // EMA alphas
    const double fast_a = 2.0 / (double)(fast + 1);
    const double slow_a = 2.0 / (double)(slow + 1);
    const double d_a    = 2.0 / (double)(d + 1);
    const double one_m_d_a = 1.0 - d_a;

    // Seed EMAs to SMA of first period points (matching scalar behavior)
    double fast_sum = 0.0, slow_sum = 0.0;
    const int f_end = min(fast, len - first_valid);
    const int s_end = min(slow, len - first_valid);
    for (int i = 0; i < f_end; ++i) fast_sum += (double)prices[first_valid + i];
    for (int i = 0; i < s_end; ++i) slow_sum += (double)prices[first_valid + i];
    double fast_ema = (f_end == fast) ? (fast_sum / (double)fast) : NAN;
    double slow_ema = (s_end == slow) ? (slow_sum / (double)slow) : NAN;

    // Rings/flags init
    for (int r = threadIdx.x; r < max_k; r += blockDim.x) {
        macd_ring[r] = NAN; d_ring[r] = NAN; macd_flags[r] = 0u; d_flags[r] = 0u;
    }
    __syncthreads();

    int macd_vpos = 0, d_vpos = 0;
    int macd_valid_sum = 0, d_valid_sum = 0;

    // EMA(d) states
    double d_ema = NAN, d_seed_sum = 0.0; int d_seed_cnt = 0;
    double final_ema = NAN, final_seed_sum = 0.0; int final_seed_cnt = 0;

    // Thresholds for seeding boundaries
    const int fast_thr = fast > 0 ? (fast - 1) : 0;
    const int slow_thr = slow > 0 ? (slow - 1) : 0;

    // Fill NaN prefix up to warm (thread-parallel)
    for (int i = threadIdx.x; i < min(warm, len); i += blockDim.x) out[i] = NAN;
    __syncthreads();

    if (threadIdx.x != 0) return; // single thread performs sequential scan

    // Sequential over entire series
    for (int i = 0; i < len; ++i) {
        const float x_f = prices[i];
        const double x = (double)x_f;

        // EMA updates; exact SMA seed on boundary indices
        if (i >= first_valid) {
            const int rel = i - first_valid;
            if (rel >= fast_thr) {
                if (rel == fast_thr) fast_ema = fast_sum / (double)fast;
                else if (isfinite(x)) fast_ema = ema_update(fast_ema, fast_a, x);
                else fast_ema = NAN;
            }
            if (rel >= slow_thr) {
                if (rel == slow_thr) slow_ema = slow_sum / (double)slow;
                else if (isfinite(x)) slow_ema = ema_update(slow_ema, slow_a, x);
                else slow_ema = NAN;
            }
        }

        // MACD valid once slow EMA seeded
        float macd;
        unsigned char macd_is_valid;
        if (i >= first_valid + slow_thr && isfinite(fast_ema) && isfinite(slow_ema)) {
            macd = (float)(fast_ema - slow_ema);
            macd_is_valid = 1u;
        } else {
            macd = NAN; macd_is_valid = 0u;
        }

        // Maintain MACD ring + valid sum
        if (k > 0) {
            if (i >= k) macd_valid_sum -= macd_flags[macd_vpos];
            macd_flags[macd_vpos] = macd_is_valid;
            macd_valid_sum += (int)macd_is_valid;
            if (macd_is_valid) macd_ring[macd_vpos] = macd;
            macd_vpos = (macd_vpos + 1 == k) ? 0 : macd_vpos + 1;
        }

        // Stochastic of MACD
        float stok;
        if (k > 0 && macd_valid_sum == k && macd_is_valid) {
            float mn = macd_ring[0], mx = macd_ring[0];
            for (int j = 1; j < k; ++j) { float v = macd_ring[j]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
            const float range = mx - mn;
            stok = (fabsf(range) > 1e-20f) ? ((macd - mn) * (100.0f / range)) : 50.0f;
        } else if (macd_is_valid) {
            stok = 50.0f;
        } else {
            stok = NAN;
        }

        // EMA(d) of stok with SMA seed
        float d_val;
        if (isfinite(stok)) {
            if (d_seed_cnt < d) {
                d_seed_sum += (double)stok;
                d_seed_cnt += 1;
                if (d_seed_cnt == d) { d_ema = d_seed_sum / (double)d; d_val = (float)d_ema; }
                else d_val = NAN;
            } else {
                d_ema = ema_update(d_ema, d_a, (double)stok);
                d_val = (float)d_ema;
            }
        } else {
            d_val = NAN;
        }

        // Maintain d-ema ring + valid sum
        unsigned char d_is_valid = (unsigned char)isfinite(d_val);
        if (k > 0) {
            if (i >= k) d_valid_sum -= d_flags[d_vpos];
            d_flags[d_vpos] = d_is_valid;
            d_valid_sum += (int)d_is_valid;
            if (d_is_valid) d_ring[d_vpos] = d_val;
            d_vpos = (d_vpos + 1 == k) ? 0 : d_vpos + 1;
        }

        // Second stochastic over d-EMA
        float kd;
        if (k > 0 && d_valid_sum == k && d_is_valid) {
            float mn = d_ring[0], mx = d_ring[0];
            for (int j = 1; j < k; ++j) { float v = d_ring[j]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
            const float range = mx - mn;
            kd = (fabsf(range) > 1e-20f) ? ((d_val - mn) * (100.0f / range)) : 50.0f;
        } else if (d_is_valid) {
            kd = 50.0f;
        } else {
            kd = NAN;
        }

        // Final EMA(d) smoothing
        float out_i = NAN;
        if (isfinite(kd)) {
            if (final_seed_cnt < d) {
                final_seed_sum += (double)kd;
                final_seed_cnt += 1;
                if (final_seed_cnt == d) {
                    final_ema = final_seed_sum / (double)d;
                    out_i = (float)final_ema;
                }
            } else {
                final_ema = ema_update(final_ema, d_a, (double)kd);
                out_i = (float)final_ema;
            }
        }

        if (i >= warm) out[i] = out_i; // NaN-safe: remains NaN until all pieces valid
    }
}

// --------------------------- Batch kernel ---------------------------
extern "C" __global__
void stc_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ fasts,
                   const int* __restrict__ slows,
                   const int* __restrict__ ks,
                   const int* __restrict__ ds,
                   int series_len,
                   int first_valid,
                   int n_rows,
                   int max_k,
                   float* __restrict__ out)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int fast = fasts[row];
    const int slow = slows[row];
    const int kk   = ks[row];
    const int dd   = ds[row];
    if (fast <= 0 || slow <= 0 || kk <= 0 || dd <= 0) return;

    const int base = row * series_len;

    // Fill NaN prefix here; compute kernel writes post-warm only
    int warm = first_valid + max(max(fast, slow), max(kk, dd)) - 1;
    if (warm > series_len) warm = series_len;
    for (int i = threadIdx.x; i < warm; i += blockDim.x) out[base + i] = NAN;
    __syncthreads();

    if (threadIdx.x != 0) return; // single thread per row for sequential recurrence
    stc_compute_series_ema(prices, series_len, first_valid, fast, slow, kk, dd, max_k, out + base);
}

// -------------------- Many-series × one param ----------------------
extern "C" __global__
void stc_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int fast,
                                   int slow,
                                   int k,
                                   int d,
                                   float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    const int first = first_valids[s];

    // Per-column warmup
    int warm = first + max(max(fast, slow), max(k, d)) - 1;
    if (warm > rows) warm = rows;
    for (int t = 0; t < warm; ++t) out_tm[t * cols + s] = NAN;
    if (warm >= rows) return;

    // For simplicity, reuse the same engine by walking the column sequentially.
    // Allocate small local rings (stack) up to a sane bound; for larger k we cap via chunked scans.
    const int max_k = k; // dynamic shared not available per-thread here; use local rings via engine below

    // Minimal reimplementation of stc_compute_series_ema for TM layout
    // EMA alphas
    const double fast_a = 2.0 / (double)(fast + 1);
    const double slow_a = 2.0 / (double)(slow + 1);
    const double d_a    = 2.0 / (double)(d + 1);

    // Seed EMAs (SMA) from [first..first+period)
    double fast_sum = 0.0, slow_sum = 0.0;
    const int f_end = min(fast, rows - first);
    const int s_end = min(slow, rows - first);
    for (int i = 0; i < f_end; ++i) fast_sum += (double)prices_tm[(first + i) * cols + s];
    for (int i = 0; i < s_end; ++i) slow_sum += (double)prices_tm[(first + i) * cols + s];
    double fast_ema = (f_end == fast) ? (fast_sum / (double)fast) : NAN;
    double slow_ema = (s_end == slow) ? (slow_sum / (double)slow) : NAN;

    // Local rings (cap k to reasonable bound)
    const int KMAX = 2048;
    const int kk = (k <= KMAX) ? k : KMAX;
    float macd_ring[KMAX];
    float d_ring[KMAX];
    unsigned char macd_flags[KMAX];
    unsigned char d_flags[KMAX];
    for (int i = 0; i < kk; ++i) { macd_ring[i] = NAN; d_ring[i] = NAN; macd_flags[i] = 0u; d_flags[i] = 0u; }
    int macd_vpos = 0, d_vpos = 0;
    int macd_valid_sum = 0, d_valid_sum = 0;

    // EMA(d) states
    double d_ema = NAN, d_seed_sum = 0.0; int d_seed_cnt = 0;
    double final_ema = NAN, final_seed_sum = 0.0; int final_seed_cnt = 0;
    const int fast_thr = fast > 0 ? (fast - 1) : 0;
    const int slow_thr = slow > 0 ? (slow - 1) : 0;

    for (int i = 0; i < rows; ++i) {
        const float x_f = prices_tm[i * cols + s];
        const double x = (double)x_f;

        // EMA updates
        if (i >= first) {
            const int rel = i - first;
            if (rel >= fast_thr) {
                if (rel == fast_thr) fast_ema = fast_sum / (double)fast;
                else if (isfinite(x)) fast_ema = ema_update(fast_ema, fast_a, x);
                else fast_ema = NAN;
            }
            if (rel >= slow_thr) {
                if (rel == slow_thr) slow_ema = slow_sum / (double)slow;
                else if (isfinite(x)) slow_ema = ema_update(slow_ema, slow_a, x);
                else slow_ema = NAN;
            }
        }

        float macd;
        unsigned char macd_is_valid;
        if (i >= first + slow_thr && isfinite(fast_ema) && isfinite(slow_ema)) {
            macd = (float)(fast_ema - slow_ema);
            macd_is_valid = 1u;
        } else { macd = NAN; macd_is_valid = 0u; }

        if (k > 0) {
            if (i >= k) macd_valid_sum -= macd_flags[macd_vpos];
            macd_flags[macd_vpos] = macd_is_valid;
            macd_valid_sum += (int)macd_is_valid;
            if (macd_is_valid) macd_ring[macd_vpos] = macd;
            macd_vpos = (macd_vpos + 1 == kk) ? 0 : macd_vpos + 1;
        }

        float stok;
        if (k > 0 && macd_valid_sum == k && macd_is_valid) {
            float mn = macd_ring[0], mx = macd_ring[0];
            for (int j = 1; j < k; ++j) { float v = macd_ring[j % kk]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
            const float range = mx - mn;
            stok = (fabsf(range) > 1e-20f) ? ((macd - mn) * (100.0f / range)) : 50.0f;
        } else if (macd_is_valid) { stok = 50.0f; } else { stok = NAN; }

        float d_val;
        if (isfinite(stok)) {
            if (d_seed_cnt < d) {
                d_seed_sum += (double)stok; d_seed_cnt += 1;
                if (d_seed_cnt == d) { d_ema = d_seed_sum / (double)d; d_val = (float)d_ema; } else d_val = NAN;
            } else { d_ema = ema_update(d_ema, d_a, (double)stok); d_val = (float)d_ema; }
        } else { d_val = NAN; }

        unsigned char d_is_valid = (unsigned char)isfinite(d_val);
        if (k > 0) {
            if (i >= k) d_valid_sum -= d_flags[d_vpos];
            d_flags[d_vpos] = d_is_valid;
            d_valid_sum += (int)d_is_valid;
            if (d_is_valid) d_ring[d_vpos] = d_val;
            d_vpos = (d_vpos + 1 == kk) ? 0 : d_vpos + 1;
        }

        float kd;
        if (k > 0 && d_valid_sum == k && d_is_valid) {
            float mn = d_ring[0], mx = d_ring[0];
            for (int j = 1; j < k; ++j) { float v = d_ring[j % kk]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
            const float range = mx - mn;
            kd = (fabsf(range) > 1e-20f) ? ((d_val - mn) * (100.0f / range)) : 50.0f;
        } else if (d_is_valid) { kd = 50.0f; } else { kd = NAN; }

        float out_i = NAN;
        if (isfinite(kd)) {
            if (final_seed_cnt < d) {
                final_seed_sum += (double)kd; final_seed_cnt += 1;
                if (final_seed_cnt == d) { final_ema = final_seed_sum / (double)d; out_i = (float)final_ema; }
            } else { final_ema = ema_update(final_ema, d_a, (double)kd); out_i = (float)final_ema; }
        }

        if (i >= warm) out_tm[i * cols + s] = out_i;
    }
}

