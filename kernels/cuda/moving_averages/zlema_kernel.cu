// Optimized CUDA kernels for ZLEMA (Zero Lag Exponential Moving Average)
// CUDA 13+, tuned for Ada (SM 89) and onward.
//
// What changed vs original:
// - EMA update uses FMA-friendly form: last_ema += alpha * (val - last_ema)
//   This reduces instruction count and rounding error.
// - Added a high-throughput tiled batch kernel that stages prices in shared
//   memory per tile with a halo of max_lag to reduce global traffic.
// - Many-series × one-param kernel splits pre-lag and main sections to
//   eliminate a per-iteration branch.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef ZLEMA_NAN
#define ZLEMA_NAN (__int_as_float(0x7fffffff))
#endif

// On pre-Ampere this can help; on Ampere/Ada it's neutral.
// Keeps code portable across archs without hurting perf.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  #define LDG(p) __ldg(p)
#else
  #define LDG(p) (*(p))
#endif

// Local min/max to avoid relying on non-standard macros
#ifndef ZL_MAX
#define ZL_MAX(a,b) (( (a) > (b) ) ? (a) : (b))
#endif
#ifndef ZL_MIN
#define ZL_MIN(a,b) (( (a) < (b) ) ? (a) : (b))
#endif

extern "C" __global__
void zlema_batch_f32(const float* __restrict__ prices,
                     const int*   __restrict__ periods,
                     const int*   __restrict__ lags,
                     const float* __restrict__ alphas,
                     int series_len,
                     int n_combos,
                     int first_valid,
                     float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int   period = periods[combo];
    const int   lag    = lags[combo];
    const float alpha  = alphas[combo];

    const int   warm   = first_valid + period - 1;
    const size_t base  = (size_t)combo * (size_t)series_len;

    // Initialize warmup prefix to NaN
    for (int i = 0; i < warm && i < series_len; ++i) {
        out[base + i] = ZLEMA_NAN;
    }
    if (first_valid >= series_len) return;

    float last_ema = LDG(prices + first_valid);

    // period == 1 -> warm == first_valid
    if (warm <= first_valid) {
        out[base + first_valid] = last_ema;
    }

    // Main loop (serial by dependency)
    for (int t = first_valid + 1; t < series_len; ++t) {
        const float cur = LDG(prices + t);
        float val;
        if (t < first_valid + lag) {
            val = cur;
        } else {
            const float lagged = LDG(prices + (t - lag));
            // val = 2*cur - lagged with one FMA
            val = fmaf(2.0f, cur, -lagged);
        }

        // EMA update in numerically-stable, FMA-friendly form
        // last_ema = last_ema + alpha * (val - last_ema)
        last_ema = fmaf(alpha, (val - last_ema), last_ema);

        if (t >= warm) {
            out[base + t] = last_ema;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Warp-scan batch kernel (one warp per combo).
// Emits 32 timesteps per iteration via an inclusive scan over affine transforms:
//   y_next = (1 - a) * y_prev + a * val_t  ==  A*y_prev + B, where A=(1-a), B=a*val_t
//
// Notes:
// - Warmup (t < warm) is handled by writing NaNs and updating EMA state sequentially for <= (period-1) steps.
// - For t >= warm, we always have t >= first_valid + lag, so val_t uses 2*cur - lagged (no per-step branch).
// - `blockDim.x` must be a multiple of 32; wrapper rounds/chooses accordingly.
// ---------------------------------------------------------------------------------------------------------------------
extern "C" __global__
void zlema_batch_warp_scan_f32(const float* __restrict__ prices,
                               const int*   __restrict__ periods,
                               int series_len,
                               int n_combos,
                               int first_valid,
                               float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block <= 0) return;

    const int combo = blockIdx.x * warps_per_block + warp_in_block;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int lag = (period - 1) >> 1; // (period - 1) / 2
    const float alpha = 2.0f / (float(period) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    const int warm = first_valid + period - 1;
    const size_t base = (size_t)combo * (size_t)series_len;

    // Warmup prefix NaNs
    for (int t = lane; t < warm && t < series_len; t += 32) {
        out[base + (size_t)t] = ZLEMA_NAN;
    }

    // period == 1 -> ZLEMA reduces to price (alpha == 1, lag == 0)
    if (period == 1) {
        for (int t = first_valid + lane; t < series_len; t += 32) {
            out[base + (size_t)t] = LDG(prices + t);
        }
        return;
    }

    if (warm >= series_len) return;

    // Warmup recurrence (sequential, <= 258 iterations for typical sweeps)
    float prev = 0.0f;
    if (lane == 0) {
        float ema = LDG(prices + first_valid);
        const int end = warm; // exclusive
        for (int t = first_valid + 1; t < end; ++t) {
            const float cur = LDG(prices + t);
            float val;
            if (t < first_valid + lag) {
                val = cur;
            } else {
                const float lagged = LDG(prices + (t - lag));
                val = fmaf(2.0f, cur, -lagged);
            }
            ema = fmaf(alpha, (val - ema), ema);
        }
        prev = ema;
    }

    const unsigned mask = 0xffffffffu;
    prev = __shfl_sync(mask, prev, 0);

    for (int t0 = warm; t0 < series_len; t0 += 32) {
        const int t = t0 + lane;

        // Affine transform for this timestep
        float A = 1.0f;
        float B = 0.0f;
        if (t < series_len) {
            const float cur = LDG(prices + t);
            const float lagged = LDG(prices + (t - lag));
            const float val = fmaf(2.0f, cur, -lagged);
            A = one_minus_alpha;
            B = alpha * val;
        }

        // Inclusive warp scan to compose transforms left-to-right
        // Composition: (A1,B1) o (A2,B2) = (A1*A2, A1*B2 + B1)
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }

        const float y = fmaf(A, prev, B);
        if (t < series_len) {
            out[base + (size_t)t] = y;
        }

        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        prev = __shfl_sync(mask, y, last_lane);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// New high-throughput batch kernel with shared-memory tiling.
// API change: extra argument `max_lag` (global maximum of lags[]).
// Shared memory per block: (ZLEMA_BATCH_TILE + max_lag) * sizeof(float)
// ---------------------------------------------------------------------------------------------------------------------

#ifndef ZLEMA_BATCH_TILE
#define ZLEMA_BATCH_TILE 1024  // Tunable
#endif

extern "C" __global__
void zlema_batch_f32_tiled_f32(const float* __restrict__ prices,
                               const int*   __restrict__ periods,
                               const int*   __restrict__ lags,
                               const float* __restrict__ alphas,
                               int series_len,
                               int n_combos,
                               int first_valid,
                               int max_lag,                 // <-- NEW
                               float* __restrict__ out)
{
    extern __shared__ float s_prices[]; // size = (ZLEMA_BATCH_TILE + max_lag)

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int   period = periods[combo];
    const int   lag    = lags[combo];
    const float alpha  = alphas[combo];
    const int   warm   = first_valid + period - 1;
    const size_t base  = (size_t)combo * (size_t)series_len;

    // Prefix NaNs
    for (int i = 0; i < warm && i < series_len; ++i) {
        out[base + i] = ZLEMA_NAN;
    }
    if (first_valid >= series_len) return;

    float last_ema = LDG(prices + first_valid);

    // period == 1 -> warm == first_valid
    if (warm <= first_valid) {
        out[base + first_valid] = last_ema;
    }

    // Tiled scan over time dimension, starting from first_valid+1
    const int start_t = first_valid + 1;
    if (start_t >= series_len) return;

    for (int tile_start = start_t; tile_start < series_len; tile_start += ZLEMA_BATCH_TILE) {
        const int load_start = ZL_MAX(tile_start - max_lag, 0);
        const int load_end   = ZL_MIN(tile_start + ZLEMA_BATCH_TILE, series_len);
        const int sh_len     = load_end - load_start;

        // Cooperative load of [load_start, load_end)
        for (int i = threadIdx.x; i < sh_len; i += blockDim.x) {
            s_prices[i] = LDG(prices + (load_start + i));
        }
        __syncthreads();

        const int t_end = load_end; // tile_end clamped to series_len

        // Within this tile, all needed (cur and lagged) should be in shared memory
        // because we loaded a halo of size max_lag before tile_start.
        for (int t = tile_start; t < t_end; ++t) {
            const float cur = s_prices[t - load_start];
            float val;
            if (t < first_valid + lag) {
                val = cur;
            } else {
                const float lagged = s_prices[(t - lag) - load_start];
                val = fmaf(2.0f, cur, -lagged);
            }
            last_ema = fmaf(alpha, (val - last_ema), last_ema);

            if (t >= warm) {
                out[base + t] = last_ema;
            }
        }
        __syncthreads();
    }
}

// Many-series × one-param (time-major layout)
// prices_tm/out_tm are time-major: index = row * num_series + series
extern "C" __global__
void zlema_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int*   __restrict__ first_valids,
                                     int num_series,
                                     int series_len,
                                     int period,
                                     float alpha,
                                     float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int stride      = num_series;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int   lag   = (period - 1) / 2;
    const int   warm  = first_valid + period - 1;
    const size_t col  = (size_t)series;

    // Initialize warmup prefix to NaN
    for (int row = 0; row < warm && row < series_len; ++row) {
        out_tm[(size_t)row * stride + col] = ZLEMA_NAN;
    }

    float last_ema = LDG(prices_tm + ((size_t)first_valid * stride + col));

    // Handle period == 1 explicitly: warm == first_valid
    if (warm <= first_valid) {
        out_tm[((size_t)first_valid * stride) + col] = last_ema;
    }

    // ---- Pre-lag part: use current only ----
    int t = first_valid + 1;
    const int prelag_end = ZL_MIN(series_len, first_valid + lag);
    for (; t < prelag_end; ++t) {
        const float cur = LDG(prices_tm + ((size_t)t * stride + col));
        last_ema = fmaf(alpha, (cur - last_ema), last_ema);
        if (t >= warm) {
            out_tm[((size_t)t * stride) + col] = last_ema;
        }
    }

    // ---- Main part: use 2*cur - lagged ----
    for (; t < series_len; ++t) {
        const float cur    = LDG(prices_tm + ((size_t)t * stride + col));
        const float lagged = LDG(prices_tm + ((size_t)(t - lag) * stride + col));
        const float val    = fmaf(2.0f, cur, -lagged);
        last_ema = fmaf(alpha, (val - last_ema), last_ema);
        if (t >= warm) {
            out_tm[((size_t)t * stride) + col] = last_ema;
        }
    }
}
