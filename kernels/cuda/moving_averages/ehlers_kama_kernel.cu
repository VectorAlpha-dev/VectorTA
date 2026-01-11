// CUDA kernels for Ehlers Kaufman Adaptive Moving Average (KAMA).
//
// Mirrors the scalar implementation: each parameter combination is handled by
// a single thread block that walks the time series sequentially while
// maintaining the rolling volatility sum used to derive the adaptive smoothing
// factor. A second kernel covers the many-series × one-parameter path operating
// on time-major input.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// --- Tunables ---------------------------------------------------------------
#ifndef KAMA_USE_STREAMING_LOADS
#define KAMA_USE_STREAMING_LOADS 1   // set 0 to disable ld.global.cs
#endif

#ifndef KAMA_USE_PREFETCH
#define KAMA_USE_PREFETCH 0          // set 1 to enable prefetch.global.L2
#endif

#ifndef KAMA_PREFETCH_DIST
#define KAMA_PREFETCH_DIST 64        // steps ahead to prefetch when enabled
#endif

// Match the scalar Rust implementation exactly (see `ehlers_kama_scalar`):
// s = ((0.6667 * ER) + 0.0645)^2
static constexpr float SC_MUL = 0.6667f;
static constexpr float SC_ADD = 0.0645f;

// Kahan compensated add for improved accuracy of rolling L1 sum
__device__ __forceinline__
void kahan_add(float x, float &sum, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// Optional streaming load (evict-first) to reduce L1 pollution on read-once inputs
__device__ __forceinline__
float ld_streaming_f32(const float* p) {
#if KAMA_USE_STREAMING_LOADS && (__CUDA_ARCH__ >= 700)
    float v;
    asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
#else
    return *p;
#endif
}

// Optional L2 prefetch for a given address
__device__ __forceinline__
void prefetch_l2(const float* p) {
#if KAMA_USE_PREFETCH && (__CUDA_ARCH__ >= 700)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#endif
}

extern "C" __global__
void ehlers_kama_fill_nan_vec_f32(float* __restrict__ out, int len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    const float nan_f = __int_as_float(0x7fc00000);
    out[idx] = nan_f;
}

extern "C" __global__
void ehlers_kama_batch_f32(const float* __restrict__ prices,
                           const int*   __restrict__ periods,
                           int first_valid,
                           int series_len,
                           int n_combos,
                           float* __restrict__ out) {
    if (series_len <= 0) return;

    const float nan_f = __int_as_float(0x7fc00000);
    const int strideGrid = blockDim.x * gridDim.x;
    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += strideGrid) {

        const int base = combo * series_len;
        float* __restrict__ out_row = out + base;

        const int period = periods[combo];
        int first = first_valid;
        if (first < 0) first = 0;

        if (period <= 0 || period > series_len || first >= series_len) {
            for (int i = 0; i < series_len; ++i) out_row[i] = nan_f;
            continue;
        }
        const int tail_len = series_len - first;
        if (tail_len < period) {
            for (int i = 0; i < series_len; ++i) out_row[i] = nan_f;
            continue;
        }

        const int warm = first + period - 1;

        if (warm >= series_len) {
            for (int i = 0; i < series_len; ++i) out_row[i] = nan_f;
            continue;
        }

        // Warmup semantics: [0, warm) = NaN
        for (int i = 0; i < warm; ++i) out_row[i] = nan_f;

        const int start = warm;
        int delta_start = start - period + 1;
        if (delta_start < first + 1) delta_start = first + 1;

        // Rolling sum of last (period-1) |Δ|
        float delta_sum = 0.0f, delta_c = 0.0f;
        for (int k = delta_start; k <= start; ++k) {
            if (k > first) {
                const float a = ld_streaming_f32(prices + k);
                const float b = ld_streaming_f32(prices + (k - 1));
                kahan_add(fabsf(a - b), delta_sum, delta_c);
            }
        }

        // Seed the first KAMA value at index 'start'
        const float prev_price = (start > 0)
            ? ld_streaming_f32(prices + (start - 1))
            : ld_streaming_f32(prices + start);
        const float cur_price  = ld_streaming_f32(prices + start);

        const int anchor_idx   = start - (period - 1);    // >= 0 by construction
        const float anchor_p   = ld_streaming_f32(prices + anchor_idx);
        const float direction  = fabsf(cur_price - anchor_p);

        float ef = (delta_sum > 0.0f) ? (direction / delta_sum) : 0.0f;
        ef = __saturatef(ef);
        float sc = fmaf(SC_MUL, ef, SC_ADD);
        sc *= sc;

        float prev = fmaf(sc, cur_price - prev_price, prev_price);
        out_row[start] = prev;

        // Main loop (window already full; anchor and drop always valid)
        for (int i = start + 1; i < series_len; ++i) {
#if KAMA_USE_PREFETCH
            if (i + KAMA_PREFETCH_DIST < series_len)
                prefetch_l2(prices + (i + KAMA_PREFETCH_DIST));
#endif
            const float newest      = ld_streaming_f32(prices + i);
            const float newest_prev = ld_streaming_f32(prices + (i - 1));
            const float newest_diff = fabsf(newest - newest_prev);

            const int drop_i        = i - period;
            if (drop_i > first) {
                const float d0 = ld_streaming_f32(prices + drop_i);
                const float d1 = ld_streaming_f32(prices + (drop_i - 1));
                const float drop = fabsf(d0 - d1);
                const float net  = newest_diff - drop;
                kahan_add(net, delta_sum, delta_c);
                if (delta_sum < 0.0f) delta_sum = 0.0f;   // numerical safety
            } else {
                kahan_add(newest_diff, delta_sum, delta_c);
            }

            const int anchor_i  = i - (period - 1);       // >= 0 from here onward
            const float anchor  = ld_streaming_f32(prices + anchor_i);
            float ef_i = (delta_sum > 0.0f) ? (fabsf(newest - anchor) / delta_sum) : 0.0f;
            ef_i = __saturatef(ef_i);

            float sc_i = fmaf(SC_MUL, ef_i, SC_ADD);
            sc_i *= sc_i;
            prev = fmaf(sc_i, newest - prev, prev);

            out_row[i] = prev;
        }
    }
}

extern "C" __global__
void ehlers_kama_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                            int period,
                                            int num_series,
                                            int series_len,
                                            const int* __restrict__ first_valids,
                                            float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }

    if (period <= 0 || series_len <= 0) {
        return;
    }

    int first = first_valids[series_idx];
    if (first < 0) {
        first = 0;
    }
    if (first >= series_len) {
        return;
    }

    const int stride = num_series;
    const int warm = first + period - 1;
    const float nan_f = __int_as_float(0x7fc00000);

    const int warm_clamped = warm < series_len ? warm : series_len;
    for (int t = 0; t < warm_clamped; ++t) {
        out_tm[t * stride + series_idx] = nan_f;
    }

    if (warm >= series_len) {
        return;
    }

    const int start = warm;
    int delta_start = (start >= period) ? (start - period + 1) : (first + 1);
    if (delta_start < first + 1) {
        delta_start = first + 1;
    }

    float delta_sum = 0.0f, delta_c = 0.0f;
    for (int k = delta_start; k <= start; ++k) {
        if (k > first) {
            const int idx_cur = k * stride + series_idx;
            const int idx_prev = (k - 1) * stride + series_idx;
            float x = fabsf(prices_tm[idx_cur] - prices_tm[idx_prev]);
            kahan_add(x, delta_sum, delta_c);
        }
    }

    float prev;
    if (start > 0) {
        prev = prices_tm[(start - 1) * stride + series_idx];
    } else {
        prev = prices_tm[start * stride + series_idx];
    }
    const float current = prices_tm[start * stride + series_idx];
    const int dir_idx = start - (period - 1);
    float direction = 0.0f;
    if (dir_idx >= 0) {
        const int anchor = dir_idx * stride + series_idx;
        direction = fabsf(current - prices_tm[anchor]);
    }

    float ef = 0.0f;
    if (delta_sum > 0.0f) {
        ef = direction / delta_sum;
        if (ef > 1.0f) {
            ef = 1.0f;
        }
    }

    float sc = fmaf(SC_MUL, ef, SC_ADD);
    sc *= sc;
    prev = fmaf(sc, current - prev, prev);
    out_tm[start * stride + series_idx] = prev;

    for (int t = start + 1; t < series_len; ++t) {
        const int cur_idx = t * stride + series_idx;
        const int prev_idx = (t - 1) * stride + series_idx;
        const float newest = prices_tm[cur_idx];
        const float newest_diff = fabsf(newest - prices_tm[prev_idx]);
        const int drop_idx = t - period;
        if (drop_idx > first) {
            const int idx_drop = drop_idx * stride + series_idx;
            const int idx_drop_prev = (drop_idx - 1) * stride + series_idx;
            const float drop = fabsf(prices_tm[idx_drop] - prices_tm[idx_drop_prev]);
            float net = newest_diff - drop;
            kahan_add(net, delta_sum, delta_c);
            if (delta_sum < 0.0f) delta_sum = 0.0f;
        } else {
            kahan_add(newest_diff, delta_sum, delta_c);
        }

        // Optional periodic exact rebase
        if ((t & 127) == 0) {
            float s = 0.0f;
            for (int u = t - (period - 1) + 1; u <= t; ++u) {
                const int u_idx = u * stride + series_idx;
                const int u_prev = (u - 1) * stride + series_idx;
                s += fabsf(prices_tm[u_idx] - prices_tm[u_prev]);
            }
            delta_sum = s;
            delta_c = 0.0f;
        }

        const int anchor_idx = t - (period - 1);
        float dir = 0.0f;
        if (anchor_idx >= 0) {
            const int anchor = anchor_idx * stride + series_idx;
            dir = fabsf(newest - prices_tm[anchor]);
        }

        float ef_t = 0.0f;
        if (delta_sum > 0.0f) {
            ef_t = dir / delta_sum;
            if (ef_t > 1.0f) {
                ef_t = 1.0f;
            }
        }

        float sc_t = fmaf(SC_MUL, ef_t, SC_ADD);
        sc_t *= sc_t;
        prev = fmaf(sc_t, newest - prev, prev);
        out_tm[cur_idx] = prev;
    }
}

// 2D tiled many-series kernel: each thread handles one series sequentially.
// Threads in a block cover a tile of series of size (blockDim.x * blockDim.y).
// Grid.x tiles across all series; mapping:
//   lane = threadIdx.y * blockDim.x + threadIdx.x
//   series_idx = blockIdx.x * (blockDim.x * blockDim.y) + lane
extern "C" __global__
void ehlers_kama_multi_series_one_param_2d_f32(const float* __restrict__ prices_tm,
                                               int period,
                                               int ring_len,
                                               int num_series,
                                               int series_len,
                                               const int* __restrict__ first_valids,
                                               float* __restrict__ out_tm) {
    const int lane        = threadIdx.y * blockDim.x + threadIdx.x;
    const int tile_series = blockDim.x * blockDim.y;
    const int series_idx  = blockIdx.x * tile_series + lane;
    if (series_idx >= num_series || series_len <= 0 || period <= 0) return;

    int first = first_valids[series_idx];
    if (first < 0) first = 0;
    if (first >= series_len) return;

    const int stride = num_series;
    const int warm = first + period - 1;
    // Assumption: caller prefilled out_tm with NaN for the whole buffer.
    if (warm >= series_len) return;

    // Dynamic shared ring: sized by the launcher as `tile_series * (period-1) * sizeof(float)`.
    // We gate usage using `ring_len` so callers can disable the ring when dynamic shared
    // allocation exceeds device limits, while keeping the same kernel entrypoint.
    extern __shared__ float s_ring[];

    // Ring length is (period-1). Use shared-memory ring when within static limits,
    // else fall back to the legacy global-load path.
    const int rb_len = period - 1;

    // Static shared ring: sized for common tiles (<=128 threads) and rb_len<=128.
    // Fall back to legacy path if constraints not met.
    const int MAX_TILE_SERIES = 128;
    const int MAX_RING        = 128;

    bool use_ring = (tile_series <= MAX_TILE_SERIES) && (rb_len > 0) && (rb_len <= MAX_RING)
        && (ring_len == rb_len);

    float delta_sum = 0.0f, delta_c = 0.0f;

    if (use_ring) {
        float* ring = &s_ring[lane * rb_len];
        int head = 0;

        // Warmup: build the initial delta_sum and fill the ring with the last (period-1) |Δ|
        const int delta_start = (warm - period + 1 > first + 1) ? (warm - period + 1) : (first + 1);
        for (int k = delta_start; k <= warm; ++k) {
            const int cur  = k * stride + series_idx;
            const int prev = (k - 1) * stride + series_idx;
            const float d = fabsf(ld_streaming_f32(prices_tm + cur) -
                                  ld_streaming_f32(prices_tm + prev));
            ring[head++] = d; if (head == rb_len) head = 0;
            kahan_add(d, delta_sum, delta_c);
        }

        // Seed at 'warm'
        const int cur0_idx   = warm * stride + series_idx;
        const int prev0_idx  = (warm > 0 ? (warm - 1) : warm) * stride + series_idx;
        const float cur0     = ld_streaming_f32(prices_tm + cur0_idx);
        const float prev_px0 = ld_streaming_f32(prices_tm + prev0_idx);

        const int anchor0_i  = warm - (period - 1);
        const int anchor0    = anchor0_i * stride + series_idx;
        const float dir0     = fabsf(cur0 - ld_streaming_f32(prices_tm + anchor0));

        float ef0 = (delta_sum > 0.0f) ? (dir0 / delta_sum) : 0.0f;
        ef0 = __saturatef(ef0);
        float sc0 = fmaf(SC_MUL, ef0, SC_ADD);
        sc0 *= sc0;

        float prev = fmaf(sc0, cur0 - prev_px0, prev_px0);
        out_tm[cur0_idx] = prev;

        // Main loop
        for (int t = warm + 1; t < series_len; ++t) {
#if KAMA_USE_PREFETCH
            if (t + KAMA_PREFETCH_DIST < series_len) {
                prefetch_l2(prices_tm + (t + KAMA_PREFETCH_DIST) * stride + series_idx);
            }
#endif
            const int cur_idx  = t * stride + series_idx;
            const int prv_idx  = (t - 1) * stride + series_idx;

            const float newest      = ld_streaming_f32(prices_tm + cur_idx);
            const float newest_prev = ld_streaming_f32(prices_tm + prv_idx);
            const float d_new       = fabsf(newest - newest_prev);

            const float d_drop = ring[head];      // drop the oldest |Δ|
            ring[head] = d_new;                   // push new |Δ|
            head = (head + 1) - (head + 1 == rb_len ? rb_len : 0);
            kahan_add(d_new - d_drop, delta_sum, delta_c);
            if (delta_sum < 0.0f) delta_sum = 0.0f;

            const int anchor_i = t - (period - 1);
            const int anchor   = anchor_i * stride + series_idx;
            float ef = (delta_sum > 0.0f)
                ? (fabsf(newest - ld_streaming_f32(prices_tm + anchor)) / delta_sum)
                : 0.0f;
            ef = __saturatef(ef);

            float sc = fmaf(SC_MUL, ef, SC_ADD);
            sc *= sc;
            prev = fmaf(sc, newest - prev, prev);

            out_tm[cur_idx] = prev;
        }
    } else {
        // Fallback: legacy path without shared ring (exactly previous semantics)
        const int start = warm;
        int delta_start = (start - period + 1);
        if (delta_start < first + 1) delta_start = first + 1;
        for (int k = delta_start; k <= start; ++k) {
            if (k > first) {
                const int idx_cur = k * stride + series_idx;
                const int idx_prev = (k - 1) * stride + series_idx;
                float x = fabsf(ld_streaming_f32(prices_tm + idx_cur) - ld_streaming_f32(prices_tm + idx_prev));
                kahan_add(x, delta_sum, delta_c);
            }
        }

        float prev;
        if (start > 0) {
            prev = ld_streaming_f32(prices_tm + (start - 1) * stride + series_idx);
        } else {
            prev = ld_streaming_f32(prices_tm + start * stride + series_idx);
        }
        const float current = ld_streaming_f32(prices_tm + start * stride + series_idx);
        const int dir_idx = start - (period - 1);
        float direction = 0.0f;
        if (dir_idx >= 0) {
            const int anchor = dir_idx * stride + series_idx;
            direction = fabsf(current - ld_streaming_f32(prices_tm + anchor));
        }

        float ef = (delta_sum > 0.0f) ? (direction / delta_sum) : 0.0f;
        ef = __saturatef(ef);
        float sc = fmaf(SC_MUL, ef, SC_ADD);
        sc *= sc;
        prev = fmaf(sc, current - prev, prev);
        out_tm[start * stride + series_idx] = prev;

        for (int t = start + 1; t < series_len; ++t) {
#if KAMA_USE_PREFETCH
            if (t + KAMA_PREFETCH_DIST < series_len) {
                prefetch_l2(prices_tm + (t + KAMA_PREFETCH_DIST) * stride + series_idx);
            }
#endif
            const int cur_idx = t * stride + series_idx;
            const int prev_idx = (t - 1) * stride + series_idx;
            const float newest = ld_streaming_f32(prices_tm + cur_idx);
            const float newest_diff = fabsf(newest - ld_streaming_f32(prices_tm + prev_idx));
            const int drop_idx = t - period;
            if (drop_idx > first) {
                const int idx_drop = drop_idx * stride + series_idx;
                const int idx_drop_prev = (drop_idx - 1) * stride + series_idx;
                const float drop = fabsf(ld_streaming_f32(prices_tm + idx_drop) - ld_streaming_f32(prices_tm + idx_drop_prev));
                float net = newest_diff - drop;
                kahan_add(net, delta_sum, delta_c);
                if (delta_sum < 0.0f) delta_sum = 0.0f;
            } else {
                kahan_add(newest_diff, delta_sum, delta_c);
            }

            const int anchor_idx = t - (period - 1);
            float dir = 0.0f;
            if (anchor_idx >= 0) {
                const int anchor = anchor_idx * stride + series_idx;
                dir = fabsf(newest - ld_streaming_f32(prices_tm + anchor));
            }

            float ef_t = (delta_sum > 0.0f) ? (dir / delta_sum) : 0.0f;
            ef_t = __saturatef(ef_t);

            float sc_t = fmaf(SC_MUL, ef_t, SC_ADD);
            sc_t *= sc_t;
            prev = fmaf(sc_t, newest - prev, prev);
            out_tm[cur_idx] = prev;
        }
    }
}

extern "C" __global__
void ehlers_kama_enforce_warm_nan_tm_f32(int period,
                                         int num_series,
                                         int series_len,
                                         const int* __restrict__ first_valids,
                                         float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;
    int first = first_valids[series_idx];
    if (first < 0) first = 0;
    if (first >= series_len) return;
    const int warm = first + period - 1;
    const int warm_clamped = warm < series_len ? warm : series_len;
    const int stride = num_series;
    const float nan_f = __int_as_float(0x7fc00000);
    for (int t = 0; t < warm_clamped; ++t) {
        out_tm[t * stride + series_idx] = nan_f;
    }
}

// Guard-rail fix: explicitly ensure first row (t=0) is NaN for any series with warm > 0.
extern "C" __global__
void ehlers_kama_fix_first_row_nan_tm_f32(int period,
                                          int num_series,
                                          const int* __restrict__ first_valids,
                                          float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;
    int first = first_valids[series_idx];
    if (first < 0) first = 0;
    const int warm = first + period - 1;
    if (warm > 0) {
        const float nan_f = __int_as_float(0x7fc00000);
        out_tm[series_idx] = nan_f; // t=0 -> 0 * stride + series_idx
    }
}
