// CUDA kernels for Chande Momentum Oscillator (CMO)
// Optimized for: one price series × many periods (combos)
//
// Changes vs. original:
// - Remove FP64 everywhere; use FP32 + compensated sum (Neumaier) for initial window.
// - Use FMA in Wilder updates for accuracy & throughput.
// - Tile the single price series into shared memory per block to eliminate
//   redundant global loads across combos; no extra build flags or dynamic
//   shared-mem sizes required.
// - Keep the same semantics, warmup, and zero-on-division-by-zero behavior.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CMO_NAN
#define CMO_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Tunables (compile-time; no wrapper changes)
#ifndef CMO_BLOCK_SIZE
#define CMO_BLOCK_SIZE 256
#endif
#ifndef CMO_TILE
#define CMO_TILE 256
#endif

// Light-weight Neumaier compensated adder for FP32 (2 registers).
// Used only for the initial window sums (period steps).
struct KBN32 {
    float s;  // running sum
    float c;  // compensation
    __device__ inline void init() { s = 0.0f; c = 0.0f; }
    __device__ inline void add(float x) {
        float t = s + x;
        if (fabsf(s) >= fabsf(x)) c += (s - t) + x;
        else                      c += (x - t) + s;
        s = t;
    }
    __device__ inline float result() const { return s + c; }
};

// Small helper: compute CMO value from avg gains/losses (handles zero denom).
__device__ inline float cmo_from_avgs(float avg_g, float avg_l) {
    float denom = avg_g + avg_l;
    if (denom == 0.0f) return 0.0f;
    float numer = avg_g - avg_l;
    return 100.0f * (numer / denom);
}

// =========================================================================
// One price series × many params (combos)
// prices: one series
// periods: per-combo period
// out:    row-major [combo][t]
// =========================================================================
extern "C" __global__ void cmo_batch_f32(
    const float*  __restrict__ prices,   // one series (FP32)
    const int*    __restrict__ periods,  // length = n_combos
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out              // length = n_combos * series_len
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + (size_t)combo * (size_t)series_len;

    // Basic validation mirroring wrapper guards
    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = CMO_NAN;
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail <= period)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = CMO_NAN;
        return;
    }

    const int fv   = first_valid;
    const int warm = fv + period; // first index with a defined output

    // Prefill NaN up to (warm-1)
    for (int i = 0; i < warm; ++i) out_row[i] = CMO_NAN;

    // Precompute alpha & beta for Wilder smoothing (FP32)
    const float beta  = 1.0f / (float)period;
    const float alpha = 1.0f - beta; // (period - 1) / period

    // ----- Initial averages over (fv+1 ..= warm) using compensated sum -----
    float prev = prices[fv];
    KBN32 sum_g, sum_l;
    sum_g.init(); sum_l.init();

    for (int i = fv + 1; i <= warm; ++i) {
        float curr = prices[i];
        float diff = curr - prev;
        prev = curr;
        float g = fmaxf(diff, 0.0f);
        float l = fmaxf(-diff, 0.0f);
        sum_g.add(g);
        sum_l.add(l);
    }
    float avg_g = sum_g.result() * beta;
    float avg_l = sum_l.result() * beta;

    // First defined output
    out_row[warm] = cmo_from_avgs(avg_g, avg_l);

    // ====== Rolling update with shared-memory tiling ======
    // Important: Threads in a block share the same tile range. Since each
    // combo can have a different 'warm', we stream tiles from fv+1 and only
    // start updating/writing once (t >= warm+1) for this thread.
    if (fv + 1 >= series_len) return;

    __shared__ float sh_prices[CMO_TILE + 1];

    int t = fv + 1; // common tile start for all threads in block
    while (t < series_len) {
        const int tile_start = t;
        const int tile_end   = min(series_len - 1, tile_start + CMO_TILE - 1);
        const int diffs_in_tile = tile_end - tile_start + 1;
        const int load_base = tile_start - 1;             // include previous element
        const int load_elems = diffs_in_tile + 1;         // T+1

        // Load tile into shared memory once per block.
        if (threadIdx.x == 0) {
            for (int j = 0; j < load_elems; ++j) {
                sh_prices[j] = prices[load_base + j];
            }
        }
        __syncthreads();

        // Stream through tile diffs
        float p0 = sh_prices[0];
        // Debug: verify second tile content for first combo/thread
        // (debug print removed)
        for (int j = 0; j < diffs_in_tile; ++j) {
            float p1 = sh_prices[j + 1];
            float diff = p1 - p0;
            p0 = p1;

            float g = fmaxf(diff, 0.0f);
            float l = fmaxf(-diff, 0.0f);

            // Only start rolling once past warm for this combo
            const int tt = t + j;
            if (tt >= warm + 1) {
                avg_g = __fmaf_rn(alpha, avg_g, beta * g);
                avg_l = __fmaf_rn(alpha, avg_l, beta * l);
                out_row[tt] = cmo_from_avgs(avg_g, avg_l);
            }
        }

        __syncthreads();
        t += diffs_in_tile;
    }
}

// =========================================================================
// Many series × one param, time-major layout (prices_tm: [row * num_series + series])
// Optimized for FP32 (remove FP64), KBN32 for initial window only, FMA in rolling.
// =========================================================================
extern "C" __global__ void cmo_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm // time-major
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(period <= 0 || period > series_len || fv < 0 || fv >= series_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = CMO_NAN;
        return;
    }
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= period)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = CMO_NAN;
        return;
    }

    const int warm = fv + period;
    const float beta  = 1.0f / (float)period;
    const float alpha = 1.0f - beta;

    // Prefill NaN up to warm-1
    {
        float* o = out_tm + series;
        for (int r = 0; r < warm; ++r, o += num_series) *o = CMO_NAN;
    }

    // Initial averages across (fv+1 ..= warm) using KBN32
    float prev = *(prices_tm + (size_t)fv * num_series + series);
    KBN32 sum_g, sum_l; sum_g.init(); sum_l.init();

    for (int r = fv + 1; r <= warm; ++r) {
        float curr = *(prices_tm + (size_t)r * num_series + series);
        float diff = curr - prev; prev = curr;
        float g = fmaxf(diff, 0.0f);
        float l = fmaxf(-diff, 0.0f);
        sum_g.add(g);
        sum_l.add(l);
    }
    float avg_g = sum_g.result() * beta;
    float avg_l = sum_l.result() * beta;

    *(out_tm + (size_t)warm * num_series + series) = cmo_from_avgs(avg_g, avg_l);

    // Rolling update across remaining rows
    for (int r = warm + 1; r < series_len; ++r) {
        float curr = *(prices_tm + (size_t)r * num_series + series);
        float diff = curr - prev; prev = curr;
        float g = fmaxf(diff, 0.0f);
        float l = fmaxf(-diff, 0.0f);
        avg_g = __fmaf_rn(alpha, avg_g, beta * g);
        avg_l = __fmaf_rn(alpha, avg_l, beta * l);
        *(out_tm + (size_t)r * num_series + series) = cmo_from_avgs(avg_g, avg_l);
    }
}
