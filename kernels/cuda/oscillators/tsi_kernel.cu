// CUDA kernels for True Strength Index (TSI)
//
// Matches scalar semantics in src/indicators/tsi.rs:
// - Warmup: first_valid + long + short elements are NaN
// - Use EMA on momentum and on abs(momentum) with periods long, short
// - NaN inputs do not advance state; output becomes NaN at those indices
// - Division by zero => NaN; outputs clamped to [-100, 100]
// - FP32 math; use FMA where beneficial
//
// This file was updated to remove FP64 from hot loops on Ada/RTX 40xx and
// to use a compensated FP32 EMA update (Kahan-style) with stable form
//   y += a * (x - y)
// to preserve accuracy without the 1/64 FP64 throughput penalty.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

// --- FP32 compensated EMA helper ---
struct EmaKahan {
    float y; // running EMA value
    float c; // compensation for low bits lost in y += incr
};

static __device__ __forceinline__ void ema_seed(EmaKahan& s, float seed) {
    s.y = seed;
    s.c = 0.0f;
}

// Stable EMA update using compensated summation on the increment:
// s.y += a * (x - s.y)
static __device__ __forceinline__ void ema_update(EmaKahan& s, float a, float x) {
    float d    = x - s.y;
    float incr = fmaf(a, d, 0.0f); // a*d in one rounding
    float u    = incr - s.c;
    float t    = s.y + u;
    s.c        = (t - s.y) - u; // new compensation
    s.y        = t;
}

extern "C" __global__
void tsi_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ longs,
                   const int* __restrict__ shorts,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    // Keep existing launch mapping (1 block per combo) to preserve wrappers.
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    if (threadIdx.x != 0) return; // sequential time scan per combo
    if (first_valid < 0 || first_valid + 1 >= series_len) return;

    const int base = combo * series_len;
    const int L = longs[combo];
    const int S = shorts[combo];
    if (L <= 0 || S <= 0) return;

    const int warm = first_valid + L + S;
    // Only initialize warmup prefix to NaN (avoid full-row writes)
    const int warm_stop = warm < series_len ? warm : series_len;
    for (int i = 0; i < warm_stop; ++i) out[base + i] = NAN;

    const float aL = 2.0f / (float(L) + 1.0f);
    const float aS = 2.0f / (float(S) + 1.0f);

    float prev  = prices[first_valid];
    float nextv = prices[first_valid + 1];
    if (!isfinite(nextv)) return; // follow scalar early-exit behavior

    float first_m = nextv - prev;
    prev = nextv;

    EmaKahan numL, numS, denL, denS;
    ema_seed(numL, first_m);
    ema_seed(numS, first_m);
    float first_am = fabsf(first_m);
    ema_seed(denL, first_am);
    ema_seed(denS, first_am);

    for (int i = first_valid + 2; i < series_len; ++i) {
        float cur = prices[i];
        if (!isfinite(cur)) {
            if (i >= warm) out[base + i] = NAN;
            continue;
        }
        float m = cur - prev; prev = cur;
        float am = fabsf(m);

        // Compensated FP32 EMA updates: stable form with FMA
        ema_update(numL, aL, m);        ema_update(numS, aS, numL.y);
        ema_update(denL, aL, am);       ema_update(denS, aS, denL.y);

        if (i >= warm) {
            float den = denS.y;
            if (den == 0.0f || !isfinite(den)) {
                out[base + i] = NAN;
            } else {
                float v = 100.0f * (numS.y / den);
                out[base + i] = clampf(v, -100.0f, 100.0f);
            }
        }
    }
}

// Many-series (time-major), one param. Output time-major [rows x cols].
extern "C" __global__
void tsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   int long_p,
                                   int short_p,
                                   int num_series,
                                   int series_len,
                                   const int* __restrict__ first_valids,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index (column)
    if (s >= num_series) return;
    if (long_p <= 0 || short_p <= 0) return;

    const float aL = 2.0f / (float(long_p) + 1.0f);
    const float aS = 2.0f / (float(short_p) + 1.0f);

    const int first = max(0, first_valids[s]);
    if (first >= series_len) return;
    const int warm = first + long_p + short_p;

    // Initialize prefix to NaN for this series
    const int warm_stop = warm < series_len ? warm : series_len;
    for (int t = 0; t < warm_stop; ++t) {
        out_tm[t * num_series + s] = NAN;
    }

    if (series_len <= first + 1) return;
    const int idx0 = first * num_series + s;
    float prev  = prices_tm[idx0];
    float nextv = prices_tm[idx0 + num_series];
    if (!isfinite(nextv)) return;

    float first_m = nextv - prev;
    prev = nextv;

    EmaKahan numL, numS, denL, denS;
    ema_seed(numL, first_m);
    ema_seed(numS, first_m);
    float first_am = fabsf(first_m);
    ema_seed(denL, first_am);
    ema_seed(denS, first_am);

    for (int t = first + 2; t < series_len; ++t) {
        const int idx = t * num_series + s;
        float cur = prices_tm[idx];
        if (!isfinite(cur)) {
            if (t >= warm) out_tm[idx] = NAN;
            continue;
        }

        float m = cur - prev; prev = cur;
        float am = fabsf(m);

        ema_update(numL, aL, m);        ema_update(numS, aS, numL.y);
        ema_update(denL, aL, am);       ema_update(denS, aS, denL.y);

        if (t >= warm) {
            float den = denS.y;
            if (den == 0.0f || !isfinite(den)) {
                out_tm[idx] = NAN;
            } else {
                float v = 100.0f * (numS.y / den);
                out_tm[idx] = clampf(v, -100.0f, 100.0f);
            }
        }
    }
}

// ----------------- New momentum precompute (one series) -----------------
extern "C" __global__
void tsi_prepare_momentum_f32(const float* __restrict__ prices,
                              int series_len,
                              int first_valid,
                              float* __restrict__ mom,   // len = series_len
                              float* __restrict__ amom)  // len = series_len
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    // Initialize outputs to NaN
    for (int i = 0; i < series_len; ++i) { mom[i] = NAN; amom[i] = NAN; }
    if (first_valid < 0 || first_valid + 1 >= series_len) return;

    float prev = prices[first_valid];
    float nextv = prices[first_valid + 1];
    if (!isfinite(nextv)) return; // match early-exit semantics

    float first_m = nextv - prev;
    mom[first_valid + 1]  = first_m;
    amom[first_valid + 1] = fabsf(first_m);
    prev = nextv;

    for (int t = first_valid + 2; t < series_len; ++t) {
        float cur = prices[t];
        if (!isfinite(cur)) {
            // do not advance prev; leave NaNs in outputs for this t
            continue;
        }
        float m = cur - prev;
        prev = cur;
        mom[t]  = m;
        amom[t] = fabsf(m);
    }
}

// -------- New param-parallel kernel: one series × many params (time-major) --------
extern "C" __global__
void tsi_one_series_many_params_tm_f32(const float* __restrict__ mom,
                                       const float* __restrict__ amom,
                                       const int*   __restrict__ longs,
                                       const int*   __restrict__ shorts,
                                       int series_len,
                                       int first_valid,
                                       int n_combos,
                                       float* __restrict__ out_tm) { // [t][combo]
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int L = longs[combo];
    const int S = shorts[combo];
    if (L <= 0 || S <= 0) return;

    const int warm = first_valid + L + S;
    const float aL = 2.0f / (float(L) + 1.0f);
    const float aS = 2.0f / (float(S) + 1.0f);

    // Warmup prefix to NaN for this combo (time-major output)
    const int warm_stop = warm < series_len ? warm : series_len;
    for (int t = 0; t < warm_stop; ++t) {
        out_tm[t * n_combos + combo] = NAN;
    }
    if (first_valid + 1 >= series_len) return;

    // Seed using first defined momentum (must match prepare kernel's early-exit)
    float first_m = mom[first_valid + 1];
    if (!isfinite(first_m)) return;

    EmaKahan numL, numS, denL, denS;
    ema_seed(numL, first_m);
    ema_seed(numS, first_m);
    float first_am = fabsf(first_m);
    ema_seed(denL, first_am);
    ema_seed(denS, first_am);

    // Time scan
    for (int t = first_valid + 2; t < series_len; ++t) {
        const float m = mom[t];
        if (!isfinite(m)) {
            if (t >= warm) out_tm[t * n_combos + combo] = NAN;
            continue;
        }
        const float am = amom[t];

        // Double-smoothing via compensated updates: L then S
        ema_update(numL, aL, m);
        ema_update(numS, aS, numL.y);

        ema_update(denL, aL, am);
        ema_update(denS, aS, denL.y);

        if (t >= warm) {
            const float den = denS.y;
            if (den == 0.0f || !isfinite(den)) {
                out_tm[t * n_combos + combo] = NAN;
            } else {
                float v = 100.0f * (numS.y / den);
                out_tm[t * n_combos + combo] = clampf(v, -100.0f, 100.0f);
            }
        }
    }
}

// ----------------- Utility: 32x32 tiled transpose (TM -> RM) -----------------
// in_tm shape: [rows x cols] with time-major layout (rows=time, cols=combos)
// out_rm shape: [cols x rows] with row-major layout (rows=combos, cols=time)
extern "C" __global__
void transpose_tm_to_rm_f32(const float* __restrict__ in_tm,
                            int rows, int cols,
                            float* __restrict__ out_rm)
{
    __shared__ float tile[32][33]; // +1 to avoid shared bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x; // column in in_tm (combo)
    int y = blockIdx.y * 32 + threadIdx.y; // row in in_tm (time)

    // Load 32x32 tile from in_tm -> shared
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int yy = y + j;
        if (x < cols && yy < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in_tm[yy * cols + x];
        }
    }
    __syncthreads();

    // Transpose coordinates for store
    x = blockIdx.y * 32 + threadIdx.x; // time becomes x
    y = blockIdx.x * 32 + threadIdx.y; // combo becomes y

    // Store transposed tile into out_rm [combo][time]
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int yy = y + j; // combo index
        if (x < rows && yy < cols) {
            out_rm[yy * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
