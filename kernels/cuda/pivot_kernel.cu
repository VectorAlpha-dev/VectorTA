// Pivot indicator CUDA kernels (FP32)
// - Batch: one series × many params (modes)
// - Many-series × one-param (time-major)
//
// Outputs layout for both kernels is row-major with rows = combos*9 (batch) or rows = rows*9 (many-series),
// columns = series length (batch: n) or number of series (many-series). Level order per combo:
//   [r4, r3, r2, r1, pp, s1, s2, s3, s4]

#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef FORCE_INLINE
#define FORCE_INLINE __forceinline__ __device__
#endif

static inline __device__ float f_nan() { return CUDART_NAN_F; }

static constexpr int LEVELS = 9;

// Core compute: assumes inputs are valid (non-NaN as required by mode).
FORCE_INLINE void pivot_compute_levels_core(
    const int mode, const float h, const float l, const float c, const float o,
    float &r4, float &r3, float &r2, float &r1, float &pp, float &s1, float &s2, float &s3, float &s4)
{
    const float d = h - l;

    // Default NaNs (for default case)
    r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();

    switch (mode) {
        // 0: Standard
        case 0: {
            pp = (h + l + c) * (1.0f / 3.0f);
            const float t2 = pp + pp;
            r1 = t2 - l;
            r2 = pp + d;
            s1 = t2 - h;
            s2 = pp - d;
            break;
        }
        // 1: Fibonacci
        case 1: {
            pp = (h + l + c) * (1.0f / 3.0f);
            r1 = fmaf(d, 0.382f, pp);
            r2 = fmaf(d, 0.618f, pp);
            r3 = fmaf(d, 1.000f, pp);
            s1 = fmaf(d, -0.382f, pp);
            s2 = fmaf(d, -0.618f, pp);
            s3 = fmaf(d, -1.000f, pp);
            break;
        }
        // 2: Demark (needs open)
        case 2: {
            const float p_lt = (h + (l + l) + c) * 0.25f;
            const float p_gt = ((h + h) + l + c) * 0.25f;
            const float p_eq = (h + l + (c + c)) * 0.25f;
            if (c < o)      pp = p_lt;
            else if (c > o) pp = p_gt;
            else            pp = p_eq;

            const float n_lt = (h + (l + l) + c) * 0.5f;
            const float n_gt = ((h + h) + l + c) * 0.5f;
            const float n_eq = (h + l + (c + c)) * 0.5f;
            const float n = (c < o) ? n_lt : ((c > o) ? n_gt : n_eq);
            r1 = n - l;
            s1 = n - h;
            break;
        }
        // 3: Camarilla
        case 3: {
            pp = (h + l + c) * (1.0f / 3.0f);
            // canonical coefficients
            const float c1 = 0.0916f, c2 = 0.183f, c3 = 0.275f, c4 = 0.55f;
            r1 = fmaf(d,  c1, c);
            r2 = fmaf(d,  c2, c);
            r3 = fmaf(d,  c3, c);
            r4 = fmaf(d,  c4, c);
            s1 = fmaf(d, -c1, c);
            s2 = fmaf(d, -c2, c);
            s3 = fmaf(d, -c3, c);
            s4 = fmaf(d, -c4, c);
            break;
        }
        // 4: Woodie (needs open)
        case 4: {
            pp = (h + l + (o + o)) * 0.25f;
            const float t2p = pp + pp;
            const float t2l = l + l;
            const float t2h = h + h;
            r1 = t2p - l;
            r2 = fmaf(d,  1.0f, pp);
            r3 = (t2p - t2l) + h;
            r4 = fmaf(d,  1.0f, r3);
            s1 = t2p - h;
            s2 = fmaf(d, -1.0f, pp);
            s3 = (l + t2p) - t2h;
            s4 = fmaf(d, -1.0f, s3);
            break;
        }
        default: { /* leave NaNs */ break; }
    }
}

// ================================================================
// Optimized one-series × many-params (modes) kernel (FP32)
// Reads H/L/C once per t; reads O once per t only if any mode needs it.
// ================================================================
extern "C" __global__
void pivot_batch_f32( // same symbol; NEW semantics: loops combos in-thread
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ open,
    const int*   __restrict__ modes,   // size = n_combos
    int n,
    int first_valid,
    int n_combos,
    float* __restrict__ out)           // size = (n_combos*9) * n, row-major planes
{
    // Back-compat: ignore any extra Y blocks the old launcher might create.
    if (blockIdx.y != 0) return;

    // Determine once per block whether any mode needs 'open'.
    __shared__ unsigned char s_need_open_any;
    if (threadIdx.x == 0) {
        unsigned char f = 0;
        for (int j = 0; j < n_combos; ++j) {
            const int m = modes[j];
            f |= (m == 2) | (m == 4); // Demark or Woodie
        }
        s_need_open_any = f;
    }
    __syncthreads();
    const bool need_open_any = (s_need_open_any != 0);

    // 1D grid-stride over the series (coalesced by t).
    const int stride = blockDim.x * gridDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride)
    {
        const float h = high[t];
        const float l = low[t];
        const float c = close[t];
        const float o = need_open_any ? open[t] : 0.0f; // read once if needed by any mode

        // Base validity for H/L/C (fast NaN test: x==x is false for NaN)
        const bool base_ok = (t >= first_valid) && (h == h) && (l == l) && (c == c);

        // Loop over all combos/modes; stores remain coalesced per plane for a given j.
        for (int j = 0; j < n_combos; ++j) {
            const int mode = modes[j];
            const bool need_o = (mode == 2) || (mode == 4);
            const bool valid  = base_ok && (!need_o || (o == o)); // only require O if the mode needs it

            // plane base for combo j at column t
            const int base = (j * LEVELS) * n + t;

            float r4, r3, r2, r1, pp, s1, s2, s3, s4;
            if (valid) {
                pivot_compute_levels_core(mode, h, l, c, o, r4, r3, r2, r1, pp, s1, s2, s3, s4);
            } else {
                r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();
            }

            // Coalesced per plane across the warp
            out[base + 0 * n] = r4;
            out[base + 1 * n] = r3;
            out[base + 2 * n] = r2;
            out[base + 3 * n] = r1;
            out[base + 4 * n] = pp;
            out[base + 5 * n] = s1;
            out[base + 6 * n] = s2;
            out[base + 7 * n] = s3;
            out[base + 8 * n] = s4;
        }
    }
}

// ================================================================
// Many-series × one-param (time-major) kernel (FP32)
// Only loads open_tm when the mode requires it; unchanged layout.
// ================================================================
extern "C" __global__
void pivot_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ open_tm,
    const int*   __restrict__ first_valids, // per series
    int cols,  // number of series (columns)
    int rows,  // length (rows) per series
    int mode,
    float* __restrict__ out_tm) // size = 9 * rows * cols (row-major planes)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series column
    if (s >= cols) return;

    const bool need_o = (mode == 2) || (mode == 4);
    const int first_valid = first_valids[s];

    // Walk time-major rows for this series column; loads/stores coalesced across s at each t.
    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;

        float r4, r3, r2, r1, pp, s1, s2, s3, s4;
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];
        const float o = need_o ? open_tm[idx] : 0.0f;

        const bool valid = (t >= first_valid) && (h == h) && (l == l) && (c == c) && (!need_o || (o == o));
        if (valid) {
            pivot_compute_levels_core(mode, h, l, c, o, r4, r3, r2, r1, pp, s1, s2, s3, s4);
        } else {
            r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();
        }

        // Write stacked planes (row-major): (level*rows + t, s)
        out_tm[(0 * rows + t) * cols + s] = r4;
        out_tm[(1 * rows + t) * cols + s] = r3;
        out_tm[(2 * rows + t) * cols + s] = r2;
        out_tm[(3 * rows + t) * cols + s] = r1;
        out_tm[(4 * rows + t) * cols + s] = pp;
        out_tm[(5 * rows + t) * cols + s] = s1;
        out_tm[(6 * rows + t) * cols + s] = s2;
        out_tm[(7 * rows + t) * cols + s] = s3;
        out_tm[(8 * rows + t) * cols + s] = s4;
    }
}

