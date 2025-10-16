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

FORCE_INLINE void pivot_compute_levels(int mode, float h, float l, float c, float o,
                                       float &r4, float &r3, float &r2, float &r1,
                                       float &pp, float &s1, float &s2, float &s3, float &s4) {
    // Default NaNs
    r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();

    // Guard required inputs per mode
    const bool need_o = (mode == 2) || (mode == 4); // Demark or Woodie
    if (isnan(h) || isnan(l) || isnan(c) || (need_o && isnan(o))) {
        return;
    }

    const float d = h - l;
    switch (mode) {
        // Standard
        case 0: {
            pp = (h + l + c) * (1.0f / 3.0f);
            const float t2 = pp + pp;
            r1 = t2 - l;
            r2 = pp + d;
            s1 = t2 - h;
            s2 = pp - d;
            // r3, r4, s3, s4 remain NaN
            break;
        }
        // Fibonacci
        case 1: {
            pp = (h + l + c) * (1.0f / 3.0f);
            r1 = fmaf(d, 0.382f, pp);
            r2 = fmaf(d, 0.618f, pp);
            r3 = fmaf(d, 1.000f, pp);
            s1 = fmaf(d, -0.382f, pp);
            s2 = fmaf(d, -0.618f, pp);
            s3 = fmaf(d, -1.000f, pp);
            // r4, s4 remain NaN
            break;
        }
        // Demark
        case 2: {
            // p uses quarters, n uses halves
            float p_lt = (h + (l + l) + c) * 0.25f;
            float p_gt = ((h + h) + l + c) * 0.25f;
            float p_eq = (h + l + (c + c)) * 0.25f;
            if (c < o)      pp = p_lt;
            else if (c > o) pp = p_gt;
            else            pp = p_eq;

            float n_lt = (h + (l + l) + c) * 0.5f;
            float n_gt = ((h + h) + l + c) * 0.5f;
            float n_eq = (h + l + (c + c)) * 0.5f;
            float n;
            if (c < o)      n = n_lt;
            else if (c > o) n = n_gt;
            else            n = n_eq;
            r1 = n - l;
            s1 = n - h;
            break;
        }
        // Camarilla
        case 3: {
            pp = (h + l + c) * (1.0f / 3.0f);
            const float c1 = 0.0916f, c2 = 0.183f, c3 = 0.275f, c4 = 0.55f;
            r1 = fmaf(d, c1, c);
            r2 = fmaf(d, c2, c);
            r3 = fmaf(d, c3, c);
            r4 = fmaf(d, c4, c);
            s1 = fmaf(d, -c1, c);
            s2 = fmaf(d, -c2, c);
            s3 = fmaf(d, -c3, c);
            s4 = fmaf(d, -c4, c);
            break;
        }
        // Woodie
        case 4: {
            pp = (h + l + (o + o)) * 0.25f;
            const float t2p = pp + pp;
            const float t2l = l + l;
            const float t2h = h + h;
            r1 = t2p - l;
            r2 = fmaf(d, 1.0f, pp);
            r3 = (t2p - t2l) + h;
            r4 = fmaf(d, 1.0f, r3);
            s1 = t2p - h;
            s2 = fmaf(d, -1.0f, pp);
            s3 = (l + t2p) - t2h;
            s4 = fmaf(d, -1.0f, s3);
            break;
        }
        default: {
            // leave NaNs
            break;
        }
    }
}

extern "C" __global__ void pivot_batch_f32(const float* __restrict__ high,
                                            const float* __restrict__ low,
                                            const float* __restrict__ close,
                                            const float* __restrict__ open,
                                            const int* __restrict__ modes,
                                            int n,
                                            int first_valid,
                                            int n_combos,
                                            float* __restrict__ out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int combo = blockIdx.y;
    if (combo >= n_combos || t >= n) return;

    const int mode = modes[combo];
    const int row_base = combo * 9; // r4,r3,r2,r1,pp,s1,s2,s3,s4
    const int base = row_base * n + t; // idx for r4

    float r4, r3, r2, r1, pp, s1, s2, s3, s4;
    if (t < first_valid) {
        r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();
    } else {
        float h = high[t];
        float l = low[t];
        float c = close[t];
        float o = open[t];
        pivot_compute_levels(mode, h, l, c, o, r4, r3, r2, r1, pp, s1, s2, s3, s4);
    }

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

extern "C" __global__ void pivot_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ open_tm,
    const int* __restrict__ first_valids, // per series
    int cols,
    int rows,
    int mode,
    float* __restrict__ out_tm // size = 9 * rows * cols (row-major)
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series column
    if (s >= cols) return;

    const int first_valid = first_valids[s];
    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s; // time-major index

        float r4, r3, r2, r1, pp, s1, s2, s3, s4;
        if (t < first_valid) {
            r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();
        } else {
            float h = high_tm[idx];
            float l = low_tm[idx];
            float c = close_tm[idx];
            float o = open_tm[idx];
            pivot_compute_levels(mode, h, l, c, o, r4, r3, r2, r1, pp, s1, s2, s3, s4);
        }

        // Write into stacked levels: level l row-major => (l*rows + t, s)
        const int plane = rows * cols;
        out_tm[(0 * rows + t) * cols + s] = r4;
        out_tm[(1 * rows + t) * cols + s] = r3;
        out_tm[(2 * rows + t) * cols + s] = r2;
        out_tm[(3 * rows + t) * cols + s] = r1;
        out_tm[(4 * rows + t) * cols + s] = pp;
        out_tm[(5 * rows + t) * cols + s] = s1;
        out_tm[(6 * rows + t) * cols + s] = s2;
        out_tm[(7 * rows + t) * cols + s] = s3;
        out_tm[(8 * rows + t) * cols + s] = s4;
        (void)plane; // silence unused warning in some builds
    }
}

