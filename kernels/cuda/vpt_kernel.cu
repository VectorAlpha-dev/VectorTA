// CUDA kernels for Volume Price Trend (VPT)
//
// Semantics mirror src/indicators/vpt.rs exactly:
// - Warmup: indices 0..=first_valid are NaN (first_valid >= 1).
// - Seed: prev = v[first_valid] * ((p[first_valid] - p[first_valid-1]) / p[first_valid-1]).
// - Thereafter t = first_valid + 1..len-1:
//       cur = v[t] * ((p[t] - p[t-1]) / p[t-1])
//       out[t] = cur + prev; prev = out[t];
// - NaN handling: if any term involved in the increment is non‑finite, write NaN.

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h> // CUDART_NAN_F

// Small helper: FP32 Kahan compensated add.
// After kahan_add(x, sum, c), 'sum' ~= previous sum + x with reduced error.
static __device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;   // compensation for next iteration
    sum = t;
}

// -------------------------------------------------------------------------------------
// Single series: one thread scans sequentially.
// -------------------------------------------------------------------------------------
extern "C" __global__ void vpt_batch_f32(
    const float* __restrict__ price,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{
    // Single-thread sequential scan (dependency across time).
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0) return;

    const float nan_f = CUDART_NAN_F;

    // Guard against invalid first_valid to avoid p[-1] even if host misconfigures.
    if (first_valid < 0) first_valid = 0;

    // Warmup prefix: NaN for indices 0..=first_valid (clamped into array)
    const int warm_end = (first_valid < len) ? first_valid : (len - 1);
    for (int i = 0; i <= warm_end; ++i) out[i] = nan_f;

    // Nothing more to compute?
    if (first_valid + 1 >= len) return;

    // If first_valid < 1, we cannot form p[fv-1]; semantics say fv>=1, but be defensive.
    if (first_valid < 1) {
        for (int t = first_valid + 1; t < len; ++t) out[t] = nan_f;
        return;
    }

    // Seed at fv using price[fv-1], price[fv], volume[fv]
    float p0 = price[first_valid - 1];
    float p1 = price[first_valid];
    float v1 = volume[first_valid];

    // If seed terms are non-finite or p0==0, all subsequent outputs are NaN (sticky).
    bool ok = isfinite(p0) && isfinite(p1) && isfinite(v1) && (p0 != 0.0f);
    if (!ok) {
        for (int t = first_valid + 1; t < len; ++t) out[t] = nan_f;
        return;
    }

    float prev_p = p1;

    // Compensated accumulator initialized with the seed increment.
    float sum = v1 * ((p1 - p0) / p0);
    float c = 0.0f;

    // Scan t = fv+1 .. len-1
    for (int t = first_valid + 1; t < len; ++t) {
        float pt = price[t];
        float vt = volume[t];

        bool good = isfinite(prev_p) && isfinite(pt) && isfinite(vt) && (prev_p != 0.0f);
        if (!good) {
            // sticky-NaN: once bad, everything after is NaN
            for (int j = t; j < len; ++j) out[j] = nan_f;
            return;
        }

        float cur = vt * ((pt - prev_p) / prev_p);
        kahan_add(cur, sum, c);
        out[t] = sum;

        prev_p = pt;
    }
}

// -------------------------------------------------------------------------------------
// Many-series × one-param (time-major layout): one thread per series.
// Iterate t in lockstep across threads to maximize coalescing.
// -------------------------------------------------------------------------------------
extern "C" __global__ void vpt_many_series_one_param_f32(
    const float* __restrict__ price_tm,   // [rows][cols], time-major
    const float* __restrict__ volume_tm,  // [rows][cols], time-major
    int cols,
    int rows,
    const int* __restrict__ first_valids, // per series, i >= 1
    float* __restrict__ out_tm)           // [rows][cols], time-major
{
    // Grid-stride over series (columns).
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const float nan_f = CUDART_NAN_F;

        int fv = first_valids[s];
        if (fv < 0) fv = 0; // defensive clamp

        // Per-series running state
        float sum = 0.0f;
        float c = 0.0f;
        float prev_p = nan_f;           // undefined until we hit fv
        bool sticky_nan = false;        // once true, all later outputs are NaN

        // Iterate time-major in lockstep for coalesced R/W across the warp.
        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            const float pt = price_tm[idx];
            const float vt = volume_tm[idx];

            if (t <= fv) {
                // Warmup region (0..=fv) is always NaN by semantics.
                out_tm[idx] = nan_f;

                // Initialize seed at exactly t == fv, but do not write an output yet.
                if (t == fv) {
                    if (fv < 1) { // cannot form p[fv-1]
                        sticky_nan = true;
                    } else {
                        const float p0 = price_tm[(t - 1) * cols + s];
                        const float v1 = vt;
                        const bool ok = isfinite(p0) && isfinite(pt) && isfinite(v1) && (p0 != 0.0f);
                        if (ok) {
                            sum = v1 * ((pt - p0) / p0); // seed with cur at fv
                            c = 0.0f;
                            prev_p = pt;
                        } else {
                            sticky_nan = true;
                        }
                    }
                }
                continue; // no output beyond NaN in warmup
            }

            // From here: t >= fv+1
            if (sticky_nan) {
                out_tm[idx] = nan_f;
                continue;
            }

            const bool good = isfinite(prev_p) && isfinite(pt) && isfinite(vt) && (prev_p != 0.0f);
            if (!good) {
                sticky_nan = true;
                out_tm[idx] = nan_f;
                continue;
            }

            const float cur = vt * ((pt - prev_p) / prev_p);
            kahan_add(cur, sum, c);
            out_tm[idx] = sum;
            prev_p = pt;
        }
    }
}

