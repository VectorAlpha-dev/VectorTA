// NVI kernels (FP32 + double-single accumulator)
// - Sequential order preserved: nvi += nvi * pct
// - Outputs float32, internal accumulator as two-float (hi,lo)
// - Many-series kernel expects time-major layout: [t * cols + s]

#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_NAN_F

// -------- double-single helpers (two-float) -------------------------------
struct dsfloat {
    float hi;
    float lo;
};

__device__ __forceinline__ dsfloat ds_make(float x) {
    dsfloat a; a.hi = x; a.lo = 0.0f; return a;
}

// Renormalize (fold a small t into (hi,lo))
__device__ __forceinline__ void ds_renorm(dsfloat& a, float t) {
    float s = a.hi + t;
    a.lo    = t - (s - a.hi);
    a.hi    = s;
}

// Error-free transform based addition: (hi,lo) + (hi,lo)
__device__ __forceinline__ dsfloat ds_add(dsfloat a, dsfloat b) {
    // TwoSum on hi parts
    float s  = a.hi + b.hi;
    float bb = s - a.hi;
    float err = (a.hi - (s - bb)) + (b.hi - bb);
    // Accumulate low parts + carry
    float t = a.lo + b.lo + err;
    // Renormalize
    dsfloat r;
    r.hi = s + t;
    r.lo = t - (r.hi - s);
    return r;
}

// Multiply (hi,lo) by scalar b with product error compensation (FMA)
__device__ __forceinline__ dsfloat ds_mul_scalar(dsfloat a, float b) {
    float p = a.hi * b;
    float e = fmaf(a.hi, b, -p);     // error of a.hi * b
    float t = a.lo * b + e;          // include low contribution
    dsfloat r;
    r.hi = p + t;
    r.lo = t - (r.hi - p);
    return r;
}

// Convert accumulator to float for output
__device__ __forceinline__ float ds_to_float(dsfloat a) {
    return a.hi + a.lo;
}

// ---------------- one-series kernel ---------------------------------------
extern "C" __global__ void nvi_batch_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{
    if (len <= 0) return;

    // One thread to preserve strict sequential dependency over time.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;

    // Warmup prefix -> NaN
    const float nan_f = CUDART_NAN_F;
    for (int i = 0; i < fv && i < len; ++i) out[i] = nan_f;
    if (fv >= len) return;

    // Initial value at first valid
    dsfloat nvi = ds_make(1000.0f);
    out[fv] = ds_to_float(nvi);
    if (fv + 1 >= len) return;

    float prev_close  = close[fv];
    float prev_volume = volume[fv];

    for (int i = fv + 1; i < len; ++i) {
        const float c = close[i];
        const float v = volume[i];

        if (v < prev_volume) {
            // pct = (c - prev_close) / prev_close
            const float pct = (c - prev_close) / prev_close;
            // nvi += nvi * pct (preserve CPU op order)
            dsfloat prod = ds_mul_scalar(nvi, pct);
            nvi = ds_add(nvi, prod);
        }
        out[i] = ds_to_float(nvi);
        prev_close = c;
        prev_volume = v;
    }
}

// --------------- many-series × one-param (time-major) ---------------------
extern "C" __global__ void nvi_many_series_one_param_f32(
    const float* __restrict__ close_tm,   // [t * cols + s]
    const float* __restrict__ volume_tm,  // [t * cols + s]
    int cols,
    int rows,
    const int* __restrict__ first_valids, // [cols]
    float* __restrict__ out_tm)           // [t * cols + s]
{
    if (rows <= 0 || cols <= 0) return;
    const float nan_f = CUDART_NAN_F;

    // Grid-stride over series
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s] < 0 ? 0 : first_valids[s];

        // If the entire column is invalid, fill with NaN and continue
        if (fv >= rows) {
            for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
            continue;
        }

        // Warmup [0..fv): NaNs
        for (int t = 0; t < fv; ++t) out_tm[t * cols + s] = nan_f;

        // Initial value at first valid
        dsfloat nvi = ds_make(1000.0f);
        out_tm[fv * cols + s] = ds_to_float(nvi);
        if (fv + 1 >= rows) continue;

        float prev_close  = close_tm[fv * cols + s];
        float prev_volume = volume_tm[fv * cols + s];

        for (int t = fv + 1; t < rows; ++t) {
            const float c = close_tm[t * cols + s];
            const float v = volume_tm[t * cols + s];

            if (v < prev_volume) {
                const float pct = (c - prev_close) / prev_close;
                dsfloat prod = ds_mul_scalar(nvi, pct);
                nvi = ds_add(nvi, prod);
            }
            out_tm[t * cols + s] = ds_to_float(nvi);
            prev_close  = c;
            prev_volume = v;
        }
    }
}

