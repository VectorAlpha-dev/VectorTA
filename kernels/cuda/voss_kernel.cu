// CUDA kernels for VOSS (Ehlers Voss Filter)
// Optimized for Ada+ (e.g., RTX 4090) with dual‑FP32 arithmetic.
// - Preserves scalar parity for warmup/NaN semantics.
// - One‑series × many‑params kernel remains the focus.
//
// Numerics:
// - All hot‑loop math in dual‑FP32 (“double‑single”) using FMA; outputs are f32.
// - Coefficients computed as f32 (cosf/sqrtf). No --use_fast_math required.

#include <cuda_runtime.h>
#include <math.h>

// Provide M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// ---- helpers ----------------------------------------------------------------

__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// float variant of voss_s1(): s1 = 1/g1 - sqrt(max(1/g1^2 - 1, 0))
__device__ __forceinline__ float voss_s1_f32(float g1) {
    const float inv_g1 = 1.0f / g1;
    const float t = fmaxf(inv_g1 * inv_g1 - 1.0f, 0.0f);
    const float root = sqrtf(t);
    return inv_g1 - root;
}

// ---- double‑single (dual‑FP32) arithmetic -----------------------------------
// dsfloat stores a number as hi+lo (|lo| <= ulp(hi)/2).
// Based on error‑free transforms (TwoSum/TwoProduct via FMA).

struct dsfloat { float hi, lo; };

__device__ __forceinline__ dsfloat ds_from_float(float x) { return {x, 0.0f}; }
__device__ __forceinline__ float  ds_to_float(const dsfloat &a) { return a.hi + a.lo; }

// Error‑free add (Knuth FastTwoSum + renormalize)
__device__ __forceinline__ dsfloat ds_add(dsfloat a, dsfloat b) {
    float s  = a.hi + b.hi;
    float z  = s - a.hi;
    float e  = (a.hi - (s - z)) + (b.hi - z) + a.lo + b.lo;
    float hi = s + e;
    float lo = e - (hi - s);
    return {hi, lo};
}

__device__ __forceinline__ dsfloat ds_sub(dsfloat a, dsfloat b) {
    b.hi = -b.hi; b.lo = -b.lo;
    return ds_add(a, b);
}

// Error‑free product with FMA: p = a*b; err = fma(a,b,-p)
__device__ __forceinline__ dsfloat two_prod_fma(float a, float b) {
    float p = a * b;
    float e = fmaf(a, b, -p);
    return {p, e};
}

// Multiply ds by scalar s (error‑free using FMA + renormalize)
__device__ __forceinline__ dsfloat ds_mul_scalar(dsfloat a, float s) {
    float p  = a.hi * s;
    float e  = fmaf(a.hi, s, -p) + a.lo * s;
    float hi = p + e;
    float lo = e - (hi - p);
    return {hi, lo};
}

// FMA: a*s + c (all in ds)
__device__ __forceinline__ dsfloat ds_fma_scalar(dsfloat a, float s, dsfloat c) {
    return ds_add(ds_mul_scalar(a, s), c);
}

// ---- kernels ----------------------------------------------------------------

// Batch: one price series × many parameter rows. Each row handled by thread 0 of the block (sequential over time).
extern "C" __global__ void voss_batch_f32(
    const double* __restrict__ prices,   // input still double for drop‑in compatibility
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const int* __restrict__ predicts,
    const double* __restrict__ bandwidths,
    int nrows,
    float* __restrict__ out_voss,
    float* __restrict__ out_filt)
{
    const int row = blockIdx.y;         // one block per row (configure blockDim.x=1)
    if (row >= nrows) return;

    if (threadIdx.x != 0) return;       // single thread per row by design

    const int p  = periods[row];
    const int q  = predicts[row];
    const float bw = (float)bandwidths[row];
    if (p <= 0 || q < 0) return;

    const int order     = 3 * q;
    const int min_index = max(max(p, 5), order);
    const int start     = first_valid + min_index;
    const int row_off   = row * len;

    // Fill warmup prefix with NaN for both outputs
    const int warm_end = (start < len ? start : len);
    for (int t = 0; t < warm_end; ++t) {
        out_voss[row_off + t] = f32_nan();
        out_filt[row_off + t] = f32_nan();
    }

    // Set filt[start-2], filt[start-1] = 0.0f if in range
    if (start - 2 >= 0 && start - 2 < len) out_filt[row_off + (start - 2)] = 0.0f;
    if (start - 1 >= 0 && start - 1 < len) out_filt[row_off + (start - 1)] = 0.0f;

    if (start >= len) return;

    // Coefficients (single precision math)
    const float TWO_PI = 6.2831853071795864769f;
    const float w0 = TWO_PI / (float)p;
    const float f1 = cosf(w0);
    const float g1 = cosf(bw * w0);
    const float s1 = voss_s1_f32(g1);
    const float c1 = 0.5f * (1.0f - s1);
    const float c2 = f1 * (1.0f + s1);
    const float c3 = -s1;
    const float scale = 0.5f * (3.0f + (float)order);

    // Recurrence state as dual‑FP32
    dsfloat prev_f1 = ds_from_float(0.0f);
    dsfloat prev_f2 = ds_from_float(0.0f);

    // Roll two prior price samples to avoid reloading x[i-2]
    float x_im2 = (float)prices[start - 2];
    float x_im1 = (float)prices[start - 1];

    if (order == 0) {
        for (int i = start; i < len; ++i) {
            const float xi   = (float)prices[i];
            const float diff = xi - x_im2;

            // t = c3*prev_f2 + c1*diff ;  f = c2*prev_f1 + t
            const dsfloat t = ds_fma_scalar(prev_f2, c3, ds_from_float(c1 * diff));
            const dsfloat f = ds_fma_scalar(prev_f1, c2, t);

            const float f_out = ds_to_float(f);
            out_filt[row_off + i] = f_out;
            out_voss[row_off + i] = scale * f_out;

            // shift states
            prev_f2 = prev_f1;
            prev_f1 = f;
            x_im2 = x_im1;
            x_im1 = xi;
        }
        return;
    }

    // Predictor via rolling sums (O(1) update) in dual‑FP32
    dsfloat a_sum = ds_from_float(0.0f);            // A = sum of last m v
    dsfloat d_sum = ds_from_float(0.0f);            // D = sum of k*v_{i-k}
    const float inv_m = 1.0f / (float)order;

    for (int i = start; i < len; ++i) {
        const float xi   = (float)prices[i];
        const float diff = xi - x_im2;

        // filter
        const dsfloat t = ds_fma_scalar(prev_f2, c3, ds_from_float(c1 * diff));
        const dsfloat f = ds_fma_scalar(prev_f1, c2, t);
        const float   f_out = ds_to_float(f);
        out_filt[row_off + i] = f_out;

        prev_f2 = prev_f1;
        prev_f1 = f;

        // predictor
        const float sumc = ds_to_float(d_sum) * inv_m;
        const float vi   = scale * f_out - sumc;
        out_voss[row_off + i] = vi;

        const float v_new_nz = isnan(vi) ? 0.0f : vi;

        // read v_old from our own output (matching scalar nz() policy)
        const int j_old = i - order;
        float v_old = 0.0f;
        if (j_old >= start) {
            const float vv = out_voss[row_off + j_old];
            v_old = isnan(vv) ? 0.0f : vv;
        }

        const dsfloat a_prev = a_sum;
        // A' = A - v_old + v_new
        a_sum = ds_add(ds_sub(a_prev, ds_from_float(v_old)), ds_from_float(v_new_nz));
        // D' = (D - A_prev) + m * v_new
        d_sum = ds_add(ds_sub(d_sum, a_prev), ds_from_float((float)order * v_new_nz));

        // advance price ring
        x_im2 = x_im1;
        x_im1 = xi;
    }
}

// Many‑series × one‑param (time‑major). Each series handled by one thread in blockDim.y (sequential over rows).
extern "C" __global__ void voss_many_series_one_param_time_major_f32(
    const double* __restrict__ data_tm,     // time‑major [rows x cols]
    const int*    __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    int predict,
    double bandwidth,
    float* __restrict__ out_voss_tm,
    float* __restrict__ out_filt_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;  // series index
    if (s >= cols) return;
    if (threadIdx.x != 0) return;                         // one thread per series in x

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        // No valid data: write NaNs
        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            out_voss_tm[idx] = f32_nan();
            out_filt_tm[idx] = f32_nan();
        }
        return;
    }

    const int order     = 3 * predict;
    const int min_index = max(max(period, 5), order);
    const int start     = fv + min_index;

    // Fill warmup with NaNs
    const int warm_end = (start < rows ? start : rows);
    for (int t = 0; t < warm_end; ++t) {
        const int idx = t * cols + s;
        out_voss_tm[idx] = f32_nan();
        out_filt_tm[idx] = f32_nan();
    }
    if (start - 2 >= 0 && start - 2 < rows) out_filt_tm[(start - 2) * cols + s] = 0.0f;
    if (start - 1 >= 0 && start - 1 < rows) out_filt_tm[(start - 1) * cols + s] = 0.0f;

    if (start >= rows) return;

    // Coefficients (f32)
    const float TWO_PI = 6.2831853071795864769f;
    const float w0 = TWO_PI / (float)period;
    const float f1 = cosf(w0);
    const float g1 = cosf((float)bandwidth * w0);
    const float s1 = voss_s1_f32(g1);
    const float c1 = 0.5f * (1.0f - s1);
    const float c2 = f1 * (1.0f + s1);
    const float c3 = -s1;
    const float scale = 0.5f * (3.0f + (float)order);

    dsfloat prev_f1 = ds_from_float(0.0f);
    dsfloat prev_f2 = ds_from_float(0.0f);

    // price ring to avoid loading x[i-2]
    float x_im2 = (float)data_tm[(start - 2) * cols + s];
    float x_im1 = (float)data_tm[(start - 1) * cols + s];

    if (order == 0) {
        for (int i = start; i < rows; ++i) {
            const int   idx  = i * cols + s;
            const float xi   = (float)data_tm[idx];
            const float diff = xi - x_im2;

            const dsfloat t = ds_fma_scalar(prev_f2, c3, ds_from_float(c1 * diff));
            const dsfloat f = ds_fma_scalar(prev_f1, c2, t);
            const float   f_out = ds_to_float(f);

            out_filt_tm[idx] = f_out;
            out_voss_tm[idx] = scale * f_out;

            prev_f2 = prev_f1;
            prev_f1 = f;

            x_im2 = x_im1;
            x_im1 = xi;
        }
        return;
    }

    dsfloat a_sum = ds_from_float(0.0f);
    dsfloat d_sum = ds_from_float(0.0f);
    const float inv_m = 1.0f / (float)order;

    for (int i = start; i < rows; ++i) {
        const int   idx  = i * cols + s;
        const float xi   = (float)data_tm[idx];
        const float diff = xi - x_im2;

        const dsfloat t = ds_fma_scalar(prev_f2, c3, ds_from_float(c1 * diff));
        const dsfloat f = ds_fma_scalar(prev_f1, c2, t);
        const float   f_out = ds_to_float(f);

        out_filt_tm[idx] = f_out;

        prev_f2 = prev_f1;
        prev_f1 = f;

        const float sumc = ds_to_float(d_sum) * inv_m;
        const float vi   = scale * f_out - sumc;
        out_voss_tm[idx] = vi;

        const float v_new_nz = isnan(vi) ? 0.0f : vi;

        const int j_old = i - order;
        float v_old = 0.0f;
        if (j_old >= start) {
            const float vv = out_voss_tm[j_old * cols + s];
            v_old = isnan(vv) ? 0.0f : vv;
        }

        const dsfloat a_prev = a_sum;
        a_sum = ds_add(ds_sub(a_prev, ds_from_float(v_old)), ds_from_float(v_new_nz));
        d_sum = ds_add(ds_sub(d_sum, a_prev), ds_from_float((float)order * v_new_nz));

        x_im2 = x_im1;
        x_im1 = xi;
    }
}
