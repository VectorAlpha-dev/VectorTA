// CUDA kernels for Kaufman Efficiency Ratio (ER) - DS-precision, FP32-only math
// Targeted for Ada+ (e.g., RTX 4090) with nvcc 13.x
//
// - Batch + prefix path (one series, many params): consumes host-computed DS prefix.
// - Rolling-denominator batch path: uses DS rolling accumulator.
// - Many-series (time-major) path: DS rolling accumulator per series.
//
// Rationale:
// * Avoid FP64 throughput penalty on GeForce (1/64 FP32) by using double-single (two-float) math.
// * Maintain accuracy for long-window sums via error-free transforms (TwoSum).
// * Scalable, grid-stride loops, no extra build flags.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ------------------- DS helpers (double-single in two floats) -------------------
struct dsf { float hi, lo; };

// TwoSum for float (exact rounding error of a+b)
static __forceinline__ __device__
void two_sumf(float a, float b, float &s, float &e) {
    float t  = a + b;
    float bp = t - a;
    e = (a - (t - bp)) + (b - bp);
    s = t;
}

// Add scalar 'y' into a dsf accumulator
static __forceinline__ __device__
dsf dsf_add_scalar(dsf x, float y) {
    float s1, e1; two_sumf(x.hi, y, s1, e1);
    float lo  = x.lo + e1;
    float s2, e2; two_sumf(s1, lo, s2, e2);
    return dsf{ s2, e2 };
}

// Subtract dsf 'b' from dsf 'a'
static __forceinline__ __device__
dsf dsf_sub(dsf a, dsf b) {
    float s1, e1; two_sumf(a.hi, -b.hi, s1, e1);
    float lo  = a.lo - b.lo + e1;
    float s2, e2; two_sumf(s1, lo, s2, e2);
    return dsf{ s2, e2 };
}

static __forceinline__ __device__
float dsf_to_float(dsf x) { return x.hi + x.lo; }

// ------------------- Batch kernel (prefix path): FP32, DS prefix -------------------
// One series, many params. Expects host-precomputed prefix of abs diffs as float2 (hi,lo).
// prefix_ds[t] equals sum_{k=0..t-1} |x[k+1]-x[k]| in double-single representation.
//
// NOTE: signature changed vs original: prefix_absdiff is now float2* instead of double*.
extern "C" __global__ void er_batch_prefix_f32(
    const float* __restrict__ data,
    const float2* __restrict__ prefix_ds, // DS prefix (hi,lo)
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm   = first_valid + period - 1;
    const size_t row_off = (size_t)combo * (size_t)len;
    const float nan_f = nanf("");

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f;
        if (t >= warm) {
            const int start = t + 1 - period;
            // denom = prefix[t] - prefix[start] in DS, then cast to float
            const float2 pt = prefix_ds[t];
            const float2 ps = prefix_ds[start];
            dsf denom_ds = dsf_sub(dsf{pt.x, pt.y}, dsf{ps.x, ps.y});
            float denom = dsf_to_float(denom_ds);
            if (denom > 0.0f) {
                float delta = fabsf(data[t] - data[start]);
                float r = delta / denom;
                // Clamp to [0,1]
                out_val = (r > 1.0f) ? 1.0f : r;
            } else {
                out_val = 0.0f;
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// ------------------- Batch kernel (rolling denom): FP32-only DS accumulator --------
// One thread per combo, sequential over time, but numerically robust and FP64-free.
extern "C" __global__ void er_batch_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > len) return;

    const size_t row_off = (size_t)combo * (size_t)len;
    const int warm = first_valid + period - 1;
    const float nan_f = nanf("");
    if (warm >= len) {
        // write NaNs up to len to be safe
        for (int t = 0; t < len; ++t) out[row_off + t] = nan_f;
        return;
    }

    // Only write NaNs up to 'warm'
    for (int t = 0; t < warm; ++t) out[row_off + t] = nan_f;

    // Build initial DS rolling denominator over [first_valid .. warm-1]
    dsf roll{0.f, 0.f};
    for (int j = first_valid; j < warm; ++j) {
        float v1 = data[j + 1];
        float v0 = data[j];
        roll = dsf_add_scalar(roll, fabsf(v1 - v0));
    }

    int start = first_valid;
    for (int i = warm; i < len; ++i) {
        float cur   = data[i];
        float old   = data[start];
        float delta = fabsf(cur - old);

        float denom = dsf_to_float(roll);
        float er = 0.0f;
        if (denom > 0.0f) {
            float r = delta / denom;
            er = (r > 1.0f) ? 1.0f : r;
        }
        out[row_off + i] = er;

        if (i + 1 == len) break;

        // Update DS rolling denom: add new diff, subtract old diff
        float add = fabsf(data[i + 1]     - data[i]);
        float sub = fabsf(data[start + 1] - data[start]);
        roll = dsf_add_scalar(roll,  add);
        roll = dsf_add_scalar(roll, -sub);
        ++start;
    }
}

// ------------------- Many-series, one-param, time-major: FP32 DS rolling -----------
// data_tm and out_tm are time-major: index = t*cols + s
extern "C" __global__ void er_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,
    int cols,   // number of series
    int rows,   // length per series
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    const float nan_f = nanf("");

    if (period <= 0 || period > rows) {
        // fill column with NaNs
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    const int first_valid = first_valids[s];
    const int warm = first_valid + period - 1;
    if (warm >= rows) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    // NaNs up to warm
    for (int t = 0; t < warm; ++t) out_tm[t * cols + s] = nan_f;

    // Build initial DS denom for this series
    dsf roll{0.f, 0.f};
    for (int j = first_valid; j < warm; ++j) {
        float v1 = data_tm[(j + 1) * cols + s];
        float v0 = data_tm[j * cols + s];
        roll = dsf_add_scalar(roll, fabsf(v1 - v0));
    }

    int start = first_valid;
    for (int i = warm; i < rows; ++i) {
        float cur   = data_tm[i * cols + s];
        float old   = data_tm[start * cols + s];
        float delta = fabsf(cur - old);

        float denom = dsf_to_float(roll);
        float er = 0.0f;
        if (denom > 0.0f) {
            float r = delta / denom;
            er = (r > 1.0f) ? 1.0f : r;
        }
        out_tm[i * cols + s] = er;

        if (i + 1 == rows) break;
        float add = fabsf(data_tm[(i + 1)     * cols + s] - data_tm[i * cols + s]);
        float sub = fabsf(data_tm[(start + 1) * cols + s] - data_tm[start * cols + s]);
        roll = dsf_add_scalar(roll,  add);
        roll = dsf_add_scalar(roll, -sub);
        ++start;
    }
}
