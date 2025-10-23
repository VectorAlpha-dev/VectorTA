// EMD kernels — FP32, shared-memory rings for ub/lb, compensated sums.
// SM 8.9+ friendly, no FP64 in the hot loop.
//
// Requires: dynamic shared memory (2 * 50 * blockDim.x * sizeof(float)) and
//            setting MAX_DYNAMIC_SHARED_SIZE_BYTES >= requested size before launch.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>        // cospif/sincospif, sqrtf

#ifndef EMD_NAN
#define EMD_NAN (__int_as_float(0x7fffffff))
#endif

// ---- small helpers ----
struct KahanF {
    float sum;
    float c;
    __device__ __forceinline__ void init() { sum = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {
        float t = sum + x;
        if (fabsf(sum) >= fabsf(x)) c += (sum - t) + x;
        else                        c += (x - t) + sum;
        sum = t;
    }
    __device__ __forceinline__ void sub(float x) { add(-x); }
    __device__ __forceinline__ float value() const { return sum + c; }
};

// Clamp cosine away from exact 0 to avoid gamma singularities
__device__ __forceinline__ float clampcos(float x) {
    const float eps = 1e-6f;
    return fmaxf(fminf(x, 1.0f - eps), -1.0f + eps);
}

// ---------- One series × many params (batch) ----------
extern "C" __global__ void emd_batch_f32(
    const float* __restrict__ prices,    // length = series_len
    const int*   __restrict__ periods,   // len = n_combos
    const float* __restrict__ deltas,    // len = n_combos
    const float* __restrict__ fractions, // len = n_combos
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ ub_out,          // flattened [combo * series_len + t]
    float* __restrict__ mb_out,
    float* __restrict__ lb_out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    float* __restrict__ ub_row = ub_out + base;
    float* __restrict__ mb_row = mb_out + base;
    float* __restrict__ lb_row = lb_out + base;

    // Warmup NaN prefixes to match scalar semantics
    const int fv = first_valid;
    const int per_up_low = 50;
    const int period = periods[combo];
    if (period <= 0 || fv < 0 || fv >= series_len) return;
    const int per_mid    = 2 * period;
    const int warm_ul = min(series_len, fv + per_up_low - 1);
    const int warm_mid = min(series_len, fv + per_mid - 1);
    for (int i0 = 0; i0 < warm_ul; ++i0) {
        ub_row[i0] = EMD_NAN;
        lb_row[i0] = EMD_NAN;
    }
    // Defer middleband warmup fill until after loop since we use mb_row as scratch

    const float delta     = deltas[combo];
    const float fraction  = fractions[combo];

    // Coefficients in FP32 (cospif expects arguments in units of π)
    const float beta  = cospif(2.0f / (float)period);
    const float cos4  = clampcos(cospif(4.0f * delta / (float)period));
    const float gamma = 1.0f / cos4;
    const float alpha = gamma - sqrtf(fmaxf(gamma * gamma - 1.0f, 0.0f));
    const float half_one_minus_alpha      = 0.5f * (1.0f - alpha);
    const float beta_times_one_plus_alpha = beta  * (1.0f + alpha);

    const float   inv_up_low = 1.0f / (float)per_up_low;
    const float   inv_mid    = 1.0f / (float)per_mid;

    // Shared memory rings for ub/lb, per-thread slab of length 50
    extern __shared__ float smem[];
    float* __restrict__ ring_sp_all = smem;                                  // [blockDim.x * 50]
    float* __restrict__ ring_sv_all = smem + (size_t)blockDim.x * per_up_low; // [blockDim.x * 50]
    const int ring_base = threadIdx.x * per_up_low;

    KahanF sum_up;  sum_up.init();
    KahanF sum_low; sum_low.init();
    KahanF sum_mid; sum_mid.init();

    float bp_prev1 = 0.0f, bp_prev2 = 0.0f;
    float peak_prev = 0.0f, valley_prev = 0.0f;
    float price_prev1 = 0.0f, price_prev2 = 0.0f;

    int i = fv;
    if (i < series_len) {
        const float p0 = prices[i];
        bp_prev1 = p0; bp_prev2 = p0; peak_prev = p0; valley_prev = p0;
        price_prev1 = p0; price_prev2 = p0;
    }
    int count = 0;
    int idx50 = 0;

    for (; i < series_len; ++i) {
        const float price = prices[i];

        const float bp_curr = (count >= 2)
            ? fmaf(half_one_minus_alpha, (price - price_prev2),
                   fmaf(beta_times_one_plus_alpha, bp_prev1, (-alpha) * bp_prev2))
            : price;

        float peak_curr = peak_prev;
        float valley_curr = valley_prev;
        if (count >= 2) {
            if (bp_prev1 > bp_curr && bp_prev1 > bp_prev2) peak_curr   = bp_prev1;
            if (bp_prev1 < bp_curr && bp_prev1 < bp_prev2) valley_curr = bp_prev1;
        }

        const float sp = peak_curr   * fraction;
        const float sv = valley_curr * fraction;

        // ub/lb 50-sample rings in shared memory
        if (count + 1 > per_up_low) {
            sum_up.sub(ring_sp_all[ring_base + idx50]);
            sum_low.sub(ring_sv_all[ring_base + idx50]);
        }
        ring_sp_all[ring_base + idx50] = sp;
        ring_sv_all[ring_base + idx50] = sv;
        sum_up.add(sp);
        sum_low.add(sv);
        idx50++; if (idx50 == per_up_low) idx50 = 0;

        // middle window (2*period) uses mb_row as scratch like original
        sum_mid.add(bp_curr);
        if (count + 1 > per_mid) {
            sum_mid.sub(mb_row[i - per_mid]);
        }

        // Scratch write (raw) first for middle-ring
        mb_row[i] = bp_curr;

        // Outputs (only once windows are mature)
        if (count + 1 >= per_up_low) {
            ub_row[i] = sum_up.value()  * inv_up_low;
            lb_row[i] = sum_low.value() * inv_up_low;
        }
        if (count + 1 >= per_mid) {
            mb_row[i] = sum_mid.value() * inv_mid;
        }

        bp_prev2 = bp_prev1; bp_prev1 = bp_curr;
        price_prev2 = price_prev1; price_prev1 = price;
        peak_prev = peak_curr; valley_prev = valley_curr;
        ++count;
    }
    // Restore NaN prefix for middleband now that computation is done
    for (int i0 = 0; i0 < warm_mid; ++i0) {
        mb_row[i0] = EMD_NAN;
    }
}

// ---------- Many series × one param (time-major) ----------
extern "C" __global__ void emd_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm, // time-major: [t * cols + s]
    int cols,                            // number of series
    int rows,                            // series_len (time)
    int period,
    float delta,
    float fraction,
    const int* __restrict__ first_valids, // len = cols
    float* __restrict__ ub_tm,
    float* __restrict__ mb_tm,
    float* __restrict__ lb_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    float* __restrict__ ub_col = ub_tm + series;
    float* __restrict__ mb_col = mb_tm + series;
    float* __restrict__ lb_col = lb_tm + series;

    const int fv = first_valids[series];
    if (period <= 0 || fv < 0 || fv >= rows) return;
    const int per_up_low = 50;
    const int per_mid = 2 * period;
    const int warm_ul = min(rows, fv + per_up_low - 1);
    const int warm_mid = min(rows, fv + per_mid - 1);
    for (int t = 0; t < warm_ul; ++t) {
        ub_col[(size_t)t * cols] = EMD_NAN;
        lb_col[(size_t)t * cols] = EMD_NAN;
    }
    // Defer middleband prefix NaN fill until after loop

    // Coefficients in FP32
    const float beta  = cospif(2.0f / (float)period);
    const float cos4  = clampcos(cospif(4.0f * delta / (float)period));
    const float gamma = 1.0f / cos4;
    const float alpha = gamma - sqrtf(fmaxf(gamma * gamma - 1.0f, 0.0f));
    const float half_one_minus_alpha      = 0.5f * (1.0f - alpha);
    const float beta_times_one_plus_alpha = beta  * (1.0f + alpha);

    const float   inv_up_low = 1.0f / (float)per_up_low;
    const float   inv_mid    = 1.0f / (float)per_mid;

    extern __shared__ float smem[];
    float* __restrict__ ring_sp_all = smem;
    float* __restrict__ ring_sv_all = smem + (size_t)blockDim.x * per_up_low;
    const int ring_base = threadIdx.x * per_up_low;

    KahanF sum_up;  sum_up.init();
    KahanF sum_low; sum_low.init();
    KahanF sum_mid; sum_mid.init();

    float bp_prev1 = 0.0f, bp_prev2 = 0.0f;
    float peak_prev = 0.0f, valley_prev = 0.0f;
    float price_prev1 = 0.0f, price_prev2 = 0.0f;

    int t = fv;
    if (t < rows) {
        const float p0 = prices_tm[(size_t)t * cols + series];
        bp_prev1 = p0; bp_prev2 = p0; peak_prev = p0; valley_prev = p0;
        price_prev1 = p0; price_prev2 = p0;
    }

    int idx_ul = 0, idx_mid = 0;
    int count = 0;
    int idx50 = 0;
    for (; t < rows; ++t) {
        const float price = prices_tm[(size_t)t * cols + series];

        const float bp_curr = (count >= 2)
            ? fmaf(half_one_minus_alpha, (price - price_prev2),
                   fmaf(beta_times_one_plus_alpha, bp_prev1, (-alpha) * bp_prev2))
            : price;

        float peak_curr = peak_prev;
        float valley_curr = valley_prev;
        if (count >= 2) {
            if (bp_prev1 > bp_curr && bp_prev1 > bp_prev2) peak_curr   = bp_prev1;
            if (bp_prev1 < bp_curr && bp_prev1 < bp_prev2) valley_curr = bp_prev1;
        }

        const float sp = peak_curr   * fraction;
        const float sv = valley_curr * fraction;

        if (count + 1 > per_up_low) {
            sum_up.sub(ring_sp_all[ring_base + idx50]);
            sum_low.sub(ring_sv_all[ring_base + idx50]);
        }
        ring_sp_all[ring_base + idx50] = sp;
        ring_sv_all[ring_base + idx50] = sv;
        sum_up.add(sp);
        sum_low.add(sv);
        idx50++; if (idx50 == per_up_low) idx50 = 0;

        sum_mid.add(bp_curr);
        if (count + 1 > per_mid) {
            sum_mid.sub(mb_col[(size_t)(t - per_mid) * cols]);
        }

        // scratch write for middle ring
        mb_col[(size_t)t * cols] = bp_curr;

        if (count + 1 >= per_up_low) {
            ub_col[(size_t)t * cols] = sum_up.value()  * inv_up_low;
            lb_col[(size_t)t * cols] = sum_low.value() * inv_up_low;
        }
        if (count + 1 >= per_mid) {
            mb_col[(size_t)t * cols] = sum_mid.value() * inv_mid;
        }

        bp_prev2 = bp_prev1; bp_prev1 = bp_curr;
        price_prev2 = price_prev1; price_prev1 = price;
        peak_prev = peak_curr; valley_prev = valley_curr;
        ++count;
    }
    // Restore NaN prefix for middleband
    for (int t0 = 0; t0 < warm_mid; ++t0) {
        mb_col[(size_t)t0 * cols] = EMD_NAN;
    }
}
