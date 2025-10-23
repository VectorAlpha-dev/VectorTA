// CUDA kernels for Average Directional Index (ADX)
//
// Drop-in optimization focused on one-series → many-params and many-series → one-param.
// Key changes:
// - FP32 arithmetic with Kahan/Neumaier compensation during warmup; FMA-friendly recurrences.
// - Warp-level streaming for the batch kernel: lane0 loads H/L/C and broadcasts via __shfl_sync.
// - Only write the NaN prefix per row; avoid full-row prefill unless explicitly requested.

#include <cuda_runtime.h>
#include <math.h>

// ---- Helpers ---------------------------------------------------------------

__device__ __forceinline__ float qnan_f32() { return __int_as_float(0x7fc00000); }

// Kahan/Neumaier compensated sum in FP32
struct KahanF {
    float sum, c;
    __device__ __forceinline__ void reset() { sum = 0.f; c = 0.f; }
    __device__ __forceinline__ void add(float x) {
        float y = x - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
};

// Optional: global prefill of NaNs (not used by wrappers, but handy for experiments)
extern "C" __global__ void fill_nan_f32(float* out, size_t n) {
    const float nanv = qnan_f32();
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        out[idx] = nanv;
    }
}

// ---- Optimized one-series-many-params kernel -------------------------------
// Works with any blockDim.x that is a multiple of 32. Each warp processes up to 32
// parameter combos in parallel; lane 0 of each warp streams H/L/C and broadcasts via shfl.

extern "C" __global__
void adx_batch_f32(const float* __restrict__ high,
                   const float* __restrict__ low,
                   const float* __restrict__ close,
                   const int* __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) {
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;   // unique thread id
    if (tid >= n_combos) return;

    const int lane = threadIdx.x & 31;                        // lane id in warp
    const int warp_id = threadIdx.x >> 5;                     // warp index within block
    const int warp_base = warp_id * 32;                       // base slot in shared arrays

    // Load this thread's parameter
    const int p = periods[tid];
    float* row = out + (size_t)tid * series_len;

    // Rejects and minimal NaN handling
    if (p <= 0 || first_valid < 0 || first_valid + p >= series_len) {
        // mark whole row as NaN to be conservative
        const float nanv = qnan_f32();
        for (int i = 0; i < series_len; ++i) row[i] = nanv;
        return;
    }

    // Fill only the invalid prefix [0 .. first_valid + 2*p - 1]
    const int warm_end_excl = min(series_len, first_valid + 2 * p);
    const float nanv = qnan_f32();
    for (int i = 0; i < warm_end_excl; ++i) row[i] = nanv;

    // Precompute FP32 coefficients
    const float rp = 1.0f / (float)p;
    const float one_minus_rp = 1.0f - rp;
    const float pm1_over_p = ((float)p - 1.0f) * rp;

    // Stream time from t0 = first_valid
    const int t0 = first_valid;

    // Each warp's lane0 loads; shfl broadcasts inside the warp
    __shared__ float sh_h[1024];
    __shared__ float sh_l[1024];
    __shared__ float sh_c[1024];
    if (lane == 0) {
        sh_h[warp_base] = high[t0];
        sh_l[warp_base] = low[t0];
        sh_c[warp_base] = close[t0];
    }
    __syncthreads();
    float prev_h = sh_h[warp_base];
    float prev_l = sh_l[warp_base];
    float prev_c = sh_c[warp_base];

    // Per-thread parameter state
    int warm_j = 0;
    KahanF tr_sum; tr_sum.reset();
    KahanF plus_sum; plus_sum.reset();
    KahanF minus_sum; minus_sum.reset();

    float atr = 0.f, plus_s = 0.f, minus_s = 0.f;
    KahanF dx_sum; dx_sum.reset();
    int dx_count = 0;
    float adx_last = 0.f;

    for (int t = t0 + 1; t < series_len; ++t) {
        // lane0 loads next prices and broadcasts to this warp
        if (lane == 0) {
            sh_h[warp_base] = high[t];
            sh_l[warp_base] = low[t];
            sh_c[warp_base] = close[t];
        }
        __syncthreads();
        float ch = sh_h[warp_base];
        float cl = sh_l[warp_base];
        float cc = sh_c[warp_base];

        const float hl  = ch - cl;
        const float hpc = fabsf(ch - prev_c);
        const float lpc = fabsf(cl - prev_c);
        const float tr  = fmaxf(fmaxf(hl, hpc), lpc);
        const float up   = ch - prev_h;
        const float down = prev_l - cl;
        const float plus_dm  = (up > down && up > 0.f)   ? up   : 0.f;
        const float minus_dm = (down > up && down > 0.f) ? down : 0.f;

        if (warm_j < p) {
            tr_sum.add(tr);
            plus_sum.add(plus_dm);
            minus_sum.add(minus_dm);
            ++warm_j;
            if (warm_j == p) {
                atr     = tr_sum.sum;
                plus_s  = plus_sum.sum;
                minus_s = minus_sum.sum;
                const float inv_atr = (atr != 0.f) ? (100.f / atr) : 0.f;
                const float plus_di_prev  = plus_s  * inv_atr;
                const float minus_di_prev = minus_s * inv_atr;
                const float sum_di_prev   = plus_di_prev + minus_di_prev;
                const float dx0 = (sum_di_prev != 0.f)
                                  ? (fabsf(plus_di_prev - minus_di_prev) * (100.f / sum_di_prev))
                                  : 0.f;
                dx_sum.add(dx0);
                dx_count = 1;
            }
        } else {
            atr     = __fmaf_rn(atr,     one_minus_rp, tr);
            plus_s  = __fmaf_rn(plus_s,  one_minus_rp, plus_dm);
            minus_s = __fmaf_rn(minus_s, one_minus_rp, minus_dm);

            const float inv_atr = (atr != 0.f) ? (100.f / atr) : 0.f;
            const float plus_di  = plus_s  * inv_atr;
            const float minus_di = minus_s * inv_atr;
            const float denom    = plus_di + minus_di;
            const float dx       = (denom != 0.f)
                                  ? (fabsf(plus_di - minus_di) * (100.f / denom))
                                  : 0.f;

            if (dx_count < p) {
                dx_sum.add(dx);
                ++dx_count;
                if (dx_count == p) {
                    adx_last = dx_sum.sum * rp;
                    row[t] = adx_last;
                }
            } else {
                adx_last = __fmaf_rn(adx_last, pm1_over_p, dx * rp);
                row[t] = adx_last;
            }
        }

        // advance previous for next bar
        prev_h = ch; prev_l = cl; prev_c = cc;
        __syncthreads();
    }
}

// ---- Many-series, one-param, time-major kernel -----------------------------
extern "C" __global__
void adx_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {

    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += blockDim.x * gridDim.x) {
        const int fv = first_valids[s];
        auto at = [&](int t) { return t * cols + s; };

        // Prefill only the invalid prefix for this series
        const int warm_end_excl = (period > 0 && fv >= 0) ? min(rows, fv + 2 * period) : rows;
        const float nanv = qnan_f32();
        for (int t = 0; t < warm_end_excl; ++t) out_tm[at(t)] = nanv;
        if (period <= 0 || fv < 0 || fv + period >= rows) continue;

        // Warmup (j=1..period) with FP32 + Kahan
        int i0 = fv;
        float prev_h = high_tm[at(i0)];
        float prev_l = low_tm[at(i0)];
        float prev_c = close_tm[at(i0)];

        KahanF tr_sum; tr_sum.reset();
        KahanF plus_sum; plus_sum.reset();
        KahanF minus_sum; minus_sum.reset();

        for (int j = 1; j <= period; ++j) {
            const int t = i0 + j;
            const float ch = high_tm[at(t)];
            const float cl = low_tm[at(t)];
            const float hl  = ch - cl;
            const float hpc = fabsf(ch - prev_c);
            const float lpc = fabsf(cl - prev_c);
            const float tr  = fmaxf(fmaxf(hl, hpc), lpc);
            const float up   = ch - prev_h;
            const float down = prev_l - cl;
            if (up > down && up > 0.f)   plus_sum.add(up);
            if (down > up && down > 0.f) minus_sum.add(down);
            tr_sum.add(tr);
            prev_h = ch; prev_l = cl; prev_c = close_tm[at(t)];
        }

        float atr = tr_sum.sum;
        float plus_s = plus_sum.sum;
        float minus_s = minus_sum.sum;

        const float rp = 1.0f / (float)period;
        const float one_minus_rp = 1.0f - rp;
        const float pm1_over_p = ((float)period - 1.0f) * rp;

        // initial DX
        KahanF dx_sum; dx_sum.reset();
        {
            const float inv_atr = (atr != 0.f) ? (100.f / atr) : 0.f;
            const float plus_di_prev  = plus_s  * inv_atr;
            const float minus_di_prev = minus_s * inv_atr;
            const float sum_di_prev   = plus_di_prev + minus_di_prev;
            dx_sum.add((sum_di_prev != 0.f)
                ? (fabsf(plus_di_prev - minus_di_prev) * (100.f / sum_di_prev))
                : 0.f);
        }
        int dx_count = 1;
        float adx_last = 0.f;

        int t = i0 + period + 1;
        float prev_h2 = high_tm[at(i0 + period)];
        float prev_l2 = low_tm[at(i0 + period)];
        float prev_c2 = close_tm[at(i0 + period)];

        for (; t < rows; ++t) {
            const float ch = high_tm[at(t)];
            const float cl = low_tm[at(t)];
            const float hl = ch - cl;
            const float hpc = fabsf(ch - prev_c2);
            const float lpc = fabsf(cl - prev_c2);
            const float tr  = fmaxf(fmaxf(hl, hpc), lpc);
            const float up   = ch - prev_h2;
            const float down = prev_l2 - cl;
            const float plus_dm  = (up > down && up > 0.f)   ? up   : 0.f;
            const float minus_dm = (down > up && down > 0.f) ? down : 0.f;

            atr     = __fmaf_rn(atr,     one_minus_rp, tr);
            plus_s  = __fmaf_rn(plus_s,  one_minus_rp, plus_dm);
            minus_s = __fmaf_rn(minus_s, one_minus_rp, minus_dm);

            const float inv_atr2 = (atr != 0.f) ? (100.f / atr) : 0.f;
            const float plus_di  = plus_s  * inv_atr2;
            const float minus_di = minus_s * inv_atr2;
            const float denom    = plus_di + minus_di;
            const float dx       = (denom != 0.f)
                                  ? (fabsf(plus_di - minus_di) * (100.f / denom))
                                  : 0.f;

            if (dx_count < period) {
                dx_sum.add(dx);
                ++dx_count;
                if (dx_count == period) {
                    adx_last = dx_sum.sum * rp;
                    out_tm[at(t)] = adx_last;
                }
            } else {
                adx_last = __fmaf_rn(adx_last, pm1_over_p, dx * rp);
                out_tm[at(t)] = adx_last;
            }

            prev_h2 = ch; prev_l2 = cl; prev_c2 = close_tm[at(t)];
        }
    }
}

