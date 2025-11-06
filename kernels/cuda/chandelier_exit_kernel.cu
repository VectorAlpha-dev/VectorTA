// Chandelier Exit CUDA kernels (FP32, optimized)
// - FP64 removed; compensated FP32 for ATR seed/smoothing
// - One-pass write (no pre-init loop to NaN/0)
// - Amortized-O(1) sliding extrema via "lazy rescan" (recompute only when needed)
// - Grid-stride loops for scalability
// - NaN/warmup semantics preserved

#include <cuda_runtime.h>
#include <math_constants.h>

static __device__ __forceinline__ float f32_nan() {
    // Quiet NaN bit pattern (keeps scalar semantics)
    return __int_as_float(0x7fffffff);
}

struct KahanF32 {
    float s;
    float c;
    __device__ __forceinline__ KahanF32() : s(0.0f), c(0.0f) {}
    __device__ __forceinline__ void add(float x) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};

extern "C" __global__ void chandelier_exit_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int    len,
    const int    first_valid,
    const int*   __restrict__ periods, // length = rows
    const float* __restrict__ mults,   // length = rows
    const int    rows,                 // number of parameter combos
    const int    use_close_flag,       // 1 => use close for extremums
    float*       __restrict__ out      // length = (2*rows) * len
)
{
    const int stride = blockDim.x * gridDim.x;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += stride)
    {
        const int   period = periods[r];
        const float mult   = mults[r];

        // Quick guards: non-positive period => all NaN after warm logic
        if (period <= 0) {
            float* long_row_ptr  = out + (size_t)(2 * r)     * len;
            float* short_row_ptr = out + (size_t)(2 * r + 1) * len;
            for (int i = 0; i < len; ++i) {
                long_row_ptr[i]  = f32_nan();
                short_row_ptr[i] = f32_nan();
            }
            continue;
        }

        const int   warm = first_valid + period - 1;
        const float invP = 1.0f / (float)period;

        float* long_row_ptr  = out + (size_t)(2 * r)     * len;
        float* short_row_ptr = out + (size_t)(2 * r + 1) * len;

        // ATR state (FP32 + compensation)
        bool  prev_close_set = false;
        float prev_close     = 0.0f;
        float atr            = CUDART_NAN_F;
        KahanF32 warm_sum;
        int   warm_count = 0;

        // Trailing raw and direction
        float long_raw_prev  = CUDART_NAN_F;
        float short_raw_prev = CUDART_NAN_F;
        int   dir_prev = 1;

        // Sliding-extremum state (lazy rescan)
        int   hi_idx = -1, lo_idx = -1;
        float hi_val = f32_nan(), lo_val = f32_nan();

        for (int i = 0; i < len; ++i) {
            const float h = high[i];
            const float l = low[i];
            const float c = close[i];

            // --- ATR (Wilder) ---
            if (i >= first_valid) {
                const float hl = fabsf(h - l);
                float tr;
                if (!prev_close_set) {
                    tr = hl;            // first bar uses only HL
                    prev_close = c;
                    prev_close_set = true;
                } else {
                    const float hc = fabsf(h - prev_close);
                    const float lc = fabsf(l - prev_close);
                    tr = fmaxf(hl, fmaxf(hc, lc));
                    prev_close = c;
                }

                if (warm_count < period) {
                    if (!isnan(tr)) warm_sum.add(tr);
                    ++warm_count;
                    if (warm_count == period) {
                        atr = warm_sum.value() * invP; // seed
                    }
                } else {
                    // Wilder smoothing: atr += (tr - atr)/period
                    if (!isnan(tr) && !isnan(atr)) {
                        const float delta = (tr - atr) * invP;
                        // Kahan update for ATR in FP32
                        float corr = 0.0f; // tiny, but keep a local comp term
                        float y = delta - corr;
                        float t = atr + y;
                        corr = (t - atr) - y;
                        atr = t;
                    }
                }
            }

            // --- Sliding extrema maintenance (amortized O(1)) ---
            // New candidates (respecting use_close_flag)
            const float x_max = use_close_flag ? c : h;
            const float x_min = use_close_flag ? c : l;

            // Incorporate new sample
            if (!isnan(x_max) && (isnan(hi_val) || x_max >= hi_val)) { hi_val = x_max; hi_idx = i; }
            if (!isnan(x_min) && (isnan(lo_val) || x_min <= lo_val)) { lo_val = x_min; lo_idx = i; }

            const int start = (i - period + 1 > 0) ? (i - period + 1) : 0;

            // If current extremum is out of window, recompute on [start, i]
            if (hi_idx < start) {
                hi_val = f32_nan(); hi_idx = -1;
                for (int j = start; j <= i; ++j) {
                    const float v = use_close_flag ? close[j] : high[j];
                    if (!isnan(v) && (isnan(hi_val) || v > hi_val)) { hi_val = v; hi_idx = j; }
                }
            }
            if (lo_idx < start) {
                lo_val = f32_nan(); lo_idx = -1;
                for (int j = start; j <= i; ++j) {
                    const float v = use_close_flag ? close[j] : low[j];
                    if (!isnan(v) && (isnan(lo_val) || v < lo_val)) { lo_val = v; lo_idx = j; }
                }
            }

            // --- Warmup prefix ---
            if (i < warm) {
                long_row_ptr[i]  = f32_nan();
                short_row_ptr[i] = f32_nan();
                continue;
            }

            // If ATR/extrema invalid -> NaN
            if (isnan(atr) || isnan(hi_val) || isnan(lo_val)) {
                long_row_ptr[i]  = f32_nan();
                short_row_ptr[i] = f32_nan();
                continue;
            }

            // Base stops (use FMA to reduce rounding)
            const float ls0 = fmaf(-mult, atr, hi_val); // highest - mult*atr
            const float ss0 = fmaf( mult, atr, lo_val); // lowest  + mult*atr

            const float lsp = (i == warm || isnan(long_raw_prev))  ? ls0 : long_raw_prev;
            const float ssp = (i == warm || isnan(short_raw_prev)) ? ss0 : short_raw_prev;

            float ls = ls0, ss = ss0;
            if (i > warm) {
                const float pc = close[i - 1];
                if (pc > lsp) ls = (ls0 > lsp) ? ls0 : lsp;   // max(lsp, ls0)
                if (pc < ssp) ss = (ss0 < ssp) ? ss0 : ssp;   // min(ssp, ss0)
            }

            int d;
            if (c > ssp) d = 1;
            else if (c < lsp) d = -1;
            else d = dir_prev;

            long_raw_prev  = ls;
            short_raw_prev = ss;
            dir_prev = d;

            long_row_ptr[i]  = (d == 1)  ? ls : f32_nan();
            short_row_ptr[i] = (d == -1) ? ss : f32_nan();
        }
    }
}

extern "C" __global__ void chandelier_exit_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int    cols,     // number of series (columns)
    const int    rows,     // number of time steps (rows)
    const int    period,
    const float  mult,
    const int*   __restrict__ first_valids, // length = cols
    const int    use_close_flag,
    float*       __restrict__ out_tm        // length = (2*rows) * cols
)
{
    const int stride = blockDim.x * gridDim.x;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += stride)
    {
        const int fv = first_valids[s];
        if (period <= 0) {
            // All NaN
            float* long_mat  = out_tm + 0;
            float* short_mat = out_tm + (size_t)rows * cols;
            for (int t = 0; t < rows; ++t) {
                long_mat[t * cols + s]  = f32_nan();
                short_mat[t * cols + s] = f32_nan();
            }
            continue;
        }

        const int   warm_base = fv + period - 1;
        const float invP      = 1.0f / (float)period;

        float* long_mat  = out_tm + 0;
        float* short_mat = out_tm + (size_t)rows * cols;

        // ATR per series
        float atr = CUDART_NAN_F;
        KahanF32 warm_sum;
        int   warm_count = 0;
        float prev_close = 0.0f; bool prev_set = false;

        // Trailing state per series
        float long_raw_prev  = CUDART_NAN_F;
        float short_raw_prev = CUDART_NAN_F;
        int   dir_prev = 1;

        // Sliding extrema (lazy rescan)
        int   hi_idx = -1, lo_idx = -1;
        float hi_val = f32_nan(), lo_val = f32_nan();

        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            const float c = close_tm[idx];

            // ATR
            if (t >= fv) {
                const float hl = fabsf(h - l);
                float tr;
                if (!prev_set) { tr = hl; prev_close = c; prev_set = true; }
                else {
                    const float hc = fabsf(h - prev_close);
                    const float lc = fabsf(l - prev_close);
                    tr = fmaxf(hl, fmaxf(hc, lc));
                    prev_close = c;
                }

                if (warm_count < period) {
                    if (!isnan(tr)) warm_sum.add(tr);
                    ++warm_count;
                    if (warm_count == period) atr = warm_sum.value() * invP;
                } else {
                    if (!isnan(tr) && !isnan(atr)) {
                        const float delta = (tr - atr) * invP;
                        float corr = 0.0f;
                        float y = delta - corr;
                        float tt = atr + y;
                        corr = (tt - atr) - y;
                        atr = tt;
                    }
                }
            }

            // Sliding extrema
            const float x_max = use_close_flag ? c : h;
            const float x_min = use_close_flag ? c : l;

            if (!isnan(x_max) && (isnan(hi_val) || x_max >= hi_val)) { hi_val = x_max; hi_idx = t; }
            if (!isnan(x_min) && (isnan(lo_val) || x_min <= lo_val)) { lo_val = x_min; lo_idx = t; }

            const int start = (t - period + 1 > 0) ? (t - period + 1) : 0;
            if (hi_idx < start) {
                hi_val = f32_nan(); hi_idx = -1;
                for (int j = start; j <= t; ++j) {
                    const float v = use_close_flag ? close_tm[j * cols + s] : high_tm[j * cols + s];
                    if (!isnan(v) && (isnan(hi_val) || v > hi_val)) { hi_val = v; hi_idx = j; }
                }
            }
            if (lo_idx < start) {
                lo_val = f32_nan(); lo_idx = -1;
                for (int j = start; j <= t; ++j) {
                    const float v = use_close_flag ? close_tm[j * cols + s] : low_tm[j * cols + s];
                    if (!isnan(v) && (isnan(lo_val) || v < lo_val)) { lo_val = v; lo_idx = j; }
                }
            }

            // Warmup prefix
            if (t < warm_base) {
                long_mat[idx]  = f32_nan();
                short_mat[idx] = f32_nan();
                continue;
            }

            if (isnan(atr) || isnan(hi_val) || isnan(lo_val)) {
                long_mat[idx]  = f32_nan();
                short_mat[idx] = f32_nan();
                continue;
            }

            const float ls0 = fmaf(-mult, atr, hi_val);
            const float ss0 = fmaf( mult, atr, lo_val);

            const float lsp = (t == warm_base || isnan(long_raw_prev))  ? ls0 : long_raw_prev;
            const float ssp = (t == warm_base || isnan(short_raw_prev)) ? ss0 : short_raw_prev;

            float ls = ls0, ss = ss0;
            if (t > warm_base) {
                const float pc = close_tm[(t - 1) * cols + s];
                if (pc > lsp) ls = (ls0 > lsp) ? ls0 : lsp;
                if (pc < ssp) ss = (ss0 < ssp) ? ss0 : ssp;
            }

            int d;
            if (c > ssp) d = 1;
            else if (c < lsp) d = -1;
            else d = dir_prev;

            long_raw_prev  = ls;
            short_raw_prev = ss;
            dir_prev = d;

            long_mat[idx]  = (d == 1)  ? ls : f32_nan();
            short_mat[idx] = (d == -1) ? ss : f32_nan();
        }
    }
}

