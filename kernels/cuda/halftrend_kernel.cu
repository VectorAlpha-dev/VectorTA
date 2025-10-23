// halftrend_kernel.cu
// Optimized CUDA kernels for the HalfTrend indicator
// - Batch path: keep existing row-major ABI for tests, but optimize internals
// - Many-series path (time-major): optimized internals
// Additionally provide a time-major batch variant for future wrapper use.

#include <cuda_runtime.h>
#include <math_constants.h>
#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif

// Read-only load helper: safe on cc>=3.5; falls back otherwise
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  #define LDG(ptr) __ldg(ptr)
#else
  #define LDG(ptr) (*(ptr))
#endif

// Quiet NaN
__device__ __forceinline__ float qnan_f() { return __int_as_float(0x7fc00000); }

// Indexer for time-major [n, rows]
#define AT_TM(t, rows, row) ((t) * (rows) + (row))

// Tuned threads per block and launch bounds hint
#define HT_THREADS_PER_BLOCK 256

extern "C" {

// ----------------------------------------------------------------------------
// One price series × many params (ROW-MAJOR ABI kept for compatibility)
// Each thread processes one param row. Precompute buffers are [rows, n].
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(HT_THREADS_PER_BLOCK, 2)
void halftrend_batch_f32(
    const float* __restrict__ high,          // [n]
    const float* __restrict__ low,           // [n]
    const float* __restrict__ close,         // [n]
    const float* __restrict__ atr_rows,      // [rows*n]
    const float* __restrict__ highma_rows,   // [rows*n]
    const float* __restrict__ lowma_rows,    // [rows*n]
    const float* __restrict__ roll_high_rows,// [rows*n]
    const float* __restrict__ roll_low_rows, // [rows*n]
    const int*   __restrict__ warms,         // [rows]
    const float* __restrict__ chdevs,        // [rows]
    int n,
    int rows,
    float* __restrict__ out_halftrend,       // [rows*n]
    float* __restrict__ out_trend,           // [rows*n]
    float* __restrict__ out_atr_high,        // [rows*n]
    float* __restrict__ out_atr_low,         // [rows*n]
    float* __restrict__ out_buy,             // [rows*n]
    float* __restrict__ out_sell)            // [rows*n]
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = tid; row < rows; row += stride) {
        const int base = row * n;
        const float* __restrict__ atr = atr_rows + base;
        const float* __restrict__ hma = highma_rows + base;
        const float* __restrict__ lma = lowma_rows + base;
        const float* __restrict__ rhi = roll_high_rows + base;
        const float* __restrict__ rlo = roll_low_rows + base;
        float* __restrict__ o_ht = out_halftrend + base;
        float* __restrict__ o_tr = out_trend + base;
        float* __restrict__ o_ah = out_atr_high + base;
        float* __restrict__ o_al = out_atr_low + base;
        float* __restrict__ o_bs = out_buy + base;
        float* __restrict__ o_ss = out_sell + base;

        const int warm_in = LDG(warms + row);
        const int warm    = (warm_in < n ? warm_in : n);
        const float ch_half = LDG(chdevs + row) * 0.5f;
        const float qnan = qnan_f();

        // Fill warmup prefix with NaN
        #pragma unroll 8
        for (int i = 0; i < warm; ++i) {
            o_ht[i] = qnan; o_tr[i] = qnan; o_ah[i] = qnan; o_al[i] = qnan; o_bs[i] = qnan; o_ss[i] = qnan;
        }
        if (warm >= n) continue;

        int   current_trend = 0; // 0 up, 1 down
        int   next_trend    = 0;
        int   prev_trend    = 0;
        float up   = 0.0f;
        float down = 0.0f;

        float prev_low  = (warm > 0) ? LDG(low  + (warm - 1))  : LDG(low  + 0);
        float prev_high = (warm > 0) ? LDG(high + (warm - 1)) : LDG(high + 0);
        float max_low_price  = prev_low;
        float min_high_price = prev_high;

        for (int i = warm; i < n; ++i) {
            // reset buy/sell to NaN each step
            o_bs[i] = qnan; o_ss[i] = qnan;

            const float high_price = LDG(rhi + i);
            const float low_price  = LDG(rlo + i);

            if (next_trend == 1) {
                max_low_price = fmaxf(max_low_price, low_price);
                if (LDG(hma + i) < max_low_price && LDG(close + i) < prev_low) {
                    current_trend  = 1;
                    next_trend     = 0;
                    min_high_price = high_price;
                }
            } else {
                min_high_price = fminf(min_high_price, high_price);
                if (LDG(lma + i) > min_high_price && LDG(close + i) > prev_high) {
                    current_trend  = 0;
                    next_trend     = 1;
                    max_low_price  = low_price;
                }
            }

            const float a    = LDG(atr + i);
            const float atr2 = 0.5f * a;
            const float dev  = a * ch_half;

            const bool flipped = (i > warm) && (prev_trend != current_trend);

            if (current_trend == 0) {
                if (flipped) {        // flip to up
                    up = down;
                    o_bs[i] = up - atr2;
                } else {
                    if (i == warm || up == 0.0f) up = max_low_price;
                    else if (max_low_price > up) up = max_low_price;
                }
                o_ht[i] = up;
                o_ah[i] = up + dev;
                o_al[i] = up - dev;
                o_tr[i] = 0.0f;
            } else {
                if (flipped) {        // flip to down
                    down = up;
                    o_ss[i] = down + atr2;
                } else {
                    if (i == warm || down == 0.0f) down = min_high_price;
                    else if (min_high_price < down) down = min_high_price;
                }
                o_ht[i] = down;
                o_ah[i] = down + dev;
                o_al[i] = down - dev;
                o_tr[i] = 1.0f;
            }

            prev_low   = LDG(low  + i);
            prev_high  = LDG(high + i);
            prev_trend = current_trend;
        }
    }
}

// ----------------------------------------------------------------------------
// One price series × many params (TIME-MAJOR variant for future use)
// Function name differs to avoid breaking existing wrappers/tests.
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(HT_THREADS_PER_BLOCK, 2)
void halftrend_batch_time_major_f32( // expects TIME-MAJOR [n, rows]
    const float* __restrict__ high,          // [n]
    const float* __restrict__ low,           // [n]
    const float* __restrict__ close,         // [n]
    const float* __restrict__ atr_tm,        // [n*rows]
    const float* __restrict__ highma_tm,     // [n*rows]
    const float* __restrict__ lowma_tm,      // [n*rows]
    const float* __restrict__ roll_high_tm,  // [n*rows]
    const float* __restrict__ roll_low_tm,   // [n*rows]
    const int*   __restrict__ warms,         // [rows]
    const float* __restrict__ chdevs,        // [rows]
    int n,
    int rows,
    float* __restrict__ out_halftrend_tm,    // [n*rows]
    float* __restrict__ out_trend_tm,        // [n*rows]
    float* __restrict__ out_atr_high_tm,     // [n*rows]
    float* __restrict__ out_atr_low_tm,      // [n*rows]
    float* __restrict__ out_buy_tm,          // [n*rows]
    float* __restrict__ out_sell_tm)         // [n*rows]
{
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride= blockDim.x * gridDim.x;

    for (int row = tid; row < rows; row += stride) {
        const float ch_half = LDG(chdevs + row) * 0.5f;
        const int   warm_in = LDG(warms  + row);
        const int   warm    = (warm_in < n ? warm_in : n);
        const float qnan = qnan_f();

        #pragma unroll 8
        for (int t = 0; t < warm; ++t) {
            const int idx = AT_TM(t, rows, row);
            out_halftrend_tm[idx] = qnan;
            out_trend_tm[idx]     = qnan;
            out_atr_high_tm[idx]  = qnan;
            out_atr_low_tm[idx]   = qnan;
            out_buy_tm[idx]       = qnan;
            out_sell_tm[idx]      = qnan;
        }
        if (warm >= n) continue;

        int   current_trend = 0;   // 0 up, 1 down
        int   next_trend    = 0;
        int   prev_trend    = 0;   // track previous written trend
        float up   = 0.0f;
        float down = 0.0f;

        float prev_low  = (warm > 0) ? LDG(low  + (warm - 1)) : LDG(low  + 0);
        float prev_high = (warm > 0) ? LDG(high + (warm - 1)) : LDG(high + 0);
        float max_low_price  = prev_low;
        float min_high_price = prev_high;

        for (int t = warm; t < n; ++t) {
            const int idx = AT_TM(t, rows, row);
            out_buy_tm[idx]  = qnan;
            out_sell_tm[idx] = qnan;

            const float high_price = LDG(roll_high_tm + idx);
            const float low_price  = LDG(roll_low_tm  + idx);

            if (next_trend == 1) {
                max_low_price = fmaxf(max_low_price, low_price);
                if (LDG(highma_tm + idx) < max_low_price && LDG(close + t) < prev_low) {
                    current_trend  = 1;
                    next_trend     = 0;
                    min_high_price = high_price;
                }
            } else {
                min_high_price = fminf(min_high_price, high_price);
                if (LDG(lowma_tm + idx) > min_high_price && LDG(close + t) > prev_high) {
                    current_trend  = 0;
                    next_trend     = 1;
                    max_low_price  = low_price;
                }
            }

            const float a    = LDG(atr_tm + idx);
            const float atr2 = 0.5f * a;
            const float dev  = a * ch_half;

            const bool flipped = (t > warm) && (prev_trend != current_trend);

            if (current_trend == 0) {
                if (flipped) {        // flip to up
                    up = down;
                    out_buy_tm[idx] = up - atr2;
                } else {
                    if (t == warm || up == 0.0f) up = max_low_price;
                    else if (max_low_price > up) up = max_low_price;
                }
                out_halftrend_tm[idx] = up;
                out_atr_high_tm[idx]  = up + dev;
                out_atr_low_tm[idx]   = up - dev;
                out_trend_tm[idx]     = 0.0f;
            } else {
                if (flipped) {        // flip to down
                    down = up;
                    out_sell_tm[idx] = down + atr2;
                } else {
                    if (t == warm || down == 0.0f) down = min_high_price;
                    else if (min_high_price < down) down = min_high_price;
                }
                out_halftrend_tm[idx] = down;
                out_atr_high_tm[idx]  = down + dev;
                out_atr_low_tm[idx]   = down - dev;
                out_trend_tm[idx]     = 1.0f;
            }

            prev_low  = LDG(low  + t);
            prev_high = LDG(high + t);
            prev_trend = current_trend;
        }
    }
}

// ----------------------------------------------------------------------------
// Many series × one param (time-major), optimized internals.
// Layout unchanged: idx = t*cols + s
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(HT_THREADS_PER_BLOCK, 2)
void halftrend_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ atr_tm,        // [rows*cols]
    const float* __restrict__ highma_tm,     // [rows*cols]
    const float* __restrict__ lowma_tm,      // [rows*cols]
    const float* __restrict__ roll_high_tm,  // [rows*cols]
    const float* __restrict__ roll_low_tm,   // [rows*cols]
    const int*   __restrict__ warms_cols,    // [cols]
    float ch_dev,
    int cols,
    int rows,
    float* __restrict__ out_halftrend_tm,
    float* __restrict__ out_trend_tm,
    float* __restrict__ out_atr_high_tm,
    float* __restrict__ out_atr_low_tm,
    float* __restrict__ out_buy_tm,
    float* __restrict__ out_sell_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series
    if (s >= cols) return;

    const float ch_half = ch_dev * 0.5f;
    const int warm_in = LDG(warms_cols + s);
    const int warm    = (warm_in < rows ? warm_in : rows);

    const float qnan = qnan_f();
    #pragma unroll 8
    for (int t = 0; t < warm; ++t) {
        const int idx = t * cols + s;
        out_halftrend_tm[idx] = qnan;
        out_trend_tm[idx]     = qnan;
        out_atr_high_tm[idx]  = qnan;
        out_atr_low_tm[idx]   = qnan;
        out_buy_tm[idx]       = qnan;
        out_sell_tm[idx]      = qnan;
    }
    if (warm >= rows) return;

    int   current_trend = 0;
    int   next_trend    = 0;
    int   prev_trend    = 0;
    float up   = 0.0f;
    float down = 0.0f;

    float prev_low  = (warm > 0) ? LDG(low_tm  + ((warm - 1) * cols + s)) : LDG(low_tm  + (0 * cols + s));
    float prev_high = (warm > 0) ? LDG(high_tm + ((warm - 1) * cols + s)) : LDG(high_tm + (0 * cols + s));
    float max_low_price  = prev_low;
    float min_high_price = prev_high;

    for (int t = warm; t < rows; ++t) {
        const int idx = t * cols + s;
        out_buy_tm[idx]  = qnan;
        out_sell_tm[idx] = qnan;

        const float high_price = LDG(roll_high_tm + idx);
        const float low_price  = LDG(roll_low_tm  + idx);

        if (next_trend == 1) {
            max_low_price = fmaxf(max_low_price, low_price);
            if (LDG(highma_tm + idx) < max_low_price && LDG(close_tm + idx) < prev_low) {
                current_trend  = 1;
                next_trend     = 0;
                min_high_price = high_price;
            }
        } else {
            min_high_price = fminf(min_high_price, high_price);
            if (LDG(lowma_tm + idx) > min_high_price && LDG(close_tm + idx) > prev_high) {
                current_trend  = 0;
                next_trend     = 1;
                max_low_price  = low_price;
            }
        }

        const float a    = LDG(atr_tm + idx);
        const float atr2 = 0.5f * a;
        const float dev  = a * ch_half;

        const bool flipped = (t > warm) && (prev_trend != current_trend);

        if (current_trend == 0) {
            if (flipped) {
                up = down;
                out_buy_tm[idx] = up - atr2;
            } else {
                if (t == warm || up == 0.0f) up = max_low_price;
                else if (max_low_price > up) up = max_low_price;
            }
            out_halftrend_tm[idx] = up;
            out_atr_high_tm[idx]  = up + dev;
            out_atr_low_tm[idx]   = up - dev;
            out_trend_tm[idx]     = 0.0f;
        } else {
            if (flipped) {
                down = up;
                out_sell_tm[idx] = down + atr2;
            } else {
                if (t == warm || down == 0.0f) down = min_high_price;
                else if (min_high_price < down) down = min_high_price;
            }
            out_halftrend_tm[idx] = down;
            out_atr_high_tm[idx]  = down + dev;
            out_atr_low_tm[idx]   = down - dev;
            out_trend_tm[idx]     = 1.0f;
        }

        prev_low  = LDG(low_tm  + (t * cols + s));
        prev_high = LDG(high_tm + (t * cols + s));
        prev_trend = current_trend;
    }
}

} // extern "C"

#undef AT_TM
#undef HT_THREADS_PER_BLOCK
#undef LDG
