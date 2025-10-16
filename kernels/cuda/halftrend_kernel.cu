// halftrend_kernel.cu
// CUDA kernels for HalfTrend indicator
// Math category: recurrence/time-scan per-parameter using precomputed ATR/SMA(High/Low)
// and rolling window extrema (high/low). Each thread processes one parameter row
// for the batch kernel, or one series for the many-series kernel (time-major layout).

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {

// one-series × many-params (rows = combos)
// All precomputed inputs are laid out as row-major [rows, n] flattened buffers.
// Outputs are also [rows, n]. Warmup per row provided via `warms[row]`.
__global__ void halftrend_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ atr_rows,      // [rows*n]
    const float* __restrict__ highma_rows,   // [rows*n]
    const float* __restrict__ lowma_rows,    // [rows*n]
    const float* __restrict__ roll_high_rows,// [rows*n]
    const float* __restrict__ roll_low_rows, // [rows*n]
    const int*   __restrict__ warms,         // [rows]
    const float* __restrict__ chdevs,        // [rows]
    int n,                                   // series length
    int rows,                                // number of parameter combos
    float* __restrict__ out_halftrend,
    float* __restrict__ out_trend,
    float* __restrict__ out_atr_high,
    float* __restrict__ out_atr_low,
    float* __restrict__ out_buy,
    float* __restrict__ out_sell)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const int base = row * n;
    const float* atr   = atr_rows + base;
    const float* hma   = highma_rows + base;
    const float* lma   = lowma_rows + base;
    const float* rhi   = roll_high_rows + base;
    const float* rlo   = roll_low_rows + base;
    float* o_ht  = out_halftrend + base;
    float* o_tr  = out_trend + base;
    float* o_ah  = out_atr_high + base;
    float* o_al  = out_atr_low + base;
    float* o_bs  = out_buy + base;
    float* o_ss  = out_sell + base;

    const int warm = warms[row] < n ? warms[row] : n;
    const float ch_half = chdevs[row] * 0.5f;

    // Fill warmup prefix with NaN to match scalar semantics
    const float qnan = CUDART_NAN_F;
    for (int i = 0; i < warm; ++i) {
        o_ht[i] = qnan; o_tr[i] = qnan; o_ah[i] = qnan; o_al[i] = qnan; o_bs[i] = qnan; o_ss[i] = qnan;
    }
    if (warm >= n) return;

    int current_trend = 0; // 0 up, 1 down
    int next_trend = 0;
    float up = 0.0f;
    float down = 0.0f;
    float max_low_price  = (warm > 0) ? low[warm - 1]  : low[0];
    float min_high_price = (warm > 0) ? high[warm - 1] : high[0];

    for (int i = warm; i < n; ++i) {
        // reset buy/sell to NaN each step
        o_bs[i] = qnan; o_ss[i] = qnan;

        const float high_price = rhi[i];
        const float low_price  = rlo[i];
        const float prev_low   = (i > 0) ? low[i - 1]  : low[0];
        const float prev_high  = (i > 0) ? high[i - 1] : high[0];

        if (next_trend == 1) {
            if (low_price > max_low_price) max_low_price = low_price;
            if (hma[i] < max_low_price && close[i] < prev_low) {
                current_trend = 1;
                next_trend = 0;
                min_high_price = high_price;
            }
        } else {
            if (high_price < min_high_price) min_high_price = high_price;
            if (lma[i] > min_high_price && close[i] > prev_high) {
                current_trend = 0;
                next_trend = 1;
                max_low_price = low_price;
            }
        }

        const float a = atr[i];
        const float atr2 = 0.5f * a;
        const float dev = a * ch_half;

        if (current_trend == 0) {
            if (i > warm && o_tr[i - 1] != 0.0f) { // flip
                up = down;
                o_bs[i] = up - atr2;
            } else {
                if (i == warm || up == 0.0f) {
                    up = max_low_price;
                } else if (max_low_price > up) {
                    up = max_low_price;
                }
            }
            o_ht[i] = up;
            o_ah[i] = up + dev;
            o_al[i] = up - dev;
            o_tr[i] = 0.0f;
        } else {
            if (i > warm && o_tr[i - 1] != 1.0f) { // flip
                down = up;
                o_ss[i] = down + atr2;
            } else {
                if (i == warm || down == 0.0f) {
                    down = min_high_price;
                } else if (min_high_price < down) {
                    down = min_high_price;
                }
            }
            o_ht[i] = down;
            o_ah[i] = down + dev;
            o_al[i] = down - dev;
            o_tr[i] = 1.0f;
        }
    }
}

// many-series × one-param (time-major): series = cols, time = rows (n_rows)
// Inputs and outputs are time-major flattened: idx = t*cols + s
__global__ void halftrend_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ atr_tm,       // [rows*cols]
    const float* __restrict__ highma_tm,    // [rows*cols]
    const float* __restrict__ lowma_tm,     // [rows*cols]
    const float* __restrict__ roll_high_tm, // [rows*cols]
    const float* __restrict__ roll_low_tm,  // [rows*cols]
    const int*   __restrict__ warms_cols,   // [cols]
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
    int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    const float ch_half = ch_dev * 0.5f;
    const int warm = warms_cols[s] < rows ? warms_cols[s] : rows;

    const float qnan = CUDART_NAN_F;
    for (int t = 0; t < warm; ++t) {
        const int idx = t * cols + s;
        out_halftrend_tm[idx] = qnan;
        out_trend_tm[idx] = qnan;
        out_atr_high_tm[idx] = qnan;
        out_atr_low_tm[idx] = qnan;
        out_buy_tm[idx] = qnan;
        out_sell_tm[idx] = qnan;
    }
    if (warm >= rows) return;

    int current_trend = 0;
    int next_trend = 0;
    float up = 0.0f, down = 0.0f;
    auto at = [&](int t)->int { return t * cols + s; };
    float max_low_price  = (warm > 0) ? low_tm[at(warm - 1)]  : low_tm[at(0)];
    float min_high_price = (warm > 0) ? high_tm[at(warm - 1)] : high_tm[at(0)];

    for (int t = warm; t < rows; ++t) {
        const int idx = at(t);
        out_buy_tm[idx] = qnan;
        out_sell_tm[idx] = qnan;

        const float high_price = roll_high_tm[idx];
        const float low_price  = roll_low_tm[idx];
        const float prev_low   = (t > 0) ? low_tm[at(t - 1)]  : low_tm[at(0)];
        const float prev_high  = (t > 0) ? high_tm[at(t - 1)] : high_tm[at(0)];

        if (next_trend == 1) {
            if (low_price > max_low_price) max_low_price = low_price;
            if (highma_tm[idx] < max_low_price && close_tm[idx] < prev_low) {
                current_trend = 1;
                next_trend = 0;
                min_high_price = high_price;
            }
        } else {
            if (high_price < min_high_price) min_high_price = high_price;
            if (lowma_tm[idx] > min_high_price && close_tm[idx] > prev_high) {
                current_trend = 0;
                next_trend = 1;
                max_low_price = low_price;
            }
        }

        const float a = atr_tm[idx];
        const float atr2 = 0.5f * a;
        const float dev = a * ch_half;

        if (current_trend == 0) {
            if (t > warm && out_trend_tm[at(t - 1)] != 0.0f) {
                up = down;
                out_buy_tm[idx] = up - atr2;
            } else {
                if (t == warm || up == 0.0f) {
                    up = max_low_price;
                } else if (max_low_price > up) {
                    up = max_low_price;
                }
            }
            out_halftrend_tm[idx] = up;
            out_atr_high_tm[idx] = up + dev;
            out_atr_low_tm[idx] = up - dev;
            out_trend_tm[idx] = 0.0f;
        } else {
            if (t > warm && out_trend_tm[at(t - 1)] != 1.0f) {
                down = up;
                out_sell_tm[idx] = down + atr2;
            } else {
                if (t == warm || down == 0.0f) {
                    down = min_high_price;
                } else if (min_high_price < down) {
                    down = min_high_price;
                }
            }
            out_halftrend_tm[idx] = down;
            out_atr_high_tm[idx] = down + dev;
            out_atr_low_tm[idx] = down - dev;
            out_trend_tm[idx] = 1.0f;
        }
    }
}

} // extern "C"
