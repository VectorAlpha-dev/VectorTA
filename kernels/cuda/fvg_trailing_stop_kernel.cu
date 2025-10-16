// CUDA kernels for FVG Trailing Stop (one-series × many-params, and many-series × one-param)
//
// Design notes:
// - Each thread scans time sequentially for its parameter row (batch) or series (many-series).
// - Uses small fixed-size per-thread buffers with conservative maxima to avoid dynamic allocation.
//   These maxima are chosen to comfortably cover typical parameter ranges and tests.
// - Smoothing over the x-series uses an O(W) windowed sum per bar (W = smoothing_length).
//   Given usual W<=64, this is acceptable and keeps the kernel simple and robust.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline bool finite3(float a, float b, float c) {
    return !isnan(a) && !isnan(b) && !isnan(c) && !isinf(a) && !isinf(b) && !isinf(c);
}

template <int MAXL>
__device__ inline void push_with_lookback(float* buf, int& cur_len, const int lookback, float v) {
    if (cur_len < lookback) {
        if (cur_len < MAXL) { buf[cur_len] = v; cur_len += 1; }
    } else if (lookback > 0) {
        // shift-left by 1 (lookback is small)
        const int L = (lookback < MAXL ? lookback : MAXL);
        #pragma unroll
        for (int k = 1; k < L; ++k) { buf[k - 1] = buf[k]; }
        buf[L - 1] = v;
        cur_len = L;
    }
}

template <int MAXL>
__device__ inline float compact_and_avg_bull(float* buf, int& len, float close_v, int& new_len_out) {
    double acc = 0.0;
    int new_len = 0;
    const int L = len < MAXL ? len : MAXL;
    for (int k = 0; k < L; ++k) {
        float v = buf[k];
        if (!(v > close_v)) { // keep when close >= v
            buf[new_len] = v; new_len += 1; acc += (double)v;
        }
    }
    len = new_len; new_len_out = new_len;
    return (new_len > 0) ? (float)(acc / (double)new_len) : CUDART_NAN_F;
}

template <int MAXL>
__device__ inline float compact_and_avg_bear(float* buf, int& len, float close_v, int& new_len_out) {
    double acc = 0.0;
    int new_len = 0;
    const int L = len < MAXL ? len : MAXL;
    for (int k = 0; k < L; ++k) {
        float v = buf[k];
        if (!(v < close_v)) { // keep when close <= v
            buf[new_len] = v; new_len += 1; acc += (double)v;
        }
    }
    len = new_len; new_len_out = new_len;
    return (new_len > 0) ? (float)(acc / (double)new_len) : CUDART_NAN_F;
}

__device__ inline float sma_last_bs_over_close(const float* close, int i, int bs) {
    if (bs <= 0) return CUDART_NAN_F;
    int start = i + 1 - bs;
    if (start < 0) return CUDART_NAN_F;
    double s = 0.0; 
    for (int j = start; j <= i; ++j) {
        float v = close[j];
        if (isnan(v) || isinf(v)) return CUDART_NAN_F;
        s += (double)v;
    }
    return (float)(s / (double)bs);
}

template <int MAXW>
__device__ inline float disp_last_w(const float* hist, int count, int w) {
    if (w <= 0 || count < w) return CUDART_NAN_F;
    double s = 0.0;
    for (int k = 0; k < w; ++k) {
        float v = hist[(count - 1) - k]; // last w values in reverse
        if (isnan(v) || isinf(v)) return CUDART_NAN_F;
        s += (double)v;
    }
    return (float)(s / (double)w);
}
}

// Conservative maxima for per-thread buffers
#define FVGTS_MAX_LOOKBACK 256
#define FVGTS_MAX_SMOOTH   256

// One-series × many-params
extern "C" __global__ void fvg_trailing_stop_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    const int*   __restrict__ lookbacks,
    const int*   __restrict__ smoothings,
    const int*   __restrict__ resets,   // 0/1 per combo
    int n_combos,
    float* __restrict__ upper_out,      // [n_combos * len]
    float* __restrict__ lower_out,      // [n_combos * len]
    float* __restrict__ upper_ts_out,   // [n_combos * len]
    float* __restrict__ lower_ts_out    // [n_combos * len]
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Find first valid OHLC once (same for all rows)
    int first_valid = len;
    for (int i = 0; i < len; ++i) {
        if (finite3(high[i], low[i], close[i])) { first_valid = i; break; }
    }
    if (first_valid >= len) {
        // Nothing to compute; still prefill outputs for any rows we touch
        for (int row = tid0; row < n_combos; row += stride) {
            float* U = upper_out    + (size_t)row * len;
            float* L = lower_out    + (size_t)row * len;
            float* UT= upper_ts_out + (size_t)row * len;
            float* LT= lower_ts_out + (size_t)row * len;
            for (int i = 0; i < len; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
        }
        return;
    }

    for (int row = tid0; row < n_combos; row += stride) {
        const int look = lookbacks[row];
        const int w    = smoothings[row];
        const bool rst = (resets[row] != 0);

        float* U  = upper_out    + (size_t)row * len;
        float* L  = lower_out    + (size_t)row * len;
        float* UT = upper_ts_out + (size_t)row * len;
        float* LT = lower_ts_out + (size_t)row * len;

        // Prefill with NaN
        for (int i = 0; i < len; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }

        if (look <= 0 || look > FVGTS_MAX_LOOKBACK || w <= 0 || w > FVGTS_MAX_SMOOTH) {
            continue;
        }

        // Small per-thread buffers
        float bull_buf[FVGTS_MAX_LOOKBACK]; int bull_len = 0;
        float bear_buf[FVGTS_MAX_LOOKBACK]; int bear_len = 0;
        float xbull_hist[FVGTS_MAX_SMOOTH]; int xbull_count = 0;
        float xbear_hist[FVGTS_MAX_SMOOTH]; int xbear_count = 0;

        int last_bull_non_na = -1;
        int last_bear_non_na = -1;
        int os = 0; // -1, 0, 1
        float ts = CUDART_NAN_F;
        float ts_prev = CUDART_NAN_F;

        for (int i = 0; i < len; ++i) {
            // FVG detection (needs i>=2)
            if (i >= 2) {
                float hi2 = high[i-2];
                float lo2 = low [i-2];
                float cm1 = close[i-1];
                float hi  = high[i];
                float lo  = low [i];
                if (finite3(hi2, lo2, cm1) && finite3(hi, lo, cm1)) {
                    if (lo > hi2 && cm1 > hi2) {
                        push_with_lookback<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, look, hi2);
                    }
                    if (hi < lo2 && cm1 < lo2) {
                        push_with_lookback<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, look, lo2);
                    }
                }
            }

            float c = close[i];
            // Mitigation and averages
            int dummy_len = 0;
            float bull_avg = compact_and_avg_bull<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, c, dummy_len);
            float bear_avg = compact_and_avg_bear<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, c, dummy_len);
            if (!isnan(bull_avg)) last_bull_non_na = i;
            if (!isnan(bear_avg)) last_bear_non_na = i;

            // Progressive SMA fallback over close
            int bull_bs = (!isnan(bull_avg)) ? 1 : ((last_bull_non_na >= 0) ? min(max(i - last_bull_non_na, 1), w) : 1);
            int bear_bs = (!isnan(bear_avg)) ? 1 : ((last_bear_non_na >= 0) ? min(max(i - last_bear_non_na, 1), w) : 1);
            float bull_sma = isnan(bull_avg) ? sma_last_bs_over_close(close, i, bull_bs) : CUDART_NAN_F;
            float bear_sma = isnan(bear_avg) ? sma_last_bs_over_close(close, i, bear_bs) : CUDART_NAN_F;
            float xbull = isnan(bull_avg) ? bull_sma : bull_avg;
            float xbear = isnan(bear_avg) ? bear_sma : bear_avg;

            // Update x-series histories and compute w-SMA of x-series
            if (xbull_count < FVGTS_MAX_SMOOTH) { xbull_hist[xbull_count] = xbull; xbull_count += 1; } else { xbull_hist[FVGTS_MAX_SMOOTH-1] = xbull; /* overflow-safe */ }
            if (xbear_count < FVGTS_MAX_SMOOTH) { xbear_hist[xbear_count] = xbear; xbear_count += 1; } else { xbear_hist[FVGTS_MAX_SMOOTH-1] = xbear; }
            float bull_disp = disp_last_w<FVGTS_MAX_SMOOTH>(xbull_hist, xbull_count, w);
            float bear_disp = disp_last_w<FVGTS_MAX_SMOOTH>(xbear_hist, xbear_count, w);

            int prev_os = os;
            if (!isnan(bear_disp) && c > bear_disp) { os = 1; }
            else if (!isnan(bull_disp) && c < bull_disp) { os = -1; }

            if (os != 0 && prev_os != 0) {
                if (os == 1 && prev_os != 1) { ts = bull_disp; }
                else if (os == -1 && prev_os != -1) { ts = bear_disp; }
                else if (os == 1) { if (!isnan(ts)) ts = fmaxf(ts, bull_disp); }
                else if (os == -1) { if (!isnan(ts)) ts = fminf(ts, bear_disp); }
            } else {
                if (os == 1 && !isnan(ts)) ts = fmaxf(ts, bull_disp);
                if (os == -1 && !isnan(ts)) ts = fminf(ts, bear_disp);
            }

            if (rst) {
                if (os == 1) {
                    if (!isnan(ts) && c < ts) { ts = CUDART_NAN_F; }
                    else if (isnan(ts) && !isnan(bear_disp) && c > bear_disp) { ts = bull_disp; }
                } else if (os == -1) {
                    if (!isnan(ts) && c > ts) { ts = CUDART_NAN_F; }
                    else if (isnan(ts) && !isnan(bull_disp) && c < bull_disp) { ts = bear_disp; }
                }
            }

            bool show = (!isnan(ts)) || (!isnan(ts_prev));
            float ts_nz = !isnan(ts) ? ts : ts_prev;
            if (os == 1 && show) {
                U[i] = CUDART_NAN_F; L[i] = bull_disp; UT[i] = CUDART_NAN_F; LT[i] = ts_nz;
            } else if (os == -1 && show) {
                U[i] = bear_disp; L[i] = CUDART_NAN_F; UT[i] = ts_nz; LT[i] = CUDART_NAN_F;
            } else {
                U[i] = L[i] = UT[i] = LT[i] = CUDART_NAN_F;
            }
            ts_prev = ts;
        }

        // Enforce warmup NaNs (first_valid + 2 + w - 1)
        int warm = first_valid + 2 + (w - 1);
        if (warm > len) warm = len;
        for (int i = 0; i < warm; ++i) { U[i]=L[i]=UT[i]=LT[i]=CUDART_NAN_F; }
    }
}

// Many-series × one-param (time-major)
extern "C" __global__ void fvg_trailing_stop_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int look,
    int w,
    int reset_on_cross,
    float* __restrict__ upper_tm_out,
    float* __restrict__ lower_tm_out,
    float* __restrict__ upper_ts_tm_out,
    float* __restrict__ lower_ts_tm_out
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; // series index (column)
    if (s >= cols) return;

    // Prefill column with NaNs
    for (int t = 0; t < rows; ++t) {
        int idx = t * cols + s;
        upper_tm_out[idx] = CUDART_NAN_F;
        lower_tm_out[idx] = CUDART_NAN_F;
        upper_ts_tm_out[idx] = CUDART_NAN_F;
        lower_ts_tm_out[idx] = CUDART_NAN_F;
    }

    if (look <= 0 || look > FVGTS_MAX_LOOKBACK || w <= 0 || w > FVGTS_MAX_SMOOTH) return;

    // Find first valid for this series
    int first_valid = rows;
    for (int t = 0; t < rows; ++t) {
        int idx = t * cols + s;
        if (finite3(high_tm[idx], low_tm[idx], close_tm[idx])) { first_valid = t; break; }
    }
    if (first_valid >= rows) return;

    float bull_buf[FVGTS_MAX_LOOKBACK]; int bull_len = 0;
    float bear_buf[FVGTS_MAX_LOOKBACK]; int bear_len = 0;
    float xbull_hist[FVGTS_MAX_SMOOTH]; int xbull_count = 0;
    float xbear_hist[FVGTS_MAX_SMOOTH]; int xbear_count = 0;
    int last_bull_non_na = -1;
    int last_bear_non_na = -1;
    int os = 0; float ts = CUDART_NAN_F; float ts_prev = CUDART_NAN_F;
    const bool rst = (reset_on_cross != 0);

    for (int t = 0; t < rows; ++t) {
        int idx = t * cols + s;
        // detection
        if (t >= 2) {
            int im2 = (t-2) * cols + s;
            int im1 = (t-1) * cols + s;
            float hi2 = high_tm[im2]; float lo2 = low_tm[im2]; float cm1 = close_tm[im1];
            float hi = high_tm[idx]; float lo = low_tm[idx];
            if (finite3(hi2, lo2, cm1) && finite3(hi, lo, cm1)) {
                if (lo > hi2 && cm1 > hi2) {
                    push_with_lookback<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, look, hi2);
                }
                if (hi < lo2 && cm1 < lo2) {
                    push_with_lookback<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, look, lo2);
                }
            }
        }

        float c = close_tm[idx];
        int dummy=0;
        float bull_avg = compact_and_avg_bull<FVGTS_MAX_LOOKBACK>(bull_buf, bull_len, c, dummy);
        float bear_avg = compact_and_avg_bear<FVGTS_MAX_LOOKBACK>(bear_buf, bear_len, c, dummy);
        if (!isnan(bull_avg)) last_bull_non_na = t;
        if (!isnan(bear_avg)) last_bear_non_na = t;

        int bull_bs = (!isnan(bull_avg)) ? 1 : ((last_bull_non_na >= 0) ? min(max(t - last_bull_non_na, 1), w) : 1);
        int bear_bs = (!isnan(bear_avg)) ? 1 : ((last_bear_non_na >= 0) ? min(max(t - last_bear_non_na, 1), w) : 1);
        float bull_sma = isnan(bull_avg) ? sma_last_bs_over_close(close_tm + s, t, bull_bs) : CUDART_NAN_F; // careful indexing below
        // sma_last_bs_over_close for time-major: we need consecutive indices; provide view via helper below

        // Implement time-major close access for SMA
        if (isnan(bull_avg)) {
            // recompute bull_sma over last bull_bs closes in time-major layout
            double ss = 0.0; bool bad=false; int start = t + 1 - bull_bs;
            if (start >= 0) {
                for (int j = start; j <= t; ++j) { float v = close_tm[j * cols + s]; if (isnan(v) || isinf(v)) { bad=true; break; } ss += (double)v; }
                bull_sma = bad ? CUDART_NAN_F : (float)(ss / (double)bull_bs);
            } else { bull_sma = CUDART_NAN_F; }
        }

        float bear_sma = CUDART_NAN_F;
        if (isnan(bear_avg)) {
            double ss = 0.0; bool bad=false; int start = t + 1 - bear_bs;
            if (start >= 0) {
                for (int j = start; j <= t; ++j) { float v = close_tm[j * cols + s]; if (isnan(v) || isinf(v)) { bad=true; break; } ss += (double)v; }
                bear_sma = bad ? CUDART_NAN_F : (float)(ss / (double)bear_bs);
            }
        }

        float xbull = isnan(bull_avg) ? bull_sma : bull_avg;
        float xbear = isnan(bear_avg) ? bear_sma : bear_avg;
        if (xbull_count < FVGTS_MAX_SMOOTH) { xbull_hist[xbull_count] = xbull; xbull_count += 1; } else { xbull_hist[FVGTS_MAX_SMOOTH-1] = xbull; }
        if (xbear_count < FVGTS_MAX_SMOOTH) { xbear_hist[xbear_count] = xbear; xbear_count += 1; } else { xbear_hist[FVGTS_MAX_SMOOTH-1] = xbear; }
        float bull_disp = disp_last_w<FVGTS_MAX_SMOOTH>(xbull_hist, xbull_count, w);
        float bear_disp = disp_last_w<FVGTS_MAX_SMOOTH>(xbear_hist, xbear_count, w);

        int prev_os = os;
        if (!isnan(bear_disp) && c > bear_disp) { os = 1; }
        else if (!isnan(bull_disp) && c < bull_disp) { os = -1; }

        if (os != 0 && prev_os != 0) {
            if (os == 1 && prev_os != 1) { ts = bull_disp; }
            else if (os == -1 && prev_os != -1) { ts = bear_disp; }
            else if (os == 1) { if (!isnan(ts)) ts = fmaxf(ts, bull_disp); }
            else if (os == -1) { if (!isnan(ts)) ts = fminf(ts, bear_disp); }
        } else {
            if (os == 1 && !isnan(ts)) ts = fmaxf(ts, bull_disp);
            if (os == -1 && !isnan(ts)) ts = fminf(ts, bear_disp);
        }

        if (rst) {
            if (os == 1) {
                if (!isnan(ts) && c < ts) { ts = CUDART_NAN_F; }
                else if (isnan(ts) && !isnan(bear_disp) && c > bear_disp) { ts = bull_disp; }
            } else if (os == -1) {
                if (!isnan(ts) && c > ts) { ts = CUDART_NAN_F; }
                else if (isnan(ts) && !isnan(bull_disp) && c < bull_disp) { ts = bear_disp; }
            }
        }

        bool show = (!isnan(ts)) || (!isnan(ts_prev));
        float ts_nz = !isnan(ts) ? ts : ts_prev;
        if (os == 1 && show) {
            upper_tm_out[idx] = CUDART_NAN_F; lower_tm_out[idx] = bull_disp; upper_ts_tm_out[idx] = CUDART_NAN_F; lower_ts_tm_out[idx] = ts_nz;
        } else if (os == -1 && show) {
            upper_tm_out[idx] = bear_disp; lower_tm_out[idx] = CUDART_NAN_F; upper_ts_tm_out[idx] = ts_nz; lower_ts_tm_out[idx] = CUDART_NAN_F;
        } else {
            // already NaN
        }
        ts_prev = ts;
    }

    // warmup NaNs per series
    int warm = first_valid + 2 + (w - 1);
    if (warm > rows) warm = rows;
    for (int t = 0; t < warm; ++t) {
        int idx = t * cols + s;
        upper_tm_out[idx] = lower_tm_out[idx] = upper_ts_tm_out[idx] = lower_ts_tm_out[idx] = CUDART_NAN_F;
    }
}

