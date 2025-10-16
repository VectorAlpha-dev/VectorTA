// CUDA kernels for Variable Length Moving Average (VLMA)
//
// Supported path in device kernels:
//   - Reference MA: SMA
//   - Deviation   : Standard deviation (population) over fixed window = max_period
//
// Behavior parity with scalar VLMA:
//   - Warmup end = first_valid + max_period - 1
//   - At index first_valid, write the initial data value
//   - For indices < warmup (except first_valid), write NaN
//   - After warmup, if input is NaN: write NaN and keep state unchanged
//   - Period adapts by ±1 within [min_period, max_period] using bands
//       [m-1.75*std, m-0.25*std] U [m+0.25*std, m+1.75*std]
//   - Smoothing is EMA-like with alpha = 2/(p+1)
//
// Batch kernel uses host-precomputed prefix sums (double) and prefix NaN counts
// to compute mean/std in O(1) per time. Many-series kernel uses O(1) rolling
// sums per series to avoid per-series prefix buffers.

#include <cuda_runtime.h>
#include <math.h>

// Helpers
__device__ __forceinline__ float fmaf_safe(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

// ========================
// One-series × many-params
// ========================
// data:           [len]
// prefix_sum:     [len+1] (double)
// prefix_sum_sq:  [len+1] (double)
// prefix_nan:     [len+1] (int, count of NaNs)
// min_periods:    [n_combos]
// max_periods:    [n_combos]
// out:            [n_combos * len] row-major
extern "C" __global__ void vlma_batch_sma_std_prefix_f32(
    const float*  __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    const int*    __restrict__ min_periods,
    const int*    __restrict__ max_periods,
    int len,
    int first_valid,
    int n_combos,
    float* __restrict__ out
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || len <= 0) return;

    const int min_p = max(1, min_periods[combo]);
    const int max_p = max(min_p, max_periods[combo]);
    if (first_valid < 0 || first_valid >= len) return;

    const int base = combo * len;

    // Initialize prefix: indices < first_valid to NaN, index==first_valid to x0
    for (int i = threadIdx.x; i < first_valid; i += blockDim.x) {
        out[base + i] = NAN;
    }

    if (threadIdx.x != 0) return; // sequential per-param path

    const float x0 = data[first_valid];
    out[base + first_valid] = x0;

    const int warm_end = min(len, first_valid + max_p - 1);
    int last_p = max_p;
    float last_val = x0;

    // Advance warmup without adaptation; write NaN for prefix (already set above)
    for (int i = first_valid + 1; i < warm_end; ++i) {
        const float x = data[i];
        if (isfinite(x)) {
            const float sc = 2.0f / (float)(last_p + 1);
            last_val = fmaf_safe(x - last_val, sc, last_val);
        }
        out[base + i] = NAN; // keep warmup NaNs except first_valid
    }

    if (warm_end >= len) return;

    // Steady-state loop
    for (int i = warm_end; i < len; ++i) {
        const float x = data[i];
        if (!isfinite(x)) {
            out[base + i] = NAN;
            continue;
        }

        // Window [i - max_p + 1, i]
        const int t1 = i + 1;
        const int t0 = max(0, t1 - max_p);
        const int nan_cnt = prefix_nan[t1] - prefix_nan[t0];

        float sc = 2.0f / (float)(last_p + 1); // default keep same period
        if (nan_cnt == 0) {
            const double sum  = prefix_sum[t1]    - prefix_sum[t0];
            const double sum2 = prefix_sum_sq[t1] - prefix_sum_sq[t0];
            const double inv  = 1.0 / (double)max_p;
            const double m    = sum * inv;
            double var        = (sum2 * inv) - m * m;
            if (var < 0.0) var = 0.0;
            const double dv   = sqrt(var);

            // Adapt period using bands around mean
            const double d175 = dv * 1.75;
            const double d025 = dv * 0.25;
            const double a = m - d175;
            const double b = m - d025;
            const double c = m + d025;
            const double d = m + d175;

            const int inc_fast = (x < a) || (x > d);
            const int inc_slow = (x >= b) && (x <= c);
            const int delta = inc_slow - inc_fast; // -1,0,+1
            int p_next = last_p + delta;
            if (p_next < min_p) p_next = min_p;
            if (p_next > max_p) p_next = max_p;
            sc = 2.0f / (float)(p_next + 1);
            last_p = p_next;
        }

        last_val = fmaf_safe(x - last_val, sc, last_val);
        out[base + i] = last_val;
    }
}

// ======================================
// Many-series × one-param (time-major IO)
// ======================================
// prices_tm: [rows * cols], time-major: prices_tm[t * cols + s]
// first_valids: [cols], first non-NaN index per series (time index)
// params: min_period, max_period (SMA reference + stddev only)
// out_tm: same layout as input
extern "C" __global__ void vlma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int min_period,
    int max_period,
    int cols,           // number of series
    int rows,           // series length (time)
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x; // one block per series (sequential in time)
    if (s >= cols || rows <= 0) return;

    int min_p = max(1, min_period);
    int max_p = max(min_p, max_period);

    int first_valid = first_valids[s];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= rows) return;

    // Prefix init to NaN
    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * cols + s] = NAN;
    }
    if (threadIdx.x != 0) return;

    // Seed at first_valid
    const float x0 = prices_tm[first_valid * cols + s];
    out_tm[first_valid * cols + s] = x0;

    const int warm_end = min(rows, first_valid + max_p - 1);
    int last_p = max_p;
    float last_val = x0;

    // Warmup advance without adaptation
    for (int t = first_valid + 1; t < warm_end; ++t) {
        const float x = prices_tm[t * cols + s];
        if (isfinite(x)) {
            const float sc = 2.0f / (float)(last_p + 1);
            last_val = fmaf_safe(x - last_val, sc, last_val);
        }
        out_tm[t * cols + s] = NAN; // warmup NaN except first_valid
    }
    if (warm_end >= rows) return;

    // Initialize rolling sums for first full window ending at warm_end-1
    double sum = 0.0, sumsq = 0.0;
    int nan_cnt = 0;
    for (int k = 0; k < max_p; ++k) {
        const float v = prices_tm[(first_valid + k) * cols + s];
        if (isfinite(v)) {
            const double dv = (double)v;
            sum += dv;
            sumsq += dv * dv;
        } else {
            ++nan_cnt;
        }
    }
    const double inv_n = 1.0 / (double)max_p;

    // Steady-state
    for (int t = warm_end; t < rows; ++t) {
        const float x = prices_tm[t * cols + s];
        if (!isfinite(x)) {
            out_tm[t * cols + s] = NAN;
        } else {
            float sc = 2.0f / (float)(last_p + 1);
            if (nan_cnt == 0) {
                const double m  = sum * inv_n;
                double var      = (sumsq * inv_n) - m * m;
                if (var < 0.0) var = 0.0;
                const double dv = sqrt(var);

                const double d175 = dv * 1.75;
                const double d025 = dv * 0.25;
                const double a = m - d175;
                const double b = m - d025;
                const double c = m + d025;
                const double d = m + d175;

                const int inc_fast = (x < a) || (x > d);
                const int inc_slow = (x >= b) && (x <= c);
                int p_next = last_p + (inc_slow - inc_fast);
                if (p_next < min_p) p_next = min_p;
                if (p_next > max_p) p_next = max_p;
                sc = 2.0f / (float)(p_next + 1);
                last_p = p_next;
            }

            last_val = fmaf_safe(x - last_val, sc, last_val);
            out_tm[t * cols + s] = last_val;
        }

        // Slide window to next time index (t+1)
        if (t + 1 < rows) {
            const int out_idx = t + 1 - max_p; // leaving index
            const float leaving = prices_tm[out_idx * cols + s];
            if (isfinite(leaving)) {
                const double dl = (double)leaving;
                sum   -= dl;
                sumsq -= dl * dl;
            } else {
                nan_cnt = max(0, nan_cnt - 1);
            }
            const float enter = prices_tm[(t + 1) * cols + s];
            if (isfinite(enter)) {
                const double de = (double)enter;
                sum   += de;
                sumsq += de * de;
            } else {
                ++nan_cnt;
            }
        }
    }
}

