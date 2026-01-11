// CUDA kernels for OTT (Optimized Trend Tracker)
//
// Two usage patterns are supported:
// 1) Apply OTT to a precomputed moving-average series (generic path for any MA):
//      - ott_apply_single_f32(ma, len, percent, out)
//      - ott_many_series_one_param_f32(ma_tm, cols, rows, percent, out_tm)
// 2) VAR-specific fast path that inlines VAR (VIDYA) computation and then applies OTT:
//      - ott_from_var_batch_f32(prices, periods, percents, len, n_combos, out)
//      - ott_from_var_many_series_one_param_f32(prices_tm, cols, rows, period, percent, out_tm)
//
// Notes:
// - All kernels execute one sequential scan per row/series (IIR/recurrence), so we use one
//   thread per row. Launch with grid.x = rows (or series) and blockDim.x >= 1; threadIdx.x==0
//   performs the scan.
// - Outputs are expected to be prefilled with qNaN; kernels only write valid positions.
// - Arithmetic uses double for internal state to mirror f64 scalar path; outputs are f32.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Find first index >= start where v[i] is finite; return len if none
static __device__ __forceinline__ int find_first_finite(const float* v, int start, int len) {
    for (int i = start; i < len; ++i) {
        if (isfinite(v[i])) return i;
    }
    return len;
}

// Apply OTT to a single moving-average row (ma length = series_len)
extern "C" __global__
void ott_apply_single_f32(const float* __restrict__ ma,
                          int series_len,
                          float percent,
                          float* __restrict__ out) {
    if (threadIdx.x != 0) return;
    if (series_len <= 0) return;

    const float fark = percent * 0.01f;               // percent / 100
    const float scale_minus = 1.0f - percent * 0.005f; // (200 - percent) / 200

    // Seed from first non-NaN MA value
    int i = find_first_finite(ma, 0, series_len);
    if (i >= series_len) return;

    float m = ma[i];
    float long_stop = fmaf(-fark, m, m);  // m * (1 - fark)
    float short_stop = fmaf( fark, m, m); // m * (1 + fark)
    int dir = 1; // 1 = long, -1 = short

    // First output
    float mt0 = long_stop; // dir == 1 at seed
    float scale0 = (m > mt0) ? (scale_minus + fark) : (scale_minus);
    out[i] = mt0 * scale0;
    ++i;

    for (; i < series_len; ++i) {
        float mf = ma[i];
        if (!isfinite(mf)) continue;
        float mavg = mf;

        float cand_long = fmaf(-fark, mavg, mavg); // mavg * (1 - fark)
        float cand_short = fmaf( fark, mavg, mavg); // mavg * (1 + fark)

        float lprev = long_stop;
        float sprev = short_stop;

        // Update long/short stops
        if (mavg > lprev) {
            long_stop = (cand_long > lprev) ? cand_long : lprev;
        } else {
            long_stop = cand_long;
        }
        if (mavg < sprev) {
            short_stop = (cand_short < sprev) ? cand_short : sprev;
        } else {
            short_stop = cand_short;
        }

        // Direction switch
        if (dir == -1 && mavg > sprev) {
            dir = 1;
        } else if (dir == 1 && mavg < lprev) {
            dir = -1;
        }

        // MT and scaled output
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (mavg > mt) ? (scale_minus + fark) : (scale_minus);
        out[i] = mt * scale;
    }
}

// VAR helper: compute next VAR state given previous state and up/down ring
static __device__ __forceinline__ float vidya_alpha_base(int period) {
    return 2.0f / ((float)period + 1.0f);
}

// Batch kernel: compute VAR (VIDYA) inline on prices, then apply OTT per combo
extern "C" __global__
void ott_from_var_batch_f32(const float* __restrict__ prices,
                            const int*   __restrict__ periods,
                             const float* __restrict__ percents,
                             int series_len,
                             int n_combos,
                             float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) return;

    const int period = periods[combo];
    const float percent = percents[combo];
    if (period <= 0 || series_len <= 0 || !isfinite(percent)) return;

    float* __restrict__ out_row = out + combo * series_len;

    // Find first finite price
    int first = -1;
    for (int i = 0; i < series_len; ++i) {
        if (isfinite(prices[i])) { first = i; break; }
    }
    if (first < 0) return;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;
    const float valpha_base = vidya_alpha_base(period);

    // VAR rings for last 9 diffs
    float ring_u[9];
    float ring_d[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) { ring_u[k] = 0.0; ring_d[k] = 0.0; }
    float u_sum = 0.0f, d_sum = 0.0f;
    int ridx = 0;

    // Seed VAR at first bar (Pine-compatible: nz(VAR[1]) = 0)
    float var = 0.0f;

    // Seed OTT from first MA value (var=0)
    float long_stop = fmaf(-fark, var, var);
    float short_stop = fmaf( fark, var, var);
    int dir = 1;

    // First output
    float mt0 = long_stop;
    float scale0 = (var > mt0) ? (scale_minus + fark) : (scale_minus);
    out_row[first] = mt0 * scale0;

    // Prefill window for next up to 8 diffs
    int pre_end = (first + 8 < series_len ? first + 8 : series_len - 1);
    for (int i = first + 1; i <= pre_end; ++i) {
        float a = prices[i - 1];
        float b = prices[i];
        if (!isfinite(a) || !isfinite(b)) continue;
        float up = b - a; if (up < 0.0f) up = 0.0f; 
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        ring_u[ridx] = up;  u_sum += up;
        ring_d[ridx] = dn;  d_sum += dn;
        ridx = (ridx + 1) % 9;

        // var remains 0.0 until full window; apply OTT with current var
        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? (scale_minus + fark) : (scale_minus);
        out_row[i] = mt * scale;
    }

    // Main loop once we have 9 diffs
    for (int i = first + 9; i < series_len; ++i) {
        float a = prices[i - 1];
        float b = prices[i];
        if (!isfinite(a) || !isfinite(b)) continue;
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        float old_u = ring_u[ridx];
        float old_d = ring_d[ridx];
        ring_u[ridx] = up; ring_d[ridx] = dn;
        ridx = (ridx + 1) % 9;
        u_sum += up - old_u;
        d_sum += dn - old_d;
        float denom = u_sum + d_sum;
        float vcmo = (denom != 0.0f) ? ((u_sum - d_sum) / denom) : 0.0f;
        float avalpha = valpha_base * fabsf(vcmo);
        var = fmaf(avalpha, b, (1.0f - avalpha) * var);

        // OTT update
        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? (scale_minus + fark) : (scale_minus);
        out_row[i] = mt * scale;
    }
}

// Many-series: apply OTT to MA values arranged time-major (rows x cols)
extern "C" __global__
void ott_many_series_one_param_f32(const float* __restrict__ ma_tm,
                                   int cols,
                                   int rows,
                                   float percent,
                                   float* __restrict__ out_tm) {
    // Time-major layout: element at (t, s) is [t * cols + s]
    const int s = blockIdx.x; // series index
    if (s >= cols || threadIdx.x != 0) return;
    if (rows <= 0) return;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;

    // Find first finite along time for this series
    int t = 0;
    for (; t < rows; ++t) { if (isfinite(ma_tm[(size_t)t * (size_t)cols + s])) break; }
    if (t >= rows) return;

    float m = ma_tm[(size_t)t * (size_t)cols + s];
    float long_stop = fmaf(-fark, m, m);
    float short_stop = fmaf( fark, m, m);
    int dir = 1;
    float mt0 = long_stop;
    float scale0 = (m > mt0) ? (scale_minus + fark) : (scale_minus);
    out_tm[(size_t)t * (size_t)cols + s] = mt0 * scale0;
    ++t;
    for (; t < rows; ++t) {
        float mf = ma_tm[(size_t)t * (size_t)cols + s];
        if (!isfinite(mf)) continue;
        float mavg = mf;
        float cand_long = fmaf(-fark, mavg, mavg);
        float cand_short = fmaf( fark, mavg, mavg);
        float lprev = long_stop, sprev = short_stop;
        if (mavg > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (mavg < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && mavg > sprev) dir = 1; else if (dir == 1 && mavg < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (mavg > mt) ? (scale_minus + fark) : (scale_minus);
        out_tm[(size_t)t * (size_t)cols + s] = mt * scale;
    }
}

// Many-series VAR path: compute VAR inline then apply OTT
extern "C" __global__
void ott_from_var_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                            int cols,
                                            int rows,
                                            int period,
                                            float percent,
                                            float* __restrict__ out_tm) {
    const int s = blockIdx.x; // series index
    if (s >= cols || threadIdx.x != 0) return;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;
    const float valpha_base = vidya_alpha_base(period);

    // First finite price
    int first = -1;
    for (int t = 0; t < rows; ++t) { if (isfinite(prices_tm[(size_t)t * (size_t)cols + s])) { first = t; break; } }
    if (first < 0) return;

    // VAR rings
    float ring_u[9];
    float ring_d[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) { ring_u[k] = 0.0; ring_d[k] = 0.0; }
    float u_sum = 0.0f, d_sum = 0.0f; int ridx = 0;
    float var = 0.0f;

    // Seed OTT
    float long_stop = fmaf(-fark, var, var);
    float short_stop = fmaf( fark, var, var);
    int dir = 1;
    float mt0 = long_stop; float scale0 = (var > mt0) ? (scale_minus + fark) : (scale_minus);
    out_tm[(size_t)first * (size_t)cols + s] = mt0 * scale0;

    int pre_end = (first + 8 < rows ? first + 8 : rows - 1);
    for (int t = first + 1; t <= pre_end; ++t) {
        float a = prices_tm[(size_t)(t - 1) * (size_t)cols + s];
        float b = prices_tm[(size_t)t * (size_t)cols + s];
        if (!isfinite(a) || !isfinite(b)) continue;
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        ring_u[ridx] = up; u_sum += up; ring_d[ridx] = dn; d_sum += dn; ridx = (ridx + 1) % 9;
        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? (scale_minus + fark) : (scale_minus);
        out_tm[(size_t)t * (size_t)cols + s] = mt * scale;
    }
    for (int t = first + 9; t < rows; ++t) {
        float a = prices_tm[(size_t)(t - 1) * (size_t)cols + s];
        float b = prices_tm[(size_t)t * (size_t)cols + s];
        if (!isfinite(a) || !isfinite(b)) continue;
        float up = b - a; if (up < 0.0f) up = 0.0f; float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        float old_u = ring_u[ridx]; float old_d = ring_d[ridx]; ring_u[ridx] = up; ring_d[ridx] = dn; ridx = (ridx + 1) % 9;
        u_sum += up - old_u; d_sum += dn - old_d; float denom = u_sum + d_sum; float vcmo = (denom != 0.0f) ? ((u_sum - d_sum) / denom) : 0.0f;
        float avalpha = valpha_base * fabsf(vcmo); var = fmaf(avalpha, b, (1.0f - avalpha) * var);
        float cand_long = fmaf(-fark, var, var); float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop; float scale = (var > mt) ? (scale_minus + fark) : (scale_minus);
        out_tm[(size_t)t * (size_t)cols + s] = mt * scale;
    }
}
