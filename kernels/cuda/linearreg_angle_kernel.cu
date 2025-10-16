// CUDA kernels for Linear Regression Angle (LRA).
//
// Batch path uses host-precomputed prefixes for sum(y), sum(k*y) with NaNs
// excluded (treated as zero in the prefix) and a prefix count of NaNs to
// guard window validity. This preserves CPU semantics and avoids poisoning
// prefix differences. Many-series path scans per series with O(1) sliding
// updates and performs an O(period) rebuild when a boundary value is NaN,
// matching the scalar behavior and allowing recovery after NaN windows.

#include <cuda_runtime.h>
#include <math.h>

#ifndef LRA_NAN_F
#define LRA_NAN_F (__int_as_float(0x7fffffff))
#endif

// -------------------------- Helpers --------------------------

static __device__ __forceinline__ int tm_idx(int row, int num_series, int series) {
    return row * num_series + series;
}

// -------------------------- Batch kernel (one series × many params) --------------------------

// Inputs:
//  - prices:        [len]
//  - prefix_sum:    [len+1] double, running sum of y with NaNs treated as 0
//  - prefix_kd:     [len+1] double, running sum of (k_abs * y) with NaNs treated as 0
//  - prefix_nan:    [len+1] int, running count of NaNs
//  - periods:       [n_combos]
//  - sum_x:         [n_combos] float, Σx for x=0..p-1 = p*(p-1)/2
//  - inv_div:       [n_combos] float, 1 / (sum_x^2 - p*Σx^2), where Σx^2 = p*(p-1)*(2p-1)/6
//  - first_valid:   first non-NaN index in prices
// Output:
//  - out:           [n_combos * len] row-major (combo-major) results in degrees

extern "C" __global__ void linearreg_angle_batch_f32(
    const float*  __restrict__ prices,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_kd,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    const float*  __restrict__ sum_x,
    const float*  __restrict__ inv_div,
    int n_combos,
    float*        __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period < 2 || period > len) return;

    const int warm = first_valid + period - 1;
    const float sx_f   = sum_x[combo];
    const float invd_f = inv_div[combo];
    const double sx    = (double)sx_f;
    const double invd  = (double)invd_f;
    const double p_d   = (double)period;
    const float rad2deg_f = 180.0f / (float)M_PI;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int row_off = combo * len;

    while (t < len) {
        float outv = LRA_NAN_F;
        if (t >= warm) {
            const int start = t + 1 - period;
            const int nan_cnt = prefix_nan[t + 1] - prefix_nan[start];
            if (nan_cnt == 0) {
                const double sum_y  = prefix_sum[t + 1] - prefix_sum[start];
                const double sum_kd = prefix_kd[t + 1] - prefix_kd[start];
                // reversed-x identity: sum_xy = i*sum_y - sum_kd
                const double sum_xy = ((double)t) * sum_y - sum_kd;
                const double num = fma(p_d, sum_xy, -sx * sum_y); // p*sum_xy - sum_x*sum_y
                const double slope = num * invd;
                const float angle = atanf((float)slope) * rad2deg_f;
                outv = angle;
            }
        }
        out[row_off + t] = outv;
        t += stride;
    }
}

// -------------------------- Many-series kernel (time-major, one param) --------------------------

extern "C" __global__ void linearreg_angle_many_series_one_param_f32(
    const float* __restrict__ prices_tm, // [rows * cols], time-major
    const int*   __restrict__ first_valids, // [cols]
    int cols,
    int rows,
    int period,
    float sum_x_f,
    float inv_div_f,
    float* __restrict__ out_tm)
{
    const int stride = blockDim.x * gridDim.x;
    const double p_d  = (double)period;
    const double sx   = (double)sum_x_f;
    const double invd = (double)inv_div_f;
    const float rad2deg_f = 180.0f / (float)M_PI;

    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += stride) {
        if (period < 2 || period > rows) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }
        const int fv = first_valids[s];
        if (fv < 0 || fv >= rows) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }
        const int tail = rows - fv;
        if (tail < period) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }

        const int warm = fv + period - 1;
        for (int r = 0; r < warm; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;

        // Warm-up: sums over first (period-1) elements using absolute index weighting
        double y_sum = 0.0;
        double sum_kd = 0.0; // Σ (k_abs * y)
        bool invalid = false;
        for (int k = 0; k < period - 1; ++k) {
            const int r0 = fv + k;
            const float v_f = prices_tm[tm_idx(r0, cols, s)];
            if (isnan(v_f)) { invalid = true; break; }
            const double v = (double)v_f;
            y_sum  += v;
            sum_kd += (double)r0 * v;
        }

        float latest_f = prices_tm[tm_idx(warm, cols, s)];

        for (int r = warm; r < rows; ++r) {
            float outv = LRA_NAN_F;
            float enter_f = latest_f;
            if (r + 1 < rows) latest_f = prices_tm[tm_idx(r + 1, cols, s)];
            const float leave_f = prices_tm[tm_idx(r - period + 1, cols, s)];

            bool need_rebuild = invalid || isnan(enter_f) || isnan(leave_f);
            if (need_rebuild) {
                // Rebuild window sums; mark invalid if any NaN in window
                y_sum = 0.0; sum_kd = 0.0; invalid = false;
                for (int k = 0; k < period; ++k) {
                    const int r0 = r - period + 1 + k;
                    const float v_f = prices_tm[tm_idx(r0, cols, s)];
                    if (isnan(v_f)) { invalid = true; break; }
                    const double v = (double)v_f;
                    y_sum  += v;
                    sum_kd += (double)r0 * v;
                }
                if (!invalid) {
                    const double sum_xy = (double)r * y_sum - sum_kd;
                    const double b_num = fma(p_d, sum_xy, -sx * y_sum);
                    const double slope = b_num * invd;
                    outv = atanf((float)slope) * rad2deg_f;
                } else {
                    outv = LRA_NAN_F;
                }
            } else {
                // O(1) slide using absolute-index identity
                y_sum  += (double)enter_f;
                sum_kd += (double)r * (double)enter_f;
                const double sum_xy = (double)r * y_sum - sum_kd;
                const double b_num = fma(p_d, sum_xy, -sx * y_sum);
                const double slope = b_num * invd;
                outv = atanf((float)slope) * rad2deg_f;

                // Roll window
                y_sum  -= (double)leave_f;
                sum_kd -= (double)(r - period + 1) * (double)leave_f;
            }

            out_tm[tm_idx(r, cols, s)] = outv;
            invalid = false;
        }
    }
}
