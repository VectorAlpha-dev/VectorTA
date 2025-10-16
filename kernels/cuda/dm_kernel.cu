// CUDA kernels for Directional Movement (DM)
//
// Follows the scalar implementation in src/indicators/dm.rs
// - Warmup: accumulate over (period - 1) steps from first_valid+1 .. first_valid+period-1
// - Then Wilder smoothing recurrence per step:
//     sum_plus  = sum_plus  * (1 - 1/p) + plus_val
//     sum_minus = sum_minus * (1 - 1/p) + minus_val
// - Write NaN before warm index; write first values at warm index, then smooth forward
//
// Semantics
// - FP32 IO, FP64 accumulations
// - Division-free (no ATR); inputs may contain NaNs before first_valid

#include <cuda_runtime.h>
#include <math.h>

__device__ inline void fill_nan_row(float* ptr, int len) {
    const float nanv = nanf("");
    for (int i = 0; i < len; ++i) ptr[i] = nanv;
}

extern "C" __global__
void dm_batch_f32(const float* __restrict__ high,
                  const float* __restrict__ low,
                  const int* __restrict__ periods,
                  int series_len,
                  int n_combos,
                  int first_valid,
                  float* __restrict__ plus_out,
                  float* __restrict__ minus_out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int p = periods[combo];
    float* plus_row = plus_out + combo * series_len;
    float* minus_row = minus_out + combo * series_len;
    fill_nan_row(plus_row, series_len);
    fill_nan_row(minus_row, series_len);
    if (p <= 0) return;
    if (first_valid < 0 || first_valid + p - 1 >= series_len) return;

    const int i0 = first_valid;
    const int warm_end = i0 + p - 1; // index where first values are written

    double prev_h = (double)high[i0];
    double prev_l = (double)low[i0];
    double sum_plus = 0.0;
    double sum_minus = 0.0;

    // Warmup accumulation over (p - 1) steps
    for (int i = i0 + 1; i <= warm_end; ++i) {
        const double ch = (double)high[i];
        const double cl = (double)low[i];
        const double dp = ch - prev_h;
        const double dm = prev_l - cl;
        if (dp > 0.0 && dp > dm) sum_plus += dp;
        else if (dm > 0.0 && dm > dp) sum_minus += dm;
        prev_h = ch;
        prev_l = cl;
    }

    plus_row[warm_end]  = (float)sum_plus;
    minus_row[warm_end] = (float)sum_minus;
    if (warm_end + 1 >= series_len) return;

    const double rp = 1.0 / (double)p;
    const double one_minus_rp = 1.0 - rp;

    // Smoothed updates forward
    for (int i = warm_end + 1; i < series_len; ++i) {
        const double ch = (double)high[i];
        const double cl = (double)low[i];
        const double dp = ch - prev_h;
        const double dm = prev_l - cl;
        prev_h = ch;
        prev_l = cl;

        const double pv = (dp > 0.0 && dp > dm) ? dp : 0.0;
        const double mv = (dm > 0.0 && dm > dp) ? dm : 0.0;

        sum_plus  = sum_plus  * one_minus_rp + pv;
        sum_minus = sum_minus * one_minus_rp + mv;

        plus_row[i]  = (float)sum_plus;
        minus_row[i] = (float)sum_minus;
    }
}

extern "C" __global__
void dm_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ plus_tm,
    float* __restrict__ minus_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index (column)
    if (s >= cols) return;

    // Initialize column with NaNs
    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;
        plus_tm[idx]  = nanf("");
        minus_tm[idx] = nanf("");
    }
    if (period <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv + period - 1 >= rows) return;

    auto at = [&](int t) { return t * cols + s; };

    const int i0 = fv;
    const int warm_end = i0 + period - 1;

    double prev_h = (double)high_tm[at(i0)];
    double prev_l = (double)low_tm[at(i0)];
    double sum_plus = 0.0;
    double sum_minus = 0.0;

    for (int t = i0 + 1; t <= warm_end; ++t) {
        const double ch = (double)high_tm[at(t)];
        const double cl = (double)low_tm[at(t)];
        const double dp = ch - prev_h;
        const double dm = prev_l - cl;
        if (dp > 0.0 && dp > dm) sum_plus += dp;
        else if (dm > 0.0 && dm > dp) sum_minus += dm;
        prev_h = ch; prev_l = cl;
    }
    plus_tm[at(warm_end)]  = (float)sum_plus;
    minus_tm[at(warm_end)] = (float)sum_minus;
    if (warm_end + 1 >= rows) return;

    const double rp = 1.0 / (double)period;
    const double one_minus_rp = 1.0 - rp;

    for (int t = warm_end + 1; t < rows; ++t) {
        const double ch = (double)high_tm[at(t)];
        const double cl = (double)low_tm[at(t)];
        const double dp = ch - prev_h;
        const double dm = prev_l - cl;
        prev_h = ch; prev_l = cl;

        const double pv = (dp > 0.0 && dp > dm) ? dp : 0.0;
        const double mv = (dm > 0.0 && dm > dp) ? dm : 0.0;

        sum_plus  = sum_plus  * one_minus_rp + pv;
        sum_minus = sum_minus * one_minus_rp + mv;

        plus_tm[at(t)]  = (float)sum_plus;
        minus_tm[at(t)] = (float)sum_minus;
    }
}

