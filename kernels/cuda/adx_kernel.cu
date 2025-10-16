// CUDA kernels for Average Directional Index (ADX)
//
// Math follows the scalar Rust implementation in src/indicators/adx.rs:
// - Warmup sums over [first_valid+1 .. first_valid+period]
// - Wilder-style smoothing of TR, +DM, -DM
// - DX accumulation for first `period` points to seed ADX
// - Then ADX recurrence: adx = (adx*(p-1) + dx) / p
//
// Semantics:
// - FP32 inputs/outputs, double accumulations for stability
// - Before warm_end = first_valid + 2*period - 1, outputs are NaN
// - Division-by-zero branches yield 0.0 exactly like the scalar path

#include <cuda_runtime.h>
#include <math.h>

__device__ inline void fill_nan_row(float* ptr, int len) {
    const float nan = nanf("");
    for (int i = 0; i < len; ++i) ptr[i] = nan;
}

extern "C" __global__
void adx_batch_f32(const float* __restrict__ high,
                   const float* __restrict__ low,
                   const float* __restrict__ close,
                   const int* __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int p = periods[combo];
    float* row = out + combo * series_len;
    fill_nan_row(row, series_len);
    if (p <= 0) return;

    // Require at least period+1 valid bars after first_valid (host validates, but keep defensive)
    if (first_valid < 0 || first_valid + p >= series_len) return;

    // Warmup accumulation j = 1..=p (relative to first_valid)
    int i0 = first_valid;
    double prev_h = (double)high[i0];
    double prev_l = (double)low[i0];
    double prev_c = (double)close[i0];

    double tr_sum = 0.0;
    double plus_dm_sum = 0.0;
    double minus_dm_sum = 0.0;
    for (int j = 1; j <= p; ++j) {
        const int i = i0 + j;
        const double ch = (double)high[i];
        const double cl = (double)low[i];
        const double hl = ch - cl;
        const double hpc = fabs(ch - prev_c);
        const double lpc = fabs(cl - prev_c);
        const double tr = fmax(fmax(hl, hpc), lpc);
        const double up = ch - prev_h;
        const double down = prev_l - cl;
        if (up > down && up > 0.0) plus_dm_sum += up;
        if (down > up && down > 0.0) minus_dm_sum += down;
        tr_sum += tr;
        prev_h = ch;
        prev_l = cl;
        prev_c = (double)close[i];
    }

    double atr = tr_sum;
    double plus_s = plus_dm_sum;
    double minus_s = minus_dm_sum;

    const double rp = 1.0 / (double)p;
    const double one_minus_rp = 1.0 - rp;
    const double pm1 = (double)p - 1.0;

    // Initial DX from the first smoothed window
    double plus_di_prev = (atr != 0.0) ? ((plus_s / atr) * 100.0) : 0.0;
    double minus_di_prev = (atr != 0.0) ? ((minus_s / atr) * 100.0) : 0.0;
    double sum_di_prev = plus_di_prev + minus_di_prev;
    double dx_sum = (sum_di_prev != 0.0) ? (fabs(plus_di_prev - minus_di_prev) / sum_di_prev) * 100.0 : 0.0;
    int dx_count = 1;
    double adx_last = 0.0;

    // Main pass i = first_valid + p + 1 .. series_len-1
    int i = i0 + p + 1;
    double prev_h2 = (double)high[i0 + p];
    double prev_l2 = (double)low[i0 + p];
    double prev_c2 = (double)close[i0 + p];
    for (; i < series_len; ++i) {
        const double ch = (double)high[i];
        const double cl = (double)low[i];
        const double hl = ch - cl;
        const double hpc = fabs(ch - prev_c2);
        const double lpc = fabs(cl - prev_c2);
        const double tr = fmax(fmax(hl, hpc), lpc);
        const double up = ch - prev_h2;
        const double down = prev_l2 - cl;
        const double plus_dm = (up > down && up > 0.0) ? up : 0.0;
        const double minus_dm = (down > up && down > 0.0) ? down : 0.0;

        atr = atr * one_minus_rp + tr;
        plus_s = plus_s * one_minus_rp + plus_dm;
        minus_s = minus_s * one_minus_rp + minus_dm;

        const double plus_di = (atr != 0.0) ? ((plus_s / atr) * 100.0) : 0.0;
        const double minus_di = (atr != 0.0) ? ((minus_s / atr) * 100.0) : 0.0;
        const double sum_di = plus_di + minus_di;
        const double dx = (sum_di != 0.0) ? (fabs(plus_di - minus_di) / sum_di) * 100.0 : 0.0;

        if (dx_count < p) {
            dx_sum += dx;
            dx_count += 1;
            if (dx_count == p) {
                adx_last = dx_sum * rp;
                row[i] = (float)adx_last;
            }
        } else {
            adx_last = (adx_last * pm1 + dx) * rp;
            row[i] = (float)adx_last;
        }

        prev_h2 = ch;
        prev_l2 = cl;
        prev_c2 = (double)close[i];
    }
}

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
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    // Initialize column with NaNs
    for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nanf("");
    if (period <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv + period >= rows) return;

    auto at = [&](int t) {
        return t * cols + s;
    };

    // Warmup j=1..=period relative to fv
    int i0 = fv;
    double prev_h = (double)high_tm[at(i0)];
    double prev_l = (double)low_tm[at(i0)];
    double prev_c = (double)close_tm[at(i0)];

    double tr_sum = 0.0, plus_sum = 0.0, minus_sum = 0.0;
    for (int j = 1; j <= period; ++j) {
        const int t = i0 + j;
        const double ch = (double)high_tm[at(t)];
        const double cl = (double)low_tm[at(t)];
        const double hl = ch - cl;
        const double hpc = fabs(ch - prev_c);
        const double lpc = fabs(cl - prev_c);
        const double tr = fmax(fmax(hl, hpc), lpc);
        const double up = ch - prev_h;
        const double down = prev_l - cl;
        if (up > down && up > 0.0) plus_sum += up;
        if (down > up && down > 0.0) minus_sum += down;
        tr_sum += tr;
        prev_h = ch; prev_l = cl; prev_c = (double)close_tm[at(t)];
    }

    double atr = tr_sum;
    double plus_s = plus_sum;
    double minus_s = minus_sum;

    const double rp = 1.0 / (double)period;
    const double one_minus_rp = 1.0 - rp;
    const double pm1 = (double)period - 1.0;

    double plus_di_prev = (atr != 0.0) ? ((plus_s / atr) * 100.0) : 0.0;
    double minus_di_prev = (atr != 0.0) ? ((minus_s / atr) * 100.0) : 0.0;
    double sum_di_prev = plus_di_prev + minus_di_prev;
    double dx_sum = (sum_di_prev != 0.0) ? (fabs(plus_di_prev - minus_di_prev) / sum_di_prev) * 100.0 : 0.0;
    int dx_count = 1;
    double adx_last = 0.0;

    int t = i0 + period + 1;
    double prev_h2 = (double)high_tm[at(i0 + period)];
    double prev_l2 = (double)low_tm[at(i0 + period)];
    double prev_c2 = (double)close_tm[at(i0 + period)];
    for (; t < rows; ++t) {
        const double ch = (double)high_tm[at(t)];
        const double cl = (double)low_tm[at(t)];
        const double hl = ch - cl;
        const double hpc = fabs(ch - prev_c2);
        const double lpc = fabs(cl - prev_c2);
        const double tr = fmax(fmax(hl, hpc), lpc);
        const double up = ch - prev_h2;
        const double down = prev_l2 - cl;
        const double plus_dm = (up > down && up > 0.0) ? up : 0.0;
        const double minus_dm = (down > up && down > 0.0) ? down : 0.0;

        atr = atr * one_minus_rp + tr;
        plus_s = plus_s * one_minus_rp + plus_dm;
        minus_s = minus_s * one_minus_rp + minus_dm;

        const double plus_di = (atr != 0.0) ? ((plus_s / atr) * 100.0) : 0.0;
        const double minus_di = (atr != 0.0) ? ((minus_s / atr) * 100.0) : 0.0;
        const double sum_di = plus_di + minus_di;
        const double dx = (sum_di != 0.0) ? (fabs(plus_di - minus_di) / sum_di) * 100.0 : 0.0;

        if (dx_count < period) {
            dx_sum += dx;
            dx_count += 1;
            if (dx_count == period) {
                adx_last = dx_sum * rp;
                out_tm[at(t)] = (float)adx_last;
            }
        } else {
            adx_last = (adx_last * pm1 + dx) * rp;
            out_tm[at(t)] = (float)adx_last;
        }

        prev_h2 = ch; prev_l2 = cl; prev_c2 = (double)close_tm[at(t)];
    }
}

