// CUDA kernels for RSMK (Relative Strength Mark)
//
// Pipeline (mirrors scalar rsmk.rs semantics):
// 1) lr[t] = ln(main[t] / compare[t]) with NaN when compare==0 or any NaN
// 2) mom[t] = lr[t] - lr[t-lookback]
// 3) indicator: EMA or SMA over momentum (scaled by 100.0)
// 4) signal: EMA or SMA over indicator
// Warmup/NaN policy matches scalar:
//   - Leading indices before first_valid are NaN
//   - Momentum becomes valid after first_valid + lookback
//   - indicator warmup = mom_first_valid + period - 1
//   - signal warmup    = indicator_warmup + signal_period - 1

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }

// --- Step 1+2: Build momentum for a given lookback (one series) ---
// Sequential kernel (one thread) to avoid auxiliary storage.
extern "C" __global__ void rsmk_momentum_f32(
    const float* __restrict__ main_in,
    const float* __restrict__ compare_in,
    int lookback,
    int first_valid, // first index where main,compare finite and compare!=0 observed on host
    int len,
    float* __restrict__ mom_out // length=len; NaN prefix up to first_valid+lookback-1
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || lookback <= 0) return;
    const int mom_fv = first_valid + lookback;
    for (int i = 0; i < min(mom_fv, len); ++i) { mom_out[i] = nanf; }
    if (mom_fv >= len) return;

    // Compute momentum via direct reads (no lr ring needed)
    for (int i = mom_fv; i < len; ++i) {
        const float a_m = main_in[i];
        const float a_c = compare_in[i];
        const float b_m = main_in[i - lookback];
        const float b_c = compare_in[i - lookback];
        float outv = nanf;
        if (!isnan(a_m) && !isnan(a_c) && !isnan(b_m) && !isnan(b_c) && a_c != 0.0f && b_c != 0.0f) {
            const float lr_new = logf(a_m / a_c);
            const float lr_old = logf(b_m / b_c);
            outv = lr_new - lr_old;
        }
        mom_out[i] = outv;
    }
}

// --- Step 3+4: Apply EMA(main) then EMA(signal) over momentum (batch, per-row sequential) ---
// Each block.y (or launch) handles a single row; sequential time loop per row.
extern "C" __global__ void rsmk_apply_mom_single_row_ema_ema_f32(
    const float* __restrict__ mom,
    int len,
    int first_valid_mom, // first_valid + lookback
    int period,
    int signal_period,
    float* __restrict__ out_indicator, // length=len (target row region)
    float* __restrict__ out_signal     // length=len (target row region)
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || period <= 0 || signal_period <= 0) return;

    const int ind_warm = first_valid_mom + period - 1;
    const int sig_warm = ind_warm + signal_period - 1;
    const float alpha_ind = 2.0f / (float(period) + 1.0f);
    const float alpha_sig = 2.0f / (float(signal_period) + 1.0f);

    // Warmup prefixes
    for (int i = 0; i < min(ind_warm, len); ++i) { out_indicator[i] = nanf; }
    for (int i = 0; i < min(sig_warm, len); ++i) { out_signal[i] = nanf; }
    if (ind_warm >= len) return;

    // Seed EMA(indicator) from SMA of first `period` momentum points (NaN-aware)
    double sum = 0.0; int cnt = 0;
    const int init_end = min(len, first_valid_mom + period);
    for (int i = first_valid_mom; i < init_end; ++i) {
        const float v = mom[i];
        if (!isnan(v)) { sum += (double)v; cnt += 1; }
    }
    if (cnt == 0) {
        // No finite seed: leave post-warmup as NaN as per scalar policy
        for (int i = ind_warm; i < len; ++i) { out_indicator[i] = nanf; }
        for (int i = sig_warm; i < len; ++i) { out_signal[i] = nanf; }
        return;
    }
    double ema_ind = (sum / double(cnt)) * 100.0;
    out_indicator[ind_warm] = (float)ema_ind;

    // Seed EMA(signal) from first `signal_period` indicator values
    double ema_sig = 0.0; bool sig_seeded = false;
    double acc_sig = ema_ind; int cnt_sig = 1;
    if (sig_warm == ind_warm) {
        ema_sig = (acc_sig / double(cnt_sig));
        out_signal[sig_warm] = (float)ema_sig; sig_seeded = true;
    }

    for (int i = ind_warm + 1; i < len; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            // ema_ind = ema_ind + alpha*(src100 - ema_ind)
            ema_ind = ((double)alpha_ind) * (src100 - ema_ind) + ema_ind;
        }
        out_indicator[i] = (float)ema_ind;

        if (!sig_seeded) {
            if (i < sig_warm) { acc_sig += ema_ind; cnt_sig += 1; }
            else if (i == sig_warm) {
                ema_sig = (acc_sig / (double)cnt_sig);
                out_signal[i] = (float)ema_sig; sig_seeded = true; continue;
            }
        } else {
            ema_sig = ((double)alpha_sig) * (ema_ind - ema_sig) + ema_sig;
            out_signal[i] = (float)ema_sig;
        }
    }
}

// --- Many-series × one-param (time-major), EMA/EMA path ---
// One thread per series (column); sequential scan over time.
extern "C" __global__ void rsmk_many_series_one_param_time_major_ema_ema_f32(
    const float* __restrict__ main_tm,
    const float* __restrict__ compare_tm,
    const int* __restrict__ first_valids, // length=cols
    int cols,
    int rows,
    int lookback,
    int period,
    int signal_period,
    float* __restrict__ out_indicator_tm,
    float* __restrict__ out_signal_tm
) {
    const int s = blockIdx.y;
    if (s >= cols) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return; // one thread per series
    const int stride = cols;
    const int fv = first_valids[s];
    const int mom_fv = fv + lookback;
    const int ind_warm = mom_fv + period - 1;
    const int sig_warm = ind_warm + signal_period - 1;
    const float nanf = qnan32();
    const float alpha_ind = 2.0f / (float(period) + 1.0f);
    const float alpha_sig = 2.0f / (float(signal_period) + 1.0f);

    for (int t = 0; t < min(ind_warm, rows); ++t) {
        const int idx = t * stride + s;
        out_indicator_tm[idx] = nanf;
    }
    for (int t = 0; t < min(sig_warm, rows); ++t) {
        const int idx = t * stride + s;
        out_signal_tm[idx] = nanf;
    }
    if (ind_warm >= rows) return;

    // Seed EMA(indicator) from SMA over first `period` momentum values
    double sum = 0.0; int cnt = 0;
    const int init_end = min(rows, mom_fv + period);
    for (int t = mom_fv; t < init_end; ++t) {
        const int i_new = t * stride + s;
        const int i_old = (t - lookback) * stride + s;
        const float m_new = main_tm[i_new];
        const float c_new = compare_tm[i_new];
        const float m_old = main_tm[i_old];
        const float c_old = compare_tm[i_old];
        if (!isnan(m_new) && !isnan(c_new) && !isnan(m_old) && !isnan(c_old) && c_new != 0.0f && c_old != 0.0f) {
            const float lr_new = logf(m_new / c_new);
            const float lr_old = logf(m_old / c_old);
            sum += (double)(lr_new - lr_old);
            cnt += 1;
        }
    }
    if (cnt == 0) {
        for (int t = ind_warm; t < rows; ++t) { out_indicator_tm[t * stride + s] = nanf; }
        for (int t = sig_warm; t < rows; ++t) { out_signal_tm[t * stride + s] = nanf; }
        return;
    }
    double ema_ind = (sum / double(cnt)) * 100.0;
    out_indicator_tm[ind_warm * stride + s] = (float)ema_ind;

    double ema_sig = 0.0; bool sig_seeded = false;
    double acc_sig = ema_ind; int cnt_sig = 1;
    if (sig_warm == ind_warm) {
        ema_sig = (acc_sig / (double)cnt_sig);
        out_signal_tm[sig_warm * stride + s] = (float)ema_sig; sig_seeded = true;
    }

    for (int t = ind_warm + 1; t < rows; ++t) {
        const int i_new = t * stride + s;
        const int i_old = (t - lookback) * stride + s;
        const float m_new = main_tm[i_new];
        const float c_new = compare_tm[i_new];
        const float m_old = main_tm[i_old];
        const float c_old = compare_tm[i_old];
        if (!isnan(m_new) && !isnan(c_new) && !isnan(m_old) && !isnan(c_old) && c_new != 0.0f && c_old != 0.0f) {
            const float lr_new = logf(m_new / c_new);
            const float lr_old = logf(m_old / c_old);
            const double mom = (double)(lr_new - lr_old);
            const double src100 = mom * 100.0;
            ema_ind = ((double)alpha_ind) * (src100 - ema_ind) + ema_ind;
        }
        out_indicator_tm[i_new] = (float)ema_ind;

        if (!sig_seeded) {
            if (t < sig_warm) { acc_sig += ema_ind; cnt_sig += 1; }
            else if (t == sig_warm) {
                ema_sig = (acc_sig / (double)cnt_sig);
                out_signal_tm[i_new] = (float)ema_sig; sig_seeded = true; continue;
            }
        } else {
            ema_sig = ((double)alpha_sig) * (ema_ind - ema_sig) + ema_sig;
            out_signal_tm[i_new] = (float)ema_sig;
        }
    }
}
