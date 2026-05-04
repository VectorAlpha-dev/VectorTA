#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }


extern "C" __global__ void rsmk_momentum_f32(
    const float* __restrict__ main_in,
    const float* __restrict__ compare_in,
    int lookback,
    int first_valid,
    int len,
    float* __restrict__ mom_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || lookback <= 0) return;
    const int mom_fv = first_valid + lookback;
    for (int i = 0; i < min(mom_fv, len); ++i) { mom_out[i] = nanf; }
    if (mom_fv >= len) return;


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


extern "C" __global__ void rsmk_apply_mom_single_row_ema_ema_f32(
    const float* __restrict__ mom,
    int len,
    int first_valid_mom,
    int period,
    int signal_period,
    float* __restrict__ out_indicator,
    float* __restrict__ out_signal
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || period <= 0 || signal_period <= 0) return;

    int first = first_valid_mom;
    if (first < 0) first = 0;
    if (first >= len) return;
    while (first < len && isnan(mom[first])) { first += 1; }
    if (first >= len) return;

    const double alpha_ind = 2.0 / (double(period) + 1.0);
    const double beta_ind = 1.0 - alpha_ind;
    const double alpha_sig = 2.0 / (double(signal_period) + 1.0);
    const double beta_sig = 1.0 - alpha_sig;

    for (int i = 0; i < first; ++i) {
        out_indicator[i] = nanf;
        out_signal[i] = nanf;
    }

    double ind_mean = (double)mom[first] * 100.0;
    int ind_count = 1;
    out_indicator[first] = (float)ind_mean;

    const int ind_warm_end = min(len, first + period);
    for (int i = first + 1; i < ind_warm_end; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ind_count += 1;
            ind_mean = (((double)(ind_count - 1) * ind_mean) + src100) / (double)ind_count;
        }
        out_indicator[i] = (float)ind_mean;
    }

    double ind_val = ind_mean;
    for (int i = ind_warm_end; i < len; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ind_val = beta_ind * ind_val + alpha_ind * src100;
        }
        out_indicator[i] = (float)ind_val;
    }

    double sig_mean = (double)out_indicator[first];
    int sig_count = 1;
    out_signal[first] = (float)sig_mean;

    const int sig_warm_end = min(len, first + signal_period);
    for (int i = first + 1; i < sig_warm_end; ++i) {
        const float iv = out_indicator[i];
        if (!isnan(iv)) {
            sig_count += 1;
            sig_mean = (((double)(sig_count - 1) * sig_mean) + (double)iv) / (double)sig_count;
        }
        out_signal[i] = (float)sig_mean;
    }

    double sig_val = sig_mean;
    for (int i = sig_warm_end; i < len; ++i) {
        const float iv = out_indicator[i];
        if (!isnan(iv)) {
            sig_val = beta_sig * sig_val + alpha_sig * (double)iv;
        }
        out_signal[i] = (float)sig_val;
    }
}


extern "C" __global__ void rsmk_indicator_from_mom_ema_f32(
    const float* __restrict__ mom,
    int len,
    int first_valid_mom,
    int period,
    float* __restrict__ out_indicator
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || period <= 0) return;

    int first = first_valid_mom;
    if (first < 0) first = 0;
    if (first >= len) return;
    while (first < len && isnan(mom[first])) { first += 1; }
    if (first >= len) return;

    const double alpha_ind = 2.0 / (double(period) + 1.0);
    const double beta_ind = 1.0 - alpha_ind;

    for (int i = 0; i < first; ++i) {
        out_indicator[i] = nanf;
    }

    double ind_mean = (double)mom[first] * 100.0;
    int ind_count = 1;
    out_indicator[first] = (float)ind_mean;

    const int ind_warm_end = min(len, first + period);
    for (int i = first + 1; i < ind_warm_end; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ind_count += 1;
            ind_mean = (((double)(ind_count - 1) * ind_mean) + src100) / (double)ind_count;
        }
        out_indicator[i] = (float)ind_mean;
    }

    double ind_val = ind_mean;
    for (int i = ind_warm_end; i < len; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ind_val = beta_ind * ind_val + alpha_ind * src100;
        }
        out_indicator[i] = (float)ind_val;
    }
}


extern "C" __global__ void rsmk_signal_from_indicator_ema_f32(
    const float* __restrict__ indicator,
    int len,
    int first_valid_mom,
    int signal_period,
    float* __restrict__ out_signal
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || signal_period <= 0) return;

    int first = first_valid_mom;
    if (first < 0) first = 0;
    if (first >= len) return;
    while (first < len && isnan(indicator[first])) { first += 1; }
    if (first >= len) return;

    for (int i = 0; i < first; ++i) {
        out_signal[i] = nanf;
    }

    const double alpha_sig = 2.0 / (double(signal_period) + 1.0);
    const double beta_sig = 1.0 - alpha_sig;
    double sig_mean = (double)indicator[first];
    int sig_count = 1;
    out_signal[first] = (float)sig_mean;

    const int sig_warm_end = min(len, first + signal_period);
    for (int i = first + 1; i < sig_warm_end; ++i) {
        const float iv = indicator[i];
        if (!isnan(iv)) {
            sig_count += 1;
            sig_mean = (((double)(sig_count - 1) * sig_mean) + (double)iv) / (double)sig_count;
        }
        out_signal[i] = (float)sig_mean;
    }

    double sig_val = sig_mean;
    for (int i = sig_warm_end; i < len; ++i) {
        const float iv = indicator[i];
        if (!isnan(iv)) {
            sig_val = beta_sig * sig_val + alpha_sig * (double)iv;
        }
        out_signal[i] = (float)sig_val;
    }
}


extern "C" __global__ void rsmk_copy_group_indicator_tiled_f32(
    const float* __restrict__ group_indicator,
    const int* __restrict__ row_group_idx,
    int len,
    int n_rows,
    float* __restrict__ out_indicator
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= n_rows || t >= len) return;
    const int group = row_group_idx[row];
    out_indicator[row * len + t] = group_indicator[group * len + t];
}


extern "C" __global__ void rsmk_apply_mom_single_row_ema_ema_classic_f32(
    const float* __restrict__ mom,
    int len,
    int first_valid_mom,
    int period,
    int signal_period,
    float* __restrict__ out_indicator,
    float* __restrict__ out_signal
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float nanf = qnan32();
    if (len <= 0 || period <= 0 || signal_period <= 0) return;

    int first = first_valid_mom;
    if (first < 0) first = 0;
    if (first >= len) return;
    const int ind_warm = first + period - 1;
    const int sig_warm = ind_warm + signal_period - 1;
    if (ind_warm >= len) return;

    const double alpha_ind = 2.0 / (double(period) + 1.0);
    const double beta_ind = 1.0 - alpha_ind;
    const double alpha_sig = 2.0 / (double(signal_period) + 1.0);
    const double beta_sig = 1.0 - alpha_sig;

    for (int i = 0; i < min(ind_warm, len); ++i) {
        out_indicator[i] = nanf;
    }
    for (int i = 0; i < min(sig_warm, len); ++i) {
        out_signal[i] = nanf;
    }

    double sum_ind = 0.0;
    int count_ind = 0;
    const int ind_seed_end = min(len, first + period);
    for (int i = first; i < ind_seed_end; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            sum_ind += (double)mv;
            count_ind += 1;
        }
    }
    if (count_ind == 0) return;

    double ema_ind = (sum_ind / (double)count_ind) * 100.0;
    out_indicator[ind_warm] = (float)ema_ind;

    for (int i = ind_warm + 1; i < len; ++i) {
        const float mv = mom[i];
        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ema_ind = beta_ind * ema_ind + alpha_ind * src100;
        }
        out_indicator[i] = (float)ema_ind;
    }

    if (sig_warm >= len) return;

    double sum_sig = 0.0;
    int count_sig = 0;
    const int sig_seed_end = min(len, ind_warm + signal_period);
    for (int i = ind_warm; i < sig_seed_end; ++i) {
        const float iv = out_indicator[i];
        if (!isnan(iv)) {
            sum_sig += (double)iv;
            count_sig += 1;
        }
    }
    if (count_sig == 0) return;

    double ema_sig = sum_sig / (double)count_sig;
    out_signal[sig_warm] = (float)ema_sig;

    for (int i = sig_warm + 1; i < len; ++i) {
        const float iv = out_indicator[i];
        if (!isnan(iv)) {
            ema_sig = beta_sig * ema_sig + alpha_sig * (double)iv;
        }
        out_signal[i] = (float)ema_sig;
    }
}


extern "C" __global__ void rsmk_many_series_one_param_time_major_ema_ema_f32(
    const float* __restrict__ main_tm,
    const float* __restrict__ compare_tm,
    const int* __restrict__ first_valids,
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
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int stride = cols;
    const int fv = first_valids[s];
    const float nanf = qnan32();
    if (rows <= 0 || lookback <= 0 || period <= 0 || signal_period <= 0) return;


    const int mom_fv = fv + lookback;
    const int ind_warm = mom_fv + period - 1;
    const int sig_warm = ind_warm + signal_period - 1;
    const double alpha_ind = 2.0 / (double(period) + 1.0);
    const double alpha_sig = 2.0 / (double(signal_period) + 1.0);


    for (int t = 0; t < min(ind_warm, rows); ++t) {
        out_indicator_tm[t * stride + s] = nanf;
    }
    for (int t = 0; t < min(sig_warm, rows); ++t) {
        out_signal_tm[t * stride + s] = nanf;
    }
    if (ind_warm >= rows) return;


    double sum = 0.0; int cnt = 0;
    const int init_end = min(rows, mom_fv + period);
    for (int t = mom_fv; t < init_end; ++t) {
        const int i_new = t * stride + s;
        const int i_old = (t - lookback) * stride + s;
        const float m_new = main_tm[i_new];
        const float c_new = compare_tm[i_new];
        const float m_old = main_tm[i_old];
        const float c_old = compare_tm[i_old];
        float mv = nanf;
        if (!isnan(m_new) && !isnan(c_new) && !isnan(m_old) && !isnan(c_old) && c_new != 0.0f && c_old != 0.0f) {
            const float lr_new = logf(m_new / c_new);
            const float lr_old = logf(m_old / c_old);
            mv = lr_new - lr_old;
        }
        if (!isnan(mv)) { sum += (double)mv; cnt += 1; }
    }

    if (cnt == 0) {
        for (int t = ind_warm; t < rows; ++t) { out_indicator_tm[t * stride + s] = nanf; }
        for (int t = sig_warm; t < rows; ++t) { out_signal_tm[t * stride + s] = nanf; }
        return;
    }

    double ema_ind = (sum / (double)cnt) * 100.0;
    out_indicator_tm[ind_warm * stride + s] = (float)ema_ind;


    double ema_sig = 0.0; bool sig_seeded = false;
    double acc_sig = ema_ind; int cnt_sig = 1;
    if (sig_warm == ind_warm) {
        ema_sig = (acc_sig / (double)cnt_sig);
        out_signal_tm[sig_warm * stride + s] = (float)ema_sig;
        sig_seeded = true;
    }

    for (int t = ind_warm + 1; t < rows; ++t) {
        const int i_new = t * stride + s;
        const int i_old = (t - lookback) * stride + s;
        const float m_new = main_tm[i_new];
        const float c_new = compare_tm[i_new];
        const float m_old = main_tm[i_old];
        const float c_old = compare_tm[i_old];
        float mv = nanf;
        if (!isnan(m_new) && !isnan(c_new) && !isnan(m_old) && !isnan(c_old) && c_new != 0.0f && c_old != 0.0f) {
            const float lr_new = logf(m_new / c_new);
            const float lr_old = logf(m_old / c_old);
            mv = lr_new - lr_old;
        }

        if (!isnan(mv)) {
            const double src100 = (double)mv * 100.0;
            ema_ind = ((src100 - ema_ind) * alpha_ind) + ema_ind;
        }
        out_indicator_tm[i_new] = (float)ema_ind;

        if (!sig_seeded) {
            if (t < sig_warm) { acc_sig += ema_ind; cnt_sig += 1; }
            else if (t == sig_warm) {
                ema_sig = (acc_sig / (double)cnt_sig);
                out_signal_tm[i_new] = (float)ema_sig; sig_seeded = true; continue;
            }
        } else {
            ema_sig = ((ema_ind - ema_sig) * alpha_sig) + ema_sig;
            out_signal_tm[i_new] = (float)ema_sig;
        }
    }
}
