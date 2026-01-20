


















#include <cuda_runtime.h>
#include <math.h>

static __device__ inline float f32_nan() {
    return __int_as_float(0x7fffffff);
}


static __device__ inline int window_has_nan(const int* __restrict__ pref_nan, int t1, int t0) {
    return (pref_nan[t1] - pref_nan[t0]) != 0;
}

static __device__ inline double window_sum(const double* __restrict__ pref, int t1, int t0) {
    return pref[t1] - pref[t0];
}




extern "C" __global__ void macz_batch_macz_tmp_f32(

    const float* __restrict__ close,
    const float* __restrict__ volume,
    const double* __restrict__ pref_close_sum,
    const double* __restrict__ pref_close_sumsq,
    const int* __restrict__ pref_close_nan,
    const double* __restrict__ pref_vol_sum,
    const double* __restrict__ pref_pv_sum,
    const int* __restrict__ pref_vol_nan,

    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    const int* __restrict__ lzs,
    const int* __restrict__ lsds,
    const float* __restrict__ a_s,
    const float* __restrict__ b_s,

    int len,
    int first_valid,
    int n_rows,
    int use_sma_for_vwap,

    float* __restrict__ macz_tmp
) {
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)blockIdx.y;
    if (row >= n_rows || t >= len) return;

    const int f = fasts[row];
    const int s = slows[row];
    const int lz = lzs[row];
    const int lsd = lsds[row];
    const float a = a_s[row];
    const float b = b_s[row];

    const int warm_m = first_valid + max(max(s, lz), lsd) - 1;
    const int row_off = row * len;
    if (t < warm_m) {
        macz_tmp[row_off + t] = f32_nan();
        return;
    }


    double mean_vwap = NAN;
    {
        const int t1 = t + 1;
        const int t0 = t + 1 - lz;
        if (!use_sma_for_vwap && volume != nullptr) {

            if (!window_has_nan(pref_close_nan, t1, t0) && !window_has_nan(pref_vol_nan, t1, t0)) {
                const double vol_sum = window_sum(pref_vol_sum, t1, t0);
                if (vol_sum > 0.0) {
                    const double pv_sum = window_sum(pref_pv_sum, t1, t0);
                    mean_vwap = pv_sum / vol_sum;
                }
            }
        } else {

            if (!window_has_nan(pref_close_nan, t1, t0)) {
                const double ssum = window_sum(pref_close_sum, t1, t0);
                mean_vwap = ssum / (double)lz;
            }
        }
    }


    double z = NAN;
    if (!isnan(mean_vwap)) {
        const int t1 = t + 1;
        const int t0 = t + 1 - lz;
        if (!window_has_nan(pref_close_nan, t1, t0)) {
            const double ssum = window_sum(pref_close_sum, t1, t0);
            const double ssum2 = window_sum(pref_close_sumsq, t1, t0);
            const double e = ssum / (double)lz;
            const double e2 = ssum2 / (double)lz;
            double var = fma(-2.0 * mean_vwap, e, e2) + (mean_vwap * mean_vwap);
            if (var > 0.0) {
                const double std = sqrt(var);
                const double x = (double)close[t];
                z = (x - mean_vwap) / std;
            } else {
                z = 0.0;
            }
        }
    }


    double macd = NAN;
    {
        const int t1s = t + 1;
        const int t0s = t + 1 - s;
        const int t1f = t + 1;
        const int t0f = t + 1 - f;
        if (!window_has_nan(pref_close_nan, t1s, t0s) && !window_has_nan(pref_close_nan, t1f, t0f)) {
            const double slow_mean = window_sum(pref_close_sum, t1s, t0s) / (double)s;
            const double fast_mean = window_sum(pref_close_sum, t1f, t0f) / (double)f;
            macd = fast_mean - slow_mean;
        }
    }


    double sd = NAN;
    {
        const int t1d = t + 1;
        const int t0d = t + 1 - lsd;
        if (!window_has_nan(pref_close_nan, t1d, t0d)) {
            const double mean = window_sum(pref_close_sum, t1d, t0d) / (double)lsd;
            const double s2 = window_sum(pref_close_sumsq, t1d, t0d) / (double)lsd;
            const double var = s2 - mean * mean;
            if (var > 0.0) sd = sqrt(var);
        }
    }

    float macz_raw = f32_nan();
    if (!isnan(z) && !isnan(macd) && !isnan(sd) && sd > 0.0) {
        const double val = (double)z * (double)a + ((double)macd / (double)sd) * (double)b;
        macz_raw = (float)val;
    }

    macz_tmp[row_off + t] = macz_raw;
}


extern "C" __global__ void macz_batch_hist_from_macz_f32(

    const float* __restrict__ macz_tmp,

    const int* __restrict__ slows,
    const int* __restrict__ sigs,
    const int* __restrict__ lzs,
    const int* __restrict__ lsds,

    int len,
    int first_valid,
    int n_rows,

    float* __restrict__ out_hist
) {
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)blockIdx.y;
    if (row >= n_rows || t >= len) return;

    const int s = slows[row];
    const int g = sigs[row];
    const int lz = lzs[row];
    const int lsd = lsds[row];
    const int warm_m = first_valid + max(max(s, lz), lsd) - 1;
    const int warm_hist = warm_m + g - 1;

    const int row_off = row * len;
    if (t < warm_hist) {
        out_hist[row_off + t] = f32_nan();
        return;
    }


    double sum = 0.0;
    bool any_nan = false;
    const int start = t + 1 - g;
    for (int j = start; j <= t; ++j) {
        const float mv = macz_tmp[row_off + j];
        if (isnan(mv)) { any_nan = true; break; }
        sum += (double)mv;
    }
    if (any_nan) {
        out_hist[row_off + t] = f32_nan();
    } else {
        const float signal = (float)(sum / (double)g);
        out_hist[row_off + t] = macz_tmp[row_off + t] - signal;
    }
}



extern "C" __global__ void macz_batch_f32(

    const float* __restrict__ close,
    const float* __restrict__ volume,
    const double* __restrict__ pref_close_sum,
    const double* __restrict__ pref_close_sumsq,
    const int* __restrict__ pref_close_nan,
    const double* __restrict__ pref_vol_sum,
    const double* __restrict__ pref_pv_sum,
    const int* __restrict__ pref_vol_nan,

    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    const int* __restrict__ sigs,
    const int* __restrict__ lzs,
    const int* __restrict__ lsds,
    const float* __restrict__ a_s,
    const float* __restrict__ b_s,
    const int* __restrict__ use_lag_s,
    const float* __restrict__ gammas,

    int len,
    int first_valid,
    int n_rows,
    int use_sma_for_vwap,

    float* __restrict__ macz_tmp,
    float* __restrict__ out_hist
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int f = fasts[row];
    const int s = slows[row];
    const int g = sigs[row];
    const int lz = lzs[row];
    const int lsd = lsds[row];
    const float a = a_s[row];
    const float b = b_s[row];
    const int use_lag = use_lag_s[row] != 0;
    const double gamma = (double)gammas[row];

    const int warm_m = first_valid + max(max(s, lz), lsd) - 1;
    const int warm_hist = warm_m + g - 1;
    const int row_off = row * len;


    for (int i = 0; i < len; ++i) {
        macz_tmp[row_off + i] = f32_nan();
        out_hist[row_off + i] = f32_nan();
    }


    double l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0;

    for (int t = warm_m; t < len; ++t) {

        double mean_vwap = NAN;
        if (t >= first_valid + lz - 1) {
            const int t1 = t + 1;
            const int t0 = t + 1 - lz;
            if (!use_sma_for_vwap && volume != nullptr) {

                if (!window_has_nan(pref_close_nan, t1, t0) && !window_has_nan(pref_vol_nan, t1, t0)) {
                    const double vol_sum = window_sum(pref_vol_sum, t1, t0);
                    if (vol_sum > 0.0) {
                        const double pv_sum = window_sum(pref_pv_sum, t1, t0);
                        mean_vwap = pv_sum / vol_sum;
                    }
                }
            } else {

                if (!window_has_nan(pref_close_nan, t1, t0)) {
                    const double ssum = window_sum(pref_close_sum, t1, t0);
                    mean_vwap = ssum / (double)lz;
                }
            }
        }


        double z = NAN;
        if (!isnan(mean_vwap)) {
            const int t1 = t + 1;
            const int t0 = t + 1 - lz;
            if (!window_has_nan(pref_close_nan, t1, t0)) {
                const double ssum = window_sum(pref_close_sum, t1, t0);
                const double ssum2 = window_sum(pref_close_sumsq, t1, t0);
                const double e = ssum / (double)lz;
                const double e2 = ssum2 / (double)lz;
                double var = fma(-2.0 * mean_vwap, e, e2) + (mean_vwap * mean_vwap);
                if (var > 0.0) {
                    const double std = sqrt(var);
                    const double x = (double)close[t];
                    z = (x - mean_vwap) / std;
                } else {
                    z = 0.0;
                }
            }
        }


        double macd = NAN;
        if (t >= first_valid + s - 1) {
            const int t1s = t + 1;
            const int t0s = t + 1 - s;
            const int t1f = t + 1;
            const int t0f = t + 1 - f;
            if (!window_has_nan(pref_close_nan, t1s, t0s) && !window_has_nan(pref_close_nan, t1f, t0f)) {
                const double slow_mean = window_sum(pref_close_sum, t1s, t0s) / (double)s;
                const double fast_mean = window_sum(pref_close_sum, t1f, t0f) / (double)f;
                macd = fast_mean - slow_mean;
            }
        }


        double sd = NAN;
        if (t >= first_valid + lsd - 1) {
            const int t1d = t + 1;
            const int t0d = t + 1 - lsd;
            if (!window_has_nan(pref_close_nan, t1d, t0d)) {
                const double mean = window_sum(pref_close_sum, t1d, t0d) / (double)lsd;
                const double s2 = window_sum(pref_close_sumsq, t1d, t0d) / (double)lsd;
                const double var = s2 - mean * mean;
                if (var > 0.0) sd = sqrt(var);
            }
        }

        float macz_raw = f32_nan();
        if (!isnan(z) && !isnan(macd) && !isnan(sd) && sd > 0.0) {
            const double val = (double)z * (double)a + ((double)macd / (double)sd) * (double)b;
            macz_raw = (float)val;
        }

        float macz_val = macz_raw;
        if (use_lag) {
            if (isnan(macz_raw)) {
                macz_val = f32_nan();
            } else {
                const double s_in = (double)macz_raw;
                const double one_minus_g = 1.0 - gamma;
                const double new_l0 = one_minus_g * s_in + gamma * l0;
                const double new_l1 = -gamma * new_l0 + l0 + gamma * l1;
                const double new_l2 = -gamma * new_l1 + l1 + gamma * l2;
                const double new_l3 = -gamma * new_l2 + l2 + gamma * l3;
                l0 = new_l0; l1 = new_l1; l2 = new_l2; l3 = new_l3;
                const double outv = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0;
                macz_val = (float)outv;
            }
        }

        macz_tmp[row_off + t] = macz_val;



        if (t >= warm_hist) {

            double sum = 0.0;
            bool any_nan = false;
            const int start = t + 1 - g;
            for (int j = start; j <= t; ++j) {
                const float mv = macz_tmp[row_off + j];
                if (isnan(mv)) { any_nan = true; break; }
                sum += (double)mv;
            }
            if (!any_nan) {
                const float signal = (float)(sum / (double)g);
                const float hv = macz_val - signal;
                out_hist[row_off + t] = hv;
            }
        }
    }
}

extern "C" __global__ void macz_many_series_one_param_time_major_f32(

    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const double* __restrict__ pref_close_sum_tm,
    const double* __restrict__ pref_close_sumsq_tm,
    const int* __restrict__ pref_close_nan_tm,
    const double* __restrict__ pref_vol_sum_tm,
    const double* __restrict__ pref_pv_sum_tm,
    const int* __restrict__ pref_vol_nan_tm,
    int cols,
    int rows,

    int fast,
    int slow,
    int sig,
    int lz,
    int lsd,
    float a,
    float b,
    int use_lag,
    float gamma_f,
    const int* __restrict__ first_valids,
    int use_sma_for_vwap,

    float* __restrict__ macz_tm,
    float* __restrict__ hist_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    const int off_pref = s * (rows + 1);

    const double* pcs = pref_close_sum_tm + off_pref;
    const double* pcsq = pref_close_sumsq_tm + off_pref;
    const int* pcn = pref_close_nan_tm + off_pref;
    const double* pvs = pref_vol_sum_tm ? (pref_vol_sum_tm + off_pref) : nullptr;
    const double* pps = pref_pv_sum_tm ? (pref_pv_sum_tm + off_pref) : nullptr;
    const int* pvn = pref_vol_nan_tm ? (pref_vol_nan_tm + off_pref) : nullptr;

    const int fv = first_valids[s];
    if (fv < 0) return;
    const int warm_m = fv + max(max(slow, lz), lsd) - 1;
    const int warm_hist = warm_m + sig - 1;

    auto at = [&](int t) { return t * cols + s; };
    for (int t = 0; t < rows; ++t) { macz_tm[at(t)] = f32_nan(); hist_tm[at(t)] = f32_nan(); }

    double l0=0.0,l1=0.0,l2=0.0,l3=0.0;
    const double gamma = (double)gamma_f;

    for (int t = warm_m; t < rows; ++t) {

        double mean_vwap = NAN;
        if (t >= fv + lz - 1) {
            const int t1 = t + 1;
            const int t0 = t + 1 - lz;
            if (!use_sma_for_vwap && volume_tm) {
                if (!window_has_nan(pcn, t1, t0) && !window_has_nan(pvn, t1, t0)) {
                    const double vs = window_sum(pvs, t1, t0);
                    if (vs > 0.0) {
                        const double pv = window_sum(pps, t1, t0);
                        mean_vwap = pv / vs;
                    }
                }
            } else {
                if (!window_has_nan(pcn, t1, t0)) {
                    mean_vwap = window_sum(pcs, t1, t0) / (double)lz;
                }
            }
        }


        double z = NAN;
        if (!isnan(mean_vwap)) {
            const int t1 = t + 1, t0 = t + 1 - lz;
            if (!window_has_nan(pcn, t1, t0)) {
                const double s2 = window_sum(pcsq, t1, t0) / (double)lz;
                const double s1 = window_sum(pcs, t1, t0) / (double)lz;
                const double var = fma(-2.0 * mean_vwap, s1, s2) + (mean_vwap * mean_vwap);
                if (var > 0.0) {
                    const double std = sqrt(var);
                    const double x = (double)close_tm[at(t)];
                    z = (x - mean_vwap) / std;
                } else {
                    z = 0.0;
                }
            }
        }


        double macd = NAN;
        if (t >= fv + slow - 1) {
            const int t1s = t + 1, t0s = t + 1 - slow;
            const int t1f = t + 1, t0f = t + 1 - fast;
            if (!window_has_nan(pcn, t1s, t0s) && !window_has_nan(pcn, t1f, t0f)) {
                const double slow_m = window_sum(pcs, t1s, t0s) / (double)slow;
                const double fast_m = window_sum(pcs, t1f, t0f) / (double)fast;
                macd = fast_m - slow_m;
            }
        }


        double sd = NAN;
        if (t >= fv + lsd - 1) {
            const int t1d = t + 1, t0d = t + 1 - lsd;
            if (!window_has_nan(pcn, t1d, t0d)) {
                const double mean = window_sum(pcs, t1d, t0d) / (double)lsd;
                const double s2 = window_sum(pcsq, t1d, t0d) / (double)lsd;
                const double var = s2 - mean * mean;
                if (var > 0.0) sd = sqrt(var);
            }
        }

        float macz_raw = f32_nan();
        if (!isnan(z) && !isnan(macd) && !isnan(sd) && sd > 0.0) {
            const double val = (double)z * (double)a + ((double)macd / (double)sd) * (double)b;
            macz_raw = (float)val;
        }

        float macz_val = macz_raw;
        if (use_lag) {
            if (isnan(macz_raw)) {
                macz_val = f32_nan();
            } else {
                const double s_in = (double)macz_raw;
                const double one_minus_g = 1.0 - gamma;
                const double new_l0 = one_minus_g * s_in + gamma * l0;
                const double new_l1 = -gamma * new_l0 + l0 + gamma * l1;
                const double new_l2 = -gamma * new_l1 + l1 + gamma * l2;
                const double new_l3 = -gamma * new_l2 + l2 + gamma * l3;
                l0 = new_l0; l1 = new_l1; l2 = new_l2; l3 = new_l3;
                const double outv = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0;
                macz_val = (float)outv;
            }
        }

        macz_tm[at(t)] = macz_val;

        if (t >= warm_hist) {
            double sum = 0.0; bool any_nan = false;
            const int start = t + 1 - sig;
            for (int j = start; j <= t; ++j) {
                const float mv = macz_tm[at(j)];
                if (isnan(mv)) { any_nan = true; break; }
                sum += (double)mv;
            }
            if (!any_nan) {
                const float signal = (float)(sum / (double)sig);
                hist_tm[at(t)] = macz_val - signal;
            }
        }
    }
}
