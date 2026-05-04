#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float f32_nan() {
    return __int_as_float(0x7fffffff);
}

extern "C" __global__ void vwmacd_build_prefix_one_series_f64(
    const float* __restrict__ prices,
    const float* __restrict__ volumes,
    int len,
    int first_valid,
    double* __restrict__ pv_prefix,
    double* __restrict__ vol_prefix)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    double acc_pv = 0.0;
    double acc_vol = 0.0;
    for (int i = 0; i < len; ++i) {
        if (i >= first_valid) {
            const double p = (double)prices[i];
            const double v = (double)volumes[i];
            if (isnan(p) || isnan(v) || isnan(acc_pv) || isnan(acc_vol)) {
                acc_pv = nan("");
                acc_vol = nan("");
            } else {
                acc_pv += p * v;
                acc_vol += v;
            }
            pv_prefix[i] = acc_pv;
            vol_prefix[i] = acc_vol;
        } else {
            pv_prefix[i] = 0.0;
            vol_prefix[i] = 0.0;
        }
    }
}


extern "C" __global__ void vwmacd_build_prefix_time_major_f64(
    const float* __restrict__ prices_tm,
    const float* __restrict__ volumes_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    double* __restrict__ pv_prefix_tm,
    double* __restrict__ vol_prefix_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols || rows < 0) return;

    const int first_valid = first_valids[series];
    double acc_pv = 0.0;
    double acc_vol = 0.0;

    for (int r = 0; r < rows; ++r) {
        const int idx = r * cols + series;
        if (r >= first_valid) {
            const double p = (double)prices_tm[idx];
            const double v = (double)volumes_tm[idx];
            if (isnan(p) || isnan(v) || isnan(acc_pv) || isnan(acc_vol)) {
                acc_pv = nan("");
                acc_vol = nan("");
            } else {
                acc_pv += p * v;
                acc_vol += v;
            }
            pv_prefix_tm[idx] = acc_pv;
            vol_prefix_tm[idx] = acc_vol;
        } else {
            pv_prefix_tm[idx] = 0.0;
            vol_prefix_tm[idx] = 0.0;
        }
    }
}


extern "C" __global__ void vwmacd_batch_f32(
    const double* __restrict__ pv_prefix,
    const double* __restrict__ vol_prefix,
    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    const int* __restrict__ sigs,
    int len,
    int first_valid,
    int n_rows,
    float* __restrict__ out_macd,
    float* __restrict__ out_signal,
    float* __restrict__ out_hist)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int f = fasts[row];
    const int s = slows[row];
    const int g = sigs[row];

    const int warm_macd = first_valid + (f > s ? f : s) - 1;
    const int warm_hist = warm_macd + g - 1;

    const int base = row * len;


    for (int i = 0; i < len; ++i) {
        out_macd[base + i] = f32_nan();
        out_signal[base + i] = f32_nan();
        out_hist[base + i] = f32_nan();
    }


    for (int t = warm_macd; t < len; ++t) {
        const int prev_f = t - f;
        const int prev_s = t - s;

        double sum_pv_f = pv_prefix[t];
        double sum_v_f  = vol_prefix[t];
        if (prev_f >= 0) {
            sum_pv_f -= pv_prefix[prev_f];
            sum_v_f  -= vol_prefix[prev_f];
        }

        double sum_pv_s = pv_prefix[t];
        double sum_v_s  = vol_prefix[t];
        if (prev_s >= 0) {
            sum_pv_s -= pv_prefix[prev_s];
            sum_v_s  -= vol_prefix[prev_s];
        }

        float macd_val = f32_nan();
        if (!isnan(sum_v_f) && !isnan(sum_v_s) && sum_v_f != 0.0 && sum_v_s != 0.0) {
            const double fast_vwma = sum_pv_f / sum_v_f;
            const double slow_vwma = sum_pv_s / sum_v_s;
            macd_val = (float)(fast_vwma - slow_vwma);
        }
        out_macd[base + t] = macd_val;
    }


    if (warm_macd < len) {
        const float alpha = 2.0f / (float)(g + 1);
        const float beta  = 1.0f - alpha;
        const int start = warm_macd;
        const int warm_end = min(start + g, len);

        if (start < len) {
            float mean = out_macd[base + start];
            out_signal[base + start] = mean;
            int count = 1;
            for (int i = start + 1; i < warm_end; ++i) {
                const float x = out_macd[base + i];

                const double m = ((double)(count) * (double)mean + (double)x) / (double)(count + 1);
                mean = (float)m;
                out_signal[base + i] = mean;
                ++count;
            }

            float prev = mean;
            for (int i = warm_end; i < len; ++i) {
                const float x = out_macd[base + i];
                prev = beta * prev + alpha * x;
                out_signal[base + i] = prev;
            }
        }
    }


    for (int i = 0; i < min(warm_hist, len); ++i) {
        out_signal[base + i] = f32_nan();
        out_hist[base + i] = f32_nan();
    }
    for (int i = warm_hist; i < len; ++i) {
        const float m = out_macd[base + i];
        const float sgn = out_signal[base + i];
        out_hist[base + i] = (!isnan(m) && !isnan(sgn)) ? (m - sgn) : f32_nan();
    }
}


extern "C" __global__ void vwmacd_batch_macd_tiled_f32(
    const double* __restrict__ pv_prefix,
    const double* __restrict__ vol_prefix,
    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    int len,
    int first_valid,
    int n_rows,
    float* __restrict__ out_macd)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= n_rows || t >= len) return;

    const int f = fasts[row];
    const int s = slows[row];
    const int warm_macd = first_valid + (f > s ? f : s) - 1;
    const int base = row * len;

    if (t < warm_macd) {
        out_macd[base + t] = f32_nan();
        return;
    }

    const int prev_f = t - f;
    const int prev_s = t - s;

    double sum_pv_f = pv_prefix[t];
    double sum_v_f  = vol_prefix[t];
    if (prev_f >= 0) {
        sum_pv_f -= pv_prefix[prev_f];
        sum_v_f  -= vol_prefix[prev_f];
    }

    double sum_pv_s = pv_prefix[t];
    double sum_v_s  = vol_prefix[t];
    if (prev_s >= 0) {
        sum_pv_s -= pv_prefix[prev_s];
        sum_v_s  -= vol_prefix[prev_s];
    }

    float macd_val = f32_nan();
    if (!isnan(sum_v_f) && !isnan(sum_v_s) && sum_v_f != 0.0 && sum_v_s != 0.0) {
        const double fast_vwma = sum_pv_f / sum_v_f;
        const double slow_vwma = sum_pv_s / sum_v_s;
        macd_val = (float)(fast_vwma - slow_vwma);
    }
    out_macd[base + t] = macd_val;
}


extern "C" __global__ void vwmacd_batch_signal_serial_f32(
    const float* __restrict__ out_macd,
    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    const int* __restrict__ sigs,
    int len,
    int first_valid,
    int n_rows,
    float* __restrict__ out_signal)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int f = fasts[row];
    const int s = slows[row];
    const int g = sigs[row];
    const int warm_macd = first_valid + (f > s ? f : s) - 1;
    const int warm_hist = warm_macd + g - 1;
    const int base = row * len;

    if (warm_macd >= len) return;

    const float alpha = 2.0f / (float)(g + 1);
    const float beta  = 1.0f - alpha;
    const int start = warm_macd;
    const int warm_end = min(start + g, len);

    float mean = out_macd[base + start];
    int count = 1;
    for (int i = start + 1; i < warm_end; ++i) {
        const float x = out_macd[base + i];
        const double m = ((double)(count) * (double)mean + (double)x) / (double)(count + 1);
        mean = (float)m;
        ++count;
    }

    if (warm_hist < len) {
        out_signal[base + warm_hist] = mean;
    }

    float prev = mean;
    for (int i = warm_end; i < len; ++i) {
        const float x = out_macd[base + i];
        prev = beta * prev + alpha * x;
        out_signal[base + i] = prev;
    }
}


extern "C" __global__ void vwmacd_batch_hist_tiled_f32(
    const float* __restrict__ out_macd,
    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    const int* __restrict__ sigs,
    int len,
    int first_valid,
    int n_rows,
    float* __restrict__ out_signal,
    float* __restrict__ out_hist)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= n_rows || t >= len) return;

    const int f = fasts[row];
    const int s = slows[row];
    const int g = sigs[row];
    const int warm_macd = first_valid + (f > s ? f : s) - 1;
    const int warm_hist = warm_macd + g - 1;
    const int base = row * len;

    if (t < warm_hist) {
        out_signal[base + t] = f32_nan();
        out_hist[base + t] = f32_nan();
        return;
    }

    const float m = out_macd[base + t];
    const float sgn = out_signal[base + t];
    out_hist[base + t] = (!isnan(m) && !isnan(sgn)) ? (m - sgn) : f32_nan();
}


extern "C" __global__ void vwmacd_many_series_one_param_time_major_f32(
    const double* __restrict__ pv_prefix_tm,
    const double* __restrict__ vol_prefix_tm,
    const int*    __restrict__ first_valids,
    int fast,
    int slow,
    int signal,
    int cols,
    int rows,
    float* __restrict__ out_macd_tm,
    float* __restrict__ out_signal_tm,
    float* __restrict__ out_hist_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    const int fv = first_valids[series];
    const int warm_macd = fv + (fast > slow ? fast : slow) - 1;
    const int warm_hist = warm_macd + signal - 1;


    for (int r = 0; r < rows; ++r) {
        const int idx = r * cols + series;
        out_macd_tm[idx] = f32_nan();
        out_signal_tm[idx] = f32_nan();
        out_hist_tm[idx] = f32_nan();
    }


    for (int r = warm_macd; r < rows; ++r) {
        const int prev_f = r - fast;
        const int prev_s = r - slow;
        const int idx = r * cols + series;

        double sum_pv_f = pv_prefix_tm[idx];
        double sum_v_f  = vol_prefix_tm[idx];
        if (prev_f >= 0) {
            const int pidx = prev_f * cols + series;
            sum_pv_f -= pv_prefix_tm[pidx];
            sum_v_f  -= vol_prefix_tm[pidx];
        }
        double sum_pv_s = pv_prefix_tm[idx];
        double sum_v_s  = vol_prefix_tm[idx];
        if (prev_s >= 0) {
            const int pidx = prev_s * cols + series;
            sum_pv_s -= pv_prefix_tm[pidx];
            sum_v_s  -= vol_prefix_tm[pidx];
        }

        float macd_val = f32_nan();
        if (!isnan(sum_v_f) && !isnan(sum_v_s) && sum_v_f != 0.0 && sum_v_s != 0.0) {
            const double fast_vwma = sum_pv_f / sum_v_f;
            const double slow_vwma = sum_pv_s / sum_v_s;
            macd_val = (float)(fast_vwma - slow_vwma);
        }
        out_macd_tm[idx] = macd_val;
    }


    if (warm_macd < rows) {
        const float alpha = 2.0f / (float)(signal + 1);
        const float beta  = 1.0f - alpha;
        const int start = warm_macd;
        const int warm_end = min(start + signal, rows);
        if (start < rows) {
            float mean = out_macd_tm[start * cols + series];
            out_signal_tm[start * cols + series] = mean;
            int count = 1;
            for (int r = start + 1; r < warm_end; ++r) {
                const float x = out_macd_tm[r * cols + series];
                const double m = ((double)(count) * (double)mean + (double)x) / (double)(count + 1);
                mean = (float)m;
                out_signal_tm[r * cols + series] = mean;
                ++count;
            }
            float prev = mean;
            for (int r = warm_end; r < rows; ++r) {
                const float x = out_macd_tm[r * cols + series];
                prev = beta * prev + alpha * x;
                out_signal_tm[r * cols + series] = prev;
            }
        }
    }


    for (int r = 0; r < min(warm_hist, rows); ++r) {
        out_signal_tm[r * cols + series] = f32_nan();
        out_hist_tm[r * cols + series] = f32_nan();
    }
    for (int r = warm_hist; r < rows; ++r) {
        const int idx = r * cols + series;
        const float m = out_macd_tm[idx];
        const float s = out_signal_tm[idx];
        out_hist_tm[idx] = (!isnan(m) && !isnan(s)) ? (m - s) : f32_nan();
    }
}
