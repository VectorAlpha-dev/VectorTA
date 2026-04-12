#include <cmath>
#include <cstdint>

extern "C" __global__ void impulse_macd_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* length_mas,
    const int* length_signals,
    int rows,
    int max_signal_length,
    double* signal_buf,
    double* out_md,
    double* out_hist,
    double* out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int length_ma = length_mas[row];
    int length_signal = length_signals[row];
    if (length_ma <= 0 || length_signal <= 0 || max_signal_length <= 0) {
        return;
    }

    const double nan = NAN;
    double* row_signal_buf =
        signal_buf + static_cast<size_t>(row) * static_cast<size_t>(max_signal_length);
    double* row_md = out_md + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_hist = out_hist + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    int hi_count = 0;
    double hi_sum = 0.0;
    double hi_value = nan;
    bool hi_ready = false;

    int lo_count = 0;
    double lo_sum = 0.0;
    double lo_value = nan;
    bool lo_ready = false;

    double ema_alpha = 2.0 / (static_cast<double>(length_ma) + 1.0);
    double ema1_value = 0.0;
    bool ema1_has = false;
    double ema2_value = 0.0;
    bool ema2_has = false;

    int signal_head = 0;
    int signal_len = 0;
    double signal_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        row_md[i] = nan;
        row_hist[i] = nan;
        row_signal[i] = nan;

        if (!(isfinite(h) && isfinite(l) && isfinite(c)) || h < l) {
            hi_count = 0;
            hi_sum = 0.0;
            hi_value = nan;
            hi_ready = false;
            lo_count = 0;
            lo_sum = 0.0;
            lo_value = nan;
            lo_ready = false;
            ema1_value = 0.0;
            ema1_has = false;
            ema2_value = 0.0;
            ema2_has = false;
            signal_head = 0;
            signal_len = 0;
            signal_sum = 0.0;
            continue;
        }

        double src = (h + l + c) / 3.0;

        if (length_ma == 1) {
            hi_value = h;
            hi_ready = true;
        } else if (!hi_ready) {
            hi_sum += h;
            hi_count += 1;
            if (hi_count == length_ma) {
                hi_value = hi_sum / static_cast<double>(length_ma);
                hi_ready = true;
            }
        } else {
            double p = static_cast<double>(length_ma);
            hi_value = (hi_value * (p - 1.0) + h) / p;
        }

        if (length_ma == 1) {
            lo_value = l;
            lo_ready = true;
        } else if (!lo_ready) {
            lo_sum += l;
            lo_count += 1;
            if (lo_count == length_ma) {
                lo_value = lo_sum / static_cast<double>(length_ma);
                lo_ready = true;
            }
        } else {
            double p = static_cast<double>(length_ma);
            lo_value = (lo_value * (p - 1.0) + l) / p;
        }

        double ema1 = ema1_has ? ema_alpha * src + (1.0 - ema_alpha) * ema1_value : src;
        ema1_value = ema1;
        ema1_has = true;

        double ema2 = ema2_has ? ema_alpha * ema1 + (1.0 - ema_alpha) * ema2_value : ema1;
        ema2_value = ema2;
        ema2_has = true;

        double mi = ema1 + (ema1 - ema2);
        double md = 0.0;
        if (hi_ready && lo_ready) {
            if (mi > hi_value) {
                md = mi - hi_value;
            } else if (mi < lo_value) {
                md = mi - lo_value;
            }
        }

        double signal_value = nan;
        if (length_signal == 1) {
            row_signal_buf[0] = md;
            signal_len = 1;
            signal_sum = md;
            signal_value = md;
        } else if (signal_len < length_signal) {
            row_signal_buf[signal_len] = md;
            signal_len += 1;
            signal_sum += md;
            if (signal_len == length_signal) {
                signal_value = signal_sum / static_cast<double>(length_signal);
            }
        } else {
            double old = row_signal_buf[signal_head];
            row_signal_buf[signal_head] = md;
            signal_head += 1;
            if (signal_head == length_signal) {
                signal_head = 0;
            }
            signal_sum += md - old;
            signal_value = signal_sum / static_cast<double>(length_signal);
        }

        row_md[i] = md;
        row_signal[i] = signal_value;
        row_hist[i] = isfinite(signal_value) ? (md - signal_value) : nan;
    }
}
