#include <cmath>
#include <cstddef>

extern "C" __global__ void price_density_market_noise_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ eval_periods,
    int rows,
    int max_length,
    int max_eval_period,
    double* __restrict__ high_ring_buf,
    double* __restrict__ low_ring_buf,
    double* __restrict__ tr_ring_buf,
    double* __restrict__ pd_ring_buf,
    double* __restrict__ out_price_density,
    double* __restrict__ out_price_density_percent
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    int eval_period = eval_periods[row];
    double* high_ring =
        high_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* low_ring =
        low_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* tr_ring =
        tr_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* pd_ring =
        pd_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_eval_period);
    double* row_out_price_density =
        out_price_density + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_price_density_percent =
        out_price_density_percent + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_price_density[i] = NAN;
        row_out_price_density_percent[i] = NAN;
    }

    if (length <= 0 ||
        eval_period <= 0 ||
        length > max_length ||
        eval_period > max_eval_period) {
        return;
    }

    int window_head = 0;
    int window_count = 0;
    int pd_head = 0;
    int pd_count = 0;
    bool have_prev_close = false;
    double prev_close = NAN;
    double tr_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            window_head = 0;
            window_count = 0;
            pd_head = 0;
            pd_count = 0;
            have_prev_close = false;
            prev_close = NAN;
            tr_sum = 0.0;
            continue;
        }

        double tr = h - l;
        if (have_prev_close) {
            double high_close = fabs(h - prev_close);
            double low_close = fabs(l - prev_close);
            if (high_close > tr) {
                tr = high_close;
            }
            if (low_close > tr) {
                tr = low_close;
            }
        }
        prev_close = c;
        have_prev_close = true;

        if (window_count < length) {
            high_ring[window_count] = h;
            low_ring[window_count] = l;
            tr_ring[window_count] = tr;
            tr_sum += tr;
            window_count += 1;
        } else {
            tr_sum -= tr_ring[window_head];
            high_ring[window_head] = h;
            low_ring[window_head] = l;
            tr_ring[window_head] = tr;
            tr_sum += tr;
            window_head += 1;
            if (window_head == length) {
                window_head = 0;
            }
        }

        if (window_count < length) {
            continue;
        }

        double highest = -INFINITY;
        double lowest = INFINITY;
        for (int j = 0; j < length; ++j) {
            int idx = (window_head + j) % length;
            double high_value = high_ring[idx];
            double low_value = low_ring[idx];
            if (high_value > highest) {
                highest = high_value;
            }
            if (low_value < lowest) {
                lowest = low_value;
            }
        }

        double denom = highest - lowest;
        double price_density = denom > 0.0 ? tr_sum / denom : NAN;
        row_out_price_density[i] = price_density;

        if (pd_count < eval_period) {
            pd_ring[pd_count] = price_density;
            pd_count += 1;
        } else {
            pd_ring[pd_head] = price_density;
            pd_head += 1;
            if (pd_head == eval_period) {
                pd_head = 0;
            }
        }

        if (pd_count < eval_period || !isfinite(price_density)) {
            continue;
        }

        bool invalid = false;
        int rank = 0;
        for (int j = 0; j < eval_period; ++j) {
            int idx = (pd_head + j) % eval_period;
            double value = pd_ring[idx];
            if (!isfinite(value)) {
                invalid = true;
                break;
            }
            if (value <= price_density) {
                rank += 1;
            }
        }

        if (!invalid) {
            row_out_price_density_percent[i] =
                (static_cast<double>(rank) / static_cast<double>(eval_period)) * 100.0;
        }
    }
}
