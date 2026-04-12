#include <cmath>
#include <cstddef>

extern "C" __global__ void premier_rsi_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ rsi_lengths,
    const int* __restrict__ stoch_lengths,
    const int* __restrict__ smooth_lengths,
    int rows,
    double* __restrict__ out_values
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int rsi_length = rsi_lengths[row];
    int stoch_length = stoch_lengths[row];
    int smooth_length = smooth_lengths[row];
    double* row_values = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_values[i] = NAN;
    }

    if (rsi_length <= 0 || stoch_length <= 0 || smooth_length <= 0) {
        return;
    }

    int ema_length = static_cast<int>(floor(sqrt(static_cast<double>(smooth_length)) + 0.5));
    if (ema_length < 1) {
        ema_length = 1;
    }
    double ema_alpha = 2.0 / (static_cast<double>(ema_length) + 1.0);

    double* stoch_window = new double[stoch_length];
    if (stoch_window == nullptr) {
        return;
    }

    bool has_prev = false;
    double prev = NAN;
    int seed_count = 0;
    double sum_gain = 0.0;
    double sum_loss = 0.0;
    bool seeded = false;
    double avg_gain = 0.0;
    double avg_loss = 0.0;

    int stoch_count = 0;
    int stoch_head = 0;

    bool has_ema1 = false;
    bool has_ema2 = false;
    double ema1 = NAN;
    double ema2 = NAN;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            has_prev = false;
            prev = NAN;
            seed_count = 0;
            sum_gain = 0.0;
            sum_loss = 0.0;
            seeded = false;
            avg_gain = 0.0;
            avg_loss = 0.0;
            stoch_count = 0;
            stoch_head = 0;
            has_ema1 = false;
            has_ema2 = false;
            ema1 = NAN;
            ema2 = NAN;
            continue;
        }

        if (!has_prev) {
            prev = value;
            has_prev = true;
            continue;
        }

        double delta = value - prev;
        prev = value;

        bool rsi_ready = false;
        double rsi = NAN;
        if (!seeded) {
            double gain = delta > 0.0 ? delta : 0.0;
            double loss = delta < 0.0 ? -delta : 0.0;
            sum_gain += gain;
            sum_loss += loss;
            seed_count += 1;
            if (seed_count == rsi_length) {
                seeded = true;
                avg_gain = sum_gain / static_cast<double>(rsi_length);
                avg_loss = sum_loss / static_cast<double>(rsi_length);
                double denom = avg_gain + avg_loss;
                rsi = denom == 0.0 ? 50.0 : 100.0 * avg_gain / denom;
                rsi_ready = true;
            }
        } else {
            double gain = delta > 0.0 ? delta : 0.0;
            double loss = delta < 0.0 ? -delta : 0.0;
            double inv_p = 1.0 / static_cast<double>(rsi_length);
            double beta = 1.0 - inv_p;
            avg_gain = avg_gain * beta + inv_p * gain;
            avg_loss = avg_loss * beta + inv_p * loss;
            double denom = avg_gain + avg_loss;
            rsi = denom == 0.0 ? 50.0 : 100.0 * avg_gain / denom;
            rsi_ready = true;
        }

        if (!rsi_ready) {
            continue;
        }

        if (stoch_count < stoch_length) {
            stoch_window[(stoch_head + stoch_count) % stoch_length] = rsi;
            stoch_count += 1;
        } else {
            stoch_window[stoch_head] = rsi;
            stoch_head += 1;
            if (stoch_head == stoch_length) {
                stoch_head = 0;
            }
        }

        if (stoch_count < stoch_length) {
            continue;
        }

        double highest = stoch_window[0];
        double lowest = stoch_window[0];
        for (int j = 1; j < stoch_count; ++j) {
            double sample = stoch_window[j];
            if (sample > highest) {
                highest = sample;
            }
            if (sample < lowest) {
                lowest = sample;
            }
        }

        double denom = highest - lowest;
        double sk = fabs(denom) <= 1.0e-12 ? 50.0 : (rsi - lowest) * (100.0 / denom);
        double nsk = 0.1 * (sk - 50.0);

        ema1 = has_ema1 ? ema_alpha * nsk + (1.0 - ema_alpha) * ema1 : nsk;
        ema2 = has_ema2 ? ema_alpha * ema1 + (1.0 - ema_alpha) * ema2 : ema1;
        has_ema1 = true;
        has_ema2 = true;

        row_values[i] = tanh(ema2 * 0.5);
    }

    delete[] stoch_window;
}
