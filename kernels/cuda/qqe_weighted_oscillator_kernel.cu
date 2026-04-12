#include <cmath>
#include <cstddef>

extern "C" __global__ void qqe_weighted_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const double* __restrict__ factors,
    const int* __restrict__ smooths,
    const double* __restrict__ weights,
    int rows,
    double* __restrict__ out_rsi,
    double* __restrict__ out_trailing_stop
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const double factor = factors[row];
    const int smooth = smooths[row];
    const double weight = weights[row];

    double* row_rsi = out_rsi + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ts = out_trailing_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_rsi[i] = NAN;
        row_ts[i] = NAN;
    }

    if (length <= 0 || length > len || smooth <= 0 || !isfinite(factor) || factor < 0.0 ||
        !isfinite(weight)) {
        return;
    }

    int first = -1;
    for (int i = 0; i < len; ++i) {
        if (isfinite(data[i])) {
            first = i;
            break;
        }
    }
    if (first < 0 || first + 1 >= len) {
        return;
    }

    const double ratio_alpha = 2.0 / (static_cast<double>(smooth) + 1.0);

    int num_count = 0;
    int den_count = 0;
    int diff_count = 0;
    double num_sum = 0.0;
    double den_sum = 0.0;
    double diff_sum = 0.0;
    double num_value = NAN;
    double den_value = NAN;
    double diff_value = NAN;
    bool num_seeded = false;
    bool den_seeded = false;
    bool diff_seeded = false;

    bool ratio_seeded = false;
    double ratio_value = NAN;

    bool has_prev_src = true;
    double prev_src = data[first];
    bool has_prev_rsi = false;
    bool has_prev_ts = false;
    double prev_rsi = NAN;
    double prev_ts = NAN;

    for (int i = first + 1; i < len; ++i) {
        const double current = data[i];
        if (!isfinite(current)) {
            has_prev_src = false;
            continue;
        }

        if (!has_prev_src) {
            prev_src = current;
            has_prev_src = true;
            continue;
        }

        const double delta = current - prev_src;
        prev_src = current;

        const double scale = (has_prev_rsi && has_prev_ts &&
                              delta * (prev_rsi - prev_ts) > 0.0)
            ? weight
            : 1.0;
        const double weighted_delta = delta * scale;

        bool num_ready = false;
        if (num_seeded) {
            num_value =
                (num_value * (static_cast<double>(length) - 1.0) + weighted_delta) /
                static_cast<double>(length);
            num_ready = true;
        } else {
            num_sum += weighted_delta;
            num_count += 1;
            if (num_count == length) {
                num_value = num_sum / static_cast<double>(length);
                num_seeded = true;
                num_ready = true;
            }
        }

        const double abs_delta = fabs(weighted_delta);
        bool den_ready = false;
        if (den_seeded) {
            den_value =
                (den_value * (static_cast<double>(length) - 1.0) + abs_delta) /
                static_cast<double>(length);
            den_ready = true;
        } else {
            den_sum += abs_delta;
            den_count += 1;
            if (den_count == length) {
                den_value = den_sum / static_cast<double>(length);
                den_seeded = true;
                den_ready = true;
            }
        }

        if (!num_ready || !den_ready || den_value == 0.0) {
            continue;
        }

        const double ratio_input = num_value / den_value;
        ratio_value = ratio_seeded
            ? ratio_alpha * ratio_input + (1.0 - ratio_alpha) * ratio_value
            : ratio_input;
        ratio_seeded = true;

        const double rsi = 50.0 * ratio_value + 50.0;
        row_rsi[i] = rsi;

        bool diff_ready = false;
        if (has_prev_rsi) {
            const double diff_input = fabs(rsi - prev_rsi);
            if (diff_seeded) {
                diff_value =
                    (diff_value * (static_cast<double>(length) - 1.0) + diff_input) /
                    static_cast<double>(length);
                diff_ready = true;
            } else {
                diff_sum += diff_input;
                diff_count += 1;
                if (diff_count == length) {
                    diff_value = diff_sum / static_cast<double>(length);
                    diff_seeded = true;
                    diff_ready = true;
                }
            }
        }

        double trailing_stop = rsi;
        if (diff_ready) {
            const bool crossover =
                has_prev_ts && rsi > prev_ts && prev_rsi <= prev_ts;
            const bool crossunder =
                has_prev_ts && rsi < prev_ts && prev_rsi >= prev_ts;
            if (crossover) {
                trailing_stop = rsi - diff_value * factor;
            } else if (crossunder) {
                trailing_stop = rsi + diff_value * factor;
            } else if (has_prev_ts) {
                if (rsi > prev_ts) {
                    trailing_stop = fmax(rsi - diff_value * factor, prev_ts);
                } else {
                    trailing_stop = fmin(rsi + diff_value * factor, prev_ts);
                }
            }
        }

        row_ts[i] = trailing_stop;
        prev_rsi = rsi;
        prev_ts = trailing_stop;
        has_prev_rsi = true;
        has_prev_ts = true;
    }
}
