#include <cmath>
#include <cstddef>

namespace {
constexpr int HISTORY_LENGTH = 10;
constexpr int MAX_BARS_FORWARD = 10;
constexpr double PI_CONST = 3.14159265358979323846;
constexpr double FLOAT_TOL = 1e-12;

constexpr int SIGNAL_MODE_PREDICT_FILTER_CROSSES = 0;
constexpr int SIGNAL_MODE_PREDICT_MIDDLE_CROSSES = 1;
constexpr int SIGNAL_MODE_FILTER_MIDDLE_CROSSES = 2;

__device__ inline double signum_with_tol(double value) {
    if (value > FLOAT_TOL) {
        return 1.0;
    }
    if (value < -FLOAT_TOL) {
        return -1.0;
    }
    return 0.0;
}

__device__ inline double signal_state(
    int signal_mode,
    double prediction,
    double filter
) {
    double lhs = prediction;
    double rhs = filter;
    if (signal_mode == SIGNAL_MODE_PREDICT_MIDDLE_CROSSES) {
        rhs = 0.0;
    } else if (signal_mode == SIGNAL_MODE_FILTER_MIDDLE_CROSSES) {
        lhs = filter;
        rhs = 0.0;
    }
    return signum_with_tol(lhs - rhs);
}

__device__ inline double go_long(
    int signal_mode,
    double prev_prediction,
    double prev_filter,
    double prediction,
    double filter
) {
    double prev_lhs = prev_prediction;
    double prev_rhs = prev_filter;
    double lhs = prediction;
    double rhs = filter;

    if (signal_mode == SIGNAL_MODE_PREDICT_MIDDLE_CROSSES) {
        prev_rhs = 0.0;
        rhs = 0.0;
    } else if (signal_mode == SIGNAL_MODE_FILTER_MIDDLE_CROSSES) {
        prev_lhs = prev_filter;
        prev_rhs = 0.0;
        lhs = filter;
        rhs = 0.0;
    }

    return (prev_lhs <= prev_rhs && lhs > rhs) ? 1.0 : 0.0;
}

__device__ inline double go_short(
    int signal_mode,
    double prev_prediction,
    double prev_filter,
    double prediction,
    double filter
) {
    double prev_lhs = prev_prediction;
    double prev_rhs = prev_filter;
    double lhs = prediction;
    double rhs = filter;

    if (signal_mode == SIGNAL_MODE_PREDICT_MIDDLE_CROSSES) {
        prev_rhs = 0.0;
        rhs = 0.0;
    } else if (signal_mode == SIGNAL_MODE_FILTER_MIDDLE_CROSSES) {
        prev_lhs = prev_filter;
        prev_rhs = 0.0;
        lhs = filter;
        rhs = 0.0;
    }

    return (prev_lhs >= prev_rhs && lhs < rhs) ? 1.0 : 0.0;
}
}

extern "C" __global__ void ehlers_linear_extrapolation_predictor_batch_f64(
    const double* data,
    int len,
    const int* high_pass_lengths,
    const int* low_pass_lengths,
    const double* gains,
    const int* bars_forwards,
    const int* signal_modes,
    int rows,
    int max_low_pass_length,
    double* out_prediction,
    double* out_filter,
    double* out_state,
    double* out_go_long,
    double* out_go_short,
    double* hp_history
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int high_pass_length = high_pass_lengths[row];
    const int low_pass_length = low_pass_lengths[row];
    const double gain = gains[row];
    const int bars_forward = bars_forwards[row];
    const int signal_mode = signal_modes[row];

    double* row_prediction = out_prediction + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_filter = out_filter + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_go_long = out_go_long + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_go_short = out_go_short + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_hp_history =
        hp_history + static_cast<size_t>(row) * static_cast<size_t>(max_low_pass_length);

    for (int i = 0; i < len; ++i) {
        row_prediction[i] = NAN;
        row_filter[i] = NAN;
        row_state[i] = NAN;
        row_go_long[i] = NAN;
        row_go_short[i] = NAN;
    }

    if (high_pass_length <= 0 || low_pass_length <= 0 || low_pass_length > max_low_pass_length
        || !isfinite(gain) || bars_forward < 0 || bars_forward > MAX_BARS_FORWARD
        || signal_mode < SIGNAL_MODE_PREDICT_FILTER_CROSSES
        || signal_mode > SIGNAL_MODE_FILTER_MIDDLE_CROSSES) {
        return;
    }

    const double angle = 1.414 * PI_CONST / static_cast<double>(high_pass_length);
    const double a1 = exp(-angle);
    const double hp_c2 = 2.0 * a1 * cos(angle);
    const double hp_c3 = -a1 * a1;
    const double hp_c1 = (1.0 + hp_c2 - hp_c3) * 0.25;
    const double pix2 = 2.0 * PI_CONST / static_cast<double>(low_pass_length + 1);

    double hann_weight_sum = 0.0;
    for (int count = 1; count <= low_pass_length; ++count) {
        hann_weight_sum += 1.0 - cos(static_cast<double>(count) * pix2);
    }

    int source_count = 0;
    double prev_source_1 = 0.0;
    double prev_source_2 = 0.0;
    double hp_prev_1 = 0.0;
    double hp_prev_2 = 0.0;
    int hp_count = 0;
    double filter_history_local[HISTORY_LENGTH];
    int filter_count = 0;
    double prev_prediction = 0.0;
    double prev_filter = 0.0;
    bool has_prev_signal = false;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];

        if (!isfinite(value)) {
            source_count = 0;
            prev_source_1 = 0.0;
            prev_source_2 = 0.0;
            hp_prev_1 = 0.0;
            hp_prev_2 = 0.0;
            hp_count = 0;
            filter_count = 0;
            prev_prediction = 0.0;
            prev_filter = 0.0;
            has_prev_signal = false;
            continue;
        }

        source_count += 1;
        const double hp =
            source_count <= 4
                ? 0.0
                : hp_c1 * (value - 2.0 * prev_source_1 + prev_source_2) + hp_c2 * hp_prev_1
                      + hp_c3 * hp_prev_2;

        prev_source_2 = prev_source_1;
        prev_source_1 = value;
        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp;

        if (hp_count < low_pass_length) {
            for (int j = hp_count; j > 0; --j) {
                row_hp_history[j] = row_hp_history[j - 1];
            }
            row_hp_history[0] = hp;
            hp_count += 1;
        } else {
            for (int j = low_pass_length - 1; j > 0; --j) {
                row_hp_history[j] = row_hp_history[j - 1];
            }
            row_hp_history[0] = hp;
        }

        if (source_count < 4 + low_pass_length - 1 || hp_count < low_pass_length) {
            continue;
        }

        double filter = 0.0;
        for (int count = 1; count <= low_pass_length; ++count) {
            const double coef = 1.0 - cos(static_cast<double>(count) * pix2);
            filter += coef * row_hp_history[count - 1];
        }
        filter /= hann_weight_sum;

        if (filter_count < HISTORY_LENGTH) {
            filter_history_local[filter_count] = filter;
            filter_count += 1;
        } else {
            for (int j = 0; j < HISTORY_LENGTH - 1; ++j) {
                filter_history_local[j] = filter_history_local[j + 1];
            }
            filter_history_local[HISTORY_LENGTH - 1] = filter;
        }

        if (filter_count < HISTORY_LENGTH) {
            continue;
        }

        const double current = filter_history_local[HISTORY_LENGTH - 1];
        const double prev = filter_history_local[HISTORY_LENGTH - 2];
        const double prediction = bars_forward == 0
                                      ? current * gain
                                      : (current + static_cast<double>(bars_forward)
                                                       * (current - prev))
                                            * gain;
        const double state = signal_state(signal_mode, prediction, filter);
        const double go_long_value =
            has_prev_signal ? go_long(signal_mode, prev_prediction, prev_filter, prediction, filter)
                            : 0.0;
        const double go_short_value =
            has_prev_signal ? go_short(signal_mode, prev_prediction, prev_filter, prediction, filter)
                            : 0.0;

        prev_prediction = prediction;
        prev_filter = filter;
        has_prev_signal = true;

        row_prediction[i] = prediction;
        row_filter[i] = filter;
        row_state[i] = state;
        row_go_long[i] = go_long_value;
        row_go_short[i] = go_short_value;
    }
}
