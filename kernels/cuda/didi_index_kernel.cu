#include <cmath>
#include <cstddef>

extern "C" __global__ void didi_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ short_lengths,
    const int* __restrict__ medium_lengths,
    const int* __restrict__ long_lengths,
    int rows,
    double* __restrict__ out_short,
    double* __restrict__ out_long,
    double* __restrict__ out_crossover,
    double* __restrict__ out_crossunder
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int short_length = short_lengths[row];
    int medium_length = medium_lengths[row];
    int long_length = long_lengths[row];

    double* row_out_short = out_short + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_long = out_long + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_crossover =
        out_crossover + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_crossunder =
        out_crossunder + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_short[i] = NAN;
        row_out_long[i] = NAN;
        row_out_crossover[i] = NAN;
        row_out_crossunder[i] = NAN;
    }

    if (short_length <= 0 || medium_length <= 0 || long_length <= 0) {
        return;
    }

    int needed = short_length;
    if (medium_length > needed) {
        needed = medium_length;
    }
    if (long_length > needed) {
        needed = long_length;
    }

    int run_length = 0;
    bool have_prev = false;
    double prev_short = NAN;
    double prev_long = NAN;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            run_length = 0;
            have_prev = false;
            prev_short = NAN;
            prev_long = NAN;
            continue;
        }

        run_length += 1;
        if (run_length < needed) {
            have_prev = false;
            continue;
        }

        double short_sum = 0.0;
        for (int j = i + 1 - short_length; j <= i; ++j) {
            short_sum += data[j];
        }

        double medium_sum = 0.0;
        for (int j = i + 1 - medium_length; j <= i; ++j) {
            medium_sum += data[j];
        }

        double long_sum = 0.0;
        for (int j = i + 1 - long_length; j <= i; ++j) {
            long_sum += data[j];
        }

        double medium_ma = medium_sum / static_cast<double>(medium_length);
        if (!isfinite(medium_ma) || medium_ma == 0.0) {
            have_prev = false;
            prev_short = NAN;
            prev_long = NAN;
            continue;
        }

        double short_value = (short_sum / static_cast<double>(short_length)) / medium_ma;
        double long_value = (long_sum / static_cast<double>(long_length)) / medium_ma;
        if (!isfinite(short_value) || !isfinite(long_value)) {
            have_prev = false;
            prev_short = NAN;
            prev_long = NAN;
            continue;
        }

        double crossover =
            (have_prev && short_value > long_value && prev_short <= prev_long) ? 1.0 : 0.0;
        double crossunder =
            (have_prev && short_value < long_value && prev_short >= prev_long) ? 1.0 : 0.0;

        row_out_short[i] = short_value;
        row_out_long[i] = long_value;
        row_out_crossover[i] = crossover;
        row_out_crossunder[i] = crossunder;

        prev_short = short_value;
        prev_long = long_value;
        have_prev = true;
    }
}
