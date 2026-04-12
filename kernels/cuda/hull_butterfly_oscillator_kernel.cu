#include <cmath>
#include <cstdint>

__device__ __forceinline__ bool crossed(double prev_a, double curr_a, double prev_b, double curr_b) {
    return (curr_a > curr_b && prev_a <= prev_b) || (curr_a < curr_b && prev_a >= prev_b);
}

extern "C" __global__ void hull_butterfly_oscillator_batch_f64(
    const double* data,
    int len,
    const int* coeff_lens,
    const double* mults,
    const double* coeffs,
    int max_coeff_len,
    int rows,
    double* out_oscillator,
    double* out_cumulative_mean,
    double* out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int coeff_len = coeff_lens[row];
    double mult = mults[row];
    if (coeff_len < 2 || coeff_len > max_coeff_len || !isfinite(mult)) {
        return;
    }

    const double nan = NAN;
    const double* row_coeffs = coeffs + static_cast<size_t>(row) * static_cast<size_t>(max_coeff_len);
    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_cumulative_mean =
        out_cumulative_mean + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    int segment_index = 0;
    double cumulative_abs = 0.0;
    double prev_hso = 0.0;
    double prev_cmean = 0.0;
    double signal_state = 0.0;
    bool has_prev = false;

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = nan;
        row_cumulative_mean[i] = nan;
        row_signal[i] = nan;

        double value = data[i];
        if (!isfinite(value)) {
            segment_index = 0;
            cumulative_abs = 0.0;
            prev_hso = 0.0;
            prev_cmean = 0.0;
            signal_state = 0.0;
            has_prev = false;
            continue;
        }

        int current_index = segment_index;
        segment_index += 1;
        if (segment_index < coeff_len) {
            continue;
        }

        int window_start = i - coeff_len + 1;
        double hma = 0.0;
        double inv_hma = 0.0;
        for (int j = 0; j < coeff_len; ++j) {
            double coeff = row_coeffs[j];
            hma += data[i - j] * coeff;
            inv_hma += data[window_start + j] * coeff;
        }

        double hso = hma - inv_hma;
        cumulative_abs += fabs(hso);
        if (current_index == 0) {
            continue;
        }

        double cmean = cumulative_abs / static_cast<double>(current_index) * mult;
        if (has_prev) {
            if (crossed(prev_hso, hso, prev_cmean, cmean)
                || crossed(prev_hso, hso, -prev_cmean, -cmean)) {
                signal_state = 0.0;
            } else if (hso < prev_hso && hso > cmean) {
                signal_state = -1.0;
            } else if (hso > prev_hso && hso < -cmean) {
                signal_state = 1.0;
            }
        }

        prev_hso = hso;
        prev_cmean = cmean;
        has_prev = true;
        row_oscillator[i] = hso;
        row_cumulative_mean[i] = cmean;
        row_signal[i] = signal_state;
    }
}
