#include <cmath>
#include <cstddef>

extern "C" __global__ void andean_oscillator_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ signal_lengths,
    int rows,
    double* __restrict__ out_bull,
    double* __restrict__ out_bear,
    double* __restrict__ out_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const int signal_length = signal_lengths[row];

    double* row_bull = out_bull + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bear = out_bear + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_bull[i] = NAN;
        row_bear[i] = NAN;
        row_signal[i] = NAN;
    }

    if (length <= 0 || signal_length <= 0) {
        return;
    }

    const double alpha = 2.0 / (static_cast<double>(length) + 1.0);
    const double signal_alpha = 2.0 / (static_cast<double>(signal_length) + 1.0);

    bool initialized = false;
    double up1 = NAN;
    double up2 = NAN;
    double dn1 = NAN;
    double dn2 = NAN;
    double signal = NAN;

    for (int i = 0; i < len; ++i) {
        const double open_i = open[i];
        const double close_i = close[i];
        if (!isfinite(open_i) || !isfinite(close_i)) {
            continue;
        }

        const double close_sq = close_i * close_i;
        const double open_sq = open_i * open_i;

        if (!initialized) {
            up1 = close_i;
            up2 = close_sq;
            dn1 = close_i;
            dn2 = close_sq;
            signal = 0.0;
            initialized = true;
            row_bull[i] = 0.0;
            row_bear[i] = 0.0;
            row_signal[i] = 0.0;
            continue;
        }

        const double up1_next = up1 - (up1 - close_i) * alpha;
        const double up2_next = up2 - (up2 - close_sq) * alpha;
        const double dn1_next = dn1 + (close_i - dn1) * alpha;
        const double dn2_next = dn2 + (close_sq - dn2) * alpha;

        up1 = fmax(close_i, fmax(open_i, up1_next));
        up2 = fmax(close_sq, fmax(open_sq, up2_next));
        dn1 = fmin(close_i, fmin(open_i, dn1_next));
        dn2 = fmin(close_sq, fmin(open_sq, dn2_next));

        const double bull = sqrt(fmax(dn2 - dn1 * dn1, 0.0));
        const double bear = sqrt(fmax(up2 - up1 * up1, 0.0));
        const double signal_input = fmax(bull, bear);
        signal = isfinite(signal)
            ? signal_alpha * signal_input + (1.0 - signal_alpha) * signal
            : signal_input;

        row_bull[i] = bull;
        row_bear[i] = bear;
        row_signal[i] = signal;
    }
}
