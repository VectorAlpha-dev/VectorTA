#include <cmath>
#include <cstddef>

namespace {
constexpr double INTRADAY_MI_PI = 3.14159265358979323846264338327950288;
constexpr double INTRADAY_MI_SQRT_2 = 1.41421356237309504880168872420969808;

__device__ inline double intraday_push_bb(
    double value,
    double* bb_values,
    unsigned char* bb_valid,
    int length_bb,
    int* bb_idx,
    int* bb_count,
    int* bb_valid_count,
    double* bb_sum,
    double* bb_sumsq
) {
    if (*bb_count >= length_bb) {
        const int old = *bb_idx;
        if (bb_valid[old] != 0) {
            *bb_valid_count -= 1;
            const double old_value = bb_values[old];
            *bb_sum -= old_value;
            *bb_sumsq -= old_value * old_value;
        }
    } else {
        *bb_count += 1;
    }

    if (isfinite(value)) {
        bb_values[*bb_idx] = value;
        bb_valid[*bb_idx] = 1;
        *bb_valid_count += 1;
        *bb_sum += value;
        *bb_sumsq += value * value;
    } else {
        bb_values[*bb_idx] = 0.0;
        bb_valid[*bb_idx] = 0;
    }

    *bb_idx += 1;
    if (*bb_idx == length_bb) {
        *bb_idx = 0;
    }

    if (*bb_count < length_bb || *bb_valid_count != length_bb) {
        return NAN;
    }

    const double n = static_cast<double>(length_bb);
    const double mean = *bb_sum / n;
    const double variance = fmax(*bb_sumsq / n - mean * mean, 0.0);
    return sqrt(variance);
}
}

extern "C" __global__ void intraday_momentum_index_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ length_mas,
    const double* __restrict__ mults,
    const int* __restrict__ length_bbs,
    const int* __restrict__ apply_smoothings,
    const int* __restrict__ low_bands,
    int rows,
    double* __restrict__ out_imi,
    double* __restrict__ out_upper_hit,
    double* __restrict__ out_lower_hit,
    double* __restrict__ out_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const int length_ma = length_mas[row];
    const double mult = mults[row];
    const int length_bb = length_bbs[row];
    const bool apply_smoothing = apply_smoothings[row] != 0;
    const int low_band = low_bands[row];

    double* row_imi = out_imi + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper_hit + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower_hit + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_imi[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
        row_signal[i] = NAN;
    }

    if (length <= 0 || length > len || length_ma <= 0 || length_ma > len ||
        length_bb <= 0 || length_bb > len || !isfinite(mult) || mult < 0.0 ||
        (apply_smoothing && low_band == 0)) {
        return;
    }

    double* gains = new double[length];
    double* losses = new double[length];
    unsigned char* valid = new unsigned char[length];
    double* bb_values = new double[length_bb];
    unsigned char* bb_valid = new unsigned char[length_bb];
    if (gains == nullptr || losses == nullptr || valid == nullptr ||
        bb_values == nullptr || bb_valid == nullptr) {
        delete[] gains;
        delete[] losses;
        delete[] valid;
        delete[] bb_values;
        delete[] bb_valid;
        return;
    }

    for (int i = 0; i < length; ++i) {
        gains[i] = 0.0;
        losses[i] = 0.0;
        valid[i] = 0;
    }
    for (int i = 0; i < length_bb; ++i) {
        bb_values[i] = 0.0;
        bb_valid[i] = 0;
    }

    const double signal_alpha = 2.0 / (static_cast<double>(length_ma) + 1.0);
    const double basis_alpha = 2.0 / (static_cast<double>(length_bb) + 1.0);

    double coeff1 = 0.0;
    double coeff2 = 0.0;
    double coeff3 = 0.0;
    if (apply_smoothing) {
        const double band = static_cast<double>(low_band);
        const double a1 = exp(-INTRADAY_MI_PI * INTRADAY_MI_SQRT_2 / band);
        coeff2 = 2.0 * a1 * cos(INTRADAY_MI_SQRT_2 * INTRADAY_MI_PI / band);
        coeff3 = -(a1 * a1);
        coeff1 = 1.0 - coeff2 - coeff3;
    }

    int idx = 0;
    int count = 0;
    int valid_count = 0;
    double sum_gain = 0.0;
    double sum_loss = 0.0;

    bool signal_seeded = false;
    bool basis_seeded = false;
    double signal_value = NAN;
    double basis_value = NAN;

    double prev_price = 0.0;
    double prev_filt1 = 0.0;
    double prev_filt2 = 0.0;

    int bb_idx = 0;
    int bb_count = 0;
    int bb_valid_count = 0;
    double bb_sum = 0.0;
    double bb_sumsq = 0.0;

    for (int i = 0; i < len; ++i) {
        const bool valid_bar = isfinite(open[i]) && isfinite(close[i]);

        if (count >= length) {
            const int old = idx;
            if (valid[old] != 0) {
                valid_count -= 1;
                sum_gain -= gains[old];
                sum_loss -= losses[old];
            }
        } else {
            count += 1;
        }

        if (valid_bar) {
            const double diff = close[i] - open[i];
            const double gain = fmax(diff, 0.0);
            const double loss = fmax(-diff, 0.0);
            gains[idx] = gain;
            losses[idx] = loss;
            valid[idx] = 1;
            valid_count += 1;
            sum_gain += gain;
            sum_loss += loss;
        } else {
            gains[idx] = 0.0;
            losses[idx] = 0.0;
            valid[idx] = 0;
        }

        idx += 1;
        if (idx == length) {
            idx = 0;
        }

        double imi = NAN;
        if (count >= length && valid_count == length) {
            const double denom = sum_gain + sum_loss;
            if (denom > 0.0 && isfinite(denom)) {
                const double raw_imi = 100.0 * (sum_gain / denom);
                if (apply_smoothing) {
                    const double filt =
                        coeff1 * (raw_imi + prev_price) * 0.5 +
                        coeff2 * prev_filt1 +
                        coeff3 * prev_filt2;
                    prev_price = raw_imi;
                    prev_filt2 = prev_filt1;
                    prev_filt1 = filt;
                    imi = filt;
                } else {
                    imi = raw_imi;
                }
            }
        }

        if (!isfinite(imi)) {
            if (apply_smoothing) {
                prev_price = 0.0;
                prev_filt1 = 0.0;
                prev_filt2 = 0.0;
            }
            intraday_push_bb(
                NAN,
                bb_values,
                bb_valid,
                length_bb,
                &bb_idx,
                &bb_count,
                &bb_valid_count,
                &bb_sum,
                &bb_sumsq
            );
            continue;
        }

        signal_value = signal_seeded
            ? fma(signal_alpha, imi, (1.0 - signal_alpha) * signal_value)
            : imi;
        basis_value = basis_seeded
            ? fma(basis_alpha, imi, (1.0 - basis_alpha) * basis_value)
            : imi;
        signal_seeded = true;
        basis_seeded = true;

        const double dev = intraday_push_bb(
            imi,
            bb_values,
            bb_valid,
            length_bb,
            &bb_idx,
            &bb_count,
            &bb_valid_count,
            &bb_sum,
            &bb_sumsq
        );

        double upper_hit = NAN;
        double lower_hit = NAN;
        if (isfinite(dev)) {
            const double upper = basis_value + mult * dev;
            const double lower = basis_value - mult * dev;
            if (imi >= upper) {
                upper_hit = imi;
            }
            if (imi <= lower) {
                lower_hit = imi;
            }
        }

        row_imi[i] = imi;
        row_upper[i] = upper_hit;
        row_lower[i] = lower_hit;
        row_signal[i] = signal_value;
    }

    delete[] gains;
    delete[] losses;
    delete[] valid;
    delete[] bb_values;
    delete[] bb_valid;
}
