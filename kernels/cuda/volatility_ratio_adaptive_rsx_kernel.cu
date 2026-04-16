#include <cmath>
#include <cstddef>

namespace {
__device__ inline bool is_valid_source(double value) {
    return isfinite(value);
}

__device__ inline double nz(double value) {
    return isfinite(value) ? value : 0.0;
}

__device__ inline double biased_std_from_sums(double sum, double sum_sq, int period) {
    const double n = static_cast<double>(period);
    const double centered = fmax(sum_sq - (sum * sum) / n, 0.0);
    return sqrt(centered / n);
}

__device__ inline void push_window_sum_sumsq(
    double* window,
    int window_len,
    int* head,
    int* count,
    int* valid,
    double* sum,
    double* sum_sq,
    double value
) {
    if (*count == window_len) {
        const double old = window[*head];
        if (isfinite(old)) {
            *valid -= 1;
            *sum -= old;
            *sum_sq -= old * old;
        }
    } else {
        *count += 1;
    }

    window[*head] = value;
    *head += 1;
    if (*head == window_len) {
        *head = 0;
    }

    if (isfinite(value)) {
        *valid += 1;
        *sum += value;
        *sum_sq += value * value;
    }
}

__device__ inline void push_window_sum(
    double* window,
    int window_len,
    int* head,
    int* count,
    int* valid,
    double* sum,
    double value
) {
    if (*count == window_len) {
        const double old = window[*head];
        if (isfinite(old)) {
            *valid -= 1;
            *sum -= old;
        }
    } else {
        *count += 1;
    }

    window[*head] = value;
    *head += 1;
    if (*head == window_len) {
        *head = 0;
    }

    if (isfinite(value)) {
        *valid += 1;
        *sum += value;
    }
}
}

extern "C" __global__ void volatility_ratio_adaptive_rsx_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    const double* __restrict__ speeds,
    int rows,
    double* __restrict__ out_line,
    double* __restrict__ out_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int period = periods[row];
    const double speed = speeds[row];

    double* row_line = out_line + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_line[i] = NAN;
        row_signal[i] = NAN;
    }

    if (period <= 0 || !isfinite(speed) || speed < 0.0 || speed > 1.0) {
        return;
    }

    double* price_window = new double[period];
    double* dev_window = new double[period];
    if (price_window == nullptr || dev_window == nullptr) {
        delete[] price_window;
        delete[] dev_window;
        return;
    }
    for (int i = 0; i < period; ++i) {
        price_window[i] = NAN;
        dev_window[i] = NAN;
    }

    double prev_src_out = NAN;
    double prev_line = NAN;
    int price_head = 0;
    int price_count = 0;
    int price_valid = 0;
    double price_sum = 0.0;
    double price_sum_sq = 0.0;
    int dev_head = 0;
    int dev_count = 0;
    int dev_valid = 0;
    double dev_sum = 0.0;
    double f28 = NAN;
    double f30 = NAN;
    double f38 = NAN;
    double f40 = NAN;
    double f48 = NAN;
    double f50 = NAN;
    double f58 = NAN;
    double f60 = NAN;
    double f68 = NAN;
    double f70 = NAN;
    double f78 = NAN;
    double f80 = NAN;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        const double src_out = is_valid_source(value) ? 100.0 * value : NAN;

        push_window_sum_sumsq(
            price_window,
            period,
            &price_head,
            &price_count,
            &price_valid,
            &price_sum,
            &price_sum_sq,
            value
        );

        const double dev = (price_count == period && price_valid == period)
            ? biased_std_from_sums(price_sum, price_sum_sq, period)
            : NAN;

        push_window_sum(
            dev_window,
            period,
            &dev_head,
            &dev_count,
            &dev_valid,
            &dev_sum,
            dev
        );

        const double devavg = (dev_count == period && dev_valid == period)
            ? dev_sum / static_cast<double>(period)
            : NAN;

        const double vol_ratio =
            isfinite(dev) && isfinite(devavg) && devavg != 0.0 ? dev / devavg : NAN;
        const double adaptive_len =
            isfinite(vol_ratio) && vol_ratio > 0.0
                ? trunc(static_cast<double>(period) / vol_ratio)
                : NAN;
        const double kg = isfinite(adaptive_len) ? 3.0 / (adaptive_len + 2.0) : NAN;
        const double hg = isfinite(kg) ? 1.0 - kg : NAN;

        const double mom0 =
            isfinite(src_out) && isfinite(prev_src_out) ? src_out - prev_src_out : NAN;
        const double moa0 = isfinite(mom0) ? fabs(mom0) : NAN;
        const double spdp1 = speed + 1.0;

        f28 = isfinite(kg) && isfinite(hg) && isfinite(mom0) ? kg * mom0 + hg * nz(f28) : NAN;
        f30 = isfinite(kg) && isfinite(hg) && isfinite(f28) ? hg * nz(f30) + kg * f28 : NAN;
        const double mom1 = isfinite(f28) && isfinite(f30) ? f28 * spdp1 - f30 * speed : NAN;

        f38 = isfinite(kg) && isfinite(hg) && isfinite(mom1) ? hg * nz(f38) + kg * mom1 : NAN;
        f40 = isfinite(kg) && isfinite(hg) && isfinite(f38) ? kg * f38 + hg * nz(f40) : NAN;
        const double mom2 = isfinite(f38) && isfinite(f40) ? f38 * spdp1 - f40 * speed : NAN;

        f48 = isfinite(kg) && isfinite(hg) && isfinite(mom2) ? hg * nz(f48) + kg * mom2 : NAN;
        f50 = isfinite(kg) && isfinite(hg) && isfinite(f48) ? kg * f48 + hg * nz(f50) : NAN;
        const double mom_out = isfinite(f48) && isfinite(f50) ? f48 * spdp1 - f50 * speed : NAN;

        f58 = isfinite(kg) && isfinite(hg) && isfinite(moa0) ? hg * nz(f58) + kg * moa0 : NAN;
        f60 = isfinite(kg) && isfinite(hg) && isfinite(f58) ? kg * f58 + hg * nz(f60) : NAN;
        const double moa1 = isfinite(f58) && isfinite(f60) ? f58 * spdp1 - f60 * speed : NAN;

        f68 = isfinite(kg) && isfinite(hg) && isfinite(moa1) ? hg * nz(f68) + kg * moa1 : NAN;
        f70 = isfinite(kg) && isfinite(hg) && isfinite(f68) ? kg * f68 + hg * nz(f70) : NAN;
        const double moa2 = isfinite(f68) && isfinite(f70) ? f68 * spdp1 - f70 * speed : NAN;

        f78 = isfinite(kg) && isfinite(hg) && isfinite(moa2) ? hg * nz(f78) + kg * moa2 : NAN;
        f80 = isfinite(kg) && isfinite(hg) && isfinite(f78) ? kg * f78 + hg * nz(f80) : NAN;
        const double moa_out = isfinite(f78) && isfinite(f80) ? f78 * spdp1 - f80 * speed : NAN;

        const double line = isfinite(mom_out) && isfinite(moa_out) && moa_out != 0.0
            ? fmin(fmax((mom_out / moa_out + 1.0) * 50.0, 0.0), 100.0)
            : NAN;
        const double signal = prev_line;

        row_line[i] = line;
        row_signal[i] = signal;

        prev_src_out = src_out;
        prev_line = line;
    }

    delete[] price_window;
    delete[] dev_window;
}
