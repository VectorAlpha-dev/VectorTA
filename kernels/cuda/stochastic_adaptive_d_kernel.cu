#include <cmath>
#include <cstddef>

static __device__ inline bool sad_valid_bar(double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close) && high >= low;
}

static __device__ inline bool sad_sma_update(
    double value,
    double* buffer,
    int period,
    int* count,
    int* head,
    double* sum,
    double* out
) {
    if (*count < period) {
        buffer[(*head + *count) % period] = value;
        *sum += value;
        *count += 1;
    } else {
        *sum -= buffer[*head];
        buffer[*head] = value;
        *sum += value;
        *head += 1;
        if (*head == period) {
            *head = 0;
        }
    }

    if (*count == period) {
        *out = *sum / static_cast<double>(period);
        return true;
    }
    return false;
}

extern "C" __global__ void stochastic_adaptive_d_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ k_lengths,
    const int* __restrict__ d_smoothings,
    const int* __restrict__ pre_smooths,
    const double* __restrict__ attenuations,
    int rows,
    double* __restrict__ out_standard_d,
    double* __restrict__ out_adaptive_d,
    double* __restrict__ out_difference
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int k_length = k_lengths[row];
    int d_smoothing = d_smoothings[row];
    int pre_smooth = pre_smooths[row];
    double attenuation = attenuations[row];

    double* row_standard = out_standard_d + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_adaptive = out_adaptive_d + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_difference = out_difference + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_standard[i] = NAN;
        row_adaptive[i] = NAN;
        row_difference[i] = NAN;
    }

    if (k_length <= 0 || d_smoothing <= 0 || pre_smooth <= 0 || !isfinite(attenuation) ||
        attenuation < 0.1) {
        return;
    }

    double* pre_high_buf = new double[pre_smooth];
    double* pre_low_buf = new double[pre_smooth];
    double* pre_close_buf = new double[pre_smooth];
    double* stoch_high_buf = new double[k_length];
    double* stoch_low_buf = new double[k_length];
    double* d_buf = new double[d_smoothing];
    if (pre_high_buf == nullptr || pre_low_buf == nullptr || pre_close_buf == nullptr ||
        stoch_high_buf == nullptr || stoch_low_buf == nullptr || d_buf == nullptr) {
        delete[] pre_high_buf;
        delete[] pre_low_buf;
        delete[] pre_close_buf;
        delete[] stoch_high_buf;
        delete[] stoch_low_buf;
        delete[] d_buf;
        return;
    }

    int pre_high_count = 0;
    int pre_low_count = 0;
    int pre_close_count = 0;
    int pre_high_head = 0;
    int pre_low_head = 0;
    int pre_close_head = 0;
    double pre_high_sum = 0.0;
    double pre_low_sum = 0.0;
    double pre_close_sum = 0.0;

    int stoch_count = 0;
    int stoch_head = 0;

    int d_count = 0;
    int d_head = 0;
    double d_sum = 0.0;

    double adaptive = 50.0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];

        if (!sad_valid_bar(h, l, c)) {
            pre_high_count = 0;
            pre_low_count = 0;
            pre_close_count = 0;
            pre_high_head = 0;
            pre_low_head = 0;
            pre_close_head = 0;
            pre_high_sum = 0.0;
            pre_low_sum = 0.0;
            pre_close_sum = 0.0;
            stoch_count = 0;
            stoch_head = 0;
            d_count = 0;
            d_head = 0;
            d_sum = 0.0;
            adaptive = 50.0;
            continue;
        }

        double s_high = NAN;
        double s_low = NAN;
        double s_close = NAN;
        if (!sad_sma_update(
                h,
                pre_high_buf,
                pre_smooth,
                &pre_high_count,
                &pre_high_head,
                &pre_high_sum,
                &s_high
            ) ||
            !sad_sma_update(
                l,
                pre_low_buf,
                pre_smooth,
                &pre_low_count,
                &pre_low_head,
                &pre_low_sum,
                &s_low
            ) ||
            !sad_sma_update(
                c,
                pre_close_buf,
                pre_smooth,
                &pre_close_count,
                &pre_close_head,
                &pre_close_sum,
                &s_close
            )) {
            continue;
        }

        if (stoch_count < k_length) {
            stoch_high_buf[(stoch_head + stoch_count) % k_length] = s_high;
            stoch_low_buf[(stoch_head + stoch_count) % k_length] = s_low;
            stoch_count += 1;
        } else {
            stoch_high_buf[stoch_head] = s_high;
            stoch_low_buf[stoch_head] = s_low;
            stoch_head += 1;
            if (stoch_head == k_length) {
                stoch_head = 0;
            }
        }

        if (stoch_count < k_length) {
            continue;
        }

        double highest = stoch_high_buf[0];
        double lowest = stoch_low_buf[0];
        for (int j = 1; j < stoch_count; ++j) {
            if (stoch_high_buf[j] > highest) {
                highest = stoch_high_buf[j];
            }
            if (stoch_low_buf[j] < lowest) {
                lowest = stoch_low_buf[j];
            }
        }

        double range = highest - lowest;
        double stoch_raw = fabs(range) <= 1.0e-12 ? 50.0 : (s_close - lowest) * (100.0 / range);

        double stoch_d_raw = NAN;
        if (!sad_sma_update(
                stoch_raw,
                d_buf,
                d_smoothing,
                &d_count,
                &d_head,
                &d_sum,
                &stoch_d_raw
            )) {
            continue;
        }

        double standard_d = 50.0 + (stoch_d_raw - 50.0) * 0.5;
        double alpha = (fabs(standard_d - 50.0) / 100.0) / attenuation;
        double src_ama = (standard_d - 50.0) / attenuation + 50.0;
        adaptive = adaptive + alpha * (src_ama - adaptive);
        double difference = 50.0 + (standard_d - adaptive) * 2.0;

        row_standard[i] = standard_d;
        row_adaptive[i] = adaptive;
        row_difference[i] = difference;
    }

    delete[] pre_high_buf;
    delete[] pre_low_buf;
    delete[] pre_close_buf;
    delete[] stoch_high_buf;
    delete[] stoch_low_buf;
    delete[] d_buf;
}
