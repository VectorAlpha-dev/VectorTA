#include <cmath>
#include <cstddef>

extern "C" __global__ void normalized_volume_true_range_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const double* __restrict__ outlier_ranges,
    const int* __restrict__ atr_lengths,
    const int* __restrict__ volume_lengths,
    const int* __restrict__ styles,
    int rows,
    double* __restrict__ out_normalized_volume,
    double* __restrict__ out_normalized_true_range,
    double* __restrict__ out_baseline,
    double* __restrict__ out_atr,
    double* __restrict__ out_average_volume
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double outlier_range = outlier_ranges[row];
    int atr_length = atr_lengths[row];
    int volume_length = volume_lengths[row];
    int style = styles[row];

    double* row_out_nv =
        out_normalized_volume + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_ntr =
        out_normalized_true_range + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_baseline =
        out_baseline + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_atr = out_atr + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_avg_vol =
        out_average_volume + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_nv[i] = NAN;
        row_out_ntr[i] = NAN;
        row_out_baseline[i] = NAN;
        row_out_atr[i] = NAN;
        row_out_avg_vol[i] = NAN;
    }

    if (!isfinite(outlier_range) || outlier_range < 0.5 || atr_length < 2 || volume_length < 2 ||
        style < 0 || style > 2) {
        return;
    }

    double* atr_ring = new double[static_cast<size_t>(atr_length)];
    double* volume_ring = new double[static_cast<size_t>(volume_length)];
    if (atr_ring == nullptr || volume_ring == nullptr) {
        if (atr_ring != nullptr) {
            delete[] atr_ring;
        }
        if (volume_ring != nullptr) {
            delete[] volume_ring;
        }
        return;
    }

    double abs_sum = 0.0;
    double volume_sum = 0.0;
    int count = 0;
    double abs_variance_sum = 0.0;
    int abs_qualifying_count = 0;
    double abs_positive_deviation = NAN;
    double volume_variance_sum = 0.0;
    int volume_qualifying_count = 0;
    double volume_positive_deviation = NAN;
    double prev_close = NAN;
    bool have_prev_close = false;
    bool atr_ready = false;
    double atr_first_value = NAN;
    int atr_head = 0;
    double atr_sum = 0.0;
    bool average_volume_ready = false;
    double average_volume_first_value = NAN;
    int average_volume_head = 0;
    double average_volume_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        bool valid = false;
        if (style == 0) {
            valid = isfinite(open[i]) && isfinite(close[i]) && isfinite(volume[i]);
        } else if (style == 1) {
            valid = isfinite(high[i]) && isfinite(low[i]) && isfinite(volume[i]);
        } else {
            valid = isfinite(close[i]) && isfinite(volume[i]);
        }

        if (!valid) {
            if (isfinite(close[i])) {
                prev_close = close[i];
                have_prev_close = true;
            }
            continue;
        }

        double start = 0.0;
        double finish = 0.0;
        if (style == 0) {
            start = open[i];
            finish = close[i];
        } else if (style == 1) {
            start = low[i];
            finish = high[i];
        } else {
            start = have_prev_close ? prev_close : close[i];
            finish = close[i];
        }

        prev_close = close[i];
        have_prev_close = true;

        double denom = fmin(start, finish);
        if (!isfinite(denom) || denom <= 0.0) {
            continue;
        }

        double abs_percent = fabs(finish - start) / denom;
        if (!isfinite(abs_percent)) {
            continue;
        }

        count += 1;
        abs_sum += abs_percent;
        volume_sum += volume[i];

        double count_f64 = static_cast<double>(count);
        double avg_abs_percent = abs_sum / count_f64;
        double avg_volume = volume_sum / count_f64;

        if (abs_percent > avg_abs_percent) {
            double delta = abs_percent - avg_abs_percent;
            abs_variance_sum += delta * delta;
            abs_qualifying_count += 1;
            if (abs_qualifying_count >= 2) {
                abs_positive_deviation =
                    sqrt(abs_variance_sum / static_cast<double>(abs_qualifying_count - 1));
            }
        }

        if (volume[i] > avg_volume) {
            double delta = volume[i] - avg_volume;
            volume_variance_sum += delta * delta;
            volume_qualifying_count += 1;
            if (volume_qualifying_count >= 2) {
                volume_positive_deviation =
                    sqrt(volume_variance_sum / static_cast<double>(volume_qualifying_count - 1));
            }
        }

        double abs_percent_max = isfinite(abs_positive_deviation)
            ? avg_abs_percent + abs_positive_deviation * outlier_range
            : NAN;
        double normalized_avg_percent =
            (isfinite(abs_percent_max) && abs_percent_max > 0.0) ? avg_abs_percent / abs_percent_max
                                                                 : NAN;
        double scale_factor =
            (isfinite(normalized_avg_percent) && normalized_avg_percent > 0.0 &&
             normalized_avg_percent < 1.0 && isfinite(volume_positive_deviation) &&
             volume_positive_deviation > 0.0)
            ? avg_volume * (1.0 - normalized_avg_percent) /
                (normalized_avg_percent * volume_positive_deviation)
            : NAN;
        double max_volume = (isfinite(scale_factor) && isfinite(volume_positive_deviation))
            ? avg_volume + volume_positive_deviation * scale_factor
            : NAN;
        double normalized_abs_percent =
            (isfinite(abs_percent_max) && abs_percent_max > 0.0)
            ? fmin(abs_percent, abs_percent_max) / abs_percent_max
            : NAN;
        double normalized_volume_ratio =
            (isfinite(max_volume) && max_volume > 0.0) ? fmin(volume[i], max_volume) / max_volume
                                                       : NAN;
        double normalized_avg_volume_ratio =
            (isfinite(max_volume) && max_volume > 0.0) ? avg_volume / max_volume : NAN;

        double nv = normalized_volume_ratio * 100.0;
        double ntr = normalized_abs_percent * 100.0;
        double baseline = normalized_avg_volume_ratio * 100.0;
        double atr_value = NAN;
        if (!atr_ready) {
            if (isfinite(ntr)) {
                atr_first_value = ntr;
                atr_ready = true;
                for (int j = 0; j < atr_length; ++j) {
                    atr_ring[j] = ntr;
                }
                atr_sum = ntr * static_cast<double>(atr_length);
                atr_head = (atr_length > 0) ? (1 % atr_length) : 0;
                atr_value = ntr;
            }
        } else {
            double sanitized = isfinite(ntr) ? ntr : atr_first_value;
            double old = atr_ring[atr_head];
            atr_ring[atr_head] = sanitized;
            atr_sum += sanitized - old;
            atr_head += 1;
            if (atr_head == atr_length) {
                atr_head = 0;
            }
            atr_value = atr_sum / static_cast<double>(atr_length);
        }

        double average_volume_value = NAN;
        if (!average_volume_ready) {
            if (isfinite(nv)) {
                average_volume_first_value = nv;
                average_volume_ready = true;
                for (int j = 0; j < volume_length; ++j) {
                    volume_ring[j] = nv;
                }
                average_volume_sum = nv * static_cast<double>(volume_length);
                average_volume_head = (volume_length > 0) ? (1 % volume_length) : 0;
                average_volume_value = nv;
            }
        } else {
            double sanitized = isfinite(nv) ? nv : average_volume_first_value;
            double old = volume_ring[average_volume_head];
            volume_ring[average_volume_head] = sanitized;
            average_volume_sum += sanitized - old;
            average_volume_head += 1;
            if (average_volume_head == volume_length) {
                average_volume_head = 0;
            }
            average_volume_value = average_volume_sum / static_cast<double>(volume_length);
        }

        if (!(isfinite(nv) && isfinite(ntr) && isfinite(baseline) && isfinite(atr_value) &&
              isfinite(average_volume_value))) {
            continue;
        }

        row_out_nv[i] = nv;
        row_out_ntr[i] = ntr;
        row_out_baseline[i] = baseline;
        row_out_atr[i] = atr_value;
        row_out_avg_vol[i] = average_volume_value;
    }

    delete[] atr_ring;
    delete[] volume_ring;
}
