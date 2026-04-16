#include <cmath>
#include <cstddef>

namespace {
constexpr int MA_WSMA = 0;
constexpr int MA_SMA = 1;
constexpr int MA_EMA = 2;
constexpr int MA_WMA = 3;
constexpr int MA_VWMA = 4;

__device__ inline bool is_valid_pair(double source, double volume) {
    return isfinite(source) && isfinite(volume);
}

__device__ inline double rsi_from_avgs(double avg_gain, double avg_loss) {
    if (avg_loss <= 0.0) {
        if (avg_gain <= 0.0) {
            return 50.0;
        }
        return 100.0;
    }
    if (avg_gain <= 0.0) {
        return 0.0;
    }
    const double rs = avg_gain / avg_loss;
    return 100.0 - 100.0 / (1.0 + rs);
}

struct WeightedRsiState {
    int period;
    double prev_source;
    bool has_prev_source;
    double gain_sum;
    double loss_sum;
    int count;
    double avg_gain;
    double avg_loss;
    bool initialized;

    __device__ void init(int value) {
        period = value;
        prev_source = NAN;
        has_prev_source = false;
        gain_sum = 0.0;
        loss_sum = 0.0;
        count = 0;
        avg_gain = 0.0;
        avg_loss = 0.0;
        initialized = false;
    }

    __device__ double update(double source, double volume) {
        if (!is_valid_pair(source, volume)) {
            if (isfinite(source)) {
                prev_source = source;
                has_prev_source = true;
            } else {
                prev_source = NAN;
                has_prev_source = false;
            }
            return NAN;
        }

        if (!has_prev_source || !isfinite(prev_source)) {
            prev_source = source;
            has_prev_source = true;
            return NAN;
        }

        const double change = source - prev_source;
        const double gain = (change > 0.0 ? change : 0.0) * volume;
        const double loss = (change < 0.0 ? -change : 0.0) * volume;
        prev_source = source;

        if (!initialized) {
            gain_sum += gain;
            loss_sum += loss;
            count += 1;
            if (count == period) {
                avg_gain = gain_sum / static_cast<double>(period);
                avg_loss = loss_sum / static_cast<double>(period);
                initialized = true;
                return rsi_from_avgs(avg_gain, avg_loss);
            }
            return NAN;
        }

        const double p = static_cast<double>(period);
        avg_gain = (avg_gain * (p - 1.0) + gain) / p;
        avg_loss = (avg_loss * (p - 1.0) + loss) / p;
        return rsi_from_avgs(avg_gain, avg_loss);
    }
};

struct StochState {
    double* window;
    int period;
    int head;
    int count;
    int valid;

    __device__ void init(int value, double* storage) {
        window = storage;
        period = value;
        head = 0;
        count = 0;
        valid = 0;
        for (int i = 0; i < value; ++i) {
            window[i] = NAN;
        }
    }

    __device__ double update(double value) {
        if (count == period) {
            const double old = window[head];
            if (isfinite(old)) {
                valid -= 1;
            }
        } else {
            count += 1;
        }

        window[head] = value;
        head += 1;
        if (head == period) {
            head = 0;
        }

        if (isfinite(value)) {
            valid += 1;
        }

        if (count < period || valid < period || !isfinite(value)) {
            return NAN;
        }

        double lowest = INFINITY;
        double highest = -INFINITY;
        for (int i = 0; i < period; ++i) {
            lowest = fmin(lowest, window[i]);
            highest = fmax(highest, window[i]);
        }
        const double denom = highest - lowest;
        if (!isfinite(denom) || denom == 0.0) {
            return NAN;
        }
        return (value - lowest) / denom * 100.0;
    }
};

struct WeightedSmaState {
    int period;
    double* numerators;
    double* weights;
    int head;
    int count;
    double numerator_sum;
    double weight_sum;

    __device__ void init(int value, double* numerator_storage, double* weight_storage) {
        period = value;
        numerators = numerator_storage;
        weights = weight_storage;
        head = 0;
        count = 0;
        numerator_sum = 0.0;
        weight_sum = 0.0;
    }

    __device__ double update(double value, double weight) {
        const double numerator = value * weight;
        if (count < period) {
            numerators[head] = numerator;
            weights[head] = weight;
            head = (head + 1) % period;
            count += 1;
            numerator_sum += numerator;
            weight_sum += weight;
        } else {
            const double old_numerator = numerators[head];
            const double old_weight = weights[head];
            numerators[head] = numerator;
            weights[head] = weight;
            head = (head + 1) % period;
            numerator_sum += numerator - old_numerator;
            weight_sum += weight - old_weight;
        }

        if (count == period && weight_sum != 0.0) {
            return numerator_sum / weight_sum;
        }
        return NAN;
    }
};

struct WeightedEmaState {
    double alpha;
    double numerator;
    double denominator;
    bool initialized;

    __device__ void init(int period) {
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        numerator = 0.0;
        denominator = 0.0;
        initialized = false;
    }

    __device__ double update(double value, double weight) {
        const double num = value * weight;
        if (!initialized) {
            numerator = num;
            denominator = weight;
            initialized = true;
        } else {
            const double beta = 1.0 - alpha;
            numerator = alpha * num + beta * numerator;
            denominator = alpha * weight + beta * denominator;
        }

        if (denominator != 0.0) {
            return numerator / denominator;
        }
        return NAN;
    }
};

struct WeightedWsmaState {
    int period;
    double numerator_sum;
    double denominator_sum;
    int count;
    double numerator_avg;
    double denominator_avg;
    bool initialized;

    __device__ void init(int value) {
        period = value;
        numerator_sum = 0.0;
        denominator_sum = 0.0;
        count = 0;
        numerator_avg = 0.0;
        denominator_avg = 0.0;
        initialized = false;
    }

    __device__ double update(double value, double weight) {
        const double numerator = value * weight;
        if (!initialized) {
            numerator_sum += numerator;
            denominator_sum += weight;
            count += 1;
            if (count == period) {
                numerator_avg = numerator_sum / static_cast<double>(period);
                denominator_avg = denominator_sum / static_cast<double>(period);
                initialized = true;
                if (denominator_avg != 0.0) {
                    return numerator_avg / denominator_avg;
                }
            }
            return NAN;
        }

        const double p = static_cast<double>(period);
        numerator_avg = (numerator_avg * (p - 1.0) + numerator) / p;
        denominator_avg = (denominator_avg * (p - 1.0) + weight) / p;
        if (denominator_avg != 0.0) {
            return numerator_avg / denominator_avg;
        }
        return NAN;
    }
};

struct WeightedWmaState {
    int period;
    double* numerators;
    double* weights;
    int head;
    int count;
    double numerator_plain_sum;
    double numerator_weighted_sum;
    double weight_plain_sum;
    double weight_weighted_sum;

    __device__ void init(int value, double* numerator_storage, double* weight_storage) {
        period = value;
        numerators = numerator_storage;
        weights = weight_storage;
        head = 0;
        count = 0;
        numerator_plain_sum = 0.0;
        numerator_weighted_sum = 0.0;
        weight_plain_sum = 0.0;
        weight_weighted_sum = 0.0;
    }

    __device__ double update(double value, double weight) {
        const double numerator = value * weight;
        if (count < period) {
            numerators[head] = numerator;
            weights[head] = weight;
            head = (head + 1) % period;
            count += 1;
            numerator_plain_sum += numerator;
            weight_plain_sum += weight;
            numerator_weighted_sum += numerator * static_cast<double>(count);
            weight_weighted_sum += weight * static_cast<double>(count);
        } else {
            const double old_numerator = numerators[head];
            const double old_weight = weights[head];
            const double prev_numerator_plain = numerator_plain_sum;
            const double prev_weight_plain = weight_plain_sum;
            numerators[head] = numerator;
            weights[head] = weight;
            head = (head + 1) % period;
            numerator_plain_sum = prev_numerator_plain - old_numerator + numerator;
            weight_plain_sum = prev_weight_plain - old_weight + weight;
            numerator_weighted_sum =
                numerator_weighted_sum - prev_numerator_plain + numerator * static_cast<double>(period);
            weight_weighted_sum =
                weight_weighted_sum - prev_weight_plain + weight * static_cast<double>(period);
        }

        if (count == period && weight_weighted_sum != 0.0) {
            return numerator_weighted_sum / weight_weighted_sum;
        }
        return NAN;
    }
};

struct WeightedMaState {
    int kind;
    WeightedWsmaState wsma;
    WeightedSmaState sma;
    WeightedEmaState ema;
    WeightedWmaState wma;
    WeightedSmaState vwma;

    __device__ void init(int ma_kind, int period, double* buf1, double* buf2) {
        kind = ma_kind;
        if (kind == MA_WSMA) {
            wsma.init(period);
        } else if (kind == MA_SMA) {
            sma.init(period, buf1, buf2);
        } else if (kind == MA_EMA) {
            ema.init(period);
        } else if (kind == MA_WMA) {
            wma.init(period, buf1, buf2);
        } else {
            vwma.init(period, buf1, buf2);
        }
    }

    __device__ double update(double value, double weight) {
        if (kind == MA_WSMA) {
            return wsma.update(value, weight);
        }
        if (kind == MA_SMA) {
            return sma.update(value, weight);
        }
        if (kind == MA_EMA) {
            return ema.update(value, weight);
        }
        if (kind == MA_WMA) {
            return wma.update(value, weight);
        }
        return vwma.update(value, weight);
    }
};
}

extern "C" __global__ void volume_weighted_stochastic_rsi_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ rsi_lengths,
    const int* __restrict__ stoch_lengths,
    const int* __restrict__ k_lengths,
    const int* __restrict__ d_lengths,
    const int* __restrict__ ma_codes,
    int rows,
    double* __restrict__ out_k,
    double* __restrict__ out_d
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_k = out_k + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_d = out_d + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_k[i] = NAN;
        row_d[i] = NAN;
    }

    const int rsi_length = rsi_lengths[row];
    const int stoch_length = stoch_lengths[row];
    const int k_length = k_lengths[row];
    const int d_length = d_lengths[row];
    const int ma_code = ma_codes[row];
    if (rsi_length <= 0 || rsi_length > len || stoch_length <= 0 || stoch_length > len ||
        k_length <= 0 || k_length > len || d_length <= 0 || d_length > len ||
        ma_code < MA_WSMA || ma_code > MA_VWMA) {
        return;
    }

    double* stoch_window = new double[stoch_length];
    double* k_buf1 = nullptr;
    double* k_buf2 = nullptr;
    double* d_buf1 = nullptr;
    double* d_buf2 = nullptr;
    if (ma_code == MA_SMA || ma_code == MA_WMA || ma_code == MA_VWMA) {
        k_buf1 = new double[k_length];
        k_buf2 = new double[k_length];
        d_buf1 = new double[d_length];
        d_buf2 = new double[d_length];
    }
    if (stoch_window == nullptr ||
        ((ma_code == MA_SMA || ma_code == MA_WMA || ma_code == MA_VWMA) &&
         (k_buf1 == nullptr || k_buf2 == nullptr || d_buf1 == nullptr || d_buf2 == nullptr))) {
        delete[] stoch_window;
        delete[] k_buf1;
        delete[] k_buf2;
        delete[] d_buf1;
        delete[] d_buf2;
        return;
    }

    WeightedRsiState rsi_state;
    StochState stoch_state;
    WeightedMaState k_ma;
    WeightedMaState d_ma;
    rsi_state.init(rsi_length);
    stoch_state.init(stoch_length, stoch_window);
    k_ma.init(ma_code, k_length, k_buf1, k_buf2);
    d_ma.init(ma_code, d_length, d_buf1, d_buf2);

    for (int i = 0; i < len; ++i) {
        const double rsi = rsi_state.update(source[i], volume[i]);
        const double stoch = stoch_state.update(rsi);
        const double k = isfinite(stoch) ? k_ma.update(stoch, volume[i]) : NAN;
        const double d = isfinite(k) ? d_ma.update(k, volume[i]) : NAN;
        row_k[i] = k;
        row_d[i] = d;
    }

    delete[] stoch_window;
    delete[] k_buf1;
    delete[] k_buf2;
    delete[] d_buf1;
    delete[] d_buf2;
}
