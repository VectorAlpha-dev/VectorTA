#include <cmath>
#include <cstddef>

namespace {
constexpr double ZERO_RANGE_DIVISOR = 9'999'999.0;
constexpr int MA_SMA = 0;
constexpr int MA_EMA = 1;
constexpr int MA_WMA = 2;
constexpr int MA_VWMA = 3;

__device__ bool is_valid_bar(double high, double low, double close, double volume) {
    return isfinite(high) && isfinite(low) && isfinite(close) && isfinite(volume) && high >= low;
}

struct SmaState {
    int period;
    double* window;
    int head;
    int count;
    double sum;

    __device__ void init(int value, double* storage) {
        period = value;
        window = storage;
        head = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ double update(double value) {
        if (count < period) {
            window[head] = value;
            head = (head + 1) % period;
            count += 1;
            sum += value;
            return count == period ? sum / static_cast<double>(period) : NAN;
        }

        const double old = window[head];
        window[head] = value;
        head = (head + 1) % period;
        sum += value - old;
        return sum / static_cast<double>(period);
    }
};

struct EmaState {
    double alpha;
    double value;
    bool initialized;

    __device__ void init(int period) {
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        value = NAN;
        initialized = false;
    }

    __device__ double update(double x) {
        if (!initialized) {
            value = x;
            initialized = true;
        } else {
            value = alpha * x + (1.0 - alpha) * value;
        }
        return value;
    }
};

struct WmaState {
    int period;
    double* window;
    int head;
    int count;
    double plain_sum;
    double weighted_sum;
    double divisor;

    __device__ void init(int value, double* storage) {
        period = value;
        window = storage;
        head = 0;
        count = 0;
        plain_sum = 0.0;
        weighted_sum = 0.0;
        divisor = static_cast<double>(value * (value + 1) / 2);
    }

    __device__ double update(double x) {
        if (count < period) {
            window[head] = x;
            head = (head + 1) % period;
            count += 1;
            plain_sum += x;
            weighted_sum += x * static_cast<double>(count);
            return count == period ? weighted_sum / divisor : NAN;
        }

        const double old = window[head];
        const double prev_plain = plain_sum;
        window[head] = x;
        head = (head + 1) % period;
        plain_sum = prev_plain - old + x;
        weighted_sum = weighted_sum - prev_plain + x * static_cast<double>(period);
        return weighted_sum / divisor;
    }
};

struct VwmaState {
    int period;
    double* values;
    double* weights;
    int head;
    int count;
    double num_sum;
    double den_sum;

    __device__ void init(int value, double* value_storage, double* weight_storage) {
        period = value;
        values = value_storage;
        weights = weight_storage;
        head = 0;
        count = 0;
        num_sum = 0.0;
        den_sum = 0.0;
    }

    __device__ double update(double x, double weight) {
        if (count < period) {
            values[head] = x;
            weights[head] = weight;
            head = (head + 1) % period;
            count += 1;
            num_sum += x * weight;
            den_sum += weight;
        } else {
            const double old_value = values[head];
            const double old_weight = weights[head];
            values[head] = x;
            weights[head] = weight;
            head = (head + 1) % period;
            num_sum += x * weight - old_value * old_weight;
            den_sum += weight - old_weight;
        }

        if (count == period && den_sum != 0.0) {
            return num_sum / den_sum;
        }
        return NAN;
    }
};

struct BaseMaState {
    int kind;
    SmaState sma;
    EmaState ema;
    WmaState wma;
    VwmaState vwma;

    __device__ void init(int ma_kind, int period, double* storage1, double* storage2) {
        kind = ma_kind;
        if (kind == MA_SMA) {
            sma.init(period, storage1);
        } else if (kind == MA_EMA) {
            ema.init(period);
        } else if (kind == MA_WMA) {
            wma.init(period, storage1);
        } else {
            vwma.init(period, storage1, storage2);
        }
    }

    __device__ double update(double value, double weight) {
        if (kind == MA_SMA) {
            return sma.update(value);
        }
        if (kind == MA_EMA) {
            return ema.update(value);
        }
        if (kind == MA_WMA) {
            return wma.update(value);
        }
        return vwma.update(value, weight);
    }
};

struct HmaState {
    bool passthrough;
    WmaState half;
    WmaState full;
    WmaState sqrt_wma;

    __device__ void init(
        int period,
        double* half_storage,
        double* full_storage,
        double* sqrt_storage
    ) {
        passthrough = period <= 1;
        const int full_period = period > 0 ? period : 1;
        const int half_period = period / 2 > 0 ? period / 2 : 1;
        int sqrt_period = static_cast<int>(floor(sqrt(static_cast<double>(period))));
        if (sqrt_period < 1) {
            sqrt_period = 1;
        }
        half.init(half_period, half_storage);
        full.init(full_period, full_storage);
        sqrt_wma.init(sqrt_period, sqrt_storage);
    }

    __device__ double update(double value) {
        if (passthrough) {
            return value;
        }
        const double half_value = half.update(value);
        const double full_value = full.update(value);
        if (!isfinite(half_value) || !isfinite(full_value)) {
            return NAN;
        }
        return sqrt_wma.update(2.0 * half_value - full_value);
    }
};
}

extern "C" __global__ void twiggs_money_flow_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smoothing_lengths,
    const int* __restrict__ ma_codes,
    int rows,
    double* __restrict__ out_tmf,
    double* __restrict__ out_smoothed
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_tmf = out_tmf + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed = out_smoothed + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_tmf[i] = NAN;
        row_smoothed[i] = NAN;
    }

    const int length = lengths[row];
    const int smoothing_length = smoothing_lengths[row];
    const int ma_code = ma_codes[row];
    if (length <= 0 || length > len || smoothing_length <= 0 || smoothing_length > len ||
        ma_code < MA_SMA || ma_code > MA_VWMA) {
        return;
    }

    double* adv_buf1 = new double[length];
    double* adv_buf2 = new double[length];
    double* vol_buf1 = new double[length];
    double* vol_buf2 = new double[length];
    const int half_period = length / 2 > 0 ? length / 2 : 1;
    const int full_period = smoothing_length > 0 ? smoothing_length : 1;
    int sqrt_period = static_cast<int>(floor(sqrt(static_cast<double>(smoothing_length))));
    if (sqrt_period < 1) {
        sqrt_period = 1;
    }
    double* hma_half_buf = new double[half_period];
    double* hma_full_buf = new double[full_period];
    double* hma_sqrt_buf = new double[sqrt_period];
    if (adv_buf1 == nullptr || adv_buf2 == nullptr || vol_buf1 == nullptr || vol_buf2 == nullptr ||
        hma_half_buf == nullptr || hma_full_buf == nullptr || hma_sqrt_buf == nullptr) {
        delete[] adv_buf1;
        delete[] adv_buf2;
        delete[] vol_buf1;
        delete[] vol_buf2;
        delete[] hma_half_buf;
        delete[] hma_full_buf;
        delete[] hma_sqrt_buf;
        return;
    }

    BaseMaState adv_ma;
    BaseMaState vol_ma;
    HmaState smoother;
    adv_ma.init(ma_code, length, adv_buf1, adv_buf2);
    vol_ma.init(ma_code, length, vol_buf1, vol_buf2);
    smoother.init(smoothing_length, hma_half_buf, hma_full_buf, hma_sqrt_buf);

    bool has_prev_close = false;
    double prev_close = NAN;

    for (int i = 0; i < len; ++i) {
        const double high_value = high[i];
        const double low_value = low[i];
        const double close_value = close[i];
        const double volume_value = volume[i];

        if (!is_valid_bar(high_value, low_value, close_value, volume_value)) {
            has_prev_close = isfinite(close_value);
            prev_close = close_value;
            continue;
        }

        if (!has_prev_close || !isfinite(prev_close)) {
            has_prev_close = true;
            prev_close = close_value;
            continue;
        }

        const double tr_h = prev_close > high_value ? prev_close : high_value;
        const double tr_l = prev_close < low_value ? prev_close : low_value;
        const double tr_c = tr_h - tr_l;
        const double denom = tr_c == 0.0 ? ZERO_RANGE_DIVISOR : tr_c;
        const double adv = volume_value * (((close_value - tr_l) - (tr_h - close_value)) / denom);

        const double wm_v = vol_ma.update(volume_value, volume_value);
        const double wm_a = adv_ma.update(adv, volume_value);
        double tmf = NAN;
        if (wm_v == 0.0) {
            tmf = 0.0;
        } else if (isfinite(wm_v) && isfinite(wm_a)) {
            tmf = wm_a / wm_v;
        }

        const double smoothed = isfinite(tmf) ? smoother.update(tmf) : NAN;
        row_tmf[i] = tmf;
        row_smoothed[i] = smoothed;
        prev_close = close_value;
        has_prev_close = true;
    }

    delete[] adv_buf1;
    delete[] adv_buf2;
    delete[] vol_buf1;
    delete[] vol_buf2;
    delete[] hma_half_buf;
    delete[] hma_full_buf;
    delete[] hma_sqrt_buf;
}
