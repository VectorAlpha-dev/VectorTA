#include <cmath>
#include <cstddef>

namespace {
constexpr int MA_SMA = 0;
constexpr int MA_EMA = 1;
constexpr int MA_WMA = 2;
constexpr int MA_TMA = 3;
constexpr int MA_VIDYA = 4;
constexpr int MA_WWMA = 5;
constexpr int MA_ZLEMA = 6;
constexpr int MA_TSF = 7;
constexpr int MA_HMA = 8;
constexpr int MA_VWMA = 9;

struct SmaState {
    int period;
    double* window;
    int head;
    int count;
    double sum;

    __device__ void init(int value, double* storage) {
        period = value;
        window = storage;
        reset();
    }

    __device__ void reset() {
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
    int period;
    double alpha;
    double decay;
    int count;
    double mean;
    bool filled;

    __device__ void init(int period) {
        this->period = period;
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        decay = 1.0 - alpha;
        reset();
    }

    __device__ void reset() {
        count = 0;
        mean = NAN;
        filled = false;
    }

    __device__ double update(double x) {
        count += 1;
        if (count == 1) {
            mean = x;
        } else if (count <= period) {
            mean = (x - mean) * (1.0 / static_cast<double>(count)) + mean;
        } else {
            mean = decay * mean + alpha * x;
        }
        if (!filled && count >= period) {
            filled = true;
        }
        return filled ? mean : NAN;
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
        divisor = static_cast<double>(value * (value + 1) / 2);
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        plain_sum = 0.0;
        weighted_sum = 0.0;
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
    double numerator_sum;
    double denominator_sum;

    __device__ void init(int value, double* value_storage, double* weight_storage) {
        period = value;
        values = value_storage;
        weights = weight_storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        numerator_sum = 0.0;
        denominator_sum = 0.0;
    }

    __device__ double update(double x, double weight) {
        if (count < period) {
            values[head] = x;
            weights[head] = weight;
            head = (head + 1) % period;
            count += 1;
            numerator_sum += x * weight;
            denominator_sum += weight;
        } else {
            const double old_value = values[head];
            const double old_weight = weights[head];
            values[head] = x;
            weights[head] = weight;
            head = (head + 1) % period;
            numerator_sum += x * weight - old_value * old_weight;
            denominator_sum += weight - old_weight;
        }

        if (count == period && denominator_sum != 0.0) {
            return numerator_sum / denominator_sum;
        }
        return NAN;
    }
};

struct TmaState {
    SmaState stage1;
    SmaState stage2;

    __device__ void init(int length, double* buf1, double* buf2) {
        const int p1 = (length + 1) / 2;
        const int p2 = length / 2 + 1;
        stage1.init(p1, buf1);
        stage2.init(p2, buf2);
    }

    __device__ void reset() {
        stage1.reset();
        stage2.reset();
    }

    __device__ double update(double value) {
        const double v1 = stage1.update(value);
        if (!isfinite(v1)) {
            return NAN;
        }
        return stage2.update(v1);
    }
};

struct WwmaState {
    double alpha;
    double state;
    bool initialized;

    __device__ void init(int length) {
        alpha = 1.0 / static_cast<double>(length);
        reset();
    }

    __device__ void reset() {
        state = NAN;
        initialized = false;
    }

    __device__ double update(double value) {
        if (!initialized) {
            state = value;
            initialized = true;
        } else {
            state = alpha * value + (1.0 - alpha) * state;
        }
        return state;
    }
};

struct VidyaState {
    double alpha;
    double prev;
    bool have_prev;
    double state;
    bool initialized;
    double ring_up[9];
    double ring_down[9];
    int head;
    int count;
    double sum_up;
    double sum_down;

    __device__ void init(int length) {
        alpha = 2.0 / (static_cast<double>(length) + 1.0);
        reset();
    }

    __device__ void reset() {
        prev = NAN;
        have_prev = false;
        state = NAN;
        initialized = false;
        head = 0;
        count = 0;
        sum_up = 0.0;
        sum_down = 0.0;
        for (int i = 0; i < 9; ++i) {
            ring_up[i] = 0.0;
            ring_down[i] = 0.0;
        }
    }

    __device__ double update(double value) {
        if (!have_prev) {
            prev = value;
            have_prev = true;
            state = value;
            initialized = true;
            return state;
        }

        const double diff = value - prev;
        prev = value;

        const double up = diff > 0.0 ? diff : 0.0;
        const double down = diff < 0.0 ? -diff : 0.0;

        if (count == 9) {
            sum_up -= ring_up[head];
            sum_down -= ring_down[head];
        } else {
            count += 1;
        }

        ring_up[head] = up;
        ring_down[head] = down;
        sum_up += up;
        sum_down += down;
        head += 1;
        if (head == 9) {
            head = 0;
        }

        const double denom = sum_up + sum_down;
        const double cmo_abs = denom > 0.0 ? fabs((sum_up - sum_down) / denom) : 0.0;
        const double adaptive_alpha = alpha * cmo_abs;

        if (!initialized) {
            state = value;
            initialized = true;
        } else {
            state = adaptive_alpha * value + (1.0 - adaptive_alpha) * state;
        }
        return state;
    }
};

struct ZlemaState {
    int period;
    int lag;
    int ring_len;
    double alpha;
    double decay;
    double last_ema;
    double* ring;
    int head;
    unsigned int idx;
    int first_idx;
    int warm_idx;

    __device__ void init(int value, double* storage) {
        period = value;
        lag = (value - 1) / 2;
        ring_len = lag + 1;
        if (ring_len < 1) {
            ring_len = 1;
        }
        alpha = 2.0 / (static_cast<double>(value) + 1.0);
        decay = 1.0 - alpha;
        ring = storage;
        reset();
    }

    __device__ void reset() {
        last_ema = NAN;
        head = 0;
        idx = 0;
        first_idx = -1;
        warm_idx = -1;
    }

    __device__ double update(double x) {
        const int pos = head;
        ring[pos] = x;
        head += 1;
        if (head == ring_len) {
            head = 0;
        }

        const int i = static_cast<int>(idx);
        idx += 1;

        if (first_idx < 0) {
            first_idx = i;
            warm_idx = i + period - 1;
            last_ema = x;
            return i >= warm_idx ? last_ema : NAN;
        }

        double value = x;
        if (lag > 0 && i >= first_idx + lag) {
            const int lag_pos = pos >= lag ? pos - lag : pos + 1;
            value = 2.0 * x - ring[lag_pos];
        }

        last_ema = alpha * value + decay * last_ema;
        return i >= warm_idx ? last_ema : NAN;
    }
};

struct TsfState {
    int period;
    double pf;
    double inv_pf;
    double pf_over_div;
    double sumx_over_div;
    double p_minus_mean_x;
    double* buffer;
    int head;
    bool filled;
    double s0;
    double s1;

    __device__ void init(int value, double* storage) {
        period = value;
        pf = static_cast<double>(value);
        double sum_x = 0.0;
        double sum_x2 = 0.0;
        for (int x = 0; x < value; ++x) {
            const double xf = static_cast<double>(x);
            sum_x += xf;
            sum_x2 += xf * xf;
        }
        const double divisor = pf * sum_x2 - sum_x * sum_x;
        inv_pf = 1.0 / pf;
        const double inv_div = 1.0 / divisor;
        pf_over_div = pf * inv_div;
        sumx_over_div = sum_x * inv_div;
        p_minus_mean_x = pf - sum_x * inv_pf;
        buffer = storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        filled = false;
        s0 = 0.0;
        s1 = 0.0;
    }

    __device__ double update(double value) {
        if (!filled) {
            buffer[head] = value;
            head += 1;
            if (head == period) {
                head = 0;
                filled = true;
                s0 = 0.0;
                s1 = 0.0;
                int idx = head;
                for (int j = 0; j < period; ++j) {
                    const double v = buffer[idx];
                    s0 += v;
                    s1 += static_cast<double>(j) * v;
                    idx += 1;
                    if (idx == period) {
                        idx = 0;
                    }
                }
                const double m = s1 * pf_over_div - s0 * sumx_over_div;
                return s0 * inv_pf + m * p_minus_mean_x;
            }
            return NAN;
        }

        const double old = buffer[head];
        buffer[head] = value;

        const double new_s0 = s0 + (value - old);
        const double new_s1 = pf * value + s1 - new_s0;
        s0 = new_s0;
        s1 = new_s1;

        head += 1;
        if (head == period) {
            head = 0;
        }

        const double m = s1 * pf_over_div - s0 * sumx_over_div;
        return s0 * inv_pf + m * p_minus_mean_x;
    }
};

struct HmaState {
    WmaState half;
    WmaState full;
    WmaState sqrt_wma;

    __device__ void init(int period, double* half_storage, double* full_storage, double* sqrt_storage) {
        int half_period = period / 2;
        if (half_period < 1) {
            half_period = 1;
        }
        int sqrt_period = static_cast<int>(floor(sqrt(static_cast<double>(period))));
        if (sqrt_period < 1) {
            sqrt_period = 1;
        }
        half.init(half_period, half_storage);
        full.init(period, full_storage);
        sqrt_wma.init(sqrt_period, sqrt_storage);
    }

    __device__ void reset() {
        half.reset();
        full.reset();
        sqrt_wma.reset();
    }

    __device__ double update(double value) {
        const double half_value = half.update(value);
        const double full_value = full.update(value);
        if (!isfinite(half_value) || !isfinite(full_value)) {
            return NAN;
        }
        return sqrt_wma.update(2.0 * half_value - full_value);
    }
};
}

extern "C" __global__ void kairi_relative_index_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ ma_codes,
    int rows,
    double* __restrict__ out
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    const int length = lengths[row];
    const int ma_code = ma_codes[row];
    if (length < 2 || length > len || ma_code < MA_SMA || ma_code > MA_VWMA) {
        return;
    }

    const int tma1 = (length + 1) / 2;
    const int tma2 = length / 2 + 1;
    const int lag = (length - 1) / 2;
    const int zlema_ring = lag + 1 > 0 ? lag + 1 : 1;
    int hma_half = length / 2;
    if (hma_half < 1) {
        hma_half = 1;
    }
    int hma_sqrt = static_cast<int>(floor(sqrt(static_cast<double>(length))));
    if (hma_sqrt < 1) {
        hma_sqrt = 1;
    }

    double* buf1 = nullptr;
    double* buf2 = nullptr;
    double* buf3 = nullptr;

    SmaState sma;
    EmaState ema;
    WmaState wma;
    TmaState tma;
    VidyaState vidya;
    WwmaState wwma;
    ZlemaState zlema;
    TsfState tsf;
    HmaState hma;
    VwmaState vwma;

    if (ma_code == MA_SMA) {
        buf1 = new double[length];
        if (buf1 == nullptr) {
            return;
        }
        sma.init(length, buf1);
    } else if (ma_code == MA_EMA) {
        ema.init(length);
    } else if (ma_code == MA_WMA) {
        buf1 = new double[length];
        if (buf1 == nullptr) {
            return;
        }
        wma.init(length, buf1);
    } else if (ma_code == MA_TMA) {
        buf1 = new double[tma1];
        buf2 = new double[tma2];
        if (buf1 == nullptr || buf2 == nullptr) {
            delete[] buf1;
            delete[] buf2;
            return;
        }
        tma.init(length, buf1, buf2);
    } else if (ma_code == MA_VIDYA) {
        vidya.init(length);
    } else if (ma_code == MA_WWMA) {
        wwma.init(length);
    } else if (ma_code == MA_ZLEMA) {
        buf1 = new double[zlema_ring];
        if (buf1 == nullptr) {
            return;
        }
        zlema.init(length, buf1);
    } else if (ma_code == MA_TSF) {
        buf1 = new double[length];
        if (buf1 == nullptr) {
            return;
        }
        tsf.init(length, buf1);
    } else if (ma_code == MA_HMA) {
        buf1 = new double[hma_half];
        buf2 = new double[length];
        buf3 = new double[hma_sqrt];
        if (buf1 == nullptr || buf2 == nullptr || buf3 == nullptr) {
            delete[] buf1;
            delete[] buf2;
            delete[] buf3;
            return;
        }
        hma.init(length, buf1, buf2, buf3);
    } else {
        buf1 = new double[length];
        buf2 = new double[length];
        if (buf1 == nullptr || buf2 == nullptr) {
            delete[] buf1;
            delete[] buf2;
            return;
        }
        vwma.init(length, buf1, buf2);
    }

    for (int i = 0; i < len; ++i) {
        const double src = source[i];
        const double vol = volume[i];
        const bool valid = isfinite(src) && (ma_code != MA_VWMA || isfinite(vol));
        if (!valid) {
            if (ma_code == MA_SMA) {
                sma.reset();
            } else if (ma_code == MA_EMA) {
                ema.reset();
            } else if (ma_code == MA_WMA) {
                wma.reset();
            } else if (ma_code == MA_TMA) {
                tma.reset();
            } else if (ma_code == MA_VIDYA) {
                vidya.reset();
            } else if (ma_code == MA_WWMA) {
                wwma.reset();
            } else if (ma_code == MA_ZLEMA) {
                zlema.reset();
            } else if (ma_code == MA_TSF) {
                tsf.reset();
            } else if (ma_code == MA_HMA) {
                hma.reset();
            } else {
                vwma.reset();
            }
            continue;
        }

        double ma = NAN;
        if (ma_code == MA_SMA) {
            ma = sma.update(src);
        } else if (ma_code == MA_EMA) {
            ma = ema.update(src);
        } else if (ma_code == MA_WMA) {
            ma = wma.update(src);
        } else if (ma_code == MA_TMA) {
            ma = tma.update(src);
        } else if (ma_code == MA_VIDYA) {
            ma = vidya.update(src);
        } else if (ma_code == MA_WWMA) {
            ma = wwma.update(src);
        } else if (ma_code == MA_ZLEMA) {
            ma = zlema.update(src);
        } else if (ma_code == MA_TSF) {
            ma = tsf.update(src);
        } else if (ma_code == MA_HMA) {
            ma = hma.update(src);
        } else {
            ma = vwma.update(src, vol);
        }

        if (!isfinite(ma)) {
            continue;
        }
        if (ma == 0.0) {
            row_out[i] = src == 0.0 ? 0.0 : NAN;
        } else {
            row_out[i] = (src - ma) * 100.0 / ma;
        }
    }

    delete[] buf1;
    delete[] buf2;
    delete[] buf3;
}
