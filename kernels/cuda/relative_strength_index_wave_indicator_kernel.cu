#include <cmath>
#include <cstddef>

namespace {
struct RsiState {
    int period;
    double inv_p;
    double beta;
    bool initialized;
    double prev;
    int deltas_seen;
    double sum_gain;
    double sum_loss;
    double avg_gain;
    double avg_loss;
    bool ready;

    __device__ void init(int value) {
        period = value;
        inv_p = 1.0 / static_cast<double>(value);
        beta = 1.0 - inv_p;
        reset();
    }

    __device__ void reset() {
        initialized = false;
        prev = NAN;
        deltas_seen = 0;
        sum_gain = 0.0;
        sum_loss = 0.0;
        avg_gain = 0.0;
        avg_loss = 0.0;
        ready = false;
    }

    __device__ double update(double value) {
        if (!initialized) {
            prev = value;
            initialized = true;
            return NAN;
        }

        const double delta = value - prev;
        prev = value;
        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);

        if (!ready) {
            sum_gain += gain;
            sum_loss += loss;
            deltas_seen += 1;
            if (deltas_seen < period) {
                return NAN;
            }
            avg_gain = sum_gain * inv_p;
            avg_loss = sum_loss * inv_p;
            ready = true;
        } else {
            avg_gain = fma(avg_gain, beta, inv_p * gain);
            avg_loss = fma(avg_loss, beta, inv_p * loss);
        }

        const double denom = avg_gain + avg_loss;
        if (denom == 0.0) {
            return 50.0;
        }
        return 100.0 * avg_gain / denom;
    }
};

struct WmaState {
    int len;
    double denom;
    double* buf;
    int pos;
    int count;
    double sum;
    double weighted_sum;

    __device__ void init(int length, double* storage) {
        len = length;
        denom = static_cast<double>(length * (length + 1) / 2);
        buf = storage;
        reset();
    }

    __device__ void reset() {
        pos = 0;
        count = 0;
        sum = 0.0;
        weighted_sum = 0.0;
    }

    __device__ double update(double value) {
        if (count < len) {
            buf[count] = value;
            count += 1;
            sum += value;
            weighted_sum += static_cast<double>(count) * value;
            if (count == len) {
                return weighted_sum / denom;
            }
            return NAN;
        }

        const double old_sum = sum;
        const double old = buf[pos];
        buf[pos] = value;
        pos += 1;
        if (pos == len) {
            pos = 0;
        }
        weighted_sum = weighted_sum + static_cast<double>(len) * value - old_sum;
        sum = old_sum + value - old;
        return weighted_sum / denom;
    }
};
}

extern "C" __global__ void relative_strength_index_wave_indicator_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    const int* __restrict__ rsi_lengths,
    const int* __restrict__ length1s,
    const int* __restrict__ length2s,
    const int* __restrict__ length3s,
    const int* __restrict__ length4s,
    int rows,
    double* __restrict__ out_rsi_ma1,
    double* __restrict__ out_rsi_ma2,
    double* __restrict__ out_rsi_ma3,
    double* __restrict__ out_rsi_ma4,
    double* __restrict__ out_state
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int rsi_length = rsi_lengths[row];
    const int length1 = length1s[row];
    const int length2 = length2s[row];
    const int length3 = length3s[row];
    const int length4 = length4s[row];

    double* row_ma1 = out_rsi_ma1 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma2 = out_rsi_ma2 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma3 = out_rsi_ma3 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma4 = out_rsi_ma4 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_ma1[i] = NAN;
        row_ma2[i] = NAN;
        row_ma3[i] = NAN;
        row_ma4[i] = NAN;
        row_state[i] = NAN;
    }

    if (rsi_length <= 0 || length1 <= 0 || length2 <= 0 || length3 <= 0 || length4 <= 0) {
        return;
    }

    double* w1_buf = new double[length1];
    double* w2_buf = new double[length2];
    double* w3_buf = new double[length3];
    double* w4_buf = new double[length4];
    if (w1_buf == nullptr || w2_buf == nullptr || w3_buf == nullptr || w4_buf == nullptr) {
        delete[] w1_buf;
        delete[] w2_buf;
        delete[] w3_buf;
        delete[] w4_buf;
        return;
    }

    RsiState rsi_source;
    RsiState rsi_high;
    RsiState rsi_low;
    rsi_source.init(rsi_length);
    rsi_high.init(rsi_length);
    rsi_low.init(rsi_length);

    WmaState wma1;
    WmaState wma2;
    WmaState wma3;
    WmaState wma4;
    wma1.init(length1, w1_buf);
    wma2.init(length2, w2_buf);
    wma3.init(length3, w3_buf);
    wma4.init(length4, w4_buf);

    bool has_prev_slo = false;
    double prev_slo = NAN;

    for (int i = 0; i < len; ++i) {
        const double source_value = source[i];
        const double high_value = high[i];
        const double low_value = low[i];

        if (!isfinite(source_value) || !isfinite(high_value) || !isfinite(low_value)) {
            rsi_source.reset();
            rsi_high.reset();
            rsi_low.reset();
            wma1.reset();
            wma2.reset();
            wma3.reset();
            wma4.reset();
            has_prev_slo = false;
            prev_slo = NAN;
            continue;
        }

        const double custom_rsi = rsi_source.update(source_value);
        const double high_rsi = rsi_high.update(high_value);
        const double low_rsi = rsi_low.update(low_value);
        if (!isfinite(custom_rsi) || !isfinite(high_rsi) || !isfinite(low_rsi)) {
            continue;
        }

        const double hlc_rsi = (high_rsi + low_rsi + 2.0 * custom_rsi) * 0.25;
        const double rsi_ma1 = wma1.update(hlc_rsi);
        const double rsi_ma2 = wma2.update(hlc_rsi);
        const double rsi_ma3 = wma3.update(hlc_rsi);
        const double rsi_ma4 = wma4.update(hlc_rsi);

        row_ma1[i] = rsi_ma1;
        row_ma2[i] = rsi_ma2;
        row_ma3[i] = rsi_ma3;
        row_ma4[i] = rsi_ma4;

        if (isfinite(rsi_ma1) && isfinite(rsi_ma2)) {
            const double slo = rsi_ma1 - rsi_ma2;
            const double prev = has_prev_slo ? prev_slo : 0.0;
            prev_slo = slo;
            has_prev_slo = true;
            if (slo > 0.0) {
                row_state[i] = slo > prev ? 2.0 : 1.0;
            } else if (slo < 0.0) {
                row_state[i] = slo < prev ? -2.0 : -1.0;
            } else {
                row_state[i] = 0.0;
            }
        }
    }

    delete[] w1_buf;
    delete[] w2_buf;
    delete[] w3_buf;
    delete[] w4_buf;
}
