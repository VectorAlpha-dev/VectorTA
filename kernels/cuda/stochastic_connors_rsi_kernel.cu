#include <cmath>
#include <cstddef>

namespace {
constexpr double SCALE_100 = 100.0;
constexpr double EPS = 1.0e-14;

struct WilderRsiState {
    int period;
    double inv_p;
    double beta;
    bool has_prev;
    double prev;
    int seed_count;
    double sum_gain;
    double sum_loss;
    double avg_gain;
    double avg_loss;
    bool seeded;

    __device__ void init(int value) {
        period = value;
        inv_p = 1.0 / static_cast<double>(value);
        beta = 1.0 - inv_p;
        reset();
    }

    __device__ void reset() {
        has_prev = false;
        prev = NAN;
        seed_count = 0;
        sum_gain = 0.0;
        sum_loss = 0.0;
        avg_gain = 0.0;
        avg_loss = 0.0;
        seeded = false;
    }

    __device__ double update(double value) {
        if (!has_prev) {
            prev = value;
            has_prev = true;
            return NAN;
        }

        const double delta = value - prev;
        prev = value;

        if (!seeded) {
            sum_gain += fmax(delta, 0.0);
            sum_loss += fmax(-delta, 0.0);
            seed_count += 1;
            if (seed_count == period) {
                seeded = true;
                avg_gain = sum_gain * inv_p;
                avg_loss = sum_loss * inv_p;
                const double denom = avg_gain + avg_loss;
                return denom == 0.0 ? 50.0 : SCALE_100 * avg_gain / denom;
            }
            return NAN;
        }

        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);
        avg_gain = fma(avg_gain, beta, inv_p * gain);
        avg_loss = fma(avg_loss, beta, inv_p * loss);
        const double denom = avg_gain + avg_loss;
        return denom == 0.0 ? 50.0 : SCALE_100 * avg_gain / denom;
    }
};

struct SmaState {
    int period;
    double* buf;
    int head;
    int len;
    double sum;

    __device__ void init(int value, double* storage) {
        period = value;
        buf = storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        len = 0;
        sum = 0.0;
    }

    __device__ double update(double value) {
        if (len < period) {
            buf[head] = value;
            sum += value;
            head += 1;
            if (head == period) {
                head = 0;
            }
            len += 1;
            if (len == period) {
                return sum / static_cast<double>(period);
            }
            return NAN;
        }

        const double old = buf[head];
        buf[head] = value;
        sum += value - old;
        head += 1;
        if (head == period) {
            head = 0;
        }
        return sum / static_cast<double>(period);
    }
};
}

extern "C" __global__ void stochastic_connors_rsi_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ stoch_lengths,
    const int* __restrict__ smooth_ks,
    const int* __restrict__ smooth_ds,
    const int* __restrict__ rsi_lengths,
    const int* __restrict__ updown_lengths,
    const int* __restrict__ roc_lengths,
    int rows,
    double* __restrict__ out_k,
    double* __restrict__ out_d
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int stoch_length = stoch_lengths[row];
    const int smooth_k = smooth_ks[row];
    const int smooth_d = smooth_ds[row];
    const int rsi_length = rsi_lengths[row];
    const int updown_length = updown_lengths[row];
    const int roc_length = roc_lengths[row];

    double* row_k = out_k + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_d = out_d + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_k[i] = NAN;
        row_d[i] = NAN;
    }

    if (stoch_length <= 0 || stoch_length > len || smooth_k <= 0 || smooth_k > len ||
        smooth_d <= 0 || smooth_d > len || rsi_length <= 0 || rsi_length > len ||
        updown_length <= 0 || updown_length > len || roc_length <= 0 || roc_length > len) {
        return;
    }

    double* roc_window = new double[roc_length];
    double* crsi_window = new double[stoch_length];
    double* k_buf = new double[smooth_k];
    double* d_buf = new double[smooth_d];
    if (roc_window == nullptr || crsi_window == nullptr || k_buf == nullptr || d_buf == nullptr) {
        delete[] roc_window;
        delete[] crsi_window;
        delete[] k_buf;
        delete[] d_buf;
        return;
    }

    WilderRsiState src_rsi;
    WilderRsiState streak_rsi;
    src_rsi.init(rsi_length);
    streak_rsi.init(updown_length);

    SmaState k_sma;
    SmaState d_sma;
    k_sma.init(smooth_k, k_buf);
    d_sma.init(smooth_d, d_buf);

    bool has_prev_source = false;
    double prev_source = NAN;
    long long streak = 0;

    int roc_head = 0;
    int roc_count = 0;
    int crsi_head = 0;
    int crsi_count = 0;

    for (int i = 0; i < len; ++i) {
        const double source = data[i];
        if (!isfinite(source)) {
            has_prev_source = false;
            prev_source = NAN;
            streak = 0;
            src_rsi.reset();
            streak_rsi.reset();
            k_sma.reset();
            d_sma.reset();
            roc_head = 0;
            roc_count = 0;
            crsi_head = 0;
            crsi_count = 0;
            continue;
        }

        const bool had_prev = has_prev_source;
        const double prev_value = prev_source;
        if (had_prev) {
            if (source > prev_value) {
                streak = streak >= 0 ? streak + 1 : 1;
            } else if (source < prev_value) {
                streak = streak <= 0 ? streak - 1 : -1;
            } else {
                streak = 0;
            }
        } else {
            streak = 0;
        }
        prev_source = source;
        has_prev_source = true;

        const double src_value = src_rsi.update(source);
        const double streak_value = streak_rsi.update(static_cast<double>(streak));

        double percent_rank = NAN;
        if (had_prev) {
            const double roc =
                (prev_value == 0.0 || !isfinite(prev_value)) ? 0.0 : fma(source / prev_value, SCALE_100, -SCALE_100);

            if (roc_count < roc_length) {
                roc_window[roc_count] = roc;
                roc_count += 1;
            } else {
                roc_window[roc_head] = roc;
                roc_head += 1;
                if (roc_head == roc_length) {
                    roc_head = 0;
                }
            }

            if (roc_count == roc_length) {
                int count = 0;
                for (int j = 0; j < roc_count; ++j) {
                    if (roc_window[j] <= roc) {
                        count += 1;
                    }
                }
                percent_rank = SCALE_100 * static_cast<double>(count) / static_cast<double>(roc_length);
            }
        }

        if (!isfinite(src_value) || !isfinite(streak_value) || !isfinite(percent_rank)) {
            continue;
        }

        const double crsi = (src_value + streak_value + percent_rank) / 3.0;
        if (crsi_count < stoch_length) {
            crsi_window[crsi_count] = crsi;
            crsi_count += 1;
        } else {
            crsi_window[crsi_head] = crsi;
            crsi_head += 1;
            if (crsi_head == stoch_length) {
                crsi_head = 0;
            }
        }

        if (crsi_count < stoch_length) {
            continue;
        }

        double ll = crsi_window[0];
        double hh = crsi_window[0];
        for (int j = 1; j < crsi_count; ++j) {
            const double value = crsi_window[j];
            if (value < ll) {
                ll = value;
            }
            if (value > hh) {
                hh = value;
            }
        }

        const double denom = hh - ll;
        const double raw_k =
            fabs(denom) < EPS ? 0.0 : (crsi - ll) * (SCALE_100 / denom);
        const double k = k_sma.update(raw_k);
        if (!isfinite(k)) {
            continue;
        }
        const double d = d_sma.update(k);

        row_k[i] = k;
        row_d[i] = d;
    }

    delete[] roc_window;
    delete[] crsi_window;
    delete[] k_buf;
    delete[] d_buf;
}
