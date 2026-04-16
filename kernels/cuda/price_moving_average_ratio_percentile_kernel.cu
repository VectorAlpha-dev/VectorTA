#include <cmath>
#include <cstddef>

namespace {
constexpr double EPS = 1e-12;
constexpr int MA_SMA = 0;
constexpr int MA_EMA = 1;
constexpr int MA_HMA = 2;
constexpr int MA_RMA = 3;
constexpr int MA_VWMA = 4;
constexpr int LINE_PMAR = 0;
constexpr int LINE_PMARP = 1;

__device__ inline int max_i(int a, int b) {
    return a > b ? a : b;
}

__device__ inline int sqrt_period(int period) {
    const double root = floor(sqrt(static_cast<double>(period)));
    const int value = static_cast<int>(root);
    return value > 0 ? value : 1;
}

struct EmaState {
    int period;
    int count;
    double alpha;
    double beta;
    double mean;
    bool has_value;

    __device__ void init(int period) {
        this->period = period;
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        beta = 1.0 - alpha;
        reset();
    }

    __device__ void reset() {
        count = 0;
        mean = NAN;
        has_value = false;
    }

    __device__ bool update(double input, double* out) {
        if (!isfinite(input)) {
            *out = has_value ? mean : NAN;
            return has_value;
        }
        count += 1;
        if (count == 1) {
            mean = input;
        } else if (count <= period) {
            mean += (input - mean) / static_cast<double>(count);
        } else {
            mean = alpha * input + beta * mean;
        }
        has_value = true;
        *out = mean;
        return true;
    }
};

struct RmaState {
    int period;
    int count;
    double sum;
    double value;
    bool seeded;

    __device__ void init(int p) {
        period = p;
        reset();
    }

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
        seeded = false;
    }

    __device__ bool update(double input, double* out) {
        if (!isfinite(input)) {
            *out = seeded ? value : NAN;
            return seeded;
        }
        if (seeded) {
            value = ((value * static_cast<double>(period - 1)) + input) / static_cast<double>(period);
            *out = value;
            return true;
        }
        count += 1;
        sum += input;
        if (count == period) {
            value = sum / static_cast<double>(period);
            seeded = true;
            *out = value;
            return true;
        }
        *out = NAN;
        return false;
    }
};

struct SmaState {
    double* ring;
    int period;
    int head;
    int count;
    double sum;

    __device__ void init(int p, double* storage) {
        period = p;
        ring = storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (!isfinite(input)) {
            *out = NAN;
            return false;
        }
        if (count < period) {
            ring[count] = input;
            count += 1;
            sum += input;
            if (count == period) {
                *out = sum / static_cast<double>(period);
                return true;
            }
            *out = NAN;
            return false;
        }
        const double old = ring[head];
        ring[head] = input;
        head += 1;
        if (head == period) {
            head = 0;
        }
        sum += input - old;
        *out = sum / static_cast<double>(period);
        return true;
    }
};

struct WmaState {
    double* ring;
    int period;
    int head;
    int count;
    double sum;
    double wsum;
    double inv_norm;

    __device__ void init(int p, double* storage) {
        period = p;
        ring = storage;
        const double norm = static_cast<double>(period) * (static_cast<double>(period) + 1.0) * 0.5;
        inv_norm = 1.0 / norm;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
        wsum = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (!isfinite(input)) {
            *out = NAN;
            return false;
        }
        if (count < period) {
            ring[count] = input;
            count += 1;
            sum += input;
            wsum += static_cast<double>(count) * input;
            if (count == period) {
                *out = wsum * inv_norm;
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old = ring[head];
        ring[head] = input;
        head += 1;
        if (head == period) {
            head = 0;
        }
        const double prev_sum = sum;
        sum = prev_sum + input - old;
        wsum = static_cast<double>(period) * input + wsum - prev_sum;
        *out = wsum * inv_norm;
        return true;
    }
};

struct VwmaState {
    double* pv_ring;
    double* vol_ring;
    int period;
    int head;
    int count;
    double pv_sum;
    double vol_sum;

    __device__ void init(int p, double* pv_storage, double* vol_storage) {
        period = p;
        pv_ring = pv_storage;
        vol_ring = vol_storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        pv_sum = 0.0;
        vol_sum = 0.0;
    }

    __device__ bool update(double value, double volume, double* out) {
        if (!isfinite(value) || !isfinite(volume)) {
            *out = NAN;
            return false;
        }
        const double pv = value * volume;
        if (count < period) {
            pv_ring[count] = pv;
            vol_ring[count] = volume;
            count += 1;
            pv_sum += pv;
            vol_sum += volume;
            if (count == period) {
                *out = fabs(vol_sum) <= EPS ? NAN : (pv_sum / vol_sum);
                return fabs(vol_sum) > EPS;
            }
            *out = NAN;
            return false;
        }

        const double old_pv = pv_ring[head];
        const double old_vol = vol_ring[head];
        pv_ring[head] = pv;
        vol_ring[head] = volume;
        head += 1;
        if (head == period) {
            head = 0;
        }
        pv_sum += pv - old_pv;
        vol_sum += volume - old_vol;
        *out = fabs(vol_sum) <= EPS ? NAN : (pv_sum / vol_sum);
        return fabs(vol_sum) > EPS;
    }
};

struct HmaState {
    WmaState wma_half;
    WmaState wma_full;
    WmaState wma_sqrt;

    __device__ void init(int period, double* half_storage, double* full_storage, double* sqrt_storage) {
        const int half = max_i(period / 2, 1);
        wma_half.init(half, half_storage);
        wma_full.init(period, full_storage);
        wma_sqrt.init(sqrt_period(period), sqrt_storage);
    }

    __device__ bool update(double input, double* out) {
        double half_value = NAN;
        double full_value = NAN;
        const bool half_ready = wma_half.update(input, &half_value);
        const bool full_ready = wma_full.update(input, &full_value);
        if (half_ready && full_ready) {
            const double diff = 2.0 * half_value - full_value;
            return wma_sqrt.update(diff, out);
        }
        *out = NAN;
        return false;
    }
};

__device__ inline double scaled_pmar_value(double pmar, double pmar_high, double pmar_low) {
    if (pmar >= 1.0) {
        const double denom = pmar_high - 1.0;
        if (fabs(denom) <= EPS) {
            return 50.0;
        }
        return (((pmar - 1.0) * (100.0 / denom)) / 2.0) + 50.0;
    }
    const double denom = 1.0 - pmar_low;
    if (fabs(denom) <= EPS) {
        return 50.0;
    }
    return ((pmar - pmar_low) * (100.0 / denom)) / 2.0;
}

__device__ bool update_ma_value(
    int ma_code,
    double value,
    double volume,
    EmaState* ema,
    RmaState* rma,
    SmaState* sma,
    WmaState* wma,
    VwmaState* vwma,
    HmaState* hma,
    double* out
) {
    switch (ma_code) {
        case MA_SMA:
            return sma->update(value, out);
        case MA_EMA:
            return ema->update(value, out);
        case MA_HMA:
            return hma->update(value, out);
        case MA_RMA:
            return rma->update(value, out);
        case MA_VWMA:
            return vwma->update(value, volume, out);
        default:
            *out = NAN;
            return false;
    }
}
}

extern "C" __global__ void price_moving_average_ratio_percentile_batch_f64(
    const double* price,
    const double* volume,
    int len,
    const int* ma_lengths,
    const int* pmarp_lookbacks,
    const int* signal_ma_lengths,
    const int* ma_codes,
    const int* signal_ma_codes,
    const int* line_modes,
    int rows,
    int scratch_cap,
    double* scratch,
    double* out_pmar,
    double* out_pmarp,
    double* out_plotline,
    double* out_signal,
    double* out_pmar_high,
    double* out_pmar_low,
    double* out_scaled_pmar
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int ma_length = ma_lengths[row];
    const int pmarp_lookback = pmarp_lookbacks[row];
    const int signal_ma_length = signal_ma_lengths[row];
    const int ma_code = ma_codes[row];
    const int signal_ma_code = signal_ma_codes[row];
    const int line_mode = line_modes[row];

    double* row_pmar = out_pmar + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_pmarp = out_pmarp + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_plotline = out_plotline + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_pmar_high = out_pmar_high + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_pmar_low = out_pmar_low + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_scaled_pmar = out_scaled_pmar + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_pmar[i] = NAN;
        row_pmarp[i] = NAN;
        row_plotline[i] = NAN;
        row_signal[i] = NAN;
        row_pmar_high[i] = NAN;
        row_pmar_low[i] = NAN;
        row_scaled_pmar[i] = NAN;
    }

    if (ma_length <= 0 || pmarp_lookback <= 0 || signal_ma_length <= 0 || scratch_cap <= 0) {
        return;
    }
    if (ma_length > scratch_cap || signal_ma_length > scratch_cap) {
        return;
    }
    if (ma_code < MA_SMA || ma_code > MA_VWMA || signal_ma_code < MA_SMA || signal_ma_code > MA_VWMA) {
        return;
    }
    if (line_mode != LINE_PMAR && line_mode != LINE_PMARP) {
        return;
    }

    double* row_scratch =
        scratch + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap) * static_cast<size_t>(10);
    double* main_a = row_scratch + static_cast<size_t>(scratch_cap) * 0;
    double* main_b = row_scratch + static_cast<size_t>(scratch_cap) * 1;
    double* main_c = row_scratch + static_cast<size_t>(scratch_cap) * 2;
    double* main_d = row_scratch + static_cast<size_t>(scratch_cap) * 3;
    double* main_e = row_scratch + static_cast<size_t>(scratch_cap) * 4;
    double* signal_a = row_scratch + static_cast<size_t>(scratch_cap) * 5;
    double* signal_b = row_scratch + static_cast<size_t>(scratch_cap) * 6;
    double* signal_c = row_scratch + static_cast<size_t>(scratch_cap) * 7;
    double* signal_d = row_scratch + static_cast<size_t>(scratch_cap) * 8;
    double* signal_e = row_scratch + static_cast<size_t>(scratch_cap) * 9;

    EmaState main_ema;
    RmaState main_rma;
    SmaState main_sma;
    WmaState main_wma;
    VwmaState main_vwma;
    HmaState main_hma;
    EmaState signal_ema;
    RmaState signal_rma;
    SmaState signal_sma;
    WmaState signal_wma;
    VwmaState signal_vwma;
    HmaState signal_hma;

    main_ema.init(ma_length);
    main_rma.init(ma_length);
    main_sma.init(ma_length, main_a);
    main_wma.init(ma_length, main_a);
    main_vwma.init(ma_length, main_d, main_e);
    main_hma.init(ma_length, main_a, main_b, main_c);

    signal_ema.init(signal_ma_length);
    signal_rma.init(signal_ma_length);
    signal_sma.init(signal_ma_length, signal_a);
    signal_wma.init(signal_ma_length, signal_a);
    signal_vwma.init(signal_ma_length, signal_d, signal_e);
    signal_hma.init(signal_ma_length, signal_a, signal_b, signal_c);

    bool seen_pmar = false;
    double pmar_high = 1.0;
    double pmar_low = 1.0;

    for (int i = 0; i < len; ++i) {
        const double current_price = price[i];
        const double current_volume = volume[i];
        double ma_value = NAN;
        if (update_ma_value(
                ma_code,
                current_price,
                current_volume,
                &main_ema,
                &main_rma,
                &main_sma,
                &main_wma,
                &main_vwma,
                &main_hma,
                &ma_value) &&
            isfinite(current_price) &&
            isfinite(ma_value) &&
            fabs(ma_value) > EPS) {
            const double pmar = current_price / ma_value;
            row_pmar[i] = pmar;
            pmar_high = fmax(pmar_high, pmar);
            pmar_low = fmin(pmar_low, pmar);
            seen_pmar = true;
        }

        if (seen_pmar) {
            row_pmar_high[i] = pmar_high;
            row_pmar_low[i] = pmar_low;
            if (isfinite(row_pmar[i])) {
                row_scaled_pmar[i] = scaled_pmar_value(row_pmar[i], pmar_high, pmar_low);
            }
        }

        if (i >= ma_length) {
            const double current = fabs(row_pmar[i]);
            if (isfinite(current)) {
                const int lookback = i < pmarp_lookback ? i : pmarp_lookback;
                if (lookback > 0) {
                    int count = 0;
                    for (int offset = 1; offset <= lookback; ++offset) {
                        const double prev = fabs(row_pmar[i - offset]);
                        if (!(isfinite(prev) && prev > current)) {
                            count += 1;
                        }
                    }
                    row_pmarp[i] = (static_cast<double>(count) / static_cast<double>(lookback)) * 100.0;
                }
            }
        }

        const double plotline = line_mode == LINE_PMAR ? row_pmar[i] : row_pmarp[i];
        row_plotline[i] = plotline;

        double signal_value = NAN;
        if (update_ma_value(
                signal_ma_code,
                plotline,
                current_volume,
                &signal_ema,
                &signal_rma,
                &signal_sma,
                &signal_wma,
                &signal_vwma,
                &signal_hma,
                &signal_value)) {
            row_signal[i] = signal_value;
        }
    }
}
