#include <cmath>
#include <cstddef>

namespace {

constexpr double PI_CONST = 3.14159265358979323846264338327950288;
constexpr double SQRT_2_CONST = 1.41421356237309504880168872420969808;

__device__ inline int corr_window_device(int avg_length, int lag) {
    if (avg_length == 0) {
        return lag > 2 ? lag : 2;
    }
    return avg_length > 2 ? avg_length : 2;
}

__device__ inline int warmup_period_device(int max_period, int avg_length) {
    return max_period + corr_window_device(avg_length, max_period) - 1;
}

__device__ inline double highpass_alpha_device(int max_period) {
    const double angle = SQRT_2_CONST * PI_CONST / static_cast<double>(max_period);
    return (cos(angle) + sin(angle) - 1.0) / cos(angle);
}

struct PeriodogramState {
    int min_period;
    int max_period;
    int avg_length;
    bool enhance;
    double prev_price_1;
    double prev_price_2;
    double hp_prev_1;
    double hp_prev_2;
    double filt_prev_1;
    double filt_prev_2;
    double* filt_history;
    int history_cap;
    int history_head;
    int history_count;
    double* corr;
    double* power;
    double* smooth;
    double dom;
    double max_pwr;
    double e;
    bool warmup_bias;
    int bars_seen;

    __device__ void init(
        int min_period_value,
        int max_period_value,
        int avg_length_value,
        bool enhance_value,
        double* history_storage,
        int history_cap_value,
        double* corr_storage,
        double* power_storage,
        double* smooth_storage
    ) {
        min_period = min_period_value;
        max_period = max_period_value;
        avg_length = avg_length_value;
        enhance = enhance_value;
        filt_history = history_storage;
        history_cap = history_cap_value;
        corr = corr_storage;
        power = power_storage;
        smooth = smooth_storage;
        reset();
    }

    __device__ void reset() {
        prev_price_1 = 0.0;
        prev_price_2 = 0.0;
        hp_prev_1 = 0.0;
        hp_prev_2 = 0.0;
        filt_prev_1 = 0.0;
        filt_prev_2 = 0.0;
        history_head = 0;
        history_count = 0;
        for (int i = 0; i < history_cap; ++i) {
            filt_history[i] = 0.0;
        }
        for (int i = 0; i <= max_period; ++i) {
            corr[i] = 0.0;
            power[i] = 0.0;
            smooth[i] = 0.0;
        }
        dom = 0.5 * static_cast<double>(min_period + max_period);
        max_pwr = 0.0;
        e = 1.0;
        warmup_bias = true;
        bars_seen = 0;
    }

    __device__ void push_filt(double value) {
        filt_history[history_head] = value;
        history_head += 1;
        if (history_head == history_cap) {
            history_head = 0;
        }
        if (history_count < history_cap) {
            history_count += 1;
        }
    }

    __device__ double filt_back(int back) const {
        if (back >= history_count) {
            return 0.0;
        }
        int idx = history_head - 1 - back;
        while (idx < 0) {
            idx += history_cap;
        }
        return filt_history[idx];
    }
};

}

extern "C" __global__ void ehlers_autocorrelation_periodogram_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ min_periods,
    const int* __restrict__ max_periods,
    const int* __restrict__ avg_lengths,
    const int* __restrict__ enhances,
    int rows,
    int scratch_cap,
    double* __restrict__ scratch_buf,
    double* __restrict__ out_dominant_cycle,
    double* __restrict__ out_normalized_power
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int min_period = min_periods[row];
    const int max_period = max_periods[row];
    const int avg_length = avg_lengths[row];
    const bool enhance = enhances[row] != 0;

    double* row_dom = out_dominant_cycle + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_pwr = out_normalized_power + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_dom[i] = NAN;
        row_pwr[i] = NAN;
    }

    if (min_period < 3 || max_period <= min_period || max_period > len) {
        return;
    }

    const int history_cap = max_period + corr_window_device(avg_length, max_period);
    const int needed = history_cap + 3 * (max_period + 1);
    if (needed > scratch_cap) {
        return;
    }

    double* row_scratch = scratch_buf + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap);
    double* history_storage = row_scratch;
    double* corr_storage = history_storage + history_cap;
    double* power_storage = corr_storage + (max_period + 1);
    double* smooth_storage = power_storage + (max_period + 1);

    PeriodogramState state;
    state.init(
        min_period,
        max_period,
        avg_length,
        enhance,
        history_storage,
        history_cap,
        corr_storage,
        power_storage,
        smooth_storage
    );

    const double alpha_hp = highpass_alpha_device(max_period);
    const double one_minus_hp = 1.0 - alpha_hp;
    const double hp_coeff = (1.0 - alpha_hp * 0.5) * (1.0 - alpha_hp * 0.5);
    const double a1 = exp(-SQRT_2_CONST * PI_CONST / static_cast<double>(min_period));
    const double b1 = 2.0 * a1 * cos(SQRT_2_CONST * PI_CONST / static_cast<double>(min_period));
    const double c2 = b1;
    const double c3 = -(a1 * a1);
    const double c1 = 1.0 - c2 - c3;
    const int warmup = warmup_period_device(max_period, avg_length);

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            state.reset();
            continue;
        }

        const double hp = hp_coeff * (value - 2.0 * state.prev_price_1 + state.prev_price_2) +
            2.0 * one_minus_hp * state.hp_prev_1 -
            (one_minus_hp * one_minus_hp) * state.hp_prev_2;
        const double filt =
            c1 * (hp + state.hp_prev_1) * 0.5 + c2 * state.filt_prev_1 + c3 * state.filt_prev_2;

        state.prev_price_2 = state.prev_price_1;
        state.prev_price_1 = value;
        state.hp_prev_2 = state.hp_prev_1;
        state.hp_prev_1 = hp;
        state.filt_prev_2 = state.filt_prev_1;
        state.filt_prev_1 = filt;
        state.push_filt(filt);
        state.bars_seen += 1;

        state.corr[0] = 0.0;
        if (state.max_period >= 1) {
            state.corr[1] = 0.0;
        }

        for (int lag = 2; lag <= state.max_period; ++lag) {
            const int window = corr_window_device(state.avg_length, lag);
            double sx = 0.0;
            double sy = 0.0;
            double sxx = 0.0;
            double syy = 0.0;
            double sxy = 0.0;
            for (int k = 0; k < window; ++k) {
                const double x = state.filt_back(k);
                const double y = state.filt_back(lag + k);
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
            const double valid = static_cast<double>(window);
            const double denom_x = valid * sxx - sx * sx;
            const double denom_y = valid * syy - sy * sy;
            const double denom = denom_x * denom_y;
            state.corr[lag] = denom > 0.0 ? (valid * sxy - sx * sy) / sqrt(denom) : 0.0;
        }

        double local_max_pwr = 0.0;
        for (int period = state.min_period; period <= state.max_period; ++period) {
            double cos_acc = 0.0;
            double sin_acc = 0.0;
            const double period_f = static_cast<double>(period);
            for (int n = 2; n <= state.max_period; ++n) {
                const double angle = 2.0 * PI_CONST * static_cast<double>(n) / period_f;
                const double corr = state.corr[n];
                cos_acc += corr * cos(angle);
                sin_acc += corr * sin(angle);
            }
            const double sq = cos_acc * cos_acc + sin_acc * sin_acc;
            const double smooth = 0.2 * sq * sq + 0.8 * state.smooth[period];
            state.smooth[period] = smooth;
            if (smooth > local_max_pwr) {
                local_max_pwr = smooth;
            }
        }

        const double diff = static_cast<double>(state.max_period - state.min_period);
        const double decay = diff > 0.0 ? pow(10.0, -0.15 / diff) : 1.0;
        if (local_max_pwr > state.max_pwr) {
            state.max_pwr = local_max_pwr;
        } else {
            state.max_pwr *= decay;
        }

        double weighted = 0.0;
        double sum_weight = 0.0;
        for (int period = state.min_period; period <= state.max_period; ++period) {
            double pwr = state.max_pwr > 0.0 ? state.smooth[period] / state.max_pwr : 0.0;
            if (state.enhance) {
                pwr = pwr * pwr * pwr;
            }
            state.power[period] = pwr;
            if (pwr >= 0.5) {
                weighted += static_cast<double>(period) * pwr;
                sum_weight += pwr;
            }
        }

        const double base = sum_weight >= 0.25 ? (weighted / sum_weight) : state.dom;
        state.dom += 0.2 * (base - state.dom);
        if (state.warmup_bias) {
            state.e *= 0.8;
            const double correction = 1.0 / (1.0 - state.e);
            state.dom *= correction;
            state.warmup_bias = state.e > 1e-10;
        }

        if (state.bars_seen <= warmup) {
            continue;
        }

        int dom_idx = static_cast<int>(llround(state.dom));
        if (dom_idx < state.min_period) {
            dom_idx = state.min_period;
        } else if (dom_idx > state.max_period) {
            dom_idx = state.max_period;
        }

        row_dom[i] = state.dom;
        row_pwr[i] = state.power[dom_idx];
    }
}
