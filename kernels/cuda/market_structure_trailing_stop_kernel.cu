#include <cmath>
#include <cstddef>

namespace {
constexpr int RESET_ON_CHOCH = 0;
constexpr int RESET_ON_ALL = 1;

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline bool is_pivot_high(
    const double* high,
    int center,
    int length
) {
    const double pivot = high[center];
    for (int idx = center - length; idx < center; ++idx) {
        if (high[idx] > pivot) {
            return false;
        }
    }
    for (int idx = center + 1; idx <= center + length; ++idx) {
        if (high[idx] >= pivot) {
            return false;
        }
    }
    return true;
}

__device__ inline bool is_pivot_low(
    const double* low,
    int center,
    int length
) {
    const double pivot = low[center];
    for (int idx = center - length; idx < center; ++idx) {
        if (low[idx] < pivot) {
            return false;
        }
    }
    for (int idx = center + 1; idx <= center + length; ++idx) {
        if (low[idx] <= pivot) {
            return false;
        }
    }
    return true;
}
}

extern "C" __global__ void market_structure_trailing_stop_batch_f64(
    const double* open,
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* lengths,
    const double* increment_factors,
    int rows,
    int reset_mode,
    double* out_trailing_stop,
    double* out_state,
    double* out_structure
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double increment_factor = increment_factors[row];

    double* row_trailing_stop =
        out_trailing_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_structure = out_structure + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_trailing_stop[i] = NAN;
        row_state[i] = NAN;
        row_structure[i] = NAN;
    }

    if (length <= 0 || !isfinite(increment_factor) || increment_factor < 0.0) {
        return;
    }

    const int needed = 2 * length + 1;
    const double incr = increment_factor / 100.0;
    int idx = 0;
    while (idx < len) {
        while (idx < len && !is_valid_ohlc(open[idx], high[idx], low[idx], close[idx])) {
            idx += 1;
        }
        const int start = idx;
        while (idx < len && is_valid_ohlc(open[idx], high[idx], low[idx], close[idx])) {
            idx += 1;
        }
        const int end = idx;
        if (end - start < needed) {
            continue;
        }

        double ph_y = NAN;
        int ph_x = 0;
        double pl_y = NAN;
        int pl_x = 0;
        bool ph_cross = false;
        bool pl_cross = false;
        double top = NAN;
        double btm = NAN;
        double max_close = NAN;
        double min_close = NAN;
        double ts = NAN;
        int os = 0;

        for (int local = 0; local < end - start; ++local) {
            const int i = start + local;
            int ms = 0;

            if (local >= 2 * length) {
                const int center = i - length;
                if (is_pivot_high(high, center, length)) {
                    ph_y = high[center];
                    ph_x = center;
                    ph_cross = false;
                }
                if (is_pivot_low(low, center, length)) {
                    pl_y = low[center];
                    pl_x = center;
                    pl_cross = false;
                }
            }

            const double c = close[i];

            if (isfinite(ph_y) && !ph_cross && c > ph_y) {
                ms = (reset_mode == RESET_ON_ALL || (reset_mode == RESET_ON_CHOCH && os == -1))
                    ? 1
                    : 0;
                ph_cross = true;
                os = 1;
                btm = low[i];
                for (int scan = i; scan > ph_x; --scan) {
                    btm = fmin(btm, low[scan]);
                }
            }

            if (isfinite(pl_y) && !pl_cross && c < pl_y) {
                ms = (reset_mode == RESET_ON_ALL || (reset_mode == RESET_ON_CHOCH && os == 1))
                    ? -1
                    : 0;
                pl_cross = true;
                os = -1;
                top = high[i];
                for (int scan = i; scan > pl_x; --scan) {
                    top = fmax(top, high[scan]);
                }
            }

            const double prev_max = max_close;
            const double prev_min = min_close;

            if (ms == 1) {
                max_close = c;
            } else if (ms == -1) {
                min_close = c;
            } else {
                if (isfinite(max_close) && c > max_close) {
                    max_close = c;
                }
                if (isfinite(min_close) && c < min_close) {
                    min_close = c;
                }
            }

            if (ms == 1) {
                ts = btm;
            } else if (ms == -1) {
                ts = top;
            } else if (os == 1) {
                ts = (isfinite(ts) && isfinite(max_close) && isfinite(prev_max))
                    ? (ts + (max_close - prev_max) * incr)
                    : NAN;
            } else {
                ts = (isfinite(ts) && isfinite(min_close) && isfinite(prev_min))
                    ? (ts + (min_close - prev_min) * incr)
                    : NAN;
            }

            row_trailing_stop[i] = ts;
            row_state[i] = static_cast<double>(os);
            row_structure[i] = static_cast<double>(ms);
        }
    }
}
