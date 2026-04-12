#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double trend_trigger_factor_value(
    double hh,
    double ll,
    double hist_hh,
    double hist_ll
) {
    double buy_power = hh - hist_ll;
    double sell_power = hist_hh - ll;
    double denom = buy_power + sell_power;
    if (isfinite(denom) && denom != 0.0) {
        return 200.0 * (buy_power - sell_power) / denom;
    }
    return CUDART_NAN;
}

extern "C" __global__ void trend_trigger_factor_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    int first_valid,
    const int* __restrict__ lengths,
    int n_combos,
    int max_length,
    int* __restrict__ maxq_idx,
    int* __restrict__ minq_idx,
    double* __restrict__ hh_history,
    double* __restrict__ ll_history,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    int* maxq = maxq_idx + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    int* minq = minq_idx + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* hh_hist =
        hh_history + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* ll_hist =
        ll_history + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length <= 0 || length > max_length) {
        return;
    }

    int warm = first_valid + length - 1;
    int maxq_head = 0;
    int maxq_size = 0;
    int minq_head = 0;
    int minq_size = 0;
    int hist_head = 0;
    int hist_size = 0;

    for (int i = first_valid; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        if (!isfinite(h) || !isfinite(l)) {
            if (i >= warm) {
                row[i] = CUDART_NAN;
            }
            continue;
        }

        int window_start = i + 1 - length;
        if (window_start < first_valid) {
            window_start = first_valid;
        }

        while (maxq_size > 0) {
            int front_idx = maxq[maxq_head];
            if (front_idx < window_start) {
                maxq_head += 1;
                if (maxq_head == length) {
                    maxq_head = 0;
                }
                maxq_size -= 1;
            } else {
                break;
            }
        }

        while (minq_size > 0) {
            int front_idx = minq[minq_head];
            if (front_idx < window_start) {
                minq_head += 1;
                if (minq_head == length) {
                    minq_head = 0;
                }
                minq_size -= 1;
            } else {
                break;
            }
        }

        while (maxq_size > 0) {
            int back_pos = maxq_head + maxq_size - 1;
            if (back_pos >= length) {
                back_pos -= length;
            }
            int back_idx = maxq[back_pos];
            if (high[back_idx] <= h) {
                maxq_size -= 1;
            } else {
                break;
            }
        }
        int max_insert = maxq_head + maxq_size;
        if (max_insert >= length) {
            max_insert -= length;
        }
        maxq[max_insert] = i;
        maxq_size += 1;

        while (minq_size > 0) {
            int back_pos = minq_head + minq_size - 1;
            if (back_pos >= length) {
                back_pos -= length;
            }
            int back_idx = minq[back_pos];
            if (low[back_idx] >= l) {
                minq_size -= 1;
            } else {
                break;
            }
        }
        int min_insert = minq_head + minq_size;
        if (min_insert >= length) {
            min_insert -= length;
        }
        minq[min_insert] = i;
        minq_size += 1;

        if (i >= warm) {
            double hh = high[maxq[maxq_head]];
            double ll = low[minq[minq_head]];
            double hist_hh = hist_size == length ? hh_hist[hist_head] : 0.0;
            double hist_ll = hist_size == length ? ll_hist[hist_head] : 0.0;
            row[i] = trend_trigger_factor_value(hh, ll, hist_hh, hist_ll);

            int hist_insert = hist_head + hist_size;
            if (hist_insert >= length) {
                hist_insert -= length;
            }
            if (hist_size < length) {
                hh_hist[hist_insert] = hh;
                ll_hist[hist_insert] = ll;
                hist_size += 1;
            } else {
                hh_hist[hist_head] = hh;
                ll_hist[hist_head] = ll;
                hist_head += 1;
                if (hist_head == length) {
                    hist_head = 0;
                }
            }
        }
    }
}
