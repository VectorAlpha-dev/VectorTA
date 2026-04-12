#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void stochastic_money_flow_index_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ stoch_k_lengths,
    const int* __restrict__ stoch_k_smooths,
    const int* __restrict__ stoch_d_smooths,
    const int* __restrict__ mfi_lengths,
    int n_combos,
    int max_flow_len,
    int max_stoch_k_length,
    int max_k_smooth,
    int max_d_smooth,
    double* __restrict__ pos_buf,
    double* __restrict__ neg_buf,
    int* __restrict__ maxdq_idx,
    double* __restrict__ maxdq_val,
    int* __restrict__ mindq_idx,
    double* __restrict__ mindq_val,
    double* __restrict__ k_buf,
    double* __restrict__ d_buf,
    double* __restrict__ out_k,
    double* __restrict__ out_d
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int stoch_k_length = stoch_k_lengths[combo_idx];
    int stoch_k_smooth = stoch_k_smooths[combo_idx];
    int stoch_d_smooth = stoch_d_smooths[combo_idx];
    int mfi_length = mfi_lengths[combo_idx];
    int flow_len = mfi_length - 1;

    double* row_k = out_k + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_d = out_d + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* pos = pos_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_flow_len);
    double* neg = neg_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_flow_len);
    int* max_idx = maxdq_idx + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_stoch_k_length);
    double* max_val = maxdq_val + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_stoch_k_length);
    int* min_idx = mindq_idx + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_stoch_k_length);
    double* min_val = mindq_val + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_stoch_k_length);
    double* k_ring = k_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_k_smooth);
    double* d_ring = d_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_d_smooth);

    for (int i = 0; i < len; ++i) {
        row_k[i] = CUDART_NAN;
        row_d[i] = CUDART_NAN;
    }

    if (stoch_k_length <= 0 || stoch_k_smooth <= 0 || stoch_d_smooth <= 0 || mfi_length <= 0) {
        return;
    }
    if (flow_len > max_flow_len || stoch_k_length > max_stoch_k_length ||
        stoch_k_smooth > max_k_smooth || stoch_d_smooth > max_d_smooth) {
        return;
    }

    int flow_head = 0;
    int flow_count = 0;
    double pos_sum = 0.0;
    double neg_sum = 0.0;
    double prev_source = CUDART_NAN;
    bool has_prev = false;
    int mfi_index = 0;
    int max_head = 0;
    int max_size = 0;
    int min_head = 0;
    int min_size = 0;
    int k_head = 0;
    int k_len = 0;
    double k_sum = 0.0;
    int d_head = 0;
    int d_len = 0;
    double d_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double src = source[i];
        double vol = volume[i];
        if (!isfinite(src) || !isfinite(vol)) {
            flow_head = 0;
            flow_count = 0;
            pos_sum = 0.0;
            neg_sum = 0.0;
            prev_source = CUDART_NAN;
            has_prev = false;
            mfi_index = 0;
            max_head = 0;
            max_size = 0;
            min_head = 0;
            min_size = 0;
            k_head = 0;
            k_len = 0;
            k_sum = 0.0;
            d_head = 0;
            d_len = 0;
            d_sum = 0.0;
            continue;
        }

        bool have_mfi = false;
        double mfi = 0.0;
        if (!has_prev) {
            prev_source = src;
            has_prev = true;
            if (mfi_length == 1) {
                have_mfi = true;
                mfi = 0.0;
            }
        } else {
            double diff = src - prev_source;
            prev_source = src;

            if (flow_len == 0) {
                have_mfi = true;
                mfi = 0.0;
            } else {
                double flow = src * vol;
                double pos_new = diff > 0.0 ? flow : 0.0;
                double neg_new = diff < 0.0 ? flow : 0.0;

                if (flow_count == flow_len) {
                    pos_sum -= pos[flow_head];
                    neg_sum -= neg[flow_head];
                } else {
                    flow_count += 1;
                }

                pos[flow_head] = pos_new;
                neg[flow_head] = neg_new;
                pos_sum += pos_new;
                neg_sum += neg_new;
                flow_head += 1;
                if (flow_head == flow_len) {
                    flow_head = 0;
                }

                if (flow_count == flow_len) {
                    double total = pos_sum + neg_sum;
                    mfi = total <= 1e-14 ? 0.0 : 100.0 * pos_sum / total;
                    have_mfi = true;
                }
            }
        }

        if (!have_mfi) {
            continue;
        }

        int window_start = mfi_index + 1 - stoch_k_length;
        if (window_start < 0) {
            window_start = 0;
        }
        while (max_size > 0 && max_idx[max_head] < window_start) {
            max_head += 1;
            if (max_head == stoch_k_length) {
                max_head = 0;
            }
            max_size -= 1;
        }
        while (min_size > 0 && min_idx[min_head] < window_start) {
            min_head += 1;
            if (min_head == stoch_k_length) {
                min_head = 0;
            }
            min_size -= 1;
        }

        while (max_size > 0) {
            int back_pos = max_head + max_size - 1;
            if (back_pos >= stoch_k_length) {
                back_pos -= stoch_k_length;
            }
            if (max_val[back_pos] <= mfi) {
                max_size -= 1;
            } else {
                break;
            }
        }
        int max_insert = max_head + max_size;
        if (max_insert >= stoch_k_length) {
            max_insert -= stoch_k_length;
        }
        max_idx[max_insert] = mfi_index;
        max_val[max_insert] = mfi;
        max_size += 1;

        while (min_size > 0) {
            int back_pos = min_head + min_size - 1;
            if (back_pos >= stoch_k_length) {
                back_pos -= stoch_k_length;
            }
            if (min_val[back_pos] >= mfi) {
                min_size -= 1;
            } else {
                break;
            }
        }
        int min_insert = min_head + min_size;
        if (min_insert >= stoch_k_length) {
            min_insert -= stoch_k_length;
        }
        min_idx[min_insert] = mfi_index;
        min_val[min_insert] = mfi;
        min_size += 1;
        mfi_index += 1;

        if (mfi_index < stoch_k_length) {
            continue;
        }

        double highest = max_size > 0 ? max_val[max_head] : mfi;
        double lowest = min_size > 0 ? min_val[min_head] : mfi;
        double raw_k = (highest - lowest) > DBL_EPSILON
            ? 100.0 * (mfi - lowest) / (highest - lowest)
            : 0.0;

        if (stoch_k_smooth == 1) {
            row_k[i] = raw_k;
        } else {
            if (k_len == stoch_k_smooth) {
                k_sum -= k_ring[k_head];
            } else {
                k_len += 1;
            }
            k_ring[k_head] = raw_k;
            k_sum += raw_k;
            k_head += 1;
            if (k_head == stoch_k_smooth) {
                k_head = 0;
            }
            if (k_len < stoch_k_smooth) {
                continue;
            }
            row_k[i] = k_sum / static_cast<double>(stoch_k_smooth);
        }

        double k_value = row_k[i];
        if (stoch_d_smooth == 1) {
            row_d[i] = k_value;
            continue;
        }
        if (d_len == stoch_d_smooth) {
            d_sum -= d_ring[d_head];
        } else {
            d_len += 1;
        }
        d_ring[d_head] = k_value;
        d_sum += k_value;
        d_head += 1;
        if (d_head == stoch_d_smooth) {
            d_head = 0;
        }
        row_d[i] = d_len < stoch_d_smooth
            ? CUDART_NAN
            : d_sum / static_cast<double>(stoch_d_smooth);
    }
}
