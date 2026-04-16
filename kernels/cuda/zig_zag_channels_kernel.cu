#include <cmath>
#include <cstddef>

namespace {

struct PivotState {
    int confirm_idx;
    double value;
};

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline void shift_left(int* deque, int count) {
    for (int i = 1; i < count; ++i) {
        deque[i - 1] = deque[i];
    }
}

__device__ inline void compute_segment_offsets_abs(
    const double* open,
    const double* close,
    int seg_start,
    int start_idx,
    int end_idx,
    double start_value,
    double end_value,
    double* up_offset,
    double* dn_offset
) {
    if (end_idx <= start_idx) {
        *up_offset = 0.0;
        *dn_offset = 0.0;
        return;
    }

    if (end_idx == start_idx + 1) {
        const int abs_idx = seg_start + end_idx;
        const double top = fmax(open[abs_idx], close[abs_idx]);
        const double bottom = fmin(open[abs_idx], close[abs_idx]);
        *up_offset = fmax(top - end_value, 0.0);
        *dn_offset = fmax(end_value - bottom, 0.0);
        return;
    }

    double max_diff_up = 0.0;
    double max_diff_dn = 0.0;
    const double denom = static_cast<double>(end_idx - start_idx - 1);
    const double span = end_value - start_value;

    for (int idx = start_idx + 1; idx <= end_idx; ++idx) {
        const double j = static_cast<double>(idx - start_idx - 1);
        const double point = start_value + (j / denom) * span;
        const int abs_idx = seg_start + idx;
        const double top = fmax(open[abs_idx], close[abs_idx]);
        const double bottom = fmin(open[abs_idx], close[abs_idx]);
        max_diff_up = fmax(max_diff_up, top - point);
        max_diff_dn = fmax(max_diff_dn, point - bottom);
    }

    *up_offset = fmax(max_diff_up, 0.0);
    *dn_offset = fmax(max_diff_dn, 0.0);
}

__device__ inline void fill_segment_abs(
    double* middle,
    double* upper,
    double* lower,
    int seg_start,
    int start_idx,
    int end_idx,
    double start_value,
    double end_value,
    double up_offset,
    double dn_offset
) {
    if (end_idx < start_idx) {
        return;
    }

    if (start_idx == end_idx) {
        const int abs_idx = seg_start + start_idx;
        middle[abs_idx] = start_value;
        upper[abs_idx] = start_value + up_offset;
        lower[abs_idx] = start_value - dn_offset;
        return;
    }

    const double denom = static_cast<double>(end_idx - start_idx);
    const double span = end_value - start_value;
    for (int idx = start_idx; idx <= end_idx; ++idx) {
        const double t = static_cast<double>(idx - start_idx) / denom;
        const double value = start_value + t * span;
        const int abs_idx = seg_start + idx;
        middle[abs_idx] = value;
        upper[abs_idx] = value + up_offset;
        lower[abs_idx] = value - dn_offset;
    }
}

__device__ void compute_run_abs(
    const double* open,
    const double* high,
    const double* low,
    const double* close,
    int seg_start,
    int seg_len,
    int length,
    bool extend,
    int* max_deque,
    int* min_deque,
    double* middle,
    double* upper,
    double* lower
) {
    if (seg_len <= length) {
        return;
    }

    int max_count = 0;
    int min_count = 0;
    int os = 0;
    bool has_last_top = false;
    bool has_last_bottom = false;
    PivotState last_top{0, NAN};
    PivotState last_bottom{0, NAN};

    for (int idx = 0; idx < seg_len; ++idx) {
        const double current_close = close[seg_start + idx];

        while (max_count > 0 && close[seg_start + max_deque[max_count - 1]] <= current_close) {
            max_count -= 1;
        }
        max_deque[max_count] = idx;
        max_count += 1;

        while (min_count > 0 && close[seg_start + min_deque[min_count - 1]] >= current_close) {
            min_count -= 1;
        }
        min_deque[min_count] = idx;
        min_count += 1;

        if (idx < length) {
            continue;
        }

        const int window_start = idx + 1 - length;
        while (max_count > 0 && max_deque[0] < window_start) {
            shift_left(max_deque, max_count);
            max_count -= 1;
        }
        while (min_count > 0 && min_deque[0] < window_start) {
            shift_left(min_deque, min_count);
            min_count -= 1;
        }

        const int candidate = idx - length;
        const double upper_close = close[seg_start + max_deque[0]];
        const double lower_close = close[seg_start + min_deque[0]];
        const int prev_os = os;
        const double candidate_close = close[seg_start + candidate];

        if (candidate_close > upper_close) {
            os = 0;
        } else if (candidate_close < lower_close) {
            os = 1;
        }

        if (os == 1 && prev_os != 1) {
            const int end_idx = candidate;
            const double end_value = low[seg_start + end_idx];
            if (has_last_top) {
                const int start_idx = last_top.confirm_idx - length;
                double up_offset = 0.0;
                double dn_offset = 0.0;
                compute_segment_offsets_abs(
                    open,
                    close,
                    seg_start,
                    start_idx,
                    end_idx,
                    last_top.value,
                    end_value,
                    &up_offset,
                    &dn_offset
                );
                fill_segment_abs(
                    middle,
                    upper,
                    lower,
                    seg_start,
                    start_idx,
                    end_idx,
                    last_top.value,
                    end_value,
                    up_offset,
                    dn_offset
                );
            }
            last_bottom.confirm_idx = idx;
            last_bottom.value = end_value;
            has_last_bottom = true;
        }

        if (os == 0 && prev_os != 0) {
            const int end_idx = candidate;
            const double end_value = high[seg_start + end_idx];
            if (has_last_bottom) {
                const int start_idx = last_bottom.confirm_idx - length;
                double up_offset = 0.0;
                double dn_offset = 0.0;
                compute_segment_offsets_abs(
                    open,
                    close,
                    seg_start,
                    start_idx,
                    end_idx,
                    last_bottom.value,
                    end_value,
                    &up_offset,
                    &dn_offset
                );
                fill_segment_abs(
                    middle,
                    upper,
                    lower,
                    seg_start,
                    start_idx,
                    end_idx,
                    last_bottom.value,
                    end_value,
                    up_offset,
                    dn_offset
                );
            }
            last_top.confirm_idx = idx;
            last_top.value = end_value;
            has_last_top = true;
        }
    }

    if (!extend) {
        return;
    }

    const int end_idx = seg_len - 1;
    const double end_value = close[seg_start + end_idx];
    if (os == 1) {
        if (has_last_bottom) {
            const int start_idx = last_bottom.confirm_idx - length;
            double up_offset = 0.0;
            double dn_offset = 0.0;
            compute_segment_offsets_abs(
                open,
                close,
                seg_start,
                start_idx,
                end_idx,
                last_bottom.value,
                end_value,
                &up_offset,
                &dn_offset
            );
            fill_segment_abs(
                middle,
                upper,
                lower,
                seg_start,
                start_idx,
                end_idx,
                last_bottom.value,
                end_value,
                up_offset,
                dn_offset
            );
        }
    } else if (has_last_top) {
        const int start_idx = last_top.confirm_idx - length;
        double up_offset = 0.0;
        double dn_offset = 0.0;
        compute_segment_offsets_abs(
            open,
            close,
            seg_start,
            start_idx,
            end_idx,
            last_top.value,
            end_value,
            &up_offset,
            &dn_offset
        );
        fill_segment_abs(
            middle,
            upper,
            lower,
            seg_start,
            start_idx,
            end_idx,
            last_top.value,
            end_value,
            up_offset,
            dn_offset
        );
    }
}

}

extern "C" __global__ void zig_zag_channels_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ extends,
    int rows,
    int scratch_cap,
    int* __restrict__ scratch_buf,
    double* __restrict__ out_middle,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const bool extend = extends[row] != 0;

    double* row_middle = out_middle + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_middle[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
    }

    if (length <= 0 || length > scratch_cap) {
        return;
    }

    int* row_scratch = scratch_buf + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap * 2);
    int* max_deque = row_scratch;
    int* min_deque = row_scratch + scratch_cap;

    int idx = 0;
    while (idx < len) {
        while (idx < len && !is_valid_ohlc(open[idx], high[idx], low[idx], close[idx])) {
            idx += 1;
        }
        if (idx >= len) {
            break;
        }

        const int seg_start = idx;
        idx += 1;
        while (idx < len && is_valid_ohlc(open[idx], high[idx], low[idx], close[idx])) {
            idx += 1;
        }
        const int seg_end = idx;
        const int seg_len = seg_end - seg_start;

        if (seg_len >= length + 1) {
            compute_run_abs(
                open,
                high,
                low,
                close,
                seg_start,
                seg_len,
                length,
                extend,
                max_deque,
                min_deque,
                row_middle,
                row_upper,
                row_lower
            );
        }
    }
}
