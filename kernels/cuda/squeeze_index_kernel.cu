#include <cmath>
#include <cstdint>

static __device__ inline double psi_from_corr(double sum_x, double sum_x2, double weighted, int length) {
    double n = static_cast<double>(length);
    double sum_y = n * (n - 1.0) * 0.5;
    double sum_y2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    double denom_x = n * sum_x2 - sum_x * sum_x;
    double denom_y = n * sum_y2 - sum_y * sum_y;
    double denom = denom_x * denom_y;
    if (denom <= 0.0 || !isfinite(denom)) {
        return NAN;
    }
    double corr = (n * weighted - sum_x * sum_y) / sqrt(denom);
    return -50.0 * corr + 50.0;
}

extern "C" __global__ void squeeze_index_batch_f64(
    const double* data,
    int len,
    const double* convs,
    const int* lengths,
    int rows,
    int max_length,
    double* ring_vals_buf,
    int* ring_valid_buf,
    double* out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    double conv = convs[row];
    int length = lengths[row];
    if (!(isfinite(conv)) || conv <= 1.0 || length <= 0 || length > max_length) {
        return;
    }

    const double nan = NAN;
    double* ring_vals = ring_vals_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* ring_valid = ring_valid_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    double max_state = 0.0;
    double min_state = 0.0;
    int head = 0;
    int filled = 0;
    int valid_count = 0;
    double sum_x = 0.0;
    double sum_x2 = 0.0;
    double weighted = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            max_state = 0.0;
            min_state = 0.0;
            head = 0;
            filled = 0;
            valid_count = 0;
            sum_x = 0.0;
            sum_x2 = 0.0;
            weighted = 0.0;
            row_out[i] = nan;
            continue;
        }

        double max_next = fmax(value, max_state - (max_state - value) / conv);
        double min_next = fmin(value, min_state + (value - min_state) / conv);
        max_state = max_next;
        min_state = min_next;

        double spread = max_next - min_next;
        bool is_valid = spread > 0.0;
        double push_value = is_valid ? log(spread) : 0.0;

        if (filled < length) {
            int pos = filled;
            ring_vals[pos] = push_value;
            ring_valid[pos] = is_valid ? 1 : 0;
            sum_x += push_value;
            sum_x2 += push_value * push_value;
            weighted += static_cast<double>(pos) * push_value;
            if (is_valid) {
                valid_count += 1;
            }
            filled += 1;
            if (filled < length) {
                row_out[i] = nan;
                continue;
            }
        } else {
            double old_value = ring_vals[head];
            int old_valid = ring_valid[head];
            double old_sum = sum_x;

            weighted = weighted - old_sum + old_value + static_cast<double>(length - 1) * push_value;
            sum_x = old_sum - old_value + push_value;
            sum_x2 = sum_x2 - old_value * old_value + push_value * push_value;
            valid_count = valid_count + (is_valid ? 1 : 0) - old_valid;

            ring_vals[head] = push_value;
            ring_valid[head] = is_valid ? 1 : 0;
            head += 1;
            if (head == length) {
                head = 0;
            }
        }

        row_out[i] = valid_count == length ? psi_from_corr(sum_x, sum_x2, weighted, length) : nan;
    }
}
