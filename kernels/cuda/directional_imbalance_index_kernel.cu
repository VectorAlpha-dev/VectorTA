#include <cmath>
#include <cstdint>

extern "C" __global__ void directional_imbalance_index_batch_f64(
    const double* high,
    const double* low,
    int len,
    const int* lengths,
    const int* periods,
    int rows,
    int max_window,
    int max_period,
    double* high_ring,
    double* low_ring,
    double* up_hits_ring,
    double* down_hits_ring,
    double* out_up,
    double* out_down,
    double* out_bulls,
    double* out_bears,
    double* out_upper,
    double* out_lower
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    int period = periods[row];
    if (length <= 0 || period <= 0) {
        return;
    }

    const double nan = NAN;
    const double inf = 1.7976931348623157e308;
    int window_cap = length + 1;
    double* row_high_ring = high_ring + static_cast<size_t>(row) * static_cast<size_t>(max_window);
    double* row_low_ring = low_ring + static_cast<size_t>(row) * static_cast<size_t>(max_window);
    double* row_up_hits = up_hits_ring + static_cast<size_t>(row) * static_cast<size_t>(max_period);
    double* row_down_hits =
        down_hits_ring + static_cast<size_t>(row) * static_cast<size_t>(max_period);
    double* row_out_up = out_up + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_down = out_down + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_bulls = out_bulls + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_bears = out_bears + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);

    int price_head = 0;
    int price_count = 0;
    int hit_head = 0;
    int hit_count = 0;
    double up_sum = 0.0;
    double down_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        if (!isfinite(h) || !isfinite(l)) {
            price_head = 0;
            price_count = 0;
            hit_head = 0;
            hit_count = 0;
            up_sum = 0.0;
            down_sum = 0.0;
            row_out_up[i] = nan;
            row_out_down[i] = nan;
            row_out_bulls[i] = nan;
            row_out_bears[i] = nan;
            row_out_upper[i] = nan;
            row_out_lower[i] = nan;
            continue;
        }

        row_high_ring[price_head] = h;
        row_low_ring[price_head] = l;
        price_head += 1;
        if (price_head == window_cap) {
            price_head = 0;
        }
        if (price_count < window_cap) {
            price_count += 1;
        }

        double upper = -inf;
        double lower = inf;
        for (int j = 0; j < price_count; ++j) {
            double window_high = row_high_ring[j];
            double window_low = row_low_ring[j];
            if (window_high > upper) {
                upper = window_high;
            }
            if (window_low < lower) {
                lower = window_low;
            }
        }

        double up_hit = (h == upper) ? 1.0 : 0.0;
        double down_hit = (l == lower) ? 1.0 : 0.0;

        if (hit_count == period) {
            up_sum -= row_up_hits[hit_head];
            down_sum -= row_down_hits[hit_head];
        } else {
            hit_count += 1;
        }
        row_up_hits[hit_head] = up_hit;
        row_down_hits[hit_head] = down_hit;
        up_sum += up_hit;
        down_sum += down_hit;
        hit_head += 1;
        if (hit_head == period) {
            hit_head = 0;
        }

        double total = up_sum + down_sum;
        row_out_up[i] = up_sum;
        row_out_down[i] = down_sum;
        row_out_bulls[i] = total > 0.0 ? (up_sum / total) * 100.0 : nan;
        row_out_bears[i] = total > 0.0 ? (down_sum / total) * 100.0 : nan;
        row_out_upper[i] = upper;
        row_out_lower[i] = lower;
    }
}
