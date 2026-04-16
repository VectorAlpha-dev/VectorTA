#include <cmath>
#include <cstddef>

namespace {
constexpr double PI = 3.14;

__device__ inline double nz(double value) {
    return isfinite(value) ? value : 0.0;
}

struct MesaLineState {
    double* ring;
    int length;
    int head;
    int count;
    double prev_1;
    double prev_2;

    __device__ void init(double* storage, int line_length) {
        ring = storage;
        length = line_length;
        head = 0;
        count = 0;
        prev_1 = NAN;
        prev_2 = NAN;
    }

    __device__ double update(double filt, double c1, double c2, double c3) {
        const double filt_nz = nz(filt);
        if (count < length) {
            ring[count] = filt_nz;
            count += 1;
        } else {
            ring[head] = filt_nz;
            head += 1;
            if (head == length) {
                head = 0;
            }
        }

        double out = NAN;
        if (isfinite(filt)) {
            double highest = filt;
            double lowest = filt;
            for (int i = 0; i < count; ++i) {
                const double value = ring[i];
                if (value > highest) {
                    highest = value;
                }
                if (value < lowest) {
                    lowest = value;
                }
            }
            if (count < length) {
                if (0.0 > highest) {
                    highest = 0.0;
                }
                if (0.0 < lowest) {
                    lowest = 0.0;
                }
            }
            const double denom = highest - lowest;
            if (denom != 0.0 && isfinite(denom)) {
                const double stoc = (filt - lowest) / denom;
                if (isfinite(stoc)) {
                    out = fma(c1, stoc, fma(c2, nz(prev_1), c3 * nz(prev_2)));
                }
            }
        }

        prev_2 = prev_1;
        prev_1 = out;
        return out;
    }
};

struct RollingSmaState {
    double* ring;
    int length;
    int head;
    int count;
    int finite_count;
    double finite_sum;

    __device__ void init(double* storage, int window_length) {
        ring = storage;
        length = window_length;
        head = 0;
        count = 0;
        finite_count = 0;
        finite_sum = 0.0;
    }

    __device__ double update(double value) {
        if (count < length) {
            ring[count] = value;
            if (isfinite(value)) {
                finite_sum += value;
                finite_count += 1;
            }
            count += 1;
        } else {
            const double old = ring[head];
            if (isfinite(old)) {
                finite_sum -= old;
                finite_count -= 1;
            }
            ring[head] = value;
            if (isfinite(value)) {
                finite_sum += value;
                finite_count += 1;
            }
            head += 1;
            if (head == length) {
                head = 0;
            }
        }

        if (count == length && finite_count == length) {
            return finite_sum / static_cast<double>(length);
        }
        return NAN;
    }
};
}

extern "C" __global__ void mesa_stochastic_multi_length_batch_f64(
    const double* __restrict__ source,
    int len,
    const int* __restrict__ length_1s,
    const int* __restrict__ length_2s,
    const int* __restrict__ length_3s,
    const int* __restrict__ length_4s,
    const int* __restrict__ trigger_lengths,
    int rows,
    int max_length,
    int max_trigger_length,
    double* __restrict__ mesa_ring_buf,
    double* __restrict__ trigger_ring_buf,
    double* __restrict__ out_mesa_1,
    double* __restrict__ out_mesa_2,
    double* __restrict__ out_mesa_3,
    double* __restrict__ out_mesa_4,
    double* __restrict__ out_trigger_1,
    double* __restrict__ out_trigger_2,
    double* __restrict__ out_trigger_3,
    double* __restrict__ out_trigger_4
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length_1 = length_1s[row];
    const int length_2 = length_2s[row];
    const int length_3 = length_3s[row];
    const int length_4 = length_4s[row];
    const int trigger_length = trigger_lengths[row];

    double* row_out_mesa_1 = out_mesa_1 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_mesa_2 = out_mesa_2 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_mesa_3 = out_mesa_3 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_mesa_4 = out_mesa_4 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger_1 =
        out_trigger_1 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger_2 =
        out_trigger_2 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger_3 =
        out_trigger_3 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger_4 =
        out_trigger_4 + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_mesa_1[i] = NAN;
        row_out_mesa_2[i] = NAN;
        row_out_mesa_3[i] = NAN;
        row_out_mesa_4[i] = NAN;
        row_out_trigger_1[i] = NAN;
        row_out_trigger_2[i] = NAN;
        row_out_trigger_3[i] = NAN;
        row_out_trigger_4[i] = NAN;
    }

    if (length_1 <= 0 || length_2 <= 0 || length_3 <= 0 || length_4 <= 0 ||
        trigger_length <= 0 || length_1 > max_length || length_2 > max_length ||
        length_3 > max_length || length_4 > max_length || trigger_length > max_trigger_length) {
        return;
    }

    double* mesa_base =
        mesa_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length) * 4u;
    double* trigger_base =
        trigger_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_trigger_length) * 4u;

    MesaLineState mesa_1_state;
    MesaLineState mesa_2_state;
    MesaLineState mesa_3_state;
    MesaLineState mesa_4_state;
    mesa_1_state.init(mesa_base, length_1);
    mesa_2_state.init(mesa_base + max_length, length_2);
    mesa_3_state.init(mesa_base + max_length * 2, length_3);
    mesa_4_state.init(mesa_base + max_length * 3, length_4);

    RollingSmaState trigger_1_state;
    RollingSmaState trigger_2_state;
    RollingSmaState trigger_3_state;
    RollingSmaState trigger_4_state;
    trigger_1_state.init(trigger_base, trigger_length);
    trigger_2_state.init(trigger_base + max_trigger_length, trigger_length);
    trigger_3_state.init(trigger_base + max_trigger_length * 2, trigger_length);
    trigger_4_state.init(trigger_base + max_trigger_length * 3, trigger_length);

    const double alpha1 =
        ((cos(0.707 * 2.0 * PI / 48.0) + sin(0.707 * 2.0 * PI / 48.0) - 1.0) /
         cos(0.707 * 2.0 * PI / 48.0));
    const double one_minus_alpha = 1.0 - alpha1;
    const double hp_coef = (1.0 - alpha1 * 0.5) * (1.0 - alpha1 * 0.5);
    const double a1 = exp(-1.414 * PI / 10.0);
    const double b1 = 2.0 * a1 * cos(1.414 * PI / 10.0);
    const double c2 = b1;
    const double c3 = -(a1 * a1);
    const double c1 = 1.0 - c2 - c3;
    const double hp_feedback_1 = 2.0 * one_minus_alpha;
    const double hp_feedback_2 = -(one_minus_alpha * one_minus_alpha);

    double prev_src_1 = NAN;
    double prev_src_2 = NAN;
    double prev_hp_1 = NAN;
    double prev_hp_2 = NAN;
    double prev_filt_1 = NAN;
    double prev_filt_2 = NAN;

    for (int i = 0; i < len; ++i) {
        const double value = source[i];
        const double hp = isfinite(value)
            ? fma(
                  hp_coef,
                  value - 2.0 * nz(prev_src_1) + nz(prev_src_2),
                  fma(hp_feedback_1, nz(prev_hp_1), hp_feedback_2 * nz(prev_hp_2)))
            : NAN;
        const double filt = isfinite(hp)
            ? fma(c1, hp, fma(c2, nz(prev_filt_1), c3 * nz(prev_filt_2)))
            : NAN;

        prev_src_2 = prev_src_1;
        prev_src_1 = value;
        prev_hp_2 = prev_hp_1;
        prev_hp_1 = hp;
        prev_filt_2 = prev_filt_1;
        prev_filt_1 = filt;

        const double mesa_1 = mesa_1_state.update(filt, c1, c2, c3);
        const double mesa_2 = mesa_2_state.update(filt, c1, c2, c3);
        const double mesa_3 = mesa_3_state.update(filt, c1, c2, c3);
        const double mesa_4 = mesa_4_state.update(filt, c1, c2, c3);
        const double trigger_1 = trigger_1_state.update(mesa_1);
        const double trigger_2 = trigger_2_state.update(mesa_2);
        const double trigger_3 = trigger_3_state.update(mesa_3);
        const double trigger_4 = trigger_4_state.update(mesa_4);

        row_out_mesa_1[i] = mesa_1;
        row_out_mesa_2[i] = mesa_2;
        row_out_mesa_3[i] = mesa_3;
        row_out_mesa_4[i] = mesa_4;
        row_out_trigger_1[i] = trigger_1;
        row_out_trigger_2[i] = trigger_2;
        row_out_trigger_3[i] = trigger_3;
        row_out_trigger_4[i] = trigger_4;
    }
}
