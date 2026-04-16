#include <cmath>
#include <cstddef>

namespace {
constexpr double PI = 3.14159265358979323846264338327950288;

template <int N>
__device__ inline double ring_get(const double (&buf)[N], int center, int off) {
    int idx = center + N - (off % N);
    if (idx >= N) {
        idx -= N;
    }
    return buf[idx];
}

__device__ inline double nz(double value) {
    return isfinite(value) ? value : 0.0;
}

__device__ inline double median3(double a, double b, double c) {
    if (!(isfinite(a) && isfinite(b) && isfinite(c))) {
        return NAN;
    }
    return (a + b + c) - fmin(a, fmin(b, c)) - fmax(a, fmax(b, c));
}
}

extern "C" __global__ void ehlers_adaptive_cyber_cycle_batch_f64(
    const double* __restrict__ data,
    int len,
    const double* __restrict__ alphas,
    int rows,
    double* __restrict__ out_cycle,
    double* __restrict__ out_trigger
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const double alpha = alphas[row];
    double* row_cycle = out_cycle + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trigger = out_trigger + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_cycle[i] = NAN;
        row_trigger[i] = NAN;
    }

    if (!isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
        return;
    }

    const double one_minus_alpha = 1.0 - alpha;
    const double cycle_coef = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    const double cycle_prev1_coef = 2.0 * one_minus_alpha;
    const double cycle_prev2_coef = one_minus_alpha * one_minus_alpha;

    double src_ring[4];
    double smooth_ring[3];
    double cycle_ring[7];
    double dp_ring[5];
    double adaptive_hist[2];
    for (int i = 0; i < 4; ++i) {
        src_ring[i] = NAN;
    }
    for (int i = 0; i < 3; ++i) {
        smooth_ring[i] = NAN;
    }
    for (int i = 0; i < 7; ++i) {
        cycle_ring[i] = NAN;
    }
    for (int i = 0; i < 5; ++i) {
        dp_ring[i] = NAN;
    }
    adaptive_hist[0] = NAN;
    adaptive_hist[1] = NAN;

    int src_idx = 0;
    int smooth_idx = 0;
    int cycle_idx = 0;
    int dp_idx = 0;
    int valid_count = 0;
    double prev_ip = NAN;
    double prev_p = NAN;
    double prev_q1 = NAN;
    double prev_i1 = NAN;

    for (int i = 0; i < len; ++i) {
        const double source = data[i];
        if (!isfinite(source)) {
            continue;
        }

        const int bar = valid_count;
        src_ring[src_idx] = source;

        const double src0 = ring_get(src_ring, src_idx, 0);
        const double src1 = ring_get(src_ring, src_idx, 1);
        const double src2 = ring_get(src_ring, src_idx, 2);
        const double src3 = ring_get(src_ring, src_idx, 3);

        const double smooth =
            isfinite(src0) && isfinite(src1) && isfinite(src2) && isfinite(src3)
                ? (src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0
                : NAN;
        smooth_ring[smooth_idx] = smooth;

        const double smooth1 = ring_get(smooth_ring, smooth_idx, 1);
        const double smooth2 = ring_get(smooth_ring, smooth_idx, 2);
        const double cycle_prev1 = ring_get(cycle_ring, cycle_idx, 1);
        const double cycle_prev2 = ring_get(cycle_ring, cycle_idx, 2);

        const double cycle_main =
            isfinite(smooth) && isfinite(smooth1) && isfinite(smooth2) &&
                    isfinite(cycle_prev1) && isfinite(cycle_prev2)
                ? cycle_coef * (smooth - 2.0 * smooth1 + smooth2) +
                      cycle_prev1_coef * cycle_prev1 - cycle_prev2_coef * cycle_prev2
                : NAN;

        const double cycle_fallback =
            isfinite(src0) && isfinite(src1) && isfinite(src2)
                ? (src0 - 2.0 * src1 + src2) / 4.0
                : NAN;

        const double cycle = bar < 7 ? cycle_fallback : cycle_main;
        cycle_ring[cycle_idx] = cycle;

        const double q1 =
            isfinite(cycle)
                ? (0.0962 * cycle + 0.5769 * nz(ring_get(cycle_ring, cycle_idx, 2)) -
                      0.5769 * nz(ring_get(cycle_ring, cycle_idx, 4)) -
                      0.0962 * nz(ring_get(cycle_ring, cycle_idx, 6))) *
                      (0.5 + 0.08 * nz(prev_ip))
                : NAN;
        const double i1 = nz(ring_get(cycle_ring, cycle_idx, 3));

        const double dp_raw =
            isfinite(q1) && isfinite(prev_q1) && q1 != 0.0 && prev_q1 != 0.0
                ? (((i1 / q1) - (nz(prev_i1) / nz(prev_q1))) /
                   (1.0 + i1 * nz(prev_i1) / (q1 * nz(prev_q1))))
                : 0.0;
        double dp = dp_raw;
        if (dp < 0.1) {
            dp = 0.1;
        } else if (dp > 1.1) {
            dp = 1.1;
        }
        dp_ring[dp_idx] = dp;

        const double md_inner = median3(
            ring_get(dp_ring, dp_idx, 2),
            ring_get(dp_ring, dp_idx, 3),
            ring_get(dp_ring, dp_idx, 4)
        );
        const double md = median3(dp, ring_get(dp_ring, dp_idx, 1), md_inner);
        const double dc = md == 0.0 ? 15.0 : (2.0 * PI / md) + 0.5;
        const double ip = 0.33 * dc + 0.67 * nz(prev_ip);
        const double p = 0.15 * ip + 0.85 * nz(prev_p);
        const double a1 = 2.0 / (p + 1.0);

        const double adaptive_main =
            isfinite(smooth) && isfinite(smooth1) && isfinite(smooth2) &&
                    isfinite(adaptive_hist[0]) && isfinite(adaptive_hist[1]) && isfinite(a1)
                ? ((1.0 - 0.5 * a1) * (1.0 - 0.5 * a1)) * (smooth - 2.0 * smooth1 + smooth2) +
                      2.0 * (1.0 - a1) * adaptive_hist[0] -
                      (1.0 - a1) * (1.0 - a1) * adaptive_hist[1]
                : NAN;

        const double adaptive_cycle =
            bar < 7 ? cycle_fallback : (isfinite(adaptive_main) ? adaptive_main : cycle_fallback);
        const double trigger = adaptive_hist[0];

        row_cycle[i] = adaptive_cycle;
        row_trigger[i] = trigger;

        prev_q1 = q1;
        prev_i1 = i1;
        prev_ip = ip;
        prev_p = p;
        adaptive_hist[1] = adaptive_hist[0];
        adaptive_hist[0] = adaptive_cycle;

        valid_count += 1;
        src_idx = (src_idx + 1) % 4;
        smooth_idx = (smooth_idx + 1) % 3;
        cycle_idx = (cycle_idx + 1) % 7;
        dp_idx = (dp_idx + 1) % 5;
    }
}
