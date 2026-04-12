#include <cmath>
#include <cstddef>

static __device__ inline double ehlers_adaptive_cg_median3(double a, double b, double c) {
    return (a + b + c) - fmin(a, fmin(b, c)) - fmax(a, fmax(b, c));
}

extern "C" __global__ void ehlers_adaptive_cg_batch_f64(
    const double* __restrict__ data,
    int len,
    const double* __restrict__ alphas,
    int rows,
    double* __restrict__ out_cg,
    double* __restrict__ out_trigger
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double alpha = alphas[row];
    double* row_out_cg = out_cg + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger = out_trigger + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_cg[i] = NAN;
        row_out_trigger[i] = NAN;
    }

    if (!isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
        return;
    }

    int first_valid = -1;
    for (int i = 0; i < len; ++i) {
        if (!isnan(data[i])) {
            first_valid = i;
            break;
        }
    }
    if (first_valid < 0 || len - first_valid < 14) {
        return;
    }

    double smooth_hist[3] = {0.0, 0.0, 0.0};
    double cycle_hist[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double q1_hist[2] = {0.0, 0.0};
    double dp_hist[5] = {0.1, 0.1, 0.1, 0.1, 0.1};
    double ip_hist[2] = {0.0, 0.0};
    double p_hist[2] = {0.0, 0.0};

    double alpha_half = 1.0 - 0.5 * alpha;
    double alpha_half_sq = alpha_half * alpha_half;
    double one_minus_alpha = 1.0 - alpha;
    double one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    for (int i = 0; i < len; ++i) {
        double cg_cur = NAN;
        double smooth_cur = 0.0;
        double cycle_cur = 0.0;
        double q1_cur = 0.0;
        double dp_cur = 0.1;
        double ip_cur = 0.0;
        double p_cur = 0.0;

        if (i > first_valid) {
            row_out_trigger[i] = row_out_cg[i - 1];
        }

        if (i >= first_valid) {
            double x0 = data[i];
            if (!isnan(x0)) {
                double x1 = (i >= 1) ? data[i - 1] : x0;
                double x2 = (i >= 2) ? data[i - 2] : x1;
                double x3 = (i >= 3) ? data[i - 3] : x2;

                smooth_cur = (x0 + 2.0 * x1 + 2.0 * x2 + x3) / 6.0;

                if (i < first_valid + 7) {
                    cycle_cur = (x0 - 2.0 * x1 + x2) * 0.25;
                } else {
                    double smooth_prev1 = smooth_hist[(i - 1) % 3];
                    double smooth_prev2 = smooth_hist[(i - 2) % 3];
                    double cycle_prev1 = cycle_hist[(i - 1) % 7];
                    double cycle_prev2 = cycle_hist[(i - 2) % 7];
                    cycle_cur = alpha_half_sq * (smooth_cur - 2.0 * smooth_prev1 + smooth_prev2)
                        + 2.0 * one_minus_alpha * cycle_prev1
                        - one_minus_alpha_sq * cycle_prev2;
                }

                double ip_prev = (i >= 1) ? ip_hist[(i - 1) % 2] : 0.0;
                if (i >= first_valid + 6) {
                    double cycle_m2 = cycle_hist[(i - 2) % 7];
                    double cycle_m4 = cycle_hist[(i - 4) % 7];
                    double cycle_m6 = cycle_hist[(i - 6) % 7];
                    q1_cur = (0.0962 * cycle_cur + 0.5769 * cycle_m2 - 0.5769 * cycle_m4 -
                              0.0962 * cycle_m6) *
                        (0.5 + 0.08 * ip_prev);
                }

                if (i >= first_valid + 7) {
                    double i1 = cycle_hist[(i - 3) % 7];
                    double prev_i1 = cycle_hist[(i - 4) % 7];
                    double prev_q = q1_hist[(i - 1) % 2];
                    if (fabs(q1_cur) > 1e-12 && fabs(prev_q) > 1e-12) {
                        double raw = (i1 / q1_cur - prev_i1 / prev_q) /
                            (1.0 + i1 * prev_i1 / (q1_cur * prev_q));
                        if (raw < 0.1) {
                            raw = 0.1;
                        } else if (raw > 1.1) {
                            raw = 1.1;
                        }
                        dp_cur = raw;
                    }
                }

                double md = 0.1;
                if (i >= first_valid + 4) {
                    md = ehlers_adaptive_cg_median3(
                        dp_cur,
                        dp_hist[(i - 1) % 5],
                        ehlers_adaptive_cg_median3(
                            dp_hist[(i - 2) % 5],
                            dp_hist[(i - 3) % 5],
                            dp_hist[(i - 4) % 5]
                        )
                    );
                }

                double dc = (2.0 * 3.14159265358979323846) / md + 0.5;
                if (i == first_valid) {
                    ip_cur = dc;
                    p_cur = ip_cur;
                } else {
                    double prev_ip = ip_hist[(i - 1) % 2];
                    double prev_p = p_hist[(i - 1) % 2];
                    ip_cur = 0.33 * dc + 0.67 * prev_ip;
                    p_cur = 0.15 * ip_cur + 0.85 * prev_p;
                }

                int window = static_cast<int>(llround(p_cur * 0.5));
                if (window < 1) {
                    window = 1;
                } else if (window > 100) {
                    window = 100;
                }

                if (i + 1 >= first_valid + window && i + 1 >= window) {
                    double numerator = 0.0;
                    double denominator = 0.0;
                    bool has_nan = false;
                    for (int lag = 0; lag < window; ++lag) {
                        double value = data[i - lag];
                        if (isnan(value)) {
                            has_nan = true;
                            break;
                        }
                        numerator += (static_cast<double>(lag) + 1.0) * value;
                        denominator += value;
                    }
                    if (!has_nan) {
                        if (fabs(denominator) > 1e-12) {
                            cg_cur = -numerator / denominator +
                                (static_cast<double>(window) + 1.0) * 0.5;
                        } else {
                            cg_cur = 0.0;
                        }
                    }
                }
            }
        }

        smooth_hist[i % 3] = smooth_cur;
        cycle_hist[i % 7] = cycle_cur;
        q1_hist[i % 2] = q1_cur;
        dp_hist[i % 5] = dp_cur;
        ip_hist[i % 2] = ip_cur;
        p_hist[i % 2] = p_cur;
        row_out_cg[i] = cg_cur;
    }
}
