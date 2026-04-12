#include <cmath>
#include <cstdint>

static __device__ inline double abto_median3(double x, double y, double z) {
    double min_xy = x < y ? x : y;
    double min_v = min_xy < z ? min_xy : z;
    double max_xy = x > y ? x : y;
    double max_v = max_xy > z ? max_xy : z;
    return (x + y + z) - min_v - max_v;
}

extern "C" __global__ void adaptive_bandpass_trigger_oscillator_batch_f64(
    const double* data,
    int len,
    const double* deltas,
    const double* alphas,
    int rows,
    double* out_in_phase,
    double* out_lead
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    double delta = deltas[row];
    double alpha = alphas[row];
    if (!(isfinite(delta) && isfinite(alpha)) || delta <= 0.0 || delta >= 1.0 || alpha <= 0.0
        || alpha >= 1.0) {
        return;
    }

    const double pi = 3.14159265358979323846;
    const double float_tol = 1e-12;
    const double nan = NAN;
    const int in_phase_warmup = 11;
    const int lead_warmup = 12;

    double price[4] = {0.0, 0.0, 0.0, 0.0};
    double smooth_hist[2] = {0.0, 0.0};
    double c_hist[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dp_hist[4] = {0.0, 0.0, 0.0, 0.0};
    double q1_prev = 0.0;
    double i1_prev = 0.0;
    double ip_prev = 0.0;
    double p_prev = 0.0;
    double bp_prev1 = 0.0;
    double bp_prev2 = 0.0;
    int valid_count = 0;

    double* row_in_phase = out_in_phase + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lead = out_lead + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            for (int j = 0; j < 4; ++j) {
                price[j] = 0.0;
            }
            smooth_hist[0] = 0.0;
            smooth_hist[1] = 0.0;
            for (int j = 0; j < 6; ++j) {
                c_hist[j] = 0.0;
            }
            for (int j = 0; j < 4; ++j) {
                dp_hist[j] = 0.0;
            }
            q1_prev = 0.0;
            i1_prev = 0.0;
            ip_prev = 0.0;
            p_prev = 0.0;
            bp_prev1 = 0.0;
            bp_prev2 = 0.0;
            valid_count = 0;
            row_in_phase[i] = nan;
            row_lead[i] = nan;
            continue;
        }

        price[3] = price[2];
        price[2] = price[1];
        price[1] = price[0];
        price[0] = value;

        int index = valid_count;
        valid_count += 1;

        double smooth =
            index >= 3 ? (price[0] + 2.0 * price[1] + 2.0 * price[2] + price[3]) / 6.0 : 0.0;

        double c = 0.0;
        if (index < 2) {
            c = 0.0;
        } else if (index < 7) {
            c = (price[0] - 2.0 * price[1] + price[2]) * 0.25;
        } else {
            double smooth_gain = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
            c = smooth_gain * (smooth - 2.0 * smooth_hist[0] + smooth_hist[1])
                + 2.0 * (1.0 - alpha) * c_hist[0]
                - (1.0 - alpha) * (1.0 - alpha) * c_hist[1];
        }

        double q1 =
            index >= 6
                ? (0.0962 * c + 0.5769 * c_hist[1] - 0.5769 * c_hist[3] - 0.0962 * c_hist[5])
                      * (0.5 + 0.08 * ip_prev)
                : 0.0;
        double i1 = index >= 3 ? c_hist[2] : 0.0;

        double dp_raw = 0.0;
        if (fabs(q1) > float_tol && fabs(q1_prev) > float_tol) {
            double denominator = 1.0 + (i1 * i1_prev) / (q1 * q1_prev);
            if (fabs(denominator) > float_tol) {
                dp_raw = ((i1 / q1) - (i1_prev / q1_prev)) / denominator;
            }
        }
        double dp = fmin(fmax(dp_raw, 0.1), 1.1);

        double md = 0.0;
        if (index >= 10) {
            md = abto_median3(dp, dp_hist[0], abto_median3(dp_hist[1], dp_hist[2], dp_hist[3]));
        }
        double dc = fabs(md) <= float_tol ? 15.0 : (2.0 * pi) / md + 0.5;
        double ip = 0.33 * dc + 0.67 * ip_prev;
        double p = 0.15 * ip + 0.85 * p_prev;

        double in_phase = nan;
        double lead = nan;
        if (index >= in_phase_warmup) {
            double length = fmax(p, 6.0);
            double beta = cos(2.0 * pi / length);
            double cos_angle = cos(4.0 * pi * delta / length);
            double denom = fabs(cos_angle) < float_tol
                               ? (cos_angle < 0.0 ? -float_tol : float_tol)
                               : cos_angle;
            double gamma = 1.0 / denom;
            double root = gamma * gamma - 1.0;
            if (root < 0.0) {
                root = 0.0;
            }
            double alpha_bp = gamma - sqrt(root);

            in_phase = 0.5 * (1.0 - alpha_bp) * (price[0] - price[2])
                + beta * (1.0 + alpha_bp) * bp_prev1 - alpha_bp * bp_prev2;
            if (index >= lead_warmup) {
                double quadrature = (in_phase - bp_prev1) * length / (2.0 * pi);
                lead = 0.5 * in_phase + 0.866 * quadrature;
            }
        }

        smooth_hist[1] = smooth_hist[0];
        smooth_hist[0] = smooth;

        c_hist[5] = c_hist[4];
        c_hist[4] = c_hist[3];
        c_hist[3] = c_hist[2];
        c_hist[2] = c_hist[1];
        c_hist[1] = c_hist[0];
        c_hist[0] = c;

        dp_hist[3] = dp_hist[2];
        dp_hist[2] = dp_hist[1];
        dp_hist[1] = dp_hist[0];
        dp_hist[0] = dp;

        q1_prev = q1;
        i1_prev = i1;
        ip_prev = ip;
        p_prev = p;

        if (isfinite(in_phase)) {
            bp_prev2 = bp_prev1;
            bp_prev1 = in_phase;
            row_in_phase[i] = in_phase;
            row_lead[i] = lead;
        } else {
            row_in_phase[i] = nan;
            row_lead[i] = nan;
        }
    }
}
