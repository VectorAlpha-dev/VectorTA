#include <cmath>
#include <cstdint>

static __device__ inline double esam_nz(double value) {
    return isfinite(value) ? value : 0.0;
}

static __device__ inline double esam_median3(double a, double b, double c) {
    if (!(isfinite(a) && isfinite(b) && isfinite(c))) {
        return NAN;
    }
    double min_ab = a < b ? a : b;
    double min_v = min_ab < c ? min_ab : c;
    double max_ab = a > b ? a : b;
    double max_v = max_ab > c ? max_ab : c;
    return (a + b + c) - min_v - max_v;
}

static __device__ inline double esam_ring_get(const double* buf, int center, int off, int size) {
    int idx = center + size - (off % size);
    if (idx >= size) {
        idx -= size;
    }
    return buf[idx];
}

extern "C" __global__ void ehlers_smoothed_adaptive_momentum_batch_f64(
    const double* data,
    int len,
    const double* alphas,
    const double* cutoffs,
    int rows,
    double* out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    double alpha = alphas[row];
    double cutoff = cutoffs[row];
    if (!(isfinite(alpha) && isfinite(cutoff)) || alpha < 0.0 || cutoff <= 0.0) {
        return;
    }

    const int src_ring_size = 76;
    const int smooth_ring_size = 3;
    const int c_ring_size = 7;
    const int dp_ring_size = 5;
    const double nan = NAN;
    const double pi = 3.14159265358979323846;

    double coef_c = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    double one_minus_alpha = 1.0 - alpha;
    double a1 = exp(-pi / cutoff);
    double b1 = 2.0 * a1 * cos(1.738 * pi / cutoff);
    double c1 = a1 * a1;
    double coef2 = b1 + c1;
    double coef3 = -(c1 + b1 * c1);
    double coef4 = c1 * c1;
    double coef1 = 1.0 - coef2 - coef3 - coef4;

    double src_ring[src_ring_size];
    double smooth_ring[smooth_ring_size];
    double c_ring[c_ring_size];
    double dp_ring[dp_ring_size];
    double f3_hist[3];

    for (int i = 0; i < src_ring_size; ++i) {
        src_ring[i] = nan;
    }
    for (int i = 0; i < smooth_ring_size; ++i) {
        smooth_ring[i] = nan;
    }
    for (int i = 0; i < c_ring_size; ++i) {
        c_ring[i] = nan;
    }
    for (int i = 0; i < dp_ring_size; ++i) {
        dp_ring[i] = nan;
    }
    for (int i = 0; i < 3; ++i) {
        f3_hist[i] = nan;
    }

    int src_idx = 0;
    int smooth_idx = 0;
    int c_idx = 0;
    int dp_idx = 0;
    int valid_count = 0;

    double prev_ip = nan;
    double prev_p = nan;
    double prev_q1 = nan;
    double prev_i1 = nan;

    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        double source = data[i];
        if (!isfinite(source)) {
            row_out[i] = nan;
            continue;
        }

        src_ring[src_idx] = source;
        double src0 = esam_ring_get(src_ring, src_idx, 0, src_ring_size);
        double src1 = esam_ring_get(src_ring, src_idx, 1, src_ring_size);
        double src2 = esam_ring_get(src_ring, src_idx, 2, src_ring_size);
        double src3 = esam_ring_get(src_ring, src_idx, 3, src_ring_size);

        double smooth =
            (isfinite(src0) && isfinite(src1) && isfinite(src2) && isfinite(src3))
                ? (src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0
                : nan;
        smooth_ring[smooth_idx] = smooth;

        double smooth1 = esam_nz(esam_ring_get(smooth_ring, smooth_idx, 1, smooth_ring_size));
        double smooth2 = esam_nz(esam_ring_get(smooth_ring, smooth_idx, 2, smooth_ring_size));
        double c_prev1 = esam_nz(esam_ring_get(c_ring, c_idx, 1, c_ring_size));
        double c_prev2 = esam_nz(esam_ring_get(c_ring, c_idx, 2, c_ring_size));
        double c_main = isfinite(smooth)
                            ? coef_c * (smooth - 2.0 * smooth1 + smooth2)
                                  + 2.0 * one_minus_alpha * c_prev1
                                  - one_minus_alpha * one_minus_alpha * c_prev2
                            : nan;
        double c_fallback =
            (isfinite(src0) && isfinite(src1) && isfinite(src2)) ? (src0 - 2.0 * src1 + src2) / 4.0
                                                                  : nan;
        double c = isfinite(c_main) ? c_main : c_fallback;
        c_ring[c_idx] = c;

        double q1 = nan;
        if (isfinite(c)) {
            double factor = 0.5 + 0.08 * esam_nz(prev_ip);
            q1 = (0.0962 * c + 0.5769 * esam_nz(esam_ring_get(c_ring, c_idx, 2, c_ring_size))
                  - 0.5769 * esam_nz(esam_ring_get(c_ring, c_idx, 4, c_ring_size))
                  - 0.0962 * esam_nz(esam_ring_get(c_ring, c_idx, 6, c_ring_size)))
                 * factor;
        }
        double i1 = esam_nz(esam_ring_get(c_ring, c_idx, 3, c_ring_size));

        double dp_raw = 0.0;
        if (isfinite(q1) && isfinite(prev_q1) && q1 != 0.0 && prev_q1 != 0.0) {
            double prev_i1_nz = esam_nz(prev_i1);
            double prev_q1_nz = esam_nz(prev_q1);
            double numer = (i1 / q1) - (prev_i1_nz / prev_q1_nz);
            double denom = 1.0 + i1 * prev_i1_nz / (q1 * prev_q1_nz);
            dp_raw = numer / denom;
        }
        double dp = dp_raw < 0.1 ? 0.1 : (dp_raw > 1.1 ? 1.1 : dp_raw);
        dp_ring[dp_idx] = dp;

        double md_inner = esam_median3(
            esam_ring_get(dp_ring, dp_idx, 2, dp_ring_size),
            esam_ring_get(dp_ring, dp_idx, 3, dp_ring_size),
            esam_ring_get(dp_ring, dp_idx, 4, dp_ring_size));
        double md = esam_median3(dp, esam_ring_get(dp_ring, dp_idx, 1, dp_ring_size), md_inner);
        double dc = md == 0.0 ? 15.0 : ((2.0 * pi / md) + 0.5);
        double ip = 0.33 * dc + 0.67 * esam_nz(prev_ip);
        double p = 0.15 * ip + 0.85 * esam_nz(prev_p);

        double pr = isfinite(p) ? round(fabs(p - 1.0)) : nan;
        double v1 = 0.0;
        if (isfinite(pr)) {
            int lookback = static_cast<int>(pr);
            if (lookback >= 1 && lookback <= 75) {
                double past = esam_ring_get(src_ring, src_idx, lookback, src_ring_size);
                v1 = isfinite(past) ? (source - past) : nan;
            } else {
                v1 = 0.0;
            }
        } else {
            v1 = 0.0;
        }

        double raw_f3 =
            isfinite(v1)
                ? coef1 * v1 + coef2 * esam_nz(f3_hist[0]) + coef3 * esam_nz(f3_hist[1])
                      + coef4 * esam_nz(f3_hist[2])
                : nan;
        double f3 = isfinite(raw_f3) ? raw_f3 : v1;

        prev_q1 = q1;
        prev_i1 = i1;
        prev_ip = ip;
        prev_p = p;
        f3_hist[2] = f3_hist[1];
        f3_hist[1] = f3_hist[0];
        f3_hist[0] = f3;

        valid_count += 1;
        src_idx = (src_idx + 1) % src_ring_size;
        smooth_idx = (smooth_idx + 1) % smooth_ring_size;
        c_idx = (c_idx + 1) % c_ring_size;
        dp_idx = (dp_idx + 1) % dp_ring_size;

        row_out[i] = valid_count <= 75 ? nan : f3;
    }
}
