#include <cmath>
#include <cstdint>

static __device__ inline double ring_get(const double* buf, int center, int off, int size) {
    int idx = center + size - (off % size);
    if (idx >= size) {
        idx -= size;
    }
    return buf[idx];
}

extern "C" __global__ void l2_ehlers_signal_to_noise_batch_f64(
    const double* source,
    const double* high,
    const double* low,
    int len,
    const int* smooth_periods,
    int rows,
    double* out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int smooth_period = smooth_periods[row];
    if (smooth_period <= 0) {
        return;
    }

    const double nan = NAN;
    const double ln10 = 2.3025850929940459;
    const double two_pi = 6.2831853071795865;
    const double period_mult = 0.075 * static_cast<double>(smooth_period) + 0.54;

    double source_ring[4] = {0.0, 0.0, 0.0, 0.0};
    double smooth_ring[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double detrender_ring[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double q1_ring[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double i1_ring[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int source_idx = 0;
    int smooth_idx = 0;
    int detrender_idx = 0;
    int q1_idx = 0;
    int i1_idx = 0;
    int valid_count = 0;

    double range_1 = 0.0;
    double i2 = 0.0;
    double q2 = 0.0;
    double re = 0.0;
    double im = 0.0;
    double period = 0.0;
    double snr = 0.0;

    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        double src = source[i];
        double hi = high[i];
        double lo = low[i];
        if (!(isfinite(src) && isfinite(hi) && isfinite(lo))) {
            row_out[i] = nan;
            continue;
        }

        range_1 = 0.1 * (hi - lo) + 0.9 * range_1;
        source_ring[source_idx] = src;

        double smooth = 0.0;
        double detrender = 0.0;
        double i1 = 0.0;
        double q1 = 0.0;

        if (valid_count > 5) {
            double x0 = ring_get(source_ring, source_idx, 0, 4);
            double x1 = ring_get(source_ring, source_idx, 1, 4);
            double x2 = ring_get(source_ring, source_idx, 2, 4);
            double x3 = ring_get(source_ring, source_idx, 3, 4);
            smooth = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) / 10.0;

            smooth_ring[smooth_idx] = smooth;
            double s0 = ring_get(smooth_ring, smooth_idx, 0, 7);
            double s2 = ring_get(smooth_ring, smooth_idx, 2, 7);
            double s4 = ring_get(smooth_ring, smooth_idx, 4, 7);
            double s6 = ring_get(smooth_ring, smooth_idx, 6, 7);
            detrender = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) * period_mult;

            detrender_ring[detrender_idx] = detrender;
            i1 = ring_get(detrender_ring, detrender_idx, 3, 7);
            i1_ring[i1_idx] = i1;

            double d0 = ring_get(detrender_ring, detrender_idx, 0, 7);
            double d2 = ring_get(detrender_ring, detrender_idx, 2, 7);
            double d4 = ring_get(detrender_ring, detrender_idx, 4, 7);
            double d6 = ring_get(detrender_ring, detrender_idx, 6, 7);
            q1 = (0.0962 * d0 + 0.5769 * d2 - 0.5769 * d4 - 0.0962 * d6) * period_mult;
            q1_ring[q1_idx] = q1;

            double i0 = ring_get(i1_ring, i1_idx, 0, 7);
            double i2_hist = ring_get(i1_ring, i1_idx, 2, 7);
            double i4 = ring_get(i1_ring, i1_idx, 4, 7);
            double i6 = ring_get(i1_ring, i1_idx, 6, 7);
            double ji = (0.0962 * i0 + 0.5769 * i2_hist - 0.5769 * i4 - 0.0962 * i6) * period_mult;

            double q0 = ring_get(q1_ring, q1_idx, 0, 7);
            double q2_hist = ring_get(q1_ring, q1_idx, 2, 7);
            double q4 = ring_get(q1_ring, q1_idx, 4, 7);
            double q6 = ring_get(q1_ring, q1_idx, 6, 7);
            double jq = (0.0962 * q0 + 0.5769 * q2_hist - 0.5769 * q4 - 0.0962 * q6) * period_mult;

            double prev_i2 = i2;
            double prev_q2 = q2;
            double prev_re = re;
            double prev_im = im;
            double prev_period = period;
            double prev_snr = snr;

            i2 = 0.2 * (i1 - jq) + 0.8 * prev_i2;
            q2 = 0.2 * (q1 + ji) + 0.8 * prev_q2;

            double re_raw = i2 * prev_i2 + q2 * prev_q2;
            double im_raw = i2 * prev_q2 - q2 * prev_i2;
            re = 0.2 * re_raw + 0.8 * prev_re;
            im = 0.2 * im_raw + 0.8 * prev_im;

            double next_period = prev_period;
            if (re != 0.0 && im != 0.0) {
                double angle = atan2(im, re);
                if (angle != 0.0) {
                    next_period = two_pi / fabs(angle);
                }
            }
            if (prev_period != 0.0) {
                double upper = 1.5 * prev_period;
                double lower = 0.67 * prev_period;
                if (next_period > upper) {
                    next_period = upper;
                }
                if (next_period < lower) {
                    next_period = lower;
                }
            }
            if (next_period < 6.0) {
                next_period = 6.0;
            }
            if (next_period > 50.0) {
                next_period = 50.0;
            }
            period = 0.2 * next_period + 0.8 * prev_period;

            double power = i1 * i1 + q1 * q1;
            double noise = range_1 * range_1;
            if (power > 0.0 && noise > 0.0) {
                double snr_raw = 10.0 * log(power / noise) / ln10 + 6.0;
                snr = 0.25 * snr_raw + 0.75 * prev_snr;
            } else {
                snr = prev_snr;
            }
        } else {
            smooth_ring[smooth_idx] = smooth;
            detrender_ring[detrender_idx] = detrender;
            i1_ring[i1_idx] = i1;
            q1_ring[q1_idx] = q1;
        }

        valid_count += 1;
        source_idx = (source_idx + 1) % 4;
        smooth_idx = (smooth_idx + 1) % 7;
        detrender_idx = (detrender_idx + 1) % 7;
        i1_idx = (i1_idx + 1) % 7;
        q1_idx = (q1_idx + 1) % 7;

        row_out[i] = valid_count <= 6 ? nan : snr;
    }
}
