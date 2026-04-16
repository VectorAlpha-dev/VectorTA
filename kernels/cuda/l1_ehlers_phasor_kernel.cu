#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline double compute_phase_angle(double real, double imaginary) {
    double phase_angle = 0.0;
    if (fabs(real) > 0.001) {
        phase_angle = atan(imaginary / real) * 180.0 / CUDART_PI;
    } else if (imaginary > 0.0) {
        phase_angle = 90.0;
    } else if (imaginary < 0.0) {
        phase_angle = -90.0;
    }
    if (real < 0.0) {
        phase_angle += 180.0;
    }
    phase_angle += 90.0;
    if (phase_angle < 0.0) {
        phase_angle += 360.0;
    }
    if (phase_angle > 360.0) {
        phase_angle -= 360.0;
    }
    return phase_angle;
}
}

extern "C" __global__ void l1_ehlers_phasor_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length <= 0 || length > len) {
        return;
    }

    double angle = 2.0 * CUDART_PI / static_cast<double>(length);
    for (int end = length - 1; end < len; ++end) {
        bool valid = true;
        double real = 0.0;
        double imaginary = 0.0;
        for (int j = 0; j < length; ++j) {
            double value = data[end - j];
            if (!isfinite(value)) {
                valid = false;
                break;
            }
            double theta = angle * static_cast<double>(j);
            real += cos(theta) * value;
            imaginary += sin(theta) * value;
        }
        if (!valid) {
            continue;
        }
        row[end] = compute_phase_angle(real, imaginary);
    }
}
