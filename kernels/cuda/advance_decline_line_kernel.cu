#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void advance_decline_line_batch_f64(
    const double* __restrict__ data,
    int len,
    double* __restrict__ out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || len <= 0) {
        return;
    }

    bool started = false;
    double sum = 0.0;

    for (int t = 0; t < len; ++t) {
        double value = data[t];
        if (!isfinite(value)) {
            out[t] = CUDART_NAN;
            started = false;
            sum = 0.0;
            continue;
        }

        if (!started) {
            started = true;
            sum = value;
        } else {
            sum += value;
        }

        out[t] = sum;
    }
}
