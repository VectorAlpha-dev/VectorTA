#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void ewma_volatility_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    const double* __restrict__ alphas,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double alpha = alphas[combo_idx];
    double beta = 1.0 - alpha;
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (period <= 0 || !isfinite(alpha) || alpha <= 0.0 || alpha > 1.0) {
        return;
    }

    int valid_count = 0;
    int seed_idx = -1;
    double seed_sum = 0.0;

    for (int i = 1; i < len; ++i) {
        double prev = data[i - 1];
        double curr = data[i];
        if (!isfinite(prev) || !isfinite(curr) || prev <= 0.0 || curr <= 0.0) {
            continue;
        }

        double ret = log(curr / prev);
        double sq = ret * ret;
        if (valid_count < period) {
            seed_sum += sq;
        }
        valid_count += 1;

        if (valid_count == period) {
            seed_idx = i;
            break;
        }
    }

    if (seed_idx < 0) {
        return;
    }

    double ema = seed_sum / static_cast<double>(period);
    row[seed_idx] = sqrt(fmax(ema, 0.0)) * 100.0;

    for (int i = seed_idx + 1; i < len; ++i) {
        double prev = data[i - 1];
        double curr = data[i];
        if (isfinite(prev) && isfinite(curr) && prev > 0.0 && curr > 0.0) {
            double ret = log(curr / prev);
            double sq = ret * ret;
            ema = beta * ema + alpha * sq;
        }
        row[i] = sqrt(fmax(ema, 0.0)) * 100.0;
    }
}
