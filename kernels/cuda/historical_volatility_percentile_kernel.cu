#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void historical_volatility_percentile_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ annual_lengths,
    int n_combos,
    double* __restrict__ out_hvp,
    double* __restrict__ out_hvp_sma
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int annual_length = annual_lengths[combo_idx];
    double* row_hvp = out_hvp + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_hvp_sma = out_hvp_sma + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (length < 2 || annual_length <= 0) {
        for (int t = 0; t < len; ++t) {
            row_hvp[t] = CUDART_NAN;
            row_hvp_sma[t] = CUDART_NAN;
        }
        return;
    }

    for (int t = 0; t < len; ++t) {
        row_hvp[t] = CUDART_NAN;
        row_hvp_sma[t] = CUDART_NAN;
    }

    for (int t = length - 1; t < len; ++t) {
        int start = t + 1 - length;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int j = start; j <= t; ++j) {
            double curr = data[j];
            if (!isfinite(curr) || curr <= 0.0) {
                valid = false;
                break;
            }

            double ret;
            if (j == 0) {
                ret = 0.0;
            } else {
                double prev = data[j - 1];
                if (!isfinite(prev) || prev <= 0.0) {
                    ret = 0.0;
                } else {
                    ret = log(curr / prev);
                }
            }

            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            continue;
        }

        double n = static_cast<double>(length);
        double mean = sum / n;
        double centered = sumsq - mean * mean * n;
        if (centered < 0.0) {
            centered = 0.0;
        }
        double sample_var = centered / static_cast<double>(length - 1);
        row_hvp_sma[t] = sqrt(sample_var) * sqrt(static_cast<double>(annual_length));
    }

    for (int t = annual_length - 1; t < len; ++t) {
        int start = t + 1 - annual_length;
        bool valid = true;
        int rank = 0;
        double current_hv = row_hvp_sma[t];

        for (int j = start; j <= t; ++j) {
            double hv = row_hvp_sma[j];
            if (!isfinite(hv)) {
                valid = false;
                break;
            }
            rank += static_cast<int>(hv < current_hv);
        }

        if (!valid) {
            continue;
        }

        row_hvp[t] = static_cast<double>(rank) * (100.0 / static_cast<double>(annual_length));
    }

    for (int t = length - 1; t < len; ++t) {
        int start = t + 1 - length;
        bool valid = true;
        double sum = 0.0;

        for (int j = start; j <= t; ++j) {
            double hvp = row_hvp[j];
            if (!isfinite(hvp)) {
                valid = false;
                break;
            }
            sum += hvp;
        }

        if (!valid) {
            row_hvp_sma[t] = CUDART_NAN;
            continue;
        }

        row_hvp_sma[t] = sum / static_cast<double>(length);
    }
}
