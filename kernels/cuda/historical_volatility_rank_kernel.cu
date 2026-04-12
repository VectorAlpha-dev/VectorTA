#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void historical_volatility_rank_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ hv_lengths,
    const int* __restrict__ rank_lengths,
    const double* __restrict__ annualization_scales,
    int n_combos,
    double* __restrict__ out_hvr,
    double* __restrict__ out_hv
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int hv_length = hv_lengths[combo_idx];
    int rank_length = rank_lengths[combo_idx];
    double annualization_scale = annualization_scales[combo_idx];
    double* row_hvr = out_hvr + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_hv = out_hv + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (hv_length <= 0 || rank_length <= 0 || !isfinite(annualization_scale) || annualization_scale <= 0.0) {
        for (int t = 0; t < len; ++t) {
            row_hvr[t] = CUDART_NAN;
            row_hv[t] = CUDART_NAN;
        }
        return;
    }

    for (int t = 0; t < len; ++t) {
        row_hvr[t] = CUDART_NAN;
        row_hv[t] = CUDART_NAN;
    }

    for (int t = hv_length; t < len; ++t) {
        int start = t + 1 - hv_length;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = start; i <= t; ++i) {
            double prev = data[i - 1];
            double curr = data[i];
            if (!isfinite(prev) || !isfinite(curr) || prev <= 0.0 || curr <= 0.0) {
                valid = false;
                break;
            }
            double ret = log(curr / prev);
            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            continue;
        }

        double n = static_cast<double>(hv_length);
        double mean = sum / n;
        double variance = (sumsq / n) - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        row_hv[t] = 100.0 * sqrt(variance) * annualization_scale;
    }

    for (int t = rank_length - 1; t < len; ++t) {
        int start = t + 1 - rank_length;
        bool valid = true;
        double min_v = CUDART_INF;
        double max_v = -CUDART_INF;
        double value = row_hv[t];

        for (int i = start; i <= t; ++i) {
            double hv = row_hv[i];
            if (!isfinite(hv)) {
                valid = false;
                break;
            }
            if (hv < min_v) {
                min_v = hv;
            }
            if (hv > max_v) {
                max_v = hv;
            }
        }

        if (!valid) {
            continue;
        }

        double range = max_v - min_v;
        if (!isfinite(range) || range <= 0.0) {
            row_hvr[t] = 0.0;
        } else {
            row_hvr[t] = 100.0 * (value - min_v) / range;
        }
    }
}
