#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double mro_safe_ratio(double num, double den) {
    if (isfinite(num) && isfinite(den) && den != 0.0) {
        return num / den;
    }
    return CUDART_NAN;
}

extern "C" __global__ void momentum_ratio_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out_line,
    double* __restrict__ out_signal
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double* row_line = out_line + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_line[t] = CUDART_NAN;
        row_signal[t] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    double alpha = 2.0 / static_cast<double>(period);
    bool has_ema = false;
    double ema_prev = 0.0;
    double emaa_prev = 0.0;
    double emab_prev = 0.0;
    double val_prev = CUDART_NAN;

    for (int t = 0; t < len; ++t) {
        double value = data[t];
        if (!isfinite(value)) {
            has_ema = false;
            ema_prev = 0.0;
            emaa_prev = 0.0;
            emab_prev = 0.0;
            val_prev = CUDART_NAN;
            continue;
        }

        double prev_ema_nz = has_ema ? ema_prev : 0.0;
        double ema = prev_ema_nz + alpha * (value - prev_ema_nz);
        double ratioa = has_ema ? mro_safe_ratio(ema, ema_prev) : CUDART_NAN;
        double emaa_input = isfinite(ratioa) && ratioa < 1.0 ? ratioa : 0.0;
        double emab_input = isfinite(ratioa) && ratioa > 1.0 ? ratioa : 0.0;
        double emaa = emaa_prev + alpha * (emaa_input - emaa_prev);
        double emab = emab_prev + alpha * (emab_input - emab_prev);
        double ratiob = mro_safe_ratio(ratioa, ratioa + emab);

        double val = CUDART_NAN;
        double denom = ratioa + ratiob * emaa;
        if (isfinite(ratioa) && isfinite(ratiob) && isfinite(emaa) && isfinite(denom) &&
            denom != 0.0) {
            val = 2.0 * ratioa / denom - 1.0;
        }

        row_line[t] = val;
        row_signal[t] = val_prev;

        has_ema = true;
        ema_prev = ema;
        emaa_prev = emaa;
        emab_prev = emab;
        val_prev = val;
    }
}
