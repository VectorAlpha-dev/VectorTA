// CUDA kernels for Polarized Fractal Efficiency (PFE)
//
// Math per time t (period = P):
//   diff = x[t] - x[t-P]
//   long = sqrt(diff^2 + P^2)
//   denom = sum_{k=t-P+1..t} sqrt(1 + (x[k] - x[k-1])^2)
//   raw = 100 * long / denom   (0.0 if denom==0)
//   signed = diff > 0 ? raw : -raw
//   EMA smoothing: y[t] = alpha*signed + (1-alpha)*y[t-1], seeded with first signed
// Warmup/NaN semantics:
//   - Outputs are NaN for indices < first_valid + period
//   - Computation starts at t = first_valid + period

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// One thread per combo; sequential over time. Computes denominators via rolling
// update without storing a ring (recomputes outgoing step in double).
extern "C" __global__
void pfe_batch_f32(const float* __restrict__ data,
                   int len,
                   int first_valid,
                   const int* __restrict__ periods,
                   const int* __restrict__ smoothings,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int smoothing = smoothings[combo];
    const int row_off = combo * len;
    const float nan_f = nanf("");
    for (int t = 0; t < len; ++t) out[row_off + t] = nan_f;
    if (period <= 0 || smoothing <= 0 || period > len) return;

    const int start = first_valid + period; // earliest output index
    if (start >= len) return;

    const double p = (double)period;
    const double p2 = p * p;
    const double alpha = 2.0 / ((double)smoothing + 1.0);
    const double one_minus_alpha = 1.0 - alpha;

    // Initialize rolling denominator over [start - period + 1 .. start]
    // which corresponds to steps j..j+1 for j in [first_valid .. start-1]
    double denom = 0.0;
    for (int j = first_valid; j < start; ++j) {
        const double d = (double)data[j + 1] - (double)data[j];
        denom += sqrt(fma(d, d, 1.0));
    }
    int head = first_valid; // oldest step index corresponds to (head -> head+1)

    // EMA state
    bool ema_started = false;
    double ema_val = 0.0;

    for (int t = start; t < len; ++t) {
        const double cur = (double)data[t];
        const double past = (double)data[t - period];
        const double diff = cur - past;
        const double long_leg = sqrt(fma(diff, diff, p2));
        double raw = 0.0;
        if (denom > 0.0) raw = 100.0 * (long_leg / denom);
        const double signed_val = diff > 0.0 ? raw : -raw;

        double y;
        if (!ema_started) { ema_started = true; ema_val = signed_val; y = signed_val; }
        else { ema_val = fma(alpha, signed_val, one_minus_alpha * ema_val); y = ema_val; }
        out[row_off + t] = (float)y;

        if (t + 1 == len) break;
        // Update rolling denominator for next t: add step (t -> t+1), drop (head -> head+1)
        const double add_d = (double)data[t + 1] - (double)data[t];
        const double add_s = sqrt(fma(add_d, add_d, 1.0));
        const double sub_d = (double)data[head + 1] - (double)data[head];
        const double sub_s = sqrt(fma(sub_d, sub_d, 1.0));
        denom = denom + add_s - sub_s;
        ++head;
    }
}

// Prefix-based batch kernel: one thread per combo; uses host-precomputed
// prefix of step lengths to compute denom in O(1) per t.
extern "C" __global__
void pfe_batch_prefix_f32(const float* __restrict__ data,
                          const double* __restrict__ prefix,
                          int len,
                          int first_valid,
                          const int* __restrict__ periods,
                          const int* __restrict__ smoothings,
                          int n_combos,
                          float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int smoothing = smoothings[combo];
    const int row_off = combo * len;
    const float nan_f = nanf("");
    for (int t = 0; t < len; ++t) out[row_off + t] = nan_f;
    if (period <= 0 || smoothing <= 0 || period > len) return;

    const int start = first_valid + period;
    if (start >= len) return;

    const double p = (double)period;
    const double p2 = p * p;
    const double alpha = 2.0 / ((double)smoothing + 1.0);
    const double one_minus_alpha = 1.0 - alpha;

    bool ema_started = false;
    double ema_val = 0.0;

    for (int t = start; t < len; ++t) {
        const double cur = (double)data[t];
        const double past = (double)data[t - period];
        const double diff = cur - past;
        const double long_leg = sqrt(fma(diff, diff, p2));
        const double denom = prefix[t] - prefix[t - period];

        if (denom > 0.0) {
            double raw = 100.0 * (long_leg / denom);
            const double signed_val = diff > 0.0 ? raw : -raw;
            if (!ema_started) { ema_started = true; ema_val = signed_val; }
            else { ema_val = fma(alpha, signed_val, one_minus_alpha * ema_val); }
            out[row_off + t] = (float)ema_val;
        } else {
            // denom <= 0 or NaN: propagate NaN like CPU (prefix-NaN case)
            out[row_off + t] = nan_f;
        }
    }
}

// Many-series × one-parameter-set (time-major). One thread per series scans time.
// data_tm, out_tm: index = t * cols + s
extern "C" __global__
void pfe_many_series_one_param_time_major_f32(const float* __restrict__ data_tm,
                                              const int*   __restrict__ first_valids,
                                              int cols,
                                              int rows,
                                              int period,
                                              int smoothing,
                                              float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;
    if (period <= 0 || smoothing <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = NAN;
        return;
    }
    const int start = fv + period;
    for (int t = 0; t < start && t < rows; ++t) out_tm[t * cols + s] = NAN;
    if (start >= rows) return;

    const double p = (double)period;
    const double p2 = p * p;
    const double alpha = 2.0 / ((double)smoothing + 1.0);
    const double one_minus_alpha = 1.0 - alpha;

    // Initialize rolling denominator over [fv .. start-1] steps
    double denom = 0.0;
    for (int j = fv; j < start; ++j) {
        const double d = (double)data_tm[j * cols + s] - (double)data_tm[(j - 1) * cols + s];
        // Beware j==fv uses (j-1); ensure inputs before fv are either NaN or not used.
        // We instead compute using forward difference j..j+1 consistent with batch path:
        // Recompute with indices on the fly to avoid complications.
    }
    // Correct denom init: use steps from j in [fv .. start-1] of (j -> j+1)
    denom = 0.0;
    for (int j = fv; j < start; ++j) {
        const double d = (double)data_tm[(j + 1) * cols + s] - (double)data_tm[j * cols + s];
        denom += sqrt(fma(d, d, 1.0));
    }
    int head = fv;

    bool ema_started = false; double ema_val = 0.0;
    for (int t = start; t < rows; ++t) {
        const double cur = (double)data_tm[t * cols + s];
        const double past = (double)data_tm[(t - period) * cols + s];
        const double diff = cur - past;
        const double long_leg = sqrt(fma(diff, diff, p2));
        double raw = 0.0; if (denom > 0.0) raw = 100.0 * (long_leg / denom);
        const double signed_val = diff > 0.0 ? raw : -raw;
        double y;
        if (!ema_started) { ema_started = true; ema_val = signed_val; y = signed_val; }
        else { ema_val = fma(alpha, signed_val, one_minus_alpha * ema_val); y = ema_val; }
        out_tm[t * cols + s] = (float)y;

        if (t + 1 == rows) break;
        const double add_d = (double)data_tm[(t + 1) * cols + s] - (double)data_tm[t * cols + s];
        const double add_s = sqrt(fma(add_d, add_d, 1.0));
        const double sub_d = (double)data_tm[(head + 1) * cols + s] - (double)data_tm[head * cols + s];
        const double sub_s = sqrt(fma(sub_d, sub_d, 1.0));
        denom = denom + add_s - sub_s;
        ++head;
    }
}
