// CUDA kernels for Kaufman Efficiency Ratio (ER)
//
// Batch kernel uses host-precomputed prefix sums of absolute diffs to compute
// denominators in O(1) per output. Many-series kernel assigns one thread per
// series and streams over time with a rolling denominator.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void er_batch_prefix_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_absdiff,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = nanf("");

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f;
        if (t >= warm) {
            const int start = t + 1 - period;
            // Denominator: sum |x[k+1]-x[k]| over window via prefix
            const double denom = prefix_absdiff[t] - prefix_absdiff[start];
            if (denom > 0.0) {
                const double a = (double)data[t];
                const double b = (double)data[start];
                const double delta = fabs(a - b);
                double er = delta / denom;
                if (er > 1.0) er = 1.0;
                out_val = (float)er;
            } else {
                out_val = 0.0f;
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// Rolling-denominator batch kernel: one thread per combo, sequential over time.
extern "C" __global__ void er_batch_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out) {
    int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int row_off = combo * len;
    const float nan_f = nanf("");
    for (int t = 0; t < len; ++t) out[row_off + t] = nan_f;
    if (period <= 0 || period > len) return;
    const int warm = first_valid + period - 1;
    if (warm >= len) return;

    // Initial denom over [first_valid .. warm-1]
    double roll = 0.0;
    for (int j = first_valid; j < warm; ++j) {
        const double v1 = (double)data[j + 1];
        const double v0 = (double)data[j];
        roll += fabs(v1 - v0);
    }
    int start = first_valid;
    for (int i = warm; i < len; ++i) {
        const double cur = (double)data[i];
        const double old = (double)data[start];
        const double delta = fabs(cur - old);
        float er = 0.0f;
        if (roll > 0.0) {
            double r = delta / roll;
            if (r > 1.0) r = 1.0;
            er = (float)r;
        }
        out[row_off + i] = er;
        if (i + 1 == len) break;
        const double add1 = (double)data[i + 1];
        const double add0 = (double)data[i];
        const double sub1 = (double)data[start + 1];
        const double sub0 = (double)data[start];
        const double add = fabs(add1 - add0);
        const double sub = fabs(sub1 - sub0);
        roll = roll + add - sub;
        ++start;
    }
}

// High-accuracy batch kernel: read double inputs, compute in double, write f32
extern "C" __global__ void er_batch_f64(
    const double* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out) {
    int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    const int period = periods[combo];
    const int row_off = combo * len;
    const float nan_f = nanf("");
    for (int t = 0; t < len; ++t) out[row_off + t] = nan_f;
    if (period <= 0 || period > len) return;
    const int warm = first_valid + period - 1;
    if (warm >= len) return;
    double roll = 0.0;
    for (int j = first_valid; j < warm; ++j) {
        roll += fabs(data[j + 1] - data[j]);
    }
    int start = first_valid;
    for (int i = warm; i < len; ++i) {
        const double delta = fabs(data[i] - data[start]);
        float er = 0.0f;
        if (roll > 0.0) {
            double r = delta / roll; if (r > 1.0) r = 1.0; er = (float)r;
        }
        out[row_off + i] = er;
        if (i + 1 == len) break;
        double add = fabs(data[i + 1] - data[i]);
        double sub = fabs(data[start + 1] - data[start]);
        roll = roll + add - sub; ++start;
    }
}

extern "C" __global__ void er_many_series_one_param_time_major_f32(
    const float* __restrict__ data_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    const float nan_f = nanf("");
    // Prefill column with NaNs
    for (int t = 0; t < rows; ++t) {
        out_tm[t * cols + s] = nan_f;
    }

    if (period <= 0 || period > rows) return;
    const int first_valid = first_valids[s];
    const int warm = first_valid + period - 1;
    if (warm >= rows) return;

    // Build initial rolling denominator over [first_valid .. warm-1]
    double roll = 0.0;
    int j = first_valid;
    while (j < warm) {
        const double v1 = (double)data_tm[(j + 1) * cols + s];
        const double v0 = (double)data_tm[j * cols + s];
        roll += fabs(v1 - v0);
        ++j;
    }

    int start = first_valid;
    int i = warm;
    while (i < rows) {
        const double cur = (double)data_tm[i * cols + s];
        const double old = (double)data_tm[start * cols + s];
        const double delta = fabs(cur - old);
        float er = 0.0f;
        if (roll > 0.0) {
            double r = delta / roll;
            if (r > 1.0) r = 1.0;
            er = (float)r;
        }
        out_tm[i * cols + s] = er;

        if (i + 1 == rows) break;
        const double add1 = (double)data_tm[(i + 1) * cols + s];
        const double add0 = (double)data_tm[i * cols + s];
        const double sub1 = (double)data_tm[(start + 1) * cols + s];
        const double sub0 = (double)data_tm[start * cols + s];
        const double add = fabs(add1 - add0);
        const double sub = fabs(sub1 - sub0);
        roll = roll + add - sub;
        ++start;
        ++i;
    }
}
