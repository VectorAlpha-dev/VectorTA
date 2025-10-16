// CUDA kernels for VOSS (Ehlers Voss Filter)
//
// Math pattern: recurrence/IIR with predictive feedback.
// - Batch (one series × many params): each parameter row is processed by a single
//   thread with a sequential time scan. Warmup and NaN semantics match the scalar
//   implementation. Two outputs: voss and filt.
// - Many-series × one-param (time-major): each series is handled by one thread in
//   block.y with a sequential scan over rows.
//
// Numerics:
// - f64 accumulations for coefficients and feedback; outputs in f32.
// - For the predictor, we compute SumC via an O(m) loop over prior voss values
//   using already-written outputs from this row; NaN values are treated as 0.0,
//   and indices < start are clamped to 0.0, matching the scalar ring-buffer nz() policy.

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// Compute s1 matching the scalar implementation (for parity)
__device__ __forceinline__ double voss_s1(double g1) {
    const double inv_g1 = 1.0 / g1;
    const double root = sqrt(fmax(inv_g1 * inv_g1 - 1.0, 0.0));
    return inv_g1 - root;
}

extern "C" __global__ void voss_batch_f32(
    const double* __restrict__ prices,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const int* __restrict__ predicts,
    const double* __restrict__ bandwidths,
    int nrows,
    float* __restrict__ out_voss,
    float* __restrict__ out_filt)
{
    const int row = blockIdx.y;
    if (row >= nrows) return;

    const int p = periods[row];
    const int q = predicts[row];
    const double bw = bandwidths[row];
    if (p <= 0 || q < 0) return;

    const int order = 3 * q;
    const int min_index = max(max(p, 5), order);
    const int start = first_valid + min_index;
    const int row_off = row * len;

    // Fill warmup prefix with NaN for both outputs
    for (int t = threadIdx.x; t < min(start, len); t += blockDim.x) {
        out_voss[row_off + t] = f32_nan();
        out_filt[row_off + t] = f32_nan();
    }
    __syncthreads();

    // Set filt[start-2] and filt[start-1] to 0.0 if in range (parity with scalar)
    if (threadIdx.x == 0) {
        if (start - 2 >= 0 && start - 2 < len) out_filt[row_off + (start - 2)] = 0.0f;
        if (start - 1 >= 0 && start - 1 < len) out_filt[row_off + (start - 1)] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // sequential per-row
    if (start >= len) return;

    // Coefficients
    const double w0 = 2.0 * M_PI / (double)p;
    const double f1 = cos(w0);
    const double g1 = cos(bw * w0);
    const double s1 = voss_s1(g1);
    const double c1 = 0.5 * (1.0 - s1);
    const double c2 = f1 * (1.0 + s1);
    const double c3 = -s1;
    const double scale = 0.5 * (3.0 + (double)order);

    double prev_f1 = 0.0;
    double prev_f2 = 0.0;

    if (order == 0) {
        for (int i = start; i < len; ++i) {
            const double xi = prices[i];
            const double xim2 = prices[i - 2];
            const double diff = xi - xim2;
            const double t = c3 * prev_f2 + c1 * diff;
            const double f = c2 * prev_f1 + t;
            out_filt[row_off + i] = (float)f;
            out_voss[row_off + i] = (float)(scale * f);
            prev_f2 = prev_f1;
            prev_f1 = f;
        }
        return;
    }

    // O(1) predictor via rolling sums (parity with scalar):
    // A = sum of last m v; D = sum of k*v_{i-k}; SumC = D/m; update:
    // A' = A - v_old + v_new; D' = (D - A) + m*v_new
    double a_sum = 0.0;
    double d_sum = 0.0;
    for (int i = start; i < len; ++i) {
        const double xi = prices[i];
        const double xim2 = prices[i - 2];
        const double diff = xi - xim2;
        const double t = c3 * prev_f2 + c1 * diff;
        const double f = c2 * prev_f1 + t;
        out_filt[row_off + i] = (float)f;
        prev_f2 = prev_f1;
        prev_f1 = f;

        const double sumc = d_sum / (double)order;
        const double vi = scale * f - sumc;
        out_voss[row_off + i] = (float)vi;

        const double v_new = isnan(vi) ? 0.0 : vi;
        const int j_old = i - order;
        double v_old = 0.0;
        if (j_old >= start) {
            const float vv = out_voss[row_off + j_old];
            v_old = isnan(vv) ? 0.0 : (double)vv;
        }
        const double a_prev = a_sum;
        a_sum = a_prev - v_old + v_new;
        d_sum = (d_sum - a_prev) + (double)order * v_new;
    }
}

extern "C" __global__ void voss_many_series_one_param_time_major_f32(
    const double* __restrict__ data_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    int predict,
    double bandwidth,
    float* __restrict__ out_voss_tm,
    float* __restrict__ out_filt_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series index
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        // No valid data: write NaNs
        for (int t = threadIdx.x; t < rows; t += blockDim.x) {
            const int idx = t * cols + s;
            out_voss_tm[idx] = f32_nan();
            out_filt_tm[idx] = f32_nan();
        }
        return;
    }

    const int order = 3 * predict;
    const int min_index = max(max(period, 5), order);
    const int start = fv + min_index;

    // Fill warmup with NaNs
    for (int t = threadIdx.x; t < min(start, rows); t += blockDim.x) {
        const int idx = t * cols + s;
        out_voss_tm[idx] = f32_nan();
        out_filt_tm[idx] = f32_nan();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (start - 2 >= 0 && start - 2 < rows) out_filt_tm[(start - 2) * cols + s] = 0.0f;
        if (start - 1 >= 0 && start - 1 < rows) out_filt_tm[(start - 1) * cols + s] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (start >= rows) return;

    const double w0 = 2.0 * M_PI / (double)period;
    const double f1 = cos(w0);
    const double g1 = cos(bandwidth * w0);
    const double s1 = voss_s1(g1);
    const double c1 = 0.5 * (1.0 - s1);
    const double c2 = f1 * (1.0 + s1);
    const double c3 = -s1;
    const double scale = 0.5 * (3.0 + (double)order);

    double prev_f1 = 0.0;
    double prev_f2 = 0.0;

    if (order == 0) {
        for (int i = start; i < rows; ++i) {
            const int idx = i * cols + s;
            const double xi = data_tm[idx];
            const double xim2 = data_tm[(i - 2) * cols + s];
            const double diff = xi - xim2;
            const double t = c3 * prev_f2 + c1 * diff;
            const double f = c2 * prev_f1 + t;
            out_filt_tm[idx] = (float)f;
            out_voss_tm[idx] = (float)(scale * f);
            prev_f2 = prev_f1;
            prev_f1 = f;
        }
        return;
    }

    double a_sum = 0.0;
    double d_sum = 0.0;
    for (int i = start; i < rows; ++i) {
        const int idx = i * cols + s;
        const double xi = data_tm[idx];
        const double xim2 = data_tm[(i - 2) * cols + s];
        const double diff = xi - xim2;
        const double t = c3 * prev_f2 + c1 * diff;
        const double f = c2 * prev_f1 + t;
        out_filt_tm[idx] = (float)f;
        prev_f2 = prev_f1;
        prev_f1 = f;

        const double sumc = d_sum / (double)order;
        const double vi = scale * f - sumc;
        out_voss_tm[idx] = (float)vi;

        const double v_new = isnan(vi) ? 0.0 : vi;
        const int j_old = i - order;
        double v_old = 0.0;
        if (j_old >= start) {
            const float vv = out_voss_tm[j_old * cols + s];
            v_old = isnan(vv) ? 0.0 : (double)vv;
        }
        const double a_prev = a_sum;
        a_sum = a_prev - v_old + v_new;
        d_sum = (d_sum - a_prev) + (double)order * v_new;
    }
}
