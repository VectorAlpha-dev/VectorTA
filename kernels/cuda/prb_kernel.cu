// PRB (Polynomial Regression Bands) CUDA kernels
//
// Math policy:
// - For each output index i after warmup, compute windowed RHS moments S[r] = sum_{j=1..n} y_{i-n+j} * j^r
// - Use a host-precomputed A^{-1} (normal matrix inverse) per (n, k) row to solve coefficients: c = A^{-1} * S
// - Evaluate polynomial at x_pos = n - regression_offset (+ equ_from which is 0 here)
// - Compute stdev over the window as sqrt(E[y^2] - E[y]^2)
// - Before warmup, write NaN. If window contains any NaN, host-provided contig_valid ensures < n so we also write NaN.

extern "C" {

__device__ inline float horner_eval(const float* coeffs, int m, float x) {
    // coeffs[0] + coeffs[1]*x + ... + coeffs[m-1]*x^(m-1)
    // Horner in single precision; accumulators in float are adequate since coeffs are already f32
    float acc = 0.0f;
    for (int p = m - 1; p >= 0; --p) {
        acc = fmaf(acc, x, coeffs[p]);
    }
    return acc;
}

__global__ void prb_batch_f32(
    const float* __restrict__ data,  // series (len)
    const int len,
    const int first_valid,
    const int* __restrict__ periods, // per-row n
    const int* __restrict__ orders,  // per-row k
    const int* __restrict__ offsets, // per-row regression_offset
    const int combos,
    const int max_m,                 // max polynomial degree+1 stored in a_inv
    const float* __restrict__ a_inv, // per-row inverse (max_m x max_m), row-major, stride=max_m*max_m
    const int a_stride,
    const int* __restrict__ contig,  // contig valid counts for this (possibly smoothed) data
    const float ndev,
    const int* __restrict__ row_indices, // map local row -> absolute row
    float* __restrict__ out_main,
    float* __restrict__ out_up,
    float* __restrict__ out_lo
) {
    const int row = blockIdx.y;
    if (row >= combos) return;
    const int abs_row = row_indices ? row_indices[row] : row;

    const int n = periods[row];
    const int k = orders[row];
    const int m = k + 1;
    const int offset = offsets[row];
    const float x_pos = (float)n - (float)offset; // equ_from = 0 for CUDA path

    const float* arow = a_inv + row * a_stride; // top-left m x m is valid

    // Walk time sequentially to honor warmup; single thread per row for simplicity/semantics
    const int warm = first_valid + n - 1;
    bool poisoned = false;
    for (int i = 0; i < len; ++i) {
        const int out_idx = abs_row * len + i;
        // Warmup or poisoned or window contains NaN => NaN
        if (i < warm || poisoned || contig[i] < n) {
            if (i >= warm && contig[i] < n) { poisoned = true; }
            out_main[out_idx] = __int_as_float(0x7fffffff);
            out_up[out_idx] = __int_as_float(0x7fffffff);
            out_lo[out_idx] = __int_as_float(0x7fffffff);
            continue;
        }

        // Window base index (inclusive)
        const int base = i - n + 1;
        // Accumulate S[r] and stats in double for stability
        double sxy[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r) sxy[r] = 0.0;
        double sum = 0.0, sumsq = 0.0;
        for (int j = 1; j <= n; ++j) {
            const float y = data[base + j - 1];
            // y should be finite here because contig >= n
            double jd = (double)j;
            double yy = (double)y;
            sxy[0] += yy;
            double pwr = jd;
            for (int r = 1; r <= k; ++r) {
                sxy[r] += yy * pwr;
                pwr *= jd;
            }
            sum += yy;
            sumsq += yy * yy;
        }

        // Solve c = A^{-1} * S using top-left m x m of arow
        float coeffs[8];
        #pragma unroll
        for (int r = 0; r < m; ++r) {
            double acc = 0.0;
            const float* arow_r = arow + r * max_m; // row stride = max_m
            for (int c = 0; c < m; ++c) {
                acc += (double)arow_r[c] * sxy[c];
            }
            coeffs[r] = (float)acc;
        }

        const float reg = horner_eval(coeffs, m, x_pos);
        const double invn = 1.0 / (double)n;
        const double mean = sum * invn;
        double var = sumsq * invn - mean * mean;
        if (var < 0.0) var = 0.0;
        const float stdev = (float)sqrt(var);
        out_main[out_idx] = reg;
        out_up[out_idx] = reg + ndev * stdev;
        out_lo[out_idx] = reg - ndev * stdev;
    }
}

__global__ void prb_many_series_one_param_f32(
    const float* __restrict__ data_tm, // time-major: t * cols + s
    const int cols,
    const int rows,
    const int period,
    const int order,
    const int offset,
    const int max_m,
    const float* __restrict__ a_inv, // single inverse (top-left m x m)
    const int a_stride,
    const int* __restrict__ contig_tm, // time-major contig valid counts
    const int* __restrict__ first_valids, // per-column first valid index
    const float ndev,
    float* __restrict__ out_main_tm,
    float* __restrict__ out_up_tm,
    float* __restrict__ out_lo_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // column index
    if (s >= cols) return;
    const int n = period;
    const int k = order;
    const int m = k + 1;
    const float x_pos = (float)n - (float)offset;
    const float* ainv = a_inv; // same for all series

    const int fv = first_valids ? first_valids[s] : 0;
    const int warm = fv + n - 1;
    for (int t = 0; t < rows; ++t) {
        const int out_idx = t * cols + s;
        const int cont = contig_tm[out_idx];
        if (t < warm || cont < n) {
            const float nan = __int_as_float(0x7fffffff);
            out_main_tm[out_idx] = nan;
            out_up_tm[out_idx] = nan;
            out_lo_tm[out_idx] = nan;
            continue;
        }
        const int base = t - n + 1;
        double sxy[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r) sxy[r] = 0.0;
        double sum = 0.0, sumsq = 0.0;
        for (int j = 1; j <= n; ++j) {
            const float y = data_tm[(base + j - 1) * cols + s];
            double jd = (double)j;
            double yy = (double)y;
            sxy[0] += yy;
            double pwr = jd;
            for (int r = 1; r <= k; ++r) { sxy[r] += yy * pwr; pwr *= jd; }
            sum += yy; sumsq += yy * yy;
        }
        float coeffs[8];
        #pragma unroll
        for (int r = 0; r < m; ++r) {
            double acc = 0.0; const float* rowp = ainv + r * max_m;
            for (int c = 0; c < m; ++c) acc += (double)rowp[c] * sxy[c];
            coeffs[r] = (float)acc;
        }
        const float reg = horner_eval(coeffs, m, x_pos);
        const double invn = 1.0 / (double)n;
        const double mean = sum * invn;
        double var = sumsq * invn - mean * mean; if (var < 0.0) var = 0.0;
        const float stdev = (float)sqrt(var);
        out_main_tm[out_idx] = reg;
        out_up_tm[out_idx] = reg + ndev * stdev;
        out_lo_tm[out_idx] = reg - ndev * stdev;
    }
}

} // extern "C"
