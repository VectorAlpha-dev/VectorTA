// Ultimate Oscillator CUDA kernels
// Category: Prefix-sum/rational. Kernels consume host-precomputed prefix sums
// for CMTL (close - true low) and TR (true range). Outputs match scalar warmup
// and NaN semantics.

extern "C" {

// Batch: one series × many params. Grid mapping:
// - block.x over time indices (0..len-1)
// - grid.y over parameter rows (combos)
// Args:
//  pcmtl  : double prefix sums of CMTL, length = len+1
//  ptr    : double prefix sums of TR,    length = len+1
//  len    : number of time samples
//  first  : first valid index (both i-1 and i must be valid)
//  p1s,p2s,p3s: per-row periods (size n_combos)
//  nrows  : number of parameter combinations
//  out    : row-major [nrows x len]
__global__ void ultosc_batch_f32(
    const double* __restrict__ pcmtl,
    const double* __restrict__ ptr,
    int len,
    int first,
    const int* __restrict__ p1s,
    const int* __restrict__ p2s,
    const int* __restrict__ p3s,
    int nrows,
    float* __restrict__ out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= nrows || i >= len) return;

    const int p1 = p1s[row];
    const int p2 = p2s[row];
    const int p3 = p3s[row];
    const int maxp = max(p1, max(p2, p3));
    const int start = first + maxp - 1;

    float* row_out = out + (size_t)row * (size_t)len;
    if (i < start) {
        row_out[i] = __int_as_float(0x7fffffff); // NaN
        return;
    }

    const double s1a = pcmtl[i + 1] - pcmtl[i + 1 - p1];
    const double s1b = ptr[i + 1] - ptr[i + 1 - p1];
    const double s2a = pcmtl[i + 1] - pcmtl[i + 1 - p2];
    const double s2b = ptr[i + 1] - ptr[i + 1 - p2];
    const double s3a = pcmtl[i + 1] - pcmtl[i + 1 - p3];
    const double s3b = ptr[i + 1] - ptr[i + 1 - p3];

    const double t1 = (s1b != 0.0) ? (s1a / s1b) : 0.0;
    const double t2 = (s2b != 0.0) ? (s2a / s2b) : 0.0;
    const double t3 = (s3b != 0.0) ? (s3a / s3b) : 0.0;

    // weights: 100/7 * (4,2,1)
    const double inv7_100 = 100.0 / 7.0;
    const double w1 = inv7_100 * 4.0;
    const double w2 = inv7_100 * 2.0;
    const double w3 = inv7_100 * 1.0;

    const double acc = w1 * t1 + w2 * t2 + w3 * t3;
    row_out[i] = (float)acc;
}

// Many-series × one-param (time-major).
// Mapping:
//  - block.x over time rows (0..rows-1)
//  - block.y over series within tile
// Args:
//  pcmtl_tm, ptr_tm: double prefix matrices of shape [(rows+1) x cols] in time-major layout.
//  cols, rows: matrix dims
//  p1,p2,p3 : shared periods
//  first_valids: per-series first valid row indices (length cols)
//  out_tm   : output matrix [rows x cols] time-major
__global__ void ultosc_many_series_one_param_f32(
    const double* __restrict__ pcmtl_tm,
    const double* __restrict__ ptr_tm,
    int cols,
    int rows,
    int p1,
    int p2,
    int p3,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x; // time row
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series column
    if (t >= rows || s >= cols) return;

    const int maxp = max(p1, max(p2, p3));
    const int first = first_valids[s];
    const int start = first + maxp - 1;

    float* out_row = out_tm + (size_t)t * (size_t)cols;
    if (t < start) {
        out_row[s] = __int_as_float(0x7fffffff); // NaN
        return;
    }

    const int idx_now = (t + 1) * cols + s;
    const int idx_1 = idx_now - p1 * cols;
    const int idx_2 = idx_now - p2 * cols;
    const int idx_3 = idx_now - p3 * cols;

    const double s1a = pcmtl_tm[idx_now] - pcmtl_tm[idx_1];
    const double s1b = ptr_tm[idx_now] - ptr_tm[idx_1];
    const double s2a = pcmtl_tm[idx_now] - pcmtl_tm[idx_2];
    const double s2b = ptr_tm[idx_now] - ptr_tm[idx_2];
    const double s3a = pcmtl_tm[idx_now] - pcmtl_tm[idx_3];
    const double s3b = ptr_tm[idx_now] - ptr_tm[idx_3];

    const double t1 = (s1b != 0.0) ? (s1a / s1b) : 0.0;
    const double t2 = (s2b != 0.0) ? (s2a / s2b) : 0.0;
    const double t3 = (s3b != 0.0) ? (s3a / s3b) : 0.0;

    const double inv7_100 = 100.0 / 7.0;
    const double w1 = inv7_100 * 4.0;
    const double w2 = inv7_100 * 2.0;
    const double w3 = inv7_100 * 1.0;
    const double acc = w1 * t1 + w2 * t2 + w3 * t3;
    out_row[s] = (float)acc;
}

} // extern "C"

