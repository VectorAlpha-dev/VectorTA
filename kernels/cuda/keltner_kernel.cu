// Simple Keltner combine kernels.
//
// These kernels do not compute ATR or the moving average themselves; instead
// they combine precomputed MA and ATR buffers into upper/middle/lower bands.
// This mirrors the composite pattern used in wrappers where we reuse existing
// CUDA MA and ATR implementations and avoid redundant work.

extern "C" __global__ void keltner_batch_f32(
    const float* __restrict__ ma_rows,           // [Prows x len]
    const float* __restrict__ atr_rows,          // [Prows x len]
    const int* __restrict__ row_period_idx,      // [R]
    const float* __restrict__ row_multipliers,   // [R]
    const int* __restrict__ row_warms,           // [R] absolute warm index per row
    int len,
    int rows,                                    // R (number of output rows / combos)
    float* __restrict__ out_upper,               // [R x len]
    float* __restrict__ out_middle,              // [R x len]
    float* __restrict__ out_lower                // [R x len]
) {
    int r = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || t >= len) return;

    int pidx = row_period_idx[r];
    int warm = row_warms[r];
    float mult = row_multipliers[r];

    // Row-major addressing
    int base_p = pidx * len;
    int base_r = r * len;

    if (t < warm) {
        out_middle[base_r + t] = NAN;
        out_upper [base_r + t] = NAN;
        out_lower [base_r + t] = NAN;
        return;
    }
    float mid = ma_rows[base_p + t];
    float a   = atr_rows[base_p + t];

    out_middle[base_r + t] = mid;
    out_upper [base_r + t] = mid + mult * a;
    out_lower [base_r + t] = mid - mult * a;
}

extern "C" __global__ void keltner_many_series_one_param_f32(
    const float* __restrict__ ma_tm,   // [rows*cols] time-major
    const float* __restrict__ atr_tm,  // [rows*cols] time-major
    const int*   __restrict__ first_valids, // [cols]
    int period,
    int cols,
    int rows,
    int elems,                         // rows * cols
    float multiplier,
    float* __restrict__ out_upper_tm,  // [rows*cols] time-major
    float* __restrict__ out_middle_tm, // [rows*cols] time-major
    float* __restrict__ out_lower_tm   // [rows*cols] time-major
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems) return;
    int t = idx / cols;
    int s = idx - t * cols;
    int fv = first_valids[s];
    int warm = fv + period - 1;
    if (t < warm) {
        out_middle_tm[idx] = NAN;
        out_upper_tm [idx] = NAN;
        out_lower_tm [idx] = NAN;
        return;
    }
    float mid = ma_tm[idx];
    float a   = atr_tm[idx];
    out_middle_tm[idx] = mid;
    out_upper_tm [idx] = mid + multiplier * a;
    out_lower_tm [idx] = mid - multiplier * a;
}
