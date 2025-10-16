// Kaufmanstop CUDA kernels
//
// Numeric contract matches the scalar implementation in src/indicators/kaufmanstop.rs:
// - warmup index per row = first_valid + period - 1; write NaN before warmup
// - base is low for long (stop below), high for short (stop above)
// - out = base + signed_mult * ma(range), where signed_mult = -mult for long, +mult for short
// - ignore NaNs exactly like scalar: NaN in base or MA propagates

extern "C" {
__global__ void kaufmanstop_axpy_row_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ ma_row,
    int len,
    float signed_mult,
    int warm,
    int base_is_low, // 1 = long (use low), 0 = short (use high)
    float* __restrict__ out_row
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    if (i < warm) {
        out_row[i] = __int_as_float(0x7fffffff); // NaN
        return;
    }

    float base = base_is_low ? low[i] : high[i];
    float ma   = ma_row[i];
    out_row[i] = base + ma * signed_mult;
}

// Many-series, one-param (time-major): process entire matrix
__global__ void kaufmanstop_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,  // [rows * cols]
    const float* __restrict__ low_tm,   // [rows * cols]
    const float* __restrict__ ma_tm,    // [rows * cols]
    const int*   __restrict__ first_valids, // [cols]
    int cols,
    int rows,
    float signed_mult,
    int base_is_low,
    int period,
    float* __restrict__ out_tm          // [rows * cols]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int s = idx % cols;    // series index
    int t = idx / cols;    // time index
    int warm = first_valids[s] + period - 1;
    if (t < warm) {
        out_tm[idx] = __int_as_float(0x7fffffff); // NaN
        return;
    }

    float base = base_is_low ? low_tm[idx] : high_tm[idx];
    float ma   = ma_tm[idx];
    out_tm[idx] = base + ma * signed_mult;
}
}

