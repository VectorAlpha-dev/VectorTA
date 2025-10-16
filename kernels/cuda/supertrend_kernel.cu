// SuperTrend CUDA kernels
//
// Pattern: Recurrence/IIR over time. Each output row (combo) or series is
// computed sequentially along t using precomputed ATR and HL2 midpoints.
//
// Inputs are f32; warmup semantics match scalar: warm = first_valid + period - 1.

extern "C" __global__ void supertrend_batch_f32(
    const float* __restrict__ hl2,                 // [len]
    const float* __restrict__ close,               // [len]
    const float* __restrict__ atr_rows,            // [Prows x len]
    const int*   __restrict__ row_period_idx,      // [R] maps row -> period row index in atr_rows
    const float* __restrict__ row_factors,         // [R]
    const int*   __restrict__ row_warms,           // [R] absolute warm index per row
    int len,
    int rows,                                      // R (number of output rows / combos)
    float* __restrict__ out_trend,                 // [R x len]
    float* __restrict__ out_changed                // [R x len]
) {
    int r = blockIdx.x; // one thread per row
    if (r >= rows || threadIdx.x != 0) return;

    const int pidx = row_period_idx[r];
    const int warm = row_warms[r];
    const float factor = row_factors[r];

    const int base_p = pidx * len;
    const int base_r = r * len;

    // Warmup prefix
    for (int t = 0; t < min(warm, len); ++t) {
        out_trend  [base_r + t] = NAN;
        out_changed[base_r + t] = NAN;
    }
    if (warm >= len) return;

    // Seed at warm
    float hl_w = hl2[warm];
    float atr_w = atr_rows[base_p + warm];
    float prev_upper = hl_w + factor * atr_w;
    float prev_lower = hl_w - factor * atr_w;
    float last_close = close[warm];

    bool upper_state;
    if (last_close <= prev_upper) {
        out_trend  [base_r + warm] = prev_upper;
        out_changed[base_r + warm] = 0.0f;
        upper_state = true;
    } else {
        out_trend  [base_r + warm] = prev_lower;
        out_changed[base_r + warm] = 0.0f;
        upper_state = false;
    }

    // Scan forward
    for (int t = warm + 1; t < len; ++t) {
        const float hl = hl2[t];
        const float a  = atr_rows[base_p + t];
        float upper_basic = hl + factor * a;
        float lower_basic = hl - factor * a;

        const float prev_c = last_close;
        float curr_upper = upper_basic;
        if (prev_c <= prev_upper) curr_upper = fminf(curr_upper, prev_upper);
        float curr_lower = lower_basic;
        if (prev_c >= prev_lower) curr_lower = fmaxf(curr_lower, prev_lower);

        const float c = close[t];
        if (upper_state) {
            if (c <= curr_upper) {
                out_trend  [base_r + t] = curr_upper;
                out_changed[base_r + t] = 0.0f;
            } else {
                out_trend  [base_r + t] = curr_lower;
                out_changed[base_r + t] = 1.0f;
                upper_state = false;
            }
        } else {
            if (c >= curr_lower) {
                out_trend  [base_r + t] = curr_lower;
                out_changed[base_r + t] = 0.0f;
            } else {
                out_trend  [base_r + t] = curr_upper;
                out_changed[base_r + t] = 1.0f;
                upper_state = true;
            }
        }

        prev_upper = curr_upper;
        prev_lower = curr_lower;
        last_close = c;
    }
}

extern "C" __global__ void supertrend_many_series_one_param_f32(
    const float* __restrict__ hl2_tm,          // [rows*cols] time-major
    const float* __restrict__ close_tm,        // [rows*cols] time-major
    const float* __restrict__ atr_tm,          // [rows*cols] time-major
    const int*   __restrict__ first_valids,    // [cols]
    int period,
    int cols,
    int rows,
    float factor,
    float* __restrict__ out_trend_tm,          // [rows*cols] time-major
    float* __restrict__ out_changed_tm         // [rows*cols] time-major
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; // one thread per series
    if (s >= cols) return;

    const int fv = first_valids[s];
    const int warm = fv + period - 1;

    // Warmup
    for (int t = 0; t < min(warm, rows); ++t) {
        int idx = t * cols + s;
        out_trend_tm  [idx] = NAN;
        out_changed_tm[idx] = NAN;
    }
    if (warm >= rows) return;

    int idx_w = warm * cols + s;
    float hl_w = hl2_tm[idx_w];
    float atr_w = atr_tm[idx_w];
    float prev_upper = hl_w + factor * atr_w;
    float prev_lower = hl_w - factor * atr_w;
    float last_close = close_tm[idx_w];

    bool upper_state;
    if (last_close <= prev_upper) {
        out_trend_tm  [idx_w] = prev_upper;
        out_changed_tm[idx_w] = 0.0f;
        upper_state = true;
    } else {
        out_trend_tm  [idx_w] = prev_lower;
        out_changed_tm[idx_w] = 0.0f;
        upper_state = false;
    }

    for (int t = warm + 1; t < rows; ++t) {
        int idx = t * cols + s;
        float hl = hl2_tm[idx];
        float a  = atr_tm[idx];
        float upper_basic = hl + factor * a;
        float lower_basic = hl - factor * a;

        const float prev_c = last_close;
        float curr_upper = upper_basic;
        if (prev_c <= prev_upper) curr_upper = fminf(curr_upper, prev_upper);
        float curr_lower = lower_basic;
        if (prev_c >= prev_lower) curr_lower = fmaxf(curr_lower, prev_lower);

        const float c = close_tm[idx];
        if (upper_state) {
            if (c <= curr_upper) {
                out_trend_tm  [idx] = curr_upper;
                out_changed_tm[idx] = 0.0f;
            } else {
                out_trend_tm  [idx] = curr_lower;
                out_changed_tm[idx] = 1.0f;
                upper_state = false;
            }
        } else {
            if (c >= curr_lower) {
                out_trend_tm  [idx] = curr_lower;
                out_changed_tm[idx] = 0.0f;
            } else {
                out_trend_tm  [idx] = curr_upper;
                out_changed_tm[idx] = 1.0f;
                upper_state = true;
            }
        }

        prev_upper = curr_upper;
        prev_lower = curr_lower;
        last_close = c;
    }
}

