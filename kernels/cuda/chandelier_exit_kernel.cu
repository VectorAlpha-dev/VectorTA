// Chandelier Exit CUDA kernels (FP32)
//
// Math pattern: recurrence/time-scan per parameter with windowed
// extremums over either close (use_close=1) or high/low (use_close=0).
// ATR computed per-thread using Wilder smoothing seeded by the average of
// the first `period` TR values after `first_valid`.
//
// Output layout (row-major):
// - Batch (one series × many params): rows = 2 * n_combos, cols = len
//   row 2*r     -> long stops for combo r
//   row 2*r + 1 -> short stops for combo r
// - Many-series, one param (time-major): rows = 2 * rows_tm, cols = cols_tm
//   upper half  -> long stops per series/time
//   lower half  -> short stops per series/time
//
// NaN/warmup semantics mirror scalar implementation.

extern "C" __global__ void chandelier_exit_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int    len,
    const int    first_valid,
    const int*   __restrict__ periods, // length = rows
    const float* __restrict__ mults,   // length = rows
    const int    rows,                 // number of parameter combos
    const int    use_close_flag,       // 1 => use close for extremums
    float*       __restrict__ out      // length = (2*rows) * len
)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    const int period = periods[r];
    const float mult = mults[r];
    const int warm = first_valid + period - 1;

    const int long_row  = 2 * r;
    const int short_row = 2 * r + 1;

    // Initialize output prefix to NaN (for indices < warm)
    for (int i = 0; i < len; ++i) {
        float* long_ptr  = out + (size_t)long_row * len + i;
        float* short_ptr = out + (size_t)short_row * len + i;
        if (i < warm) {
            *long_ptr  = __int_as_float(0x7fffffff); // NaN
            *short_ptr = __int_as_float(0x7fffffff);
        } else {
            // Assign later in main loop
            *long_ptr  = 0.0f;
            *short_ptr = 0.0f;
        }
    }

    // Wilder ATR state
    float prev_close = 0.0f;
    bool  prev_close_set = false;
    double atr = NAN; // keep in double for accumulation
    double warm_tr_sum = 0.0;
    int warm_count = 0;

    // Trailing state
    double long_raw_prev = NAN;
    double short_raw_prev = NAN;
    int    dir_prev = 1;

    for (int i = 0; i < len; ++i) {
        const float h = high[i];
        const float l = low[i];
        const float c = close[i];

        // Build ATR
        float tr = NAN;
        if (i >= first_valid) {
            const float hl = fabsf(h - l);
            if (!prev_close_set) {
                tr = hl; // seed uses only HL
                prev_close = c;
                prev_close_set = true;
            } else {
                const float hc = fabsf(h - prev_close);
                const float lc = fabsf(l - prev_close);
                tr = fmaxf(hl, fmaxf(hc, lc));
                prev_close = c;
            }

            if (warm_count < period) {
                if (!isnan(tr)) {
                    warm_tr_sum += (double)tr;
                } else {
                    // If TR is NaN, keep sum as-is; scalar ATR would treat
                    // non-finite inputs as invalid and warm will delay anyway.
                }
                ++warm_count;
                if (warm_count == period) {
                    atr = warm_tr_sum / (double)period;
                }
            } else {
                // Wilder smoothing
                if (!isnan(tr) && !isnan(atr)) {
                    atr = atr + ((double)tr - atr) / (double)period;
                }
            }
        }

        if (i < warm) {
            continue; // prefix already NaN
        }

        // Window extremums over last `period` bars
        float highest = __int_as_float(0x7fffffff); // start as NaN
        float lowest  = __int_as_float(0x7fffffff);
        int start = i - period + 1;
        if (start < 0) start = 0;
        for (int j = start; j <= i; ++j) {
            float vx_max = use_close_flag ? close[j] : high[j];
            float vx_min = use_close_flag ? close[j] : low[j];
            if (!isnan(vx_max)) {
                if (isnan(highest) || vx_max > highest) highest = vx_max;
            }
            if (!isnan(vx_min)) {
                if (isnan(lowest) || vx_min < lowest) lowest = vx_min;
            }
        }

        // If ATR or extremums are NaN, outputs should be NaN as well
        float* long_ptr  = out + (size_t)long_row * len + i;
        float* short_ptr = out + (size_t)short_row * len + i;
        if (isnan((float)atr) || isnan(highest) || isnan(lowest)) {
            *long_ptr  = __int_as_float(0x7fffffff);
            *short_ptr = __int_as_float(0x7fffffff);
            continue;
        }

        const double ls0 = (double)highest - (double)mult * atr;
        const double ss0 = (double)lowest  + (double)mult * atr;
        const double lsp = (i == warm || isnan(long_raw_prev)) ? ls0 : long_raw_prev;
        const double ssp = (i == warm || isnan(short_raw_prev)) ? ss0 : short_raw_prev;

        const bool use_prev = (i > warm);
        double ls = ls0;
        double ss = ss0;
        if (use_prev) {
            const double pc = (double)close[i - 1];
            if (pc > lsp) ls = ls0 > lsp ? ls0 : lsp; // max
            if (pc < ssp) ss = ss0 < ssp ? ss0 : ssp; // min
        }

        int d;
        if ((double)c > ssp) d = 1;
        else if ((double)c < lsp) d = -1;
        else d = dir_prev;

        long_raw_prev = ls;
        short_raw_prev = ss;
        dir_prev = d;

        *long_ptr  = (d == 1) ? (float)ls : __int_as_float(0x7fffffff);
        *short_ptr = (d == -1) ? (float)ss : __int_as_float(0x7fffffff);
    }
}

extern "C" __global__ void chandelier_exit_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int    cols,
    const int    rows,
    const int    period,
    const float  mult,
    const int*   __restrict__ first_valids, // length = cols (per-series)
    const int    use_close_flag,
    float*       __restrict__ out_tm        // length = (2*rows) * cols
)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x; // series index (column)
    if (s >= cols) return;

    const int warm_base = first_valids[s] + period - 1;
    const int long_offset  = 0;              // first matrix
    const int short_offset = rows * cols;    // second matrix stacked after rows

    // Initialize prefix to NaN
    for (int t = 0; t < rows; ++t) {
        float* lp = out_tm + long_offset  + t * cols + s;
        float* sp = out_tm + short_offset + t * cols + s;
        if (t < warm_base) {
            *lp = __int_as_float(0x7fffffff);
            *sp = __int_as_float(0x7fffffff);
        } else {
            *lp = 0.0f; *sp = 0.0f; // will be written later
        }
    }

    // Wilder ATR per series
    double atr = NAN;
    double warm_tr_sum = 0.0;
    int warm_count = 0;
    float prev_close = 0.0f; bool prev_set = false;

    // Trailing state per series
    double long_raw_prev = NAN;
    double short_raw_prev = NAN;
    int    dir_prev = 1;

    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];

        float tr = NAN;
        if (t >= first_valids[s]) {
            const float hl = fabsf(h - l);
            if (!prev_set) { tr = hl; prev_close = c; prev_set = true; }
            else {
                const float hc = fabsf(h - prev_close);
                const float lc = fabsf(l - prev_close);
                tr = fmaxf(hl, fmaxf(hc, lc));
                prev_close = c;
            }
            if (warm_count < period) {
                if (!isnan(tr)) warm_tr_sum += (double)tr;
                ++warm_count;
                if (warm_count == period) atr = warm_tr_sum / (double)period;
            } else {
                if (!isnan(tr) && !isnan(atr)) atr = atr + ((double)tr - atr) / (double)period;
            }
        }

        if (t < warm_base) continue;

        float highest = __int_as_float(0x7fffffff);
        float lowest  = __int_as_float(0x7fffffff);
        int start = t - period + 1; if (start < 0) start = 0;
        for (int j = start; j <= t; ++j) {
            const int jdx = j * cols + s;
            float vx_max = use_close_flag ? close_tm[jdx] : high_tm[jdx];
            float vx_min = use_close_flag ? close_tm[jdx] : low_tm[jdx];
            if (!isnan(vx_max)) { if (isnan(highest) || vx_max > highest) highest = vx_max; }
            if (!isnan(vx_min)) { if (isnan(lowest)  || vx_min < lowest)  lowest  = vx_min; }
        }

        float* lp = out_tm + long_offset  + t * cols + s;
        float* sp = out_tm + short_offset + t * cols + s;
        if (isnan((float)atr) || isnan(highest) || isnan(lowest)) {
            *lp = __int_as_float(0x7fffffff);
            *sp = __int_as_float(0x7fffffff);
            continue;
        }

        const double ls0 = (double)highest - (double)mult * atr;
        const double ss0 = (double)lowest  + (double)mult * atr;
        const double lsp = (t == warm_base || isnan(long_raw_prev)) ? ls0 : long_raw_prev;
        const double ssp = (t == warm_base || isnan(short_raw_prev)) ? ss0 : short_raw_prev;

        const bool use_prev = (t > warm_base);
        double ls = ls0; double ss = ss0;
        if (use_prev) {
            const double pc = (double)close_tm[(t - 1) * cols + s];
            if (pc > lsp) ls = ls0 > lsp ? ls0 : lsp;
            if (pc < ssp) ss = ss0 < ssp ? ss0 : ssp;
        }

        int d;
        if ((double)c > ssp) d = 1;
        else if ((double)c < lsp) d = -1;
        else d = dir_prev;
        long_raw_prev = ls; short_raw_prev = ss; dir_prev = d;
        *lp = (d == 1) ? (float)ls : __int_as_float(0x7fffffff);
        *sp = (d == -1) ? (float)ss : __int_as_float(0x7fffffff);
    }
}

