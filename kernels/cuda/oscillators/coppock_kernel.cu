// CUDA kernels for Coppock Curve (sum of two ROCs smoothed by WMA/SMA)
//
// Semantics:
// - Warmup = first_valid + max(short,long) + (ma_period - 1)
// - Before warmup: write NaN
// - Inputs may contain NaN; any NaN in the active window yields NaN at output
// - Accumulation done in float; adequate for default ranges and parity with other CUDA paths

extern "C" __global__ void coppock_batch_f32(
    const float* __restrict__ price, // [len]
    const float* __restrict__ inv,   // [len] precomputed 1/price
    int len,
    int first_valid,
    const int* __restrict__ shorts,      // [n_combos]
    const int* __restrict__ longs,       // [n_combos]
    const int* __restrict__ ma_periods,  // [n_combos]
    int n_combos,
    float* __restrict__ out              // [n_combos * len] row-major
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // time index
    int row = blockIdx.y;                           // parameter row
    if (row >= n_combos) return;
    if (i >= len) return;

    const int s = shorts[row];
    const int l = longs[row];
    const int m = ma_periods[row];
    const int largest = s > l ? s : l;
    const int warm = first_valid + largest + (m - 1);

    float* row_out = out + row * (size_t)len;
    if (i < warm) { row_out[i] = __int_as_float(0x7fffffff); return; } // NaN

    // Weighted Moving Average (default for Coppock). We implement WMA and SMA via the same kernel.
    // Denominator for WMA
    const float denom_w = 0.5f * (float)m * (float)(m + 1);

    // Compute WMA of sum_roc over window [i-m+1 .. i]
    float weighted = 0.0f;
    float sum = 0.0f; // for SMA
    bool has_nan = false;
    int w = 1;
    const int start = i - m + 1;
    for (int j = start; j <= i; ++j, ++w) {
        // sum_roc(j) = 100*((c/s - 1) + (c/l - 1))
        int js = j - s;
        int jl = j - l;
        // j is guaranteed >= first_valid + largest here (since i >= warm), so js,jl in-range
        float c = price[j];
        float ps = price[js];
        float pl = price[jl];
        // Propagate NaN semantics like CPU: any NaN in window -> NaN output
        if (__isnanf(c) || __isnanf(ps) || __isnanf(pl)) { has_nan = true; break; }
        float v = (c * (inv[js] + inv[jl]) - 2.0f) * 100.0f;
        weighted += v * (float)w;
        sum += v;
    }

    // Default Coppock is WMA; we emit WMA here. If has_nan, write NaN.
    row_out[i] = has_nan ? __int_as_float(0x7fffffff) : (weighted / denom_w);
}

// Many-series × one-param (time-major): each thread processes a series sequentially across time.
extern "C" __global__ void coppock_many_series_one_param_f32(
    const float* __restrict__ price_tm, // [rows * cols], time-major: t*cols + s
    const float* __restrict__ inv_tm,   // [rows * cols], 1/price_tm
    const int* __restrict__ first_valids, // [cols]
    int cols, int rows,
    int short_p, int long_p, int ma_period,
    float* __restrict__ out_tm // [rows * cols]
)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    const int first_valid = first_valids[s];
    const int largest = short_p > long_p ? short_p : long_p;
    const int warm = first_valid + largest + (ma_period - 1);
    const float denom_w = 0.5f * (float)ma_period * (float)(ma_period + 1);

    // leading NaNs
    for (int t = 0; t < rows; ++t) {
        float* dst = out_tm + (size_t)t * (size_t)cols + s;
        if (t < warm) { *dst = __int_as_float(0x7fffffff); continue; }

        // WMA window over sum_roc
        bool has_nan = false;
        float weighted = 0.0f;
        int w = 1;
        const int start = t - ma_period + 1;
        for (int j = start; j <= t; ++j, ++w) {
            const int js = j - short_p;
            const int jl = j - long_p;
            const size_t idxj  = (size_t)j  * (size_t)cols + s;
            const size_t idxjs = (size_t)js * (size_t)cols + s;
            const size_t idxjl = (size_t)jl * (size_t)cols + s;
            const float c  = price_tm[idxj];
            const float ps = price_tm[idxjs];
            const float pl = price_tm[idxjl];
            if (__isnanf(c) || __isnanf(ps) || __isnanf(pl)) { has_nan = true; break; }
            float v = (c * (inv_tm[idxjs] + inv_tm[idxjl]) - 2.0f) * 100.0f;
            weighted += v * (float)w;
        }
        *dst = has_nan ? __int_as_float(0x7fffffff) : (weighted / denom_w);
    }
}
