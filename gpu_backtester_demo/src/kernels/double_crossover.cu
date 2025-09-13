// GPU double-crossover backtest kernel for one price series.
// Assumes ALMA values already computed on device for tiles of fast and slow parameter sets.
// Layout:
//  - fast_ma: [Pf_tile, T] row-major (each row contiguous in time)
//  - slow_ma: [Ps_tile, T] row-major
//  - prices:  [T]
//  - fast_periods: [Pf_total] (we index by global fast index)
//  - slow_periods: [Ps_total]
//  - metrics_out: [F_total * S_total, M] row-major along metric dimension (stride M)
// Index mapping for global output: pair_global = (f_global) * S_total + (s_global)

extern "C" __global__
void double_cross_backtest_f32(
    const float* __restrict__ fast_ma, // [Pf_tile, T]
    int Pf_tile,
    int Pf_total,
    int f_offset, // starting fast index within global grid

    const float* __restrict__ slow_ma, // [Ps_tile, T]
    int Ps_tile,
    int Ps_total,
    int s_offset, // starting slow index within global grid

    const int* __restrict__ fast_periods, // [Pf_total]
    const int* __restrict__ slow_periods, // [Ps_total]

    const float* __restrict__ prices, // [T]
    int T,
    int first_valid,

    float commission, // commission fraction applied on entries/exits (e.g., 0.0005)
    int M,           // metrics per pair
    float* __restrict__ metrics_out // [F_total * S_total, M]
) {
    const int pair_local = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = Pf_tile * Ps_tile;
    if (pair_local >= pairs) return;

    // Map local pair to (pf_local, ps_local) and global indices
    const int pf_local = pair_local / Ps_tile;
    const int ps_local = pair_local % Ps_tile;
    const int pf_global = f_offset + pf_local;
    const int ps_global = s_offset + ps_local;

    const int period_f = fast_periods[pf_global];
    const int period_s = slow_periods[ps_global];
    const int t0 = first_valid + max(period_f, period_s) - 1;

    // Row pointers into MA tiles
    const float* f_row = fast_ma + (size_t)pf_local * (size_t)T;
    const float* s_row = slow_ma + (size_t)ps_local * (size_t)T;

    // Backtest state
    float equity = 1.0f;
    float peak = 1.0f;
    float max_dd = 0.0f;
    int trades = 0;
    int pos = 0; // -1 = short, 0 = flat, 1 = long
    long long exposure_steps = 0;

    // Stats for returns
    double mean = 0.0;
    double m2 = 0.0;
    long long n = 0;

    if (t0 + 1 >= T) {
        // Not enough data to simulate; write zeros/NaNs
        const size_t pair_global = (size_t)pf_global * (size_t)Ps_total + (size_t)ps_global;
        const size_t base = pair_global * (size_t)M;
        if (M > 0) metrics_out[base + 0] = 0.0f;      // total_return
        if (M > 1) metrics_out[base + 1] = 0.0f;      // trades
        if (M > 2) metrics_out[base + 2] = 0.0f;      // max_dd
        if (M > 3) metrics_out[base + 3] = 0.0f;      // mean_ret
        if (M > 4) metrics_out[base + 4] = 0.0f;      // std_ret
        return;
    }

    // Initialize position based on initial relation (no trade fee at init)
    pos = (f_row[t0] > s_row[t0]) ? 1 : ((f_row[t0] < s_row[t0]) ? -1 : 0);

    // Iterate from t0+1 to T-1
    for (int t = t0 + 1; t < T; ++t) {
        const float f_prev = f_row[t - 1];
        const float s_prev = s_row[t - 1];
        const float f_cur = f_row[t];
        const float s_cur = s_row[t];

        // Generate signals
        const int sign_prev = (f_prev > s_prev) ? 1 : ((f_prev < s_prev) ? -1 : 0);
        const int sign_cur  = (f_cur  > s_cur) ? 1 : ((f_cur  < s_cur) ? -1 : 0);

        if (sign_cur != pos) {
            if (pos == 0 && sign_cur != 0) {
                // Enter new side
                if (commission > 0.0f) equity *= (1.0f - commission);
                trades += 1;
                pos = sign_cur;
            } else if (pos != 0 && sign_cur == 0) {
                // Exit to flat
                if (commission > 0.0f) equity *= (1.0f - commission);
                trades += 1;
                pos = 0;
            } else if (pos != 0 && sign_cur != 0) {
                // Flip position: pay exit + entry commissions
                if (commission > 0.0f) equity *= (1.0f - commission) * (1.0f - commission);
                trades += 2;
                pos = sign_cur;
            }
        }

        // Price return
        const float ret_t = prices[t] / prices[t - 1] - 1.0f;
        const float strat_ret = (pos == 0) ? 0.0f : (float)pos * ret_t;

        // Update equity and drawdown
        equity *= (1.0f + strat_ret);
        if (equity > peak) peak = equity;
        const float dd = (peak > 0.0f) ? (peak - equity) / peak : 0.0f;
        if (dd > max_dd) max_dd = dd;

        // Welford for strat returns
        n += 1;
        const double x = (double)strat_ret;
        const double delta = x - mean;
        mean += delta / (double)n;
        m2 += delta * (x - mean);

        // Exposure
        exposure_steps += (long long)pos;
    }

    const double variance = (n > 1) ? (m2 / (double)(n - 1)) : 0.0;
    const double stddev = variance > 0.0 ? sqrt(variance) : 0.0;

    const size_t pair_global = (size_t)pf_global * (size_t)Ps_total + (size_t)ps_global;
    const size_t base = pair_global * (size_t)M;
    if (M > 0) metrics_out[base + 0] = equity - 1.0f;     // total_return
    if (M > 1) metrics_out[base + 1] = (float)trades;     // trades
    if (M > 2) metrics_out[base + 2] = max_dd;            // max_drawdown
    if (M > 3) metrics_out[base + 3] = (float)mean;       // mean_ret per-step
    if (M > 4) metrics_out[base + 4] = (float)stddev;     // std_ret per-step
}
