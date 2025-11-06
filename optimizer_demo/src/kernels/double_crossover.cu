// Same kernel as in gpu_backtester_demo to compute metrics only for ALMA crossovers
extern "C" __global__
void double_cross_backtest_f32(
    const float* __restrict__ fast_ma, // [Pf_tile, T]
    int Pf_tile,
    int Pf_total,
    int f_offset,
    const float* __restrict__ slow_ma, // [Ps_tile, T]
    int Ps_tile,
    int Ps_total,
    int s_offset,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const float* __restrict__ prices,
    int T,
    int first_valid,
    float commission,
    int M,
    float* __restrict__ metrics_out // [F_total * S_total, M]
) {
    const int pair_local = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = Pf_tile * Ps_tile;
    if (pair_local >= pairs) return;
    const int pf_local = pair_local / Ps_tile;
    const int ps_local = pair_local % Ps_tile;
    const int pf_global = f_offset + pf_local;
    const int ps_global = s_offset + ps_local;
    const int period_f = fast_periods[pf_global];
    const int period_s = slow_periods[ps_global];
    const int t0 = first_valid + max(period_f, period_s) - 1;

    const float* f_row = fast_ma + (size_t)pf_local * (size_t)T;
    const float* s_row = slow_ma + (size_t)ps_local * (size_t)T;

    float equity = 1.0f;
    float peak = 1.0f;
    float max_dd = 0.0f;
    int trades = 0;
    int pos = 0; // -1,0,1
    long long exposure_steps = 0;
    double mean = 0.0, m2 = 0.0; long long n = 0;

    if (t0 + 1 >= T) {
        const size_t pair_global = (size_t)pf_global * (size_t)Ps_total + (size_t)ps_global;
        const size_t base = pair_global * (size_t)M;
        for (int k = 0; k < M; ++k) metrics_out[base + k] = 0.0f;
        return;
    }
    pos = (f_row[t0] > s_row[t0]) ? 1 : ((f_row[t0] < s_row[t0]) ? -1 : 0);
    for (int t = t0 + 1; t < T; ++t) {
        const float f_prev = f_row[t - 1];
        const float s_prev = s_row[t - 1];
        const float f_cur = f_row[t];
        const float s_cur = s_row[t];
        const int sign_prev = (f_prev > s_prev) ? 1 : ((f_prev < s_prev) ? -1 : 0);
        const int sign_cur  = (f_cur  > s_cur) ? 1 : ((f_cur  < s_cur) ? -1 : 0);
        if (sign_cur != pos) {
            if (pos == 0 && sign_cur != 0) { if (commission > 0.0f) equity *= (1.0f - commission); trades += 1; pos = sign_cur; }
            else if (pos != 0 && sign_cur == 0) { if (commission > 0.0f) equity *= (1.0f - commission); trades += 1; pos = 0; }
            else if (pos != 0 && sign_cur != 0) { if (commission > 0.0f) equity *= (1.0f - commission) * (1.0f - commission); trades += 2; pos = sign_cur; }
        }
        const float ret_t = prices[t] / prices[t - 1] - 1.0f;
        const float strat_ret = (pos == 0) ? 0.0f : (float)pos * ret_t;
        equity *= (1.0f + strat_ret);
        if (equity > peak) peak = equity;
        const float dd = (peak > 0.0f) ? (peak - equity) / peak : 0.0f;
        if (dd > max_dd) max_dd = dd;
        n += 1; const double x = (double)strat_ret; const double d = x - mean; mean += d / (double)n; m2 += d * (x - mean);
        exposure_steps += (long long)(pos != 0);
    }
    const double variance = (n > 1) ? (m2 / (double)(n - 1)) : 0.0;
    const double stddev = variance > 0.0 ? sqrt(variance) : 0.0;
    const size_t pair_global = (size_t)pf_global * (size_t)Ps_total + (size_t)ps_global;
    const size_t base = pair_global * (size_t)M;
    if (M > 0) metrics_out[base + 0] = equity - 1.0f;
    if (M > 1) metrics_out[base + 1] = (float)trades;
    if (M > 2) metrics_out[base + 2] = max_dd;
    if (M > 3) metrics_out[base + 3] = (float)mean;
    if (M > 4) metrics_out[base + 4] = (float)stddev;
}

