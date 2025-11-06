// TrendFlex CUDA kernels — CUDA 13, sm_89+ (Ada OK)
// Decision note:
// - Many-series × one-param kernel uses the fused pass (SSF + volatility in one sweep),
//   FMA, and prev_price carry. Tests show accuracy within tolerance and speed is improved.
// - Batch (one series × many params) retains a two-pass structure but computes SSF and
//   normalization in double internally (casting to f32 for storage) to match the f64 CPU
//   baseline within strict unit-test tolerances. This avoids tiny drift observed with the
//   fully fused f32 path on some period sweeps.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef TRENDFLEX_NAN
#define TRENDFLEX_NAN (__int_as_float(0x7fffffff))
#endif

// --- Tunables ---------------------------------------------------------------
// If 1, we assume the host prefills the output buffer with NaNs once up-front.
// If 0, this kernel fills [0, warm) to NaN per row/series.
#ifndef TRENDFLEX_ASSUME_OUT_PREFILLED
#define TRENDFLEX_ASSUME_OUT_PREFILLED 0
#endif

// If 1, use rsqrtf + one NR refinement (~1 ulp, faster). If 0, use sqrtf.
#ifndef TRENDFLEX_USE_RSQRT_NR
#define TRENDFLEX_USE_RSQRT_NR 0
#endif
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float trendflex_round_half(float v) {
    return roundf(v);
}

static __device__ __forceinline__ float inv_sqrt_pos(float x) {
#if TRENDFLEX_USE_RSQRT_NR
    // rsqrtf + one Newton-Raphson step: fast and very accurate (nearly 1 ulp).
    // See CUDA libdevice __nv_rsqrtf and common NR refinement guidance.
    float y = rsqrtf(x);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
#else
    return 1.0f / sqrtf(x);
#endif
}

// ------------------------------ Batch: one series × many parameter combos ---
extern "C" __global__ void trendflex_batch_f32(const float* __restrict__ prices,
                                               const int*   __restrict__ periods,
                                               int series_len,
                                               int n_combos,
                                               int first_valid,
                                               float* __restrict__ ssf_buf,
                                               float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];

    // Validate early to avoid useless memory traffic or prefix writes
    if (series_len <= 0 || period <= 0 || period >= series_len) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int base = combo * series_len;
    float* __restrict__ row_out = out     + base;
    float* __restrict__ row_ssf = ssf_buf + base; // used as rolling ring/scratch

    const int tail_len = series_len - first_valid;
    if (tail_len < period) return;

    // Super-smoother coefficients for ss_period = round(period/2)
    const float PI    = 3.14159265358979323846f;
    const float ROOT2 = 1.41421356237f;

    int ss_period = (int)trendflex_round_half(0.5f * (float)period);
    if (ss_period < 1) ss_period = 1;
    if (tail_len < ss_period) return;

    // Compute coefficients in double for closer parity with CPU scalar path
    const double inv_ss = 1.0 / (double)ss_period;
    const double a_d    = exp(-ROOT2 * PI * inv_ss);
    const double a_sq_d = a_d * a_d;
    const double b_d    = 2.0 * a_d * cos(ROOT2 * PI * inv_ss);
    const double c_d    = 0.5 * (1.0 + a_sq_d - b_d);

    // Warm index and prefix NaNs
    const int warm = first_valid + period;
    // Clear output and scratch (match original behavior for determinism)
    for (int i = 0; i < series_len; ++i) {
        row_out[i] = TRENDFLEX_NAN;
        row_ssf[i] = 0.0f;
    }

    if (warm >= series_len) return;

    // Build super smoother sequence in scratch buffer (aligned with output row)
    const int first_idx = first_valid;
    double prev2 = (double)prices[first_idx];
    row_ssf[first_idx] = (float)prev2;
    double prev1 = prev2;
    if (tail_len > 1) {
        prev1 = (double)prices[first_idx + 1];
        row_ssf[first_idx + 1] = (float)prev1;
    }

    for (int t = 2; t < tail_len; ++t) {
        const int idx = first_idx + t;
        const double cur_price = (double)prices[idx];
        const double prev_price = (double)prices[idx - 1];
        const double ss = c_d * (cur_price + prev_price) + b_d * prev1 - a_sq_d * prev2;
        row_ssf[idx] = (float)ss;
        prev2 = prev1;
        prev1 = ss;
    }

    double rolling_sum = 0.0;
    for (int t = 0; t < period; ++t) {
        rolling_sum += (double)row_ssf[first_idx + t];
    }

    const double tp_f = (double)period;
    const double inv_tp = 1.0 / tp_f;
    double ms_prev = 0.0;

    for (int idx = warm; idx < series_len; ++idx) {
        const double ss = (double)row_ssf[idx];
        const double my_sum = (tp_f * ss - rolling_sum) * inv_tp;
        const double ms_current = 0.04 * my_sum * my_sum + 0.96 * ms_prev;
        ms_prev = ms_current;

        float out_val = 0.0f;
        if (ms_current > 0.0) {
            out_val = (float)(my_sum / sqrt(ms_current));
        }
        row_out[idx] = out_val;

        const double ss_old = (double)row_ssf[idx - period];
        rolling_sum += ss - ss_old;
    }
}

// --------------- Many-series × one-parameter (time-major, single period) ---
extern "C" __global__ void trendflex_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ ssf_tm,
    float* __restrict__ out_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;
    if (series_len <= 0 || period <= 0 || period >= series_len) return;

    const int stride = num_series;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int tail_len = series_len - first_valid;
    if (tail_len < period) return;

    const double PI    = 3.1415926535897932384626433832795;
    const double ROOT2 = 1.4142135623730951;

    int ss_period = (int)trendflex_round_half(0.5f * (float)period);
    if (ss_period < 1) ss_period = 1;
    if (tail_len < ss_period) return;

    const double inv_ss = 1.0 / (double)ss_period;
    const double k = ROOT2 * PI * inv_ss;
    const double a_d    = exp(-k);
    const double a_sq_d = a_d * a_d;
    const double b_d    = 2.0 * a_d * cos(k);
    const double c_d    = 0.5 * (1.0 + a_sq_d - b_d);
    const float  a      = (float)a_d;
    const float  a_sq   = (float)a_sq_d;
    const float  b      = (float)b_d;
    const float  c      = (float)c_d;

    // helper for time-major addressing
    auto at = [stride, series](int row) { return row * stride + series; };

    const int warm = first_valid + period;
#if !TRENDFLEX_ASSUME_OUT_PREFILLED
    const int nan_end = warm < series_len ? warm : series_len;
    for (int row = 0; row < nan_end; ++row) {
        out_tm[at(row)] = TRENDFLEX_NAN;
    }
#endif
    if (warm >= series_len) return;

    // Seed ss and rolling sum on [first_valid, first_valid+period)
    const int fidx = first_valid;

    // t=0
    float prev2 = prices_tm[at(fidx)];
    ssf_tm[at(fidx)] = prev2;
    float rolling_sum = prev2;

    // t=1
    float prev1, prev_price;
    if (tail_len > 1) {
        const float p1 = prices_tm[at(fidx + 1)];
        prev1 = p1;
        ssf_tm[at(fidx + 1)] = prev1;
        rolling_sum += prev1;
        prev_price = p1;
    } else {
        return;
    }

    // t=2..period-1
    for (int t = 2; t < period; ++t) {
        const float cur_price = prices_tm[at(fidx + t)];
        const float ss = fmaf(c, (cur_price + prev_price),
                              fmaf(b, prev1, -a_sq * prev2));
        ssf_tm[at(fidx + t)] = ss;
        rolling_sum += ss;
        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }

    // main loop
    const float tp_f   = (float)period;
    const float inv_tp = 1.0f / tp_f;
    float ms_prev = 0.0f;

    for (int row = warm; row < series_len; ++row) {
        const float cur_price = prices_tm[at(row)];
        const float ss = fmaf(c, (cur_price + prev_price),
                              fmaf(b, prev1, -a_sq * prev2));

        const float my_sum  = (tp_f * ss - rolling_sum) * inv_tp;
        const float my_sum2 = my_sum * my_sum;
        const float ms_current = fmaf(0.04f, my_sum2, 0.96f * ms_prev);
        ms_prev = ms_current;

        float out_val = 0.0f;
        if (ms_current > 0.0f) {
            out_val = my_sum * inv_sqrt_pos(ms_current);
        }
        out_tm[at(row)] = out_val;

        const float ss_old = ssf_tm[at(row - period)];
        rolling_sum += ss - ss_old;
        ssf_tm[at(row)] = ss;

        prev2      = prev1;
        prev1      = ss;
        prev_price = cur_price;
    }
}
