// CUDA kernels for Percentage Price Oscillator (PPO)
//
// Math patterns supported:
// - SMA mode (ma_mode = 0): prefix-sum/rational. Host provides f64 prefix sums
//   over the valid segment starting at first_valid. Kernel computes window sums
//   in O(1) per output and writes NaN for the warmup prefix (first_valid+slow-1).
// - EMA mode (ma_mode = 1): per-row sequential recurrence that mirrors the
//   scalar classic EMA seeding (first slow samples form slow SMA; fast seeded
//   from its SMA and then advanced until alignment; then standard EMA updates).
//
// Outputs are f32; critical accumulations use f64 when applicable.

#include <cuda_runtime.h>
#include <math.h>

// Write IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// One-series × many-params (row-major out: [combo][t])
// data: len
// prefix_sum: len+1 (only used for SMA mode; can be len=1 dummy otherwise)
// fasts, slows: length n_combos
// ma_mode: 0=SMA, 1=EMA
extern "C" __global__ void ppo_batch_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,
    int len,
    int first_valid,
    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    int ma_mode,
    int n_combos,
    float* __restrict__ out)
{
    if (len <= 0) return;
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int fast = fasts[combo];
    const int slow = slows[combo];
    if (fast <= 0 || slow <= 0) return;
    const int start_idx = first_valid + slow - 1; // warmup end index written first
    const int row_off = combo * len;
    const float nanf = f32_nan();

    if (ma_mode == 0) {
        // SMA path: parallel over time using prefix sums
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < len) {
            float y = nanf;
            if (t >= start_idx) {
                const int tr = t + 1;
                const double s_fast = prefix_sum[tr] - prefix_sum[tr - fast];
                const double s_slow = prefix_sum[tr] - prefix_sum[tr - slow];
                if (isfinite(s_fast) && isfinite(s_slow) && s_slow != 0.0) {
                    const double ratio = (s_fast * (double)slow) / (s_slow * (double)fast);
                    y = (float)(ratio * 100.0 - 100.0);
                } else {
                    y = nanf;
                }
            }
            out[row_off + t] = y;
            t += stride;
        }
        return;
    }

    // EMA path: only thread 0 performs sequential scan; others help prefix NaN init
    // Initialize prefix [0..start_idx) to NaN in parallel
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < min(start_idx, len); idx += gridDim.x * blockDim.x) {
        out[row_off + idx] = nanf;
    }
    __syncthreads();

    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (start_idx >= len) return;

    const double fa = 2.0 / (double)(fast + 1);
    const double sa = 2.0 / (double)(slow + 1);
    const double fb = 1.0 - fa;
    const double sb = 1.0 - sa;

    // Seed: sums over the first `slow` valid samples; fast_sum over overlap tail
    double slow_sum = 0.0;
    double fast_sum = 0.0;
    const int overlap = slow - fast;
    for (int k = 0; k < slow; ++k) {
        const double v = (double)data[first_valid + k];
        slow_sum += v;
        if (k >= overlap) fast_sum += v;
    }

    double fast_ema = fast_sum / (double)fast;
    double slow_ema = slow_sum / (double)slow;

    // Advance fast EMA until alignment at start_idx
    for (int i = first_valid + fast; i <= start_idx; ++i) {
        const double x = (double)data[i];
        fast_ema = fa * x + fb * fast_ema;
    }

    // First PPO at start_idx
    float y0 = nanf;
    if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
        const double ratio = fast_ema / slow_ema;
        y0 = (float)(ratio * 100.0 - 100.0);
    }
    out[row_off + start_idx] = y0;

    // Main loop
    for (int j = start_idx + 1; j < len; ++j) {
        const double x = (double)data[j];
        fast_ema = fa * x + fb * fast_ema;
        slow_ema = sa * x + sb * slow_ema;
        float y = nanf;
        if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
            const double ratio = fast_ema / slow_ema;
            y = (float)(ratio * 100.0 - 100.0);
        }
        out[row_off + j] = y;
    }
}

// Many-series × one-param (time-major)
// prices_tm: rows x cols (time-major: idx = t*cols + s)
// prefix_sum_tm: (rows*cols)+1, time-major running sums per series (only SMA). May be dummy for EMA.
// first_valids: length cols
// fast, slow, ma_mode
extern "C" __global__ void ppo_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const double* __restrict__ prefix_sum_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int fast,
    int slow,
    int ma_mode,
    float* __restrict__ out_tm)
{
    if (cols <= 0 || rows <= 0) return;
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series/column
    if (s >= cols) return;
    const int fv = max(0, first_valids[s]);
    const int start_idx = fv + slow - 1;
    const float nanf = f32_nan();

    if (ma_mode == 0) {
        // SMA path: parallelize over time
        const int tx = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        for (int t = tx; t < rows; t += stride) {
            float y = nanf;
            if (t >= start_idx) {
                const int wr = (t * cols + s) + 1;
                const int wl_fast = ((t - fast) * cols + s) + 1;
                const int wl_slow = ((t - slow) * cols + s) + 1;
                const double s_fast = prefix_sum_tm[wr] - prefix_sum_tm[wl_fast];
                const double s_slow = prefix_sum_tm[wr] - prefix_sum_tm[wl_slow];
                if (isfinite(s_fast) && isfinite(s_slow) && s_slow != 0.0) {
                    const double ratio = (s_fast * (double)slow) / (s_slow * (double)fast);
                    y = (float)(ratio * 100.0 - 100.0);
                }
            }
            out_tm[t * cols + s] = y;
        }
        return;
    }

    // EMA path: let lane 0 (threadIdx.x==0 && threadIdx.y==0) do the sequential scan for series s
    if (!(threadIdx.x == 0)) return; // only one thread along x for recurrence

    // Prefix NaN init
    for (int t = 0; t < min(start_idx, rows); ++t) {
        out_tm[t * cols + s] = nanf;
    }
    if (start_idx >= rows) return;

    const double fa = 2.0 / (double)(fast + 1);
    const double sa = 2.0 / (double)(slow + 1);
    const double fb = 1.0 - fa;
    const double sb = 1.0 - sa;

    // Seed from first `slow` values of this series
    double slow_sum = 0.0;
    double fast_sum = 0.0;
    const int overlap = slow - fast;
    for (int k = 0; k < slow; ++k) {
        const double v = (double)prices_tm[(fv + k) * cols + s];
        slow_sum += v;
        if (k >= overlap) fast_sum += v;
    }
    double fast_ema = fast_sum / (double)fast;
    double slow_ema = slow_sum / (double)slow;

    for (int i = fv + fast; i <= start_idx; ++i) {
        const double x = (double)prices_tm[i * cols + s];
        fast_ema = fa * x + fb * fast_ema;
    }
    float y0 = nanf;
    if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
        const double ratio = fast_ema / slow_ema;
        y0 = (float)(ratio * 100.0 - 100.0);
    }
    out_tm[start_idx * cols + s] = y0;

    for (int t = start_idx + 1; t < rows; ++t) {
        const double x = (double)prices_tm[t * cols + s];
        fast_ema = fa * x + fb * fast_ema;
        slow_ema = sa * x + sb * slow_ema;
        float y = nanf;
        if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
            const double ratio = fast_ema / slow_ema;
            y = (float)(ratio * 100.0 - 100.0);
        }
        out_tm[t * cols + s] = y;
    }
}

// Elementwise PPO from precomputed MA series (batch):
// fast_ma: nf x len (row-major), slow_ma: ns x len (row-major)
// out: (nf*ns) x len, with row mapping r = fi*ns + si
extern "C" __global__ void ppo_from_ma_batch_f32(
    const float* __restrict__ fast_ma,
    const float* __restrict__ slow_ma,
    int len,
    int nf,
    int ns,
    int first_valid,
    const int* __restrict__ slow_periods,
    int row_start,
    float* __restrict__ out)
{
    const int r = row_start + blockIdx.y; // global output row
    if (r >= nf * ns) return;
    const int fi = r / ns;
    const int si = r - fi * ns;
    const int fast_off = fi * len;
    const int slow_off = si * len;
    const int stride = gridDim.x * blockDim.x;
    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const float nanf = f32_nan();
    const int warm = first_valid + slow_periods[si] - 1;
    for (int t = t0; t < len; t += stride) {
        const float sf = slow_ma[slow_off + t];
        const float ff = fast_ma[fast_off + t];
        float y = nanf;
        if (t >= warm && isfinite(sf) && isfinite(ff) && sf != 0.0f) {
            y = (ff / sf) * 100.0f - 100.0f;
        }
        out[r * len + t] = y;
    }
}

// Elementwise PPO from precomputed MA series (many-series, time-major):
// fast_ma_tm and slow_ma_tm: rows x cols (time-major)
extern "C" __global__ void ppo_from_ma_many_series_one_param_time_major_f32(
    const float* __restrict__ fast_ma_tm,
    const float* __restrict__ slow_ma_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    int slow,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= cols) return;
    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nanf = f32_nan();
    const int warm = first_valids[s] + slow - 1;
    for (int t = t0; t < rows; t += stride) {
        const float sf = slow_ma_tm[t * cols + s];
        const float ff = fast_ma_tm[t * cols + s];
        float y = nanf;
        if (t >= warm && isfinite(sf) && isfinite(ff) && sf != 0.0f) {
            y = (ff / sf) * 100.0f - 100.0f;
        }
        out_tm[t * cols + s] = y;
    }
}
