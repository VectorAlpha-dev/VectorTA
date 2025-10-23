// CUDA kernels for Fisher Transform (Ehlers variant) in FP32, optimized.
//
// Key changes vs the previous version:
// - O(N) sliding-window min/max via per-block monotonic deques in dynamic
//   shared memory for the "one series × many params" batch kernel.
// - FP64 removed; use FP32 math + FMA for the recurrence.
// - Fisher nonlinearity computed via atanhf() for improved stability near |x|→1.
// - Parallel warmup NaN prefill; single-thread sequential pass for the time
//   recurrence (per block).
//
// Behavior mirrors src/indicators/fisher.rs (scalar path):
// - Inputs are HL2 midpoints (0.5 * (high + low)) precomputed on host for batch
//   and many-series kernels to reduce memory traffic.
// - Warmup prefix: indices < (first_valid + period - 1) are NaN.
// - Recurrence:
//     val1 = 0.67*val1 + 0.66*norm, where norm = (x-min)/max(range,1e-3) - 0.5.
//     val1 is clamped to [-0.999, 0.999]
//     signal[t] = prev_fish
//     fisher[t] = atanhf(val1) + 0.5*prev_fish
// - Any NaN in the active window propagates via FP ops; host precompute ensures
//   first_valid respects NaN gating across high and low.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// --- helpers -----------------------------------------------------------------

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// ring-buffer helpers (period-sized capacity)
__device__ __forceinline__ int rb_dec(int x, int cap) { return (x == 0) ? (cap - 1) : (x - 1); }
__device__ __forceinline__ int rb_inc(int x, int cap) { return (x + 1 == cap) ? 0 : (x + 1); }

// One-series × many-params (batch). Each block handles one period row.
// Dynamic shared memory: 2 * period * sizeof(int) per block for deques.
extern "C" __global__ void fisher_batch_f32(const float* __restrict__ hl,
                                             const int*   __restrict__ periods,
                                             int series_len,
                                             int n_combos,
                                             int first_valid,
                                             float* __restrict__ out_fisher,
                                             float* __restrict__ out_signal) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    // Fast invalid-path: parallel fill of NaNs, then return
    auto fill_all_nan = [&](int len){
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            out_fisher[base + i] = NAN;
            out_signal[base + i] = NAN;
        }
    };

    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        fill_all_nan(series_len);
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        fill_all_nan(series_len);
        return;
    }

    const int warm = first_valid + period - 1;

    // Warmup prefix: NaNs
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_fisher[base + i] = NAN;
        out_signal[base + i] = NAN;
    }
    __syncthreads();

    // From here on, only thread 0 performs the sequential recurrence.
    if (threadIdx.x != 0) return;

    // Dynamic shared memory for deques (indices)
    extern __shared__ int s[];
    int* dq_min = s;               // capacity = period
    int* dq_max = s + period;      // capacity = period

    int hmin = 0, tmin = 0; // head/tail indices for min deque
    int hmax = 0, tmax = 0; // head/tail indices for max deque
    int cmin = 0, cmax = 0; // element counts to disambiguate empty vs full

    // Sequential state
    float prev_fish = 0.0f;
    float val1 = 0.0f;

    // Core loop: maintain deques and emit from the first full window (i >= warm)
    for (int i = first_valid; i < series_len; ++i) {
        const float xi = hl[i];

        // If we already have a full window, evict expired indices BEFORE pushing to avoid
        // ring-buffer overflow (t == h means empty AND full; eviction keeps size <= period).
        if (i >= warm) {
            const int window_start = i - period + 1;
            while (cmin > 0 && dq_min[hmin] < window_start) { hmin = rb_inc(hmin, period); --cmin; }
            while (cmax > 0 && dq_max[hmax] < window_start) { hmax = rb_inc(hmax, period); --cmax; }
        }

        // --- push xi into MIN deque (maintain nondecreasing values) ---
        while (cmin > 0) {
            int last = rb_dec(tmin, period);
            if (xi <= hl[dq_min[last]]) {
                tmin = last; // pop back
                --cmin;
            } else {
                break;
            }
        }
        dq_min[tmin] = i;
        tmin = rb_inc(tmin, period);
        ++cmin;

        // --- push xi into MAX deque (maintain nonincreasing values) ---
        while (cmax > 0) {
            int last = rb_dec(tmax, period);
            if (xi >= hl[dq_max[last]]) {
                tmax = last; // pop back
                --cmax;
            } else {
                break;
            }
        }
        dq_max[tmax] = i;
        tmax = rb_inc(tmax, period);
        ++cmax;

        // Emit only after first full window is formed
        if (i >= warm) {
            const float minv  = hl[dq_min[hmin]];
            const float maxv  = hl[dq_max[hmax]];
            const float range = fmaxf(maxv - minv, 1.0e-3f); // Ehlers guard
            const float norm  = (xi - minv) / range - 0.5f;

            // val1 = 0.67*val1 + 0.66*norm (one fused op improves both perf + accuracy)
            val1 = fmaf(0.67f, val1, 0.66f * norm);
            val1 = clampf(val1, -0.999f, 0.999f);

            out_signal[base + i] = prev_fish;
            // atanhf(x) == 0.5f * logf((1+x)/(1-x))
            const float fish = atanhf(val1) + 0.5f * prev_fish;
            out_fisher[base + i] = fish;
            prev_fish = fish;
        }
    }
}

// Many-series × one-param (time-major). Each thread handles one series (column).
extern "C" __global__ void fisher_many_series_one_param_f32(
    const float* __restrict__ hl_tm,        // time-major: [row * num_series + series]
    const int*   __restrict__ first_valids, // per-series first valid index
    int num_series,
    int series_len,
    int period,
    float* __restrict__ fisher_tm,
    float* __restrict__ signal_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        for (int r = 0; r < series_len; ++r) {
            const int idx = r * num_series + series;
            fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
        }
        return;
    }

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (UNLIKELY(first_valid >= series_len || (series_len - first_valid) < period)) {
        for (int r = 0; r < series_len; ++r) {
            const int idx = r * num_series + series;
            fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int r = 0; r < warm; ++r) {
        const int idx = r * num_series + series;
        fisher_tm[idx] = NAN; signal_tm[idx] = NAN;
    }

    float prev_fish = 0.0f;
    float val1 = 0.0f;
    for (int r = warm; r < series_len; ++r) {
        const int start = r + 1 - period;
        float minv = CUDART_INF_F;
        float maxv = -CUDART_INF_F;
        // Window scan with time-major stride (O(P) per step)
        for (int k = 0; k < period; ++k) {
            const int idx = (start + k) * num_series + series;
            const float x = hl_tm[idx];
            minv = fminf(minv, x);
            maxv = fmaxf(maxv, x);
        }
        const float range = fmaxf(maxv - minv, 1.0e-3f);
        const float x = hl_tm[r * num_series + series];
        const float norm = (x - minv) / range - 0.5f;

        val1 = fmaf(0.67f, val1, 0.66f * norm);
        val1 = clampf(val1, -0.999f, 0.999f);

        const int idxo = r * num_series + series;
        signal_tm[idxo] = prev_fish;
        const float fish = atanhf(val1) + 0.5f * prev_fish;
        fisher_tm[idxo] = fish;
        prev_fish = fish;
    }
}
