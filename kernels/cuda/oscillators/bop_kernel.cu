// CUDA kernels for Balance of Power (BOP)
// bop = (close[t] - open[t]) / (high[t] - low[t])
// if (high[t] - low[t]) <= 0 -> 0.0
// warmup: write NaN for t < first_valid

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef BOP_NAN_F
#define BOP_NAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Small device helper to keep math single-place and inlined.
static __forceinline__ __device__ float bop_core(float o, float h, float l, float c) {
    const float den = h - l;
    return (den <= 0.0f) ? 0.0f : (c - o) / den;
}

// One series × “many‑params” (degenerate; Grid‑Y must be 1).
extern "C" __global__ void bop_batch_f32(const float* __restrict__ open,
                                         const float* __restrict__ high,
                                         const float* __restrict__ low,
                                         const float* __restrict__ close,
                                         int len,
                                         int first_valid,
                                         float* __restrict__ out)
{
    // Kept for API parity; expect combo == 0
    const int combo = blockIdx.y;
    if (UNLIKELY(combo > 0)) return;

    // ILP factor: how many elements a thread tries per tile.
    // 4 works well on Ada/Lovelace; change only if profiling indicates otherwise.
    constexpr int ILP = 4;

    const int tid   = threadIdx.x;
    const int bdim  = blockDim.x;
    const int gdim  = gridDim.x;

    // Each tile covers ILP * blockDim.x elements, so we stride by that in grid space.
    int base = blockIdx.x * bdim * ILP;
    const int step = gdim * bdim * ILP;

    for (; base < len; base += step) {
        // Unroll the ILP work so the compiler can overlap independent loads/divides.
        #pragma unroll
        for (int k = 0; k < ILP; ++k) {
            const int t = base + tid + k * bdim;
            if (t >= len) continue;

            if (LIKELY(t >= first_valid)) {
                const float o = open[t];
                const float h = high[t];
                const float l = low[t];
                const float c = close[t];
                out[t] = bop_core(o, h, l, c);
            } else {
                out[t] = BOP_NAN_F;
            }
        }
    }
}

// Many series × one‑param (time‑major layout: idx = t*num_series + s).
extern "C" __global__ void bop_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series id
    if (s >= num_series) return;

    const int fv = first_valids[s];
    if (UNLIKELY(fv < 0 || fv >= series_len)) {
        // All-NaN/invalid first_valid -> fill NaN
        float* o = out_tm + s;
        for (int t = 0; t < series_len; ++t, o += num_series) { *o = BOP_NAN_F; }
        return;
    }

    // Fill NaNs before first_valid (coalesced across series).
    {
        float* o = out_tm + s;
        for (int t = 0; t < fv; ++t, o += num_series) { *o = BOP_NAN_F; }
    }

    // Compute from first_valid onward; each thread walks the time axis of its series.
    const float* po = open_tm  + (size_t)fv * num_series + s;
    const float* ph = high_tm  + (size_t)fv * num_series + s;
    const float* pl = low_tm   + (size_t)fv * num_series + s;
    const float* pc = close_tm + (size_t)fv * num_series + s;
    float*       pd = out_tm   + (size_t)fv * num_series + s;

    // This loop has perfectly coalesced traffic across the warp at each t.
    // A modest unroll increases ILP without register bloat.
    #pragma unroll 4
    for (int t = fv; t < series_len; ++t) {
        const float v = bop_core(*po, *ph, *pl, *pc);
        *pd = v;

        po += num_series; ph += num_series; pl += num_series; pc += num_series; pd += num_series;
    }
}

