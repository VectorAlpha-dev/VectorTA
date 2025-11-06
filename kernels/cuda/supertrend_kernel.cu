// SuperTrend CUDA kernels (optimized)
//
// Key ideas:
// - Warp-synchronous time loop with intra-warp broadcasts of shared inputs
//   (hl2, close) via __shfl_sync to reduce redundant loads.
// - Dynamic ATR sharing when a warp's period index is uniform.
// - FP32 throughout with fmaf for speed and better numerical behavior.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Quiet NaN bitpattern (no dependency on host memset)
__device__ __forceinline__ float qnan_f32() {
    return __int_as_float(0x7fc00000);
}

// Warp-wide integer minimum
__device__ __forceinline__ int warp_min_int(int v, unsigned mask) {
    for (int ofs = WARP_SIZE / 2; ofs > 0; ofs >>= 1) {
        int o = __shfl_down_sync(mask, v, ofs);
        v = (o < v) ? o : v;
    }
    return v;
}

// --- SuperTrend: one series, many parameter rows (optimized) ---
extern "C" __global__ void supertrend_batch_f32(
    const float* __restrict__ hl2,                 // [len]
    const float* __restrict__ close,               // [len]
    const float* __restrict__ atr_rows,            // [Prows x len]
    const int*   __restrict__ row_period_idx,      // [R]
    const float* __restrict__ row_factors,         // [R]
    const int*   __restrict__ row_warms,           // [R] absolute warm index per row
    int len,
    int rows,                                      // R (number of parameter combos)
    float* __restrict__ out_trend,                 // [R x len] row-major
    float* __restrict__ out_changed                // [R x len] row-major
) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x; // one thread per row
    if (r >= rows) return;

    // Row-specific metadata and base pointers
    const int   pidx   = row_period_idx[r];
    const int   warm   = row_warms[r];
    const float factor = row_factors[r];

    const int base_p = pidx * len;
    const int base_r = r    * len;

    const float* __restrict__ atr_row = atr_rows + base_p;
    float* __restrict__ out_tr = out_trend   + base_r;
    float* __restrict__ out_ch = out_changed + base_r;

    // Warp cooperation state
    const unsigned mask = __activemask();
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int src  = __ffs(mask) - 1; // leader lane

    // Prefill up to the warp-min warm with NaN, in lockstep
    const int warp_min_warm = warp_min_int(warm, mask);
    for (int t = 0; t < warp_min_warm; ++t) {
        out_tr[t] = qnan_f32();
        out_ch[t] = qnan_f32();
    }

    // Detect whether all lanes in this warp share the same period index
    const int warp_p0 = __shfl_sync(mask, pidx, src);
    const int same_p  = __all_sync(mask, pidx == warp_p0);
    const int base_p0 = warp_p0 * len; // valid if same_p

    // Per-row running state
    int   upper_state = 0; // 1 == upper, 0 == lower
    float prev_upper  = 0.0f;
    float prev_lower  = 0.0f;
    float last_close  = 0.0f;
    bool  active      = false; // becomes true at t == warm

    // Main time scan, warp-synchronous
    for (int t = warp_min_warm; t < len; ++t) {
        // Load shared series data once per warp and broadcast
        float hl_b = 0.0f, c_b = 0.0f, a_b = 0.0f;
        if (lane == src) {
            hl_b = hl2[t];
            c_b  = close[t];
            if (same_p) a_b = atr_rows[base_p0 + t];
        }
        const float hl = __shfl_sync(mask, hl_b, src);
        const float c  = __shfl_sync(mask, c_b,  src);
        const float a  = same_p ? __shfl_sync(mask, a_b, src) : atr_row[t];

        // Handle per-row warmup inline to preserve lockstep over t
        if (t < warm) {
            out_tr[t] = qnan_f32();
            out_ch[t] = qnan_f32();
            continue;
        }

        if (!active) {
            // Seed at warm
            prev_upper  = fmaf(factor,  a, hl);   // hl + factor * a
            prev_lower  = fmaf(-factor, a, hl);   // hl - factor * a
            last_close  = c;
            upper_state = (last_close <= prev_upper);
            out_tr[t]   = upper_state ? prev_upper : prev_lower;
            out_ch[t]   = 0.0f;
            active      = true;
            continue;
        }

        // Recurrence step
        const float upper_basic = fmaf(factor,  a, hl);
        const float lower_basic = fmaf(-factor, a, hl);

        const float curr_upper = (last_close <= prev_upper) ? fminf(upper_basic, prev_upper) : upper_basic;
        const float curr_lower = (last_close >= prev_lower) ? fmaxf(lower_basic, prev_lower) : lower_basic;

        float outv, changed = 0.0f;
        if (upper_state) {
            if (c <= curr_upper) { outv = curr_upper; }
            else { outv = curr_lower; changed = 1.0f; upper_state = 0; }
        } else {
            if (c >= curr_lower) { outv = curr_lower; }
            else { outv = curr_upper; changed = 1.0f; upper_state = 1; }
        }

        out_tr[t] = outv;
        out_ch[t] = changed;

        prev_upper = curr_upper;
        prev_lower = curr_lower;
        last_close = c;
    }
}

// --- SuperTrend: many series, one param (time-major, coalesced) ---
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
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // one thread per series
    if (s >= cols) return;

    const int fv   = first_valids[s];
    const int warm = fv + period - 1;

    // Pointer-stride iterators (reduce index math)
    const int stride = cols;
    const float* __restrict__ p_hl    = hl2_tm   + s;
    const float* __restrict__ p_close = close_tm + s;
    const float* __restrict__ p_atr   = atr_tm   + s;
    float* __restrict__ p_out_tr = out_trend_tm   + s;
    float* __restrict__ p_out_ch = out_changed_tm + s;

    // Warmup: write NaNs up to warm
    int t = 0;
    for (; t < rows && t < warm; ++t) {
        p_out_tr[ t*stride ] = qnan_f32();
        p_out_ch[ t*stride ] = qnan_f32();
    }
    if (t >= rows) return;

    // Seed at warm
    const float hl_w    = p_hl   [ t*stride ];
    const float atr_w   = p_atr  [ t*stride ];
    const float close_w = p_close[ t*stride ];

    float prev_upper = fmaf(factor,  atr_w, hl_w);
    float prev_lower = fmaf(-factor, atr_w, hl_w);
    float last_close = close_w;
    int   upper_state = (last_close <= prev_upper);

    p_out_tr[ t*stride ] = upper_state ? prev_upper : prev_lower;
    p_out_ch[ t*stride ] = 0.0f;

    // Scan forward
    for (++t; t < rows; ++t) {
        const float hl = p_hl   [ t*stride ];
        const float a  = p_atr  [ t*stride ];
        const float c  = p_close[ t*stride ];

        const float upper_basic = fmaf(factor,  a, hl);
        const float lower_basic = fmaf(-factor, a, hl);

        const float curr_upper = (last_close <= prev_upper) ? fminf(upper_basic, prev_upper) : upper_basic;
        const float curr_lower = (last_close >= prev_lower) ? fmaxf(lower_basic, prev_lower) : lower_basic;

        float outv, changed = 0.0f;
        if (upper_state) {
            if (c <= curr_upper) { outv = curr_upper; }
            else { outv = curr_lower; changed = 1.0f; upper_state = 0; }
        } else {
            if (c >= curr_lower) { outv = curr_lower; }
            else { outv = curr_upper; changed = 1.0f; upper_state = 1; }
        }

        p_out_tr[ t*stride ] = outv;
        p_out_ch[ t*stride ] = changed;

        prev_upper = curr_upper;
        prev_lower = curr_lower;
        last_close = c;
    }
}

