// CUDA kernels for Buff Averages batch computation using prefix sums.
//
// Sliding-sum MA category: we use precomputed prefix sums for
//   pv[t] = sum_{i< t}(price[i] * volume[i] if both are finite else 0)
//   vv[t] = sum_{i< t}(volume[i]            if finite          else 0)
// which yield O(1) per-output window evaluation:
//   sum_pv = pv[t+1] - pv[t+1-period]
//   sum_vv = vv[t+1] - vv[t+1-period]
//   out    = (sum_vv != 0) ? sum_pv / sum_vv : 0
//
// Kernels provided:
// - buff_averages_batch_prefix_f32                      : plain 1D launch (baseline)
// - buff_averages_batch_prefix_tiled_f32_*              : tiled 1D variants (tile=128/256)
// - buff_averages_many_series_one_param_f32             : many-series × one-param (time-major), 1D
// - buff_averages_many_series_one_param_tiled2d_tx128_ty2/ty4 : many-series × one-param, 2D tiled
//
// Notes:
// - Outputs at t < warm (warm = first_valid + slow_period - 1) are set to NaN.
// - Kernel grid.y indexes the (fast, slow) combo; grid.x covers time.
// - This file intentionally avoids shared memory: O(1) arithmetic dominates,
//   and coalesced global loads suffice. Tiled variants exist for consistency
//   with the ALMA integration and to control launch geometry deterministically.

#include <cuda_runtime.h>
#include <math.h>

// FP32 two-float expansion helpers (hi+lo)
struct f2 { float hi, lo; };

__device__ __forceinline__ f2 two_sum(float a, float b) {
    float s = a + b;
    float bp = s - a;
    float e = (a - (s - bp)) + (b - bp);
    f2 r; r.hi = s; r.lo = e; return r;
}

__device__ __forceinline__ f2 add_f2(f2 x, f2 y) {
    f2 s = two_sum(x.hi, y.hi);
    float t = x.lo + y.lo;
    f2 r = two_sum(s.hi, s.lo + t);
    return r;
}

__device__ __forceinline__ f2 sub_f2(f2 x, f2 y) {
    f2 s = two_sum(x.hi, -y.hi);
    float t = x.lo - y.lo;
    f2 r = two_sum(s.hi, s.lo + t);
    return r;
}

__device__ __forceinline__ float div_f2(f2 n, f2 d) {
    // Same semantics as existing code: 0 on exact-zero denominator
    if (d.hi == 0.0f && d.lo == 0.0f) return 0.0f;
    // FP32 reciprocal + one Newton refinement
    float rcp = __frcp_rn(d.hi);
    rcp = fmaf(rcp, (2.0f - d.hi * rcp), 0.0f);
    // correction using FMAs and the expansion tails
    float q0 = n.hi * rcp;
    float r  = fmaf(-q0, d.hi, n.hi);
    r        = fmaf(-q0, d.lo, r);
    r       += n.lo;
    float q1 = r * rcp;
    return q0 + q1;
}

extern "C" __global__ void buff_averages_batch_prefix_f32(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int fast_period = fast_periods[combo];
    const int slow_period = slow_periods[combo];
    if (fast_period <= 0 || slow_period <= 0) {
        return;
    }

    const int warm = first_valid + slow_period - 1;
    const int row_offset = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float fast_val = nan_f;
        float slow_val = nan_f;

        if (t >= warm) {
            int fast_start = t + 1 - fast_period;
            if (fast_start < 0) {
                fast_start = 0;
            }
            int slow_start = t + 1 - slow_period;
            if (slow_start < 0) {
                slow_start = 0;
            }

            const float slow_num = prefix_pv[t + 1] - prefix_pv[slow_start];
            const float slow_den = prefix_vv[t + 1] - prefix_vv[slow_start];
            if (slow_den != 0.0f) {
                float rcp = __frcp_rn(slow_den);
                rcp = fmaf(rcp, (2.0f - slow_den * rcp), 0.0f);
                slow_val = slow_num * rcp;
            } else {
                slow_val = 0.0f;
            }

            const float fast_num = prefix_pv[t + 1] - prefix_pv[fast_start];
            const float fast_den = prefix_vv[t + 1] - prefix_vv[fast_start];
            if (fast_den != 0.0f) {
                float rcp = __frcp_rn(fast_den);
                rcp = fmaf(rcp, (2.0f - fast_den * rcp), 0.0f);
                fast_val = fast_num * rcp;
            } else {
                fast_val = 0.0f;
            }
        }

        fast_out[row_offset + t] = fast_val;
        slow_out[row_offset + t] = slow_val;
        t += stride;
    }
}

// Tiled 1D variant: blockDim.x == TILE, grid.x covers ceil(len/TILE).
// Each thread computes at most one output within its tile.
template<int TILE>
__device__ __forceinline__ void buff_averages_batch_prefix_tiled_f32_impl(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int fast_period = fast_periods[combo];
    const int slow_period = slow_periods[combo];
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valid + slow_period - 1;
    const int row_offset = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    const int t0 = blockIdx.x * TILE;
    const int t  = t0 + threadIdx.x;
    if (t >= len) return;

    float fast_val = nan_f;
    float slow_val = nan_f;

    if (t >= warm) {
        int fast_start = t + 1 - fast_period;
        if (fast_start < 0) fast_start = 0;
        int slow_start = t + 1 - slow_period;
        if (slow_start < 0) slow_start = 0;

        const float slow_num = prefix_pv[t + 1] - prefix_pv[slow_start];
        const float slow_den = prefix_vv[t + 1] - prefix_vv[slow_start];
        if (slow_den != 0.0f) {
            float rcp = __frcp_rn(slow_den);
            rcp = fmaf(rcp, (2.0f - slow_den * rcp), 0.0f);
            slow_val = slow_num * rcp;
        } else {
            slow_val = 0.0f;
        }

        const float fast_num = prefix_pv[t + 1] - prefix_pv[fast_start];
        const float fast_den = prefix_vv[t + 1] - prefix_vv[fast_start];
        if (fast_den != 0.0f) {
            float rcp = __frcp_rn(fast_den);
            rcp = fmaf(rcp, (2.0f - fast_den * rcp), 0.0f);
            fast_val = fast_num * rcp;
        } else {
            fast_val = 0.0f;
        }
    }

    fast_out[row_offset + t] = fast_val;
    slow_out[row_offset + t] = slow_val;
}

extern "C" __global__ void buff_averages_batch_prefix_tiled_f32_tile128(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    buff_averages_batch_prefix_tiled_f32_impl<128>(
        prefix_pv, prefix_vv, len, first_valid,
        fast_periods, slow_periods, n_combos, fast_out, slow_out);
}

extern "C" __global__ void buff_averages_batch_prefix_tiled_f32_tile256(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    buff_averages_batch_prefix_tiled_f32_impl<256>(
        prefix_pv, prefix_vv, len, first_valid,
        fast_periods, slow_periods, n_combos, fast_out, slow_out);
}

// ------------------------ Many-series (time-major) -------------------------
// Inputs are time-major matrices of prefix sums with one extra leading row
// (length = rows + 1). Each output is computed as:
//   sum_pv = pv_prefix[(t+1, s)] - pv_prefix[(t+1-fast, s)]
//   sum_vv = vv_prefix[(t+1, s)] - vv_prefix[(t+1-slow, s)]
// with the starts clamped to zero. Outputs for t < warm are set to NaN.

extern "C" __global__ void buff_averages_many_series_one_param_f32(
    const float* __restrict__ pv_prefix_tm,  // (rows+1) x cols
    const float* __restrict__ vv_prefix_tm,  // (rows+1) x cols
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,         // rows x cols
    float* __restrict__ slow_out_tm) {       // rows x cols
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valids[series] + slow_period - 1;
    const int stride = num_series;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + series;
        if (t < warm) {
            fast_out_tm[out_idx] = __int_as_float(0x7fffffff);
            slow_out_tm[out_idx] = __int_as_float(0x7fffffff);
        } else {
            const int t1 = t + 1;
            int fstart = t1 - fast_period;
            if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period;
            if (sstart < 0) sstart = 0;

            const int p_idx = t1 * stride + series;
            const int f_idx = fstart * stride + series;
            const int s_idx = sstart * stride + series;

            const float fast_num = pv_prefix_tm[p_idx] - pv_prefix_tm[f_idx];
            const float fast_den = vv_prefix_tm[p_idx] - vv_prefix_tm[f_idx];
            const float slow_num = pv_prefix_tm[p_idx] - pv_prefix_tm[s_idx];
            const float slow_den = vv_prefix_tm[p_idx] - vv_prefix_tm[s_idx];

            if (fast_den != 0.0f) {
                float rcp = __frcp_rn(fast_den);
                rcp = fmaf(rcp, (2.0f - fast_den * rcp), 0.0f);
                fast_out_tm[out_idx] = fast_num * rcp;
            } else {
                fast_out_tm[out_idx] = 0.0f;
            }
            if (slow_den != 0.0f) {
                float rcp = __frcp_rn(slow_den);
                rcp = fmaf(rcp, (2.0f - slow_den * rcp), 0.0f);
                slow_out_tm[out_idx] = slow_num * rcp;
            } else {
                slow_out_tm[out_idx] = 0.0f;
            }
        }
        t += step;
    }
}

template<int TX, int TY>
__device__ __forceinline__ void buff_averages_many_series_one_param_tiled2d_impl(
    const float* __restrict__ pv_prefix_tm,  // (rows+1) x cols
    const float* __restrict__ vv_prefix_tm,  // (rows+1) x cols
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    const int s = blockIdx.y * TY + threadIdx.y;
    if (s >= num_series) return;
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valids[s] + slow_period - 1;
    const int stride = num_series;

    const int t0 = blockIdx.x * TX;
    const int t = t0 + threadIdx.x;
    if (t >= series_len) return;

    const int out_idx = t * stride + s;
    if (t < warm) {
        fast_out_tm[out_idx] = __int_as_float(0x7fffffff);
        slow_out_tm[out_idx] = __int_as_float(0x7fffffff);
        return;
    }

    const int t1 = t + 1;
    int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
    int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

    const int p_idx = t1 * stride + s;
    const int f_idx = fstart * stride + s;
    const int s_idx = sstart * stride + s;

    const float fast_num = pv_prefix_tm[p_idx] - pv_prefix_tm[f_idx];
    const float fast_den = vv_prefix_tm[p_idx] - vv_prefix_tm[f_idx];
    const float slow_num = pv_prefix_tm[p_idx] - pv_prefix_tm[s_idx];
    const float slow_den = vv_prefix_tm[p_idx] - vv_prefix_tm[s_idx];

    if (fast_den != 0.0f) {
        float rcp = __frcp_rn(fast_den);
        rcp = fmaf(rcp, (2.0f - fast_den * rcp), 0.0f);
        fast_out_tm[out_idx] = fast_num * rcp;
    } else {
        fast_out_tm[out_idx] = 0.0f;
    }
    if (slow_den != 0.0f) {
        float rcp = __frcp_rn(slow_den);
        rcp = fmaf(rcp, (2.0f - slow_den * rcp), 0.0f);
        slow_out_tm[out_idx] = slow_num * rcp;
    } else {
        slow_out_tm[out_idx] = 0.0f;
    }
}

extern "C" __global__ void buff_averages_many_series_one_param_tiled2d_f32_tx128_ty2(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    buff_averages_many_series_one_param_tiled2d_impl<128, 2>(
        pv_prefix_tm, vv_prefix_tm, fast_period, slow_period,
        num_series, series_len, first_valids, fast_out_tm, slow_out_tm);
}

extern "C" __global__ void buff_averages_many_series_one_param_tiled2d_f32_tx128_ty4(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    buff_averages_many_series_one_param_tiled2d_impl<128, 4>(
        pv_prefix_tm, vv_prefix_tm, fast_period, slow_period,
        num_series, series_len, first_valids, fast_out_tm, slow_out_tm);
}

// ------------------------ Expansion-based variants -------------------------
// Batch (one series, many (fast, slow) combos); expects four prefix arrays.
extern "C" __global__ void buff_averages_batch_prefix_exp2_f32(
    const float* __restrict__ pv_hi,
    const float* __restrict__ pv_lo,
    const float* __restrict__ vv_hi,
    const float* __restrict__ vv_lo,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int fast_period = fast_periods[combo];
    const int slow_period = slow_periods[combo];
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valid + slow_period - 1;
    const int row_offset = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float fast_val = nan_f;
        float slow_val = nan_f;
        if (t >= warm) {
            const int t1 = t + 1;
            int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

            f2 pv_t  = { pv_hi[t1],  pv_lo[t1] };
            f2 pv_f0 = { pv_hi[fstart], pv_lo[fstart] };
            f2 pv_s0 = { pv_hi[sstart], pv_lo[sstart] };
            f2 vv_t  = { vv_hi[t1],  vv_lo[t1] };
            f2 vv_f0 = { vv_hi[fstart], vv_lo[fstart] };
            f2 vv_s0 = { vv_hi[sstart], vv_lo[sstart] };

            f2 fast_num = sub_f2(pv_t, pv_f0);
            f2 fast_den = sub_f2(vv_t, vv_f0);
            f2 slow_num = sub_f2(pv_t, pv_s0);
            f2 slow_den = sub_f2(vv_t, vv_s0);

            fast_val = div_f2(fast_num, fast_den);
            slow_val = div_f2(slow_num, slow_den);
        }
        fast_out[row_offset + t] = fast_val;
        slow_out[row_offset + t] = slow_val;
        t += stride;
    }
}

// Many-series (time-major) expansion variant; prefixes are (rows+1) x cols.
extern "C" __global__ void buff_averages_many_series_one_param_exp2_f32(
    const float* __restrict__ pv_hi_tm,
    const float* __restrict__ pv_lo_tm,
    const float* __restrict__ vv_hi_tm,
    const float* __restrict__ vv_lo_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    const int s = blockIdx.y;
    if (s >= num_series) return;
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valids[s] + slow_period - 1;
    const int stride = num_series;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + s;
        if (t < warm) {
            fast_out_tm[out_idx] = __int_as_float(0x7fffffff);
            slow_out_tm[out_idx] = __int_as_float(0x7fffffff);
        } else {
            const int t1 = t + 1;
            int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

            const int p = t1 * stride + s;
            const int f = fstart * stride + s;
            const int q = sstart * stride + s;

            f2 pv_t  = { pv_hi_tm[p], pv_lo_tm[p] };
            f2 pv_f0 = { pv_hi_tm[f], pv_lo_tm[f] };
            f2 pv_s0 = { pv_hi_tm[q], pv_lo_tm[q] };
            f2 vv_t  = { vv_hi_tm[p], vv_lo_tm[p] };
            f2 vv_f0 = { vv_hi_tm[f], vv_lo_tm[f] };
            f2 vv_s0 = { vv_hi_tm[q], vv_lo_tm[q] };

            f2 fast_num = sub_f2(pv_t, pv_f0);
            f2 fast_den = sub_f2(vv_t, vv_f0);
            f2 slow_num = sub_f2(pv_t, pv_s0);
            f2 slow_den = sub_f2(vv_t, vv_s0);

            fast_out_tm[out_idx] = div_f2(fast_num, fast_den);
            slow_out_tm[out_idx] = div_f2(slow_num, slow_den);
        }
        t += step;
    }
}
