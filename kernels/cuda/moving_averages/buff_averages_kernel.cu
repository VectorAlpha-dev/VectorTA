






















#include <cuda_runtime.h>
#include <math.h>



#ifndef BA_ENABLE_L2_PREFETCH
#define BA_ENABLE_L2_PREFETCH 1
#endif

#ifndef BA_EXP2_NR_STEPS
#define BA_EXP2_NR_STEPS 1
#endif

__device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}


__device__ __forceinline__ float ratio_from_prefix(float pv_t, float pv_s,
                                                   float vv_t, float vv_s) {
    float den = vv_t - vv_s;
    if (den != 0.0f) {
        float rcp = __frcp_rn(den);
        rcp = fmaf(rcp, (2.0f - den * rcp), 0.0f);
        return (pv_t - pv_s) * rcp;
    }
    return 0.0f;
}


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

    if (d.hi == 0.0f && d.lo == 0.0f) return 0.0f;

    float rcp = __frcp_rn(d.hi);
#if BA_EXP2_NR_STEPS >= 1
    rcp = fmaf(rcp, (2.0f - d.hi * rcp), 0.0f);
#endif

    float q0 = n.hi * rcp;
    float r  = fmaf(-q0, d.hi, n.hi);
    r        = fmaf(-q0, d.lo, r);
    r       += n.lo;
    float q1 = r * rcp;
#if BA_EXP2_NR_STEPS >= 2

    float r2 = fmaf(-(q0 + q1), d.hi, n.hi);
    r2       = fmaf(-(q0 + q1), d.lo, r2);
    r2      += n.lo;
    q1      += r2 * rcp;
#endif
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
    if (combo >= n_combos) return;

    const int fast_period = fast_periods[combo];
    const int slow_period = slow_periods[combo];
    if (fast_period <= 0 || slow_period <= 0) return;

    const int warm = first_valid + slow_period - 1;
    const int row_offset = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (; t < len; t += stride) {
        float fo = qnan32();
        float so = qnan32();

#if BA_ENABLE_L2_PREFETCH

        int t_pref = t + stride;
        if (t_pref + 1 < len + 1) {
            const float* p0 = prefix_pv + (t_pref + 1);
            const float* p1 = prefix_vv + (t_pref + 1);
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
            asm volatile ("prefetch.global.L2 [%0];" :: "l"(p0));
            asm volatile ("prefetch.global.L2 [%0];" :: "l"(p1));
#endif
        }
#endif

        if (t >= warm) {
            const int t1 = t + 1;
            int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

            const float pv_t = prefix_pv[t1];
            const float vv_t = prefix_vv[t1];
            so = ratio_from_prefix(pv_t, prefix_pv[sstart], vv_t, prefix_vv[sstart]);
            fo = ratio_from_prefix(pv_t, prefix_pv[fstart], vv_t, prefix_vv[fstart]);
        }

        fast_out[row_offset + t] = fo;
        slow_out[row_offset + t] = so;
    }
}



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


extern "C" __global__ void buff_averages_batch_prefix_tiled_f32_tile512(
    const float* __restrict__ prefix_pv,
    const float* __restrict__ prefix_vv,
    int len,
    int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int n_combos,
    float* __restrict__ fast_out,
    float* __restrict__ slow_out) {
    buff_averages_batch_prefix_tiled_f32_impl<512>(
        prefix_pv, prefix_vv, len, first_valid,
        fast_periods, slow_periods, n_combos, fast_out, slow_out);
}








extern "C" __global__ void buff_averages_many_series_one_param_f32(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
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
            fast_out_tm[out_idx] = qnan32();
            slow_out_tm[out_idx] = qnan32();
        } else {
            const int t1 = t + 1;
            int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
            int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

            const int p_idx = t1 * stride + series;
            const int f_idx = fstart * stride + series;
            const int s_idx = sstart * stride + series;

            const float pv_t = pv_prefix_tm[p_idx];
            const float vv_t = vv_prefix_tm[p_idx];

            fast_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[f_idx],
                                                     vv_t, vv_prefix_tm[f_idx]);
            slow_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[s_idx],
                                                     vv_t, vv_prefix_tm[s_idx]);
        }
        t += step;
    }
}

template<int TX, int TY>
__device__ __forceinline__ void buff_averages_many_series_one_param_tiled2d_impl(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
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
        fast_out_tm[out_idx] = qnan32();
        slow_out_tm[out_idx] = qnan32();
        return;
    }

    const int t1 = t + 1;
    int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
    int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

    const int p_idx = t1 * stride + s;
    const int f_idx = fstart * stride + s;
    const int s_idx = sstart * stride + s;

    const float pv_t = pv_prefix_tm[p_idx];
    const float vv_t = vv_prefix_tm[p_idx];

    fast_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[f_idx],
                                             vv_t, vv_prefix_tm[f_idx]);
    slow_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[s_idx],
                                             vv_t, vv_prefix_tm[s_idx]);
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



template<int SX, int TY>
__device__ __forceinline__ void buff_averages_many_series_one_param_tiled2d_swizzled_f32(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {

    if (fast_period <= 0 || slow_period <= 0) return;

    const int s = blockIdx.y * SX + threadIdx.x;
    if (s >= num_series) return;

    const int t = blockIdx.x * TY + threadIdx.y;
    if (t >= series_len) return;

    const int warm = first_valids[s] + slow_period - 1;
    const int stride = num_series;

    const int out_idx = t * stride + s;
    if (t < warm) {
        fast_out_tm[out_idx] = qnan32();
        slow_out_tm[out_idx] = qnan32();
        return;
    }

    const int t1 = t + 1;
    int fstart = t1 - fast_period; if (fstart < 0) fstart = 0;
    int sstart = t1 - slow_period; if (sstart < 0) sstart = 0;

    const int p_idx = t1 * stride + s;
    const int f_idx = fstart * stride + s;
    const int s_idx = sstart * stride + s;

    const float pv_t = pv_prefix_tm[p_idx];
    const float vv_t = vv_prefix_tm[p_idx];

    fast_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[f_idx],
                                             vv_t, vv_prefix_tm[f_idx]);
    slow_out_tm[out_idx] = ratio_from_prefix(pv_t, pv_prefix_tm[s_idx],
                                             vv_t, vv_prefix_tm[s_idx]);
}


extern "C" __global__ void buff_averages_many_series_one_param_tiled2d_f32_sx128_ty1(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    buff_averages_many_series_one_param_tiled2d_swizzled_f32<128,1>(
        pv_prefix_tm, vv_prefix_tm, fast_period, slow_period,
        num_series, series_len, first_valids, fast_out_tm, slow_out_tm);
}

extern "C" __global__ void buff_averages_many_series_one_param_tiled2d_f32_sx128_ty2(
    const float* __restrict__ pv_prefix_tm,
    const float* __restrict__ vv_prefix_tm,
    int fast_period,
    int slow_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ fast_out_tm,
    float* __restrict__ slow_out_tm) {
    buff_averages_many_series_one_param_tiled2d_swizzled_f32<128,2>(
        pv_prefix_tm, vv_prefix_tm, fast_period, slow_period,
        num_series, series_len, first_valids, fast_out_tm, slow_out_tm);
}



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
            fast_out_tm[out_idx] = qnan32();
            slow_out_tm[out_idx] = qnan32();
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
