











#include <cuda_runtime.h>
#include <math.h>
#include <limits.h>


__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }



struct Float2 { float hi, lo; };

__device__ __forceinline__ Float2 f2_make(float a)    { return {a, 0.0f}; }


__device__ __forceinline__ Float2 f2_two_sum(float a, float b) {
    Float2 r;
    float s  = a + b;
    float bp = s - a;
    float e  = (a - (s - bp)) + (b - bp);

    float t  = s + e;
    r.lo     = e - (t - s);
    r.hi     = t;
    return r;
}


__device__ __forceinline__ Float2 f2_add_f(Float2 a, float b) {
    Float2 s = f2_two_sum(a.hi, b);
    s.lo += a.lo;

    float t = s.hi + s.lo;
    s.lo = s.lo - (t - s.hi);
    s.hi = t;
    return s;
}


__device__ __forceinline__ Float2 f2_mul_f(Float2 a, float b) {
    float ph = a.hi * b;
    float pe = fmaf(a.hi, b, -ph) + a.lo * b;
    float t  = ph + pe;
    Float2 r = { t, pe - (t - ph) };
    return r;
}


__device__ __forceinline__ Float2 f2_fma(float a, float b, Float2 c) {
    float ph = fmaf(a, b, c.hi);
    float pe = fmaf(a, b, - (ph - c.hi)) + c.lo;
    float t  = ph + pe;
    Float2 r = { t, pe - (t - ph) };
    return r;
}


__device__ __forceinline__ Float2 f2_div_int(Float2 a, int den) {
    float d      = (float)den;
    float inv_d  = 1.0f / d;

    float q0     = (a.hi + a.lo) * inv_d;

    float r      = (a.hi + a.lo) - q0 * d;
    float q1     = r * inv_d;
    Float2 q     = f2_make(q0 + q1);
    return q;
}


__device__ __forceinline__ float f2_ratio(Float2 num, Float2 den) {
    float N  = num.hi + num.lo;
    float D  = den.hi + den.lo;
    float invD = 1.0f / D;
    float y  = N * invD;

    float corr = (num.lo - y * den.lo) * invD;
    return y + corr;
}


__device__ __forceinline__ int warp_max_i(int v, unsigned mask) {
    const int lane = (int)(threadIdx.x & 31);
    for (int ofs = 16; ofs; ofs >>= 1) {
        const int src_lane = lane + ofs;
        const int other = __shfl_down_sync(mask, v, ofs);
        if (src_lane < 32 && (mask & (1u << src_lane))) v = max(v, other);
    }
    return v;
}
__device__ __forceinline__ int warp_min_i(int v, unsigned mask) {
    const int lane = (int)(threadIdx.x & 31);
    for (int ofs = 16; ofs; ofs >>= 1) {
        const int src_lane = lane + ofs;
        const int other = __shfl_down_sync(mask, v, ofs);
        if (src_lane < 32 && (mask & (1u << src_lane))) v = min(v, other);
    }
    return v;
}






extern "C" __global__ void ppo_batch_ema_manyparams_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    int n_combos,
    float* __restrict__ out)
{
    if (len <= 0 || n_combos <= 0) return;

    const unsigned lane  = threadIdx.x & 31;
    const unsigned warp  = threadIdx.x >> 5;
    const unsigned wpb   = blockDim.x >> 5;
    if (wpb == 0) return;

    const int combos_per_block = (int)(wpb * 32);
    const int base_combo = (int)blockIdx.y * combos_per_block + (int)warp * 32;
    const int combo      = base_combo + (int)lane;



    const unsigned full_mask  = __activemask();
    const bool     valid_lane = (combo < n_combos);
    const unsigned mask       = __ballot_sync(full_mask, valid_lane);
    if (mask == 0u) return;
    if (!valid_lane) return;


    int fast = 0, slow = 0;
    if (valid_lane) {
        fast = fasts[combo];
        slow = slows[combo];
    }

    const bool periods_ok = valid_lane && (fast > 0) && (slow > 0);
    const float nanf = f32_nan();


    int start_idx = 0;
    if (periods_ok) start_idx = first_valid + slow - 1;


    if (periods_ok) {
        const int row_off = combo * len;
        for (int t = 0; t < min(start_idx, len); ++t) {
            out[row_off + t] = nanf;
        }
    }



    int warp_slow_min = periods_ok ? slow : INT_MAX;
    int warp_slow_max = periods_ok ? slow : 0;
    int warp_fast_min = periods_ok ? fast : INT_MAX;

    warp_slow_min = warp_min_i(warp_slow_min, mask);
    warp_slow_max = warp_max_i(warp_slow_max, mask);
    warp_fast_min = warp_min_i(warp_fast_min, mask);


    Float2 slow_sum = f2_make(0.0f);
    Float2 fast_sum = f2_make(0.0f);
    int overlap = 0;
    if (periods_ok) overlap = slow - fast;

    for (int k = 0; k < warp_slow_max && k + first_valid < len; ++k) {
        float v = 0.0f;
        if (lane == 0u) v = data[first_valid + k];
        v = __shfl_sync(mask, v, 0);
        if (periods_ok) {
            if (k < slow) {
                slow_sum = f2_add_f(slow_sum, v);
                if (k >= overlap) fast_sum = f2_add_f(fast_sum, v);
            }
        }
    }


    Float2 fast_ema = f2_make(0.0f), slow_ema = f2_make(0.0f);
    float fa = 0.0f, fb = 0.0f, sa = 0.0f, sb = 0.0f;
    int row_off = 0;
    if (periods_ok) {
        fast_ema = f2_div_int(fast_sum, max(fast, 1));
        slow_ema = f2_div_int(slow_sum, max(slow, 1));
        fa = 2.0f / (float)(fast + 1);
        fb = 1.0f - fa;
        sa = 2.0f / (float)(slow + 1);
        sb = 1.0f - sa;
        row_off = combo * len;
    }



    const int i_begin = first_valid + warp_fast_min;
    const int i_end   = first_valid + warp_slow_max - 1;
    for (int i = i_begin; i <= i_end && i < len; ++i) {
        float x = 0.0f;
        if (lane == 0u) x = data[i];
        x = __shfl_sync(mask, x, 0);
        if (periods_ok) {
            if (i >= first_valid + fast && i <= first_valid + slow - 1) {

                Float2 tmp = f2_mul_f(fast_ema, fb);
                fast_ema = f2_fma(fa, x, tmp);
            }
        }
    }


    if (periods_ok && start_idx < len) {
        float y0 = nanf;
        float den = slow_ema.hi + slow_ema.lo;
        if (isfinite(den) && den != 0.0f) {
            float ratio = f2_ratio(fast_ema, slow_ema);
            y0 = ratio * 100.0f - 100.0f;
        }
        out[row_off + start_idx] = y0;
    }


    int warp_start_min = periods_ok ? start_idx : INT_MAX;
    warp_start_min = warp_min_i(warp_start_min, mask);
    for (int t = warp_start_min + 1; t < len; ++t) {
        float x = 0.0f;
        if (lane == 0u) x = data[t];
        x = __shfl_sync(mask, x, 0);
        if (periods_ok && t > start_idx) {

            fast_ema = f2_fma(fa, x, f2_mul_f(fast_ema, fb));
            slow_ema = f2_fma(sa, x, f2_mul_f(slow_ema, sb));

            float y = nanf;
            float den = slow_ema.hi + slow_ema.lo;
            if (isfinite(den) && den != 0.0f) {
                float ratio = f2_ratio(fast_ema, slow_ema);
                y = ratio * 100.0f - 100.0f;
            }
            out[row_off + t] = y;
        }
    }
}






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
    const int warm_idx = first_valid + max(fast, slow) - 1;
    const int row_off = combo * len;
    const float nanf = f32_nan();

    if (ma_mode == 0) {

        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < len) {
            float y = nanf;
            if (t >= warm_idx) {
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



    const int start_idx = first_valid + slow - 1;
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


    for (int i = first_valid + fast; i <= start_idx; ++i) {
        const double x = (double)data[i];
        fast_ema = fa * x + fb * fast_ema;
    }


    float y0 = nanf;
    if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
        const double ratio = fast_ema / slow_ema;
        y0 = (float)(ratio * 100.0 - 100.0);
    }
    out[row_off + start_idx] = y0;


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
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= cols) return;
    const int fv = max(0, first_valids[s]);
    const int warm_idx = fv + max(fast, slow) - 1;
    const float nanf = f32_nan();

    if (ma_mode == 0) {



        const int tx = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        for (int t = tx; t < rows; t += stride) {
            float y = nanf;
            if (t >= warm_idx) {
                const int wr = (t * cols + s) + 1;
                const int lfast_t = max(t - fast, fv - 1);
                const int lslow_t = max(t - slow, fv - 1);
                const int wl_fast = (lfast_t >= 0) ? (lfast_t * cols + s) + 1 : 0;
                const int wl_slow = (lslow_t >= 0) ? (lslow_t * cols + s) + 1 : 0;
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


    if (!(threadIdx.x == 0)) return;


    const int start_idx = fv + slow - 1;
    for (int t = 0; t < min(start_idx, rows); ++t) {
        out_tm[t * cols + s] = nanf;
    }
    if (start_idx >= rows) return;

    const double fa = 2.0 / (double)(fast + 1);
    const double sa = 2.0 / (double)(slow + 1);
    const double fb = 1.0 - fa;
    const double sb = 1.0 - sa;


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
    const int r = row_start + blockIdx.y;
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
            const double ratio = (double)ff / (double)sf;
            y = (float)(ratio * 100.0 - 100.0);
        }
        out[r * len + t] = y;
    }
}



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
            const double ratio = (double)ff / (double)sf;
            y = (float)(ratio * 100.0 - 100.0);
        }
        out_tm[t * cols + s] = y;
    }
}
