

















#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


__device__ __forceinline__ float qnan_f32() {
    return __int_as_float(0x7FC00000u);
}


struct f2 { float hi, lo; };

__device__ __forceinline__ f2 make_f2(float hi, float lo) {
    
    float s = hi + lo;
    float z = s - hi;
    float e = (hi - (s - z)) + (lo - z);
    f2 r; r.hi = s; r.lo = e; return r;
}


__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}


__device__ __forceinline__ void two_prod_fma(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p); 
}


__device__ __forceinline__ f2 f2_add(const f2 &x, const f2 &y) {
    float s, e;
    two_sum(x.hi, y.hi, s, e);
    e += (x.lo + y.lo);
    return make_f2(s, e);
}


__device__ __forceinline__ f2 f2_mul_scalar(const f2 &x, float b) {
    float p1, e1; two_prod_fma(x.hi, b, p1, e1);
    float p2 = x.lo * b;
    float s, e; two_sum(p1, p2, s, e);
    e += e1;
    return make_f2(s, e);
}



__device__ __forceinline__ f2 ema_update_f2(float a, float x, float b, const f2 &y) {
    
    float p1, pe1; two_prod_fma(a, x, p1, pe1);
    float p2, pe2; two_prod_fma(b, y.hi, p2, pe2);
    float s, e; two_sum(p1, p2, s, e);
    e += (pe1 + pe2 + y.lo * b);
    return make_f2(s, e);
}



__device__ __forceinline__ float warp_broadcast_load(const float* __restrict__ prices, int idx) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & 31;
    float v = 0.0f;
    if (lane == 0) {
        
        v = prices[idx];
    }
    v = __shfl_sync(mask, v, 0);
    return v;
}



template <bool UseWarpBroadcast>
__device__ __forceinline__
void range_filter_scan_one_combo(
    int combo,
    const float* __restrict__ prices,     
    const float* __restrict__ range_sizes,
    const int*   __restrict__ range_periods,
    const int*   __restrict__ smooth_flags,
    const int*   __restrict__ smooth_periods,
    int series_len,
    int first_valid,
    
    float* __restrict__ filter_out,
    float* __restrict__ high_out,
    float* __restrict__ low_out
){
    if (combo < 0) return;
    const float rs_f = range_sizes[combo];
    const int   rp   = range_periods[combo];
    const int   sflag= smooth_flags[combo];
    const int   sp   = smooth_periods[combo];
    if (series_len <= 0 || rp <= 0) return;

    float* __restrict__ f_row = filter_out + combo * series_len;
    float* __restrict__ h_row = high_out   + combo * series_len;
    float* __restrict__ l_row = low_out    + combo * series_len;

    
    const int warm_extra = sflag ? (rp > sp ? rp : sp) : rp;
    const int warm_end   = first_valid + warm_extra;
    const int capped_warm_end = (warm_end < series_len ? warm_end : series_len);

    
    const float qnan = qnan_f32();
    for (int i = 0; i < capped_warm_end; ++i) {
        f_row[i] = qnan; h_row[i] = qnan; l_row[i] = qnan;
    }
    if (first_valid >= series_len - 1) return;

    
    float prev_filter = prices[first_valid];
    float prev_price  = prev_filter;

    bool ac_initialized = false;
    f2   ac_ema = {0.f, 0.f};

    bool range_initialized = false;
    f2   range_ema = {0.f, 0.f};

    const float alpha_ac = 2.0f / (float(rp) + 1.0f);
    const float one_minus_alpha_ac = 1.0f - alpha_ac;
    const float alpha_range = sflag ? (2.0f / (float(sp) + 1.0f)) : 0.0f;
    const float one_minus_alpha_range = 1.0f - alpha_range;

    
    
    for (int t = first_valid + 1; t < series_len; ++t) {
        float price = UseWarpBroadcast ? warp_broadcast_load(prices, t) : prices[t];
        const float d = price - prev_price;
        const float abs_change = fabsf(d);

        if (!isnan(abs_change)) {
            if (!ac_initialized) {
                ac_ema = make_f2(abs_change, 0.0f);
                ac_initialized = true;
            } else {
                ac_ema = ema_update_f2(alpha_ac, abs_change, one_minus_alpha_ac, ac_ema);
            }
        }
        if (!ac_initialized) {
            prev_price = price;
            continue;
        }

        
        float range_unsmoothed = fmaf(ac_ema.hi, rs_f, ac_ema.lo * rs_f);

        float range_cur;
        if (sflag) {
            if (!range_initialized) {
                range_ema = make_f2(range_unsmoothed, 0.0f);
                range_initialized = true;
            } else {
                range_ema = ema_update_f2(alpha_range, range_unsmoothed, one_minus_alpha_range, range_ema);
            }
            range_cur = range_ema.hi + range_ema.lo;
        } else {
            range_cur = range_unsmoothed;
        }

        const float min_b = price - range_cur;
        const float max_b = price + range_cur;
        
        float current = fminf(fmaxf(prev_filter, min_b), max_b);

        if (t >= capped_warm_end) {
            f_row[t] = current;
            h_row[t] = current + range_cur;
            l_row[t] = current - range_cur;
        }

        prev_filter = current;
        prev_price  = price;
    }
}





extern "C" __global__
void range_filter_batch_f32(const float* __restrict__ prices,
                            const float* __restrict__ range_sizes,
                            const int*   __restrict__ range_periods,
                            const int*   __restrict__ smooth_flags,
                            const int*   __restrict__ smooth_periods,
                            int series_len,
                            int n_combos,
                            int first_valid,
                            float* __restrict__ filter_out,
                            float* __restrict__ high_out,
                            float* __restrict__ low_out) {

    
    
    
    if (gridDim.y > 1) {
        const int combo = blockIdx.y;
        if (combo >= n_combos || threadIdx.x != 0) return;
        range_filter_scan_one_combo<false>(combo, prices, range_sizes, range_periods, smooth_flags, smooth_periods,
                                           series_len, first_valid, filter_out, high_out, low_out);
        return;
    }

    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int combo = tid; combo < n_combos; combo += stride) {
        range_filter_scan_one_combo<true>(combo, prices, range_sizes, range_periods, smooth_flags, smooth_periods,
                                          series_len, first_valid, filter_out, high_out, low_out);
    }
}





extern "C" __global__
void range_filter_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                            float range_size_f,
                                            int   range_period,
                                            int   smooth_flag,
                                            int   smooth_period,
                                            int   num_series,
                                            int   series_len,
                                            const int* __restrict__ first_valids,
                                            float* __restrict__ filter_tm,
                                            float* __restrict__ high_tm,
                                            float* __restrict__ low_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || threadIdx.x != 0) return;
    if (series_len <= 0 || range_period <= 0) return;

    const int first_valid = first_valids[series];
    const int warm_extra = smooth_flag ? (range_period > smooth_period ? range_period : smooth_period) : range_period;
    const int warm_end   = first_valid + warm_extra;
    const int capped_warm_end = (warm_end < series_len ? warm_end : series_len);

    const float qnan = qnan_f32();
    for (int t = 0; t < capped_warm_end; ++t) {
        const int idx = t * num_series + series;
        filter_tm[idx] = qnan; high_tm[idx] = qnan; low_tm[idx] = qnan;
    }
    if (first_valid >= series_len - 1) return;

    float prev_filter = prices_tm[first_valid * num_series + series];
    float prev_price  = prev_filter;

    bool ac_initialized = false;
    f2   ac_ema = {0.f, 0.f};

    bool range_initialized = false;
    f2   range_ema = {0.f, 0.f};

    const float alpha_ac = 2.0f / (float(range_period) + 1.0f);
    const float one_minus_alpha_ac = 1.0f - alpha_ac;
    const float alpha_range = smooth_flag ? (2.0f / (float(smooth_period) + 1.0f)) : 0.0f;
    const float one_minus_alpha_range = 1.0f - alpha_range;

    for (int t = first_valid + 1; t < series_len; ++t) {
        const int idx = t * num_series + series;
        const float price = prices_tm[idx];
        const float d = price - prev_price;
        const float abs_change = fabsf(d);

        if (!isnan(abs_change)) {
            if (!ac_initialized) {
                ac_ema = make_f2(abs_change, 0.0f);
                ac_initialized = true;
            } else {
                ac_ema = ema_update_f2(alpha_ac, abs_change, one_minus_alpha_ac, ac_ema);
            }
        }
        if (!ac_initialized) {
            prev_price = price; continue;
        }

        float range_unsmoothed = fmaf(ac_ema.hi, range_size_f, ac_ema.lo * range_size_f);

        float range_cur;
        if (smooth_flag) {
            if (!range_initialized) {
                range_ema = make_f2(range_unsmoothed, 0.0f);
                range_initialized = true;
            } else {
                range_ema = ema_update_f2(alpha_range, range_unsmoothed, one_minus_alpha_range, range_ema);
            }
            range_cur = range_ema.hi + range_ema.lo;
        } else {
            range_cur = range_unsmoothed;
        }

        const float min_b = price - range_cur;
        const float max_b = price + range_cur;
        const float current = fminf(fmaxf(prev_filter, min_b), max_b);

        if (t >= capped_warm_end) {
            filter_tm[idx] = current;
            high_tm[idx]   = current + range_cur;
            low_tm[idx]    = current - range_cur;
        }

        prev_filter = current;
        prev_price  = price;
    }
}
