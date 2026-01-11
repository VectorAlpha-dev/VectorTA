


















#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float fmaf_safe(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}











extern "C" __global__ void vlma_batch_sma_std_prefix_f32(
    const float*  __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    const int*    __restrict__ min_periods,
    const int*    __restrict__ max_periods,
    int len,
    int first_valid,
    int n_combos,
    float* __restrict__ out
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || len <= 0) return;

    const int min_p = max(1, min_periods[combo]);
    const int max_p = max(min_p, max_periods[combo]);
    if (first_valid < 0 || first_valid >= len) return;

    const int base = combo * len;

    
    for (int i = threadIdx.x; i < first_valid; i += blockDim.x) {
        out[base + i] = NAN;
    }

    if (threadIdx.x != 0) return; 

    const float x0 = data[first_valid];
    out[base + first_valid] = x0;

    const int warm_end = min(len, first_valid + max_p - 1);
    int last_p = max_p;
    float last_val = x0;

    
    for (int i = first_valid + 1; i < warm_end; ++i) {
        const float x = data[i];
        if (isfinite(x)) {
            const float sc = 2.0f / (float)(last_p + 1);
            last_val = fmaf_safe(x - last_val, sc, last_val);
        }
        out[base + i] = NAN; 
    }

    if (warm_end >= len) return;

    
    for (int i = warm_end; i < len; ++i) {
        const float x = data[i];
        if (!isfinite(x)) {
            out[base + i] = NAN;
            continue;
        }

        
        const int t1 = i + 1;
        const int t0 = max(0, t1 - max_p);
        const int nan_cnt = prefix_nan[t1] - prefix_nan[t0];

        float sc = 2.0f / (float)(last_p + 1); 
        if (nan_cnt == 0) {
            const double sum  = prefix_sum[t1]    - prefix_sum[t0];
            const double sum2 = prefix_sum_sq[t1] - prefix_sum_sq[t0];
            const double inv  = 1.0 / (double)max_p;
            const double m    = sum * inv;
            double var        = (sum2 * inv) - m * m;
            if (var < 0.0) var = 0.0;
            const double dv   = sqrt(var);

            
            const double d175 = dv * 1.75;
            const double d025 = dv * 0.25;
            const double a = m - d175;
            const double b = m - d025;
            const double c = m + d025;
            const double d = m + d175;

            const int inc_fast = (x < a) || (x > d);
            const int inc_slow = (x >= b) && (x <= c);
            const int delta = inc_slow - inc_fast; 
            int p_next = last_p + delta;
            if (p_next < min_p) p_next = min_p;
            if (p_next > max_p) p_next = max_p;
            sc = 2.0f / (float)(p_next + 1);
            last_p = p_next;
        }

        last_val = fmaf_safe(x - last_val, sc, last_val);
        out[base + i] = last_val;
    }
}








extern "C" __global__ void vlma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int min_period,
    int max_period,
    int cols,           
    int rows,           
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x; 
    if (s >= cols || rows <= 0) return;

    int min_p = max(1, min_period);
    int max_p = max(min_p, max_period);

    int first_valid = first_valids[s];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= rows) return;

    
    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * cols + s] = NAN;
    }
    if (threadIdx.x != 0) return;

    
    const float x0 = prices_tm[first_valid * cols + s];
    out_tm[first_valid * cols + s] = x0;

    const int warm_end = min(rows, first_valid + max_p - 1);
    int last_p = max_p;
    float last_val = x0;

    
    for (int t = first_valid + 1; t < warm_end; ++t) {
        const float x = prices_tm[t * cols + s];
        if (isfinite(x)) {
            const float sc = 2.0f / (float)(last_p + 1);
            last_val = fmaf_safe(x - last_val, sc, last_val);
        }
        out_tm[t * cols + s] = NAN; 
    }
    if (warm_end >= rows) return;

    
    double sum = 0.0, sumsq = 0.0;
    int nan_cnt = 0;
    for (int k = 0; k < max_p; ++k) {
        const float v = prices_tm[(first_valid + k) * cols + s];
        if (isfinite(v)) {
            const double dv = (double)v;
            sum += dv;
            sumsq += dv * dv;
        } else {
            ++nan_cnt;
        }
    }
    const double inv_n = 1.0 / (double)max_p;

    
    for (int t = warm_end; t < rows; ++t) {
        const float x = prices_tm[t * cols + s];
        if (!isfinite(x)) {
            out_tm[t * cols + s] = NAN;
        } else {
            float sc = 2.0f / (float)(last_p + 1);
            if (nan_cnt == 0) {
                const double m  = sum * inv_n;
                double var      = (sumsq * inv_n) - m * m;
                if (var < 0.0) var = 0.0;
                const double dv = sqrt(var);

                const double d175 = dv * 1.75;
                const double d025 = dv * 0.25;
                const double a = m - d175;
                const double b = m - d025;
                const double c = m + d025;
                const double d = m + d175;

                const int inc_fast = (x < a) || (x > d);
                const int inc_slow = (x >= b) && (x <= c);
                int p_next = last_p + (inc_slow - inc_fast);
                if (p_next < min_p) p_next = min_p;
                if (p_next > max_p) p_next = max_p;
                sc = 2.0f / (float)(p_next + 1);
                last_p = p_next;
            }

            last_val = fmaf_safe(x - last_val, sc, last_val);
            out_tm[t * cols + s] = last_val;
        }

        
        if (t + 1 < rows) {
            const int out_idx = t + 1 - max_p; 
            const float leaving = prices_tm[out_idx * cols + s];
            if (isfinite(leaving)) {
                const double dl = (double)leaving;
                sum   -= dl;
                sumsq -= dl * dl;
            } else {
                nan_cnt = max(0, nan_cnt - 1);
            }
            const float enter = prices_tm[(t + 1) * cols + s];
            if (isfinite(enter)) {
                const double de = (double)enter;
                sum   += de;
                sumsq += de * de;
            } else {
                ++nan_cnt;
            }
        }
    }
}

