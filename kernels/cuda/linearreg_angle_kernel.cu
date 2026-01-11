











#include <cuda_runtime.h>
#include <math.h>



#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef LRA_NAN_F
#define LRA_NAN_F (__int_as_float(0x7fffffff))
#endif



static __device__ __forceinline__ int tm_idx(int row, int num_series, int series) {
    return row * num_series + series;
}

static __device__ __forceinline__ float kRad2Deg() {
    return 57.2957795130823208767981548141051703f; 
}


struct df32 {
    float hi;
    float lo;
};

static __device__ __forceinline__ df32 df32_make(float x) {
    df32 r; r.hi = x; r.lo = 0.0f; return r;
}


static __device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}


static __device__ __forceinline__ df32 df32_add_f(df32 a, float b) {
    float s, e; two_sum(a.hi, b, s, e);
    e += a.lo;
    float s2, e2; two_sum(s, e, s2, e2);
    return {s2, e2};
}


static __device__ __forceinline__ df32 df32_sub_f(df32 a, float b) {
    return df32_add_f(a, -b);
}


static __device__ __forceinline__ df32 df32_add(df32 a, df32 b) {
    float s, e; two_sum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    float s2, e2; two_sum(s, e, s2, e2);
    return {s2, e2};
}


static __device__ __forceinline__ df32 df32_sub(df32 a, df32 b) {
    return df32_add(a, { -b.hi, -b.lo });
}


static __device__ __forceinline__ df32 df32_add_prod(df32 acc, float a, float b) {
    float p = a * b;
    float err = fmaf(a, b, -p);      
    acc = df32_add_f(acc, p);
    acc = df32_add_f(acc, err);
    return acc;
}


static __device__ __forceinline__ df32 df32_sub_prod(df32 acc, float a, float b) {
    float p = a * b;
    float err = fmaf(a, b, -p);
    acc = df32_sub_f(acc, p);
    acc = df32_sub_f(acc, err);
    return acc;
}


static __device__ __forceinline__ df32 df32_mul_scalar(df32 a, float s) {
    float p = a.hi * s;
    float err = fmaf(a.hi, s, -p);
    err += a.lo * s;
    float s2, e2; two_sum(p, err, s2, e2);
    return {s2, e2};
}


static __device__ __forceinline__ df32 df32_from_float2(const float2 v) {
    df32 r; r.hi = v.x; r.lo = v.y; return r;
}
static __device__ __forceinline__ float2 float2_from_df32(const df32 v) {
    return make_float2(v.hi, v.lo);
}


static __device__ __forceinline__ float df32_to_float(df32 a) {
    return a.hi + a.lo;
}















extern "C" __global__ void linearreg_angle_batch_f32(
    const float*   __restrict__ prices,
    const float2*  __restrict__ prefix_sum2,
    const float2*  __restrict__ prefix_kd2,
    const int*     __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*     __restrict__ periods,
    const float*   __restrict__ sum_x,
    const float*   __restrict__ inv_div,
    int n_combos,
    float*         __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period < 2 || period > len) return;

    const int warm = first_valid + period - 1;
    const float sx_f   = sum_x[combo];
    const float invd_f = inv_div[combo];
    const float rad2deg = kRad2Deg();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int row_off = combo * len;

    while (t < len) {
        float outv = LRA_NAN_F;
        if (t >= warm) {
            const int start = t + 1 - period;
            const int nan_cnt = prefix_nan[t + 1] - prefix_nan[start];
            if (nan_cnt == 0) {
                df32 sum_y  = df32_sub(df32_from_float2(prefix_sum2[t + 1]),
                                       df32_from_float2(prefix_sum2[start]));
                df32 sum_kd = df32_sub(df32_from_float2(prefix_kd2[t + 1]),
                                       df32_from_float2(prefix_kd2[start]));

                
                df32 sum_xy = df32_sub(df32_mul_scalar(sum_y, (float)t), sum_kd);
                
                df32 num = df32_sub(df32_mul_scalar(sum_xy, (float)period),
                                    df32_mul_scalar(sum_y, sx_f));
                const float slope = df32_to_float(num) * invd_f;
                outv = atanf(slope) * rad2deg;
            }
        }
        out[row_off + t] = outv;
        t += stride;
    }
}



extern "C" __global__ void linearreg_angle_many_series_one_param_f32(
    const float* __restrict__ prices_tm, 
    const int*   __restrict__ first_valids, 
    int cols,
    int rows,
    int period,
    float sum_x_f,
    float inv_div_f,
    float* __restrict__ out_tm)
{
    const int stride = blockDim.x * gridDim.x;
    const float p_f = (float)period;
    const float sx_f = sum_x_f;
    const float invd_f = inv_div_f;
    const float rad2deg = kRad2Deg();

    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += stride) {
        if (period < 2 || period > rows) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }
        const int fv = first_valids[s];
        if (fv < 0 || fv >= rows) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }
        const int tail = rows - fv;
        if (tail < period) {
            for (int r = 0; r < rows; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;
            continue;
        }

        const int warm = fv + period - 1;
        for (int r = 0; r < warm; ++r) out_tm[tm_idx(r, cols, s)] = LRA_NAN_F;

        
        df32 y_sum = df32_make(0.0f);
        df32 sum_kd = df32_make(0.0f);
        int nan_count = 0;

        for (int k = 0; k < period; ++k) {
            const int r0 = warm - period + 1 + k;
            const float v = prices_tm[tm_idx(r0, cols, s)];
            if (isnan(v)) {
                nan_count++;
            } else {
                y_sum  = df32_add_f(y_sum, v);
                sum_kd = df32_add_prod(sum_kd, (float)r0, v);
            }
        }

        
        {
            float outv = LRA_NAN_F;
            if (nan_count == 0) {
                df32 sum_xy = df32_sub(df32_mul_scalar(y_sum, (float)warm), sum_kd);
                df32 num = df32_sub(df32_mul_scalar(sum_xy, p_f),
                                    df32_mul_scalar(y_sum, sx_f));
                const float slope = df32_to_float(num) * invd_f;
                outv = atanf(slope) * rad2deg;
            }
            out_tm[tm_idx(warm, cols, s)] = outv;

            
            if (nan_count == 0) {
                const int leave0_idx = warm - period + 1;
                const float leave0 = prices_tm[tm_idx(leave0_idx, cols, s)];
                y_sum  = df32_sub_f(y_sum, leave0);
                sum_kd = df32_sub_prod(sum_kd, (float)leave0_idx, leave0);
            }
        }

        
        float next_enter = (warm + 1 < rows) ? prices_tm[tm_idx(warm + 1, cols, s)] : LRA_NAN_F;

        
        for (int r = warm + 1; r < rows; ++r) {
            const float enter = next_enter;
            if (r + 1 < rows) next_enter = prices_tm[tm_idx(r + 1, cols, s)];
            const float leave = prices_tm[tm_idx(r - period + 1, cols, s)];

            const bool enter_nan = isnan(enter);
            const bool leave_nan = isnan(leave);
            const int prev_nan_count = nan_count;
            if (enter_nan) nan_count++;
            if (leave_nan) nan_count--;

            float outv = LRA_NAN_F;

            if (nan_count == 0) {
                if (prev_nan_count == 0) {
                    
                    y_sum  = df32_add_f(y_sum, enter);
                    sum_kd = df32_add_prod(sum_kd, (float)r, enter);

                    df32 sum_xy = df32_sub(df32_mul_scalar(y_sum, (float)r), sum_kd);
                    df32 num = df32_sub(df32_mul_scalar(sum_xy, p_f),
                                        df32_mul_scalar(y_sum, sx_f));
                    const double slope_d = (double)df32_to_float(num) * (double)invd_f;
                    outv = atanf((float)slope_d) * rad2deg;

                    
                    y_sum  = df32_sub_f(y_sum, leave);
                    sum_kd = df32_sub_prod(sum_kd, (float)(r - period + 1), leave);
                } else {
                    
                    y_sum  = df32_make(0.0f);
                    sum_kd = df32_make(0.0f);
                    for (int k = 0; k < period; ++k) {
                        const int r0 = r - period + 1 + k;
                        const float v = prices_tm[tm_idx(r0, cols, s)];
                        
                        y_sum  = df32_add_f(y_sum, v);
                        sum_kd = df32_add_prod(sum_kd, (float)r0, v);
                    }
                    df32 sum_xy = df32_sub(df32_mul_scalar(y_sum, (float)r), sum_kd);
                    df32 num = df32_sub(df32_mul_scalar(sum_xy, p_f),
                                        df32_mul_scalar(y_sum, sx_f));
                    const double slope_d = (double)df32_to_float(num) * (double)invd_f;
                    outv = atanf((float)slope_d) * rad2deg;

                    
                    y_sum  = df32_sub_f(y_sum, leave);
                    sum_kd = df32_sub_prod(sum_kd, (float)(r - period + 1), leave);
                }
            } else {
                
                outv = LRA_NAN_F;
            }

            out_tm[tm_idx(r, cols, s)] = outv;
        }
    }
}
