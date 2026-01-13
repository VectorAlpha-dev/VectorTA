






#include <cuda_runtime.h>
#include <math.h>




__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }



__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void two_diff(float a, float b, float &s, float &e) {
    s = a - b;
    float bb = s - a;
    e = (a - (s - bb)) - b;
}
__device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}
__device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p);
}

struct f2 { float hi, lo; };

__device__ __forceinline__ f2 f2_make(float x) { f2 r; r.hi = x; r.lo = 0.0f; return r; }


__device__ __forceinline__ void ema_update_f2(f2 &ema, float x, float alpha)
{
    float s, s_err; two_sum(ema.hi, ema.lo, s, s_err);
    float d_hi, d_err; two_diff(x, s, d_hi, d_err);
    float delta_hi = d_hi;
    float delta_lo = d_err - s_err;

    float p_hi, p_lo; two_prod(alpha, delta_hi, p_hi, p_lo);
    p_lo = fmaf(alpha, delta_lo, p_lo);

    float y_hi, y_lo; two_sum(s, p_hi, y_hi, y_lo);
    y_lo += p_lo;
    quick_two_sum(y_hi, y_lo, ema.hi, ema.lo);
}



__device__ __forceinline__ float rcp_nr(float c)
{
    float r = __fdividef(1.0f, c);
    r = r * fmaf(-c, r, 2.0f);
    return r;
}











__device__ __forceinline__ void warp_inclusive_scan_affine(float &A, float &B, unsigned lane, unsigned mask) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float A_prev = __shfl_up_sync(mask, A, offset);
        const float B_prev = __shfl_up_sync(mask, B, offset);
        if (lane >= static_cast<unsigned>(offset)) {
            const float A_cur = A;
            const float B_cur = B;
            A = A_cur * A_prev;
            B = __fmaf_rn(A_cur, B_prev, B_cur);
        }
    }
}

extern "C" __global__ void kvo_batch_f32(
    const float* __restrict__ vf,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    int n_combos,
    float* __restrict__ out)
{
    if (len <= 0 || n_combos <= 0) return;


    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    for (int combo = blockIdx.x * warps_per_block + warp_id;
         combo < n_combos;
         combo += gridDim.x * warps_per_block)
    {
        const int s = shorts[combo];
        const int l = longs[combo];
        if (s <= 0 || l < s) continue;

        const int warm = first_valid + 1;
        float* __restrict__ row_out = out + (size_t)combo * (size_t)len;

        const float nanv = f32_nan();
        const int warm_end = (warm < len ? warm : len);
        for (int t = lane; t < warm_end; t += 32) row_out[t] = nanv;
        if (warm >= len) continue;

        const float alpha_s = 2.0f / (float)(s + 1);
        const float alpha_l = 2.0f / (float)(l + 1);
        const float beta_s = 1.0f - alpha_s;
        const float beta_l = 1.0f - alpha_l;

        const float seed = vf[warm];
        float ema_s_prev = seed;
        float ema_l_prev = seed;

        if (lane == 0) row_out[warm] = 0.0f;

        for (int t0 = warm + 1; t0 < len; t0 += 32) {
            const int t = t0 + lane;
            float x = 0.0f;
            if (t < len) x = vf[t];

            float As = beta_s;
            float Bs = alpha_s * x;
            float Al = beta_l;
            float Bl = alpha_l * x;

            warp_inclusive_scan_affine(As, Bs, lane, mask);
            warp_inclusive_scan_affine(Al, Bl, lane, mask);

            const float ema_s = __fmaf_rn(As, ema_s_prev, Bs);
            const float ema_l = __fmaf_rn(Al, ema_l_prev, Bl);

            if (t < len) row_out[t] = ema_s - ema_l;

            const int remain = len - 1 - t0;
            const int last_lane = (remain < 31 ? remain : 31);
            ema_s_prev = __shfl_sync(mask, ema_s, last_lane);
            ema_l_prev = __shfl_sync(mask, ema_l, last_lane);
        }
    }
}






extern "C" __global__ void kvo_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int short_p,
    int long_p,
    float* __restrict__ out_tm)
{

    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s];
        if (fv < 0 || fv >= rows) {
            for (int t = 0; t < rows; ++t) out_tm[(size_t)t * (size_t)cols + s] = f32_nan();
            continue;
        }

        const int warm = fv + 1;

        const int warm_end = (warm < rows ? warm : rows);
        for (int t = 0; t < warm_end; ++t) out_tm[(size_t)t * (size_t)cols + s] = f32_nan();
        if (warm >= rows) continue;

        const float alpha_s = 2.0f / (float)(short_p + 1);
        const float alpha_l = 2.0f / (float)(long_p + 1);

        const size_t idx0 = (size_t)fv * (size_t)cols + s;
        double prev_h = (double)high_tm[idx0];
        double prev_l = (double)low_tm[idx0];
        double prev_c = (double)close_tm[idx0];
        double prev_hlc = prev_h + prev_l + prev_c;
        double prev_dm  = prev_h - prev_l;
        int    trend    = -1;
        double cm       = 0.0;


        {
            const size_t idx = (size_t)warm * (size_t)cols + s;
            const double h = (double)high_tm[idx];
            const double l = (double)low_tm[idx];
            const double c = (double)close_tm[idx];
            const double v = (double)volume_tm[idx];
            const double hlc = h + l + c;
            const double dm  = h - l;

            if (hlc > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; }
            else if (hlc < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; }
            cm += dm;

            const double ratio = dm / cm;
            const double temp  = fabs((ratio * 2.0) - 1.0);
            const double sign  = (trend == 1) ? 1.0 : -1.0;
            const float vf     = (float)(v * temp * 100.0 * sign);

            float ema_s = vf;
            float ema_l = vf;
            out_tm[idx] = 0.0f;

            prev_hlc = hlc;
            prev_dm  = dm;

            #pragma unroll 1
            for (int t = warm + 1; t < rows; ++t) {
                const size_t j = (size_t)t * (size_t)cols + s;
                const double h2 = (double)high_tm[j];
                const double l2 = (double)low_tm[j];
                const double c2 = (double)close_tm[j];
                const double v2 = (double)volume_tm[j];
                const double hlc2 = h2 + l2 + c2;
                const double dm2  = h2 - l2;

                if (hlc2 > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; }
                else if (hlc2 < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; }
                cm += dm2;

                const double ratio2 = dm2 / cm;
                const double temp2  = fabs((ratio2 * 2.0) - 1.0);
                const double sign2  = (trend == 1) ? 1.0 : -1.0;
                const float vf2     = (float)(v2 * temp2 * 100.0 * sign2);


                ema_s = fmaf(alpha_s, (vf2 - ema_s), ema_s);
                ema_l = fmaf(alpha_l, (vf2 - ema_l), ema_l);
                out_tm[j] = ema_s - ema_l;

                prev_hlc = hlc2;
                prev_dm  = dm2;
            }
        }
    }
}
