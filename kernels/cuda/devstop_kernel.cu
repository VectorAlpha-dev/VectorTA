

















#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }
__device__ __forceinline__ int wrap_inc(int x, int cap) { int nx = x + 1; return (nx == cap) ? 0 : nx; }


struct kahan_t { float s, c; };
__device__ __forceinline__ void kahan_add(kahan_t &K, float x) {
    float y = x - K.c;
    float t = K.s + y;
    K.c = (t - K.s) - y;
    K.s = t;
}
__device__ __forceinline__ void kahan_sub(kahan_t &K, float x) { kahan_add(K, -x); }


struct ds_t { float hi, lo; };
__device__ __forceinline__ ds_t ds_from2(float2 v) { ds_t r; r.hi = v.x; r.lo = v.y; return r; }
__device__ __forceinline__ float ds_to_f(ds_t a) { return a.hi + a.lo; }
__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b; float bb = s - a; e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b; e = b - (s - a);
}
__device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b; e = __fmaf_rn(a, b, -p);
}
__device__ __forceinline__ ds_t ds_add(ds_t x, ds_t y) {
    float s1, e1; two_sum(x.hi, y.hi, s1, e1);
    float s2, e2; two_sum(x.lo, y.lo, s2, e2);
    float s3, e3; two_sum(s1, s2, s3, e3);
    float e = e1 + e2 + e3; float hi, lo; quick_two_sum(s3, e, hi, lo); ds_t r{hi, lo}; return r;
}
__device__ __forceinline__ ds_t ds_sub(ds_t x, ds_t y) { ds_t r{ x.hi - y.hi, x.lo - y.lo };
    float s, e; two_sum(x.hi, -y.hi, s, e); float t, f; two_sum(x.lo, -y.lo, t, f); float u, g; two_sum(s, t, u, g);
    float hi, lo; quick_two_sum(u, e + f + g, hi, lo); return ds_t{hi, lo}; }
__device__ __forceinline__ ds_t ds_scale(ds_t x, float b) {
    float p, e; two_prod(x.hi, b, p, e); float s, t; two_sum(p, x.lo * b, s, t); float hi, lo; quick_two_sum(s, e + t, hi, lo); return ds_t{hi, lo};
}
__device__ __forceinline__ ds_t ds_mul(ds_t x, ds_t y) {
    float p, e; two_prod(x.hi, y.hi, p, e);
    float c1 = __fmaf_rn(x.hi, y.lo, 0.0f); float c2 = __fmaf_rn(x.lo, y.hi, 0.0f);
    float s, t; two_sum(p, c1 + c2, s, t); float err = e + t + __fmaf_rn(x.lo, y.lo, 0.0f);
    float hi, lo; quick_two_sum(s, err, hi, lo); return ds_t{hi, lo};
}





extern "C" __global__ void devstop_batch_grouped_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float2* __restrict__ p1,
    const float2* __restrict__ p2,
    const int* __restrict__ pc,
    int len,
    int first_valid,
    int period,
    const float* __restrict__ mults,
    int n_combos,
    int is_long,
    int out_row_base,
    float* __restrict__ out
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || period <= 0) return;

    const int warm = first_valid + 2 * period - 1;
    const int row = out_row_base + combo;
    const int row_off = row * len;


    const int warm_clamp = (warm < len) ? warm : len;
    for (int t = threadIdx.x; t < warm_clamp; t += blockDim.x) { out[row_off + t] = qnan32(); }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (warm >= len) return;

    extern __shared__ unsigned char smem[];
    float* base_ring = reinterpret_cast<float*>(smem);
    int* dq_idx = reinterpret_cast<int*>(base_ring + period);
    for (int i = 0; i < period; ++i) { base_ring[i] = qnan32(); dq_idx[i] = 0; }

    int dq_head = 0, dq_len = 0;
    const int cap = period;


    auto dq_back_at = [&](int len_) { int pos = dq_head + len_ - 1; if (pos >= cap) pos -= cap; return dq_idx[pos]; };
    auto dq_push_back = [&](int value) { int pos = dq_head + dq_len; if (pos >= cap) pos -= cap; dq_idx[pos] = value; dq_len += 1; };
    auto dq_pop_back = [&]() { dq_len -= 1; };
    auto dq_pop_front = [&]() { dq_head = wrap_inc(dq_head, cap); dq_len -= 1; };
    auto dq_front = [&]() { return dq_idx[dq_head]; };

    const float mult = mults[combo];
    const int start_base = first_valid + period;
    const int start_final = start_base + period - 1;

    int slot = start_base % period;
    for (int i = start_base; i < len; ++i) {

        const int t1 = i + 1;
        int a = t1 - period; if (a < 0) a = 0;
        const int cnt = pc[t1] - pc[a];
        float base = qnan32();
        if (cnt > 0) {

            ds_t S1 = ds_sub(ds_from2(p1[t1]), ds_from2(p1[a]));
            ds_t S2 = ds_sub(ds_from2(p2[t1]), ds_from2(p2[a]));
            const float inv = 1.0f / (float)cnt;
            ds_t mean_ds = ds_scale(S1, inv);
            ds_t m2_ds   = ds_scale(S2, inv);
            ds_t var_ds  = ds_sub(m2_ds, ds_mul(mean_ds, mean_ds));
            const float mean = ds_to_f(mean_ds);
            float var = ds_to_f(var_ds);
            if (var < 0.0f) var = 0.0f;
            const float sigma = sqrtf(var);
            const float h = high[i];
            const float l = low[i];
            if (is_long) {
                if (!isnan(h)) {
                    base = h - mean - mult * sigma;
                }
            } else {
                if (!isnan(l)) {
                    base = l + mean + mult * sigma;
                }
            }
        }



        if (isnan(base)) { base = is_long ? -INFINITY : INFINITY; }
        base_ring[slot] = base;
        {

            const int cut = i + 1 - period;
            while (dq_len > 0 && dq_front() < cut) dq_pop_front();

            if (is_long) {

                while (dq_len > 0) {
                    int j = dq_back_at(dq_len);
                    float bj = base_ring[j % period];
                    if (isnan(bj) || bj <= base) dq_pop_back(); else break;
                }
            } else {

                while (dq_len > 0) {
                    int j = dq_back_at(dq_len);
                    float bj = base_ring[j % period];
                    if (isnan(bj) || bj >= base) dq_pop_back(); else break;
                }
            }
            dq_push_back(i);
        }


        const int cut = i + 1 - period;
        while (dq_len > 0 && dq_front() < cut) { dq_pop_front(); }

        if (i >= start_final) {
            float out_val = qnan32();
            if (dq_len > 0) {
                int j = dq_front();
                out_val = base_ring[j % period];
            }
            out[row_off + i] = out_val;
        }
        slot = wrap_inc(slot, cap);
    }
}






extern "C" __global__ void devstop_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float mult,
    int is_long,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= cols || period <= 0) return;


    const int fv = first_valids[s];
    const int start_base  = fv + period;
    const int start_final = start_base + period - 1;


    const int warm_clamp = (start_final < rows) ? start_final : rows;
    for (int t = threadIdx.x; t < warm_clamp; t += blockDim.x) { out_tm[t * cols + s] = qnan32(); }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (start_base >= rows) return;

    extern __shared__ unsigned char smem_uc[];
    float* r_ring = reinterpret_cast<float*>(smem_uc);
    float* base_ring = r_ring + period;
    int* dq_idx = reinterpret_cast<int*>(base_ring + period);
    for (int i = 0; i < period; ++i) { r_ring[i] = qnan32(); base_ring[i] = qnan32(); dq_idx[i] = 0; }

    int r_pos = 0; int r_inserted = 0; int cnt = 0;
    kahan_t S1{0.0f, 0.0f}, S2{0.0f, 0.0f};
    float prev_h = high_tm[fv * cols + s];
    float prev_l = low_tm [fv * cols + s];


    for (int k = fv + 1; k < min(start_base, rows); ++k) {
        const float h = high_tm[k * cols + s];
        const float l = low_tm [k * cols + s];
        float r = qnan32();
        if (!isnan(h) && !isnan(l) && !isnan(prev_h) && !isnan(prev_l)) {
            const float hi2 = (h > prev_h) ? h : prev_h;
            const float lo2 = (l < prev_l) ? l : prev_l;
            r = hi2 - lo2;
        }
        r_ring[r_pos] = r; r_pos = (r_pos + 1) % period; r_inserted += 1;
        if (!isnan(r)) { kahan_add(S1, r); kahan_add(S2, __fmaf_rn(r, r, 0.0f)); cnt += 1; }
        prev_h = h; prev_l = l;
    }
    r_pos = (period - 1) % period;

    int dq_head = 0, dq_len = 0; const int cap = period;
    auto dq_back_at = [&](int len_) { int pos = dq_head + len_ - 1; if (pos >= cap) pos -= cap; return dq_idx[pos]; };
    auto dq_push_back = [&](int value) { int pos = dq_head + dq_len; if (pos >= cap) pos -= cap; dq_idx[pos] = value; dq_len += 1; };
    auto dq_pop_back = [&]() { dq_len -= 1; };
    auto dq_pop_front = [&]() { dq_head = wrap_inc(dq_head, cap); dq_len -= 1; };
    auto dq_front = [&]() { return dq_idx[dq_head]; };

    for (int i = start_base; i < rows; ++i) {
        const float h = high_tm[i * cols + s];
        const float l = low_tm [i * cols + s];

        float r_new = qnan32();
        if (!isnan(h) && !isnan(l) && !isnan(prev_h) && !isnan(prev_l)) {
            const float hi2 = (h > prev_h) ? h : prev_h;
            const float lo2 = (l < prev_l) ? l : prev_l;
            r_new = hi2 - lo2;
        }
        prev_h = h; prev_l = l;

        const bool had_full = (r_inserted >= period);
        const float old = had_full ? r_ring[r_pos] : qnan32();
        if (had_full && !isnan(old)) { kahan_sub(S1, old); kahan_sub(S2, __fmaf_rn(old, old, 0.0f)); cnt -= 1; }
        r_ring[r_pos] = r_new; r_pos = (r_pos + 1) % period; r_inserted += 1;
        if (!isnan(r_new)) { kahan_add(S1, r_new); kahan_add(S2, __fmaf_rn(r_new, r_new, 0.0f)); cnt += 1; }

        float base = qnan32();
        if (cnt > 0) {
            const float inv = 1.0f / (float)cnt;
            const float mean = S1.s * inv;
            float var = __fmaf_rn(-mean, mean, S2.s * inv);
            if (var < 0.0f) var = 0.0f;
            const float sigma = sqrtf(var);
            if (is_long) {
                if (!isnan(h)) base = h - mean - mult * sigma;
            } else {
                if (!isnan(l)) base = l + mean + mult * sigma;
            }
        }

        const int slot = i % period;
        if (isnan(base)) { base = is_long ? -INFINITY : INFINITY; }
        base_ring[slot] = base;
        {

            const int cut = i + 1 - period;
            while (dq_len > 0 && dq_front() < cut) dq_pop_front();

            if (is_long) {
                while (dq_len > 0) {
                    int j = dq_back_at(dq_len);
                    float bj = base_ring[j % period];
                    if (isnan(bj) || bj <= base) dq_pop_back(); else break;
                }
            } else {
                while (dq_len > 0) {
                    int j = dq_back_at(dq_len);
                    float bj = base_ring[j % period];
                    if (isnan(bj) || bj >= base) dq_pop_back(); else break;
                }
            }
            dq_push_back(i);
        }

        if (i >= start_final) {
            float out_val = qnan32();
            if (dq_len > 0) { int j = dq_front(); out_val = base_ring[j % period]; }
            out_tm[i * cols + s] = out_val;
        }
    }
}
