










#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CMO_NAN
#define CMO_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


#ifndef CMO_BLOCK_SIZE
#define CMO_BLOCK_SIZE 256
#endif
#ifndef CMO_TILE
#define CMO_TILE 256
#endif



struct KBN32 {
    float s;
    float c;
    __device__ inline void init() { s = 0.0f; c = 0.0f; }
    __device__ inline void add(float x) {
        float t = s + x;
        if (fabsf(s) >= fabsf(x)) c += (s - t) + x;
        else                      c += (x - t) + s;
        s = t;
    }
    __device__ inline float result() const { return s + c; }
};


__device__ inline float cmo_from_avgs(float avg_g, float avg_l) {
    float denom = avg_g + avg_l;
    if (denom == 0.0f) return 0.0f;
    float numer = avg_g - avg_l;
    return 100.0f * (numer / denom);
}







extern "C" __global__ void cmo_batch_f32(
    const float*  __restrict__ prices,
    const int*    __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {

    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned warps_per_block = blockDim.x >> 5;
    const int combo = (int)(blockIdx.x * warps_per_block + warp);
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + (size_t)combo * (size_t)series_len;


    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = CMO_NAN;
        return;
    }
    const int fv   = first_valid;
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= period)) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = CMO_NAN;
        return;
    }

    const int warm = fv + period;


    for (int i = (int)lane; i < warm; i += 32) out_row[i] = CMO_NAN;


    const float beta  = 1.0f / (float)period;
    const float alpha = 1.0f - beta;


    float avg_g = 0.0f;
    float avg_l = 0.0f;
    if (lane == 0) {
        float prev = prices[fv];
        KBN32 sum_g, sum_l;
        sum_g.init();
        sum_l.init();
        for (int i = fv + 1; i <= warm; ++i) {
            float curr = prices[i];
            float diff = curr - prev;
            prev = curr;
            float g = fmaxf(diff, 0.0f);
            float l = fmaxf(-diff, 0.0f);
            sum_g.add(g);
            sum_l.add(l);
        }
        avg_g = sum_g.result() * beta;
        avg_l = sum_l.result() * beta;
        out_row[warm] = cmo_from_avgs(avg_g, avg_l);
    }

    const unsigned mask = 0xffffffffu;
    avg_g = __shfl_sync(mask, avg_g, 0);
    avg_l = __shfl_sync(mask, avg_l, 0);


    for (int t0 = warm + 1; t0 < series_len; t0 += 32) {
        const int t = t0 + (int)lane;

        float A  = 1.0f;
        float Bg = 0.0f;
        float Bl = 0.0f;
        if (t < series_len) {
            const float p1 = prices[t];
            const float p0 = prices[t - 1];
            const float diff = p1 - p0;
            const float g = fmaxf(diff, 0.0f);
            const float l = fmaxf(-diff, 0.0f);
            A  = alpha;
            Bg = beta * g;
            Bl = beta * l;
        }



        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev  = __shfl_up_sync(mask, A, offset);
            const float Bg_prev = __shfl_up_sync(mask, Bg, offset);
            const float Bl_prev = __shfl_up_sync(mask, Bl, offset);
            if (lane >= (unsigned)offset) {
                const float A_cur  = A;
                const float Bg_cur = Bg;
                const float Bl_cur = Bl;
                A  = A_cur * A_prev;
                Bg = __fmaf_rn(A_cur, Bg_prev, Bg_cur);
                Bl = __fmaf_rn(A_cur, Bl_prev, Bl_cur);
            }
        }

        const float yg = __fmaf_rn(A, avg_g, Bg);
        const float yl = __fmaf_rn(A, avg_l, Bl);

        if (t < series_len) {
            out_row[t] = cmo_from_avgs(yg, yl);
        }


        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        avg_g = __shfl_sync(mask, yg, last_lane);
        avg_l = __shfl_sync(mask, yl, last_lane);
    }
}





extern "C" __global__ void cmo_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(period <= 0 || period > series_len || fv < 0 || fv >= series_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = CMO_NAN;
        return;
    }
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= period)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = CMO_NAN;
        return;
    }

    const int warm = fv + period;
    const float beta  = 1.0f / (float)period;
    const float alpha = 1.0f - beta;


    {
        float* o = out_tm + series;
        for (int r = 0; r < warm; ++r, o += num_series) *o = CMO_NAN;
    }


    float prev = *(prices_tm + (size_t)fv * num_series + series);
    KBN32 sum_g, sum_l; sum_g.init(); sum_l.init();

    for (int r = fv + 1; r <= warm; ++r) {
        float curr = *(prices_tm + (size_t)r * num_series + series);
        float diff = curr - prev; prev = curr;
        float g = fmaxf(diff, 0.0f);
        float l = fmaxf(-diff, 0.0f);
        sum_g.add(g);
        sum_l.add(l);
    }
    float avg_g = sum_g.result() * beta;
    float avg_l = sum_l.result() * beta;

    *(out_tm + (size_t)warm * num_series + series) = cmo_from_avgs(avg_g, avg_l);


    for (int r = warm + 1; r < series_len; ++r) {
        float curr = *(prices_tm + (size_t)r * num_series + series);
        float diff = curr - prev; prev = curr;
        float g = fmaxf(diff, 0.0f);
        float l = fmaxf(-diff, 0.0f);
        avg_g = __fmaf_rn(alpha, avg_g, beta * g);
        avg_l = __fmaf_rn(alpha, avg_l, beta * l);
        *(out_tm + (size_t)r * num_series + series) = cmo_from_avgs(avg_g, avg_l);
    }
}
