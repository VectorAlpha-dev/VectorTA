
















#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif





#ifndef HMA_ASSUME_OUT_PREFILLED
#define HMA_ASSUME_OUT_PREFILLED 0
#endif




#ifndef HMA_RING_IN_SHARED
#define HMA_RING_IN_SHARED 1
#endif


#include <cuda_runtime.h>
#include <math.h>

#ifndef HMA_NAN
#define HMA_NAN (__int_as_float(0x7fffffff))
#endif

static __device__ __forceinline__ int clamp_positive(int v) { return v > 0 ? v : 0; }

extern "C" __global__
void hma_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   int max_sqrt_len,
                   float* __restrict__ x_buf,
                   float* __restrict__ out) {

#if HMA_RING_IN_SHARED
    extern __shared__ float sh_ring[];
#endif


    const int stride = blockDim.x * gridDim.x;
    for (int combo = blockIdx.x * blockDim.x + threadIdx.x; combo < n_combos; combo += stride) {

        const int base = combo * series_len;

        const int period = periods[combo];
        const int half   = period >> 1;
#if !HMA_ASSUME_OUT_PREFILLED
        if (period < 2 || half < 1) {
            for (int i = 0; i < series_len; ++i) { out[base + i] = HMA_NAN; }
            continue;
        }
#else
        if (period < 2 || half < 1) { continue; }
#endif

        int sqrt_len = (int)sqrtf((float)period);
        if (sqrt_len < 1) sqrt_len = 1;
#if !HMA_ASSUME_OUT_PREFILLED
        if (sqrt_len > max_sqrt_len) {
            for (int i = 0; i < series_len; ++i) { out[base + i] = HMA_NAN; }
            continue;
        }
#else
        if (sqrt_len > max_sqrt_len) { continue; }
#endif

#if !HMA_ASSUME_OUT_PREFILLED
        if ((unsigned)first_valid >= (unsigned)series_len) {
            for (int i = 0; i < series_len; ++i) { out[base + i] = HMA_NAN; }
            continue;
        }
#else
        if ((unsigned)first_valid >= (unsigned)series_len) { continue; }
#endif

        const int tail_len = series_len - first_valid;
#if !HMA_ASSUME_OUT_PREFILLED
        if (tail_len < period + sqrt_len - 1) {
            for (int i = 0; i < series_len; ++i) { out[base + i] = HMA_NAN; }
            continue;
        }



        int warmup_end = first_valid + period + sqrt_len - 2;
        if (warmup_end > series_len) warmup_end = series_len;
        for (int i = 0; i < warmup_end; ++i) { out[base + i] = HMA_NAN; }
#else
        if (tail_len < period + sqrt_len - 1) { continue; }
#endif


        const float f_half   = (float)half;
        const float f_full   = (float)period;
        const float f_sqrt   = (float)sqrt_len;

        const float ws_half  = 0.5f * f_half * (f_half + 1.0f);
        const float ws_full  = 0.5f * f_full * (f_full + 1.0f);
        const float ws_sqrt  = 0.5f * f_sqrt * (f_sqrt + 1.0f);

        const float inv_ws_half = 1.0f / ws_half;
        const float inv_ws_full = 1.0f / ws_full;
        const float inv_ws_sqrt = 1.0f / ws_sqrt;


        float sum_half = 0.0f, wsum_half = 0.0f;
        float sum_full = 0.0f, wsum_full = 0.0f;


        float sum_x = 0.0f, wsum_x = 0.0f;
        int   ring_head = 0;
        int   ring_count = 0;


#if HMA_RING_IN_SHARED
        float* ring = sh_ring + threadIdx.x * max_sqrt_len;
#else
        float* ring = x_buf + combo * max_sqrt_len;
#endif



        for (int j = 0; j < tail_len; ++j) {
            const int idx = first_valid + j;

            const float val = prices[idx];


            if (j < period) {
                const float jf = (float)(j + 1);
                wsum_full = fmaf(jf, val, wsum_full);
                sum_full  += val;
            } else {
                const float old = prices[idx - period];
                const float prev_sum = sum_full;
                sum_full = prev_sum + val - old;
                wsum_full = fmaf((float)period, val, wsum_full - prev_sum);
            }


            if (j < half) {
                const float jf = (float)(j + 1);
                wsum_half = fmaf(jf, val, wsum_half);
                sum_half  += val;
            } else {
                const float old = prices[idx - half];
                const float prev_sum = sum_half;
                sum_half = prev_sum + val - old;
                wsum_half = fmaf((float)half, val, wsum_half - prev_sum);
            }


            if (j + 1 < period) { continue; }

            const float wma_full = wsum_full * inv_ws_full;
            const float wma_half = wsum_half * inv_ws_half;
            const float x_val    = 2.0f * wma_half - wma_full;

            if (ring_count < sqrt_len) {
                ring[ring_count] = x_val;
                const float rc1 = (float)(ring_count + 1);
                wsum_x = fmaf(rc1, x_val, wsum_x);
                sum_x  += x_val;
                ++ring_count;

                if (ring_count == sqrt_len) {
                    out[base + idx] = wsum_x * inv_ws_sqrt;
                }
            } else {
                const float old_x = ring[ring_head];
                ring[ring_head] = x_val;
                ++ring_head; if (ring_head == sqrt_len) ring_head = 0;

                const float prev_sum = sum_x;
                sum_x   = prev_sum + x_val - old_x;
                wsum_x  = fmaf((float)sqrt_len, x_val, wsum_x - prev_sum);

                out[base + idx] = wsum_x * inv_ws_sqrt;
            }
        }
    }
}

extern "C" __global__
void hma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int num_series,
                                   int series_len,
                                   int period,
                                   int max_sqrt_len,
                                   float* __restrict__ x_buf,
                                   float* __restrict__ out_tm) {

#if HMA_RING_IN_SHARED
    extern __shared__ float sh_ring[];
#endif

    if (period < 2) return;
    const int half = period >> 1;
    if (half < 1) return;

    int sqrt_len = (int)sqrtf((float)period);
    if (sqrt_len < 1) sqrt_len = 1;
    if (sqrt_len > max_sqrt_len) return;

    const float f_half   = (float)half;
    const float f_full   = (float)period;
    const float f_sqrt   = (float)sqrt_len;

    const float ws_half  = 0.5f * f_half * (f_half + 1.0f);
    const float ws_full  = 0.5f * f_full * (f_full + 1.0f);
    const float ws_sqrt  = 0.5f * f_sqrt * (f_sqrt + 1.0f);

    const float inv_ws_half = 1.0f / ws_half;
    const float inv_ws_full = 1.0f / ws_full;
    const float inv_ws_sqrt = 1.0f / ws_sqrt;

    const int stride = blockDim.x * gridDim.x;

    for (int series = blockIdx.x * blockDim.x + threadIdx.x; series < num_series; series += stride) {

        const int first_valid = first_valids[series];
#if !HMA_ASSUME_OUT_PREFILLED
        if ((unsigned)first_valid >= (unsigned)series_len) {
            for (int row = 0; row < series_len; ++row) {
                out_tm[row * num_series + series] = HMA_NAN;
            }
            continue;
        }
#else
        if ((unsigned)first_valid >= (unsigned)series_len) { continue; }
#endif

        const int tail_len = series_len - first_valid;
#if !HMA_ASSUME_OUT_PREFILLED
        if (tail_len < period + sqrt_len - 1) {
            for (int row = 0; row < series_len; ++row) {
                out_tm[row * num_series + series] = HMA_NAN;
            }
            continue;
        }

        int warmup_end = first_valid + period + sqrt_len - 2;
        if (warmup_end > series_len) warmup_end = series_len;
        for (int row = 0; row < warmup_end; ++row) {
            out_tm[row * num_series + series] = HMA_NAN;
        }
#else
        if (tail_len < period + sqrt_len - 1) { continue; }
#endif


        float sum_half = 0.0f, wsum_half = 0.0f;
        float sum_full = 0.0f, wsum_full = 0.0f;

        float sum_x = 0.0f, wsum_x = 0.0f;
        int   ring_head = 0;
        int   ring_count = 0;

#if HMA_RING_IN_SHARED
        float* ring = sh_ring + threadIdx.x * max_sqrt_len;
#else
        float* ring = x_buf + series * max_sqrt_len;
#endif


        for (int j = 0; j < tail_len; ++j) {
            const int row = first_valid + j;
            const int a   = row * num_series + series;
            const float val = prices_tm[a];

            if (j < period) {
                const float jf = (float)(j + 1);
                wsum_full = fmaf(jf, val, wsum_full);
                sum_full  += val;
            } else {
                const float old = prices_tm[(row - period) * num_series + series];
                const float prev_sum = sum_full;
                sum_full = prev_sum + val - old;
                wsum_full = fmaf(f_full, val, wsum_full - prev_sum);
            }

            if (j < half) {
                const float jf = (float)(j + 1);
                wsum_half = fmaf(jf, val, wsum_half);
                sum_half  += val;
            } else {
                const float old = prices_tm[(row - half) * num_series + series];
                const float prev_sum = sum_half;
                sum_half = prev_sum + val - old;
                wsum_half = fmaf(f_half, val, wsum_half - prev_sum);
            }

            if (j + 1 < period) { continue; }

            const float wma_full = wsum_full * inv_ws_full;
            const float wma_half = wsum_half * inv_ws_half;
            const float x_val    = 2.0f * wma_half - wma_full;

            if (ring_count < sqrt_len) {
                ring[ring_count] = x_val;
                const float rc1 = (float)(ring_count + 1);
                wsum_x = fmaf(rc1, x_val, wsum_x);
                sum_x  += x_val;
                ++ring_count;

                if (ring_count == sqrt_len) {
                    out_tm[a] = wsum_x * inv_ws_sqrt;
                }
            } else {
                const float old_x = ring[ring_head];
                ring[ring_head] = x_val;
                ++ring_head; if (ring_head == sqrt_len) ring_head = 0;

                const float prev_sum = sum_x;
                sum_x  = prev_sum + x_val - old_x;
                wsum_x = fmaf(f_sqrt, x_val, wsum_x - prev_sum);

                out_tm[a] = wsum_x * inv_ws_sqrt;
            }
        }
    }
}
