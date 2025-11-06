// CUDA kernels for IFT RSI (Inverse Fisher Transform of RSI) in FP32.
//
// Drop-in FP32 rewrite: removes FP64 from hot paths, adds compensated
// summation for LWMA, and uses warp-parallel seeding where available.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// --------- small utils ---------
static __device__ __forceinline__ float f32_qnan() { return __int_as_float(0x7fffffff); }
static __device__ __forceinline__ int   imax(int a,int b){ return a>b? a:b; }
static __device__ __forceinline__ int   imin(int a,int b){ return a<b? a:b; }

// Kahan-style compensated adder for FP32
struct KahanF32 {
    float sum;
    float c; // compensation
    __device__ __forceinline__ void init(float s = 0.f){ sum=s; c=0.f; }
    __device__ __forceinline__ void add(float x){
        float y = x - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
};

// warp reduction (sum) over up to 32-lane warp
static __device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = __activemask();
    // tree reduce
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v,  8);
    v += __shfl_down_sync(mask, v,  4);
    v += __shfl_down_sync(mask, v,  2);
    v += __shfl_down_sync(mask, v,  1);
    return v;
}

// ------------------ KERNEL 1: one series × many params ------------------
// - Each block (ideally one warp) processes multiple combos in a grid-stride loop.
// - Lane 0 executes the sequential recurrence; other lanes help seed RSI.
extern "C" __global__ void ift_rsi_batch_f32(
    const float* __restrict__ data,
    int series_len,
    int n_combos,
    int first_valid,
    const int* __restrict__ rsi_periods,
    const int* __restrict__ wma_periods,
    float* __restrict__ out_values)
{
    // grid-stride over combos to keep device busy when n_combos is small
    for (int combo = blockIdx.x; combo < n_combos; combo += gridDim.x) {

        const int rp = rsi_periods[combo];
        const int wp = wma_periods[combo];
        const int base = combo * series_len;

        // basic arg checks
        if (UNLIKELY(rp <= 0 || wp <= 0 || rp > series_len || wp > series_len)) {
            for (int t = threadIdx.x; t < series_len; t += blockDim.x) out_values[base + t] = f32_qnan();
            continue;
        }
        if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
            for (int t = threadIdx.x; t < series_len; t += blockDim.x) out_values[base + t] = f32_qnan();
            continue;
        }

        const int tail = series_len - first_valid;
        const int need = imax(rp, wp);
        if (UNLIKELY(tail < need)) {
            for (int t = threadIdx.x; t < series_len; t += blockDim.x) out_values[base + t] = f32_qnan();
            continue;
        }

        const int warm = first_valid + rp + wp - 1;
        for (int t = threadIdx.x; t < imin(warm, series_len); t += blockDim.x) out_values[base + t] = f32_qnan();

        // dynamic shared memory ring buffer: host must pass max_wp*sizeof(float)
        extern __shared__ float shmem[];
        float* ring = shmem; // ring[0..wp-1] per block
        // we unconditionally require ring memory
        if (UNLIKELY(wp <= 0)) continue; // already screened; defensive

        // -------- seed Wilder's SMMA over the first rp diffs --------
        const int lane = threadIdx.x & 31;
        const int seed_start = first_valid + 1;
        const int seed_end   = seed_start + rp - 1;

        float gain_seed = 0.f, loss_seed = 0.f;

        if (blockDim.x >= 32) {
            // strided over [seed_start .. seed_end]; each lane sums its slice
            float gain_part = 0.f, loss_part = 0.f;
            for (int i = seed_start + lane; i <= seed_end; i += 32) {
                float cur  = data[i];
                float prev = data[i - 1];
                float d = cur - prev;
                if (d > 0.f) gain_part += d; else loss_part += -d;
            }
            // warp reduction
            gain_seed = warp_sum(gain_part);
            loss_seed = warp_sum(loss_part);
        } else {
            // Fallback for tiny blocks: compute seed on lane 0 sequentially
            if (threadIdx.x == 0) {
                float g = 0.f, l = 0.f;
                for (int i = seed_start; i <= seed_end; ++i) {
                    float d = data[i] - data[i - 1];
                    if (d > 0.f) g += d; else l += -d;
                }
                gain_seed = g; loss_seed = l;
            }
        }

        // lane 0 finalizes seed and runs the sequential recurrences
        if (lane == 0) {
            const float rp_rcp = 1.0f / (float)rp;
            float avg_gain = gain_seed * rp_rcp;
            float avg_loss = loss_seed * rp_rcp;
            const float alpha = rp_rcp;
            const float beta  = 1.0f - alpha;

            // LWMA state for x = 0.1*(RSI - 50)
            const float wp_f = (float)wp;
            const float denom_rcp = 2.0f / (wp_f * (wp_f + 1.0f));
            int head = 0, filled = 0;
            KahanF32 S1; S1.init(0.f);
            KahanF32 S2; S2.init(0.f);

            // initialize streaming prev to avoid 2nd load per step
            float prev = data[first_valid + rp];

            // iterate absolute offset i = rp..(tail-1), abs_t = first_valid + i
            for (int i = rp; i < tail; ++i) {
                if (i > rp) {
                    const int abs_idx = first_valid + i;
                    float curr = data[abs_idx];
                    float d = curr - prev;
                    prev = curr;
                    float g = (d > 0.f) ? d : 0.f;
                    float l = (d > 0.f) ? 0.f : -d;
                    // Wilder SMMA recurrence in FP32 with FMA
                    avg_gain = __fmaf_rn(alpha, g, beta * avg_gain);
                    avg_loss = __fmaf_rn(alpha, l, beta * avg_loss);
                }

                // RSI -> x
                float rs  = (avg_loss != 0.f) ? (avg_gain / avg_loss) : 100.f;
                float rsi = 100.f - 100.f / (1.f + rs);
                float x   = 0.1f * (rsi - 50.f);

                if (filled < wp) {
                    S1.add(x);
                    S2.add((float)(filled + 1) * x);
                    ring[head] = x;
                    head = (head + 1 == wp) ? 0 : head + 1;
                    filled += 1;
                    if (filled == wp) {
                        float wma = S2.sum * denom_rcp;
                        const int abs_t = first_valid + i;
                        out_values[base + abs_t] = tanhf(wma);
                    }
                } else {
                    float x_old = ring[head];
                    ring[head]  = x;
                    head = (head + 1 == wp) ? 0 : head + 1;

                    float S1_prev = S1.sum;
                    // S1' = S1 + x - x_old    (compensated)
                    S1.add(x);
                    S1.add(-x_old);

                    // S2' = S2 - S1_prev + wp*x   (compensated)
                    S2.add(-S1_prev);
                    S2.add(wp_f * x);

                    float wma = S2.sum * denom_rcp;
                    const int abs_t = first_valid + i;
                    out_values[base + abs_t] = tanhf(wma);
                }
            }
        }
    }
}

// ------------------ KERNEL 2: many series × one param (time-major) ------------------
// Each thread handles one series. Uses per-thread slice in shared memory for the ring.
extern "C" __global__ void ift_rsi_many_series_one_param_f32(
    const float* __restrict__ data_tm,     // time-major [row * num_series + series]
    const int*   __restrict__ first_valids,// per-series first valid index
    int num_series,
    int series_len,
    int rsi_period,
    int wma_period,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int rp = rsi_period;
    const int wp = wma_period;

    if (UNLIKELY(rp <= 0 || wp <= 0 || rp > series_len || wp > series_len)) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }
    int first = first_valids ? first_valids[series] : 0;
    if (first < 0) first = 0;
    if (UNLIKELY(first >= series_len)) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }
    const int tail = series_len - first;
    if (UNLIKELY(tail < imax(rp, wp))) {
        for (int r = 0; r < series_len; ++r) out_tm[r * num_series + series] = f32_qnan();
        return;
    }

    const int warm = first + rp + wp - 1;
    for (int r = 0; r < imin(warm, series_len); ++r) out_tm[r * num_series + series] = f32_qnan();

    // seed Wilder over [first+1 .. first+rp]
    float gain_part = 0.f, loss_part = 0.f;
    const int seed_start = first + 1;
    const int seed_end   = seed_start + rp - 1;
    for (int i = seed_start; i <= seed_end; ++i) {
        const float cur  = data_tm[i * num_series + series];
        const float prev = data_tm[(i - 1) * num_series + series];
        const float d = cur - prev;
        if (d > 0.f) gain_part += d; else loss_part += -d;
    }
    const float rp_rcp = 1.0f / (float)rp;
    float avg_gain = gain_part * rp_rcp;
    float avg_loss = loss_part * rp_rcp;
    const float alpha = rp_rcp;
    const float beta  = 1.0f - alpha;

    // LWMA state
    const float wp_f = (float)wp;
    const float denom_rcp = 2.0f / (wp_f * (wp_f + 1.0f));
    int head = 0, filled = 0;
    KahanF32 S1; S1.init(0.f);
    KahanF32 S2; S2.init(0.f);

    extern __shared__ float shbuf[];
    float* ring = shbuf + threadIdx.x * wp;

    // streaming prev
    float prev = data_tm[(first + rp) * num_series + series];

    for (int r = first + rp; r < series_len; ++r) {
        if (r > first + rp) {
            const float curr = data_tm[r * num_series + series];
            const float d = curr - prev;
            prev = curr;
            const float g = (d > 0.f) ? d : 0.f;
            const float l = (d > 0.f) ? 0.f : -d;
            avg_gain = __fmaf_rn(alpha, g, beta * avg_gain);
            avg_loss = __fmaf_rn(alpha, l, beta * avg_loss);
        }

        const float rs  = (avg_loss != 0.f) ? (avg_gain / avg_loss) : 100.f;
        const float rsi = 100.f - 100.f / (1.f + rs);
        const float x   = 0.1f * (rsi - 50.f);

        if (filled < wp) {
            S1.add(x);
            S2.add((float)(filled + 1) * x);
            ring[head] = x;
            head = (head + 1 == wp) ? 0 : head + 1;
            filled += 1;
            if (filled == wp) {
                const float wma = S2.sum * denom_rcp;
                out_tm[r * num_series + series] = tanhf(wma);
            }
        } else {
            const float x_old = ring[head];
            ring[head] = x;
            head = (head + 1 == wp) ? 0 : head + 1;

            const float S1_prev = S1.sum;
            S1.add(x);
            S1.add(-x_old);

            S2.add(-S1_prev);
            S2.add(wp_f * x);

            const float wma = S2.sum * denom_rcp;
            out_tm[r * num_series + series] = tanhf(wma);
        }
    }
}
