// CUDA kernels for the Predictive Moving Average (PMA).
//
// This implementation mirrors the scalar PMA in src/indicators/pma.rs which
// uses O(1) rolling recurrences for two 7-tap LWMAs (WMA1 over prices and
// WMA2 over WMA1), and a 4-tap LWMA trigger over the predict line.
//
// Important: This PMA variant does NOT use the 1-bar lag used by the
// Ehlers-PMA module. Warmups therefore are:
//  - predict warm at: first_valid + 6 (first 7 prices)
//  - trigger warm at: first_valid + 9 (predict LWMA4)
//
#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float nan32() { return nanf(""); }

// Kahan-compensated 7-tap LWMA (weights 1..7 oldest..newest)
struct lwma7_recur_f32 {
    float buf[7];
    int   head;
    int   count;
    int   ticks;
    float s1, c1;   // simple sum
    float s2, c2;   // weighted sum (1..7)

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int i = 0; i < 7; ++i) buf[i] = 0.f;
        head = 0; count = 0; ticks = 0; s1 = c1 = 0.f; s2 = c2 = 0.f;
    }

    __device__ __forceinline__ void kahan_add(float y, float &s, float &c) {
        float t = __fadd_rn(s, __fsub_rn(y, c));
        c = __fsub_rn(__fsub_rn(t, s), __fsub_rn(y, c));
        s = t;
    }

    __device__ __forceinline__ void push(float x) {
        const float old = buf[head];
        buf[head] = x;
        head++; if (head == 7) head = 0;
        if (count < 7) count++;

        const float s1_old = s1;
        // S2' = S2 + (N*x_new - S1_old)  (zero-padded semantics before full)
        kahan_add(__fmaf_rn(7.f, x, -s1_old), s2, c2);
        // S1' = S1 + x_new - x_old
        kahan_add(x, s1, c1);
        kahan_add(-old, s1, c1);

        // Periodic renormalization
        ticks++;
        if ((ticks & 0x3FF) == 0) {
            float ns1 = 0.f, nc1 = 0.f, ns2 = 0.f, nc2 = 0.f;
            #pragma unroll
            for (int i = 0; i < 7; ++i) {
                const int idx = (head + i) % 7; // oldest at i=0
                const float v = buf[idx];
                kahan_add(v, ns1, nc1);
                kahan_add(__fmul_rn((float)(i + 1), v), ns2, nc2);
            }
            s1 = ns1; c1 = nc1; s2 = ns2; c2 = nc2;
        }
    }

    __device__ __forceinline__ float value() const { return __fmul_rn(s2, 1.f / 28.f); }

    // Seed with 7 values provided oldest..newest. Initializes internal buffers
    // and exact s1/s2 (no compensation needed at seed point to match scalar).
    __device__ __forceinline__ void seed_from7(const float x[7]) {
        #pragma unroll
        for (int i = 0; i < 7; ++i) buf[i] = x[i];
        head = 0; count = 7; ticks = 0;
        float sum = 0.f; float wsum = 0.f;
        #pragma unroll
        for (int i = 0; i < 7; ++i) { sum = __fadd_rn(sum, x[i]); wsum = __fadd_rn(wsum, __fmul_rn((float)(i+1), x[i])); }
        s1 = sum; c1 = 0.f; s2 = wsum; c2 = 0.f;
    }
};

// 4-tap LWMA using the same recurrence pattern.
struct lwma4_recur_f32 {
    float buf[4];
    int   head;
    int   count;
    int   ticks;
    float s1, c1, s2, c2; // Kahan for simple and weighted sums

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int i = 0; i < 4; ++i) buf[i] = 0.f;
        head = 0; count = 0; ticks = 0; s1 = c1 = s2 = c2 = 0.f;
    }
    __device__ __forceinline__ void kahan_add(float y, float &s, float &c) {
        float t = __fadd_rn(s, __fsub_rn(y, c));
        c = __fsub_rn(__fsub_rn(t, s), __fsub_rn(y, c));
        s = t;
    }
    __device__ __forceinline__ void push(float x) {
        if (count < 4) {
            buf[head] = x; head++; if (head == 4) head = 0; count++;
            kahan_add(x, s1, c1);
            kahan_add(__fmul_rn((float)count, x), s2, c2);
        } else {
            const float old = buf[head]; buf[head] = x; head++; if (head == 4) head = 0;
            const float s1_old = s1;
            kahan_add(__fmaf_rn(4.f, x, -s1_old), s2, c2);
            kahan_add(x, s1, c1);
            kahan_add(-old, s1, c1);
            ticks++;
            if ((ticks & 0x3FF) == 0) {
                float ns1 = 0.f, nc1 = 0.f, ns2 = 0.f, nc2 = 0.f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int idx = (head + i) % 4;
                    const float v = buf[idx];
                    kahan_add(v, ns1, nc1);
                    kahan_add(__fmul_rn((float)(i + 1), v), ns2, nc2);
                }
                s1 = ns1; c1 = nc1; s2 = ns2; c2 = nc2;
            }
        }
    }
    __device__ __forceinline__ float value() const { return __fmul_rn(s2, 0.1f); }
};

static __device__ __forceinline__ void pma_batch_core(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (threadIdx.x != 0) return; // sequential scan per combo

    const float nan_f = nan32();
    if (series_len <= 0) return;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const int warm_predict = first_valid + 6; // first 7 prices
    const int warm_trigger = first_valid + 9; // predict LWMA4

    float* predict_row = out_predict + combo * series_len;
    float* trigger_row = out_trigger + combo * series_len;

    // NaN prefixes
    {
        int stop = (series_len < warm_predict) ? series_len : warm_predict;
        for (int i = 0; i < stop; ++i) predict_row[i] = nan_f;
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int i = 0; i < stop; ++i) trigger_row[i] = nan_f;
    }

    if (warm_predict >= series_len) return;

    // Seed WMA1 accumulators at j0 = first_valid + 6
    const int j0 = warm_predict;

    lwma7_recur_f32 wma1; wma1.init();
    float seed7[7];
    #pragma unroll
    for (int k = 0; k < 7; ++k) seed7[k] = prices[j0 - 6 + k];
    wma1.seed_from7(seed7);

    // WMA2 over WMA1 using the same recurrence + a 7-ring of w1 values
    lwma7_recur_f32 wma2; wma2.init();

    // Trigger LWMA4 over predict
    lwma4_recur_f32 trig; trig.init();

    float w1 = wma1.value();
    wma2.push(w1);
    float w2 = wma2.value();
    float pr = 2.f * w1 - w2;
    predict_row[j0] = pr;
    trig.push(pr);
    // trigger_row[j0] stays NaN per scalar semantics

    for (int j = j0 + 1; j < series_len; ++j) {
        const float x_new = prices[j];
        wma1.push(x_new);
        w1 = wma1.value();

        wma2.push(w1);
        w2 = wma2.value();

        pr = 2.f * w1 - w2;
        predict_row[j] = pr;

        trig.push(pr);
        if (j >= warm_trigger) {
            trigger_row[j] = trig.value();
        } else {
            trigger_row[j] = nan_f;
        }
    }
}

extern "C" __global__ void pma_batch_f32(const float* __restrict__ prices,
                                          int series_len,
                                          int n_combos,
                                          int first_valid,
                                          float* __restrict__ out_predict,
                                          float* __restrict__ out_trigger) {
    pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

// Tiled batch variants (symbol parity with ALMA wrappers). Tiles are ignored
// because this is a strictly sequential recurrence per combo.
extern "C" __global__ void pma_batch_tiled_f32_tile128(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger) {
    pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

extern "C" __global__ void pma_batch_tiled_f32_tile256(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger) {
    pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

static __device__ __forceinline__ void pma_many_series_core(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm)
{
    const int series = (blockIdx.y > 0) ? (blockIdx.x * blockDim.y + threadIdx.y)
                                        : (blockIdx.x);
    // We dispatch kernels with (grid.x, blockDim.y) layouts in wrappers; support both 1D/2D.
    if (series >= num_series) return;
    if (threadIdx.x != 0) return;

    const int stride = num_series;
    const float nan_f = nan32();

    int fv = first_valids ? first_valids[series] : 0;
    if (fv < 0) fv = 0;
    if (fv >= series_len) return;

    const int warm_predict = fv + 6;
    const int warm_trigger = fv + 9;

    // NaN prefixes
    {
        int stop = (series_len < warm_predict) ? series_len : warm_predict;
        for (int row = 0; row < stop; ++row) out_predict_tm[row * stride + series] = nan_f;
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) out_trigger_tm[row * stride + series] = nan_f;
    }

    if (warm_predict >= series_len) return;

    // Seed at j0
    const int j0 = warm_predict;
    lwma7_recur_f32 wma1; wma1.init();
    float seed7tm[7];
    #pragma unroll
    for (int k = 0; k < 7; ++k) seed7tm[k] = prices_tm[(j0 - 6 + k) * stride + series];
    wma1.seed_from7(seed7tm);
    lwma7_recur_f32 wma2; wma2.init();
    lwma4_recur_f32 trig; trig.init();

    float w1 = wma1.value();
    wma2.push(w1);
    float w2 = wma2.value();
    float pr = 2.f * w1 - w2;
    out_predict_tm[j0 * stride + series] = pr;
    trig.push(pr);

    for (int row = j0 + 1; row < series_len; ++row) {
        const float x_new = prices_tm[row * stride + series];
        wma1.push(x_new);
        w1 = wma1.value();
        wma2.push(w1);
        w2 = wma2.value();
        pr = 2.f * w1 - w2;
        const int idx = row * stride + series;
        out_predict_tm[idx] = pr;
        trig.push(pr);
        if (row >= warm_trigger) {
            out_trigger_tm[idx] = trig.value();
        } else {
            out_trigger_tm[idx] = nan_f;
        }
    }
}

extern "C" __global__ void pma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    pma_many_series_core(prices_tm, num_series, series_len, first_valids, out_predict_tm, out_trigger_tm);
}

extern "C" __global__ void pma_ms1p_tiled_f32_tx1_ty2(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    pma_many_series_core(prices_tm, num_series, series_len, first_valids, out_predict_tm, out_trigger_tm);
}

extern "C" __global__ void pma_ms1p_tiled_f32_tx1_ty4(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    pma_many_series_core(prices_tm, num_series, series_len, first_valids, out_predict_tm, out_trigger_tm);
}
