// CUDA kernels for the Ehlers Predictive Moving Average (PMA).
//
// Each block processes either a single "parameter" combination (the batch
// kernel) or a single series (the many-series variant).
//
// Implementation strategy (FP32 throughput + accuracy):
// - Use FP32 FMA intrinsics and compensated summation (Kahan) for the 7-tap and
//   4-tap LWMAs.
// - Use a float-float TwoSum for 2*wma1 - wma2 to reduce cancellation error.
// - Keep device storage FP32 for interoperability; arithmetic stays FP32 with
//   compensation. This maintains high throughput while tracking the f64 scalar
//   reference within tight tolerances.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float nan32() {
    return nanf("");
}

// --- FP32 compensated building blocks ---------------------------------------
static __device__ __forceinline__ void kahan_add(float y, float& s, float& c) {
    float t = __fadd_rn(s, __fsub_rn(y, c));
    c = __fsub_rn(__fsub_rn(t, s), __fsub_rn(y, c));
    s = t;
}

static __device__ __forceinline__ void kahan_add_prod(float a, float b, float& s, float& c) {
    float p = __fmul_rn(a, b);
    float r = __fmaf_rn(a, b, -p);
    kahan_add(p, s, c);
    kahan_add(r, s, c);
}

struct ff {
    float hi;
    float lo;
};

static __device__ __forceinline__ ff two_sum(float a, float b) {
    ff res;
    float s = __fadd_rn(a, b);
    float bb = __fsub_rn(s, a);
    float e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
    res.hi = s;
    res.lo = e;
    return res;
}

// --- Rolling LWMA helpers with compensated updates --------------------------
// Maintains LWMA over floats with O(1) updates using S1/S2 and Kahan sums.
struct lwma7_f32 {
    float buf[7];
    int   head;
    int   count;
    int   ticks;
    float s1, c1;   // simple sum
    float s2, c2;   // weighted sum (weights 1..7 oldest..newest)

    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < 7; ++i) buf[i] = 0.f;
        head = 0; count = 0; ticks = 0; s1 = c1 = 0.f; s2 = c2 = 0.f;
    }

    __device__ __forceinline__ void push(float x) {
        if (count < 7) {
            buf[head] = x;
            head++; if (head == 7) head = 0;
            count++;
            // Initial fill: first inserted becomes oldest -> weight = count
            kahan_add(x, s1, c1);
            kahan_add(__fmul_rn(static_cast<float>(count), x), s2, c2);
        } else {
            const float old = buf[head];          // oldest element
            buf[head] = x;
            head++; if (head == 7) head = 0;

            const float s1_old = s1;              // S2' = S2 + (N*x_new - S1_old)
            kahan_add(__fmaf_rn(7.f, x, -s1_old), s2, c2);

            kahan_add(x, s1, c1);                 // S1' = S1 + x_new - x_old
            kahan_add(-old, s1, c1);

            // Periodic renormalization to curb drift on long runs
            ticks++;
            if ((ticks & 0x3FF) == 0) { // every 1024 updates
                float ns1 = 0.f, nc1 = 0.f;
                float ns2 = 0.f, nc2 = 0.f;
#pragma unroll
                for (int i = 0; i < 7; ++i) {
                    const int idx = (head + i) % 7;   // oldest at i=0
                    const float v = buf[idx];
                    kahan_add(v, ns1, nc1);
                    kahan_add(__fmul_rn(static_cast<float>(i + 1), v), ns2, nc2);
                }
                s1 = ns1; c1 = nc1; s2 = ns2; c2 = nc2;
            }
        }
    }

    __device__ __forceinline__ bool full() const { return count >= 7; }
    __device__ __forceinline__ float value() const { return __fmul_rn(s2, 1.0f / 28.0f); }
    __device__ __forceinline__ float newest() const {
        int idx = head - 1; if (idx < 0) idx += 7; return buf[idx];
    }
};

// 4-tap LWMA over (hi,lo) using two sets of compensated S1/S2
struct lwma4_ff {
    ff    buf[4];
    int   head;
    int   count;
    int   ticks;
    // hi parts
    float s1h, c1h, s2h, c2h;
    // low parts
    float s1l, c1l, s2l, c2l;

    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < 4; ++i) { buf[i].hi = 0.f; buf[i].lo = 0.f; }
        head = 0; count = 0; ticks = 0;
        s1h = c1h = s2h = c2h = 0.f;
        s1l = c1l = s2l = c2l = 0.f;
    }

    __device__ __forceinline__ void push(const ff& p) {
        if (count < 4) {
            buf[head] = p;
            head++; if (head == 4) head = 0;
            count++;

            kahan_add(p.hi, s1h, c1h);
            kahan_add(__fmul_rn(static_cast<float>(count), p.hi), s2h, c2h);
            kahan_add(p.lo, s1l, c1l);
            kahan_add(__fmul_rn(static_cast<float>(count), p.lo), s2l, c2l);
        } else {
            const ff old = buf[head];
            buf[head] = p;
            head++; if (head == 4) head = 0;

            const float s1h_old = s1h, s1l_old = s1l;
            // hi parts
            kahan_add(__fmaf_rn(4.f, p.hi, -s1h_old), s2h, c2h);
            kahan_add(p.hi, s1h, c1h);
            kahan_add(-old.hi, s1h, c1h);
            // low parts
            kahan_add(__fmaf_rn(4.f, p.lo, -s1l_old), s2l, c2l);
            kahan_add(p.lo, s1l, c1l);
            kahan_add(-old.lo, s1l, c1l);

            // Periodic renormalization to curb drift on long runs
            ticks++;
            if ((ticks & 0x3FF) == 0) { // every 1024 updates
                float ns1h = 0.f, nc1h = 0.f, ns2h = 0.f, nc2h = 0.f;
                float ns1l = 0.f, nc1l = 0.f, ns2l = 0.f, nc2l = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int idx = (head + i) % 4;   // oldest at i=0
                    const ff v = buf[idx];
                    const float w = static_cast<float>(i + 1);
                    // hi
                    kahan_add(v.hi, ns1h, nc1h);
                    kahan_add(__fmul_rn(w, v.hi), ns2h, nc2h);
                    // lo
                    kahan_add(v.lo, ns1l, nc1l);
                    kahan_add(__fmul_rn(w, v.lo), ns2l, nc2l);
                }
                s1h = ns1h; c1h = nc1h; s2h = ns2h; c2h = nc2h;
                s1l = ns1l; c1l = nc1l; s2l = ns2l; c2l = nc2l;
            }
        }
    }

    __device__ __forceinline__ bool full() const { return count >= 4; }
    __device__ __forceinline__ float value() const {
        // (S2_hi + S2_lo) / (1+2+3+4) = / 10
        return __fmul_rn(__fadd_rn(s2h, s2l), 0.1f);
    }
};

// 7-tap LWMA from contiguous prices (weights 7..1)/28, using FP32 Kahan+FMA
static __device__ __forceinline__ float wma7_from_prices_f32(const float* __restrict__ prices,
                                                             int idx) {
    float s = 0.f, c = 0.f;
#pragma unroll
    for (int k = 1, w = 7; k <= 7; ++k, --w) {
        kahan_add_prod(static_cast<float>(w), prices[idx - k], s, c);
    }
    return __fmul_rn(s, 1.0f / 28.0f);
}

// 7-tap LWMA from a 7-ring of WMA1 values, head points to next write position
static __device__ __forceinline__ float wma7_from_ring_f32(const float ring[7], int head) {
    float s = 0.f, c = 0.f;
    const float v0 = ring[(head + 6) % 7];
    const float v1 = ring[(head + 5) % 7];
    const float v2 = ring[(head + 4) % 7];
    const float v3 = ring[(head + 3) % 7];
    const float v4 = ring[(head + 2) % 7];
    const float v5 = ring[(head + 1) % 7];
    const float v6 = ring[(head + 0) % 7];
    kahan_add_prod(7.f, v0, s, c);
    kahan_add_prod(6.f, v1, s, c);
    kahan_add_prod(5.f, v2, s, c);
    kahan_add_prod(4.f, v3, s, c);
    kahan_add_prod(3.f, v4, s, c);
    kahan_add_prod(2.f, v5, s, c);
    kahan_add_prod(1.f, v6, s, c);
    return __fmul_rn(s, 1.0f / 28.0f);
}

// 7-tap LWMA from time-major prices using stride (weights 7..1)/28
static __device__ __forceinline__ float wma7_from_prices_tm_f32(const float* __restrict__ prices_tm,
                                                                int idx, int stride) {
    float s = 0.f, c = 0.f;
    kahan_add_prod(7.f, prices_tm[idx - stride], s, c);
    kahan_add_prod(6.f, prices_tm[idx - 2 * stride], s, c);
    kahan_add_prod(5.f, prices_tm[idx - 3 * stride], s, c);
    kahan_add_prod(4.f, prices_tm[idx - 4 * stride], s, c);
    kahan_add_prod(3.f, prices_tm[idx - 5 * stride], s, c);
    kahan_add_prod(2.f, prices_tm[idx - 6 * stride], s, c);
    kahan_add_prod(1.f, prices_tm[idx - 7 * stride], s, c);
    return __fmul_rn(s, 1.0f / 28.0f);
}

// 4-tap trigger LWMA over ff ring (weights 4..1)/10
static __device__ __forceinline__ float trigger4_from_ff_ring(const ff pr[4], int head) {
    float s = 0.f, c = 0.f;
    const ff p0 = pr[(head + 0) % 4];
    const ff p1 = pr[(head + 1) % 4];
    const ff p2 = pr[(head + 2) % 4];
    const ff p3 = pr[(head + 3) % 4];
    // accumulate hi parts
    kahan_add_prod(1.f, p0.hi, s, c);
    kahan_add_prod(2.f, p1.hi, s, c);
    kahan_add_prod(3.f, p2.hi, s, c);
    kahan_add_prod(4.f, p3.hi, s, c);
    // fold in low parts with same weights
    kahan_add_prod(1.f, p0.lo, s, c);
    kahan_add_prod(2.f, p1.lo, s, c);
    kahan_add_prod(3.f, p2.lo, s, c);
    kahan_add_prod(4.f, p3.lo, s, c);
    return __fmul_rn(s, 0.1f);
}

static __device__ __forceinline__ void ehlers_pma_batch_core(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (threadIdx.x != 0) return;

    const float nan_f = nan32();
    if (series_len <= 0) return;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    // Warmup thresholds (unchanged semantics)
    const int warm_wma1    = first_valid + 7;
    const int warm_wma2    = first_valid + 13;
    const int warm_trigger = warm_wma2 + 3;

    float* predict_row = out_predict + combo * series_len;
    float* trigger_row = out_trigger + combo * series_len;

    // Write only the minimal NaN prefix; rest will be written when available
    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int i = 0; i < stop; ++i) { predict_row[i] = nan_f; }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int i = 0; i < stop; ++i) { trigger_row[i] = nan_f; }
    }

    // If we can never warm up, exit early
    if (warm_wma1 >= series_len) return;

    // Rolling LWMAs
    lwma7_f32 price_w7;  price_w7.init();   // last 7 prices (previous bars)
    lwma7_f32 wma1_w7;   wma1_w7.init();    // last 7 wma1 values
    lwma4_ff  trig_w4;   trig_w4.init();    // last 4 (hi,lo) predictive values

    // Walk forward. At loop entry, price_w7 holds the previous 0..7 prices (idx-1..idx-7).
    for (int idx = first_valid; idx < series_len; ++idx) {

        // 1) wma1 from previous 7 prices (if full)
        float wma1_val = nan_f;
        if (price_w7.full()) {
            wma1_val = price_w7.value();    // (weights 1..7 oldest..newest) / 28
        }

        // 2) Update WMA-of-WMA and compute predict when ready
        if (idx >= warm_wma1) {
            wma1_w7.push(wma1_val);

            if (wma1_w7.full()) {
                const float wma2_val = wma1_w7.value();
                const float current_wma1 = wma1_w7.newest();
                const float two_m = __fadd_rn(current_wma1, current_wma1);
                const ff     pred  = two_sum(two_m, -wma2_val);
                predict_row[idx]   = __fadd_rn(pred.hi, pred.lo);

                // 3) Update trigger (4-tap LWMA of pred) and write when ready
                trig_w4.push(pred);
                if (trig_w4.full() && idx >= warm_trigger) {
                    trigger_row[idx] = trig_w4.value();
                }
            }
        }

        // 4) Feed the current price so next idx sees the prior-7 window
        const float p_new = prices[idx];
        price_w7.push(p_new);
    }
}

extern "C" __global__ void ehlers_pma_batch_f32(const float* __restrict__ prices,
                                                 int series_len,
                                                 int n_combos,
                                                 int first_valid,
                                                 float* __restrict__ out_predict,
                                                 float* __restrict__ out_trigger) {
    ehlers_pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

// Tiled batch variants (naming parity with ALMA). For PMA the computation along
// time is inherently sequential due to the second-stage WMA and trigger over
// previous outputs. These variants therefore map one combo per block and use
// only thread 0 for the sequential walk while preserving the symbol names used
// by the Rust wrapper's policy/introspection. The tile size is ignored here.
extern "C" __global__ void ehlers_pma_batch_tiled_f32_tile128(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger) {
    ehlers_pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

extern "C" __global__ void ehlers_pma_batch_tiled_f32_tile256(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger) {
    ehlers_pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

extern "C" __global__ void ehlers_pma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const float nan_f = nan32();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;     // == first_valid + 13
    const int warm_trigger = warm_wma2 + 3;

    // Minimal NaN prefix only
    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    if (first_valid + 7 >= series_len) return;

    lwma7_f32 price_w7; price_w7.init();
    lwma7_f32 wma1_w7;  wma1_w7.init();
    lwma4_ff  trig_w4;  trig_w4.init();

    for (int row = first_valid; row < series_len; ++row) {
        float wma1_val = nan_f;
        if (price_w7.full()) { wma1_val = price_w7.value(); }

        if (row >= warm_wma1) {
            wma1_w7.push(wma1_val);

            if (wma1_w7.full()) {
                const float wma2_val = wma1_w7.value();
                const float current_wma1 = wma1_w7.newest();
                const float two_m = __fadd_rn(current_wma1, current_wma1);
                const ff pred = two_sum(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __fadd_rn(pred.hi, pred.lo);

                trig_w4.push(pred);
                if (trig_w4.full() && row >= first_valid + 16) { // warm_trigger
                    out_trigger_tm[idx] = trig_w4.value();
                }
            }
        }

        // push current price sample for this series
        const int pidx = row * stride + series;
        price_w7.push(prices_tm[pidx]);
    }
}

// 2D tiled many-series variants. We parallelize across the series dimension
// (ty threads per block handle distinct series). Each thread still walks time
// sequentially for its assigned series to preserve the dependency across the
// internal WMA-of-WMA and trigger stages. TX is fixed to 1 for PMA.
extern "C" __global__ void ehlers_pma_ms1p_tiled_f32_tx1_ty2(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    int series0 = static_cast<int>(blockIdx.x) * 2;
    int local = static_cast<int>(threadIdx.y);
    int series = series0 + local;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const float nan_f = nan32();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) { first_valid = 0; }
    if (first_valid >= series_len) { return; }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) { return; }

    // Minimal NaN prefix only
    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    lwma7_f32 price_w7; price_w7.init();
    lwma7_f32 wma1_w7;  wma1_w7.init();
    lwma4_ff  trig_w4;  trig_w4.init();

    for (int row = first_valid; row < series_len; ++row) {
        float wma1_val = nan_f;
        if (price_w7.full()) { wma1_val = price_w7.value(); }

        if (row >= warm_wma1) {
            wma1_w7.push(wma1_val);

            if (wma1_w7.full()) {
                const float wma2_val = wma1_w7.value();
                const float current_wma1 = wma1_w7.newest();
                const float two_m = __fadd_rn(current_wma1, current_wma1);
                const ff pred = two_sum(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __fadd_rn(pred.hi, pred.lo);

                trig_w4.push(pred);
                if (trig_w4.full() && row >= first_valid + 16) {
                    out_trigger_tm[idx] = trig_w4.value();
                }
            }
        }

        const int pidx = row * stride + series;
        price_w7.push(prices_tm[pidx]);
    }
}

extern "C" __global__ void ehlers_pma_ms1p_tiled_f32_tx1_ty4(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    int series0 = static_cast<int>(blockIdx.x) * 4;
    int local = static_cast<int>(threadIdx.y);
    int series = series0 + local;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const float nan_f = nan32();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) { first_valid = 0; }
    if (first_valid >= series_len) { return; }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) { return; }

    // Minimal NaN prefix only
    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    lwma7_f32 price_w7; price_w7.init();
    lwma7_f32 wma1_w7;  wma1_w7.init();
    lwma4_ff  trig_w4;  trig_w4.init();

    for (int row = first_valid; row < series_len; ++row) {
        float wma1_val = nan_f;
        if (price_w7.full()) { wma1_val = price_w7.value(); }

        if (row >= warm_wma1) {
            wma1_w7.push(wma1_val);

            if (wma1_w7.full()) {
                const float wma2_val = wma1_w7.value();
                const float current_wma1 = wma1_w7.newest();
                const float two_m = __fadd_rn(current_wma1, current_wma1);
                const ff pred = two_sum(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __fadd_rn(pred.hi, pred.lo);

                trig_w4.push(pred);
                if (trig_w4.full() && row >= first_valid + 16) {
                    out_trigger_tm[idx] = trig_w4.value();
                }
            }
        }

        const int pidx = row * stride + series;
        price_w7.push(prices_tm[pidx]);
    }
}
