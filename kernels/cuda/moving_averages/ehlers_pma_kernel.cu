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

static __device__ __forceinline__ void ehlers_pma_batch_core(const float* __restrict__ prices,
                                                             int series_len,
                                                             int n_combos,
                                                             int first_valid,
                                                             float* __restrict__ out_predict,
                                                             float* __restrict__ out_trigger) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (threadIdx.x != 0) return;

    const float nan_f = nan32();
    float* predict_row = out_predict + combo * series_len;
    float* trigger_row = out_trigger + combo * series_len;
    for (int i = 0; i < series_len; ++i) { predict_row[i] = nan_f; trigger_row[i] = nan_f; }

    if (series_len <= 0) return;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = first_valid + 13;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) return;

    float wma1_ring[7];
    ff predict_ring[4];
    for (int i = 0; i < 7; ++i) { wma1_ring[i] = nan32(); }
    for (int i = 0; i < 4; ++i) { predict_ring[i].hi = nan32(); predict_ring[i].lo = 0.f; }
    int ring_head = 0;
    int predict_head = 0;

    for (int idx = first_valid; idx < series_len; ++idx) {
        float wma1_val = nan32();
        if (idx >= warm_wma1) { wma1_val = wma7_from_prices_f32(prices, idx); }
        wma1_ring[ring_head] = wma1_val;
        ring_head = (ring_head + 1) % 7;
        if (idx >= warm_wma2) {
            const float wma2_val = wma7_from_ring_f32(wma1_ring, ring_head);
            const float current_wma1 = wma1_ring[(ring_head + 6) % 7];
            const float two_m = __fadd_rn(current_wma1, current_wma1);
            const ff pred = two_sum(two_m, -wma2_val);
            const float predict_val = __fadd_rn(pred.hi, pred.lo);
            predict_row[idx] = predict_val;
            predict_ring[predict_head] = pred;
            predict_head = (predict_head + 1) % 4;
            if (idx >= warm_trigger) {
                const float trigger_val = trigger4_from_ff_ring(predict_ring, predict_head);
                trigger_row[idx] = trigger_val;
            }
        }
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
    if (series >= num_series) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    const int stride = num_series;
    const float nan_f = nan32();
    for (int row = 0; row < series_len; ++row) {
        const int idx = row * stride + series;
        out_predict_tm[idx] = nan_f;
        out_trigger_tm[idx] = nan_f;
    }

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) {
        first_valid = 0;
    }
    if (first_valid >= series_len) {
        return;
    }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = first_valid + 13;
    const int warm_trigger = warm_wma2 + 3;

    if (warm_wma1 >= series_len) {
        return;
    }

    float wma1_ring[7];
    ff predict_ring[4];
    for (int i = 0; i < 7; ++i) { wma1_ring[i] = nan32(); }
    for (int i = 0; i < 4; ++i) { predict_ring[i].hi = nan32(); predict_ring[i].lo = 0.f; }

    int ring_head = 0;
    int predict_head = 0;

    for (int row = first_valid; row < series_len; ++row) {
        float wma1_val = nan32();
        if (row >= warm_wma1) {
            const int idx = row * stride + series;
            wma1_val = wma7_from_prices_tm_f32(prices_tm, idx, stride);
        }
        wma1_ring[ring_head] = wma1_val;
        ring_head = (ring_head + 1) % 7;

        if (row >= warm_wma2) {
            const float wma2_val = wma7_from_ring_f32(wma1_ring, ring_head);
            const float current_wma1 = wma1_ring[(ring_head + 6) % 7];
            const float two_m = __fadd_rn(current_wma1, current_wma1);
            const ff pred = two_sum(two_m, -wma2_val);
            const float predict_val = __fadd_rn(pred.hi, pred.lo);
            const int idx = row * stride + series;
            out_predict_tm[idx] = predict_val;

            predict_ring[predict_head] = pred;
            predict_head = (predict_head + 1) % 4;

            if (row >= warm_trigger) {
                const float trigger_val = trigger4_from_ff_ring(predict_ring, predict_head);
                out_trigger_tm[idx] = trigger_val;
            }
        }
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
    for (int row = 0; row < series_len; ++row) {
        const int idx = row * stride + series;
        out_predict_tm[idx] = nan_f;
        out_trigger_tm[idx] = nan_f;
    }

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) { first_valid = 0; }
    if (first_valid >= series_len) { return; }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = first_valid + 13;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) { return; }

    float wma1_ring[7];
    ff predict_ring[4];
    for (int i = 0; i < 7; ++i) { wma1_ring[i] = nan32(); }
    for (int i = 0; i < 4; ++i) { predict_ring[i].hi = nan32(); predict_ring[i].lo = 0.f; }
    int ring_head = 0;
    int predict_head = 0;

    for (int row = first_valid; row < series_len; ++row) {
        float wma1_val = nan32();
        if (row >= warm_wma1) {
            const int idx = row * stride + series;
            wma1_val = wma7_from_prices_tm_f32(prices_tm, idx, stride);
        }
        wma1_ring[ring_head] = wma1_val;
        ring_head = (ring_head + 1) % 7;

        if (row >= warm_wma2) {
            const float wma2_val = wma7_from_ring_f32(wma1_ring, ring_head);
            const float current_wma1 = wma1_ring[(ring_head + 6) % 7];
            const float two_m = __fadd_rn(current_wma1, current_wma1);
            const ff pred = two_sum(two_m, -wma2_val);
            const float predict_val = __fadd_rn(pred.hi, pred.lo);
            const int idx = row * stride + series;
            out_predict_tm[idx] = predict_val;

            predict_ring[predict_head] = pred;
            predict_head = (predict_head + 1) % 4;

            if (row >= warm_trigger) {
                const float trigger_val = trigger4_from_ff_ring(predict_ring, predict_head);
                out_trigger_tm[idx] = trigger_val;
            }
        }
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
    for (int row = 0; row < series_len; ++row) {
        const int idx = row * stride + series;
        out_predict_tm[idx] = nan_f;
        out_trigger_tm[idx] = nan_f;
    }

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) { first_valid = 0; }
    if (first_valid >= series_len) { return; }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = first_valid + 13;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) { return; }

    double wma1_ring[7];
    double predict_ring[4];
    for (int i = 0; i < 7; ++i) { wma1_ring[i] = nan64(); }
    for (int i = 0; i < 4; ++i) { predict_ring[i] = nan64(); }
    int ring_head = 0;
    int predict_head = 0;

    for (int row = first_valid; row < series_len; ++row) {
        double wma1_val = nan64();
        if (row >= warm_wma1) {
            const int idx = row * stride + series;
            const double p1 = static_cast<double>(prices_tm[idx - stride]);
            const double p2 = static_cast<double>(prices_tm[idx - 2 * stride]);
            const double p3 = static_cast<double>(prices_tm[idx - 3 * stride]);
            const double p4 = static_cast<double>(prices_tm[idx - 4 * stride]);
            const double p5 = static_cast<double>(prices_tm[idx - 5 * stride]);
            const double p6 = static_cast<double>(prices_tm[idx - 6 * stride]);
            const double p7 = static_cast<double>(prices_tm[idx - 7 * stride]);
            const double num = 7.0 * p1 + 6.0 * p2 + 5.0 * p3 + 4.0 * p4 + 3.0 * p5 + 2.0 * p6 + 1.0 * p7;
            wma1_val = num / 28.0;
        }
        wma1_ring[ring_head] = wma1_val;
        ring_head = (ring_head + 1) % 7;

        if (row >= warm_wma2) {
            const double wma2_val = wma7_from_ring(wma1_ring, ring_head);
            const double current_wma1 = wma1_ring[(ring_head + 6) % 7];
            const double predict_val = 2.0 * current_wma1 - wma2_val;
            const int idx = row * stride + series;
            out_predict_tm[idx] = static_cast<float>(predict_val);

            predict_ring[predict_head] = predict_val;
            predict_head = (predict_head + 1) % 4;

            if (row >= warm_trigger) {
                const double p0 = predict_ring[(predict_head + 0) % 4];
                const double p1 = predict_ring[(predict_head + 1) % 4];
                const double p2 = predict_ring[(predict_head + 2) % 4];
                const double p3 = predict_ring[(predict_head + 3) % 4];
                const double trigger_val = (4.0 * p3 + 3.0 * p2 + 2.0 * p1 + 1.0 * p0) / 10.0;
                out_trigger_tm[idx] = static_cast<float>(trigger_val);
            }
        }
    }
}
