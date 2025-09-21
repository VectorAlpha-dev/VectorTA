// CUDA kernels for the Ehlers Predictive Moving Average (PMA).
//
// Each block processes either a single "parameter" combination (the batch
// kernel) or a single series (the many-series variant). Arithmetic executes in
// FP64 to mirror the scalar reference implementation, while device storage stays
// FP32 to interoperate with the zero-copy wrappers used across the project.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ double nan64() {
    return nan("");
}

static __device__ __forceinline__ float nan32() {
    return nanf("");
}

static __device__ __forceinline__ double wma7_from_prices(const float* __restrict__ prices,
                                                          int idx) {
    const double p1 = static_cast<double>(prices[idx - 1]);
    const double p2 = static_cast<double>(prices[idx - 2]);
    const double p3 = static_cast<double>(prices[idx - 3]);
    const double p4 = static_cast<double>(prices[idx - 4]);
    const double p5 = static_cast<double>(prices[idx - 5]);
    const double p6 = static_cast<double>(prices[idx - 6]);
    const double p7 = static_cast<double>(prices[idx - 7]);
    const double num = 7.0 * p1 + 6.0 * p2 + 5.0 * p3 + 4.0 * p4 + 3.0 * p5 + 2.0 * p6 + 1.0 * p7;
    return num / 28.0;
}

static __device__ __forceinline__ double wma7_from_ring(const double ring[7], int head) {
    const double v0 = ring[(head + 6) % 7];
    const double v1 = ring[(head + 5) % 7];
    const double v2 = ring[(head + 4) % 7];
    const double v3 = ring[(head + 3) % 7];
    const double v4 = ring[(head + 2) % 7];
    const double v5 = ring[(head + 1) % 7];
    const double v6 = ring[(head + 0) % 7];
    const double num = 7.0 * v0 + 6.0 * v1 + 5.0 * v2 + 4.0 * v3 + 3.0 * v4 + 2.0 * v5 + 1.0 * v6;
    return num / 28.0;
}

extern "C" __global__ void ehlers_pma_batch_f32(const float* __restrict__ prices,
                                                 int series_len,
                                                 int n_combos,
                                                 int first_valid,
                                                 float* __restrict__ out_predict,
                                                 float* __restrict__ out_trigger) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    const float nan_f = nan32();
    float* predict_row = out_predict + combo * series_len;
    float* trigger_row = out_trigger + combo * series_len;
    for (int i = 0; i < series_len; ++i) {
        predict_row[i] = nan_f;
        trigger_row[i] = nan_f;
    }

    if (series_len <= 0) {
        return;
    }
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

    double wma1_ring[7];
    double predict_ring[4];
    for (int i = 0; i < 7; ++i) {
        wma1_ring[i] = nan64();
    }
    for (int i = 0; i < 4; ++i) {
        predict_ring[i] = nan64();
    }

    int ring_head = 0;
    int predict_head = 0;

    for (int idx = first_valid; idx < series_len; ++idx) {
        double wma1_val = nan64();
        if (idx >= warm_wma1) {
            wma1_val = wma7_from_prices(prices, idx);
        }
        wma1_ring[ring_head] = wma1_val;
        ring_head = (ring_head + 1) % 7;

        if (idx >= warm_wma2) {
            const double wma2_val = wma7_from_ring(wma1_ring, ring_head);
            const double current_wma1 = wma1_ring[(ring_head + 6) % 7];
            const double predict_val = 2.0 * current_wma1 - wma2_val;
            predict_row[idx] = static_cast<float>(predict_val);

            predict_ring[predict_head] = predict_val;
            predict_head = (predict_head + 1) % 4;

            if (idx >= warm_trigger) {
                const double p0 = predict_ring[(predict_head + 0) % 4];
                const double p1 = predict_ring[(predict_head + 1) % 4];
                const double p2 = predict_ring[(predict_head + 2) % 4];
                const double p3 = predict_ring[(predict_head + 3) % 4];
                const double trigger_val = (4.0 * p3 + 3.0 * p2 + 2.0 * p1 + 1.0 * p0) / 10.0;
                trigger_row[idx] = static_cast<float>(trigger_val);
            }
        }
    }
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

    double wma1_ring[7];
    double predict_ring[4];
    for (int i = 0; i < 7; ++i) {
        wma1_ring[i] = nan64();
    }
    for (int i = 0; i < 4; ++i) {
        predict_ring[i] = nan64();
    }

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
            const double num =
                7.0 * p1 + 6.0 * p2 + 5.0 * p3 + 4.0 * p4 + 3.0 * p5 + 2.0 * p6 + 1.0 * p7;
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
