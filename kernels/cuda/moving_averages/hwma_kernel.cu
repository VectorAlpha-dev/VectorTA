// CUDA kernels for the Holt-Winters Moving Average (HWMA).
//
// Drop-in optimized versions focusing on:
// - Arithmetic strength reduction + FMA (y + beta*(x - y))
// - Hoisting loop-invariant constants
// - Grid-stride loops and __launch_bounds__ occupancy hints
// - Pointer arithmetic to avoid inner-loop multiplies

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>  // fma()

// Canonical quiet NaN (qNaN)
static __device__ __forceinline__ float qnan_f() { return __int_as_float(0x7fffffff); }

static __device__ __forceinline__ int clamp_int(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// --------------------------------------------
// 1) single-series × many-parameter sweep
// --------------------------------------------
extern "C" __global__
void hwma_batch_f32(const double* __restrict__ prices,
                    const double* __restrict__ nas,
                    const double* __restrict__ nbs,
                    const double* __restrict__ ncs,
                    int first_valid,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    int first = clamp_int(first_valid, 0, series_len);

    const double dna = nas[combo];
    const double dnb = nbs[combo];
    const double dnc = ncs[combo];

    const int base = combo * series_len;
    const float nan_f = qnan_f();
    for (int t = 0; t < first; ++t) { out[base + t] = nan_f; }
    if (first >= series_len) { return; }

    // Compute in FP64 to reduce accumulation error; store as FP32.
    double f = prices[first];
    double v = 0.0;
    double a = 0.0;
    const double dh  = 0.5;

    for (int t = first; t < series_len; ++t) {
        const double price = prices[t];
        const double s_prev = dh * a + (f + v);
        const double f_new = dna * price + (1.0 - dna) * s_prev;
        const double v_new = dnb * (f_new - f) + (1.0 - dnb) * (v + a);
        const double a_new = dnc * (v_new - v) + (1.0 - dnc) * a;
        const double s_new = dh * a_new + (f_new + v_new);
        out[base + t] = (float)s_new;
        f = f_new; v = v_new; a = a_new;
    }
}

// --------------------------------------------
// 2) many-series × one-parameter (time-major)
// --------------------------------------------
extern "C" __global__ __launch_bounds__(256, 2)
void hwma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     float na,
                                     float nb,
                                     float nc,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    for (int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
         series_idx < num_series;
         series_idx += blockDim.x * gridDim.x)
    {
        if (series_len <= 0) return;

        const int stride = num_series;

        int first = clamp_int(first_valids[series_idx], 0, series_len);

        // Pre-fill leading NaNs (time-major indexing)
        const float nan_f = qnan_f();
        int idx = series_idx;
        for (int t = 0; t < first; ++t, idx += stride) {
            out_tm[idx] = nan_f;
        }
        if (first >= series_len) continue;

        // Invariants cast once
        const double dna = (double)na;
        const double dnb = (double)nb;
        const double dnc = (double)nc;
        const double dh  = 0.5;

        // Seed from first valid element
        int first_idx = first * stride + series_idx;
        double f = (double)prices_tm[first_idx];
        double v = 0.0;
        double a = 0.0;

        // Advance idx to first write position
        idx = first_idx;

        // Main recurrence with pointer-stride
        for (int t = first; t < series_len; ++t, idx += stride) {
            const double price = (double)prices_tm[idx];

            double s_prev = (f + v) + dh * a;

            double nap = dna * price;
            double f_new = fma((1.0 - dna), s_prev, nap);

            double vy    = v + a;
            double nbd   = dnb * (f_new - f);
            double v_new = fma((1.0 - dnb), vy, nbd);

            double ncv   = dnc * (v_new - v);
            double a_new = fma((1.0 - dnc), a, ncv);

            double s_new = (f_new + v_new) + dh * a_new;

            out_tm[idx] = (float)s_new;

            f = f_new;
            v = v_new;
            a = a_new;
        }
    }
}
