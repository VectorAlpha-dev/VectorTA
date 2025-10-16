// CUDA kernels for Mesa Sine Wave (MSW)
//
// Two entry points:
//  - msw_batch_f32: one series × many params (periods array)
//  - msw_many_series_one_param_time_major_f32: many series (time‑major) × one param
//
// Math: windowed projection onto sin/cos bases with step = 2π/period.
// For each output t ≥ warm, compute:
//   rp = Σ_j cos(j*step) * x[t-j]
//   ip = Σ_j sin(j*step) * x[t-j]
//   phase = atan(ip/rp) with branch/quad fixes matching scalar path
//   sine = sin(phase)
//   lead = (sin(phase) + cos(phase)) * √½

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef MSW_BLOCK_X
#define MSW_BLOCK_X 256
#endif

// Shared memory layout helper:
// [cos_weights (P) | sin_weights (P) | tile (blockDim.x + P - 1)]
static __device__ __forceinline__ void msw_build_weights(float* __restrict__ cosw,
                                                         float* __restrict__ sinw,
                                                         int period) {
    const float step = CUDART_PI_F * 2.0f / (float)period;
    // We use angles 0, step, ..., (P-1)*step exactly like the scalar path
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        float ang = step * (float)i;
        float s = __sinf(ang);
        float c = __cosf(ang);
        sinw[i] = s;
        cosw[i] = c;
    }
}

static __device__ __forceinline__ float msw_phase_from_rp_ip(float rp, float ip) {
    // Match scalar: piecewise atan with rp threshold, then quadrant fix and +pi/2, wrap to [0,2pi]
    float phase;
    if (fabsf(rp) > 0.001f) {
        phase = atanf(ip / rp);
    } else {
        phase = (ip < 0.0f ? -CUDART_PI_F : CUDART_PI_F);
    }
    if (rp < 0.0f) phase += CUDART_PI_F;
    phase += 0.5f * CUDART_PI_F;
    if (phase < 0.0f) phase += CUDART_PI_F * 2.0f;
    if (phase > CUDART_PI_F * 2.0f) phase -= CUDART_PI_F * 2.0f;
    return phase;
}

extern "C" __global__
void msw_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) // layout: rows = 2*n_combos, cols = series_len
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    // Shared memory layout
    extern __shared__ float shmem[];
    float* __restrict__ cosw = shmem;
    float* __restrict__ sinw = cosw + period;
    float* __restrict__ tile = sinw + period; // capacity: blockDim.x + period - 1

    const int warm = first_valid + period - 1;
    const int row_sine = combo * 2;
    const int row_lead = row_sine + 1;
    const int base_sine = row_sine * series_len;
    const int base_lead = row_lead * series_len;

    // 1) Warmup NaNs for both outputs
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop = min(warm, series_len);
        for (; t < stop; t += stride) {
            out[base_sine + t] = NAN;
            out[base_lead + t] = NAN;
        }
    }

    // 2) Build sin/cos tables once per block
    msw_build_weights(cosw, sinw, period);
    __syncthreads();

    // 3) Tiled compute for t in [warm, series_len)
    const int stride = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride) {
        const int t_begin = max(base_t, warm);
        const int t_end = min(base_t + blockDim.x - 1, series_len - 1);
        if (t_begin <= t_end) {
            const int tile_in_start = t_begin - (period - 1);
            const int tile_len = (t_end - t_begin + 1) + (period - 1);

            // Cooperative load of contiguous input segment
            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                tile[i] = prices[tile_in_start + i];
            }
            __syncthreads();

            const int t = base_t + threadIdx.x;
            if (t >= t_begin && t <= t_end) {
                const int start = t - t_begin; // offset in tile
                float rp = 0.0f;
                float ip = 0.0f;
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    const float w = tile[start + k];
                    rp = fmaf(cosw[k], w, rp);
                    ip = fmaf(sinw[k], w, ip);
                }
                const float phase = msw_phase_from_rp_ip(rp, ip);
                float s, c;
                s = __sinf(phase);
                c = __cosf(phase);
                out[base_sine + t] = s;
                out[base_lead + t] = (s + c) * 0.70710678118654752440f; // √½
            }
            __syncthreads();
        }
    }
}

extern "C" __global__
void msw_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm, // [time][series]
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids, // per-series
    float* __restrict__ out_tm)            // [rows=series_len][cols=2*num_series] stacked: [sine | lead]
{
    if (period <= 0) return;
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    extern __shared__ float shmem[];
    float* __restrict__ cosw = shmem;
    float* __restrict__ sinw = cosw + period;
    float* __restrict__ tile = sinw + period; // capacity: blockDim.x + period - 1

    const int warm = first_valids[series_idx] + period - 1;
    const int col_sine = series_idx;
    const int col_lead = series_idx + num_series;

    // 1) Warmup NaNs for this series (for both outputs)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop = min(warm, series_len);
        for (; t < stop; t += stride) {
            out_tm[t * (2 * num_series) + col_sine] = NAN;
            out_tm[t * (2 * num_series) + col_lead] = NAN;
        }
    }

    msw_build_weights(cosw, sinw, period);
    __syncthreads();

    const int stride = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride) {
        const int t_begin = max(base_t, warm);
        const int t_end = min(base_t + blockDim.x - 1, series_len - 1);
        if (t_begin <= t_end) {
            const int tile_in_start = t_begin - (period - 1);
            const int tile_len = (t_end - t_begin + 1) + (period - 1);
            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                const int tt = tile_in_start + i;
                tile[i] = prices_tm[tt * num_series + series_idx];
            }
            __syncthreads();

            const int t = base_t + threadIdx.x;
            if (t >= t_begin && t <= t_end) {
                const int start = t - t_begin;
                float rp = 0.0f, ip = 0.0f;
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    const float w = tile[start + k];
                    rp = fmaf(cosw[k], w, rp);
                    ip = fmaf(sinw[k], w, ip);
                }
                const float phase = msw_phase_from_rp_ip(rp, ip);
                float s = __sinf(phase);
                float c = __cosf(phase);
                out_tm[t * (2 * num_series) + col_sine] = s;
                out_tm[t * (2 * num_series) + col_lead] = (s + c) * 0.70710678118654752440f;
            }
            __syncthreads();
        }
    }
}

