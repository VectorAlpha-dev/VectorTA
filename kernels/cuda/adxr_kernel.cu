// CUDA kernels for ADXR (Average Directional Index Rating)
//
// Semantics mirror the scalar Rust implementation in src/indicators/adxr.rs:
// - Warmup: values before (first_valid + 2*period) are NaN
// - Wilder smoothing with alpha = 1/period for smoothed +DM/-DM
// - DX = 100 * |+DM_s - -DM_s| / (+DM_s + -DM_s), denominator 0 -> 0
// - ADX builds as mean of first `period` DX values, then Wilder recursion
// - ADXR[t] = 0.5 * (ADX[t] + ADX[t - period]) once the ring is full

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

static __forceinline__ __device__ float fmax3(float a, float b, float c) {
    return fmaxf(a, fmaxf(b, c));
}

// Tiny helper: Kahan-compensated sum in FP32
struct KahanF32 {
    float sum, c;
    __device__ inline void init(float s0=0.f){ sum=s0; c=0.f; }
    __device__ inline void add(float x){
        float y = x - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
};

extern "C" __global__
void adxr_batch_f32(const float* __restrict__ high,
                    const float* __restrict__ low,
                    const float* __restrict__ close,
                    const int* __restrict__ periods,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.y * gridDim.x + blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int base = combo * series_len;

    // Fill entire row with NaNs cooperatively
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    // Only one thread per row runs the sequential recurrence
    if (threadIdx.x != 0) return;

    // Bootstrap sums over first window: i in [first+1 .. first+period]
    int i = first_valid + 1;
    const int stop = min(first_valid + period, series_len - 1);
    // Accumulate in FP32 (ATR/TR removed; not needed for DX)
    float pdm_sum = 0.f;
    float mdm_sum = 0.f;
    while (i <= stop) {
        const float ch = high[i];
        const float cl = low[i];
        const float ph = high[i - 1];
        const float pl = low[i - 1];
        const float up = ch - ph;
        const float down = pl - cl;
        if (up > down && up > 0.0f) pdm_sum += up;
        if (down > up && down > 0.0f) mdm_sum += down;
        ++i;
    }

    // Initial DX from bootstrap sums (ATR cancels)
    const float denom0 = pdm_sum + mdm_sum;
    const float initial_dx = (denom0 > 0.f) ? (100.f * fabsf(pdm_sum - mdm_sum) / denom0) : 0.f;

    const float p = (float)period;
    const float inv_p = 1.0f / p;
    const float one_minus = 1.0f - inv_p;
    const float pm1 = p - 1.0f;
    const int warmup_start = first_valid + 2 * period;

    // Wilder running state (FP32)
    float pdm_s = pdm_sum;
    float mdm_s = mdm_sum;

    KahanF32 dx_sum; dx_sum.init(initial_dx);
    int dx_count = 1;
    float adx_last = NAN;
    bool have_adx = false;

    // Use a local ring buffer for ADX history. Periods are small in practice.
    float ring_local[256];
    const bool use_local = (period <= 256);
    if (use_local) {
        for (int k = 0; k < period; ++k) ring_local[k] = NAN;
    }
    int head = 0;

    i = first_valid + period + 1;
    while (i < series_len) {
        const float ch = high[i];
        const float cl = low[i];
        const float ph = high[i - 1];
        const float pl = low[i - 1];
        const float up = ch - ph;
        const float down = pl - cl;
        const float plus_dm = (up > down && up > 0.0f) ? up : 0.0f;
        const float minus_dm = (down > up && down > 0.0f) ? down : 0.0f;

        // Wilder smoothing (FP32)
        pdm_s = fmaf(pdm_s, one_minus, plus_dm);
        mdm_s = fmaf(mdm_s, one_minus, minus_dm);

        const float denom = pdm_s + mdm_s;
        const float dx = (denom > 0.f) ? (100.f * fabsf(pdm_s - mdm_s) / denom) : 0.f;

        if (dx_count < period) {
            dx_sum.add(dx);
            dx_count += 1;
            if (dx_count == period) {
                adx_last = dx_sum.sum * inv_p;
                have_adx = true;
                // push into ring
                float prev = use_local ? ring_local[head] : NAN;
                if (use_local) ring_local[head] = adx_last;
                head += 1; if (head == period) head = 0;
                if (i >= warmup_start && !isnan(prev)) {
                    out[base + i] = 0.5f * (adx_last + prev);
                }
            }
        } else if (have_adx) {
            const float adx_curr = (adx_last * pm1 + dx) * inv_p;
            adx_last = adx_curr;
            float prev = use_local ? ring_local[head] : NAN;
            if (use_local) ring_local[head] = adx_curr;
            head += 1; if (head == period) head = 0;
            if (i >= warmup_start && !isnan(prev)) {
                out[base + i] = 0.5f * (adx_curr + prev);
            }
        }

        ++i;
    }
}

// Many-series × one-param (time-major). Each block handles one series with a single thread.
extern "C" __global__
void adxr_many_series_one_param_f32(const float* __restrict__ high_tm,
                                    const float* __restrict__ low_tm,
                                    const float* __restrict__ close_tm,
                                    const int* __restrict__ first_valids,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int stride = num_series;

    // Fill with NaNs
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    // Bootstrap over [first+1 .. first+period]
    int i = first_valid + 1;
    const int stop = min(first_valid + period, series_len - 1);
    float pdm_sum = 0.f;
    float mdm_sum = 0.f;
    while (i <= stop) {
        const float ch = high_tm[i * stride + series];
        const float cl = low_tm[i * stride + series];
        const float ph = high_tm[(i - 1) * stride + series];
        const float pl = low_tm[(i - 1) * stride + series];
        const float up = ch - ph;
        const float down = pl - cl;
        if (up > down && up > 0.0f) pdm_sum += up;
        if (down > up && down > 0.0f) mdm_sum += down;
        ++i;
    }

    const float denom0 = pdm_sum + mdm_sum;
    const float initial_dx = (denom0 > 0.f) ? (100.f * fabsf(pdm_sum - mdm_sum) / denom0) : 0.f;

    const float p = (float)period;
    const float inv_p = 1.0f / p;
    const float one_minus = 1.0f - inv_p;
    const float pm1 = p - 1.0f;
    const int warmup_start = first_valid + 2 * period;

    float pdm_s = pdm_sum;
    float mdm_s = mdm_sum;

    KahanF32 dx_sum; dx_sum.init(initial_dx);
    int dx_count = 1;
    float adx_last = NAN;
    bool have_adx = false;

    int head = 0;
    // Use small local ring for typical periods; fall back to noop if too large
    const bool use_local = (period <= 256);
    float ring_local[256];
    if (use_local) {
        for (int k = 0; k < period; ++k) ring_local[k] = NAN;
    }

    i = first_valid + period + 1;
    while (i < series_len) {
        const float ch = high_tm[i * stride + series];
        const float cl = low_tm[i * stride + series];
        const float ph = high_tm[(i - 1) * stride + series];
        const float pl = low_tm[(i - 1) * stride + series];
        const float up = ch - ph;
        const float down = pl - cl;
        const float plus_dm = (up > down && up > 0.0f) ? up : 0.0f;
        const float minus_dm = (down > up && down > 0.0f) ? down : 0.0f;

        pdm_s = fmaf(pdm_s, one_minus, plus_dm);
        mdm_s = fmaf(mdm_s, one_minus, minus_dm);

        const float denom = pdm_s + mdm_s;
        const float dx = (denom > 0.f) ? (100.f * fabsf(pdm_s - mdm_s) / denom) : 0.f;

        if (dx_count < period) {
            dx_sum.add(dx);
            dx_count += 1;
            if (dx_count == period) {
                adx_last = dx_sum.sum * inv_p;
                have_adx = true;
                float prev = use_local ? ring_local[head] : NAN;
                if (use_local) ring_local[head] = adx_last;
                head += 1; if (head == period) head = 0;
                if (i >= warmup_start && !isnan(prev)) {
                    out_tm[i * stride + series] = 0.5f * (adx_last + prev);
                }
            }
        } else if (have_adx) {
            const float adx_curr = (adx_last * pm1 + dx) * inv_p;
            adx_last = adx_curr;
            float prev = use_local ? ring_local[head] : NAN;
            if (use_local) ring_local[head] = adx_curr;
            head += 1; if (head == period) head = 0;
            if (i >= warmup_start && !isnan(prev)) {
                out_tm[i * stride + series] = 0.5f * (adx_curr + prev);
            }
        }

        ++i;
    }
}

// -----------------------------------------------------------------------------
// Optimized one series × many params kernel (throughput-oriented, tiled)
// Adds global ring workspace (adx_ring, ring_pitch) to support any period.
// -----------------------------------------------------------------------------
extern "C" __global__
void adxr_one_series_many_params_f32_opt(const float* __restrict__ high,
                                         const float* __restrict__ low,
                                         const float* __restrict__ close,
                                         const int*   __restrict__ periods,
                                         int series_len,
                                         int first_valid,
                                         int n_periods,
                                         // ADX ring workspace (size: n_periods * ring_pitch)
                                         float* __restrict__ adx_ring,
                                         int ring_pitch,
                                         // Output (row-major): out[row * series_len + t]
                                         float* __restrict__ out)
{
    if (first_valid < 0 || first_valid >= series_len || n_periods <= 0) return;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_periods) return;

    const int period = periods[tid];
    if (period <= 0) return;

    const int row_base  = tid * series_len;
    const int ring_base = tid * ring_pitch;

    // Fill this row with NaNs once per row
    if (threadIdx.x == 0) {
        for (int t = 0; t < series_len; ++t) {
            out[row_base + t] = NAN;
        }
    }

    // Initialize ring slots this period will use
    for (int k = 0; k < period; ++k) {
        adx_ring[ring_base + k] = NAN;
    }
    __syncthreads();

    const float p         = (float)period;
    const float inv_p     = 1.0f / p;
    const float one_minus = 1.0f - inv_p;
    const float pm1       = p - 1.0f;
    const int   warmup_start = first_valid + 2 * period;

    // Bootstrap sums (period-dependent) over [first_valid+1 .. first_valid+period]
    const int init_i0 = first_valid + 1;
    const int init_i1 = min(first_valid + period, series_len - 1);
    float pdm_s = 0.f;
    float mdm_s = 0.f;
    for (int i = init_i0; i <= init_i1; ++i) {
        const float ch = high[i];
        const float cl = low[i];
        const float ph = high[i - 1];
        const float pl = low[i - 1];
        const float up   = ch - ph;
        const float down = pl - cl;
        if (up   > down && up   > 0.0f) pdm_s += up;
        if (down > up   && down > 0.0f) mdm_s += down;
    }

    float dx0 = 0.f;
    {
        const float denom0 = pdm_s + mdm_s;
        dx0 = (denom0 > 0.f ? 100.f * fabsf(pdm_s - mdm_s) / denom0 : 0.f);
    }

    KahanF32 dx_sum; dx_sum.init(0.f); dx_sum.add(dx0);
    int   dx_count = 1;
    float adx_last = NAN;
    bool  have_adx = false;

    int head = 0; // ring head for this period

    // Tiled pass over time using dynamic shared memory
    extern __shared__ float smem[];
    const int TILE = 256;
    float* pdm_tile = smem;
    float* mdm_tile = smem + TILE;

    int i_global = init_i0;
    while (i_global < series_len) {
        const int tile_start = i_global;
        const int tile_end   = min(tile_start + TILE, series_len);
        const int count      = tile_end - tile_start;

        // Tile loader: compute +DM/-DM once per tile per block
        for (int j = threadIdx.x; j < count; j += blockDim.x) {
            const int i = tile_start + j;
            const float ch = high[i];
            const float cl = low[i];
            const float ph = high[i - 1];
            const float pl = low[i - 1];
            const float up   = ch - ph;
            const float down = pl - cl;
            pdm_tile[j] = (up   > down && up   > 0.0f) ? up   : 0.0f;
            mdm_tile[j] = (down > up   && down > 0.0f) ? down : 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < count; ++j) {
            const int i = tile_start + j;

            if (i <= first_valid + period) {
                // still within bootstrap window; skip
            } else {
                // Wilder updates
                pdm_s = fmaf(pdm_s, one_minus, pdm_tile[j]);
                mdm_s = fmaf(mdm_s, one_minus, mdm_tile[j]);

                const float denom = pdm_s + mdm_s;
                const float dx = (denom > 0.f ? 100.f * fabsf(pdm_s - mdm_s) / denom : 0.f);

                if (dx_count < period) {
                    dx_sum.add(dx);
                    dx_count += 1;
                    if (dx_count == period) {
                        adx_last = dx_sum.sum * inv_p;
                        have_adx = true;
                        float prev = adx_ring[ring_base + head];
                        adx_ring[ring_base + head] = adx_last;
                        head += 1; if (head == period) head = 0;
                        if (i >= warmup_start && !isnan(prev)) {
                            out[row_base + i] = 0.5f * (adx_last + prev);
                        }
                    }
                } else if (have_adx) {
                    const float adx_curr = (adx_last * pm1 + dx) * inv_p;
                    adx_last = adx_curr;
                    float prev = adx_ring[ring_base + head];
                    adx_ring[ring_base + head] = adx_curr;
                    head += 1; if (head == period) head = 0;
                    if (i >= warmup_start && !isnan(prev)) {
                        out[row_base + i] = 0.5f * (adx_curr + prev);
                    }
                }
            }
        }

        __syncthreads();
        i_global = tile_end;
    }
}

// -----------------------------------------------------------------------------
// Optional: Many-series × one-param variant with global-ring fallback
// Keeps time-major layout and preserves original signature via a new function.
// -----------------------------------------------------------------------------
extern "C" __global__
void adxr_many_series_one_param_time_major_f32_opt(const float* __restrict__ high_tm,
                                                   const float* __restrict__ low_tm,
                                                   const float* __restrict__ close_tm,
                                                   const int*   __restrict__ first_valids,
                                                   int period,
                                                   int num_series,
                                                   int series_len,
                                                   // Global ring workspace (size: num_series * ring_pitch)
                                                   float* __restrict__ adx_ring,
                                                   int ring_pitch,
                                                   float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int stride = num_series;

    // Fill with NaNs
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    // Bootstrap
    int i = first_valid + 1;
    const int stop = min(first_valid + period, series_len - 1);
    float pdm_sum = 0.f;
    float mdm_sum = 0.f;
    while (i <= stop) {
        const float ch = high_tm[i * stride + series];
        const float cl = low_tm[i * stride + series];
        const float ph = high_tm[(i - 1) * stride + series];
        const float pl = low_tm[(i - 1) * stride + series];
        const float up = ch - ph;
        const float down = pl - cl;
        if (up > down && up > 0.0f) pdm_sum += up;
        if (down > up && down > 0.0f) mdm_sum += down;
        ++i;
    }

    const float denom0 = pdm_sum + mdm_sum;
    const float initial_dx = (denom0 > 0.f) ? (100.f * fabsf(pdm_sum - mdm_sum) / denom0) : 0.f;

    const float p = (float)period;
    const float inv_p = 1.0f / p;
    const float one_minus = 1.0f - inv_p;
    const float pm1 = p - 1.0f;
    const int warmup_start = first_valid + 2 * period;

    float pdm_s = pdm_sum;
    float mdm_s = mdm_sum;
    KahanF32 dx_sum; dx_sum.init(initial_dx);
    int dx_count = 1;
    float adx_last = NAN;
    bool have_adx = false;

    int head = 0;
    const bool use_local = (period <= 256);
    float ring_local[256];
    if (use_local) {
        for (int k = 0; k < period; ++k) ring_local[k] = NAN;
    } else {
        // initialize global ring for this series
        const int ring_base = series * ring_pitch;
        for (int k = 0; k < period; ++k) adx_ring[ring_base + k] = NAN;
    }

    i = first_valid + period + 1;
    while (i < series_len) {
        const float ch = high_tm[i * stride + series];
        const float cl = low_tm[i * stride + series];
        const float ph = high_tm[(i - 1) * stride + series];
        const float pl = low_tm[(i - 1) * stride + series];
        const float up = ch - ph;
        const float down = pl - cl;
        const float plus_dm = (up > down && up > 0.0f) ? up : 0.0f;
        const float minus_dm = (down > up && down > 0.0f) ? down : 0.0f;

        pdm_s = fmaf(pdm_s, one_minus, plus_dm);
        mdm_s = fmaf(mdm_s, one_minus, minus_dm);

        const float denom = pdm_s + mdm_s;
        const float dx = (denom > 0.0f) ? (100.f * fabsf(pdm_s - mdm_s) / denom) : 0.0f;

        if (dx_count < period) {
            dx_sum.add(dx);
            dx_count += 1;
            if (dx_count == period) {
                adx_last = dx_sum.sum * inv_p;
                have_adx = true;
                float prev;
                if (use_local) {
                    prev = ring_local[head];
                    ring_local[head] = adx_last;
                } else {
                    const int ring_base = series * ring_pitch;
                    prev = adx_ring[ring_base + head];
                    adx_ring[ring_base + head] = adx_last;
                }
                head += 1; if (head == period) head = 0;
                if (i >= warmup_start && !isnan(prev)) {
                    out_tm[i * stride + series] = 0.5f * (adx_last + prev);
                }
            }
        } else if (have_adx) {
            const float adx_curr = (adx_last * pm1 + dx) * inv_p;
            adx_last = adx_curr;
            float prev;
            if (use_local) {
                prev = ring_local[head];
                ring_local[head] = adx_curr;
            } else {
                const int ring_base = series * ring_pitch;
                prev = adx_ring[ring_base + head];
                adx_ring[ring_base + head] = adx_curr;
            }
            head += 1; if (head == period) head = 0;
            if (i >= warmup_start && !isnan(prev)) {
                out_tm[i * stride + series] = 0.5f * (adx_curr + prev);
            }
        }

        ++i;
    }
}
