#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void willr_build_base_and_nan_f32(const float* __restrict__ high,
                                  const float* __restrict__ low,
                                  int series_len,
                                  float* __restrict__ st_max,
                                  float* __restrict__ st_min,
                                  int* __restrict__ nan_flags) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < series_len;
         idx += blockDim.x * gridDim.x) {
        const float h = high[idx];
        const float l = low[idx];
        const bool has_nan = isnan(h) || isnan(l);
        st_max[idx] = has_nan ? -INFINITY : h;
        st_min[idx] = has_nan ? INFINITY : l;
        nan_flags[idx] = has_nan ? 1 : 0;
    }
}

extern "C" __global__
void willr_prefix_nan_psum_i32(const int* __restrict__ nan_flags,
                               int series_len,
                               int* __restrict__ nan_psum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    nan_psum[0] = 0;
    int acc = 0;
    for (int i = 0; i < series_len; ++i) {
        acc += nan_flags[i];
        nan_psum[i + 1] = acc;
    }
}

extern "C" __global__
void willr_build_sparse_level_f32(float* __restrict__ st_max,
                                  float* __restrict__ st_min,
                                  int prev_offset,
                                  int curr_offset,
                                  int curr_len,
                                  int half_offset) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < curr_len;
         idx += blockDim.x * gridDim.x) {
        const int left = prev_offset + idx;
        const int right = left + half_offset;
        st_max[curr_offset + idx] = fmaxf(st_max[left], st_max[right]);
        st_min[curr_offset + idx] = fminf(st_min[left], st_min[right]);
    }
}

extern "C" __global__
void willr_batch_f32(const float* __restrict__ close,
                     const int* __restrict__ periods,
                     const int* __restrict__ log2_tbl,
                     const int* __restrict__ level_offsets,
                     const float* __restrict__ st_max,
                     const float* __restrict__ st_min,
                     const int* __restrict__ nan_psum,
                     int series_len,
                     int first_valid,
                     int level_count,
                     int n_combos,
                     float* __restrict__ out) {

    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    float* __restrict__ out_row = out + base;


    const int period = periods[combo];
    if (period <= 0 || first_valid >= series_len) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x)
            out_row[i] = NAN;
        return;
    }

    const int warm = first_valid + period - 1;


    const int warm_clamped = (warm < series_len) ? warm : series_len;
    for (int i = threadIdx.x; i < warm_clamped; i += blockDim.x)
        out_row[i] = NAN;

    if (warm >= series_len) return;


    const int k = log2_tbl[period];
    if (k < 0 || k >= level_count) {
        for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x)
            out_row[t] = NAN;
        return;
    }
    const int offset     = 1 << k;
    const int level_base = level_offsets[k];


    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const float c = close[t];
        if (isnan(c)) { out_row[t] = NAN; continue; }

        const int start = t - period + 1;


        if (nan_psum[t + 1] - nan_psum[start] != 0) {
            out_row[t] = NAN;
            continue;
        }


        const int idx_a  = level_base + start;
        const int idx_b  = level_base + (t + 1 - offset);
        const float hmax = fmaxf(st_max[idx_a], st_max[idx_b]);
        const float lmin = fminf(st_min[idx_a], st_min[idx_b]);

        const float denom = hmax - lmin;
        out_row[t] = (denom == 0.0f) ? 0.0f : ((hmax - c) / denom) * -100.0f;
    }
}

extern "C" __global__
void willr_batch_period_levels_f32(const float* __restrict__ close,
                                   const int* __restrict__ periods,
                                   const int* __restrict__ period_levels,
                                   const int* __restrict__ level_offsets,
                                   const float* __restrict__ st_max,
                                   const float* __restrict__ st_min,
                                   const int* __restrict__ nan_psum,
                                   int series_len,
                                   int first_valid,
                                   int level_count,
                                   int n_combos,
                                   float* __restrict__ out) {

    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    float* __restrict__ out_row = out + base;

    const int period = periods[combo];
    if (period <= 0 || first_valid >= series_len) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x)
            out_row[i] = NAN;
        return;
    }

    const int warm = first_valid + period - 1;
    const int warm_clamped = (warm < series_len) ? warm : series_len;
    for (int i = threadIdx.x; i < warm_clamped; i += blockDim.x)
        out_row[i] = NAN;
    if (warm >= series_len) return;

    const int k = period_levels[combo];
    if (k < 0 || k >= level_count) {
        for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x)
            out_row[t] = NAN;
        return;
    }
    const int offset = 1 << k;
    const int level_base = level_offsets[k];

    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const float c = close[t];
        if (isnan(c)) { out_row[t] = NAN; continue; }

        const int start = t - period + 1;
        if (nan_psum[t + 1] - nan_psum[start] != 0) {
            out_row[t] = NAN;
            continue;
        }

        const int idx_a = level_base + start;
        const int idx_b = level_base + (t + 1 - offset);
        const float hmax = fmaxf(st_max[idx_a], st_max[idx_b]);
        const float lmin = fminf(st_min[idx_a], st_min[idx_b]);
        const float denom = hmax - lmin;
        out_row[t] = (denom == 0.0f) ? 0.0f : ((hmax - c) / denom) * -100.0f;
    }
}


extern "C" __global__
void willr_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    if (period <= 0) {

        for (int t = 0; t < rows; ++t) out_tm[t * cols + series] = NAN;
        return;
    }

    const int first_valid = first_valids[series];
    const int warm = first_valid + period - 1;


    const int wclamp = (warm < rows) ? warm : rows;
    for (int t = 0; t < wclamp; ++t)
        out_tm[t * cols + series] = NAN;

    if (warm >= rows) return;

    for (int t = warm; t < rows; ++t) {
        const int idx = t * cols + series;
        const float c = close_tm[idx];
        if (isnan(c)) { out_tm[idx] = NAN; continue; }

        const int start = t - period + 1;
        float h = -INFINITY, l = INFINITY;
        bool any_nan = false;


        for (int j = start; j <= t; ++j) {
            const int jidx = j * cols + series;
            const float hj = high_tm[jidx];
            const float lj = low_tm[jidx];
            if (isnan(hj) || isnan(lj)) { any_nan = true; break; }
            if (hj > h) h = hj;
            if (lj < l) l = lj;
        }

        if (any_nan) { out_tm[idx] = NAN; continue; }

        const float denom = h - l;
        out_tm[idx] = (denom == 0.0f) ? 0.0f : ((h - c) / denom) * -100.0f;
    }
}
