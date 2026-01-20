









#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __forceinline__ __device__ bool both_finite(float h, float l) {
    return isfinite(h) && isfinite(l);
}




extern "C" __global__
void aroon_batch_f32(const float* __restrict__ high,
                     const float* __restrict__ low,
                     const int*   __restrict__ lengths,
                     int series_len,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out_up,
                     float* __restrict__ out_down) {

    const int combo = blockIdx.x + blockIdx.y * gridDim.x;
    if (combo >= n_combos) return;

    const int length = lengths[combo];
    if (length <= 0 || first_valid < 0 || first_valid >= series_len) return;

    const int base = combo * series_len;
    const int W = length + 1;
    const int warm = first_valid + length;
    if (warm >= series_len) {

        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_up  [base + i] = NAN;
            out_down[base + i] = NAN;
        }
        return;
    }


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_up  [base + i] = NAN;
        out_down[base + i] = NAN;
    }
    __syncthreads();


    extern __shared__ int s_deques[];
    int* __restrict__ dq_max = s_deques;
    int* __restrict__ dq_min = s_deques + W;


    if (threadIdx.x != 0) return;


    int h_head = 0, h_tail = 0;
    int h_head_idx = 0, h_tail_idx = 0;
    int l_head = 0, l_tail = 0;
    int l_head_idx = 0, l_tail_idx = 0;

    const float scale = 100.0f / (float)length;
    int last_bad = -0x3fffffff;


    for (int t = 0; t < series_len; ++t) {
        const int start = t - length;

        while (h_tail > h_head && dq_max[h_head_idx] < start) {
            ++h_head;
            h_head_idx = (h_head_idx + 1 == W) ? 0 : (h_head_idx + 1);
        }
        while (l_tail > l_head && dq_min[l_head_idx] < start) {
            ++l_head;
            l_head_idx = (l_head_idx + 1 == W) ? 0 : (l_head_idx + 1);
        }

        const float h = high[t];
        const float l = low[t];


        if (!both_finite(h, l)) {
            last_bad = t;
        } else {

            while (h_tail > h_head) {
                const int last_slot = (h_tail_idx == 0) ? (W - 1) : (h_tail_idx - 1);
                const int idx = dq_max[last_slot];
                if (high[idx] < h) {
                    --h_tail;
                    h_tail_idx = last_slot;
                } else {
                    break;
                }
            }
            dq_max[h_tail_idx] = t;
            ++h_tail;
            h_tail_idx = (h_tail_idx + 1 == W) ? 0 : (h_tail_idx + 1);

            while (l_tail > l_head) {
                const int last_slot = (l_tail_idx == 0) ? (W - 1) : (l_tail_idx - 1);
                const int idx = dq_min[last_slot];
                if (low[idx] > l) {
                    --l_tail;
                    l_tail_idx = last_slot;
                } else {
                    break;
                }
            }
            dq_min[l_tail_idx] = t;
            ++l_tail;
            l_tail_idx = (l_tail_idx + 1 == W) ? 0 : (l_tail_idx + 1);
        }

        if (t >= warm) {
            if (last_bad >= start) {
                out_up  [base + t] = NAN;
                out_down[base + t] = NAN;
            } else {
                const int idx_hi = (h_tail > h_head) ? dq_max[h_head_idx] : -1;
                const int idx_lo = (l_tail > l_head) ? dq_min[l_head_idx] : -1;
                if (idx_hi < 0 || idx_lo < 0) {
                    out_up  [base + t] = NAN;
                    out_down[base + t] = NAN;
                } else {
                    const int dist_hi = t - idx_hi;
                    const int dist_lo = t - idx_lo;

                    const float up = (dist_hi == 0) ? 100.0f
                                     : (dist_hi >= length ? 0.0f
                                     : fmaf(-(float)dist_hi, scale, 100.0f));
                    const float dn = (dist_lo == 0) ? 100.0f
                                     : (dist_lo >= length ? 0.0f
                                     : fmaf(-(float)dist_lo, scale, 100.0f));
                    out_up  [base + t] = up;
                    out_down[base + t] = dn;
                }
            }
        }
    }
}




extern "C" __global__
void aroon_many_series_one_param_f32(const float* __restrict__ high_tm,
                                     const float* __restrict__ low_tm,
                                     const int*   __restrict__ first_valids,
                                     int length,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_up_tm,
                                     float* __restrict__ out_down_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || length <= 0) return;

    const int first = first_valids[s];
    if (first < 0 || first >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_up_tm  [t * num_series + s] = NAN;
            out_down_tm[t * num_series + s] = NAN;
        }
        return;
    }

    const int W = length + 1;
    const int warm = first + length;
    const int stride = num_series;


    for (int t = threadIdx.x; t < (warm < series_len ? warm : series_len); t += blockDim.x) {
        out_up_tm  [t * stride + s] = NAN;
        out_down_tm[t * stride + s] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    extern __shared__ int s_deques[];
    int* __restrict__ dq_max = s_deques;
    int* __restrict__ dq_min = s_deques + W;

    int h_head = 0, h_tail = 0;
    int h_head_idx = 0, h_tail_idx = 0;
    int l_head = 0, l_tail = 0;
    int l_head_idx = 0, l_tail_idx = 0;
    const float scale = 100.0f / (float)length;
    int last_bad = -0x3fffffff;

    for (int t = 0; t < series_len; ++t) {
        const int start = t - length;
        while (h_tail > h_head && dq_max[h_head_idx] < start) {
            ++h_head;
            h_head_idx = (h_head_idx + 1 == W) ? 0 : (h_head_idx + 1);
        }
        while (l_tail > l_head && dq_min[l_head_idx] < start) {
            ++l_head;
            l_head_idx = (l_head_idx + 1 == W) ? 0 : (l_head_idx + 1);
        }

        const float h = high_tm[t * stride + s];
        const float l = low_tm [t * stride + s];

        if (!both_finite(h, l)) {
            last_bad = t;
        } else {

            while (h_tail > h_head) {
                const int last_slot = (h_tail_idx == 0) ? (W - 1) : (h_tail_idx - 1);
                const int idx = dq_max[last_slot];
                if (high_tm[idx * stride + s] < h) {
                    --h_tail;
                    h_tail_idx = last_slot;
                } else {
                    break;
                }
            }
            dq_max[h_tail_idx] = t;
            ++h_tail;
            h_tail_idx = (h_tail_idx + 1 == W) ? 0 : (h_tail_idx + 1);

            while (l_tail > l_head) {
                const int last_slot = (l_tail_idx == 0) ? (W - 1) : (l_tail_idx - 1);
                const int idx = dq_min[last_slot];
                if (low_tm[idx * stride + s] > l) {
                    --l_tail;
                    l_tail_idx = last_slot;
                } else {
                    break;
                }
            }
            dq_min[l_tail_idx] = t;
            ++l_tail;
            l_tail_idx = (l_tail_idx + 1 == W) ? 0 : (l_tail_idx + 1);
        }

        if (t >= warm) {
            if (last_bad >= start) {
                out_up_tm  [t * stride + s] = NAN;
                out_down_tm[t * stride + s] = NAN;
            } else {
                const int idx_hi = (h_tail > h_head) ? dq_max[h_head_idx] : -1;
                const int idx_lo = (l_tail > l_head) ? dq_min[l_head_idx] : -1;
                if (idx_hi < 0 || idx_lo < 0) {
                    out_up_tm  [t * stride + s] = NAN;
                    out_down_tm[t * stride + s] = NAN;
                } else {
                    const int dist_hi = t - idx_hi;
                    const int dist_lo = t - idx_lo;
                    const float up = (dist_hi == 0) ? 100.0f
                                     : (dist_hi >= length ? 0.0f
                                     : fmaf(-(float)dist_hi, scale, 100.0f));
                    const float dn = (dist_lo == 0) ? 100.0f
                                     : (dist_lo >= length ? 0.0f
                                     : fmaf(-(float)dist_lo, scale, 100.0f));
                    out_up_tm  [t * stride + s] = up;
                    out_down_tm[t * stride + s] = dn;
                }
            }
        }
    }
}
