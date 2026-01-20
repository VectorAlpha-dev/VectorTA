














#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

extern "C" __global__
void minmax_batch_f32(const float* __restrict__ high,
                      const float* __restrict__ low,
                      int series_len,
                      int first_valid,
                      const int* __restrict__ orders,
                      int n_combos,
                      float* __restrict__ out_is_min,
                      float* __restrict__ out_is_max,
                      float* __restrict__ out_last_min,
                      float* __restrict__ out_last_max) {
    if (series_len <= 0) return;
    const int row = blockIdx.y;
    if (row >= n_combos) return;

    const int base = row * series_len;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += blockDim.x * gridDim.x) {
        out_is_min[base + i] = CUDART_NAN_F;
        out_is_max[base + i] = CUDART_NAN_F;
        out_last_min[base + i] = CUDART_NAN_F;
        out_last_max[base + i] = CUDART_NAN_F;
    }
    __syncthreads();


    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int order = orders[row];
    if (order <= 0) return;
    if (first_valid >= series_len) return;

    float last_min = CUDART_NAN_F;
    float last_max = CUDART_NAN_F;

    for (int i = max(0, first_valid); i < series_len; ++i) {
        float min_here = CUDART_NAN_F;
        float max_here = CUDART_NAN_F;

        const bool in_bounds = (i >= order) && (i + order < series_len);
        const float ch = high[i];
        const float cl = low[i];
        if (in_bounds && isfinite(ch) && isfinite(cl)) {

            bool left_ok_low = true, right_ok_low = true;
            float lmin = CUDART_INF_F, rmin = CUDART_INF_F;
            for (int o = 1; o <= order; ++o) {
                const float ll = low[i - o];
                const float rl = low[i + o];
                if (!isfinite(ll)) { left_ok_low = false; break; }
                if (!isfinite(rl)) { right_ok_low = false; break; }
                lmin = fminf(lmin, ll);
                rmin = fminf(rmin, rl);
            }
            if (left_ok_low && right_ok_low && cl < lmin && cl < rmin) {
                min_here = cl;
            }


            bool left_ok_high = true, right_ok_high = true;
            float lmax = -CUDART_INF_F, rmax = -CUDART_INF_F;
            for (int o = 1; o <= order; ++o) {
                const float lh = high[i - o];
                const float rh = high[i + o];
                if (!isfinite(lh)) { left_ok_high = false; break; }
                if (!isfinite(rh)) { right_ok_high = false; break; }
                lmax = fmaxf(lmax, lh);
                rmax = fmaxf(rmax, rh);
            }
            if (left_ok_high && right_ok_high && ch > lmax && ch > rmax) {
                max_here = ch;
            }
        }

        out_is_min[base + i] = min_here;
        out_is_max[base + i] = max_here;
        if (isfinite(min_here)) { last_min = min_here; }
        if (isfinite(max_here)) { last_max = max_here; }
        out_last_min[base + i] = last_min;
        out_last_max[base + i] = last_max;
    }
}


extern "C" __global__
void minmax_many_series_one_param_time_major_f32(const float* __restrict__ high_tm,
                                                 const float* __restrict__ low_tm,
                                                 const int* __restrict__ first_valids,
                                                 int num_series,
                                                 int series_len,
                                                 int order,
                                                 float* __restrict__ out_is_min_tm,
                                                 float* __restrict__ out_is_max_tm,
                                                 float* __restrict__ out_last_min_tm,
                                                 float* __restrict__ out_last_max_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || series_len <= 0 || order <= 0) return;
    const int stride = num_series;
    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];


    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        const int idx = t * stride + s;
        out_is_min_tm[idx] = CUDART_NAN_F;
        out_is_max_tm[idx] = CUDART_NAN_F;
        out_last_min_tm[idx] = CUDART_NAN_F;
        out_last_max_tm[idx] = CUDART_NAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    if (fv >= series_len) return;

    float last_min = CUDART_NAN_F;
    float last_max = CUDART_NAN_F;

    for (int t = max(0, fv); t < series_len; ++t) {
        const int idx = t * stride + s;
        float min_here = CUDART_NAN_F;
        float max_here = CUDART_NAN_F;

        const bool in_bounds = (t >= order) && (t + order < series_len);
        const float ch = high_tm[idx];
        const float cl = low_tm[idx];
        if (in_bounds && isfinite(ch) && isfinite(cl)) {
            bool left_ok_low = true, right_ok_low = true;
            float lmin = CUDART_INF_F, rmin = CUDART_INF_F;
            for (int o = 1; o <= order; ++o) {
                const float ll = low_tm[(t - o) * stride + s];
                const float rl = low_tm[(t + o) * stride + s];
                if (!isfinite(ll)) { left_ok_low = false; break; }
                if (!isfinite(rl)) { right_ok_low = false; break; }
                lmin = fminf(lmin, ll);
                rmin = fminf(rmin, rl);
            }
            if (left_ok_low && right_ok_low && cl < lmin && cl < rmin) {
                min_here = cl;
            }

            bool left_ok_high = true, right_ok_high = true;
            float lmax = -CUDART_INF_F, rmax = -CUDART_INF_F;
            for (int o = 1; o <= order; ++o) {
                const float lh = high_tm[(t - o) * stride + s];
                const float rh = high_tm[(t + o) * stride + s];
                if (!isfinite(lh)) { left_ok_high = false; break; }
                if (!isfinite(rh)) { right_ok_high = false; break; }
                lmax = fmaxf(lmax, lh);
                rmax = fmaxf(rmax, rh);
            }
            if (left_ok_high && right_ok_high && ch > lmax && ch > rmax) {
                max_here = ch;
            }
        }

        out_is_min_tm[idx] = min_here;
        out_is_max_tm[idx] = max_here;
        if (isfinite(min_here)) { last_min = min_here; }
        if (isfinite(max_here)) { last_max = max_here; }
        out_last_min_tm[idx] = last_min;
        out_last_max_tm[idx] = last_max;
    }
}







#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


static __device__ __forceinline__ int ilog2_floor_u32(unsigned int x) {

    return 31 - __clz(x);
}

static __device__ __forceinline__ float fminf2(float a, float b) { return fminf(a, b); }
static __device__ __forceinline__ float fmaxf2(float a, float b) { return fmaxf(a, b); }
static __device__ __forceinline__ uint8_t min_u8(uint8_t a, uint8_t b) { return a < b ? a : b; }


static __device__ __forceinline__
float rmq_min_f32(const float* __restrict__ st, int series_len, int l, int r) {
    unsigned int span = (unsigned int)(r - l + 1);
    int k = ilog2_floor_u32(span);
    int offset = k * series_len;
    const float a = st[offset + l];
    const int r0 = r - (1 << k) + 1;
    const float b = st[offset + r0];
    return fminf2(a, b);
}

static __device__ __forceinline__
float rmq_max_f32(const float* __restrict__ st, int series_len, int l, int r) {
    unsigned int span = (unsigned int)(r - l + 1);
    int k = ilog2_floor_u32(span);
    int offset = k * series_len;
    const float a = st[offset + l];
    const int r0 = r - (1 << k) + 1;
    const float b = st[offset + r0];
    return fmaxf2(a, b);
}

static __device__ __forceinline__
uint8_t rmq_min_u8(const uint8_t* __restrict__ st, int series_len, int l, int r) {
    unsigned int span = (unsigned int)(r - l + 1);
    int k = ilog2_floor_u32(span);
    int offset = k * series_len;
    const uint8_t a = st[offset + l];
    const int r0 = r - (1 << k) + 1;
    const uint8_t b = st[offset + r0];
    return min_u8(a, b);
}



extern "C" __global__
void st_init_level0_minmax_valid_f32(const float* __restrict__ low,
                                     const float* __restrict__ high,
                                     int series_len,
                                     float* __restrict__ low_min_st,
                                     float* __restrict__ high_max_st,
                                     uint8_t* __restrict__ valid_low_st,
                                     uint8_t* __restrict__ valid_high_st) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= series_len) return;

    const float cl = low[i];
    const float ch = high[i];
    const bool fl = isfinite(cl);
    const bool fh = isfinite(ch);


    low_min_st[i]  = fl ? cl : CUDART_INF_F;
    high_max_st[i] = fh ? ch : -CUDART_INF_F;
    valid_low_st[i]  = fl ? 1u : 0u;
    valid_high_st[i] = fh ? 1u : 0u;
}


extern "C" __global__
void st_build_level_k_minmax_valid_f32(int series_len,
                                       int k,
                                       float* __restrict__ low_min_st,
                                       float* __restrict__ high_max_st,
                                       uint8_t* __restrict__ valid_low_st,
                                       uint8_t* __restrict__ valid_high_st) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = 1 << (k - 1);
    const int span = 1 << k;
    if (i > series_len - span) return;

    const int prev = (k - 1) * series_len;

    const float aL = low_min_st[prev + i];
    const float bL = low_min_st[prev + i + half];
    low_min_st[k * series_len + i] = fminf2(aL, bL);

    const float aH = high_max_st[prev + i];
    const float bH = high_max_st[prev + i + half];
    high_max_st[k * series_len + i] = fmaxf2(aH, bH);

    const uint8_t avL = valid_low_st[prev + i];
    const uint8_t bvL = valid_low_st[prev + i + half];
    valid_low_st[k * series_len + i] = min_u8(avL, bvL);

    const uint8_t avH = valid_high_st[prev + i];
    const uint8_t bvH = valid_high_st[prev + i + half];
    valid_high_st[k * series_len + i] = min_u8(avH, bvH);
}


extern "C" __global__
void minmax_batch_rmq_f32(const float* __restrict__ high,
                          const float* __restrict__ low,
                          int series_len,
                          int first_valid,
                          const int* __restrict__ orders,
                          int n_combos,
                          const float* __restrict__ low_min_st,
                          const float* __restrict__ high_max_st,
                          const uint8_t* __restrict__ valid_low_st,
                          const uint8_t* __restrict__ valid_high_st,
                          float* __restrict__ out_is_min,
                          float* __restrict__ out_is_max) {
    if (series_len <= 0) return;
    const int row = blockIdx.y;
    if (row >= n_combos) return;

    const int k = orders[row];
    const int base = row * series_len;

    if (k <= 0) {

        for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < series_len; t += blockDim.x * gridDim.x) {
            out_is_min[base + t] = CUDART_NAN_F;
            out_is_max[base + t] = CUDART_NAN_F;
        }
        return;
    }




    const unsigned ku = (unsigned)k;
    const int klog = ilog2_floor_u32(ku);
    const int step = 1 << klog;
    const int off = klog * series_len;

    const int fv = (first_valid < 0 ? 0 : first_valid);
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < series_len; t += blockDim.x * gridDim.x) {

        out_is_min[base + t] = CUDART_NAN_F;
        out_is_max[base + t] = CUDART_NAN_F;
        if (t < fv) continue;
        const bool inb = (t >= k) && (t + k < series_len);
        if (!inb) continue;

        const float cl = low[t];
        const float ch = high[t];
        if (!(isfinite(cl) && isfinite(ch))) continue;


        const int l0 = t - k;
        const int r0 = t - 1;
        const int r0b = t - step;
        const int l1 = t + 1;
        const int r1 = t + k;
        const int r1b = t + k - step + 1;
        uint8_t left_ok_low  = min_u8(valid_low_st[off + l0], valid_low_st[off + r0b]);
        uint8_t right_ok_low = min_u8(valid_low_st[off + l1], valid_low_st[off + r1b]);
        if (left_ok_low && right_ok_low) {
            float lmin = fminf2(low_min_st[off + l0], low_min_st[off + r0b]);
            float rmin = fminf2(low_min_st[off + l1], low_min_st[off + r1b]);
            if (cl < lmin && cl < rmin) {
                out_is_min[base + t] = cl;
            }
        }


        uint8_t left_ok_high  = min_u8(valid_high_st[off + l0], valid_high_st[off + r0b]);
        uint8_t right_ok_high = min_u8(valid_high_st[off + l1], valid_high_st[off + r1b]);
        if (left_ok_high && right_ok_high) {
            float lmax = fmaxf2(high_max_st[off + l0], high_max_st[off + r0b]);
            float rmax = fmaxf2(high_max_st[off + l1], high_max_st[off + r1b]);
            if (ch > lmax && ch > rmax) {
                out_is_max[base + t] = ch;
            }
        }
    }
}


static __device__ __forceinline__ float combine_last_finite(float a, float b) {
    return isfinite(b) ? b : a;
}

static __device__ __forceinline__ float warp_scan_last(float x, unsigned mask) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float y = __shfl_up_sync(mask, x, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            x = combine_last_finite(y, x);
        }
    }
    return x;
}

extern "C" __global__
void forward_fill_two_streams_f32(const float* __restrict__ in_is_min,
                                  const float* __restrict__ in_is_max,
                                  int series_len,
                                  int n_rows,
                                  float* __restrict__ out_last_min,
                                  float* __restrict__ out_last_max) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int base = row * series_len;
    const unsigned full_mask = 0xffffffffu;
    __shared__ float warp_totals_min[WARP_SIZE];
    __shared__ float warp_totals_max[WARP_SIZE];

    float carry_min = CUDART_NAN_F;
    float carry_max = CUDART_NAN_F;

    for (int t0 = 0; t0 < series_len; t0 += blockDim.x) {
        const int t = t0 + threadIdx.x;
        float xm = (t < series_len) ? in_is_min[base + t] : CUDART_NAN_F;
        float xM = (t < series_len) ? in_is_max[base + t] : CUDART_NAN_F;

        float pm = warp_scan_last(xm, full_mask);
        float pM = warp_scan_last(xM, full_mask);

        const int lane = threadIdx.x & (WARP_SIZE - 1);
        const int wid  = threadIdx.x >> 5;

        if (lane == WARP_SIZE - 1) {
            warp_totals_min[wid] = pm;
            warp_totals_max[wid] = pM;
        }
        __syncthreads();

        if (wid == 0) {
            const int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
            float vmin = (threadIdx.x < nwarps) ? warp_totals_min[lane] : CUDART_NAN_F;
            float vmax = (threadIdx.x < nwarps) ? warp_totals_max[lane] : CUDART_NAN_F;
            vmin = warp_scan_last(vmin, full_mask);
            vmax = warp_scan_last(vmax, full_mask);
            if (lane < nwarps) {
                warp_totals_min[lane] = vmin;
                warp_totals_max[lane] = vmax;
            }
        }
        __syncthreads();

        if (wid > 0) {
            float warp_prefix_min = warp_totals_min[wid - 1];
            float warp_prefix_max = warp_totals_max[wid - 1];
            pm = combine_last_finite(warp_prefix_min, pm);
            pM = combine_last_finite(warp_prefix_max, pM);
        }

        pm = combine_last_finite(carry_min, pm);
        pM = combine_last_finite(carry_max, pM);

        if (t < series_len) {
            out_last_min[base + t] = pm;
            out_last_max[base + t] = pM;
        }

        __syncthreads();
        if (threadIdx.x == blockDim.x - 1 || (t == series_len - 1)) {
            warp_totals_min[0] = pm;
            warp_totals_max[0] = pM;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            carry_min = warp_totals_min[0];
            carry_max = warp_totals_max[0];
        }
        __syncthreads();
    }
}
