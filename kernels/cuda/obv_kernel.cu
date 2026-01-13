



















#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef OBV_BLOCK_SIZE
#define OBV_BLOCK_SIZE 256
#endif
#ifndef OBV_ITEMS_PER_THREAD
#define OBV_ITEMS_PER_THREAD 8
#endif


struct FPair { float hi, lo; };

__device__ __forceinline__ FPair make_zero_pair() { return {0.0f, 0.0f}; }

__device__ __forceinline__ FPair two_sum_fp32(float a, float b) {

    float s  = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    return {s, err};
}

__device__ __forceinline__ FPair fp_add_pair(FPair x, FPair y) {
    FPair t = two_sum_fp32(x.hi, y.hi);
    float lo = x.lo + y.lo;
    FPair u = two_sum_fp32(t.hi, t.lo + lo);
    return {u.hi, u.lo};
}
__device__ __forceinline__ FPair fp_add_f(FPair x, float y) {
    FPair t = two_sum_fp32(x.hi, y);
    FPair u = two_sum_fp32(t.hi, t.lo + x.lo);
    return {u.hi, u.lo};
}
__device__ __forceinline__ FPair fp_sub_pair(FPair x, FPair y) {
    return fp_add_pair(x, { -y.hi, -y.lo });
}
__device__ __forceinline__ float fp_collapse(FPair x) { return x.hi + x.lo; }


__device__ __forceinline__ FPair warp_inclusive_scan(FPair v, unsigned mask) {
    int lane = threadIdx.x & 31;

    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float hi = __shfl_up_sync(mask, v.hi, offset);
        float lo = __shfl_up_sync(mask, v.lo, offset);
        if (lane >= offset) v = fp_add_pair(v, {hi, lo});
    }
    return v;
}



template<int NUM_WARPS>
__device__ __forceinline__
FPair block_exclusive_offset(FPair thread_total, FPair* warp_buf) {
    unsigned full = 0xFFFFFFFFu;
    int lane  = threadIdx.x & 31;
    int wid   = threadIdx.x >> 5;

    FPair incl = warp_inclusive_scan(thread_total, full);

    if (lane == 31) warp_buf[wid] = incl;
    __syncthreads();


    if (wid == 0) {
        FPair x = (lane < NUM_WARPS) ? warp_buf[lane] : make_zero_pair();
        FPair x_incl = warp_inclusive_scan(x, full);

        FPair x_excl = fp_sub_pair(x_incl, x);
        if (lane < NUM_WARPS) warp_buf[lane] = x_excl;
    }
    __syncthreads();


    FPair warp_off = warp_buf[wid];
    FPair excl_intra = fp_sub_pair(incl, thread_total);
    return fp_add_pair(warp_off, excl_intra);
}




extern "C" __global__
void obv_batch_f32_pass1_tilescan(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int series_len,
    int ,
    int first_valid,
    float* __restrict__ out,
    FPair* __restrict__ block_sums,
    int tiles_per_row
){
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    const int base = 0;

    if (series_len <= 0 || bid >= tiles_per_row) return;
    const int fv = first_valid < 0 ? 0 : first_valid;

    constexpr int ITEMS = OBV_ITEMS_PER_THREAD;
    const int tile_size = blockDim.x * ITEMS;
    const int tile_beg  = bid * tile_size;
    const int tile_end  = min(series_len, tile_beg + tile_size);


    constexpr int NUM_WARPS = (OBV_BLOCK_SIZE + 31) / 32;
    __shared__ FPair warp_buf[NUM_WARPS];
    __shared__ FPair seg_sum_shared;


    FPair seg_base = make_zero_pair();

    int lane  = tid & 31;
    unsigned full = 0xFFFFFFFFu;


    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        int i = tile_beg + j * blockDim.x + tid;
        float inc = 0.0f;




        float ci = 0.0f;
        if (i < series_len) ci = close[i];
        float cim1_warp = __shfl_up_sync(full, ci, 1);

        if (i < series_len) {
            if (i < fv) {
                out[base + i] = CUDART_NAN_F;
            } else if (i == fv) {
                out[base + i] = 0.0f;
            } else {


                float cim1 = (lane > 0) ? cim1_warp : ((i > 0) ? close[i - 1] : ci);

                int gt = (ci > cim1);
                int lt = (ci < cim1);
                float sgn = static_cast<float>(gt - lt);
                inc = sgn * volume[i];
            }
        }


        FPair v = {inc, 0.0f};
        FPair excl = block_exclusive_offset<NUM_WARPS>(v, warp_buf);
        FPair incl = fp_add_pair(excl, v);
        FPair full_prefix = fp_add_pair(seg_base, incl);

        if (i < series_len && i > fv) {
            out[base + i] = fp_collapse(full_prefix);
        }


        if (tid == (blockDim.x - 1)) {
            seg_sum_shared = incl;
        }
        __syncthreads();
        seg_base = fp_add_pair(seg_base, seg_sum_shared);
    }


    if (tid == 0) {
        block_sums[bid] = seg_base;
    }
}


extern "C" __global__
void obv_batch_f32_pass2_scan_block_sums(
    const FPair* __restrict__ block_sums,
    int num_tiles,
    FPair* __restrict__ block_offsets
){
    if (num_tiles <= 0) return;
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        FPair acc = make_zero_pair();
        for (int b = 0; b < num_tiles; ++b) {
            block_offsets[b] = acc;
            acc = fp_add_pair(acc, block_sums[b]);
        }
    }
}


extern "C" __global__
void obv_batch_f32_pass3_add_offsets(
    int series_len,
    int ,
    int first_valid,
    float* __restrict__ out,
    const FPair* __restrict__ block_offsets,
    int tiles_per_row
){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if (bid >= tiles_per_row) return;

    const int fv = first_valid < 0 ? 0 : first_valid;
    constexpr int ITEMS = OBV_ITEMS_PER_THREAD;
    const int tile_size = blockDim.x * ITEMS;
    const int tile_beg  = bid * tile_size;

    FPair off = block_offsets[bid];

    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        int i = tile_beg + j * blockDim.x + tid;
        if (i >= series_len) break;
        if (i <= fv) continue;

        FPair s = two_sum_fp32(out[i], off.hi);
        s = two_sum_fp32(s.hi, s.lo + off.lo);
        out[i] = fp_collapse(s);
    }
}


extern "C" __global__
void obv_batch_f32_replicate_rows(
    const float* __restrict__ row0,
    int series_len,
    int n_combos,
    float* __restrict__ out
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < series_len; i += stride) {
        float v = row0[i];
        for (int r = 1; r < n_combos; ++r) {
            out[r * series_len + i] = v;
        }
    }
}


extern "C" __global__
void obv_batch_f32_serial_ref(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) return;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int fv = first_valid < 0 ? 0 : first_valid;
    for (int i = tid; i < fv && i < series_len; i += stride) {
        out[combo * series_len + i] = CUDART_NAN_F;
    }

    if (tid == 0) {
        const int base = combo * series_len;
        if (fv < series_len) {
            out[base + fv] = 0.0f;

            FPair obv = make_zero_pair();
            float prev_close = close[fv];
            for (int i = fv + 1; i < series_len; ++i) {
                float c = close[i];
                float v = volume[i];
                int gt = (c > prev_close);
                int lt = (c < prev_close);
                float s = float(gt - lt);
                obv = fp_add_f(obv, s * v);
                out[base + i] = fp_collapse(obv);
                prev_close = c;
            }
        }
    }
}


extern "C" __global__ void obv_many_series_one_param_time_major_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || rows <= 0) return;

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];

    for (int t = 0; t < rows && t < fv; ++t) {
        out_tm[t * cols + s] = CUDART_NAN_F;
    }
    if (fv >= rows) return;

    int idx0 = fv * cols + s;
    out_tm[idx0] = 0.0f;


    FPair obv = make_zero_pair();
    float prev_close = close_tm[idx0];
    for (int t = fv + 1; t < rows; ++t) {
        int idx = t * cols + s;
        float c = close_tm[idx];
        float v = volume_tm[idx];
        int gt = (c > prev_close);
        int lt = (c < prev_close);
        float sgn = float(gt - lt);
        obv = fp_add_f(obv, sgn * v);
        out_tm[idx] = fp_collapse(obv);
        prev_close = c;
    }
}

