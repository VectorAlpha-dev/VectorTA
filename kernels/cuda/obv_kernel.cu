// CUDA kernels for On-Balance Volume (OBV) — Parallel scan + dual-FP32 accumulator.
// Targets CUDA 13 / Ada+ (e.g., RTX 4090). No FP64 used in the fast path.
// Shapes kept compatible with "batch" layout: out is [n_combos][series_len] row-major.
//
// Math recap:
//   inc[i] = sign(close[i] - close[i-1]) * volume[i]              for i > first_valid
//          = 0                                                    for i <= first_valid
//   OBV[i] = inclusive_scan(inc)[i], with out[i<fv]=NaN, out[fv]=0.
//
// Strategy (3-pass):
//   Pass 1: Per-block tile scan on 'inc' directly into out row 0. Also writes per-block totals (float2).
//   Pass 2: Scan per-block totals (exclusive) to produce offsets (float2).
//   Pass 3: Add each block's offset to its tile.
// If n_combos > 1, replicate row 0 -> other rows (GPU copy kernel).
//
// Notes:
//  - Dual-FP32 ("double-single"): error-free transforms (TwoSum) to carry a residual.
//  - Warp shuffles provide close[i-1] for most lanes; lane 0 in each warp does a single fallback global read.
//  - Heuristic: wrapper should pick fast path for series_len >= ~4096; otherwise use serial fallback.

#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef OBV_BLOCK_SIZE
#define OBV_BLOCK_SIZE 256
#endif
#ifndef OBV_ITEMS_PER_THREAD
#define OBV_ITEMS_PER_THREAD 8
#endif

// --- Dual-FP32 accumulation primitives (error-free transform) ----------------
struct FPair { float hi, lo; };  // value ≈ hi + lo

__device__ __forceinline__ FPair make_zero_pair() { return {0.0f, 0.0f}; }

__device__ __forceinline__ FPair two_sum_fp32(float a, float b) {
    // Error-free transform of a+b into hi+lo where hi = fl(a+b), lo = exact error.
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

// --- Warp/block scan helpers over FPair --------------------------------------
__device__ __forceinline__ FPair warp_inclusive_scan(FPair v, unsigned mask) {
    int lane = threadIdx.x & 31;
    // Kogge–Stone inclusive scan using shuffles
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float hi = __shfl_up_sync(mask, v.hi, offset);
        float lo = __shfl_up_sync(mask, v.lo, offset);
        if (lane >= offset) v = fp_add_pair(v, {hi, lo});
    }
    return v;
}

// Returns: exclusive offset for this thread's "thread_total". Also
//          writes per-warp sums to shared for computing block totals.
template<int NUM_WARPS>
__device__ __forceinline__
FPair block_exclusive_offset(FPair thread_total, FPair* warp_buf) {
    unsigned full = 0xFFFFFFFFu;
    int lane  = threadIdx.x & 31;
    int wid   = threadIdx.x >> 5;

    FPair incl = warp_inclusive_scan(thread_total, full);
    // Save per-warp inclusive sum in the last lane
    if (lane == 31) warp_buf[wid] = incl;
    __syncthreads();

    // First warp scans warp sums to build exclusive warp offsets
    if (wid == 0) {
        FPair x = (lane < NUM_WARPS) ? warp_buf[lane] : make_zero_pair();
        FPair x_incl = warp_inclusive_scan(x, full);
        // exclusive = inclusive - self
        FPair x_excl = fp_sub_pair(x_incl, x);
        if (lane < NUM_WARPS) warp_buf[lane] = x_excl;
    }
    __syncthreads();

    // exclusive offset for this thread = warp_offset + (inclusive - self)
    FPair warp_off = warp_buf[wid];
    FPair excl_intra = fp_sub_pair(incl, thread_total);
    return fp_add_pair(warp_off, excl_intra);
}

// --- PASS 1: compute increments, scan per tile, write partial OBV + block sums
//   - close, volume: one series (length series_len)
//   - Writes out row 0 (combo 0). If you want N rows, call replicate later.
extern "C" __global__
void obv_batch_f32_pass1_tilescan(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int series_len,
    int /*n_combos*/,   // kept for shape parity; we only compute row 0 here
    int first_valid,
    float* __restrict__ out,
    FPair* __restrict__ block_sums,   // size = num_tiles (gridDim.x)
    int tiles_per_row                  // = gridDim.x at launch
){
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;  // tile id along time
    // only produce the first row; others are later replicated (saves recompute)
    const int base = 0; // combo 0 row base offset

    if (series_len <= 0 || bid >= tiles_per_row) return;
    const int fv = first_valid < 0 ? 0 : first_valid;

    constexpr int ITEMS = OBV_ITEMS_PER_THREAD;
    const int tile_size = blockDim.x * ITEMS;
    const int tile_beg  = bid * tile_size;
    const int tile_end  = min(series_len, tile_beg + tile_size);

    // Shared memory for warp prefixes
    constexpr int NUM_WARPS = (OBV_BLOCK_SIZE + 31) / 32;
    __shared__ FPair warp_buf[NUM_WARPS];
    __shared__ FPair seg_sum_shared;

    // Running sum of all prior segments (each segment is blockDim.x items).
    FPair seg_base = make_zero_pair();

    int lane  = tid & 31;
    unsigned full = 0xFFFFFFFFu;

    // Scan each coalesced segment (j) in time order, carrying seg_base forward.
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        int i = tile_beg + j * blockDim.x + tid;
        float inc = 0.0f;

        // NOTE: All lanes must participate in warp shuffles. We pre-load close[i]
        // (or 0 for out-of-range) and compute the shuffle unconditionally, then
        // only use it when i is in-range and past warmup.
        float ci = 0.0f;
        if (i < series_len) ci = close[i];
        float cim1_warp = __shfl_up_sync(full, ci, 1);

        if (i < series_len) {
            if (i < fv) {
                out[base + i] = CUDART_NAN_F;  // warmup NaN
            } else if (i == fv) {
                out[base + i] = 0.0f;
            } else {
                // Compute sign(close[i] - close[i-1]) * volume[i]
                // obtain close[i-1] via warp shuffle; lane 0 falls back to global
                float cim1 = (lane > 0) ? cim1_warp : ((i > 0) ? close[i - 1] : ci);
                // branchless sign in {-1,0,+1}
                int gt = (ci > cim1);
                int lt = (ci < cim1);
                float sgn = static_cast<float>(gt - lt);
                inc = sgn * volume[i];
            }
        }

        // Inclusive scan of this segment across threads (time order within segment).
        FPair v = {inc, 0.0f};
        FPair excl = block_exclusive_offset<NUM_WARPS>(v, warp_buf);
        FPair incl = fp_add_pair(excl, v);
        FPair full_prefix = fp_add_pair(seg_base, incl);

        if (i < series_len && i > fv) {
            out[base + i] = fp_collapse(full_prefix);
        }

        // Segment total (sum over all threads) is the last thread's inclusive value.
        if (tid == (blockDim.x - 1)) {
            seg_sum_shared = incl;
        }
        __syncthreads();
        seg_base = fp_add_pair(seg_base, seg_sum_shared);
    }

    // Total tile sum (used by pass2) is the sum of all segments.
    if (tid == 0) {
        block_sums[bid] = seg_base;
    }
}

// --- PASS 2: scan block totals (per row 0) to produce exclusive offsets
extern "C" __global__
void obv_batch_f32_pass2_scan_block_sums(
    const FPair* __restrict__ block_sums,   // len = num_tiles
    int num_tiles,
    FPair* __restrict__ block_offsets       // len = num_tiles (exclusive)
){
    if (num_tiles <= 0) return;
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        FPair acc = make_zero_pair();
        for (int b = 0; b < num_tiles; ++b) {
            block_offsets[b] = acc;            // exclusive offset for tile b
            acc = fp_add_pair(acc, block_sums[b]);
        }
    }
}

// --- PASS 3: add per-tile offsets to the partial results
extern "C" __global__
void obv_batch_f32_pass3_add_offsets(
    int series_len,
    int /*n_combos*/,
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
        if (i <= fv) continue; // preserve NaN and 0
        // Accumulate offset into existing float using error-free transform
        FPair s = two_sum_fp32(out[i], off.hi);
        s = two_sum_fp32(s.hi, s.lo + off.lo);
        out[i] = fp_collapse(s);
    }
}

// --- Optional: replicate row 0 to [1..n_combos-1]
extern "C" __global__
void obv_batch_f32_replicate_rows(
    const float* __restrict__ row0,
    int series_len,
    int n_combos,
    float* __restrict__ out // destination is [n_combos][series_len]
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

// --- Serial fallback for tiny series (kept for parity & testing) --------------
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
            // Dual-FP32 running sum (no FP64)
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

// --- Many-series × one-param (time-major) — light cleanup, avoid FP64 --------
extern "C" __global__ void obv_many_series_one_param_time_major_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int*   __restrict__ first_valids,
    int cols, // number of series
    int rows, // series length
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series id
    if (s >= cols || rows <= 0) return;

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];

    for (int t = 0; t < rows && t < fv; ++t) {
        out_tm[t * cols + s] = CUDART_NAN_F;
    }
    if (fv >= rows) return;

    int idx0 = fv * cols + s;
    out_tm[idx0] = 0.0f;

    // Dual-FP32 accumulation per series (serial along time)
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

