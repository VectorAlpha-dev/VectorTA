// CUDA kernels for Nadaraya–Watson Envelope (Gaussian kernel regression + MAE bands).
#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LDG
// On SM70+ (Volta and newer), default loads are cached (PTX ld.ca). Prefer normal loads.
// Keep __ldg for pre-Volta (SM<70) when beneficial.
#  if __CUDA_ARCH__ >= 700
#    define LDG(p) (*(p))
#  elif __CUDA_ARCH__ >= 350
#    define LDG(p) __ldg(p)
#  else
#    define LDG(p) (*(p))
#  endif
#endif

__device__ __forceinline__ float qnan_f32() { return __int_as_float(0x7fffffff); }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compensated FP32 helpers (Neumaier/Kahan-style) and warp/block reductions
///////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void kbn_acc(float &sum, float &c, float x) {
    // Neumaier's compensated sum: numerically robust with single compensation term.
    float t = sum + x;
    if (fabsf(sum) >= fabsf(x)) c += (sum - t) + x;
    else                        c += (x   - t) + sum;
    sum = t;
}

__device__ __forceinline__ void warp_reduce_kbn(float &sum, float &c, unsigned mask) {
    // Reduce (sum,c) pair across a warp using shuffle; combine via Neumaier.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float s2 = __shfl_down_sync(mask, sum, offset);
        float c2 = __shfl_down_sync(mask, c,   offset);
        // incorporate other warp lane's (sum,c)
        kbn_acc(sum, c, s2);
        kbn_acc(sum, c, c2);
    }
}

template<int MaxWarps>
__device__ __forceinline__ float block_reduce_kbn(float sum, float c, float *smem_sum, float *smem_comp) {
    // Per-warp reduce then reduce across warps using warp 0
    const unsigned mask = __activemask();
    warp_reduce_kbn(sum, c, mask);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) {
        smem_sum[warp]  = sum;
        smem_comp[warp] = c;
    }
    __syncthreads();

    float out_sum = 0.0f, out_comp = 0.0f;
    if (warp == 0) {
        // Number of warps actually used by the block
        const int num_warps = (blockDim.x + 31) >> 5;
        float v_sum = (lane < num_warps) ? smem_sum[lane]  : 0.0f;
        float v_comp= (lane < num_warps) ? smem_comp[lane] : 0.0f;
        warp_reduce_kbn(v_sum, v_comp, mask);
        if (lane == 0) { out_sum = v_sum; out_comp = v_comp; }
    }
    __syncthreads();
    // broadcast final result via shared memory slot 0
    if (threadIdx.x == 0) { smem_sum[0] = out_sum + out_comp; }
    __syncthreads();
    return smem_sum[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// One-series × many-params (primary optimized kernel)
// Each combo = one block. Threads cooperate on dot products; time is processed sequentially in tiles.
// Weights are pre-scaled by 1/den on host; weights are laid out with stride = max_lookback.
///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void nadaraya_watson_envelope_batch_f32(const float* __restrict__ data,
                                        const float* __restrict__ weights_flat,
                                        const int*   __restrict__ lookbacks,
                                        const float* __restrict__ multipliers,
                                        int series_len,
                                        int n_combos,
                                        int first_valid,
                                        int max_lookback,
                                        float* __restrict__ out_upper,
                                        float* __restrict__ out_lower)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    // --- Constants / policy ---
    const int MAE_LEN = 499;              // keep as in your original
    const int TILE_T  = 64;               // number of outputs computed per data tile
    const int L       = lookbacks[combo];
    const float mult  = multipliers[combo];

    if (L <= 0) return;

    const int warm_out   = first_valid + L - 1;
    const int warm_total = warm_out + MAE_LEN - 1;

    const int base  = combo * series_len;
    const int wbase = combo * max_lookback;

    // --- Parallel warm prefix fill with NaNs ---
    const int prefix = (warm_total < series_len) ? warm_total : series_len;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < prefix; i += blockDim.x) {
        out_upper[base + i] = qnan_f32();
        out_lower[base + i] = qnan_f32();
    }
    __syncthreads();
    if (warm_total >= series_len) return;

    // --- Shared memory layout ---
    // s_w      [0..L-1]                     -> per-combo weights
    // s_x      [0..(L+TILE_T-1)-1]          -> time tile of data
    // s_mask   [0..(L+TILE_T-1)-1]          -> 1.0f if s_x[i] is NaN else 0.0f
    // smem_sum/comp are small per-warp buffers reused for reductions
    extern __shared__ float s[];
    float *s_w    = s;
    float *s_x    = s_w + max_lookback;                   // reserve max_lookback for alignment simplicity
    float *s_mask = s_x + (max_lookback + TILE_T - 1);    // worst-case tile span
    // Per-warp reduction scratch (max 32 warps == 1024 threads)
    __shared__ float smem_sum[32], smem_comp[32];
    __shared__ float s_ring[MAE_LEN]; // rolling MAE ring is only touched by thread 0; keep in SMEM to avoid spills
    __shared__ int   s_nan_win_count; // broadcast window NaN-count to all threads

    // --- Load weights into shared memory once per combo ---
    for (int k = threadIdx.x; k < L; k += blockDim.x) {
        s_w[k] = LDG(&weights_flat[wbase + k]);
    }
    __syncthreads();

    // --- Initialize MAE ring (thread 0) ---
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < MAE_LEN; ++i) s_ring[i] = qnan_f32();
    }
    __syncthreads();

    // --- Sliding state for MAE (thread 0 owns it) ---
    int   mae_head   = 0;
    int   mae_filled = 0;
    float mae_sum    = 0.0f;
    int   mae_nan_ct = 0;

    // --- Process outputs in time tiles ---
    // We start at warm_out; each tile covers [t0, t0+tile-1].
    for (int t0 = warm_out; t0 < series_len; t0 += TILE_T)
    {
        const int tile_T = min(TILE_T, series_len - t0);
        const int tile_x_start = t0 - (L - 1);          // first x needed for y[t0]
        const int tile_x_end   = t0 + tile_T - 1;       // last x needed for y[t0 + tile_T-1]
        const int tile_span    = tile_x_end - tile_x_start + 1; // = L + tile_T - 1

        // --- Load the time tile into SMEM (coalesced) and build a NaN mask (0/1) ---
        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            float xi = LDG(&data[tile_x_start + i]);
            s_x[i]   = xi;
            // store mask as float 0.0f/1.0f so we can sum with reductions if needed
            s_mask[i]= isnan(xi) ? 1.0f : 0.0f;
        }
        __syncthreads();

        // --- Initial NaN count in the regression window for t = t0 ---
        if (threadIdx.x == 0) {
            int nc = 0;
            // window indices in s_mask: [0 .. L-1]
            for (int i = 0; i < L; ++i) nc += (s_mask[i] > 0.0f);
            s_nan_win_count = nc;
        }
        __syncthreads();

        // --- Loop over each output in this tile ---
        for (int u = 0; u < tile_T; ++u) {
            const int t        = t0 + u;
            const int x_off    = (L - 1) + u;   // s_x[x_off] corresponds to data[t]
            const bool window_ok = (s_nan_win_count == 0);

            float y = qnan_f32();

            if (window_ok) {
                // Parallel dot product: y = sum_{k=0..L-1} s_w[k] * s_x[x_off - k]
                float sum = 0.0f, comp = 0.0f;
                // Each thread accumulates a compensated partial sum over its k-stride
                for (int k = threadIdx.x; k < L; k += blockDim.x) {
                    // fmaf helps keep one rounding per product+sum path
                    float prod = s_w[k] * s_x[x_off - k];
                    kbn_acc(sum, comp, prod);
                }
                // Reduce across block with compensation; Max 32 warps supported here
                y = block_reduce_kbn<32>(sum, comp, smem_sum, smem_comp);
            }

            // --- Thread 0: update MAE ring and write outputs once past warm_total ---
            if (threadIdx.x == 0) {
                const float x_t = s_x[x_off];
                const bool y_isnan = isnan(y);
                const bool x_isnan = isnan(x_t);

                float resid = (!x_isnan && !y_isnan) ? fabsf(x_t - y) : qnan_f32();

                // pop if full
                if (mae_filled == MAE_LEN) {
                    float old = s_ring[mae_head];
                    if (isnan(old)) { if (mae_nan_ct > 0) mae_nan_ct -= 1; }
                    else            { mae_sum -= old; }
                } else {
                    mae_filled += 1;
                }

                // push new
                s_ring[mae_head] = resid;
                if (isnan(resid)) mae_nan_ct += 1; else mae_sum += resid;
                mae_head += 1; if (mae_head == MAE_LEN) mae_head = 0;

                // write when fully warmed
                if (t >= warm_total) {
                    float upper = qnan_f32();
                    float lower = qnan_f32();
                    if (mae_nan_ct == 0 && !y_isnan) {
                        float mae = (mae_sum / (float)MAE_LEN) * mult;
                        upper = y + mae;
                        lower = y - mae;
                    }
                    out_upper[base + t] = upper;
                    out_lower[base + t] = lower;
                }
            }
            __syncthreads();

            // --- Slide NaN window for next t: drop oldest, add newest ---
            if (u + 1 < tile_T) {
                if (threadIdx.x == 0) {
                    // window covers s_mask[x_off .. x_off-(L-1)]
                    // next window adds s_mask[(x_off+1)] and drops s_mask[(x_off-(L-1))]
                    int addv = (s_mask[x_off + 1] > 0.0f) ? 1 : 0;
                    int dropv= (s_mask[x_off - (L - 1)] > 0.0f) ? 1 : 0;
                    s_nan_win_count += addv - dropv;
                }
                __syncthreads();
            }
        } // tile outputs
    } // tiles
}


// Many-series × one-param, time-major layout (cols=series_len, rows=num_series).
// Minor cleanup: parallel warm fill; keep scalar inner loop to maintain simplicity.
// If you want the same block-level dot-product/tiling strategy here, it can be
// applied similarly, but this version already benefits from time-major coalescing.
extern "C" __global__
void nadaraya_watson_envelope_many_series_one_param_f32(const float* __restrict__ data_tm,
                                                        const float* __restrict__ weights,
                                                        int lookback,
                                                        float multiplier,
                                                        int num_series,
                                                        int series_len,
                                                        const int* __restrict__ first_valids,
                                                        float* __restrict__ out_upper_tm,
                                                        float* __restrict__ out_lower_tm)
{
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int L = lookback;
    const int MAE_LEN = 499;

    const int warm_out   = first_valids[series] + L - 1;
    const int warm_total = warm_out + MAE_LEN - 1;

    // Parallel warm fill
    for (int t = threadIdx.x; t < min(warm_total, series_len); t += blockDim.x) {
        const int idx = t * num_series + series;
        out_upper_tm[idx] = qnan_f32();
        out_lower_tm[idx] = qnan_f32();
    }
    __syncthreads();
    if (warm_total >= series_len) return;

    // Rolling MAE state (thread 0)
    __shared__ float ring[MAE_LEN];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < MAE_LEN; ++i) ring[i] = qnan_f32();
    }
    __syncthreads();

    int head = 0, filled = 0, nan_count = 0;
    float sum = 0.0f;

    // Single-thread time loop to keep state logic simple (others idle)
    if (threadIdx.x == 0) {
        for (int t = warm_out; t < series_len; ++t) {
            bool any_nan = false;
            // FP32 compensated accumulation
            float acc = 0.0f, c = 0.0f;
            #pragma unroll 1
            for (int k = 0; k < L; ++k) {
                int idx = (t - k) * num_series + series;
                float x = LDG(&data_tm[idx]);
                if (isnan(x)) { any_nan = true; break; }
                float wk = LDG(&weights[k]);
                float prod = x * wk;
                float tmp = acc + prod;
                if (fabsf(acc) >= fabsf(prod)) c += (acc - tmp) + prod;
                else                            c += (prod - tmp) + acc;
                acc = tmp;
            }
            float y = any_nan ? qnan_f32() : (acc + c);

            const int idx_t = t * num_series + series;
            float x_t = LDG(&data_tm[idx_t]);
            float resid = (!isnan(x_t) && !isnan(y)) ? fabsf(x_t - y) : qnan_f32();

            if (filled == MAE_LEN) {
                float old = ring[head];
                if (isnan(old)) { if (nan_count > 0) nan_count -= 1; } else { sum -= old; }
            } else {
                filled += 1;
            }
            ring[head] = resid;
            if (isnan(resid)) nan_count += 1; else sum += resid;
            head += 1; if (head == MAE_LEN) head = 0;

            if (t >= warm_total) {
                float upper = qnan_f32();
                float lower = qnan_f32();
                if (nan_count == 0 && !isnan(y)) {
                    float mae = (sum / (float)MAE_LEN) * multiplier;
                    upper = y + mae;
                    lower = y - mae;
                }
                out_upper_tm[idx_t] = upper;
                out_lower_tm[idx_t] = lower;
            }
        }
    }
}

