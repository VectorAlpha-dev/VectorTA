


#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <cub/block/block_radix_sort.cuh>



#ifndef NEG_INF_F
#define NEG_INF_F (-1.0f / 0.0f)
#endif




static __device__ __forceinline__ int sgn_eps(float f, float s, float eps) {
    return (f > s + eps) - (f < s - eps);
}

static __device__ __forceinline__ float ld_tm(const float* __restrict__ a, int dim, int t, int i) {
#if __CUDA_ARCH__ >= 350
    return __ldg(a + (size_t)t * (size_t)dim + (size_t)i);
#else
    return a[(size_t)t * (size_t)dim + (size_t)i];
#endif
}






extern "C" __global__
void transpose_row_to_tm(const float* __restrict__ in,
                         int rows, int cols,
                         float* __restrict__ out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    for (int k = tid; k < total; k += blockDim.x * gridDim.x) {
        const int r = k / cols;
        const int c = k % cols;
        out[(size_t)c * (size_t)rows + (size_t)r] = in[(size_t)r * (size_t)cols + (size_t)c];
    }
}





extern "C" __global__
void compute_log_returns_f32(const float* __restrict__ prices,
                             int T,
                             float* __restrict__ lr_out)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < T; t += blockDim.x * gridDim.x) {
        if (t == 0) { lr_out[0] = 0.0f; continue; }
        const float p  = prices[t];
        const float pm = prices[t-1];
        lr_out[t] = (p > 0.0f && pm > 0.0f && isfinite(p) && isfinite(pm)) ? logf(p) - logf(pm) : 0.0f;
    }
}




#define STRAT_LONG_ONLY              (1u<<0)
#define STRAT_NO_FLIP                (1u<<1)
#define STRAT_TRADE_ON_NEXT_BAR      (1u<<2)
#define STRAT_ENFORCE_FAST_LT_SLOW   (1u<<3)
#define STRAT_SIGNED_EXPOSURE        (1u<<4)





extern "C" __global__
void double_cross_backtest_tm_flex_f32(
    const float* __restrict__ fast_ma_T,
    const float* __restrict__ slow_ma_T,
    const float* __restrict__ lr,

    int T,
    int Pf_tile, int Ps_tile,
    int Pf_total, int Ps_total,
    int f_offset, int s_offset,

    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int first_valid,

    float commission,
    float eps_rel,
    unsigned int flags,
    int M,
    float* __restrict__ metrics_out
){
    const float log_comm = (commission > 0.0f) ? log1pf(-commission) : 0.0f;
    const int need_trades = (M > 1);
    const int need_max_dd = (M > 2);
    const int need_stats  = (M > 3);
    const int need_std    = (M > 4);
    const int need_expo   = (M > 5);

    const int total_pairs = Pf_tile * Ps_tile;
    for (int pair_local = blockIdx.x * blockDim.x + threadIdx.x;
         pair_local < total_pairs;
         pair_local += blockDim.x * gridDim.x)
    {
        const int pf_local  = pair_local / Ps_tile;
        const int ps_local  = pair_local % Ps_tile;
        const int pf_global = f_offset + pf_local;
        const int ps_global = s_offset + ps_local;

        const int period_f = fast_periods[pf_global];
        const int period_s = slow_periods[ps_global];

        const size_t pair_global = (size_t)pf_global * (size_t)Ps_total + (size_t)ps_global;
        const size_t base = pair_global * (size_t)M;

        if ((flags & STRAT_ENFORCE_FAST_LT_SLOW) && (period_f >= period_s)) {
            if (M > 0) metrics_out[base + 0] = 0.0f;
            if (M > 1) metrics_out[base + 1] = 0.0f;
            if (M > 2) metrics_out[base + 2] = 0.0f;
            if (M > 3) metrics_out[base + 3] = 0.0f;
            if (M > 4) metrics_out[base + 4] = 0.0f;
            if (M > 5) metrics_out[base + 5] = 0.0f;
            if (M > 6) metrics_out[base + 6] = 0.0f;
            continue;
        }

        const int t_valid = first_valid + max(period_f, period_s) - 1;
        const int start_t = t_valid + ((flags & STRAT_TRADE_ON_NEXT_BAR) ? 1 : 0);

        if (start_t >= T) {
            if (M > 0) metrics_out[base + 0] = 0.0f;
            if (M > 1) metrics_out[base + 1] = 0.0f;
            if (M > 2) metrics_out[base + 2] = 0.0f;
            if (M > 3) metrics_out[base + 3] = 0.0f;
            if (M > 4) metrics_out[base + 4] = 0.0f;
            if (M > 5) metrics_out[base + 5] = 0.0f;
            if (M > 6) metrics_out[base + 6] = 0.0f;
            continue;
        }

        int   pos         = 0;
        float log_eq      = 0.0f;
        float log_peak    = 0.0f;
        float max_log_dd  = 0.0f;
        int   trades      = 0;
        long long sum_abs_pos = 0;
        long long sum_pos     = 0;

        float mean = 0.0f, m2 = 0.0f;
        long long n = 0;

        for (int t = start_t; t < T; ++t) {
            const int t_sig = t - ((flags & STRAT_TRADE_ON_NEXT_BAR) ? 1 : 0);

            const float f = ld_tm(fast_ma_T, Pf_tile, t_sig, pf_local);
            const float s = ld_tm(slow_ma_T, Ps_tile, t_sig, ps_local);
            const float eps = (eps_rel > 0.0f) ? (eps_rel * fmaxf(1.0f, fabsf(s))) : 0.0f;

            int sign = sgn_eps(f, s, eps);
            if (flags & STRAT_LONG_ONLY) sign = max(sign, 0);

            if (sign != pos) {
                if (pos == 0 || sign == 0) {
                    if (commission > 0.0f) log_eq += log_comm;
                    if (need_trades) trades += 1;
                    pos = (sign == 0) ? 0 : sign;
                } else {
                    if (flags & STRAT_NO_FLIP) {
                        if (commission > 0.0f) log_eq += log_comm;
                        if (need_trades) trades += 1;
                        pos = 0;
                    } else {
                        if (commission > 0.0f) log_eq += 2.0f * log_comm;
                        if (need_trades) trades += 2;
                        pos = sign;
                    }
                }
            }

            const float lr_t = lr[t];
            log_eq += (float)pos * lr_t;

            if (need_max_dd) {
                if (log_eq > log_peak) log_peak = log_eq;
                const float dd_log = log_peak - log_eq;
                if (dd_log > max_log_dd) max_log_dd = dd_log;
            }

            if (need_stats) {
                const float step_r = (pos == 0) ? 0.0f : (float)pos * expm1f(lr_t);
                n += 1;
                const float delta = step_r - mean;
                mean += delta / (float)n;
                m2   += delta * (step_r - mean);

                if (need_expo) {
                    sum_abs_pos += llabs((long long)pos);
                    sum_pos     += (long long)pos;
                }
            }
        }

        const float variance = (need_std && n > 1) ? (m2 / (float)(n - 1)) : 0.0f;
        const float max_dd = (need_max_dd && max_log_dd > 0.0f) ? -expm1f(-max_log_dd) : 0.0f;
        const float exposure = (need_expo && n > 0) ? (float)((double)sum_abs_pos / (double)n) : 0.0f;
        const float net_expo = (need_expo && n > 0) ? (float)((double)sum_pos / (double)n)     : 0.0f;

        if (M > 0) metrics_out[base + 0] = expf(log_eq) - 1.0f;
        if (M > 1) metrics_out[base + 1] = (float)trades;
        if (M > 2) metrics_out[base + 2] = max_dd;
        if (M > 3) metrics_out[base + 3] = mean;
        if (M > 4) metrics_out[base + 4] = sqrtf(variance);
        if (M > 5) metrics_out[base + 5] = exposure;
        if (M > 6) metrics_out[base + 6] = (flags & STRAT_SIGNED_EXPOSURE) ? net_expo : 0.0f;
    }
}



static __device__ __forceinline__ float objective_score_f32(const float* __restrict__ metrics, size_t base, int M, int objective) {
    // objective: 0 = pnl, 1 = sharpe, 2 = -max_dd
    if (objective == 0) {
        return (M > 0) ? metrics[base + 0] : NEG_INF_F;
    }
    if (objective == 1) {
        if (M <= 4) return NEG_INF_F;
        const float mean = metrics[base + 3];
        const float std  = metrics[base + 4];
        return (std > 0.0f) ? (mean / std) : 0.0f;
    }
    if (objective == 2) {
        return (M > 2) ? -metrics[base + 2] : NEG_INF_F;
    }
    return NEG_INF_F;
}

static __device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    if (!isfinite(value)) return;
    int* addr_i = (int*)addr;
    int old = *addr_i;
    while (true) {
        const float old_f = __int_as_float(old);
        if (!(value > old_f)) break;
        const int assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

static __device__ __forceinline__ int bin_period_i32(int v, int v_min, int v_max, int bins) {
    if (bins <= 1) return 0;
    const int denom = v_max - v_min;
    if (denom <= 0) return 0;
    const int num = v - v_min;
    long long b = ((long long)num * (long long)(bins - 1)) / (long long)denom;
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    return (int)b;
}

extern "C" __global__
void fill_f32(float* __restrict__ out, int N, float value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        out[i] = value;
    }
}

extern "C" __global__
void heatmap_update_scores_tm_f32(
    const float* __restrict__ metrics,
    int pairs,
    int M,
    int Pf_tile,
    int Ps_tile,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int objective,
    int bins_fast,
    int bins_slow,
    int fast_min,
    int fast_max,
    int slow_min,
    int slow_max,
    float* __restrict__ heatmap_out
) {
    const int stride = blockDim.x * gridDim.x;
    for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < pairs; pair += stride) {
        const int pf_local = pair / Ps_tile;
        const int ps_local = pair - pf_local * Ps_tile;
        if (pf_local < 0 || pf_local >= Pf_tile || ps_local < 0 || ps_local >= Ps_tile) continue;

        const int f = fast_periods[pf_local];
        const int s = slow_periods[ps_local];
        if (f >= s) continue;

        const size_t base = (size_t)pair * (size_t)M;
        const float score = objective_score_f32(metrics, base, M, objective);
        if (!isfinite(score)) continue;

        const int fb = bin_period_i32(f, fast_min, fast_max, bins_fast);
        const int sb = bin_period_i32(s, slow_min, slow_max, bins_slow);
        const int idx = fb * bins_slow + sb;
        atomicMaxFloat(&heatmap_out[idx], score);
    }
}

#define TOPK_BLOCK_THREADS 256
#define TOPK_ITEMS_PER_THREAD 2
#define TOPK_BLOCK_ITEMS (TOPK_BLOCK_THREADS * TOPK_ITEMS_PER_THREAD)

extern "C" __global__
void select_topk_from_pairs_tm_f32(
    const float* __restrict__ metrics,
    int pairs,
    int M,
    int Pf_tile,
    int Ps_tile,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int objective,
    int topk,
    int* __restrict__ out_pair_idx
) {
    const int block_start = (int)blockIdx.x * (int)TOPK_BLOCK_ITEMS;
    float keys[TOPK_ITEMS_PER_THREAD];
    int vals[TOPK_ITEMS_PER_THREAD];

#pragma unroll
    for (int item = 0; item < TOPK_ITEMS_PER_THREAD; ++item) {
        const int idx = block_start + (int)threadIdx.x * TOPK_ITEMS_PER_THREAD + item;
        float score = NEG_INF_F;
        int val = -1;
        if (idx < pairs) {
            const int pf_local = idx / Ps_tile;
            const int ps_local = idx - pf_local * Ps_tile;
            if (pf_local >= 0 && pf_local < Pf_tile && ps_local >= 0 && ps_local < Ps_tile) {
                const int f = fast_periods[pf_local];
                const int s = slow_periods[ps_local];
                if (f < s) {
                    const size_t base = (size_t)idx * (size_t)M;
                    score = objective_score_f32(metrics, base, M, objective);
                    if (!isfinite(score)) score = NEG_INF_F;
                }
            }
            val = idx;
        }
        keys[item] = score;
        vals[item] = val;
    }

    typedef cub::BlockRadixSort<float, TOPK_BLOCK_THREADS, TOPK_ITEMS_PER_THREAD, int> BlockSort;
    __shared__ typename BlockSort::TempStorage temp_storage;
    BlockSort(temp_storage).SortDescending(keys, vals);

#pragma unroll
    for (int item = 0; item < TOPK_ITEMS_PER_THREAD; ++item) {
        const int rank = (int)threadIdx.x * TOPK_ITEMS_PER_THREAD + item;
        if (rank < topk) {
            out_pair_idx[(int)blockIdx.x * topk + rank] = vals[item];
        }
    }
}

extern "C" __global__
void select_topk_from_candidates_tm_f32(
    const float* __restrict__ metrics,
    int pairs,
    int M,
    int Pf_tile,
    int Ps_tile,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    int objective,
    int topk,
    const int* __restrict__ in_pair_idx,
    int in_len,
    int* __restrict__ out_pair_idx
) {
    const int block_start = (int)blockIdx.x * (int)TOPK_BLOCK_ITEMS;
    float keys[TOPK_ITEMS_PER_THREAD];
    int vals[TOPK_ITEMS_PER_THREAD];

#pragma unroll
    for (int item = 0; item < TOPK_ITEMS_PER_THREAD; ++item) {
        const int in_pos = block_start + (int)threadIdx.x * TOPK_ITEMS_PER_THREAD + item;
        int pair = -1;
        float score = NEG_INF_F;
        if (in_pos < in_len) {
            pair = in_pair_idx[in_pos];
            if (pair >= 0 && pair < pairs) {
                const int pf_local = pair / Ps_tile;
                const int ps_local = pair - pf_local * Ps_tile;
                if (pf_local >= 0 && pf_local < Pf_tile && ps_local >= 0 && ps_local < Ps_tile) {
                    const int f = fast_periods[pf_local];
                    const int s = slow_periods[ps_local];
                    if (f < s) {
                        const size_t base = (size_t)pair * (size_t)M;
                        score = objective_score_f32(metrics, base, M, objective);
                        if (!isfinite(score)) score = NEG_INF_F;
                    }
                }
            }
        }
        keys[item] = score;
        vals[item] = pair;
    }

    typedef cub::BlockRadixSort<float, TOPK_BLOCK_THREADS, TOPK_ITEMS_PER_THREAD, int> BlockSort;
    __shared__ typename BlockSort::TempStorage temp_storage;
    BlockSort(temp_storage).SortDescending(keys, vals);

#pragma unroll
    for (int item = 0; item < TOPK_ITEMS_PER_THREAD; ++item) {
        const int rank = (int)threadIdx.x * TOPK_ITEMS_PER_THREAD + item;
        if (rank < topk) {
            out_pair_idx[(int)blockIdx.x * topk + rank] = vals[item];
        }
    }
}

extern "C" __global__
void gather_topk_pairs_tm_f32(
    const float* __restrict__ metrics,
    int pairs,
    int M,
    int Pf_tile,
    int Ps_tile,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const int* __restrict__ top_pair_idx,
    int topk,
    int* __restrict__ out_fast_len,
    int* __restrict__ out_slow_len,
    float* __restrict__ out_metrics
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < topk; i += blockDim.x * gridDim.x) {
        const int pair = top_pair_idx[i];
        if (pair < 0 || pair >= pairs) {
            out_fast_len[i] = 0;
            out_slow_len[i] = 0;
            for (int m = 0; m < M; ++m) out_metrics[(size_t)i * (size_t)M + (size_t)m] = NAN;
            continue;
        }
        const int pf_local = pair / Ps_tile;
        const int ps_local = pair - pf_local * Ps_tile;
        if (pf_local < 0 || pf_local >= Pf_tile || ps_local < 0 || ps_local >= Ps_tile) {
            out_fast_len[i] = 0;
            out_slow_len[i] = 0;
            for (int m = 0; m < M; ++m) out_metrics[(size_t)i * (size_t)M + (size_t)m] = NAN;
            continue;
        }
        out_fast_len[i] = fast_periods[pf_local];
        out_slow_len[i] = slow_periods[ps_local];
        const size_t base_in = (size_t)pair * (size_t)M;
        const size_t base_out = (size_t)i * (size_t)M;
        for (int m = 0; m < M; ++m) out_metrics[base_out + (size_t)m] = metrics[base_in + (size_t)m];
    }
}
