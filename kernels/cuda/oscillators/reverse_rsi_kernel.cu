// CUDA kernels for Reverse RSI (price level that would yield a target RSI)
//
// Semantics mirror the scalar implementation in src/indicators/reverse_rsi.rs:
// - Warmup index: warm_idx = first_valid + (2*rsi_length-1) - 1
// - Outputs before warm_idx are NaN
// - Warmup seed uses SMA of gains/losses over ema_len = 2*n - 1, which yields
//   alpha = 2/(ema_len+1) == 1/n (Wilder smoothing equivalence)
// - Subsequent outputs use EMA recurrence on up/down with alpha/beta
// - Let n = rsi_length, L = rsi_level (0<L<100).
//   rs_target = L/(100-L), rs_coeff = (n-1)*rs_target, neg_scale=(100-L)/L
//   x = rs_coeff*down_ema - (n-1)*up_ema
//   scale = (x >= 0 ? 1.0 : neg_scale)
//   out[i] = price[i] + x * scale
// - If out[i] is non-finite and x < 0.0, write 0.0 (matches scalar guard)
// - Deltas treat NaNs as no-change: diff=0.0 if cur or prev is not finite

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

#ifndef RRSI_NAN
#define RRSI_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ bool is_finite_f(float x) { return isfinite(x); }

// Fast bitwise finiteness check for IEEE-754 float (true iff not NaN/Inf)
static __device__ __forceinline__ bool is_finite_bits(float x) {
    return ((__float_as_uint(x) & 0x7f800000u) != 0x7f800000u);
}

// Kahan-compensated add: sum += x (updating compensation c)
static __device__ __forceinline__ void kahan_add(float x, float& sum, float& c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

extern "C" __global__ void reverse_rsi_batch_f32(
    const float* __restrict__ prices,   // one series (FP32)
    const int*   __restrict__ lengths,  // rsi_length per combo
    const float* __restrict__ levels,   // rsi_level per combo (0<L<100)
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out             // length = n_combos * series_len
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    // Per-thread parameters
    const int n = lengths[combo];
    const float L = levels[combo];
    float* out_row = out + combo * series_len;

    // Validate inputs; mirror scalar guard semantics with NaN outputs.
    if (UNLIKELY(n <= 0 || !(L > 0.0f && L < 100.0f) || !is_finite_bits(L))) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }

    const int ema_len = (2 * n) - 1;
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail <= ema_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }

    // Indices
    const int warm_end = first_valid + ema_len; // exclusive
    const int warm_idx = warm_end - 1;

    // Constants (pure FP32 path)
    const float alpha = 1.0f / float(n);       // == 2/(ema_len+1)
    const float beta  = 1.0f - alpha;
    const float inv   = 100.0f - L;
    const float rs_target = L / inv;           // L / (100-L)
    const float neg_scale = inv / L;           // (100-L)/L
    const float n_minus_1 = float(n - 1);
    const float rs_coeff  = n_minus_1 * rs_target;

#if __CUDA_ARCH__ >= 800
    // Ampere+/Ada path: async copy to shared + double-buffered pipeline
    constexpr int TILE = 256; // tuned for sm_89; 2*TILE + TILE + TILE = 1024 floats (4 KB)

    cg::thread_block cta = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;
    auto pipe = cuda::make_pipeline(cta, &pss);

    // Dynamic shared memory region layout:
    // [0 .. 2*TILE) -> double-buffered price tiles
    // [2*TILE .. 3*TILE) -> up[] tile
    // [3*TILE .. 4*TILE) -> dn[] tile
    extern __shared__ float smem[];
    float* prices_buf = smem;
    float* up_buf     = smem + 2 * TILE;
    float* dn_buf     = smem + 3 * TILE;

    const int num_tiles = (series_len + TILE - 1) / TILE;

    // Producer helper
    auto prefetch_tile = [&](int t) {
        if (t >= num_tiles) return;
        const int stage = t & 1;
        float* dst  = prices_buf + stage * TILE;
        const int g0   = t * TILE;
        const int len  = min(TILE, series_len - g0);

        pipe.producer_acquire();
        // Cooperative async copy: each thread pulls a strided subset
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            cuda::memcpy_async(cta, dst + i, prices + g0 + i, sizeof(float), pipe);
        }
        pipe.producer_commit();
    };

    // Warmup accumulators (Kahan) and EMA state (per thread)
    float sum_up = 0.f, c_up = 0.f;
    float sum_dn = 0.f, c_dn = 0.f;
    float up_ema = 0.f, dn_ema = 0.f;

    // Cross-tile delta carry for building up/dn (shared, built once per tile)
    __shared__ float    prev_carry;
    __shared__ unsigned prev_carry_is_finite;
    if (threadIdx.x == 0) { prev_carry = 0.0f; prev_carry_is_finite = 1u; }
    __syncthreads();

    // Prefetch first tile
    prefetch_tile(0);

    for (int t = 0; t < num_tiles; ++t) {
        // Wait for current tile, then expose it to all threads
        pipe.consumer_wait();
        __syncthreads();

        const int stage = t & 1;
        float* p = prices_buf + stage * TILE;
        const int start = t * TILE;
        const int len   = min(TILE, series_len - start);

        // Build per-sample up/dn once (thread 0), honoring NaN/Inf semantics
        if (threadIdx.x == 0) {
            float    prev = prev_carry;
            unsigned pfin = prev_carry_is_finite;
            for (int j = 0; j < len; ++j) {
                const int r = start + j;
                const float cur = p[j];
                const unsigned cfin = is_finite_bits(cur);
                float d = 0.0f;
                if (r >= first_valid) {
                    // first delta uses prev=0.0 (finite true), otherwise prior sample
                    const float    prev_used = (r == first_valid) ? 0.0f : prev;
                    const unsigned p_used_ok = (r == first_valid) ? 1u   : pfin;
                    d = (cfin & p_used_ok) ? (cur - prev_used) : 0.0f;
                    up_buf[j] = d > 0.0f ? d : 0.0f;
                    dn_buf[j] = d < 0.0f ? -d : 0.0f;
                } else {
                    up_buf[j] = 0.0f; dn_buf[j] = 0.0f;
                }
                prev = cur; pfin = cfin;
            }
            if (len > 0) {
                prev_carry = p[len - 1];
                prev_carry_is_finite = is_finite_bits(prev_carry);
            }
        }
        __syncthreads();

        // While consuming this tile, prefetch the next
        prefetch_tile(t + 1);

        // Consume tile for this thread's parameter combo
        for (int j = 0; j < len; ++j) {
            const int r = start + j;

            if (r < warm_idx) {
                out_row[r] = RRSI_NAN;
                continue;
            }

            if (r < warm_end) {
                // Warmup: Kahan sums for SMA of up/down over ema_len samples
                kahan_add(up_buf[j], sum_up, c_up);
                kahan_add(dn_buf[j], sum_dn, c_dn);

                if (r == warm_idx) {
                    up_ema = sum_up / float(ema_len);
                    dn_ema = sum_dn / float(ema_len);

                    const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
                    const float m = (x >= 0.0f) ? 1.0f : 0.0f;
                    const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
                    const float v = p[j] + x * scale;
                    out_row[r] = (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
                } else {
                    out_row[r] = RRSI_NAN;
                }
            } else {
                // EMA recurrence with FMA: y += alpha*(x - y)
                up_ema = fmaf(alpha, (up_buf[j] - up_ema), up_ema);
                dn_ema = fmaf(alpha, (dn_buf[j] - dn_ema), dn_ema);

                const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
                const float m = (x >= 0.0f) ? 1.0f : 0.0f;
                const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
                const float v = p[j] + x * scale;
                out_row[r] = (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
            }
        }

        pipe.consumer_release();
    }

#else
    // Fallback path (pre-Ampere): still FP32 only, Kahan warmup, FMA EMA
    // Fill NaNs up to warm_idx
    for (int i = 0; i < warm_idx; ++i) out_row[i] = RRSI_NAN;

    float sum_up = 0.f, c_up = 0.f;
    float sum_dn = 0.f, c_dn = 0.f;

    float prev = 0.0f; // first delta uses prev=0.0
    for (int r = first_valid; r < warm_end; ++r) {
        const float cur = prices[r];
        const unsigned ok = (is_finite_bits(cur) & is_finite_bits(prev));
        const float d = ok ? (cur - prev) : 0.0f;
        kahan_add((d > 0.f) ? d : 0.f, sum_up, c_up);
        kahan_add((d < 0.f) ? -d : 0.f, sum_dn, c_dn);
        prev = cur;
    }
    float up_ema = sum_up / float(ema_len);
    float dn_ema = sum_dn / float(ema_len);

    // First output at warm_idx
    {
        const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const float m = (x >= 0.0f) ? 1.0f : 0.0f;
        const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
        const float v = prices[warm_idx] + x * scale;
        out_row[warm_idx] = (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
    }

    float prevd = prices[warm_idx];
    for (int r = warm_end; r < series_len; ++r) {
        const float cur = prices[r];
        const unsigned ok = (is_finite_bits(cur) & is_finite_bits(prevd));
        const float d = ok ? (cur - prevd) : 0.0f;
        const float up = (d > 0.f) ? d : 0.0f;
        const float dn = (d < 0.f) ? -d : 0.0f;
        up_ema = fmaf(alpha, (up - up_ema), up_ema);
        dn_ema = fmaf(alpha, (dn - dn_ema), dn_ema);
        const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const float m = (x >= 0.0f) ? 1.0f : 0.0f;
        const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
        const float v = cur + x * scale;
        out_row[r] = (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
        prevd = cur;
    }
#endif
}

// Many-series × one-param, time-major layout
// prices_tm: [row * num_series + series]
extern "C" __global__ void reverse_rsi_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int rsi_length,
    float rsi_level,
    float* __restrict__ out_tm // time-major
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(rsi_length <= 0 || !(rsi_level > 0.0f && rsi_level < 100.0f) || fv < 0 || fv >= series_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = RRSI_NAN;
        return;
    }
    const int ema_len = (2 * rsi_length) - 1;
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= ema_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = RRSI_NAN;
        return;
    }

    const int warm_end = fv + ema_len; // exclusive
    const int warm_idx = warm_end - 1;

    // Prefill NaNs
    {
        float* o = out_tm + series;
        for (int r = 0; r < warm_idx; ++r, o += num_series) *o = RRSI_NAN;
    }

    // Constants
    // Switch to FP32-only math with Kahan warmup and FMA EMA
    const float nf = static_cast<float>(rsi_length);
    const float n_minus_1 = nf - 1.0f;
    const float inv = 100.0f - rsi_level;
    const float rs_target = rsi_level / inv;
    const float rs_coeff = n_minus_1 * rs_target;
    const float neg_scale = inv / rsi_level;
    const float alpha = 1.0f / nf; // == 2/(ema_len+1)

    // Warmup using Kahan summation
    float sum_up = 0.0f, c_up = 0.0f;
    float sum_dn = 0.0f, c_dn = 0.0f;
    float prev = 0.0f;
    for (int r = fv; r < warm_end; ++r) {
        const float cf = *(prices_tm + static_cast<size_t>(r) * num_series + series);
        const unsigned ok = (is_finite_bits(cf) & is_finite_bits(prev));
        const float d = ok ? (cf - prev) : 0.0f;
        const float up = d > 0.0f ? d : 0.0f;
        const float dn = d < 0.0f ? -d : 0.0f;
        kahan_add(up, sum_up, c_up);
        kahan_add(dn, sum_dn, c_dn);
        prev = cf;
    }
    float up_ema = sum_up / static_cast<float>(ema_len);
    float dn_ema = sum_dn / static_cast<float>(ema_len);

    // First output
    {
        const float base = *(prices_tm + static_cast<size_t>(warm_idx) * num_series + series);
        const float x0 = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const float m0 = (x0 >= 0.0f) ? 1.0f : 0.0f;
        const float scale0 = fmaf(m0, (1.0f - neg_scale), neg_scale);
        const float v0 = base + x0 * scale0;
        *(out_tm + static_cast<size_t>(warm_idx) * num_series + series) =
            (is_finite_bits(v0) || x0 >= 0.0f) ? v0 : 0.0f;
    }

    // Main loop
    float prevd = *(prices_tm + static_cast<size_t>(warm_idx) * num_series + series);
    for (int r = warm_end; r < series_len; ++r) {
        const float cf = *(prices_tm + static_cast<size_t>(r) * num_series + series);
        const unsigned ok = (is_finite_bits(cf) & is_finite_bits(prevd));
        const float d = ok ? (cf - prevd) : 0.0f;
        const float up = d > 0.0f ? d : 0.0f;
        const float dn = d < 0.0f ? -d : 0.0f;
        up_ema = fmaf(alpha, (up - up_ema), up_ema);
        dn_ema = fmaf(alpha, (dn - dn_ema), dn_ema);
        const float x = fmaf(rs_coeff, dn_ema, -n_minus_1 * up_ema);
        const float m = (x >= 0.0f) ? 1.0f : 0.0f;
        const float scale = fmaf(m, (1.0f - neg_scale), neg_scale);
        const float v = cf + x * scale;
        *(out_tm + static_cast<size_t>(r) * num_series + series) =
            (is_finite_bits(v) || x >= 0.0f) ? v : 0.0f;
        prevd = cf;
    }
}

