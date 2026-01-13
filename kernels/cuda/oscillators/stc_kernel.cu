











#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


#ifndef STC_BLOCK_X
#define STC_BLOCK_X 1
#endif

#ifndef STC_SMALL_K
#define STC_SMALL_K 16
#endif



#define STC_RANGE_EPS 2.2204460492503131e-16f


#define STC_BATCH_SMEM_BYTES(max_k) ((size_t)(max_k) * (2*sizeof(float) + 4*sizeof(int)))


static __device__ __forceinline__ float ema_update_f32(float prev, float a, float x) {

    return __fmaf_rn(a, (x - prev), prev);
}


static __device__ __forceinline__ float div_rn_f32(float num, float den) {
    return __fdiv_rn(num, den);
}


struct KahanF32 {
    float s;
    float c;
    __device__ __forceinline__ void reset() { s = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {
        float t = s + x;
        if (fabsf(s) >= fabsf(x)) c += (s - t) + x;
        else                      c += (x - t) + s;
        s = t;
    }
    __device__ __forceinline__ float result() const { return s + c; }
};



struct IndexDeque {
    int*  buf;
    int   head;
    int   len;
    int   cap;
    float* ring;
    bool  is_min;

    __device__ __forceinline__ void init(int* storage, int capacity, float* ring_ptr, bool as_min) {
        buf = storage; cap = capacity; ring = ring_ptr; is_min = as_min; head = 0; len = 0;
    }
    __device__ __forceinline__ void reset() { head = 0; len = 0; }
    __device__ __forceinline__ void push(int idx, float v) {

        while (len > 0) {
            int last = head + len - 1; if (last >= cap) last -= cap;
            float backv = ring[ buf[last] % cap ];
            if (is_min ? (backv >= v) : (backv <= v)) { len--; }
            else break;
        }

        int tail_pos = head + len; if (tail_pos >= cap) tail_pos -= cap;
        buf[tail_pos] = idx;
        if (len < cap) len++;
    }
    __device__ __forceinline__ void pop_expired(int min_idx_allowed) {
        while (len > 0 && buf[head] < min_idx_allowed) {
            head++; if (head == cap) head = 0; len--;
        }
    }
    __device__ __forceinline__ bool empty() const { return len == 0; }
    __device__ __forceinline__ float front_val() const { return ring[ buf[head] % cap ]; }
};




static __device__ __forceinline__ void stc_compute_series_f32(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int fast,
    int slow,
    int k,
    int d,
    int max_k,
    float* __restrict__ out)
{
    if (len <= 0 || first_valid >= len) return;


    extern __shared__ unsigned char shmem[];
    float* macd_ring = reinterpret_cast<float*>(shmem);
    float* d_ring    = macd_ring + max_k;
    int* macd_min_idx = reinterpret_cast<int*>(d_ring + max_k);
    int* macd_max_idx = macd_min_idx + max_k;
    int* d_min_idx    = macd_max_idx + max_k;
    int* d_max_idx    = d_min_idx + max_k;


    const int warm = first_valid + max(max(fast, slow), max(k, d)) - 1;

    for (int i = 0; i < min(warm, len); ++i) out[i] = NAN;
    if (warm >= len) return;


    const float fast_a = div_rn_f32(2.0f, (float)(fast + 1));
    const float slow_a = div_rn_f32(2.0f, (float)(slow + 1));
    const float d_a    = div_rn_f32(2.0f, (float)(d + 1));


    KahanF32 fast_acc; fast_acc.reset();
    KahanF32 slow_acc; slow_acc.reset();
    bool fast_seed_nan = false, slow_seed_nan = false;
    const int f_end = min(fast, len - first_valid);
    const int s_end = min(slow, len - first_valid);
    for (int i = 0; i < f_end; ++i) { float v = prices[first_valid + i]; if (!isfinite(v)) { fast_seed_nan = true; break; } fast_acc.add(v); }
    for (int i = 0; i < s_end; ++i) { float v = prices[first_valid + i]; if (!isfinite(v)) { slow_seed_nan = true; break; } slow_acc.add(v); }
    float fast_ema = (f_end == fast && !fast_seed_nan) ? div_rn_f32(fast_acc.result(), (float)fast) : NAN;
    float slow_ema = (s_end == slow && !slow_seed_nan) ? div_rn_f32(slow_acc.result(), (float)slow) : NAN;


    IndexDeque macd_min, macd_max, d_min, d_max;
    macd_min.init(macd_min_idx, k, macd_ring, true);
    macd_max.init(macd_max_idx, k, macd_ring, false);
    d_min.init(d_min_idx, k, d_ring, true);
    d_max.init(d_max_idx, k, d_ring, false);


    int macd_run = 0, d_run = 0;


    KahanF32 d_seed_acc; d_seed_acc.reset(); int d_seed_cnt = 0; float d_ema = NAN;
    KahanF32 final_seed_acc; final_seed_acc.reset(); int final_seed_cnt = 0; float final_ema = NAN;

    const int fast_thr = fast > 0 ? (fast - 1) : 0;
    const int slow_thr = slow > 0 ? (slow - 1) : 0;


    for (int i = 0; i < len; ++i) {
        const float x = prices[i];


        if (i >= first_valid) {
            const int rel = i - first_valid;
            if (rel >= fast_thr) {
                if (rel != fast_thr) { if (isfinite(x) && isfinite(fast_ema)) fast_ema = ema_update_f32(fast_ema, fast_a, x); else fast_ema = NAN; }
            }
            if (rel >= slow_thr) {
                if (rel != slow_thr) { if (isfinite(x) && isfinite(slow_ema)) slow_ema = ema_update_f32(slow_ema, slow_a, x); else slow_ema = NAN; }
            }
        }


        float macd; unsigned char macd_is_valid;
        if (i >= first_valid + slow_thr && isfinite(fast_ema) && isfinite(slow_ema)) { macd = fast_ema - slow_ema; macd_is_valid = 1u; }
        else { macd = NAN; macd_is_valid = 0u; }


        float stok = NAN;
        if (macd_is_valid) {

            macd_ring[i % k] = macd;
            macd_run += 1;
            if (k <= STC_SMALL_K) {

                float mn = macd_ring[(i - (macd_run-1)) % k];
                float mx = mn;
                int start = i - min(macd_run, k) + 1;
                for (int j = 0; j < min(macd_run, k); ++j) { float v = macd_ring[(start + j) % k]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
                if (macd_run >= k) {
                    const float range = mx - mn;
                    stok = (fabsf(range) > STC_RANGE_EPS) ? ((macd - mn) * div_rn_f32(100.0f, range)) : 50.0f;
                } else { stok = 50.0f; }
            } else {

                macd_min.push(i, macd); macd_max.push(i, macd);
                const int left = i - k + 1;
                macd_min.pop_expired(left); macd_max.pop_expired(left);
                if (macd_run >= k && !macd_min.empty() && !macd_max.empty()) {
                    const float mn = macd_min.front_val();
                    const float mx = macd_max.front_val();
                    const float range = mx - mn;
                    stok = (fabsf(range) > STC_RANGE_EPS) ? ((macd - mn) * div_rn_f32(100.0f, range)) : 50.0f;
                } else { stok = 50.0f; }
            }
        } else {
            macd_run = 0; macd_min.reset(); macd_max.reset(); stok = NAN;
        }


        float d_val = NAN;
        if (isfinite(stok)) {
            if (d_seed_cnt < d) {
                d_seed_acc.add(stok);
                d_seed_cnt += 1;
                const float sum = d_seed_acc.result();
                if (d_seed_cnt == d) { d_ema = div_rn_f32(sum, (float)d); d_val = d_ema; }
                else { d_val = div_rn_f32(sum, (float)d_seed_cnt); }
            } else {
                d_ema = ema_update_f32(d_ema, d_a, stok);
                d_val = d_ema;
            }
        } else {
            if (d_seed_cnt == 0) d_val = NAN;
            else if (d_seed_cnt < d) d_val = div_rn_f32(d_seed_acc.result(), (float)d_seed_cnt);
            else d_val = d_ema;
        }


        float kd = NAN;
        if (isfinite(d_val)) {
            d_ring[i % k] = d_val; d_run += 1;
            if (k <= STC_SMALL_K) {
                float mn = d_ring[(i - (d_run-1)) % k]; float mx = mn;
                int start = i - min(d_run, k) + 1;
                for (int j = 0; j < min(d_run, k); ++j) { float v = d_ring[(start + j) % k]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
                if (d_run >= k) { const float range = mx - mn; kd = (fabsf(range) > STC_RANGE_EPS) ? ((d_val - mn) * div_rn_f32(100.0f, range)) : 50.0f; } else { kd = 50.0f; }
            } else {
                d_min.push(i, d_val); d_max.push(i, d_val);
                const int left = i - k + 1;
                d_min.pop_expired(left); d_max.pop_expired(left);
                if (d_run >= k && !d_min.empty() && !d_max.empty()) {
                    const float mn = d_min.front_val(); const float mx = d_max.front_val();
                    const float range = mx - mn; kd = (fabsf(range) > STC_RANGE_EPS) ? ((d_val - mn) * div_rn_f32(100.0f, range)) : 50.0f;
                } else { kd = 50.0f; }
            }
        } else { d_min.reset(); d_max.reset(); }


        float out_i = NAN;
        if (isfinite(kd)) {
            if (final_seed_cnt < d) {
                final_seed_acc.add(kd);
                final_seed_cnt += 1;
                const float sum = final_seed_acc.result();
                if (final_seed_cnt == d) { final_ema = div_rn_f32(sum, (float)d); out_i = final_ema; }
                else { out_i = div_rn_f32(sum, (float)final_seed_cnt); }
            } else {
                final_ema = ema_update_f32(final_ema, d_a, kd);
                out_i = final_ema;
            }
        } else {
            if (final_seed_cnt == 0) out_i = NAN;
            else if (final_seed_cnt < d) out_i = div_rn_f32(final_seed_acc.result(), (float)final_seed_cnt);
            else out_i = final_ema;
        }

        if (i >= warm) out[i] = out_i;
    }
}


extern "C" __global__ __launch_bounds__(1)
void stc_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ fasts,
                   const int* __restrict__ slows,
                   const int* __restrict__ ks,
                   const int* __restrict__ ds,
                   int series_len,
                   int first_valid,
                   int n_rows,
                   int max_k,
                   float* __restrict__ out)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int fast = fasts[row];
    const int slow = slows[row];
    const int kk   = ks[row];
    const int dd   = ds[row];
    if (fast <= 0 || slow <= 0 || kk <= 0 || dd <= 0) return;

    const int base = row * series_len;


    if (threadIdx.x != 0) return;
    stc_compute_series_f32(prices, series_len, first_valid, fast, slow, kk, dd, max_k, out + base);
}


extern "C" __global__
void stc_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int fast,
                                   int slow,
                                   int k,
                                   int d,
                                   float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    const int first = first_valids[s];


    int warm = first + max(max(fast, slow), max(k, d)) - 1;
    if (warm > rows) warm = rows;
    for (int t = 0; t < warm; ++t) out_tm[t * cols + s] = NAN;
    if (warm >= rows) return;


    const float fast_a = div_rn_f32(2.0f, (float)(fast + 1));
    const float slow_a = div_rn_f32(2.0f, (float)(slow + 1));
    const float d_a    = div_rn_f32(2.0f, (float)(d + 1));


    KahanF32 fast_acc; fast_acc.reset();
    KahanF32 slow_acc; slow_acc.reset();
    const int f_end = min(fast, rows - first);
    const int s_end = min(slow, rows - first);
    for (int i = 0; i < f_end; ++i) fast_acc.add(prices_tm[(first + i) * cols + s]);
    for (int i = 0; i < s_end; ++i) slow_acc.add(prices_tm[(first + i) * cols + s]);
    float fast_ema = (f_end == fast) ? div_rn_f32(fast_acc.result(), (float)fast) : NAN;
    float slow_ema = (s_end == slow) ? div_rn_f32(slow_acc.result(), (float)slow) : NAN;


    const int KMAX = 2048;
    const int kk = (k <= KMAX) ? k : KMAX;
    float macd_ring[KMAX];
    float d_ring[KMAX];
    for (int i = 0; i < kk; ++i) { macd_ring[i] = NAN; d_ring[i] = NAN; }


    KahanF32 d_seed_acc; d_seed_acc.reset(); int d_seed_cnt = 0; float d_ema = NAN;
    KahanF32 final_seed_acc; final_seed_acc.reset(); int final_seed_cnt = 0; float final_ema = NAN;
    const int fast_thr = fast > 0 ? (fast - 1) : 0;
    const int slow_thr = slow > 0 ? (slow - 1) : 0;
    int macd_run = 0, d_run = 0;

    for (int i = 0; i < rows; ++i) {
        const float x = prices_tm[i * cols + s];


        if (i >= first) {
            const int rel = i - first;
            if (rel >= fast_thr) {
                if (rel != fast_thr) { if (isfinite(x) && isfinite(fast_ema)) fast_ema = ema_update_f32(fast_ema, fast_a, x); else fast_ema = NAN; }
            }
            if (rel >= slow_thr) {
                if (rel != slow_thr) { if (isfinite(x) && isfinite(slow_ema)) slow_ema = ema_update_f32(slow_ema, slow_a, x); else slow_ema = NAN; }
            }
        }

        float macd; unsigned char macd_is_valid;
        if (i >= first + slow_thr && isfinite(fast_ema) && isfinite(slow_ema)) { macd = fast_ema - slow_ema; macd_is_valid = 1u; }
        else { macd = NAN; macd_is_valid = 0u; }

        float stok = NAN;
        if (macd_is_valid) {
            macd_ring[i % kk] = macd; macd_run += 1;
            if (macd_run >= k) {
                float mn = macd_ring[(i - (k-1)) % kk], mx = mn;
                for (int j = 1; j < k; ++j) { float v = macd_ring[(i - (k-1) + j) % kk]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
                const float range = mx - mn; stok = (fabsf(range) > STC_RANGE_EPS) ? ((macd - mn) * div_rn_f32(100.0f, range)) : 50.0f;
            } else { stok = 50.0f; }
        } else { macd_run = 0; }

        float d_val = NAN;
        if (isfinite(stok)) {
            if (d_seed_cnt < d) {
                d_seed_acc.add(stok);
                d_seed_cnt += 1;
                const float sum = d_seed_acc.result();
                if (d_seed_cnt == d) { d_ema = div_rn_f32(sum, (float)d); d_val = d_ema; }
                else { d_val = div_rn_f32(sum, (float)d_seed_cnt); }
            } else {
                d_ema = ema_update_f32(d_ema, d_a, stok);
                d_val = d_ema;
            }
        } else {
            if (d_seed_cnt == 0) d_val = NAN;
            else if (d_seed_cnt < d) d_val = div_rn_f32(d_seed_acc.result(), (float)d_seed_cnt);
            else d_val = d_ema;
        }

        float kd = NAN;
        if (isfinite(d_val)) {
            d_ring[i % kk] = d_val; d_run += 1;
            if (d_run >= k) {
                float mn = d_ring[(i - (k-1)) % kk], mx = mn;
                for (int j = 1; j < k; ++j) { float v = d_ring[(i - (k-1) + j) % kk]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
                const float range = mx - mn; kd = (fabsf(range) > STC_RANGE_EPS) ? ((d_val - mn) * div_rn_f32(100.0f, range)) : 50.0f;
            } else { kd = 50.0f; }
        } else { d_run = 0; }

        float out_i = NAN;
        if (isfinite(kd)) {
            if (final_seed_cnt < d) {
                final_seed_acc.add(kd);
                final_seed_cnt += 1;
                const float sum = final_seed_acc.result();
                if (final_seed_cnt == d) { final_ema = div_rn_f32(sum, (float)d); out_i = final_ema; }
                else { out_i = div_rn_f32(sum, (float)final_seed_cnt); }
            } else {
                final_ema = ema_update_f32(final_ema, d_a, kd);
                out_i = final_ema;
            }
        } else {
            if (final_seed_cnt == 0) out_i = NAN;
            else if (final_seed_cnt < d) out_i = div_rn_f32(final_seed_acc.result(), (float)final_seed_cnt);
            else out_i = final_ema;
        }

        if (i >= warm) out_tm[i * cols + s] = out_i;
    }
}

