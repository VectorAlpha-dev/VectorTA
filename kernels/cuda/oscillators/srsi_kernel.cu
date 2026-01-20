



#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>


#if __CUDA_ARCH__ >= 350
  #define LDG(ptr) __ldg(ptr)
#else
  #define LDG(ptr) (*(ptr))
#endif



__device__ __forceinline__ float ftz_f32(float x) {
    return (fabsf(x) < FLT_MIN) ? 0.0f : x;
}


struct Deque {
    int*   idx;
    float* val;
    int    cap;
    int    head;
    int    tail;
};

__device__ __forceinline__ void dq_init(Deque* d, int* idx_buf, float* val_buf, int cap) {
    d->idx = idx_buf; d->val = val_buf; d->cap = cap; d->head = 0; d->tail = 0;
}
__device__ __forceinline__ bool dq_empty(const Deque* d) { return d->head == d->tail; }
__device__ __forceinline__ int dq_dec(const Deque* d, int x) { return (x == 0 ? d->cap - 1 : x - 1); }
__device__ __forceinline__ int dq_inc(const Deque* d, int x) { return (x + 1 == d->cap ? 0 : x + 1); }


__device__ __forceinline__ void dq_expire(Deque* d, int start_idx) {
    if (!dq_empty(d) && d->idx[d->head] < start_idx) { d->head = dq_inc(d, d->head); }
}


__device__ __forceinline__ void dq_push_max(Deque* d, int idx, float v) {
    int t = d->tail;
    if (!dq_empty(d)) {
        int pos = dq_dec(d, t);
        while (pos != d->head && d->val[pos] < v) { t = pos; pos = dq_dec(d, pos); }
        if (pos == d->head && d->val[pos] < v) { t = d->head; d->head = dq_inc(d, d->head); }
    }
    d->idx[t] = idx; d->val[t] = v; d->tail = dq_inc(d, t);
}


__device__ __forceinline__ void dq_push_min(Deque* d, int idx, float v) {
    int t = d->tail;
    if (!dq_empty(d)) {
        int pos = dq_dec(d, t);
        while (pos != d->head && d->val[pos] > v) { t = pos; pos = dq_dec(d, pos); }
        if (pos == d->head && d->val[pos] > v) { t = d->head; d->head = dq_inc(d, d->head); }
    }
    d->idx[t] = idx; d->val[t] = v; d->tail = dq_inc(d, t);
}


struct Kahan {
    float s, c;
    __device__ __forceinline__ void reset() { s = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s; }
};














extern "C" __global__
void srsi_fk_batch_f32(const float* __restrict__ rsi,
                       const int*   __restrict__ stoch_periods,
                       const int*   __restrict__ k_periods,
                       const int*   __restrict__ d_periods,
                       int series_len,
                       int first_valid,
                       int rsi_period,
                       int n_combos,
                       float* __restrict__ out_k,
                       float* __restrict__ out_d)
{
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) return;

    const int sp = stoch_periods[combo];

    const int row_off = combo * series_len;
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len ||
        rsi_period <= 0 || sp <= 0) {
        while (t < series_len) { out_d[row_off + t] = NAN; t += stride; }
        return;
    }

    const int rsi_warmup   = first_valid + rsi_period;
    const int stoch_warmup = rsi_warmup + sp - 1;
    if (rsi_warmup >= series_len || stoch_warmup >= series_len) {
        while (t < series_len) { out_d[row_off + t] = NAN; t += stride; }
        return;
    }

    while (t < series_len) {
        float fk = NAN;
        if (t >= stoch_warmup) {
            const float rv = ftz_f32(LDG(&rsi[t]));
            const int start = t + 1 - sp;
            float hi = -1e30f;
            float lo =  1e30f;
            for (int i = start; i <= t; ++i) {
                const float v = ftz_f32(LDG(&rsi[i]));
                hi = fmaxf(hi, v);
                lo = fminf(lo, v);
            }
            const float denom = hi - lo;
            fk = (denom >= FLT_MIN) ? ((rv - lo) * 100.0f) / denom : 50.0f;
        }
        out_d[row_off + t] = fk;
        t += stride;
    }
}

extern "C" __global__
void srsi_sma_k_batch_f32(const float* __restrict__ rsi,
                          const int*   __restrict__ stoch_periods,
                          const int*   __restrict__ k_periods,
                          const int*   __restrict__ d_periods,
                          int series_len,
                          int first_valid,
                          int rsi_period,
                          int n_combos,
                          float* __restrict__ out_k,
                          float* __restrict__ out_d)
{
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) return;

    const int sp = stoch_periods[combo];
    const int kp = k_periods[combo];

    const int row_off = combo * series_len;
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len ||
        rsi_period <= 0 || sp <= 0 || kp <= 0) {
        while (t < series_len) { out_k[row_off + t] = NAN; t += stride; }
        return;
    }

    const int rsi_warmup   = first_valid + rsi_period;
    const int stoch_warmup = rsi_warmup + sp - 1;
    const int k_warmup     = stoch_warmup + kp - 1;
    if (rsi_warmup >= series_len || stoch_warmup >= series_len || k_warmup >= series_len) {
        while (t < series_len) { out_k[row_off + t] = NAN; t += stride; }
        return;
    }

    while (t < series_len) {
        float slow_k = NAN;
        if (t >= k_warmup) {
            const int start = t + 1 - kp;
            float sum = 0.0f;
            for (int i = start; i <= t; ++i) {
                sum += out_d[row_off + i];
            }
            slow_k = sum * (1.0f / (float)kp);
        }
        out_k[row_off + t] = slow_k;
        t += stride;
    }
}

extern "C" __global__
void srsi_sma_d_batch_f32(const float* __restrict__ rsi,
                          const int*   __restrict__ stoch_periods,
                          const int*   __restrict__ k_periods,
                          const int*   __restrict__ d_periods,
                          int series_len,
                          int first_valid,
                          int rsi_period,
                          int n_combos,
                          float* __restrict__ out_k,
                          float* __restrict__ out_d)
{
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) return;

    const int sp = stoch_periods[combo];
    const int kp = k_periods[combo];
    const int dp = d_periods[combo];

    const int row_off = combo * series_len;
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int stride = (int)gridDim.x * (int)blockDim.x;

    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len ||
        rsi_period <= 0 || sp <= 0 || kp <= 0 || dp <= 0) {
        while (t < series_len) { out_d[row_off + t] = NAN; t += stride; }
        return;
    }

    const int rsi_warmup   = first_valid + rsi_period;
    const int stoch_warmup = rsi_warmup + sp - 1;
    const int k_warmup     = stoch_warmup + kp - 1;
    const int d_warmup     = k_warmup + dp - 1;
    if (rsi_warmup >= series_len || stoch_warmup >= series_len ||
        k_warmup >= series_len || d_warmup >= series_len) {
        while (t < series_len) { out_d[row_off + t] = NAN; t += stride; }
        return;
    }

    while (t < series_len) {
        float slow_d = NAN;
        if (t >= d_warmup) {
            const int start = t + 1 - dp;
            float sum = 0.0f;
            for (int i = start; i <= t; ++i) {
                sum += out_k[row_off + i];
            }
            slow_d = sum * (1.0f / (float)dp);
        }
        out_d[row_off + t] = slow_d;
        t += stride;
    }
}


extern "C" __global__
void srsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    int cols,
                                    int rows,
                                    int rsi_period,
                                    int stoch_period,
                                    int k_period,
                                    int d_period,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ k_out_tm,
                                    float* __restrict__ d_out_tm) {
    const int s = blockIdx.x;
    if (s >= cols) return;
    if (rsi_period <= 0 || stoch_period <= 0 || k_period <= 0 || d_period <= 0) return;

    const int stride = cols;
    int first = first_valids[s]; if (first < 0) first = 0; if (first >= rows) return;
    const int rsi_warmup   = first + rsi_period;
    const int stoch_warmup = rsi_warmup + stoch_period - 1;
    const int k_warmup     = stoch_warmup + k_period - 1;
    const int d_warmup     = k_warmup + d_period - 1;


    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        if (t < k_warmup) k_out_tm[t * stride + s] = NAN;
        if (t < d_warmup) d_out_tm[t * stride + s] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;


    float avg_gain = 0.0f, avg_loss = 0.0f;
    float prev = LDG(&prices_tm[first * stride + s]);
    for (int i = first + 1; i <= first + rsi_period && i < rows; ++i) {
        float cur = LDG(&prices_tm[i * stride + s]);
        const float ch = cur - prev; prev = cur;
        if (ch > 0.0f) avg_gain += ch; else avg_loss += -ch;
    }
    avg_gain /= (float)rsi_period; avg_loss /= (float)rsi_period;
    const float alpha = 1.0f / (float)rsi_period;


    extern __shared__ unsigned char smem2[];
    int*   max_idx = (int*)smem2;
    float* rsi_ring = (float*)(max_idx + stoch_period);
    int*   min_idx = (int*)(rsi_ring + stoch_period);
    float* min_val = (float*)(min_idx + stoch_period);
    float* ring_k  = (float*)(min_val + stoch_period);
    float* ring_d  = (float*)(ring_k + k_period);




    int rpos = 0; int rcnt = 0;
    float rsi = 50.0f;
    if (rsi_warmup < rows) {
        rsi = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));
    }
    rsi = ftz_f32(rsi);
    if (stoch_period > 1) {
        rsi_ring[rpos] = rsi;
        rpos = (rpos + 1 == stoch_period ? 0 : rpos + 1);
        if (rcnt < stoch_period) ++rcnt;

        for (int t = rsi_warmup + 1; t < rsi_warmup + stoch_period - 1 && t < rows; ++t) {
            float x = LDG(&prices_tm[t * stride + s]);
            const float prevp = LDG(&prices_tm[(t - 1) * stride + s]);
            const float ch = x - prevp;
            const float gain = (ch > 0.0f ? ch : 0.0f);
            const float loss = (ch < 0.0f ? -ch : 0.0f);
            avg_gain = fmaf(gain - avg_gain, alpha, avg_gain);
            avg_loss = fmaf(loss - avg_loss, alpha, avg_loss);
            rsi = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));
            rsi = ftz_f32(rsi);
            rsi_ring[rpos] = rsi;
            rpos = (rpos + 1 == stoch_period ? 0 : rpos + 1);
            if (rcnt < stoch_period) ++rcnt;
        }
    }

    float sum_k = 0.0f, sum_d = 0.0f; int head_k = 0, head_d = 0, cnt_k = 0, cnt_d = 0;
    const float inv_k = 1.0f / (float)k_period;
    const float inv_d = 1.0f / (float)d_period;

    for (int t = stoch_warmup; t < rows; ++t) {
        const float x = LDG(&prices_tm[t * stride + s]);
        const float prevp = LDG(&prices_tm[(t - 1) * stride + s]);
        const float ch = x - prevp;
        const float gain = (ch > 0.0f ? ch : 0.0f);
        const float loss = (ch < 0.0f ? -ch : 0.0f);
        avg_gain = fmaf(gain - avg_gain, alpha, avg_gain);
        avg_loss = fmaf(loss - avg_loss, alpha, avg_loss);
        rsi = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));
        rsi = ftz_f32(rsi);

        rsi_ring[rpos] = rsi; rpos = (rpos + 1 == stoch_period ? 0 : rpos + 1); if (rcnt < stoch_period) ++rcnt;
        float hi = rsi, lo = rsi;
        int cnt = rcnt < stoch_period ? rcnt : stoch_period;
        for (int j = 0; j < cnt - 1; ++j) {
            float v = rsi_ring[(rpos + j) % stoch_period];
            hi = fmaxf(hi, v);
            lo = fminf(lo, v);
        }

        const float denom = hi - lo;

        float fk = (isfinite(hi) && isfinite(lo) && denom >= FLT_MIN) ? ((rsi - lo) * 100.0f) / denom : 50.0f;

        if (cnt_k < k_period) { sum_k += fk; ring_k[head_k] = fk; ++cnt_k; if (++head_k == k_period) head_k = 0; }
        else                   { sum_k += fk - ring_k[head_k]; ring_k[head_k] = fk; if (++head_k == k_period) head_k = 0; }
        if (t >= k_warmup) {
            const float slow_k = sum_k * inv_k; k_out_tm[t * stride + s] = slow_k;
            if (cnt_d < d_period) { sum_d += slow_k; ring_d[head_d] = slow_k; ++cnt_d; if (++head_d == d_period) head_d = 0; }
            else                   { sum_d += slow_k - ring_d[head_d]; ring_d[head_d] = slow_k; if (++head_d == d_period) head_d = 0; }
            if (t >= d_warmup) d_out_tm[t * stride + s] = sum_d * inv_d;
        }
    }
}

