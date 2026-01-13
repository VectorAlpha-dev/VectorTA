














#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef TTM_QNAN_F
#define TTM_QNAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ bool is_finite_f(float x) { return isfinite(x); }



struct NeumaierSumF {
    float s, c;
    __device__ __forceinline__ void reset() { s = 0.f; c = 0.f; }
    __device__ __forceinline__ void add(float x) {
        float t = s + x;
        if (fabsf(s) >= fabsf(x)) c += (s - t) + x;
        else                      c += (x - t) + s;
        s = t;
    }
    __device__ __forceinline__ float val() const { return s + c; }
};


struct DequeI {
    int *buf; int cap; int head; int tail;
    __device__ __forceinline__ DequeI(int* p, int c): buf(p), cap(c), head(0), tail(0) {}
    __device__ __forceinline__ bool empty() const { return head == tail; }
    __device__ __forceinline__ int  size()  const { int d = tail - head; return (d >= 0) ? d : d + cap; }
    __device__ __forceinline__ int  front() const { int i = head; return buf[i]; }
    __device__ __forceinline__ int  back()  const { int i = tail - 1; if (i < 0) i += cap; return buf[i]; }
    __device__ __forceinline__ void pop_front() { head = (head + 1 == cap) ? 0 : head + 1; }
    __device__ __forceinline__ void pop_back()  { tail = (tail == 0) ? cap - 1 : tail - 1; }
    __device__ __forceinline__ void push_back(int v) { buf[tail] = v; tail = (tail + 1 == cap) ? 0 : tail + 1; }
};


static __device__ __forceinline__ float true_range_idx_f32(
    int i, const float* __restrict__ high, const float* __restrict__ low, const float* __restrict__ close
){
    const float h = high[i];
    const float l = low[i];
    if (i == 0) return fabsf(h - l);
    const float pc  = close[i - 1];
    const float tr1 = fabsf(h - l);
    const float tr2 = fabsf(h - pc);
    const float tr3 = fabsf(l - pc);
    return fmaxf(fmaxf(tr1, tr2), tr3);
}


extern "C" __global__ void ttm_squeeze_batch_f32(

    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,

    const int*   __restrict__ length_arr,
    const float* __restrict__ bb_mult_arr,
    const float* __restrict__ kc_high_arr,
    const float* __restrict__ kc_mid_arr,
    const float* __restrict__ kc_low_arr,

    int series_len,
    int n_combos,
    int first_valid,

    float* __restrict__ out_momentum,
    float* __restrict__ out_squeeze
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int base = combo * series_len;

    const int   L = length_arr[combo];
    auto fill_all_nan = [&]() {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_momentum[base + i] = TTM_QNAN_F;
            out_squeeze[base + i]  = TTM_QNAN_F;
        }
    };

    if (UNLIKELY(L <= 0 || first_valid < 0 || first_valid >= series_len)) {
        fill_all_nan();
        return;
    }
    const int warm  = first_valid + L - 1;
    if (UNLIKELY(warm >= series_len)) {
        fill_all_nan();
        return;
    }


    __shared__ int seed_ok_i;
    if (threadIdx.x == 0) {
        bool seed_ok = true;
        for (int j = 0; j < L && j < series_len; ++j) {
            if (!is_finite_f(close[j]) || !is_finite_f(high[j]) || !is_finite_f(low[j])) { seed_ok = false; break; }
        }
        seed_ok_i = seed_ok ? 1 : 0;
    }
    __syncthreads();
    if (UNLIKELY(seed_ok_i == 0)) {
        fill_all_nan();
        return;
    }


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_momentum[base + i] = TTM_QNAN_F;
        out_squeeze[base + i]  = TTM_QNAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;


    const float n    = (float)L;
    const float invL = 1.0f / n;
    const float sx   = 0.5f * n * (n - 1.0f);
    const float sx2  = (n - 1.0f) * n * (2.0f * n - 1.0f) / 6.0f;
    const float den  = n * sx2 - sx * sx;
    const float inv_den = (den != 0.0f) ? (1.0f / den) : 0.0f;

    const float bb_sq = bb_mult_arr[combo] * bb_mult_arr[combo];
    const float kh_sq = kc_high_arr[combo] * kc_high_arr[combo];
    const float km_sq = kc_mid_arr[combo]  * kc_mid_arr[combo];
    const float kl_sq = kc_low_arr[combo]  * kc_low_arr[combo];


    extern __shared__ unsigned char __ttm_smem[];
    int   *dq_max_buf = (int*)  (__ttm_smem);
    int   *dq_min_buf = dq_max_buf + L;
    float *ring_c     = (float*)(dq_min_buf + L);
    float *ring_tr    = ring_c     + L;
    unsigned char *v_in = (unsigned char*)(ring_tr + L);
    unsigned char *v_tr = v_in + L;

    DequeI dq_max(dq_max_buf, L);
    DequeI dq_min(dq_min_buf, L);


    const int start0 = warm - L + 1;
    NeumaierSumF sumc;  sumc.reset();
    NeumaierSumF sumc2; sumc2.reset();
    NeumaierSumF sumtr; sumtr.reset();
    float sumxc = 0.0f;

    int bad_in_window = 0;
    int bad_tr_window = 0;

    for (int k = 0; k < L; ++k) {
        const int idx = start0 + k;
        const float h = high[idx];
        const float l = low[idx];
        const float c = close[idx];

        const unsigned char fin = (unsigned char)(is_finite_f(h) & is_finite_f(l) & is_finite_f(c));
        v_in[k] = fin;
        if (!fin) ++bad_in_window;

        ring_c[k] = c;
        if (fin) { sumc.add(c); sumc2.add(c * c); sumxc = fmaf((float)k, c, sumxc); }


        const float tr = true_range_idx_f32(idx, high, low, close);
        const unsigned char ftr = (unsigned char)is_finite_f(tr);
        v_tr[k] = ftr;
        if (!ftr) ++bad_tr_window;
        ring_tr[k] = tr;
        if (ftr) sumtr.add(tr);


        while (!dq_max.empty() && high[dq_max.back()] <= h) dq_max.pop_back();
        dq_max.push_back(idx);
        while (!dq_min.empty() && low[dq_min.back()] >= l) dq_min.pop_back();
        dq_min.push_back(idx);
    }


    int ring_head = 0;


    if (bad_in_window == 0 && bad_tr_window == 0) {
        const float mean = sumc.val() * invL;
        const float var  = fmaxf(sumc2.val() * invL - mean * mean, 0.0f);
        const float dkc  = sumtr.val() * invL;
        const float dkc2 = dkc * dkc;


        const float bbv = bb_sq * var;
        const float t_low  = kl_sq * dkc2;
        const float t_mid  = km_sq * dkc2;
        const float t_high = kh_sq * dkc2;
        out_squeeze[base + warm] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));


        const float highest = high[dq_max.front()];
        const float lowest  = low [dq_min.front()];
        const float midpoint = 0.5f * (highest + lowest);
        const float avg = 0.5f * (midpoint + mean);
        const float S0 = sumc.val() - n * avg;
        const float S1 = sumxc - avg * sx;
        const float slope = (den != 0.0f) ? ( (n * S1 - sx * S0) * inv_den ) : 0.0f;
        const float intercept = (S0 - slope * sx) * (1.0f / n);
        const float yhat_last = intercept + slope * (n - 1.0f);
        out_momentum[base + warm] = yhat_last;
    } else {
        out_squeeze[base + warm]  = TTM_QNAN_F;
        out_momentum[base + warm] = TTM_QNAN_F;
    }


    for (int i = warm + 1; i < series_len; ++i) {
        const int idx_new = i;
        const int idx_old = i - L;
        const int slot    = ring_head;


        while (!dq_max.empty() && dq_max.front() <= idx_old) dq_max.pop_front();
        while (!dq_min.empty() && dq_min.front() <= idx_old) dq_min.pop_front();


        const float h_new = high[idx_new];
        const float l_new = low [idx_new];
        const float c_new = close[idx_new];
        const unsigned char fin_new = (unsigned char)(is_finite_f(h_new) & is_finite_f(l_new) & is_finite_f(c_new));

        const float tr_new = true_range_idx_f32(idx_new, high, low, close);
        const unsigned char ftr_new = (unsigned char)is_finite_f(tr_new);


        const float c_old = ring_c[slot];
        const float tr_old = ring_tr[slot];
        const unsigned char fin_old = v_in[slot];
        const unsigned char ftr_old = v_tr[slot];


        bad_in_window += (int)!fin_new - (int)!fin_old;
        bad_tr_window += (int)!ftr_new - (int)!ftr_old;


        const float sumc_before = sumc.val();
        if (fin_old) { sumc.add(-c_old); sumc2.add(-(c_old * c_old)); }
        if (fin_new) { sumc.add( c_new); sumc2.add(  c_new * c_new ); }


        float adj_old = (fin_old ? c_old : 0.0f);
        float adj_new = (fin_new ? c_new : 0.0f);
        sumxc = fmaf(-1.0f, (sumc_before - adj_old), sumxc);
        sumxc = fmaf((float)(L - 1), adj_new,        sumxc);

        if (ftr_old) sumtr.add(-tr_old);
        if (ftr_new) sumtr.add( tr_new);


        ring_c[slot] = c_new; v_in[slot] = fin_new;
        ring_tr[slot] = tr_new; v_tr[slot] = ftr_new;
        ring_head = (ring_head + 1 == L) ? 0 : ring_head + 1;


        while (!dq_max.empty() && high[dq_max.back()] <= h_new) dq_max.pop_back();
        dq_max.push_back(idx_new);
        while (!dq_min.empty() && low[dq_min.back()] >= l_new) dq_min.pop_back();
        dq_min.push_back(idx_new);

        if (bad_in_window == 0 && bad_tr_window == 0) {
            const float mean = sumc.val() * invL;
            const float var  = fmaxf(sumc2.val() * invL - mean * mean, 0.0f);
            const float dkc  = sumtr.val() * invL;
            const float dkc2 = dkc * dkc;

            const float bbv = bb_sq * var;
            const float t_low  = kl_sq * dkc2;
            const float t_mid  = km_sq * dkc2;
            const float t_high = kh_sq * dkc2;
            out_squeeze[base + i] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));

            const float highest = high[dq_max.front()];
            const float lowest  = low [dq_min.front()];
            const float midpoint = 0.5f * (highest + lowest);
            const float avg = 0.5f * (midpoint + mean);
            const float S0 = sumc.val() - n * avg;
            const float S1 = sumxc - avg * sx;
            const float slope = (den != 0.0f) ? ( (n * S1 - sx * S0) * inv_den ) : 0.0f;
            const float intercept = (S0 - slope * sx) * (1.0f / n);
            const float yhat_last = intercept + slope * (n - 1.0f);
            out_momentum[base + i] = yhat_last;
        } else {
            out_squeeze[base + i]  = TTM_QNAN_F;
            out_momentum[base + i] = TTM_QNAN_F;
        }
    }
}


extern "C" __global__ void ttm_squeeze_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int length,
    float bb_mult,
    float kc_high,
    float kc_mid,
    float kc_low,
    float* __restrict__ out_momentum_tm,
    float* __restrict__ out_squeeze_tm
) {
    const int s   = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;
    if (tid != 0) return;


    float* mo = out_momentum_tm + s;
    float* sq = out_squeeze_tm + s;
    auto fill_all_nan = [&]() {
        for (int t = 0; t < series_len; ++t) {
            mo[t * num_series] = TTM_QNAN_F;
            sq[t * num_series] = TTM_QNAN_F;
        }
    };

    const int L = length;
    const int fv = first_valids[s];
    if (UNLIKELY(L <= 0 || fv < 0 || fv >= series_len)) {
        fill_all_nan();
        return;
    }
    const int warm = fv + L - 1;
    if (UNLIKELY(warm >= series_len)) {
        fill_all_nan();
        return;
    }


    for (int t = 0; t < warm; ++t) {
        mo[t * num_series] = TTM_QNAN_F;
        sq[t * num_series] = TTM_QNAN_F;
    }


    auto H = [&](int t){ return high_tm[(size_t)t * num_series + s]; };
    auto Lw= [&](int t){ return  low_tm[(size_t)t * num_series + s]; };
    auto C = [&](int t){ return close_tm[(size_t)t * num_series + s]; };

    auto TR = [&](int t){
        if (t == 0) return fabsf(H(t) - Lw(t));
        const float pc = C(t - 1);
        const float tr1 = fabsf(H(t) - Lw(t));
        const float tr2 = fabsf(H(t) - pc);
        const float tr3 = fabsf(Lw(t) - pc);
        return fmaxf(fmaxf(tr1, tr2), tr3);
    };


    const float n    = (float)L;
    const float invL = 1.0f / n;
    const float sx   = 0.5f * n * (n - 1.0f);
    const float sx2  = (n - 1.0f) * n * (2.0f * n - 1.0f) / 6.0f;
    const float den  = n * sx2 - sx * sx;
    const float inv_den = (den != 0.0f) ? (1.0f / den) : 0.0f;

    const float bb_sq = bb_mult * bb_mult;
    const float kh_sq = kc_high * kc_high;
    const float km_sq = kc_mid  * kc_mid;
    const float kl_sq = kc_low  * kc_low;


    extern __shared__ unsigned char __ttm_smem[];
    int   *dq_max_buf = (int*)  (__ttm_smem);
    int   *dq_min_buf = dq_max_buf + L;
    float *ring_c     = (float*)(dq_min_buf + L);
    float *ring_tr    = ring_c     + L;
    unsigned char *v_in = (unsigned char*)(ring_tr + L);
    unsigned char *v_tr = v_in + L;

    DequeI dq_max(dq_max_buf, L);
    DequeI dq_min(dq_min_buf, L);




    bool seed_ok = true;

    for (int j = fv; j < fv + L && j < series_len; ++j) {
        float ch = H(j), cl = Lw(j), cc = C(j);
        if (!is_finite_f(ch) || !is_finite_f(cl) || !is_finite_f(cc)) { seed_ok = false; break; }
    }
    if (UNLIKELY(!seed_ok)) {
        fill_all_nan();
        return;
    }


    const int start0 = warm - L + 1;
    NeumaierSumF sumc;  sumc.reset();
    NeumaierSumF sumc2; sumc2.reset();
    NeumaierSumF sumtr; sumtr.reset();
    float sumxc = 0.0f;

    int bad_in_window = 0, bad_tr_window = 0;
    for (int k = 0; k < L; ++k) {
        const int idx = start0 + k;
        const float h = H(idx);
        const float l = Lw(idx);
        const float c = C(idx);
        const unsigned char fin = (unsigned char)(is_finite_f(h) & is_finite_f(l) & is_finite_f(c));
        v_in[k] = fin; ring_c[k] = c;
        if (!fin) ++bad_in_window;
        else { sumc.add(c); sumc2.add(c * c); sumxc = fmaf((float)k, c, sumxc); }

        const float tr = TR(idx);
        const unsigned char ftr = (unsigned char)is_finite_f(tr);
        v_tr[k] = ftr; ring_tr[k] = tr;
        if (!ftr) ++bad_tr_window;
        else sumtr.add(tr);

        while (!dq_max.empty() && H(dq_max.back()) <= h) dq_max.pop_back();
        dq_max.push_back(idx);
        while (!dq_min.empty() && Lw(dq_min.back()) >= l) dq_min.pop_back();
        dq_min.push_back(idx);
    }
    int ring_head = 0;


    if (bad_in_window == 0 && bad_tr_window == 0) {
        const float mean = sumc.val() * invL;
        const float var  = fmaxf(sumc.val() * invL * mean * 0.f , 0.f);
        (void)var;

        const float highest = H(dq_max.front());
        const float lowest  = Lw(dq_min.front());
        const float midpoint = 0.5f * (highest + lowest);
        const float mean_c = sumc.val() * invL;
        const float var_c  = fmaxf(sumc2.val() * invL - mean_c * mean_c, 0.0f);
        const float dkc    = sumtr.val() * invL;
        const float dkc2   = dkc * dkc;

        const float bbv = bb_sq * var_c;
        const float t_low  = kl_sq * dkc2;
        const float t_mid  = km_sq * dkc2;
        const float t_high = kh_sq * dkc2;
        sq[warm * num_series] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));

        const float avg = 0.5f * (midpoint + mean_c);
        const float S0  = sumc.val() - n * avg;
        const float S1  = sumxc - avg * sx;
        const float slope = (den != 0.0f) ? ( (n * S1 - sx * S0) * inv_den ) : 0.0f;
        const float intercept = (S0 - slope * sx) * (1.0f / n);
        const float yhat_last = intercept + slope * (n - 1.0f);
        mo[warm * num_series] = yhat_last;
    } else {
        mo[warm * num_series] = TTM_QNAN_F;
        sq[warm * num_series] = TTM_QNAN_F;
    }

    for (int i = warm + 1; i < series_len; ++i) {
        const int idx_new = i;
        const int idx_old = i - L;
        const int slot = ring_head;

        while (!dq_max.empty() && dq_max.front() <= idx_old) dq_max.pop_front();
        while (!dq_min.empty() && dq_min.front() <= idx_old) dq_min.pop_front();

        const float h_new = H(idx_new);
        const float l_new = Lw(idx_new);
        const float c_new = C(idx_new);
        const unsigned char fin_new = (unsigned char)(is_finite_f(h_new) & is_finite_f(l_new) & is_finite_f(c_new));
        const float tr_new = TR(idx_new);
        const unsigned char ftr_new = (unsigned char)is_finite_f(tr_new);

        const float c_old = ring_c[slot];
        const float tr_old = ring_tr[slot];
        const unsigned char fin_old = v_in[slot];
        const unsigned char ftr_old = v_tr[slot];

        bad_in_window += (int)!fin_new - (int)!fin_old;
        bad_tr_window += (int)!ftr_new - (int)!ftr_old;

        const float sumc_before = sumc.val();
        if (fin_old) { sumc.add(-c_old); sumc2.add(-(c_old * c_old)); }
        if (fin_new) { sumc.add( c_new); sumc2.add(  c_new * c_new ); }

        float adj_old = (fin_old ? c_old : 0.0f);
        float adj_new = (fin_new ? c_new : 0.0f);
        sumxc = fmaf(-1.0f, (sumc_before - adj_old), sumxc);
        sumxc = fmaf((float)(L - 1), adj_new,        sumxc);

        if (ftr_old) sumtr.add(-tr_old);
        if (ftr_new) sumtr.add( tr_new);

        ring_c[slot] = c_new; v_in[slot] = fin_new;
        ring_tr[slot] = tr_new; v_tr[slot] = ftr_new;
        ring_head = (ring_head + 1 == L) ? 0 : ring_head + 1;

        while (!dq_max.empty() && H(dq_max.back()) <= h_new) dq_max.pop_back();
        dq_max.push_back(idx_new);
        while (!dq_min.empty() && Lw(dq_min.back()) >= l_new) dq_min.pop_back();
        dq_min.push_back(idx_new);

        if (bad_in_window == 0 && bad_tr_window == 0) {
            const float mean_c = sumc.val() * invL;
            const float var_c  = fmaxf(sumc2.val() * invL - mean_c * mean_c, 0.0f);
            const float dkc    = sumtr.val() * invL;
            const float dkc2   = dkc * dkc;

            const float bbv = bb_sq * var_c;
            const float t_low  = kl_sq * dkc2;
            const float t_mid  = km_sq * dkc2;
            const float t_high = kh_sq * dkc2;
            sq[i * num_series] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));

            const float highest = H(dq_max.front());
            const float lowest  = Lw(dq_min.front());
            const float avg = 0.5f * (0.5f * (highest + lowest) + mean_c);
            const float S0  = sumc.val() - n * avg;
            const float S1  = sumxc - avg * sx;
            const float slope = (den != 0.0f) ? ( (n * S1 - sx * S0) * inv_den ) : 0.0f;
            const float intercept = (S0 - slope * sx) * (1.0f / n);
            const float yhat_last = intercept + slope * (n - 1.0f);
            mo[i * num_series] = yhat_last;
        } else {
            mo[i * num_series] = TTM_QNAN_F;
            sq[i * num_series] = TTM_QNAN_F;
        }
    }
}

