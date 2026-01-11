





#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>



#ifndef CG_NAN
#define CG_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


static __device__ __forceinline__ bool cg_bad_den(float den) {
    return (!isfinite(den)) || fabsf(den) <= 1.1920929e-7f; 
}



struct CompSum {
    float s;
    float c;
    __device__ __forceinline__ CompSum() : s(0.0f), c(0.0f) {}
    __device__ __forceinline__ void add(float x) {
        float t = s + x;
        c += (fabsf(s) >= fabsf(x)) ? (s - t) + x : (x - t) + s;
        s  = t;
    }
    __device__ __forceinline__ void sub(float x) { add(-x); }
    __device__ __forceinline__ float val() const { return s + c; }
};





extern "C" __global__ void cg_batch_f32(const float* __restrict__ prices,
                                        const int*   __restrict__ periods,
                                        int series_len,
                                        int n_combos,
                                        int first_valid,
                                        float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    float* __restrict__ out_ptr = out + base;

    
    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;   
    const int window = period - 1;             

    
    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;

    if (window <= 0) {
        
        for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0f;
        return;
    }

    
    
    CompSum S_acc, T_acc;
    int nan_count = 0;

    
    for (int k = 0; k < window; ++k) {
        const float p = prices[warm - k];
        if (isfinite(p)) {
            S_acc.add(p);
            T_acc.add(fmaf((float)(k + 1), p, 0.0f));
        } else {
            nan_count++;
        }
    }

    
    {
        const float S = S_acc.val();
        out_ptr[warm] = (nan_count > 0 || cg_bad_den(S)) ? 0.0f : (-T_acc.val() / S);
    }

    
    
    
    
    const int REFRESH_EVERY = 512;
    int since_refresh = 0;
    for (int i = warm; i < series_len - 1; ++i) {
        const float add  = prices[i + 1];               
        const float drop = prices[i - window + 1];      

        if (isfinite(add))  S_acc.add(add); else ++nan_count;
        if (isfinite(drop)) S_acc.sub(drop); else --nan_count;

        
        T_acc.add(S_acc.val());
        
        if (isfinite(drop)) T_acc.sub((float)window * drop);

        
        const float S = S_acc.val();
        out_ptr[i + 1] = (nan_count > 0 || cg_bad_den(S)) ? 0.0f : (-T_acc.val() / S);

        
        if (++since_refresh >= REFRESH_EVERY) {
            since_refresh = 0;
            
            CompSum S_new, T_new;
            int nc = 0;
            const int cur = i + 1;
            for (int k = 0; k < window; ++k) {
                const float p = prices[cur - k];
                if (isfinite(p)) { S_new.add(p); T_new.add(fmaf((float)(k + 1), p, 0.0f)); }
                else { nc++; }
            }
            S_acc = S_new;
            T_acc = T_new;
            nan_count = nc;
        }
    }
}





extern "C" __global__ void cg_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const float* __restrict__ col_in  = prices_tm + series;
    float*       __restrict__ col_out = out_tm    + series;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        
        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;

    
    for (int row = 0; row < warm; ++row)
        col_out[(size_t)row * num_series] = CG_NAN;

    if (window <= 0) {
        for (int row = warm; row < series_len; ++row)
            col_out[(size_t)row * num_series] = 0.0f;
        return;
    }

    
    CompSum S_acc, T_acc;
    int nan_count = 0;
    for (int k = 0; k < window; ++k) {
        const float p = col_in[(size_t)(warm - k) * num_series];
        if (isfinite(p)) {
            S_acc.add(p);
            T_acc.add(fmaf((float)(k + 1), p, 0.0f));
        } else {
            nan_count++;
        }
    }

    {
        const float S = S_acc.val();
        col_out[(size_t)warm * num_series] =
            (nan_count > 0 || cg_bad_den(S)) ? 0.0f : (-T_acc.val() / S);
    }

    const int REFRESH_EVERY = 512;
    int since_refresh = 0;
    for (int row = warm; row < series_len - 1; ++row) {
        const float add  = col_in[(size_t)(row + 1) * num_series];
        const float drop = col_in[(size_t)(row - window + 1) * num_series];

        if (isfinite(add))  S_acc.add(add); else ++nan_count;
        if (isfinite(drop)) S_acc.sub(drop); else --nan_count;

        T_acc.add(S_acc.val());
        if (isfinite(drop)) T_acc.sub((float)window * drop);

        const float S = S_acc.val();
        col_out[(size_t)(row + 1) * num_series] =
            (nan_count > 0 || cg_bad_den(S)) ? 0.0f : (-T_acc.val() / S);

        if (++since_refresh >= REFRESH_EVERY) {
            since_refresh = 0;
            CompSum S_new, T_new;
            int nc = 0;
            const int cur = row + 1;
            for (int k = 0; k < window; ++k) {
                const float p = col_in[(size_t)(cur - k) * num_series];
                if (isfinite(p)) { S_new.add(p); T_new.add(fmaf((float)(k + 1), p, 0.0f)); }
                else { nc++; }
            }
            S_acc = S_new;
            T_acc = T_new;
            nan_count = nc;
        }
    }
}










extern "C" __global__ void cg_prefix_prepare_f32(const float* __restrict__ prices,
                                                 int series_len,
                                                 float* __restrict__ P,
                                                 float* __restrict__ Q,
                                                 int*   __restrict__ C)
{
    
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float ps = 0.0f;
    float qs = 0.0f;
    int   cs = 0;
    for (int i = 0; i < series_len; ++i) {
        const float p = prices[i];
        if (isfinite(p)) {
            ps += p;
            qs = fmaf((float)i, p, qs);
        } else {
            cs += 1;
        }
        P[i] = ps;
        Q[i] = qs;
        C[i] = cs;
    }
}


extern "C" __global__ void cg_batch_f32_from_prefix(
    const float* __restrict__ /*prices*/,   
    const int*   __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    const float* __restrict__ P,
    const float* __restrict__ Q,
    const int*   __restrict__ C,
    float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* __restrict__ out_ptr = out + (size_t)combo * series_len;

    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;

    
    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;
    if (window <= 0) {
        for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0f;
        return;
    }

    
    for (int i = warm; i < series_len; ++i) {
        const int a = i - window + 1;
        const int b = i;

        const float sumP = (P[b] - (a > 0 ? P[a - 1] : 0.0f));
        const float sumQ = (Q[b] - (a > 0 ? Q[a - 1] : 0.0f));
        const int   nans = (C[b] - (a > 0 ? C[a - 1] : 0));

        if (nans > 0 || cg_bad_den(sumP)) {
            out_ptr[i] = 0.0f;
        } else {
            
            const float num = fmaf((float)(i + 1), sumP, -sumQ);
            out_ptr[i] = -num / sumP;
        }
    }
}

