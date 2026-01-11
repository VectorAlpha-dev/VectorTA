






#include <cuda_runtime.h>
#include <math.h>

#ifndef PRB_BATCH_CHUNK_LEN
#define PRB_BATCH_CHUNK_LEN 4096
#endif

extern "C" {


__constant__ float PRB_BINOM_SIGN[8][8] = {
    {  1.0f,  0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f },
    { -1.0f,  1.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f },
    {  1.0f, -2.0f,   1.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f },
    { -1.0f,  3.0f,  -3.0f,   1.0f,   0.0f,   0.0f,   0.0f,   0.0f },
    {  1.0f, -4.0f,   6.0f,  -4.0f,   1.0f,   0.0f,   0.0f,   0.0f },
    { -1.0f,  5.0f, -10.0f,  10.0f,  -5.0f,   1.0f,   0.0f,   0.0f },
    {  1.0f, -6.0f,  15.0f, -20.0f,  15.0f,  -6.0f,   1.0f,   0.0f },
    { -1.0f,  7.0f, -21.0f,  35.0f, -35.0f,  21.0f,  -7.0f,   1.0f }
};

__device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }

__device__ __forceinline__ float horner_eval(const float* coeffs, int m, float x) {
    
    float acc = 0.0f;
    #pragma unroll
    for (int p = m - 1; p >= 0; --p) {
        acc = fmaf(acc, x, coeffs[p]); 
    }
    return acc;
}


__device__ __forceinline__ float kahan_add(float sum, float x, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    return t;
}

__device__ __forceinline__ void solve_coeffs_kahan(
    const float* __restrict__ arow,
    int max_m,
    int m,
    const float* __restrict__ S,
    float* __restrict__ coeffs) {
    #pragma unroll
    for (int r = 0; r < m; ++r) {
        float acc = 0.0f, c = 0.0f;
        const float* rowp = arow + r * max_m; 
        #pragma unroll
        for (int cidx = 0; cidx < m; ++cidx) {
            acc = kahan_add(acc, rowp[cidx] * S[cidx], c);
        }
        coeffs[r] = acc;
    }
}

__global__ void prb_batch_f32(   
    const float* __restrict__ data,  
    const int len,
    const int first_valid,
    const int* __restrict__ periods, 
    const int* __restrict__ orders,  
    const int* __restrict__ offsets, 
    const int combos,
    const int max_m,                 
    const float* __restrict__ a_inv, 
    const int a_stride,
    const int* __restrict__ contig,  
    const float ndev,
    const int* __restrict__ row_indices, 
    float* __restrict__ out_main,
    float* __restrict__ out_up,
    float* __restrict__ out_lo)
{
    const int row = blockIdx.y;
    if (row >= combos) return;

    const int abs_row = row_indices ? row_indices[row] : row;
    const int n = periods[row];
    const int k = orders[row];
    const int m = k + 1;
    const int offset = offsets[row];
    const float x_pos = float(n) - float(offset); 

    const float* arow = a_inv + row * a_stride;   

    
    const int warm = first_valid + n - 1;
    const float nan = qnan32();

    
    float npow[8]; npow[0] = 1.0f;
    #pragma unroll
    for (int r = 1; r <= k; ++r) npow[r] = npow[r-1] * float(n);

    
    for (int i = 0; i < warm && i < len; ++i) {
        const int out_idx = abs_row * len + i;
        out_main[out_idx] = nan;
        out_up[out_idx]   = nan;
        out_lo[out_idx]   = nan;
    }
    if (warm >= len) return;

    
    if (contig[warm] < n) {
        for (int i = warm; i < len; ++i) {
            const int out_idx = abs_row * len + i;
            out_main[out_idx] = nan;
            out_up[out_idx]   = nan;
            out_lo[out_idx]   = nan;
        }
        return;
    }

    
    float S[8];  
    float cS[8]; 
    #pragma unroll
    for (int r = 0; r < 8; ++r) { S[r] = 0.0f; cS[r] = 0.0f; }

    float sum = 0.0f, csum = 0.0f;
    float sumsq = 0.0f, csum2 = 0.0f;

    const int base0 = warm - n + 1;
    for (int j = 1; j <= n; ++j) {
        const float y = data[base0 + j - 1];
        
        sum   = kahan_add(sum, y, csum);
        sumsq = kahan_add(sumsq, y * y, csum2);

        
        float pwr = float(j);
        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            S[r] = kahan_add(S[r], y * pwr, cS[r]);
            pwr *= float(j);
        }
    }
    S[0] = sum;

    
    {
        float coeffs[8];
        solve_coeffs_kahan(arow, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean);
        if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        const int out_idx = abs_row * len + warm;
        out_main[out_idx] = reg;
        out_up[out_idx]   = reg + ndev * stdev;
        out_lo[out_idx]   = reg - ndev * stdev;
    }

    
    bool poisoned = false;
    float S_old[8];

    for (int i = warm + 1; i < len; ++i) {
        const int out_idx = abs_row * len + i;

        if (poisoned || contig[i] < n) {
            poisoned = true;
            out_main[out_idx] = nan;
            out_up[out_idx]   = nan;
            out_lo[out_idx]   = nan;
            continue;
        }

        
        #pragma unroll
        for (int r = 0; r <= k; ++r) S_old[r] = S[r];

        const float y_old = data[i - n];
        const float y_new = data[i];

        
        sum   = kahan_add(sum, -y_old, csum);
        sum   = kahan_add(sum,  y_new, csum);
        S[0]  = sum;
        sumsq = kahan_add(sumsq, -y_old * y_old, csum2);
        sumsq = kahan_add(sumsq,  y_new * y_new, csum2);

        
        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            float acc = 0.0f, c = 0.0f;
            #pragma unroll
            for (int p = 0; p <= r; ++p) {
                acc = kahan_add(acc, PRB_BINOM_SIGN[r][p] * S_old[p], c);
            }
            
            S[r] = fmaf(y_new, npow[r], acc);
        }

        
        float coeffs[8];
        solve_coeffs_kahan(arow, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean);
        if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        out_main[out_idx] = reg;
        out_up[out_idx]   = reg + ndev * stdev;
        out_lo[out_idx]   = reg - ndev * stdev;
    }
}





__global__ void prb_batch_chunked_f32(
    const float* __restrict__ data,
    const int len,
    const int first_valid,
    const int* __restrict__ periods,
    const int* __restrict__ orders,
    const int* __restrict__ offsets,
    const int combos,
    const int max_m,
    const float* __restrict__ a_inv,
    const int a_stride,
    const int* __restrict__ contig,
    const float ndev,
    const int* __restrict__ row_indices,
    float* __restrict__ out_main,
    float* __restrict__ out_up,
    float* __restrict__ out_lo)
{
    (void)contig; 

    const int row = (int)blockIdx.y;
    if (row >= combos) return;

    const int chunk_id = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int chunk_start = chunk_id * PRB_BATCH_CHUNK_LEN;
    if (chunk_start >= len) return;
    const int chunk_end = min(chunk_start + PRB_BATCH_CHUNK_LEN, len);

    const int abs_row = row_indices ? row_indices[row] : row;
    const int n = periods[row];
    const int k = orders[row];
    const int m = k + 1;
    const int offset = offsets[row];
    const float x_pos = float(n) - float(offset); 

    const float* arow = a_inv + row * a_stride;
    const int warm = first_valid + n - 1;
    const float nan = qnan32();

    
    float npow[8]; npow[0] = 1.0f;
    #pragma unroll
    for (int r = 1; r <= k; ++r) npow[r] = npow[r - 1] * float(n);

    
    if (chunk_end <= warm) {
        for (int i = chunk_start; i < chunk_end; ++i) {
            const int out_idx = abs_row * len + i;
            out_main[out_idx] = nan;
            out_up[out_idx]   = nan;
            out_lo[out_idx]   = nan;
        }
        return;
    }

    int i0 = chunk_start;
    for (; i0 < warm && i0 < chunk_end; ++i0) {
        const int out_idx = abs_row * len + i0;
        out_main[out_idx] = nan;
        out_up[out_idx]   = nan;
        out_lo[out_idx]   = nan;
    }
    if (i0 >= chunk_end) return;

    
    float S[8];
    float cS[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) { S[r] = 0.0f; cS[r] = 0.0f; }

    float sum = 0.0f, csum = 0.0f;
    float sumsq = 0.0f, csum2 = 0.0f;

    const int base0 = i0 - n + 1;
    for (int j = 1; j <= n; ++j) {
        const float y = data[base0 + j - 1];
        sum   = kahan_add(sum, y, csum);
        sumsq = kahan_add(sumsq, y * y, csum2);

        float pwr = float(j);
        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            S[r] = kahan_add(S[r], y * pwr, cS[r]);
            pwr *= float(j);
        }
    }
    S[0] = sum;

    
    {
        float coeffs[8];
        solve_coeffs_kahan(arow, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean);
        if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        const int out_idx = abs_row * len + i0;
        out_main[out_idx] = reg;
        out_up[out_idx]   = reg + ndev * stdev;
        out_lo[out_idx]   = reg - ndev * stdev;
    }

    
    float S_old[8];
    for (int i = i0 + 1; i < chunk_end; ++i) {
        const int out_idx = abs_row * len + i;

        #pragma unroll
        for (int r = 0; r <= k; ++r) S_old[r] = S[r];

        const float y_old = data[i - n];
        const float y_new = data[i];

        sum   = kahan_add(sum, -y_old, csum);
        sum   = kahan_add(sum,  y_new, csum);
        S[0]  = sum;
        sumsq = kahan_add(sumsq, -y_old * y_old, csum2);
        sumsq = kahan_add(sumsq,  y_new * y_new, csum2);

        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            float acc = 0.0f, c = 0.0f;
            #pragma unroll
            for (int p = 0; p <= r; ++p) {
                acc = kahan_add(acc, PRB_BINOM_SIGN[r][p] * S_old[p], c);
            }
            S[r] = fmaf(y_new, npow[r], acc);
        }

        float coeffs[8];
        solve_coeffs_kahan(arow, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean);
        if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        out_main[out_idx] = reg;
        out_up[out_idx]   = reg + ndev * stdev;
        out_lo[out_idx]   = reg - ndev * stdev;
    }
}

__global__ void prb_many_series_one_param_f32( 
    const float* __restrict__ data_tm, 
    const int cols,
    const int rows,
    const int period,
    const int order,
    const int offset,
    const int max_m,
    const float* __restrict__ a_inv, 
    const int a_stride,
    const int* __restrict__ contig_tm,    
    const int* __restrict__ first_valids, 
    const float ndev,
    float* __restrict__ out_main_tm,
    float* __restrict__ out_up_tm,
    float* __restrict__ out_lo_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; 
    if (s >= cols) return;

    const int n = period;
    const int k = order;
    const int m = k + 1;
    const float x_pos = float(n) - float(offset);
    const float* ainv = a_inv;

    const float nan = qnan32();
    const int fv = first_valids ? first_valids[s] : 0;
    const int warm = fv + n - 1;

    
    float npow[8]; npow[0] = 1.0f;
    #pragma unroll
    for (int r = 1; r <= k; ++r) npow[r] = npow[r-1] * float(n);

    
    for (int t = 0; t < rows && t < warm; ++t) {
        const int idx = t * cols + s;
        out_main_tm[idx] = nan;
        out_up_tm[idx]   = nan;
        out_lo_tm[idx]   = nan;
    }
    if (warm >= rows) return;

    
    if (contig_tm[warm * cols + s] < n) {
        for (int t = warm; t < rows; ++t) {
            const int idx = t * cols + s;
            out_main_tm[idx] = nan;
            out_up_tm[idx]   = nan;
            out_lo_tm[idx]   = nan;
        }
        return;
    }

    
    float S[8];
    float cS[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) { S[r] = 0.0f; cS[r] = 0.0f; }

    float sum = 0.0f, csum = 0.0f;
    float sumsq = 0.0f, csum2 = 0.0f;

    const int base0 = warm - n + 1;
    for (int j = 1; j <= n; ++j) {
        const float y = data_tm[(base0 + j - 1) * cols + s];
        sum   = kahan_add(sum, y, csum);
        sumsq = kahan_add(sumsq, y * y, csum2);

        float pwr = float(j);
        #pragma unroll
        for (int r = 1; r <= k; ++r) { S[r] = kahan_add(S[r], y * pwr, cS[r]); pwr *= float(j); }
    }
    S[0] = sum;

    
    {
        float coeffs[8];
        solve_coeffs_kahan(ainv, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean); if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        const int idx = warm * cols + s;
        out_main_tm[idx] = reg;
        out_up_tm[idx]   = reg + ndev * stdev;
        out_lo_tm[idx]   = reg - ndev * stdev;
    }

    
    bool poisoned = false;
    float S_old[8];

    for (int t = warm + 1; t < rows; ++t) {
        const int idx = t * cols + s;

        if (poisoned || contig_tm[idx] < n) {
            poisoned = true;
            out_main_tm[idx] = nan;
            out_up_tm[idx]   = nan;
            out_lo_tm[idx]   = nan;
            continue;
        }

        #pragma unroll
        for (int r = 0; r <= k; ++r) S_old[r] = S[r];

        const float y_old = data_tm[(t - n) * cols + s];
        const float y_new = data_tm[t * cols + s];

        sum   = kahan_add(sum, -y_old, csum);
        sum   = kahan_add(sum,  y_new, csum);
        S[0]  = sum;
        sumsq = kahan_add(sumsq, -y_old * y_old, csum2);
        sumsq = kahan_add(sumsq,  y_new * y_new, csum2);

        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            float acc = 0.0f, c = 0.0f;
            #pragma unroll
            for (int p = 0; p <= r; ++p) acc = kahan_add(acc, PRB_BINOM_SIGN[r][p] * S_old[p], c);
            S[r] = fmaf(y_new, npow[r], acc);
        }

        float coeffs[8]; solve_coeffs_kahan(ainv, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean); if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        out_main_tm[idx] = reg;
        out_up_tm[idx]   = reg + ndev * stdev;
        out_lo_tm[idx]   = reg - ndev * stdev;
    }
}

} 
