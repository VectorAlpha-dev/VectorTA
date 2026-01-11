













#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif



static __forceinline__ __device__
void hp2_coeffs_f32(float period, float &c, float &two_oma, float &oma_sq) {
    const float p = fmaxf(period, 1.0f);
    float s, co;
    
    sincospif(1.4142135623730951f / p, &s, &co);
    const float alpha = 1.0f + ((s - 1.0f) / co);
    const float t = 1.0f - 0.5f * alpha;
    c = t * t;
    const float oma = 1.0f - alpha;
    two_oma = 2.0f * oma;
    oma_sq = oma * oma;
}


#ifndef DECOSC_TILE_T
#define DECOSC_TILE_T 2048
#endif


extern "C" __global__ void dec_osc_batch_f32(
    const float* __restrict__ prices,   
    const int*   __restrict__ periods,  
    const float* __restrict__ ks,       
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out             
){
    
    const int blocks_needed = (n_combos + blockDim.x - 1) / blockDim.x;
    if (blockIdx.x >= blocks_needed) return;

    
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (combo < n_combos);

    __shared__ float s_prices[DECOSC_TILE_T];

    
    int    period    = 0;
    float  kf        = 0.0f;
    int    base_idx  = 0;
    if (active) {
        period   = periods[combo];
        kf       = ks[combo];
        base_idx = combo * series_len;

        
        const int prefix_len = (first_valid + 2 < series_len) ? (first_valid + 2) : series_len;
        for (int i = 0; i < prefix_len; ++i) {
            out[base_idx + i] = CUDART_NAN_F;
        }
    }

    
    bool valid = false;
    if (active) {
        valid = (period >= 2 && period <= series_len &&
                 first_valid >= 0 && first_valid < series_len &&
                 (series_len - first_valid) >= 2);
    }

    
    float c1=0, two_oma1=0, oma1_sq=0;
    float c2=0, two_oma2=0, oma2_sq=0;
    float scale = 0.0f;
    if (valid) {
        const float p  = (float)period;
        const float hp = 0.5f * p;
        hp2_coeffs_f32(p,  c1, two_oma1, oma1_sq);
        hp2_coeffs_f32(hp, c2, two_oma2, oma2_sq);
        scale = 100.0f * kf;
    }

    
    float x2=0.0f, x1=0.0f;
    float hp_prev_2=0.0f, hp_prev_1=0.0f;
    float decosc_prev_2=0.0f, decosc_prev_1=0.0f;

    if (valid) {
        const int i0 = first_valid;
        const int i1 = first_valid + 1;
        x2 = prices[i0];
        x1 = prices[i1];
        hp_prev_2 = x2;
        hp_prev_1 = x1;
        decosc_prev_2 = 0.0f;
        decosc_prev_1 = 0.0f;
    }

    
    for (int tile_start = first_valid + 2; tile_start < series_len; tile_start += DECOSC_TILE_T) {
        const int tile_end = min(series_len, tile_start + DECOSC_TILE_T);
        const int tile_len = tile_end - tile_start;

        
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            s_prices[t] = prices[tile_start + t];
        }
        __syncthreads();

        if (valid) {
            
            for (int t = 0; t < tile_len; ++t) {
                const int i = tile_start + t;
                const float d0 = s_prices[t];

                const float dx  = d0 - 2.0f * x1 + x2;
                const float hp0 = fmaf(c1, dx, fmaf(two_oma1, hp_prev_1, -oma1_sq * hp_prev_2));

                const float dec    = d0 - hp0;
                const float d_dec1 = x1 - hp_prev_1;
                const float d_dec2 = x2 - hp_prev_2;
                const float decdx  = dec - 2.0f * d_dec1 + d_dec2;
                const float osc0   = fmaf(c2, decdx, fmaf(two_oma2, decosc_prev_1, -oma2_sq * decosc_prev_2));

                out[base_idx + i] = scale * (osc0 / d0);

                
                hp_prev_2      = hp_prev_1;
                hp_prev_1      = hp0;
                decosc_prev_2  = decosc_prev_1;
                decosc_prev_1  = osc0;
                x2 = x1;
                x1 = d0;
            }
        }

        __syncthreads();
    }
}


extern "C" __global__ void dec_osc_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,  
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float k,
    float* __restrict__ out_tm           
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    const int first = first_valids[s];
    if (UNLIKELY(period < 2 || period > series_len ||
                 first < 0 || first >= series_len ||
                 series_len - first < 2)) {
        
        for (int t = 0; t < series_len; ++t) {
            out_tm[t * num_series + s] = CUDART_NAN_F;
        }
        return;
    }

    
    const int prefix_len = first + 2;
    for (int t = 0; t < prefix_len; ++t) {
        out_tm[t * num_series + s] = CUDART_NAN_F;
    }

    float c1, two_oma1, oma1_sq;
    float c2, two_oma2, oma2_sq;
    const float p  = (float)period;
    const float hp = 0.5f * p;
    hp2_coeffs_f32(p,  c1, two_oma1, oma1_sq);
    hp2_coeffs_f32(hp, c2, two_oma2, oma2_sq);

    const float scale = 100.0f * k;

    auto load_tm  = [&](int t) { return prices_tm[t * num_series + s]; };
    auto store_tm = [&](int t, float v) { out_tm[t * num_series + s] = v; };

    const int i0 = first;
    const int i1 = first + 1;

    float x2 = load_tm(i0);
    float x1 = load_tm(i1);
    float hp_prev_2 = x2;
    float hp_prev_1 = x1;
    float decosc_prev_2 = 0.0f;
    float decosc_prev_1 = 0.0f;

    for (int t = first + 2; t < series_len; ++t) {
        const float d0 = load_tm(t);
        const float dx  = d0 - 2.0f * x1 + x2;
        const float hp0 = fmaf(c1, dx, fmaf(two_oma1, hp_prev_1, -oma1_sq * hp_prev_2));

        const float dec    = d0 - hp0;
        const float d_dec1 = x1 - hp_prev_1;
        const float d_dec2 = x2 - hp_prev_2;
        const float decdx  = dec - 2.0f * d_dec1 + d_dec2;
        const float osc0   = fmaf(c2, decdx, fmaf(two_oma2, decosc_prev_1, -oma2_sq * decosc_prev_2));

        store_tm(t, scale * (osc0 / d0));

        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp0;
        decosc_prev_2 = decosc_prev_1;
        decosc_prev_1 = osc0;
        x2 = x1;
        x1 = d0;
    }
}
