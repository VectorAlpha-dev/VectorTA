








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

namespace {


    __device__ __forceinline__ constexpr int kTile() { return 8; }



    __device__ __forceinline__ double epma_weight_sum(int p1, int offset) {
        return 0.5 * static_cast<double>(p1) *
               (static_cast<double>(p1) + 3.0 - 2.0 * static_cast<double>(offset));
    }
}

extern "C" __global__
void epma_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ offsets,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int offset = offsets[combo];
    const int p1     = period - 1;
    if (p1 <= 0) return;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);


    const int warm = first_valid + period + offset + 1;

    const int base_out = combo * series_len;

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            out[base_out + t] = NAN;
        }
        if (t_end <= warm) continue;


        int t0 = (t_start < warm ? warm : t_start);


        int a = t0 + 1 - p1;
        int b = t0;




        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = static_cast<double>(prices[idx]);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }


        out[base_out + t0] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);


        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = static_cast<double>(prices[old_a]);
            double entering = static_cast<double>(prices[b]);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            out[base_out + t] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}

extern "C" __global__
void epma_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int offset,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int p1 = period - 1;
    if (p1 <= 0) return;

    const int s = blockIdx.y;
    if (s >= num_series) return;

    const int warm = first_valids[s] + period + offset + 1;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    auto load_tm = [&](int t) -> double {
        long long in_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
        return static_cast<double>(prices_tm[in_idx]);
    };

    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = NAN;
        }
        if (t_end <= warm) continue;

        int t0 = (t_start < warm ? warm : t_start);

        int a = t0 + 1 - p1;
        int b = t0;

        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = load_tm(idx);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }

        {
            long long out_idx = static_cast<long long>(t0) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }

        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = load_tm(old_a);
            double entering = load_tm(b);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}
