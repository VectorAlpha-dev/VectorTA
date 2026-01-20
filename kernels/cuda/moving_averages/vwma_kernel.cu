






#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>



#ifndef LDG
#  if __CUDA_ARCH__ >= 350
#    define LDG(p) __ldg(p)
#  else
#    define LDG(p) (*(p))
#  endif
#endif


__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }

extern "C" __global__
void vwma_prefix_pv_vol_f64_f32(const float* __restrict__ prices,
                                const float* __restrict__ volumes,
                                int series_len,
                                int first_valid,
                                double* __restrict__ pv_prefix,
                                double* __restrict__ vol_prefix) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (series_len <= 0) return;

    if (first_valid < 0) first_valid = 0;
    if (first_valid > series_len) first_valid = series_len;

    double acc_pv = 0.0;
    double acc_vol = 0.0;

    for (int i = 0; i < first_valid; ++i) {
        pv_prefix[i] = 0.0;
        vol_prefix[i] = 0.0;
    }

    for (int i = first_valid; i < series_len; ++i) {
        const float p = prices[i];
        const float v = volumes[i];
        if (isnan(p) || isnan(v) || isnan(acc_pv) || isnan(acc_vol)) {
            acc_pv = NAN;
            acc_vol = NAN;
        } else {
            acc_pv += (double)p * (double)v;
            acc_vol += (double)v;
        }
        pv_prefix[i] = acc_pv;
        vol_prefix[i] = acc_vol;
    }
}

extern "C" __global__
void vwma_batch_f32(const double* __restrict__ pv_prefix,
                    const double* __restrict__ vol_prefix,
                    const int*    __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int warm   = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        float value;
        if (t < warm) {
            value = nan_f32();
        } else {

            const int prev = t - period;


            double sum_pv  = LDG(&pv_prefix[t]);
            double sum_vol = LDG(&vol_prefix[t]);

            if (prev >= 0) {
                sum_pv  -= LDG(&pv_prefix[prev]);
                sum_vol -= LDG(&vol_prefix[prev]);
            }

            value = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                     : nan_f32();
        }
        out[base_out + t] = value;
        t += stride;
    }
}

extern "C" __global__
void vwma_multi_series_one_param_f32(const double* __restrict__ pv_prefix_tm,
                                     const double* __restrict__ vol_prefix_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const int warm = first_valids[series_idx] + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = nan_f32();
        } else {
            const int prev = t - period;
            const int idx  = out_idx;

            double sum_pv  = LDG(&pv_prefix_tm[idx]);
            double sum_vol = LDG(&vol_prefix_tm[idx]);

            if (prev >= 0) {
                const int prev_idx = prev * num_series + series_idx;
                sum_pv  -= LDG(&pv_prefix_tm[prev_idx]);
                sum_vol -= LDG(&vol_prefix_tm[prev_idx]);
            }

            out_tm[out_idx] = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                               : nan_f32();
        }
        t += stride;
    }
}



extern "C" __global__
void vwma_multi_series_one_param_tm_coalesced_f32(const double* __restrict__ pv_prefix_tm,
                                                  const double* __restrict__ vol_prefix_tm,
                                                  int period,
                                                  int num_series,
                                                  int series_len,
                                                  const int* __restrict__ first_valids,
                                                  float* __restrict__ out_tm)
{

    const int series_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;



    const int warm = first_valids[series_idx] + period - 1;


    for (int t = blockIdx.x * blockDim.y + threadIdx.y;
         t < series_len;
         t += gridDim.x * blockDim.y)
    {
        const int out_idx = t * num_series + series_idx;

        if (t < warm) {
            out_tm[out_idx] = nan_f32();
            continue;
        }

        const int prev = t - period;


        double sum_pv  = LDG(&pv_prefix_tm[out_idx]);
        double sum_vol = LDG(&vol_prefix_tm[out_idx]);

        if (prev >= 0) {
            const int prev_idx = prev * num_series + series_idx;
            sum_pv  -= LDG(&pv_prefix_tm[prev_idx]);
            sum_vol -= LDG(&vol_prefix_tm[prev_idx]);
        }

        out_tm[out_idx] = (sum_vol != 0.0) ? __double2float_rn(sum_pv / sum_vol)
                                           : nan_f32();
    }
}
