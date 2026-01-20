








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


constexpr double REFLEX_PI_D = 3.14159265358979323846264338327950288;
constexpr double REFLEX_SQRT2_APPROX_D = 1.414;


static __device__ __forceinline__ int wrap_inc(int idx, int len) {
    idx += 1;
    return idx - (idx == len) * len;
}

static __device__ __forceinline__ float reflex_out_if_valid(double ms, double my_sum) {
    if (ms > 0.0 && isfinite(ms)) {
        return static_cast<float>(my_sum / sqrt(ms));
    }
    return 0.0f;
}




extern "C" __global__
void reflex_batch_f32(const float* __restrict__ prices,
                      const int*   __restrict__ periods,
                      int series_len,
                      int n_combos,
                      int ,
                      float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) { return; }

    const int period = __ldg(periods + combo);
    if (period < 2 || series_len <= 0) { return; }

    float* __restrict__ out_row = out + combo * series_len;


    const int warm = period < series_len ? period : series_len;
    for (int i = 0; i < warm; ++i) { out_row[i] = 0.0f; }


    int half_period_i = period / 2; if (half_period_i < 1) half_period_i = 1;
    const double half_p = static_cast<double>(half_period_i);
    const double a  = exp(-REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double a2 = a * a;
    const double b  = 2.0 * a * cos(REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double c  = 0.5 * (1.0 + a2 - b);


    extern __shared__ double sdata[];
    double* __restrict__ ring = sdata;
    const int ring_len = period + 1;


    if (series_len > 0) ring[0] = static_cast<double>(__ldg(prices + 0));
    if (series_len > 1) ring[1] = static_cast<double>(__ldg(prices + 1));


    double ssf_sum = ((series_len > 0) ? ring[0] : 0.0) + ((series_len > 1) ? ring[1] : 0.0);

    const double inv_p = 1.0 / static_cast<double>(period);
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta  = 1.0 - alpha;

    double ms = 0.0;


    int idx    = 2;
    int idx_im1 = 1;
    int idx_im2 = 0;
    int idx_ip  = 0;


    double dim1 = (series_len > 1) ? static_cast<double>(__ldg(prices + 1)) : 0.0;


    int i = 2;
    for (; i < series_len; ) {
    #pragma unroll 4
        for (int u = 0; u < 4; ++u) {
            if (i >= series_len) break;


            const double di = static_cast<double>(__ldg(prices + i));


            const double ssf_im1 = ring[idx_im1];
            const double ssf_im2 = ring[idx_im2];
            const double t0 = c * (di + dim1);
            const double t1 = fma(-a2, ssf_im2, t0);
            const double ssf_i = fma(b, ssf_im1, t1);


            ring[idx] = ssf_i;

            if (i < period) {
                ssf_sum += ssf_i;
            } else {
                const double ssf_old = ring[idx_ip];
                const double mean_lp = ssf_sum * inv_p;
                const double my_sum  = fma(beta, ssf_i, alpha * ssf_old) - mean_lp;

                ms = fma(0.96, ms, 0.04 * my_sum * my_sum);
                out_row[i] = reflex_out_if_valid(ms, my_sum);


                ssf_sum += ssf_i - ssf_old;


                idx_ip = wrap_inc(idx_ip, ring_len);
            }


            idx    = wrap_inc(idx,    ring_len);
            idx_im1 = wrap_inc(idx_im1, ring_len);
            idx_im2 = wrap_inc(idx_im2, ring_len);

            dim1 = di;
            ++i;
        }
    }
}




extern "C" __global__
void reflex_batch_f32_precomp(const float* __restrict__ prices,
                              const int*   __restrict__ periods,
                              const double* __restrict__ a2s,
                              const double* __restrict__ bs,
                              const double* __restrict__ cs,
                              int series_len,
                              int n_combos,
                              int ,
                              float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) { return; }

    const int period = __ldg(periods + combo);
    if (period < 2 || series_len <= 0) { return; }

    float* __restrict__ out_row = out + combo * series_len;
    const int warm = period < series_len ? period : series_len;
    for (int i = 0; i < warm; ++i) { out_row[i] = 0.0f; }

    const double a2 = __ldg(a2s + combo);
    const double b  = __ldg(bs  + combo);
    const double c  = __ldg(cs  + combo);

    extern __shared__ double sdata[];
    double* __restrict__ ring = sdata;
    const int ring_len = period + 1;

    if (series_len > 0) ring[0] = static_cast<double>(__ldg(prices + 0));
    if (series_len > 1) ring[1] = static_cast<double>(__ldg(prices + 1));

    double ssf_sum = ((series_len > 0) ? ring[0] : 0.0) + ((series_len > 1) ? ring[1] : 0.0);

    const double inv_p = 1.0 / static_cast<double>(period);
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta  = 1.0 - alpha;
    double ms = 0.0;

    int idx = 2, idx_im1 = 1, idx_im2 = 0, idx_ip = 0;
    double dim1 = (series_len > 1) ? static_cast<double>(__ldg(prices + 1)) : 0.0;

    int i = 2;
    for (; i < series_len; ) {
    #pragma unroll 4
        for (int u = 0; u < 4; ++u) {
            if (i >= series_len) break;

            const double di = static_cast<double>(__ldg(prices + i));

            const double ssf_im1 = ring[idx_im1];
            const double ssf_im2 = ring[idx_im2];
            const double t0 = c * (di + dim1);
            const double t1 = fma(-a2, ssf_im2, t0);
            const double ssf_i = fma(b, ssf_im1, t1);

            ring[idx] = ssf_i;

            if (i < period) {
                ssf_sum += ssf_i;
            } else {
                const double ssf_old = ring[idx_ip];
                const double mean_lp = ssf_sum * inv_p;
                const double my_sum  = fma(beta, ssf_i, alpha * ssf_old) - mean_lp;

                ms = fma(0.96, ms, 0.04 * my_sum * my_sum);
                out_row[i] = reflex_out_if_valid(ms, my_sum);

                ssf_sum += ssf_i - ssf_old;

                idx_ip = wrap_inc(idx_ip, ring_len);
            }

            idx    = wrap_inc(idx,    ring_len);
            idx_im1 = wrap_inc(idx_im1, ring_len);
            idx_im2 = wrap_inc(idx_im2, ring_len);

            dim1 = di;
            ++i;
        }
    }
}




extern "C" __global__
void reflex_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                      int period,
                                      int num_series,
                                      int series_len,
                                      const int* __restrict__ ,
                                      float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || threadIdx.x != 0) { return; }
    if (period < 2 || series_len <= 0) { return; }


    for (int t = 0; t < series_len; ++t) { out_tm[t * num_series + series] = 0.0f; }
    const int warm = period < series_len ? period : series_len;
    for (int t = 0; t < warm; ++t) { out_tm[t * num_series + series] = 0.0f; }

    int half_period_i = period / 2; if (half_period_i < 1) half_period_i = 1;
    const double half_p = static_cast<double>(half_period_i);
    const double a  = exp(-REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double a2 = a * a;
    const double b  = 2.0 * a * cos(REFLEX_SQRT2_APPROX_D * REFLEX_PI_D / half_p);
    const double c  = 0.5 * (1.0 + a2 - b);

    extern __shared__ double sdata[];
    double* ring = sdata;
    const int ring_len = period + 1;
    if (series_len > 0) ring[0] = static_cast<double>(prices_tm[0 * num_series + series]);
    if (series_len > 1) ring[1] = static_cast<double>(prices_tm[1 * num_series + series]);

    double ssf_sum = 0.0;
    if (period == 1) {
        ssf_sum = (series_len > 0) ? ring[0] : 0.0;
    } else {
        ssf_sum = ((series_len > 0) ? ring[0] : 0.0)
                + ((series_len > 1) ? ring[1] : 0.0);
    }
    const double inv_p = 1.0 / static_cast<double>(period);
    const double alpha = 0.5 * (1.0 + inv_p);
    const double beta  = 1.0 - alpha;
    double ms = 0.0;

    for (int t = 2; t < series_len; ++t) {
        const int idx     = t % ring_len;
        const int idx_im1 = (t - 1) % ring_len;
        const int idx_im2 = (t - 2) % ring_len;

        const double di   = static_cast<double>(prices_tm[t * num_series + series]);
        const double dim1 = static_cast<double>(prices_tm[(t - 1) * num_series + series]);
        const double ssf_im1 = ring[idx_im1];
        const double ssf_im2 = ring[idx_im2];

        const double t0 = c * (di + dim1);
        const double t1 = (-a2) * ssf_im2 + t0;
        const double ssf_t = b * ssf_im1 + t1;
        ring[idx] = ssf_t;

        if (t < period) { ssf_sum += ssf_t; continue; }

        const int idx_ip = (t - period) % ring_len;
        const double ssf_ip = ring[idx_ip];
        const double mean_lp = ssf_sum * inv_p;
        const double my_sum = beta * ssf_t + alpha * ssf_ip - mean_lp;

        ms = fma(0.96, ms, 0.04 * my_sum * my_sum);
        out_tm[t * num_series + series] = reflex_out_if_valid(ms, my_sum);


        ssf_sum += ssf_t - ssf_ip;
    }
}
