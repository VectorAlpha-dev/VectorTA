






#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>



#ifndef WTO_UTILS_H
#define WTO_UTILS_H


static __device__ __forceinline__ float wto_nan() {
    return __int_as_float(0x7fc00000u);
}



struct fpair {
    float hi, lo;
    __device__ __forceinline__ void init() { hi = 0.f; lo = 0.f; }
    __device__ __forceinline__ void add(float x) {
        float s  = hi + x;
        float bb = s - hi;
        float e  = (hi - (s - bb)) + (x - bb);
        float t  = lo + e;
        float st = s + t;
        lo = t - (st - s);
        hi = st;
    }
    __device__ __forceinline__ float value() const { return hi + lo; }
};



static __device__ __forceinline__ float bcast_price(const float* __restrict__ prices, int t) {
    unsigned mask   = __activemask();
    int      leader = __ffs(mask) - 1;
    float    v      = 0.f;
    int      lane   = threadIdx.x & 31;
    if (lane == leader) { v = __ldg(prices + t); }
    return __shfl_sync(mask, v, leader);
}

#endif



__device__ inline void fill_nan(float* ptr, int len) {
    const float nanv = wto_nan();
    for (int i = 0; i < len; ++i) {
        ptr[i] = nanv;
    }
}



extern "C" __global__
void wto_fill_nan3_f32(float* __restrict__ out_wt1,
                       float* __restrict__ out_wt2,
                       float* __restrict__ out_hist,
                       size_t total_elems) {
    const float qnan = wto_nan();
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elems;
         idx += (size_t)blockDim.x * gridDim.x) {
        out_wt1[idx] = qnan;
        out_wt2[idx] = qnan;
        out_hist[idx] = qnan;
    }
}

extern "C" __global__
void wto_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ channel_lengths,
                   const int*   __restrict__ average_lengths,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ wt1_out,
                   float* __restrict__ wt2_out,
                   float* __restrict__ hist_out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int chan = channel_lengths[combo];
    const int avg = average_lengths[combo];

    float* wt1_row  = wt1_out  + (size_t)combo * series_len;
    float* wt2_row  = wt2_out  + (size_t)combo * series_len;
    float* hist_row = hist_out + (size_t)combo * series_len;

    const float qnan = wto_nan();


    if (chan <= 0 || avg <= 0 || first_valid < 0 || first_valid >= series_len || series_len <= 0) {
        for (int t = 0; t < series_len; ++t) {
            wt1_row[t]  = qnan;
            wt2_row[t]  = qnan;
            hist_row[t] = qnan;
        }
        return;
    }
    const int start_ci = first_valid + chan - 1;
    if (start_ci >= series_len) {
        for (int t = 0; t < series_len; ++t) {
            wt1_row[t]  = qnan;
            wt2_row[t]  = qnan;
            hist_row[t] = qnan;
        }
        return;
    }


    const double alpha_ch = 2.0 / (double(chan) + 1.0);
    const double beta_ch  = 1.0 - alpha_ch;
    const double alpha_av = 2.0 / (double(avg) + 1.0);
    const double beta_av  = 1.0 - alpha_av;


    bool   esa_init = false;
    double esa = 0.0, d = 0.0, wt1 = 0.0;


    double ring[4] = {0.0, 0.0, 0.0, 0.0};
    double rsum = 0.0;
    int    rlen = 0;
    int    rpos = 0;


    const unsigned mask   = __activemask();
    const int      lane   = threadIdx.x & 31;
    const int      leader = __ffs(mask) - 1;

    for (int t = 0; t < series_len; ++t) {

        float price_f32 = 0.0f;
        if (lane == leader) {
            price_f32 = __ldg(prices + t);
        }
        price_f32 = __shfl_sync(mask, price_f32, leader);
        const bool   priceFinite = isfinite(price_f32);
        const double price      = static_cast<double>(price_f32);


        float wt1_f  = qnan;
        float wt2_f  = qnan;
        float hist_f = qnan;

        if (t < first_valid) {
            wt1_row[t]  = wt1_f;
            wt2_row[t]  = wt2_f;
            hist_row[t] = hist_f;
            continue;
        }

        if (!esa_init) {
            if (!priceFinite) {
                wt1_row[t]  = wt1_f;
                wt2_row[t]  = wt2_f;
                hist_row[t] = hist_f;
                continue;
            }
            esa = price; esa_init = true;
        } else if (priceFinite) {
            esa = fma(beta_ch, esa, alpha_ch * price);
        }

        const double diff = price - esa;

        if (t < start_ci) {
            wt1_row[t]  = wt1_f;
            wt2_row[t]  = wt2_f;
            hist_row[t] = hist_f;
            continue;
        }

        if (t == start_ci) {

            const double absdiff0 = priceFinite ? fabs(diff) : __longlong_as_double(0x7ff8000000000000ULL);
            d = absdiff0;
            const double denom0 = 0.015 * d;
            double ci0 = 0.0;
            if (denom0 != 0.0 && isfinite(denom0)) {
                if (priceFinite) { ci0 = (price - esa) / denom0; }
                else { ci0 = __longlong_as_double(0x7ff8000000000000ULL); }
            } else {
                ci0 = 0.0;
            }
            wt1 = ci0;

            wt1_f = static_cast<float>(wt1);
            ring[0] = wt1;
            rsum = wt1;
            rlen = 1;


            rpos = 0;
        } else if (t > start_ci) {
            const double abs_diff = fabs(diff);
            if (isfinite(abs_diff)) { d = fma(beta_ch, d, alpha_ch * abs_diff); }
            double ci;
            const double denom = 0.015 * d;
            if (priceFinite) {
                if (denom != 0.0 && isfinite(denom)) { ci = diff / denom; }
                else { ci = 0.0; }
            } else {
                ci = __longlong_as_double(0x7ff8000000000000ULL);
            }
            if (isfinite(ci)) { wt1 = fma(beta_av, wt1, alpha_av * ci); }
            wt1_f = static_cast<float>(wt1);

            if (rlen < 4) {
                ring[rlen] = wt1;
                rsum += wt1;
                ++rlen;
            } else {
                rsum += wt1 - ring[rpos];
                ring[rpos] = wt1;
                rpos = (rpos + 1) & 3;
            }
            if (rlen == 4) {
                const double wt2d = 0.25 * rsum;
                wt2_f  = static_cast<float>(wt2d);
                hist_f = static_cast<float>(wt1 - wt2d);
            }
        }

        wt1_row[t]  = wt1_f;
        wt2_row[t]  = wt2_f;
        hist_row[t] = hist_f;
    }
}

extern "C" __global__
void wto_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int cols,
    int rows,
    int channel_length,
    int average_length,
    const int* __restrict__ first_valids,
    float* __restrict__ wt1_tm,
    float* __restrict__ wt2_tm,
    float* __restrict__ hist_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) {
        return;
    }

    float* wt1_col  = wt1_tm + series;
    float* wt2_col  = wt2_tm + series;
    float* hist_col = hist_tm + series;


    const float qnan = wto_nan();
    for (int t = 0; t < rows; ++t) {
        wt1_col[t * cols]  = qnan;
        wt2_col[t * cols]  = qnan;
        hist_col[t * cols] = qnan;
    }

    if (channel_length <= 0 || average_length <= 0) {
        return;
    }

    const int first_valid = first_valids[series];
    const int start_ci = first_valid + channel_length - 1;

    const double alpha_ch = 2.0 / (double(channel_length) + 1.0);
    const double beta_ch  = 1.0 - alpha_ch;
    const double alpha_av = 2.0 / (double(average_length) + 1.0);
    const double beta_av  = 1.0 - alpha_av;

    bool   esa_init = false, d_init = false, wt1_init = false;
    double esa = 0.0, d = 0.0, wt1 = 0.0;


    double ring[4] = {0.0, 0.0, 0.0, 0.0};
    double rsum = 0.0;
    int    rlen = 0;
    int    rpos = 0;

    for (int t = 0; t < rows; ++t) {
        const float  price_f32 = __ldg(prices_tm + (size_t)t * cols + series);
        const bool   priceFinite = isfinite(price_f32);
        const double price = static_cast<double>(price_f32);

        if (t < first_valid) { continue; }

        if (!esa_init) { if (!priceFinite) continue; esa = price; esa_init = true; }
        else if (priceFinite) { esa = fma(beta_ch, esa, alpha_ch * price); }

        const double diff     = price - esa;
        const double abs_diff = fabs(diff);

        if (t == start_ci) {
            const double absdiff0 = priceFinite ? fabs(price - esa) : __longlong_as_double(0x7ff8000000000000ULL);
            d = absdiff0; d_init = true;
            const double denom0 = 0.015 * d;
            double ci0 = 0.0;
            if (denom0 != 0.0 && isfinite(denom0)) {
                if (priceFinite) { ci0 = (price - esa) / denom0; }
                else { ci0 = __longlong_as_double(0x7ff8000000000000ULL); }
            } else { ci0 = 0.0; }
            wt1 = ci0; wt1_init = true;
            wt1_col[t * cols] = static_cast<float>(wt1);

            ring[0] = wt1;
            rsum = wt1;
            rlen = 1;
            rpos = 1;
        } else if (t > start_ci) {
            if (isfinite(abs_diff)) { d = fma(beta_ch, d, alpha_ch * abs_diff); }
            const double denom = 0.015 * d;
            double ci;
            if (priceFinite) {
                if (denom != 0.0 && isfinite(denom)) { ci = diff / denom; }
                else { ci = 0.0; }
            } else { ci = __longlong_as_double(0x7ff8000000000000ULL); }
            if (isfinite(ci)) { wt1 = fma(beta_av, wt1, alpha_av * ci); }
            wt1_col[t * cols] = static_cast<float>(wt1);

            if (rlen < 4) {
                ring[rlen] = wt1;
                rsum += wt1;
                ++rlen;
            } else {
                rsum += wt1 - ring[rpos];
                ring[rpos] = wt1;
                rpos = (rpos + 1) & 3;
            }
            if (rlen == 4) {
                const double wt2d  = 0.25 * rsum;
                wt2_col[t * cols]  = static_cast<float>(wt2d);
                hist_col[t * cols] = static_cast<float>(wt1 - wt2d);
            }
        }
    }
}
