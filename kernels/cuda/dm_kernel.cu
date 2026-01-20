
















#include <cuda_runtime.h>
#include <math.h>

#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif




__device__ __forceinline__ float qnan() {
    return __int_as_float(0x7fc00000);
}




template <typename T>
__device__ __forceinline__ T ro_load(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}


__device__ __forceinline__ void fill_nan_prefix(float* ptr, int len) {
    const float nanv = qnan();
    for (int i = 0; i < len; ++i) ptr[i] = nanv;
}


__device__ __forceinline__ void dm_step(float ch, float cl, float& prev_h, float& prev_l,
                                        float& plus_val, float& minus_val)
{
    const float dp = ch - prev_h;
    const float dm = prev_l - cl;
    prev_h = ch;
    prev_l = cl;

    const float ap = (dp > 0.0f) ? dp : 0.0f;
    const float am = (dm > 0.0f) ? dm : 0.0f;


    const bool take_p = (ap > am);
    plus_val  = take_p ? ap : 0.0f;
    minus_val = take_p ? 0.0f : am;
}


struct CompSum {
    float s;
    float c;
    __device__ __forceinline__ void init() { s = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {

        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};



struct CompEMA {
    float s;
    float c;
    __device__ __forceinline__ void init(float s0) { s = s0; c = 0.0f; }
    __device__ __forceinline__ void update(float one_minus_rp, float x) {

        float prod = s * one_minus_rp;
        float perr = __fmaf_rn(s, one_minus_rp, -prod);

        float y = (x + perr) - c;
        float t = prod + y;
        c = (t - prod) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};



extern "C" __global__
void dm_batch_f32(const float* __restrict__ high,
                  const float* __restrict__ low,
                  const int*   __restrict__ periods,
                  int series_len,
                  int n_combos,
                  int first_valid,
                  float* __restrict__ plus_out,
                  float* __restrict__ minus_out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* plus_row  = plus_out  + combo * series_len;
    float* minus_row = minus_out + combo * series_len;

    const int p = periods[combo];
    if (p <= 0) {

        fill_nan_prefix(plus_row, series_len);
        fill_nan_prefix(minus_row, series_len);
        return;
    }
    if (first_valid < 0 || first_valid + p - 1 >= series_len) {
        fill_nan_prefix(plus_row, series_len);
        fill_nan_prefix(minus_row, series_len);
        return;
    }

    const int i0 = first_valid;
    const int warm_end = i0 + p - 1;


    if (warm_end > 0) {
        fill_nan_prefix(plus_row,  warm_end);
        fill_nan_prefix(minus_row, warm_end);
    }


    float prev_h = ro_load(high + i0);
    float prev_l = ro_load(low  + i0);


    CompSum wplus, wminus; wplus.init(); wminus.init();
    for (int i = i0 + 1; i <= warm_end; ++i) {
        const float ch = ro_load(high + i);
        const float cl = ro_load(low  + i);
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);
        if (pv != 0.0f) wplus.add(pv);
        if (mv != 0.0f) wminus.add(mv);
    }


    plus_row [warm_end] = wplus.value();
    minus_row[warm_end] = wminus.value();


    if (warm_end + 1 >= series_len) return;

    const float rp = 1.0f / (float)p;
    const float one_minus_rp = 1.0f - rp;


    CompEMA splus, sminus;
    splus.init(plus_row [warm_end]);
    sminus.init(minus_row[warm_end]);

    for (int i = warm_end + 1; i < series_len; ++i) {
        const float ch = ro_load(high + i);
        const float cl = ro_load(low  + i);

        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);

        splus.update(one_minus_rp, pv);
        sminus.update(one_minus_rp, mv);

        plus_row [i] = splus.value();
        minus_row[i] = sminus.value();
    }
}



extern "C" __global__
void dm_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ plus_tm,
    float* __restrict__ minus_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (period <= 0 || fv < 0 || fv + period - 1 >= rows) {

        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            plus_tm [idx] = qnan();
            minus_tm[idx] = qnan();
        }
        return;
    }


    auto at = [&](int t) { return t * cols + s; };

    const int i0 = fv;
    const int warm_end = i0 + period - 1;


    for (int t = 0; t < warm_end; ++t) {
        const int idx = at(t);
        plus_tm [idx] = qnan();
        minus_tm[idx] = qnan();
    }

    float prev_h = ro_load(high_tm + at(i0));
    float prev_l = ro_load(low_tm  + at(i0));


    CompSum wplus, wminus; wplus.init(); wminus.init();
    for (int t = i0 + 1; t <= warm_end; ++t) {
        const float ch = ro_load(high_tm + at(t));
        const float cl = ro_load(low_tm  + at(t));
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);
        if (pv != 0.0f) wplus.add(pv);
        if (mv != 0.0f) wminus.add(mv);
    }

    plus_tm [at(warm_end)] = wplus.value();
    minus_tm[at(warm_end)] = wminus.value();

    if (warm_end + 1 >= rows) return;

    const float rp = 1.0f / (float)period;
    const float one_minus_rp = 1.0f - rp;

    CompEMA splus, sminus;
    splus.init(plus_tm [at(warm_end)]);
    sminus.init(minus_tm[at(warm_end)]);

    for (int t = warm_end + 1; t < rows; ++t) {
        const float ch = ro_load(high_tm + at(t));
        const float cl = ro_load(low_tm  + at(t));
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);

        splus.update(one_minus_rp, pv);
        sminus.update(one_minus_rp, mv);

        plus_tm [at(t)] = splus.value();
        minus_tm[at(t)] = sminus.value();
    }
}
