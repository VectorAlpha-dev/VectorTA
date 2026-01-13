











#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "../ds_float2.cuh"

static __device__ __forceinline__ float qnan() {
    return __int_as_float(0x7fffffff);
}





extern "C" __global__
void mfi_batch_f32(const float* __restrict__ typical,
                   const float* __restrict__ volume,
                   int series_len,
                   int first_valid,
                   const int* __restrict__ periods,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;
    const int period = periods[combo];
    const int row_off = combo * series_len;
    const int warm = first_valid + period - 1;


    if (blockIdx.x != 0) return;


    for (int t = threadIdx.x; t < min(warm, series_len); t += blockDim.x) {
        out[row_off + t] = qnan();
    }

    if (threadIdx.x != 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;
    if (warm >= series_len) return;


    dsf pos_sum = ds_set(0.0f), neg_sum = ds_set(0.0f);
    float prev = typical[first_valid];
    for (int i = first_valid + 1; i <= warm; ++i) {
        const float tp = typical[i];
        const float vol = volume[i];
        const float diff = tp - prev;
        prev = tp;
        const float flow = tp * vol;

        const float posf = (diff > 0.0f) ? flow : 0.0f;
        const float negf = (diff < 0.0f) ? flow : 0.0f;
        pos_sum = ds_add(pos_sum, ds_set(posf));
        neg_sum = ds_add(neg_sum, ds_set(negf));
    }

    float pos0 = ds_to_f(pos_sum);
    float neg0 = ds_to_f(neg_sum);
    float tot = pos0 + neg0;
    out[row_off + warm] = (tot <= 1e-14f) ? 0.0f : (100.0f * (pos0 / tot));


    for (int t = warm + 1; t < series_len; ++t) {

        const float tp_new = typical[t];
        const float vol_new = volume[t];
        const float diff_new = tp_new - typical[t - 1];
        const float flow_new = tp_new * vol_new;
        if (diff_new > 0.0f) pos_sum = ds_add(pos_sum, ds_set(flow_new));
        else if (diff_new < 0.0f) neg_sum = ds_add(neg_sum, ds_set(flow_new));


        {
            const int i = t - period;
            const float tp_old = typical[i];
            const float diff_old = tp_old - typical[i - 1];
            const float flow_old = tp_old * volume[i];
            if (diff_old > 0.0f) pos_sum = ds_sub(pos_sum, ds_set(flow_old));
            else if (diff_old < 0.0f) neg_sum = ds_sub(neg_sum, ds_set(flow_old));
        }

        pos0 = ds_to_f(pos_sum);
        neg0 = ds_to_f(neg_sum);
        tot = pos0 + neg0;
        out[row_off + t] = (tot <= 1e-14f) ? 0.0f : (100.0f * (pos0 / tot));
    }
}





extern "C" __global__
void mfi_many_series_one_param_f32(const float* __restrict__ typical_tm,
                                   const float* __restrict__ volume_tm,
                                   const int* __restrict__ first_valids,
                                   int period,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || series_len <= 0 || period <= 0) return;
    const int first = first_valids[s];
    const int stride = num_series;


    if (first < 0 || first >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * stride + s] = qnan();
        }
        return;
    }
    const int warm = first + period - 1;
    for (int t = threadIdx.x; t < min(warm, series_len); t += blockDim.x) {
        out_tm[t * stride + s] = qnan();
    }

    if (threadIdx.x != 0) return;


    extern __shared__ float2 shared[];
    float2* pos_buf = shared;
    float2* neg_buf = shared + period;

    for (int i = 0; i < period; ++i) { pos_buf[i] = make_float2(0.0f, 0.0f); neg_buf[i] = make_float2(0.0f, 0.0f); }


    dsf pos_sum = ds_set(0.0f), neg_sum = ds_set(0.0f);
    float prev = typical_tm[first * stride + s];
    int ring = 0;
    for (int t = first + 1; t <= warm && t < series_len; ++t) {
        const float tp = typical_tm[t * stride + s];
        const float vol = volume_tm[t * stride + s];
        const float diff = tp - prev;
        prev = tp;
        const float flow = tp * vol;
        const float posv = (diff > 0.0f) ? flow : 0.0f;
        const float negv = (diff < 0.0f) ? flow : 0.0f;
        pos_buf[ring] = make_float2(posv, 0.0f);
        neg_buf[ring] = make_float2(negv, 0.0f);
        pos_sum = ds_add(pos_sum, ds_set(posv));
        neg_sum = ds_add(neg_sum, ds_set(negv));
        ring += 1; if (ring == period) ring = 0;
    }

    if (warm < series_len) {

        const float tot0 = ds_to_f(pos_sum) + ds_to_f(neg_sum);
        out_tm[warm * stride + s] = (tot0 <= 1e-14f) ? 0.0f : (100.0f * (ds_to_f(pos_sum) / tot0));
    }


    for (int t = warm + 1; t < series_len; ++t) {
        const float tp = typical_tm[t * stride + s];
        const float vol = volume_tm[t * stride + s];
        const float diff = tp - prev;
        prev = tp;
        const float flow = tp * vol;


        dsf old_pos = ds_make(pos_buf[ring].x, pos_buf[ring].y);
        dsf old_neg = ds_make(neg_buf[ring].x, neg_buf[ring].y);
        pos_sum = ds_sub(pos_sum, old_pos); neg_sum = ds_sub(neg_sum, old_neg);


        const float posv = (diff > 0.0f) ? flow : 0.0f;
        const float negv = (diff < 0.0f) ? flow : 0.0f;
        pos_buf[ring] = make_float2(posv, 0.0f);
        neg_buf[ring] = make_float2(negv, 0.0f);
        pos_sum = ds_add(pos_sum, ds_set(posv));
        neg_sum = ds_add(neg_sum, ds_set(negv));
        ring += 1; if (ring == period) ring = 0;

        const float tot = ds_to_f(pos_sum) + ds_to_f(neg_sum);
        out_tm[t * stride + s] = (tot <= 1e-14f) ? 0.0f : (100.0f * (ds_to_f(pos_sum) / tot));
    }
}






extern "C" __global__
void mfi_prefix_stage1_transform_scan_ds(const float* __restrict__ typical,
                                         const float* __restrict__ volume,
                                         int series_len,
                                         int first_valid,
                                         float2* __restrict__ pos_ps,
                                         float2* __restrict__ neg_ps,
                                         float2* __restrict__ blk_tot_pos,
                                         float2* __restrict__ blk_tot_neg) {
    const int N = series_len;
    const int gid0 = blockIdx.x * blockDim.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int n_in_tile = min(blockDim.x, N - gid0);
    if (n_in_tile <= 0) return;

    int i = gid0 + tid;
    float posv = 0.f, negv = 0.f;
    if (tid < n_in_tile) {
        const float tp = typical[i];
        const float vol = volume[i];
        if (i > first_valid) {
            const float diff = tp - typical[i - 1];
            const float flow = tp * vol;
            posv = (diff > 0.f) ? flow : 0.f;
            negv = (diff < 0.f) ? flow : 0.f;
        }
    }

    dsf vpos = ds_make(posv, 0.f);
    dsf vneg = ds_make(negv, 0.f);

    unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offs = 1; offs < 32; offs <<= 1) {
        float hi_pos_up = __shfl_up_sync(mask, vpos.hi, offs);
        float lo_pos_up = __shfl_up_sync(mask, vpos.lo, offs);
        float hi_neg_up = __shfl_up_sync(mask, vneg.hi, offs);
        float lo_neg_up = __shfl_up_sync(mask, vneg.lo, offs);
        if (lane >= offs) {
            vpos = ds_add(vpos, ds_make(hi_pos_up, lo_pos_up));
            vneg = ds_add(vneg, ds_make(hi_neg_up, lo_neg_up));
        }
    }

    __shared__ dsf warp_tot_pos[32];
    __shared__ dsf warp_tot_neg[32];

    if (lane == 31) {
        warp_tot_pos[warp] = vpos;
        warp_tot_neg[warp] = vneg;
    }
    __syncthreads();

    dsf warp_prefix_pos = ds_set(0.f), warp_prefix_neg = ds_set(0.f);
    if (warp == 0) {
        const int nwarps = (blockDim.x + 31) / 32;
        dsf wvpos = (lane < nwarps) ? warp_tot_pos[lane] : ds_set(0.f);
        dsf wvneg = (lane < nwarps) ? warp_tot_neg[lane] : ds_set(0.f);
#pragma unroll
        for (int offs = 1; offs < 32; offs <<= 1) {
            float hi_pos_up = __shfl_up_sync(mask, wvpos.hi, offs);
            float lo_pos_up = __shfl_up_sync(mask, wvpos.lo, offs);
            float hi_neg_up = __shfl_up_sync(mask, wvneg.hi, offs);
            float lo_neg_up = __shfl_up_sync(mask, wvneg.lo, offs);
            if (lane >= offs) {
                wvpos = ds_add(wvpos, ds_make(hi_pos_up, lo_pos_up));
                wvneg = ds_add(wvneg, ds_make(hi_neg_up, lo_neg_up));
            }
        }
        warp_tot_pos[lane] = wvpos;
        warp_tot_neg[lane] = wvneg;
    }
    __syncthreads();

    if (warp > 0) {
        warp_prefix_pos = warp_tot_pos[warp - 1];
        warp_prefix_neg = warp_tot_neg[warp - 1];
        vpos = ds_add(vpos, warp_prefix_pos);
        vneg = ds_add(vneg, warp_prefix_neg);
    }

    if (tid < n_in_tile) {
        pos_ps[i] = make_float2(vpos.hi, vpos.lo);
        neg_ps[i] = make_float2(vneg.hi, vneg.lo);
    }

    if (tid == n_in_tile - 1) {
        blk_tot_pos[blockIdx.x] = make_float2(vpos.hi, vpos.lo);
        blk_tot_neg[blockIdx.x] = make_float2(vneg.hi, vneg.lo);
    }
}


extern "C" __global__
void mfi_prefix_stage2_scan_block_totals(const float2* __restrict__ blk_tot_pos,
                                         const float2* __restrict__ blk_tot_neg,
                                         float2* __restrict__ blk_off_pos,
                                         float2* __restrict__ blk_off_neg,
                                         int num_blocks) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    dsf run_pos = ds_set(0.f), run_neg = ds_set(0.f);
    for (int b = 0; b < num_blocks; ++b) {
        blk_off_pos[b] = make_float2(run_pos.hi, run_pos.lo);
        blk_off_neg[b] = make_float2(run_neg.hi, run_neg.lo);
        dsf tbp = ds_make(blk_tot_pos[b].x, blk_tot_pos[b].y);
        dsf tbn = ds_make(blk_tot_neg[b].x, blk_tot_neg[b].y);
        run_pos = ds_add(run_pos, tbp);
        run_neg = ds_add(run_neg, tbn);
    }
}


extern "C" __global__
void mfi_prefix_stage3_add_offsets(float2* __restrict__ pos_ps,
                                   float2* __restrict__ neg_ps,
                                   const float2* __restrict__ blk_off_pos,
                                   const float2* __restrict__ blk_off_neg,
                                   int series_len) {
    const int gid0 = blockIdx.x * blockDim.x;
    const int i = gid0 + threadIdx.x;
    if (i >= series_len) return;

    dsf off_pos = ds_make(blk_off_pos[blockIdx.x].x, blk_off_pos[blockIdx.x].y);
    dsf off_neg = ds_make(blk_off_neg[blockIdx.x].x, blk_off_neg[blockIdx.x].y);

    dsf vpos = ds_make(pos_ps[i].x, pos_ps[i].y);
    dsf vneg = ds_make(neg_ps[i].x, neg_ps[i].y);

    vpos = ds_add(vpos, off_pos);
    vneg = ds_add(vneg, off_neg);

    pos_ps[i] = make_float2(vpos.hi, vpos.lo);
    neg_ps[i] = make_float2(vneg.hi, vneg.lo);
}


extern "C" __global__
void mfi_batch_from_prefix_ds_f32(const float2* __restrict__ pos_ps,
                                  const float2* __restrict__ neg_ps,
                                  int series_len,
                                  int first_valid,
                                  const int* __restrict__ periods,
                                  int n_combos,
                                  float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int row_off = combo * series_len;
    const int warm = first_valid + period - 1;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int t = t0; t < series_len; t += stride) {
        if (first_valid < 0 || first_valid >= series_len || warm >= series_len) {
            out[row_off + t] = qnan();
            continue;
        }
        if (t < warm) { out[row_off + t] = qnan(); continue; }

        dsf p_t  = ds_make(pos_ps[t].x, pos_ps[t].y);
        dsf n_t  = ds_make(neg_ps[t].x, neg_ps[t].y);
        dsf p_lo = ds_set(0.f);
        dsf n_lo = ds_set(0.f);
        const int idx_old = t - period;
        if (idx_old >= 0) {
            p_lo = ds_make(pos_ps[idx_old].x, pos_ps[idx_old].y);
            n_lo = ds_make(neg_ps[idx_old].x, neg_ps[idx_old].y);
        }
        dsf pos = ds_sub(p_t, p_lo);
        dsf neg = ds_sub(n_t, n_lo);
        const float posf = ds_to_f(pos);
        const float totf = posf + ds_to_f(neg);
        out[row_off + t] = (totf <= 1e-14f) ? 0.0f : (100.0f * (posf / totf));
    }
}
