// Drop-in optimized rewrite for CUDA 13 / sm_89+
#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <limits.h>

// ---------- helpers ----------
#if __CUDACC_VER_MAJOR__ >= 9
// Read-only cached load. On modern GPUs this maps to the read-only path and is
// effective for broadcast/warp-uniform loads. Fallback to plain load if arch<3.5.
template <typename T>
__device__ __forceinline__ T ld_ro(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}
#else
template <typename T>
__device__ __forceinline__ T ld_ro(const T* p) { return *p; }
#endif

// Optional L2 prefetch (disabled by default).
#ifndef VWAP_PREFETCH_DISTANCE
#define VWAP_PREFETCH_DISTANCE 0
#endif

#if (VWAP_PREFETCH_DISTANCE > 0)
__device__ __forceinline__ void prefetch_l2(const void* ptr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
#endif
}
#endif

// ---------- Kernel 1: many-params × one time series (column-major output) ----------
struct VwapSeg2 {
    float vol;
    float pv;
    int head;
};

struct VwapSegOp {
    __device__ __forceinline__ VwapSeg2 operator()(const VwapSeg2& a, const VwapSeg2& b) const {
        if (b.head) return b;
        VwapSeg2 out;
        out.vol = a.vol + b.vol;
        out.pv = a.pv + b.pv;
        out.head = a.head;
        return out;
    }
};

extern "C" __global__
void vwap_batch_f32(const long long* __restrict__ timestamps,
                    const float* __restrict__ volumes,
                    const float* __restrict__ prices,
                    const int* __restrict__ counts,
                    const int* __restrict__ unit_codes,
                    const long long* __restrict__ divisors,
                    const int* __restrict__ first_valids,
                    const int* __restrict__ month_ids,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out)
{
    constexpr int BLOCK_THREADS = 256;
    if (blockDim.x != BLOCK_THREADS) return;

    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int count = counts[combo];
    const int unit = unit_codes[combo]; // 0=m,1=h,2=d,3=M
    long long divisor = divisors[combo];
    int warm = first_valids[combo];

    const int base = combo * series_len;
    const float nan = __int_as_float(0x7fffffff); // canonical qNaN

    if (count <= 0 || series_len <= 0) {
        for (int t = threadIdx.x; t < series_len; t += BLOCK_THREADS) {
            out[base + t] = nan;
        }
        return;
    }
    if (unit != 3 && divisor <= 0) divisor = 1;

    if (warm < 0) warm = 0;
    if (warm > series_len) warm = series_len;

    for (int t = threadIdx.x; t < warm; t += BLOCK_THREADS) {
        out[base + t] = nan;
    }
    if (warm >= series_len) return;

    using BlockScan = cub::BlockScan<VwapSeg2, BLOCK_THREADS>;
    __shared__ typename BlockScan::TempStorage scan_tmp;
    __shared__ long long sh_gid[BLOCK_THREADS];
    __shared__ float carry_vol;
    __shared__ float carry_pv;
    __shared__ long long carry_gid;

    if (threadIdx.x == 0) {
        carry_vol = 0.0f;
        carry_pv = 0.0f;
        carry_gid = LLONG_MIN;
    }
    __syncthreads();

    const int month_div = (unit == 3 && divisor > 0) ? static_cast<int>(divisor) : 1;
    const long long div = (unit != 3 && divisor > 0) ? divisor : 1;

#pragma unroll 1
    for (int chunk = warm; chunk < series_len; chunk += BLOCK_THREADS) {
        const int t = chunk + threadIdx.x;
        const bool active = (t < series_len);

        float vol = 0.0f;
        float pv = 0.0f;
        long long gid = 0;

        if (active) {
            vol = ld_ro(&volumes[t]);
            const float price = ld_ro(&prices[t]);
            pv = fmaf(vol, price, 0.0f);

            if (unit == 3) {
                const int mid = month_ids ? ld_ro(&month_ids[t]) : 0;
                const int md = (month_div > 0 ? month_div : 1);
                gid = static_cast<long long>(mid / md);
            } else {
                const long long ts = ld_ro(&timestamps[t]);
                gid = ts / div;
            }
        }

        sh_gid[threadIdx.x] = gid;
        __syncthreads();

        int head = 1;
        if (active) {
            if (t == warm) {
                head = 1;
            } else {
                const long long prev_gid = (threadIdx.x == 0) ? carry_gid : sh_gid[threadIdx.x - 1];
                head = (gid != prev_gid) ? 1 : 0;
            }
        }

        VwapSeg2 in{active ? vol : 0.0f, active ? pv : 0.0f, head};
        VwapSeg2 scanned;
        BlockScan(scan_tmp).InclusiveScan(in, scanned, VwapSegOp{});

        float vol_sum = scanned.vol;
        float pv_sum = scanned.pv;
        if (active && scanned.head == 0) {
            vol_sum += carry_vol;
            pv_sum += carry_pv;
        }

        if (active) {
            out[base + t] = (vol_sum > 0.0f) ? (pv_sum / vol_sum) : nan;
        }

        int valid = series_len - chunk;
        if (valid > BLOCK_THREADS) valid = BLOCK_THREADS;
        const int last = valid - 1;
        if (threadIdx.x == last) {
            carry_vol = vol_sum;
            carry_pv = pv_sum;
            carry_gid = gid;
        }
        __syncthreads();
    }
}

// ---------- Kernel 2: many series × one param (time-major I/O) ----------
extern "C" __global__
void vwap_multi_series_one_param_f32(const long long* __restrict__ timestamps,
                                     const float* __restrict__ volumes_tm,
                                     const float* __restrict__ prices_tm,
                                     int count,
                                     int unit_code,           // 0=m,1=h,2=d,3=M
                                     long long divisor,       // ms for m/h/d; ignored for 'M'
                                     const int* __restrict__ first_valids, // per series
                                     const int* __restrict__ month_ids,    // per time (rows), if unit_code==3
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    const int warm_raw = first_valids ? first_valids[series_idx] : 0;
    int warm = warm_raw;
    if (warm < 0) warm = 0;
    if (warm > series_len) warm = series_len;

    const float nan = __int_as_float(0x7fffffff); // canonical qNaN

    // Warmup NaNs
    for (int t = 0; t < warm; ++t)
        out_tm[t * num_series + series_idx] = nan;

    float volume_sum    = 0.0f;
    float vol_price_sum = 0.0f;

    const int  month_div = (unit_code == 3 && count > 0) ? count : 1;
    const long long div  = (unit_code != 3 && divisor > 0) ? divisor : 1;

    long long current_gid = LLONG_MIN;
    long long next_boundary_ll = LLONG_MIN; // for ts path
    int       next_boundary_i  = INT_MIN;   // for month path

    long long last_ts = LLONG_MIN;
    bool monotonic_ts = true;

#pragma unroll 1
    for (int t = warm; t < series_len; ++t) {

#if (VWAP_PREFETCH_DISTANCE > 0)
        const int tp = t + VWAP_PREFETCH_DISTANCE;
        if (tp < series_len) {
            const int next_idx = tp * num_series + series_idx;
            prefetch_l2(&volumes_tm[next_idx]);
            prefetch_l2(&prices_tm[next_idx]);
            prefetch_l2(&timestamps[tp]);
            if (unit_code == 3 && month_ids) prefetch_l2(&month_ids[tp]);
        }
#endif

        if (unit_code == 3) {
            const int mid = month_ids ? ld_ro(&month_ids[t]) : 0;
            if (t == warm) {
                current_gid = static_cast<long long>(mid / (month_div > 0 ? month_div : 1));
                next_boundary_i = (static_cast<int>(current_gid) + 1) * (month_div > 0 ? month_div : 1);
                volume_sum = 0.0f; vol_price_sum = 0.0f;
            } else if (mid >= next_boundary_i) {
                const int md = (month_div > 0 ? month_div : 1);
                const int adv = ((mid - next_boundary_i) / md) + 1;
                current_gid  += adv;
                next_boundary_i += adv * md;
                volume_sum = 0.0f; vol_price_sum = 0.0f;
            }
        } else {
            const long long ts = ld_ro(&timestamps[t]);
            if (t == warm) {
                current_gid = ts / div;
                const long long rem = ts - (current_gid * div);
                next_boundary_ll = ts - rem + div;
                last_ts = ts;
                volume_sum = 0.0f; vol_price_sum = 0.0f;
            } else {
                if (monotonic_ts && ts >= last_ts) {
                    if (ts >= next_boundary_ll) {
                        const long long adv = ((ts - next_boundary_ll) / div) + 1;
                        current_gid      += adv;
                        next_boundary_ll += adv * div;
                        volume_sum = 0.0f; vol_price_sum = 0.0f;
                    }
                } else {
                    const long long gid = ts / div;
                    if (gid != current_gid) {
                        current_gid = gid;
                        const long long rem = ts - (gid * div);
                        next_boundary_ll = ts - rem + div;
                        volume_sum = 0.0f; vol_price_sum = 0.0f;
                    }
                    if (ts < last_ts) monotonic_ts = false;
                }
                last_ts = ts;
            }
        }

        const int idx   = t * num_series + series_idx;
        const float vol = ld_ro(&volumes_tm[idx]);
        const float pr  = ld_ro(&prices_tm[idx]);

        volume_sum    += vol;
        vol_price_sum  = fmaf(vol, pr, vol_price_sum); // fused multiply-add

        out_tm[idx] = (volume_sum > 0.0f) ? (vol_price_sum / volume_sum) : nan;
    }
}
