// CUDA kernels for NET MyRSI (Ehlers' MyRSI + Noise Elimination Technique)
//
// Reference implementation focused on parity with the scalar CPU path.
// Two-pass structure per row/series:
// 1) Compute MyRSI r[t] using an O(period) window-sum in double precision.
// 2) Compute NET using O(period^2) pairwise sign counts over r-window.
//
// Warmup/NaN semantics:
// - warm = first_valid + period - 1
// - out[warm] = 0.0f (if period > 1) else NaN
// - [0, warm) are NaN
// - Division by zero emits 0.0f

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Quiet NaN
static __device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }

// Max supported period for ring buffers (bounded local storage); host validates.
#ifndef NET_MYRSI_MAX_PERIOD
#define NET_MYRSI_MAX_PERIOD 2048
#endif

// No extra helpers required

// --- Batch path: one price series x many params (warp-per-combo mapping) ---
// Warp-parallel: one warp per combo, shared-memory FP64 rings.
// This accelerates NET's O(period) update loop while preserving parity.
static __device__ __forceinline__ int warp_reduce_sum_i32(int v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

extern "C" __global__
void net_myrsi_batch_f32_warp_dbl(const float* __restrict__ prices,   // one series
                                 const int*   __restrict__ periods,  // [n_combos]
                                 int series_len,
                                 int n_combos,
                                 int first_valid,
                                 int max_period,                     // max(periods) for this launch
                                 float* __restrict__ out)            // [n_combos, series_len]
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int combo = (int)(blockIdx.x * warps_per_block + warp);
    if (combo >= n_combos) return;

    int period = (lane == 0) ? periods[combo] : 0;
    period = __shfl_sync(0xffffffffu, period, 0);

    float* out_row = out + (size_t)combo * series_len;

    // Validate & prefill
    if (UNLIKELY(period <= 0 || period > max_period || max_period <= 0 ||
                 max_period > NET_MYRSI_MAX_PERIOD ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = lane; i < series_len; i += 32) out_row[i] = qnan32();
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < (period + 1))) {
        for (int i = lane; i < series_len; i += 32) out_row[i] = qnan32();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = lane; i < warm; i += 32) out_row[i] = qnan32();
    if (lane == 0) out_row[warm] = (period > 1) ? 0.0f : qnan32();

    // Shared memory layout (per block):
    // [ diffs[warps_per_block][max_period] ][ myr[warps_per_block][max_period] ]
    extern __shared__ double smem_dbl[];
    double* diffs = smem_dbl + (size_t)warp * (size_t)max_period;
    double* myr = smem_dbl + (size_t)warps_per_block * (size_t)max_period +
                  (size_t)warp * (size_t)max_period;

    for (int j = lane; j < period; j += 32) {
        myr[j] = 0.0;
    }
    __syncwarp();

    double cu = 0.0, cd = 0.0;
    int d_head = 0, d_count = 0;
    int m_head = 0, m_count = 0;
    int num = 0; // lane 0 only
    const float denom = 0.5f * (float)period * (float)(period - 1);

    for (int i = first_valid + 1; i < series_len; ++i) {
        double r = 0.0;
        if (lane == 0) {
            double newer = (double)prices[i];
            double older = (double)prices[i - 1];
            double diff  = newer - older;

            cu += ((diff > 0.0) ? 1.0 : 0.0) * diff;
            cd += ((diff < 0.0) ? 1.0 : 0.0) * (-diff);

            if (d_count < period) {
                diffs[d_head] = diff;
                d_head++; if (d_head == period) d_head = 0;
                d_count++;
            } else {
                double old = diffs[d_head];
                cu -= ((old > 0.0) ? 1.0 : 0.0) * old;
                cd -= ((old < 0.0) ? 1.0 : 0.0) * (-old);
                diffs[d_head] = diff;
                d_head++; if (d_head == period) d_head = 0;
            }

            double sum = cu + cd;
            r = (sum != 0.0) ? ((cu - cd) / sum) : 0.0;
        }
        r = __shfl_sync(0xffffffffu, r, 0);

        if (i >= first_valid + period) {
            if (m_count < period) {
                int local = 0;
                for (int j = lane; j < m_head; j += 32) {
                    double u = myr[j];
                    local += (u < r) - (u > r);
                }
                int add = warp_reduce_sum_i32(local);
                if (lane == 0) {
                    num += add;
                    myr[m_head] = r;
                }
                __syncwarp();

                // lockstep update across lanes
                m_head++; if (m_head == period) m_head = 0;
                m_count++;
            } else {
                double z = 0.0;
                if (lane == 0) z = myr[m_head];
                z = __shfl_sync(0xffffffffu, z, 0);

                int local_rm = 0;
                for (int j = lane; j < period; j += 32) {
                    if (j == m_head) continue;
                    double u = myr[j];
                    local_rm += (u < z) - (u > z);
                }
                int rm = warp_reduce_sum_i32(local_rm);

                int local_ad = 0;
                for (int j = lane; j < period; j += 32) {
                    if (j == m_head) continue;
                    double u = myr[j];
                    local_ad += (u < r) - (u > r);
                }
                int ad = warp_reduce_sum_i32(local_ad);

                if (lane == 0) {
                    num += rm;
                    num += ad;
                    myr[m_head] = r;
                }
                __syncwarp();

                // lockstep update across lanes
                m_head++; if (m_head == period) m_head = 0;
            }

            if (lane == 0) {
                out_row[i] = (denom != 0.0f) ? (float)((double)num / (double)denom) : 0.0f;
            }
        }
    }
}

// --- Batch path: one price series × many params (thread-per-combo mapping) ---
// Fast path: keep ring buffers in shared memory (transposed layout) to avoid huge
// per-thread local arrays (which spill to local/global memory) for common periods.
extern "C" __global__
void net_myrsi_batch_f32_shared_fast(const float* __restrict__ prices,   // one series
                                     const int*   __restrict__ periods,  // [n_combos]
                                     int series_len,
                                     int n_combos,
                                     int first_valid,
                                     int max_period,                     // max(periods) for this launch
                                     float* __restrict__ out)            // [n_combos, series_len]
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + (size_t)combo * series_len;

    // Validate & prefill
    if (UNLIKELY(period <= 0 || period > max_period || max_period <= 0 ||
                 max_period > NET_MYRSI_MAX_PERIOD ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) out_row[i] = qnan32();
    out_row[warm] = (period > 1) ? 0.0f : qnan32();

    // Shared memory (transposed): [max_period][blockDim.x]
    extern __shared__ float smem[];
    const int pitch = (int)blockDim.x;
    float* diffs = smem;
    double* myr  = (double*)(diffs + (size_t)max_period * pitch);

    // Init MyRSI ring to zeros (only indices < period are read before being overwritten)
    for (int j = 0; j < period; ++j) {
        myr[(size_t)j * pitch + threadIdx.x] = 0.0;
    }

    double cu = 0.0, cd = 0.0;
    int d_head = 0, d_count = 0;
    int m_head = 0, m_count = 0;
    int num = 0;
    const double denom = 0.5 * (double)period * (double)(period - 1);

    for (int i = first_valid + 1; i < series_len; ++i) {
        float diff = prices[i] - prices[i - 1];
        if (diff > 0.0f) cu += (double)diff;
        else if (diff < 0.0f) cd += (double)(-diff);

        if (d_count < period) {
            diffs[(size_t)d_head * pitch + threadIdx.x] = diff;
            d_head++; if (d_head == period) d_head = 0;
            d_count++;
        } else {
            float old = diffs[(size_t)d_head * pitch + threadIdx.x];
            if (old > 0.0f) cu -= (double)old;
            else if (old < 0.0f) cd -= (double)(-old);
            diffs[(size_t)d_head * pitch + threadIdx.x] = diff;
            d_head++; if (d_head == period) d_head = 0;
        }

        if (d_count >= period) {
            double sum = cu + cd;
            double r = (sum != 0.0) ? ((cu - cd) / sum) : 0.0;

            if (m_count < period) {
                int add = 0;
                for (int j = 0; j < m_head; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    add += (u < r) - (u > r);
                }
                num += add;
                myr[(size_t)m_head * pitch + threadIdx.x] = r;
                m_head++; if (m_head == period) m_head = 0;
                m_count++;
            } else {
                double z = myr[(size_t)m_head * pitch + threadIdx.x];
                int rm = 0;
                for (int j = m_head + 1; j < period; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    rm += (u < z) - (u > z);
                }
                for (int j = 0; j < m_head; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    rm += (u < z) - (u > z);
                }
                num += rm;

                int ad = 0;
                for (int j = m_head + 1; j < period; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    ad += (u < r) - (u > r);
                }
                for (int j = 0; j < m_head; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    ad += (u < r) - (u > r);
                }
                num += ad;

                myr[(size_t)m_head * pitch + threadIdx.x] = r;
                m_head++; if (m_head == period) m_head = 0;
            }

            out_row[i] = (denom != 0.0) ? (float)((double)num / denom) : 0.0f;
        }
    }
}

extern "C" __global__
void net_myrsi_batch_f32_shared_dbl(const float* __restrict__ prices,   // one series
                                    const int*   __restrict__ periods,  // [n_combos]
                                    int series_len,
                                    int n_combos,
                                    int first_valid,
                                    int max_period,                     // max(periods) for this launch
                                    float* __restrict__ out)            // [n_combos, series_len]
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + (size_t)combo * series_len;

    // Validate & prefill
    if (UNLIKELY(period <= 0 || period > max_period || max_period <= 0 ||
                 max_period > NET_MYRSI_MAX_PERIOD ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) out_row[i] = qnan32();
    out_row[warm] = (period > 1) ? 0.0f : qnan32();

    // Shared memory (transposed): [max_period][blockDim.x]
    extern __shared__ double smem_dbl[];
    const int pitch = (int)blockDim.x;
    double* diffs = smem_dbl;
    double* myr   = diffs + (size_t)max_period * pitch;

    // Init MyRSI ring to zeros (only indices < period are read before being overwritten)
    for (int j = 0; j < period; ++j) {
        myr[(size_t)j * pitch + threadIdx.x] = 0.0;
    }

    double cu = 0.0, cd = 0.0;
    int d_head = 0, d_count = 0;
    int m_head = 0, m_count = 0;
    int num = 0;
    const float denom = 0.5f * (float)period * (float)(period - 1);

    for (int i = first_valid + 1; i < series_len; ++i) {
        double newer = (double)prices[i];
        double older = (double)prices[i - 1];
        double diff  = newer - older;

        cu += ((diff > 0.0) ? 1.0 : 0.0) * diff;
        cd += ((diff < 0.0) ? 1.0 : 0.0) * (-diff);

        if (d_count < period) {
            diffs[(size_t)d_head * pitch + threadIdx.x] = diff;
            d_head++; if (d_head == period) d_head = 0;
            d_count++;
        } else {
            double old = diffs[(size_t)d_head * pitch + threadIdx.x];
            cu -= ((old > 0.0) ? 1.0 : 0.0) * old;
            cd -= ((old < 0.0) ? 1.0 : 0.0) * (-old);
            diffs[(size_t)d_head * pitch + threadIdx.x] = diff;
            d_head++; if (d_head == period) d_head = 0;
        }

        if (d_count >= period) {
            double sum = cu + cd;
            double r = (sum != 0.0) ? ((cu - cd) / sum) : 0.0;

            if (m_count < period) {
                int add = 0;
                for (int j = 0; j < m_head; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    add += (u < r) - (u > r);
                }
                num += add;
                myr[(size_t)m_head * pitch + threadIdx.x] = r;
                m_head++; if (m_head == period) m_head = 0;
                m_count++;
            } else {
                double z = myr[(size_t)m_head * pitch + threadIdx.x];
                int rm = 0;
                for (int j = m_head + 1; j < period; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    rm += (u < z) - (u > z);
                }
                for (int j = 0; j < m_head; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    rm += (u < z) - (u > z);
                }
                num += rm;

                int ad = 0;
                for (int j = m_head + 1; j < period; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    ad += (u < r) - (u > r);
                }
                for (int j = 0; j < m_head; ++j) {
                    double u = myr[(size_t)j * pitch + threadIdx.x];
                    ad += (u < r) - (u > r);
                }
                num += ad;

                myr[(size_t)m_head * pitch + threadIdx.x] = r;
                m_head++; if (m_head == period) m_head = 0;
            }

            out_row[i] = (denom != 0.0f) ? (float)((double)num / (double)denom) : 0.0f;
        }
    }
}

extern "C" __global__
void net_myrsi_batch_f32(const float* __restrict__ prices,   // one series
                         const int*   __restrict__ periods,  // [n_combos]
                         int series_len,
                         int n_combos,
                         int first_valid,
                         float* __restrict__ out)            // [n_combos, series_len]
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + (size_t)combo * series_len;

    // Validate & prefill
    if (UNLIKELY(period <= 0 || period > NET_MYRSI_MAX_PERIOD ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) out_row[i] = qnan32();
    out_row[warm] = (period > 1) ? 0.0f : qnan32();

    // Streaming parity: use diff ring (size=period) and myr ring
    double cu = 0.0, cd = 0.0;
    double diffs[NET_MYRSI_MAX_PERIOD];
    int d_head = 0, d_count = 0;
    double myr[NET_MYRSI_MAX_PERIOD];
    for (int j = 0; j < period; ++j) myr[j] = 0.0;
    int m_head = 0, m_count = 0;
    int num = 0;
    const float denom = 0.5f * (float)period * (float)(period - 1);

    for (int i = first_valid + 1; i < series_len; ++i) {
        double newer = (double)prices[i];
        double older = (double)prices[i - 1];
        double diff  = newer - older;

        cu += ((diff > 0.0) ? 1.0 : 0.0) * diff;
        cd += ((diff < 0.0) ? 1.0 : 0.0) * (-diff);

        if (d_count < period) {
            diffs[d_head] = diff;
            d_head++; if (d_head == period) d_head = 0;
            d_count++;
        } else {
            double old = diffs[d_head];
            cu -= ((old > 0.0) ? 1.0 : 0.0) * old;
            cd -= ((old < 0.0) ? 1.0 : 0.0) * (-old);
            diffs[d_head] = diff;
            d_head++; if (d_head == period) d_head = 0;
        }

        if (d_count >= period) {
            double sum = cu + cd;
            double r = (sum != 0.0) ? ((cu - cd) / sum) : 0.0;

            if (m_count < period) {
                int add = 0;
                for (int j = 0; j < m_head; ++j) { double u = myr[j]; add += (u < r) - (u > r); }
                num += add;
                myr[m_head] = r;
                m_head++; if (m_head == period) m_head = 0;
                m_count++;
            } else {
                double z = myr[m_head];
                int rm = 0;
                for (int j = m_head + 1; j < period; ++j) { double u = myr[j]; rm += (u < z) - (u > z); }
                for (int j = 0; j < m_head; ++j)       { double u = myr[j]; rm += (u < z) - (u > z); }
                num += rm;

                int ad = 0;
                for (int j = m_head + 1; j < period; ++j) { double u = myr[j]; ad += (u < r) - (u > r); }
                for (int j = 0; j < m_head; ++j)       { double u = myr[j]; ad += (u < r) - (u > r); }
                num += ad;

                myr[m_head] = r;
                m_head++; if (m_head == period) m_head = 0;
            }

            out_row[i] = (denom != 0.0f) ? (float)((double)num / (double)denom) : 0.0f;
        }
    }
}

// --- Many-series × one-param (time-major): thread-per-series mapping ---
extern "C" __global__
void net_myrsi_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm, // [rows * cols], time-major
    int cols, int rows, int period,
    const int* __restrict__ first_valids, // [cols]
    float* __restrict__ out_tm            // [rows * cols], time-major
){
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    const int fv = first_valids[series];
    float* col_out = out_tm + series;
    if (UNLIKELY(period <= 0 || period > NET_MYRSI_MAX_PERIOD || fv < 0 || fv >= rows)) {
        for (int t = 0; t < rows; ++t) col_out[t * cols] = qnan32();
        return;
    }
    const int tail = rows - fv;
    if (UNLIKELY(tail < (period + 1))) {
        for (int t = 0; t < rows; ++t) col_out[t * cols] = qnan32();
        return;
    }

    const int warm = fv + period - 1;
    for (int t = 0; t < warm; ++t) col_out[t * cols] = qnan32();
    col_out[warm * cols] = (period > 1) ? 0.0f : qnan32();


    // Streaming parity with explicit diff ring (time-major)
    double cu = 0.0, cd = 0.0;
    double diffs[NET_MYRSI_MAX_PERIOD];
    int d_head = 0, d_count = 0;
    double myr[NET_MYRSI_MAX_PERIOD];
    for (int j = 0; j < period; ++j) myr[j] = 0.0;
    int m_head = 0, m_count = 0, num = 0;
    const float denom = 0.5f * (float)period * (float)(period - 1);

    for (int i = fv + 1; i < rows; ++i) {
        double newer = (double)prices_tm[(size_t)i * cols + series];
        double older = (double)prices_tm[(size_t)(i - 1) * cols + series];
        double diff  = newer - older;
        cu += ((diff > 0.0) ? 1.0 : 0.0) * diff;
        cd += ((diff < 0.0) ? 1.0 : 0.0) * (-diff);

        if (d_count < period) {
            diffs[d_head] = diff;
            d_head++; if (d_head == period) d_head = 0;
            d_count++;
        } else {
            double old = diffs[d_head];
            cu -= ((old > 0.0) ? 1.0 : 0.0) * old;
            cd -= ((old < 0.0) ? 1.0 : 0.0) * (-old);
            diffs[d_head] = diff;
            d_head++; if (d_head == period) d_head = 0;
        }

        if (d_count >= period) {
            double sum = cu + cd;
            double r = (sum != 0.0) ? ((cu - cd) / sum) : 0.0;

            // insert into ring
            if (m_count < period) {
                myr[m_head] = r;
                m_head++; if (m_head == period) m_head = 0;
                m_count++;
            } else {
                myr[m_head] = r;
                m_head++; if (m_head == period) m_head = 0;
            }

            // recompute numerator from scratch over current ring contents
            int cur_len = m_count;
            int latest = (m_head - 1 + period) % period; // position of newest r
            double rw[NET_MYRSI_MAX_PERIOD];
            for (int kk = 0; kk < cur_len; ++kk) {
                int pos = (latest - kk + period) % period;
                rw[kk] = myr[pos];
            }
            int local = 0;
            for (int i2 = 1; i2 < cur_len; ++i2) {
                double vi = rw[i2];
                for (int k2 = 0; k2 < i2; ++k2) {
                    double vk = rw[k2];
                    double d2 = vi - vk;
                    if (d2 > 0.0) local -= 1; else if (d2 < 0.0) local += 1;
                }
            }
            col_out[i * cols] = (denom != 0.0f) ? (float)((double)local / (double)denom) : 0.0f;
            }
    }
}
