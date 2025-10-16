// CUDA kernels for Relative Volatility Index (RVI)
//
// Semantics mirror src/indicators/rvi.rs (scalar path):
// - Output length equals input length (row-major for batch).
// - Warmup prefix: NaN for indices < warm = first_valid + (period-1) + (ma_len-1).
// - Two smoothing modes: SMA (matype=0) and EMA with SMA seed (matype=1).
// - Deviation types supported: 0=StdDev (rolling O(1) with valid-flag ring),
//   1=MeanAbsDev (MAD; ring + full abs-sum each step). Devtype=2 (median-abs-dev)
//   is not implemented in this initial GPU kernel and will behave like MAD if
//   passed inadvertently (wrapper prevents such launches).
// - FP32 outputs; internal accumulations use double where beneficial.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef RVI_BLOCK_X
#define RVI_BLOCK_X 256
#endif

// Helpers for SMA/EMA smoothing (NaN-aware), matching scalar semantics closely.
static __device__ __forceinline__ float smooth_sma_push(
    float x,
    float* __restrict__ ring,
    int* __restrict__ head,
    int* __restrict__ count,
    int ma_len,
    double* __restrict__ sum,
    double inv_m)
{
    if (!isfinite(x)) { // reset on NaN input
        *sum = 0.0;
        *count = 0;
        *head = 0;
        return NAN;
    }
    if (*count < ma_len) {
        ring[*head] = x;
        *sum += (double)x;
        *head = (*head + 1 == ma_len) ? 0 : *head + 1;
        (*count)++;
        if (*count == ma_len) {
            return (float)((*sum) * inv_m);
        } else {
            return NAN;
        }
    } else {
        const float old = ring[*head];
        ring[*head] = x;
        *head = (*head + 1 == ma_len) ? 0 : *head + 1;
        *sum += (double)x - (double)old;
        return (float)((*sum) * inv_m);
    }
}

static __device__ __forceinline__ float smooth_ema_push(
    float x,
    bool* __restrict__ started,
    double* __restrict__ seed_sum,
    int* __restrict__ seed_cnt,
    int ma_len,
    double inv_m,
    double alpha,
    double one_m_alpha,
    double* __restrict__ prev)
{
    if (!isfinite(x)) {
        // Reset EMA state on gaps
        *started = false;
        *seed_sum = 0.0;
        *seed_cnt = 0;
        return NAN;
    }
    if (!*started) {
        *seed_sum += (double)x;
        (*seed_cnt)++;
        if (*seed_cnt == ma_len) {
            *prev = (*seed_sum) * inv_m;
            *started = true;
            return (float)(*prev);
        }
        return NAN;
    } else {
        *prev = fma(one_m_alpha, *prev, alpha * (double)x);
        return (float)(*prev);
    }
}

// Compute one RVI row (single series) sequentially. Uses dynamic shared memory layout:
// [ up_ring (max_ma_len floats) | dn_ring (max_ma_len floats) | dev_ring (max_period floats) | vflag (max_period bytes) ]
static __device__ __forceinline__ void rvi_compute_series(
    const float* __restrict__ prices,
    int len,
    int first_valid,
    int period,
    int ma_len,
    int matype,
    int devtype,
    int max_period,
    int max_ma_len,
    float* __restrict__ out)
{
    if (len <= 0 || first_valid >= len || period <= 0 || ma_len <= 0) return;

    extern __shared__ unsigned char shraw[];
    float* up_ring = reinterpret_cast<float*>(shraw);
    float* dn_ring = up_ring + max_ma_len;
    float* dev_ring = dn_ring + max_ma_len;        // used for MAD
    unsigned char* vflag = reinterpret_cast<unsigned char*>(dev_ring + max_period); // for StdDev

    const int warm = first_valid + (period - 1) + (ma_len - 1);

    // Deviation state
    double sum = 0.0, sumsq = 0.0; // for stddev
    int head = 0, filled = 0;      // for MAD ring / stddev flags
    double ring_sum = 0.0;         // for MAD mean

    // Smoothing state
    const double inv_m = 1.0 / (double)ma_len;
    const double alpha = 2.0 / ((double)ma_len + 1.0);
    const double one_m_alpha = 1.0 - alpha;
    bool up_started = false, dn_started = false;
    double up_seed_sum = 0.0, dn_seed_sum = 0.0;
    int up_seed_cnt = 0, dn_seed_cnt = 0;
    int up_h = 0, dn_h = 0, up_cnt = 0, dn_cnt = 0;
    double up_sum = 0.0, dn_sum = 0.0; // for SMA smoothing
    double up_prev = 0.0, dn_prev = 0.0; // EMA prevs

    float prev = prices[0];

    // Initialize MAD path bookkeeping
    if (devtype != 0) {
        // MAD: ring initializes empty
        filled = 0;
        head = 0;
        ring_sum = 0.0;
    }

    for (int i = 0; i < len; ++i) {
        const float x = prices[i];
        float d;
        if (i == 0 || !isfinite(x) || !isfinite(prev)) d = NAN; else d = x - prev;
        prev = x;

        float dev;
        if (i + 1 < period) {
            dev = NAN;
        } else if (devtype == 0) {
            // StdDev with rebuild-on-NaN (matches scalar general path)
            if (i == period - 1) {
                sum = 0.0; sumsq = 0.0;
                bool ok = true;
                for (int k = 0; k < period; ++k) {
                    const float v = prices[k];
                    if (!isfinite(v)) { ok = false; break; }
                    sum += (double)v;
                    sumsq += (double)v * (double)v;
                }
                if (ok) {
                    const double mean = sum / (double)period;
                    const double mean_sq = sumsq / (double)period;
                    dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                } else {
                    dev = NAN;
                }
            } else {
                const float leaving = prices[i - period];
                if (!isfinite(leaving) || !isfinite(x)) {
                    sum = 0.0; sumsq = 0.0;
                    bool ok = true;
                    for (int k = i - period + 1; k <= i; ++k) {
                        const float v = prices[k];
                        if (!isfinite(v)) { ok = false; break; }
                        sum += (double)v;
                        sumsq += (double)v * (double)v;
                    }
                    if (ok) {
                        const double mean = sum / (double)period;
                        const double mean_sq = sumsq / (double)period;
                        dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                    } else {
                        dev = NAN;
                    }
                } else {
                    sum += (double)x - (double)leaving;
                    sumsq += (double)x * (double)x - (double)leaving * (double)leaving;
                    const double mean = sum / (double)period;
                    const double mean_sq = sumsq / (double)period;
                    dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                }
            }
        } else { // MAD (devtype==1) or fallback for unsupported
            if (!isfinite(x)) {
                // reset ring on gap
                filled = 0;
                head = 0;
                ring_sum = 0.0;
                dev = NAN;
            } else if (filled < period) {
                dev_ring[head] = x;
                ring_sum += (double)x;
                head = (head + 1 == period) ? 0 : head + 1;
                filled += 1;
                dev = (filled == period) ? 0.0f : NAN; // first valid will recompute below
                if (filled == period) {
                    // compute mean and MAD for the first time
                    const double mean = ring_sum / (double)period;
                    double abs_sum = 0.0;
                    for (int k = 0; k < period; ++k) {
                        abs_sum += fabs((double)dev_ring[k] - mean);
                    }
                    dev = (float)(abs_sum / (double)period);
                }
            } else {
                // steady-state: ring full
                const float old = dev_ring[head];
                dev_ring[head] = x;
                head = (head + 1 == period) ? 0 : head + 1;
                ring_sum += (double)x - (double)old;
                const double mean = ring_sum / (double)period;
                double abs_sum = 0.0;
                for (int k = 0; k < period; ++k) {
                    abs_sum += fabs((double)dev_ring[k] - mean);
                }
                dev = (float)(abs_sum / (double)period);
            }
        }

        float up_i, dn_i;
        if (!isfinite(d) || !isfinite(dev)) {
            up_i = NAN; dn_i = NAN;
        } else if (d > 0.0f) {
            up_i = dev; dn_i = 0.0f;
        } else if (d < 0.0f) {
            up_i = 0.0f; dn_i = dev;
        } else { up_i = 0.0f; dn_i = 0.0f; }

        float up_s, dn_s;
        if (matype == 0) {
            up_s = smooth_sma_push(up_i, up_ring, &up_h, &up_cnt, ma_len, &up_sum, inv_m);
            dn_s = smooth_sma_push(dn_i, dn_ring, &dn_h, &dn_cnt, ma_len, &dn_sum, inv_m);
        } else {
            up_s = smooth_ema_push(up_i, &up_started, &up_seed_sum, &up_seed_cnt, ma_len, inv_m, alpha, one_m_alpha, &up_prev);
            dn_s = smooth_ema_push(dn_i, &dn_started, &dn_seed_sum, &dn_seed_cnt, ma_len, inv_m, alpha, one_m_alpha, &dn_prev);
        }

        if (i >= warm) {
            if (!isfinite(up_s) || !isfinite(dn_s)) {
                out[i] = NAN;
            } else {
                const double denom_d = (double)up_s + (double)dn_s;
                out[i] = (fabs(denom_d) <= 1e-15) ? NAN : (100.0f * (up_s / (float)denom_d));
            }
        }
    }
}

// --------------------------- Batch kernel ---------------------------
extern "C" __global__
void rvi_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   const int* __restrict__ ma_lens,
                   const int* __restrict__ matypes,
                   const int* __restrict__ devtypes,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   int max_period,
                   int max_ma_len,
                   float* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= n_combos) return;

    const int period = periods[row];
    const int ma_len = ma_lens[row];
    const int matype = matypes[row];
    const int devtype = devtypes[row];
    if (period <= 0 || ma_len <= 0) return;

    const int base = row * series_len;

    // Warmup NaNs
    int warm = first_valid + (period - 1) + (ma_len - 1);
    if (warm > series_len) warm = series_len;
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single thread scans sequentially
    rvi_compute_series(prices, series_len, first_valid, period, ma_len, matype, devtype, max_period, max_ma_len, out + base);
}

// -------------------- Many-series × one param ----------------------
extern "C" __global__
void rvi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   int ma_len,
                                   int matype,
                                   int devtype,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    if (period <= 0 || ma_len <= 0) return;
    const int first = first_valids[s];

    // Warmup prefix per column
    int warm = first + (period - 1) + (ma_len - 1);
    if (warm > rows) warm = rows;
    for (int t = 0; t < warm; ++t) {
        out_tm[t * cols + s] = NAN;
    }
    if (warm >= rows) return;

    // Build contiguous view of this series (sequential walk)
    // We avoid extra shared memory by indexing time-major arrays directly.
    // Use the same sequential engine but mapping indices.
    // Local smoothing and deviation states are allocated on the stack via rvi_compute_series

    // To reuse rvi_compute_series, we expose a small wrapper that reads prices from time-major layout.
    // Copy the column into a temporary ring buffer would cost O(rows) memory; instead, call the same
    // logic inline below for clarity.

    // Re-implement minimal sequential loop for time-major layout (mirrors rvi_compute_series but without smem):
    // Since period and ma_len are identical for all series in this kernel, we can place small fixed-size rings
    // on the stack for smoothing (reasonable as ma_len defaults are small). For safety with large ma_len, we
    // degrade to EMA-only when ma_len exceeds 1024 to avoid large stack usage.

    const bool use_sma = (matype == 0) && (ma_len <= 1024);
    // Smoothing state
    const double inv_m = 1.0 / (double)ma_len;
    const double alpha = 2.0 / ((double)ma_len + 1.0);
    const double one_m_alpha = 1.0 - alpha;
    bool up_started = false, dn_started = false;
    double up_seed_sum = 0.0, dn_seed_sum = 0.0;
    int up_seed_cnt = 0, dn_seed_cnt = 0;
    int up_h = 0, dn_h = 0, up_cnt = 0, dn_cnt = 0;
    double up_sum = 0.0, dn_sum = 0.0; // for SMA
    double up_prev = 0.0, dn_prev = 0.0; // EMA prevs

    // Deviation state
    double sum = 0.0, sumsq = 0.0;
    int valid = 0, head = 0, filled = 0;
    double ring_sum = 0.0;

    float prev = prices_tm[0 * cols + s];

    // Local rings (bounded). If ma_len > 1024 we will use EMA only and ignore SMA path.
    float up_ring_local[ (1024) ];
    float dn_ring_local[ (1024) ];
    float* up_ring = up_ring_local;
    float* dn_ring = dn_ring_local;
    // MAD ring allocated in local memory up to a sane bound; else recompute by scanning directly from TM buffer
    const bool mad_local_ok = (period <= 2048);
    float dev_ring_local[ (2048) ];

    // MAD bookkeeping init
    if (devtype != 0) {
        filled = 0; head = 0; ring_sum = 0.0;
    }

    for (int i = 0; i < rows; ++i) {
        const float x = prices_tm[i * cols + s];
        float d;
        if (i == 0 || !isfinite(x) || !isfinite(prev)) d = NAN; else d = x - prev;
        prev = x;

        float dev;
        if (i + 1 < period) {
            dev = NAN;
        } else if (devtype == 0) {
            // StdDev with rebuild-on-NaN
            if (i == period - 1) {
                sum = 0.0; sumsq = 0.0; bool ok = true;
                for (int k = 0; k < period; ++k) {
                    const float v = prices_tm[k * cols + s];
                    if (!isfinite(v)) { ok = false; break; }
                    sum += (double)v; sumsq += (double)v * (double)v;
                }
                if (ok) {
                    const double mean = sum / (double)period;
                    const double mean_sq = sumsq / (double)period;
                    dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                } else { dev = NAN; }
            } else {
                const float leaving = prices_tm[(i - period) * cols + s];
                if (!isfinite(leaving) || !isfinite(x)) {
                    sum = 0.0; sumsq = 0.0; bool ok = true;
                    for (int k = i - period + 1; k <= i; ++k) {
                        const float v = prices_tm[k * cols + s];
                        if (!isfinite(v)) { ok = false; break; }
                        sum += (double)v; sumsq += (double)v * (double)v;
                    }
                    if (ok) {
                        const double mean = sum / (double)period;
                        const double mean_sq = sumsq / (double)period;
                        dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                    } else { dev = NAN; }
                } else {
                    sum += (double)x - (double)leaving;
                    sumsq += (double)x * (double)x - (double)leaving * (double)leaving;
                    const double mean = sum / (double)period;
                    const double mean_sq = sumsq / (double)period;
                    dev = (float)sqrt(fmax(0.0, mean_sq - mean * mean));
                }
            }
        } else { // MAD
            if (!isfinite(x)) { filled = 0; head = 0; ring_sum = 0.0; dev = NAN; }
            else if (filled < period) {
                if (mad_local_ok) dev_ring_local[head] = x;
                ring_sum += (double)x; head = (head + 1 == period) ? 0 : head + 1; filled++;
                if (filled == period) {
                    const double mean = ring_sum / (double)period; double abs_sum = 0.0;
                    if (mad_local_ok) { for (int k = 0; k < period; ++k) abs_sum += fabs((double)dev_ring_local[k] - mean); }
                    else { for (int k = i - period + 1; k <= i; ++k) { float v = prices_tm[k * cols + s]; abs_sum += fabs((double)v - mean); } }
                    dev = (float)(abs_sum / (double)period);
                } else dev = NAN;
            } else {
                const float old = mad_local_ok ? dev_ring_local[head] : prices_tm[(i - period) * cols + s];
                if (mad_local_ok) dev_ring_local[head] = x;
                head = (head + 1 == period) ? 0 : head + 1; ring_sum += (double)x - (double)old;
                const double mean = ring_sum / (double)period; double abs_sum = 0.0;
                if (mad_local_ok) { for (int k = 0; k < period; ++k) abs_sum += fabs((double)dev_ring_local[k] - mean); }
                else { for (int k = i - period + 1; k <= i; ++k) { float v = prices_tm[k * cols + s]; abs_sum += fabs((double)v - mean); } }
                dev = (float)(abs_sum / (double)period);
            }
        }

        float up_i, dn_i;
        if (!isfinite(d) || !isfinite(dev)) { up_i = NAN; dn_i = NAN; }
        else if (d > 0.0f) { up_i = dev; dn_i = 0.0f; }
        else if (d < 0.0f) { up_i = 0.0f; dn_i = dev; }
        else { up_i = 0.0f; dn_i = 0.0f; }

        float up_s, dn_s;
        if (use_sma) {
            up_s = smooth_sma_push(up_i, up_ring, &up_h, &up_cnt, ma_len, &up_sum, inv_m);
            dn_s = smooth_sma_push(dn_i, dn_ring, &dn_h, &dn_cnt, ma_len, &dn_sum, inv_m);
        } else {
            up_s = smooth_ema_push(up_i, &up_started, &up_seed_sum, &up_seed_cnt, ma_len, inv_m, alpha, one_m_alpha, &up_prev);
            dn_s = smooth_ema_push(dn_i, &dn_started, &dn_seed_sum, &dn_seed_cnt, ma_len, inv_m, alpha, one_m_alpha, &dn_prev);
        }

        if (i >= warm) {
            if (!isfinite(up_s) || !isfinite(dn_s)) out_tm[i * cols + s] = NAN;
            else {
                const double denom_d = (double)up_s + (double)dn_s;
                out_tm[i * cols + s] = (fabs(denom_d) <= 1e-15) ? NAN : (100.0f * (up_s / (float)denom_d));
            }
        }
    }
}
