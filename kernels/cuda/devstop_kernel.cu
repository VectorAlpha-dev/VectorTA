// CUDA kernels for DevStop (Deviation Stop) indicator
//
// Batch (one series × many params):
//   - devstop_batch_grouped_f32
//       Each block handles one parameter combo with a fixed period (grouped by the host).
//       Threads in the block first initialize warmup outputs to NaN, then thread 0 runs a
//       sequential scan using prefix sums over the two-bar range r to compute the base value
//       and a monotonic deque to emit the final rolling extrema over `base`.
//       Semantics match the scalar CPU implementation:
//         warm = first_valid + 2*period - 1; outputs < warm are NaN.
//         For long direction: base = H - mean(r) - mult * std(r); final = rolling max(base).
//         For short direction: base = L + mean(r) + mult * std(r); final = rolling min(base).
//       We only support devtype=0 (standard deviation) and SMA for the range mean (by construction
//       of the prefix sums). EMA variants are left to the scalar/GPU many‑series path.
//
// Many-series × one param (time-major):
//   - devstop_many_series_one_param_f32
//       One block per series; thread 0 performs a sequential scan while other threads help
//       initialize the warmup NaNs. Uses a ring for r, and maintains Σr/Σr² to compute mean/std.
//       Uses a monotonic deque over `base` to produce the final extrema, matching CPU policy.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }

// -------------------- Batch: one series × many params (grouped by period) --------------------
// Dynamic shared memory layout per block:
//   float base_ring[period]
//   int   dq_idx   [period]
extern "C" __global__ void devstop_batch_grouped_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const double* __restrict__ p1,   // prefix sum of r
    const double* __restrict__ p2,   // prefix sum of r^2
    const int* __restrict__ pc,      // prefix count of finite r
    int len,
    int first_valid,
    int period,
    const float* __restrict__ mults,
    int n_combos,
    int is_long,            // 1 = long, 0 = short
    int out_row_base,       // global row base for this period group
    float* __restrict__ out // [rows=len*n_combos_total]
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || period <= 0) return;

    const int warm = first_valid + 2 * period - 1;
    const int row = out_row_base + combo;
    const int row_off = row * len;

    // Initialize entire row to NaN in parallel
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        out[row_off + t] = qnan32();
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single lane performs the sequential scan

    if (warm >= len) return;

    extern __shared__ unsigned char smem[];
    float* base_ring = reinterpret_cast<float*>(smem);
    int* dq_idx = reinterpret_cast<int*>(base_ring + period);
    for (int i = 0; i < period; ++i) { base_ring[i] = qnan32(); dq_idx[i] = 0; }

    int dq_head = 0, dq_len = 0;
    const int cap = period;

    // Helpers (match CPU policy)
    auto dq_back_at = [&](int len) {
        int pos = (dq_head + len - 1) % cap; return dq_idx[pos];
    };
    auto dq_push_back = [&](int value) {
        int pos = (dq_head + dq_len) % cap; dq_idx[pos] = value; dq_len += 1;
    };
    auto dq_pop_back = [&]() { dq_len -= 1; };
    auto dq_pop_front = [&]() { dq_head = (dq_head + 1) % cap; dq_len -= 1; };
    auto dq_front = [&]() { return dq_idx[dq_head]; };

    const float mult = mults[combo];
    const int start_base = first_valid + period;           // first index where base is defined
    const int start_final = start_base + period - 1;       // first valid output index

    for (int i = start_base; i < len; ++i) {
        // Compute mean and std over window [i-period+1, i]
        const int t1 = i + 1;
        int a = t1 - period; if (a < 0) a = 0; // clamp (matches scalar prefix handling)
        const int cnt = pc[t1] - pc[a];
        float base = qnan32();
        if (cnt > 0) {
            const double S1 = p1[t1] - p1[a];
            const double S2 = p2[t1] - p2[a];
            const double inv = 1.0 / static_cast<double>(cnt);
            const double mean = S1 * inv;                 // E[r]
            double var = (S2 * inv) - (mean * mean);      // E[r^2] - (E[r])^2
            if (var < 0.0) var = 0.0;                     // numeric safety
            const double sigma = sqrt(var);
            const float h = high[i];
            const float l = low[i];
            if (is_long) {
                if (!isnan(h) && !isnan((float)mean) && !isnan((float)sigma)) {
                    base = h - (float)mean - mult * (float)sigma;
                }
            } else {
                if (!isnan(l) && !isnan((float)mean) && !isnan((float)sigma)) {
                    base = l + (float)mean + mult * (float)sigma;
                }
            }
        }

        // Update deque over base using a ring
        const int slot = i % period;
        base_ring[slot] = base;
        if (is_long) {
            // Long: pop while bj <= base (decreasing deque -> front holds max)
            while (dq_len > 0) {
                int j = dq_back_at(dq_len);
                float bj = base_ring[j % period];
                if (isnan(bj) || bj <= base) dq_pop_back(); else break;
            }
        } else {
            // Short: pop while bj >= base (increasing deque -> front holds min)
            while (dq_len > 0) {
                int j = dq_back_at(dq_len);
                float bj = base_ring[j % period];
                if (isnan(bj) || bj >= base) dq_pop_back(); else break;
            }
        }
        dq_push_back(i);

        // Expire old indices
        const int cut = i + 1 - period;
        while (dq_len > 0 && dq_front() < cut) { dq_pop_front(); }

        if (i >= start_final) {
            float out_val = qnan32();
            if (dq_len > 0) {
                int j = dq_front();
                out_val = base_ring[j % period];
            }
            out[row_off + i] = out_val;
        }
    }
}

// -------------------- Many series × one param (time-major) --------------------
// Dynamic shared memory per block:
//   float r_ring   [period]
//   float base_ring[period]
//   int   dq_idx   [period]
extern "C" __global__ void devstop_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int* __restrict__ first_valids, // per series (column)
    int cols,
    int rows,
    int period,
    float mult,
    int is_long,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x; // series index (column)
    if (s >= cols || period <= 0) return;

    // Initialize outputs to NaN in parallel
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        out_tm[t * cols + s] = qnan32();
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single lane per series

    const int fv = first_valids[s];
    const int start_base = fv + period;
    const int start_final = start_base + period - 1;
    if (start_base >= rows) return;

    extern __shared__ unsigned char smem_uc[];
    float* r_ring = reinterpret_cast<float*>(smem_uc);
    float* base_ring = r_ring + period;
    int* dq_idx = reinterpret_cast<int*>(base_ring + period);
    for (int i = 0; i < period; ++i) { r_ring[i] = qnan32(); base_ring[i] = qnan32(); dq_idx[i] = 0; }

    int r_pos = 0; int r_inserted = 0; double sum = 0.0, sum2 = 0.0; int cnt = 0;
    float prev_h = high_tm[fv * cols + s];
    float prev_l = low_tm [fv * cols + s];

    // Prefill r over (fv+1 .. start_base-1)
    for (int k = fv + 1; k < min(start_base, rows); ++k) {
        const float h = high_tm[k * cols + s];
        const float l = low_tm [k * cols + s];
        float r = qnan32();
        if (!isnan(h) && !isnan(l) && !isnan(prev_h) && !isnan(prev_l)) {
            const float hi2 = (h > prev_h) ? h : prev_h;
            const float lo2 = (l < prev_l) ? l : prev_l;
            r = hi2 - lo2;
        }
        r_ring[r_pos] = r; r_pos = (r_pos + 1) % period; r_inserted += 1;
        if (!isnan(r)) { double rd = (double)r; sum += rd; sum2 += rd * rd; cnt += 1; }
        prev_h = h; prev_l = l;
    }
    r_pos = (period - 1) % period;

    int dq_head = 0, dq_len = 0; const int cap = period;
    auto dq_back_at = [&](int len) { int pos = (dq_head + len - 1) % cap; return dq_idx[pos]; };
    auto dq_push_back = [&](int value) { int pos = (dq_head + dq_len) % cap; dq_idx[pos] = value; dq_len += 1; };
    auto dq_pop_back = [&]() { dq_len -= 1; };
    auto dq_pop_front = [&]() { dq_head = (dq_head + 1) % cap; dq_len -= 1; };
    auto dq_front = [&]() { return dq_idx[dq_head]; };

    for (int i = start_base; i < rows; ++i) {
        const float h = high_tm[i * cols + s];
        const float l = low_tm [i * cols + s];

        float r_new = qnan32();
        if (!isnan(h) && !isnan(l) && !isnan(prev_h) && !isnan(prev_l)) {
            const float hi2 = (h > prev_h) ? h : prev_h;
            const float lo2 = (l < prev_l) ? l : prev_l;
            r_new = hi2 - lo2;
        }
        prev_h = h; prev_l = l;

        const bool had_full = (r_inserted >= period);
        const float old = had_full ? r_ring[r_pos] : qnan32();
        if (had_full && !isnan(old)) { double od = (double)old; sum -= od; sum2 -= od * od; cnt -= 1; }
        r_ring[r_pos] = r_new; r_pos = (r_pos + 1) % period; r_inserted += 1;
        if (!isnan(r_new)) { double rd = (double)r_new; sum += rd; sum2 += rd * rd; cnt += 1; }

        float base = qnan32();
        if (cnt > 0) {
            const double inv = 1.0 / (double)cnt;
            const double mean = sum * inv;
            double var = (sum2 * inv) - (mean * mean);
            if (var < 0.0) var = 0.0;
            const float sigma = (float)sqrt(var);
            if (is_long) {
                if (!isnan(h)) base = h - (float)mean - mult * sigma;
            } else {
                if (!isnan(l)) base = l + (float)mean + mult * sigma;
            }
        }

        const int slot = i % period;
        base_ring[slot] = base;
        if (is_long) {
            while (dq_len > 0) {
                int j = dq_back_at(dq_len);
                float bj = base_ring[j % period];
                if (isnan(bj) || bj <= base) dq_pop_back(); else break;
            }
        } else {
            while (dq_len > 0) {
                int j = dq_back_at(dq_len);
                float bj = base_ring[j % period];
                if (isnan(bj) || bj >= base) dq_pop_back(); else break;
            }
        }
        dq_push_back(i);

        const int cut = i + 1 - period;
        while (dq_len > 0 && dq_front() < cut) dq_pop_front();

        if (i >= start_final) {
            float out_val = qnan32();
            if (dq_len > 0) { int j = dq_front(); out_val = base_ring[j % period]; }
            out_tm[i * cols + s] = out_val;
        }
    }
}

