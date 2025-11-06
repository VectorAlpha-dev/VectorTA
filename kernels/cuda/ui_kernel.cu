// CUDA kernels for Ulcer Index (UI)
//
// UI[i] = sqrt( avg_{k=0..period-1}( ((price[j] - max_{window})/max_{window})^2 ) ) * |scalar|
// Warmup: write NaN for indices < first + (2*period - 2)
// Division-by-zero guard: if rolling max is ~0 or non-finite, mark drawdown invalid.
// Validity policy: emit value only when the last `period` drawdowns are all valid; else NaN.

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include "ds_float2.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// -----------------------------
// Helpers
// -----------------------------

// FP32 compensated summation (Neumaier)
static __forceinline__ __device__ void neumaier_add(float y, float &sum, float &c) {
    float t = sum + y;
    if (fabsf(sum) >= fabsf(y)) c += (sum - t) + y; else c += (y - t) + sum;
    sum = t;
}

// FP32 compensated summation (classic Kahan)
static __forceinline__ __device__ void kahan_add(float y, float &sum, float &c) {
    float y2 = y - c;
    float t  = sum + y2;
    c        = (t - sum) - y2;
    sum      = t;
}

// Shared-memory helpers for a small circular deque and ring buffers.
// We implement a monotonic deque of indices of length `period` to track the rolling max.

// Single-series kernel computing base UI with scalar=1.0.
extern "C" __global__ void ui_single_series_f32(
    const float* __restrict__ prices,
    int series_len,
    int first_valid,
    int period,
    float* __restrict__ out)
{
    if (series_len <= 0 || period <= 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return; // single worker

    // dynamic shared memory layout (must match Rust wrapper sizing):
    // [deq_idx:int p][D region: p*sizeof(double) bytes][valid:uint8 p]
    // For the single-series base we use the D region as a double ring (sq_ring[p]).
    extern __shared__ __align__(16) unsigned char shraw[];
    unsigned char* base = shraw;
    const int p = period;
    // deq indices
    int* deq_idx = reinterpret_cast<int*>(base);
    // align to 8 bytes as in wrapper
    size_t off = static_cast<size_t>(p) * sizeof(int);
    const size_t a = sizeof(double) - 1;
    off = (off + a) & ~a;
    // use the double-sized region for double sq_ring
    double* sq_ring = reinterpret_cast<double*>(base + off);
    // valid ring comes after the double region
    unsigned char* valid_ring = reinterpret_cast<unsigned char*>(base + off + static_cast<size_t>(p) * sizeof(double));

    const int fv = first_valid < 0 ? 0 : first_valid;
    const int warm_end = fv + (2 * p - 2);

    // init rings
    for (int i = 0; i < p; ++i) { sq_ring[i] = 0.0; valid_ring[i] = 0u; }
    // warmup NaNs
    const int warm_write = (warm_end < series_len) ? warm_end : series_len;
    for (int i = 0; i < warm_write; ++i) out[i] = CUDART_NAN_F;

    int head = 0, tail = 0, dsize = 0;
    int ring_idx = 0;
    double sum = 0.0; // FP64 accumulator for higher accuracy
    int count = 0;

    for (int i = fv; i < series_len; ++i) {
        const int start = (i + 1 >= p) ? (i + 1 - p) : 0;

        // expire stale from front
        while (dsize != 0 && deq_idx[head] < start) {
            head = (head + 1); if (head == p) head = 0; dsize--;
        }

        const float xi = prices[i];
        const bool xi_finite = isfinite(xi);
        if (xi_finite) {
            // pop from back while <= xi (reload values from global)
            while (dsize != 0) {
                int back = (tail == 0) ? (p - 1) : (tail - 1);
                const int j = deq_idx[back];
                const float xj = prices[j];
                if (xj <= xi) { tail = back; dsize--; }
                else break;
            }
            // push xi
            deq_idx[tail] = i;
            tail += 1; if (tail == p) tail = 0; dsize++;
        }

        // compute new squared drawdown
        unsigned char new_valid = 0u;
        float new_sq = 0.0f;
        if (i + 1 >= fv + p && dsize != 0) {
            const float m = prices[deq_idx[head]];
            if (xi_finite && isfinite(m) && fabsf(m) > 1e-20f) {
                const double dd = (static_cast<double>(xi) - static_cast<double>(m)) / static_cast<double>(m);
                new_sq = static_cast<float>(dd * dd);
                new_valid = 1u;
            }
        }

        // slide ring (compensated sum)
        if (new_valid)             { sum += (double)new_sq; count++; }
        if (valid_ring[ring_idx])  { sum -= sq_ring[ring_idx]; count--; }
        sq_ring[ring_idx] = (double)new_sq; valid_ring[ring_idx] = new_valid;
        ring_idx += 1; if (ring_idx == p) ring_idx = 0;

        // output
        if (i >= warm_end) {
            if (count == p) {
                double avg_d = sum / (double)p;
                if (avg_d < 0.0) avg_d = 0.0;
                out[i] = static_cast<float>(sqrt(avg_d));
            } else {
                out[i] = CUDART_NAN_F;
            }
        }
    }
}

// Expand base -> rows by scaling with |scalar|.
extern "C" __global__ void ui_scale_rows_from_base_f32(
    const float* __restrict__ base,
    const float* __restrict__ scalars,
    int series_len,
    int n_rows,
    float* __restrict__ out)
{
    const int row = blockIdx.y;
    if (row >= n_rows) return;
    const float s = fabsf(scalars[row]);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float* dst = out + row * series_len;
    for (int i = tid; i < series_len; i += stride) {
        const float v = base[i];
        dst[i] = static_cast<float>(static_cast<double>(v) * static_cast<double>(s));
    }
}

// Many-series × one-param (time-major): params are (period, scalar).
// prices_tm: [rows][cols] laid out as t*cols + s
extern "C" __global__ void ui_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float scalar,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x;               // one series per block
    if (s >= cols || rows <= 0 || period <= 0) return;
    if (threadIdx.x != 0) return;            // single worker per block

    // dynamic shared memory layout compatible with wrapper
    extern __shared__ __align__(16) unsigned char shraw[];
    unsigned char* base = shraw;
    const int p = period;
    int* deq_idx = reinterpret_cast<int*>(base);
    size_t off = static_cast<size_t>(p) * sizeof(int);
    const size_t a = sizeof(double) - 1;
    off = (off + a) & ~a;
    float* deq_val = reinterpret_cast<float*>(base + off);
    float* sq_ring = reinterpret_cast<float*>(base + off + static_cast<size_t>(p) * sizeof(float));
    unsigned char* valid_ring = reinterpret_cast<unsigned char*>(base + off + static_cast<size_t>(p) * sizeof(double));

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const int warm_end = fv + (2 * p - 2);
    for (int i = 0; i < p; ++i) { sq_ring[i] = 0.0f; valid_ring[i] = 0u; }
    for (int t = 0; t < rows && t < warm_end; ++t) { out_tm[t * cols + s] = CUDART_NAN_F; }

    int head = 0, tail = 0, dsize = 0;
    int ring_idx = 0;
    float sum = 0.0f, comp = 0.0f;
    int count = 0;
    const float s_abs = fabsf(scalar);

    for (int t = fv; t < rows; ++t) {
        const int start = (t + 1 >= p) ? (t + 1 - p) : 0;

        while (dsize != 0 && deq_idx[head] < start) {
            head = (head + 1); if (head == p) head = 0; dsize--;
        }
        const int idx = t * cols + s;
        const float xi = prices_tm[idx];
        const bool xi_finite = isfinite(xi);
        if (xi_finite) {
            while (dsize != 0) {
                int back = (tail == 0) ? (p - 1) : (tail - 1);
                const float xj = deq_val[back];
                if (xj <= xi) { tail = back; dsize--; } else { break; }
            }
            deq_idx[tail] = t; deq_val[tail] = xi;
            tail += 1; if (tail == p) tail = 0; dsize++;
        }

        unsigned char new_valid = 0u; float new_sq = 0.0f;
        if (t + 1 >= fv + p && dsize != 0) {
            const float m = deq_val[head];
            if (xi_finite && isfinite(m) && fabsf(m) > 1e-20f) {
                const float dd = (xi - m) / m;
                new_sq = dd * dd;
                new_valid = 1u;
            }
        }
        if (valid_ring[ring_idx]) { neumaier_add(-sq_ring[ring_idx], sum, comp); count--; }
        if (new_valid)             { neumaier_add( new_sq,               sum, comp); count++; }
        sq_ring[ring_idx] = new_sq; valid_ring[ring_idx] = new_valid;
        ring_idx += 1; if (ring_idx == p) ring_idx = 0;

        if (t >= warm_end) {
            if (count == p) {
                float avg = (sum + comp) / (float)p;
                if (avg < 0.0f) avg = 0.0f;
                out_tm[idx] = sqrtf(avg) * s_abs;
            } else {
                out_tm[idx] = CUDART_NAN_F;
            }
        }
    }
}

// NEW: one price series × many (period, scalar) parameters.
// Not used by the current Rust wrapper yet, but compiled and available.
extern "C" __global__ void ui_one_series_many_params_f32(
    const float* __restrict__ prices,
    int series_len,
    const int*  __restrict__ periods,   // [n_params]
    const float* __restrict__ scalars,  // [n_params]
    int n_params,
    int first_valid,
    int max_period,                     // host-provided max(periods)
    float* __restrict__ out)            // [n_params][series_len]
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    int param_id = blockIdx.x * warps_per_block + warp;
    if (param_id >= n_params) return;

    // Shared memory layout per block (compat with 8B alignment from host isn't required here):
    // [deq_idx:int warps*maxP][D region: warps*maxP*sizeof(double) bytes][valid:uchar warps*maxP]
    // We split each warp's D region into two FP32 arrays: deq_val[p] and sq_ring[p].
    extern __shared__ __align__(16) unsigned char shraw[];
    unsigned char* base = shraw;

    size_t stride_i = (size_t)warps_per_block * (size_t)max_period;
    int*   deq_idx_base = reinterpret_cast<int*>(base);
    size_t off = stride_i * sizeof(int);
    const size_t a = sizeof(double) - 1;
    off = (off + a) & ~a;
    float* deq_val_base = reinterpret_cast<float*>(base + off);
    float* sq_ring_base = reinterpret_cast<float*>(base + off + stride_i * sizeof(float));
    unsigned char* valid_base = reinterpret_cast<unsigned char*>(base + off + stride_i * sizeof(double));

    // Pointers for this warp (parameter)
    int*   deq_idx = deq_idx_base + warp * max_period;
    float* deq_val = deq_val_base + warp * max_period;
    float* sq_ring = sq_ring_base + warp * max_period;
    unsigned char* valid_ring = valid_base + warp * max_period;

    const int p = periods[param_id];
    if (p <= 0 || p > max_period) return;
    const int fv = first_valid < 0 ? 0 : first_valid;
    const int warm_end = fv + (2 * p - 2);

    // Parallel init within warp
    for (int k = lane; k < p; k += WARP_SIZE) { sq_ring[k] = 0.0f; valid_ring[k] = 0u; }
    // Warmup NaNs for this param
    float* out_row = out + (size_t)param_id * (size_t)series_len;
    for (int i = lane; i < series_len && i < warm_end; i += WARP_SIZE) { out_row[i] = CUDART_NAN_F; }
    __syncwarp();

    // Lane 0 runs the sequential scan for this parameter
    if (lane == 0) {
        int head = 0, tail = 0, dsize = 0;
        int ring_idx = 0;
        float sum = 0.0f, comp = 0.0f;
        int count = 0;

        for (int i = fv; i < series_len; ++i) {
            const int start = (i + 1 >= p) ? (i + 1 - p) : 0;
            while (dsize != 0 && deq_idx[head] < start) { head = (head + 1); if (head == p) head = 0; dsize--; }

            const float xi = prices[i];
            const bool xi_finite = isfinite(xi);
            if (xi_finite) {
                while (dsize != 0) {
                    int back = (tail == 0) ? (p - 1) : (tail - 1);
                    const float xj = deq_val[back];
                    if (xj <= xi) { tail = back; dsize--; } else break;
                }
                deq_idx[tail] = i; deq_val[tail] = xi;
                tail += 1; if (tail == p) tail = 0; dsize++;
            }

            unsigned char new_valid = 0u; float new_sq = 0.0f;
            if (i + 1 >= fv + p && dsize != 0) {
                const float m = deq_val[head];
                if (xi_finite && isfinite(m) && fabsf(m) > 1e-20f) {
                    const float dd = (xi - m) / m;
                    new_sq = dd * dd; new_valid = 1u;
                }
            }

            if (valid_ring[ring_idx]) { neumaier_add(-sq_ring[ring_idx], sum, comp); count--; }
            if (new_valid)             { neumaier_add( new_sq,               sum, comp); count++; }
            sq_ring[ring_idx] = new_sq; valid_ring[ring_idx] = new_valid;
            ring_idx += 1; if (ring_idx == p) ring_idx = 0;

            if (i >= warm_end) {
                if (count == p) {
                    float avg = sum / (float)p;
                    if (avg < 0.0f) avg = 0.0f;
                    out_row[i] = sqrtf(avg) * fabsf(scalars[param_id]);
                } else {
                    out_row[i] = CUDART_NAN_F;
                }
            }
        }
    }
}
