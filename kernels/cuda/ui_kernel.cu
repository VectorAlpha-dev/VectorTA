// CUDA kernels for Ulcer Index (UI)
//
// UI[i] = sqrt( avg_{k=0..period-1}( ((price[j] - max_{window})/max_{window})^2 ) ) * |scalar|
// Warmup: write NaN for indices < first + (2*period - 2)
// Division-by-zero guard: if rolling max is ~0 or non-finite, mark drawdown invalid.
// Validity policy: emit value only when the last `period` drawdowns are all valid; else NaN.

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

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
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if (bid != 0) return; // single block is sufficient; we use only tid 0 for scan

    // dynamic shared memory layout: [deque indices][sq_ring][valid_ring]
    extern __shared__ __align__(16) unsigned char shraw[];
    // Layout with alignment padding to keep double aligned
    unsigned char* base = shraw;
    int* deq = reinterpret_cast<int*>(base);
    size_t off = static_cast<size_t>(period) * sizeof(int);
    const size_t a = sizeof(double) - 1;
    off = (off + a) & ~a; // round up to next multiple of sizeof(double)
    double* sq_ring = reinterpret_cast<double*>(base + off);
    unsigned char* valid_ring = reinterpret_cast<unsigned char*>(
        reinterpret_cast<unsigned char*>(sq_ring) + static_cast<size_t>(period) * sizeof(double));

    // Only a single thread performs the sequential scan due to dependencies.
    if (tid == 0) {
        const int p = period;
        const int cap = p;
        const int fv = first_valid < 0 ? 0 : first_valid;
        const int warm_end = fv + (2 * p - 2);

        // Initialize rings
        for (int i = 0; i < p; ++i) {
            sq_ring[i] = 0.0;
            valid_ring[i] = 0u;
        }
        // Write NaN prefix up to warmup end (clamped to series_len)
        const int warm_write = warm_end < series_len ? warm_end : series_len;
        for (int i = 0; i < warm_write; ++i) {
            out[i] = CUDART_NAN_F;
        }

        int head = 0; // front index in circular deque
        int tail = 0; // next write position (one past back)
        int dsize = 0; // number of valid entries in deque
        int ring_idx = 0;
        double sum = 0.0; // accumulate in FP64
        int count = 0;

        for (int i = fv; i < series_len; ++i) {
            // Window start
            const int start = (i + 1 >= p) ? (i + 1 - p) : 0;
            // Expire stale indices from front
            while (dsize != 0) {
                const int j = deq[head];
                if (j < start) {
                    head = head + 1; if (head == cap) head = 0; dsize -= 1;
                } else {
                    break;
                }
            }

            const float xi = prices[i];
            const bool xi_finite = isfinite(xi);
            if (xi_finite) {
                // Maintain monotonic deque in descending values
                while (dsize != 0) {
                    int back = tail == 0 ? (cap - 1) : (tail - 1);
                    const int j = deq[back];
                    const float xj = prices[j];
                    if (xj <= xi) {
                        tail = back; dsize -= 1;
                    } else {
                        break;
                    }
                }
                deq[tail] = i; tail += 1; if (tail == cap) tail = 0; dsize += 1;
            }

            // Compute squared drawdown when first rolling max is available
            unsigned char new_valid = 0u;
            double new_sq = 0.0;
            if (i + 1 >= fv + p && dsize != 0) {
                const int jmax = deq[head];
                const float m = prices[jmax];
                if (xi_finite && isfinite(m) && fabsf(m) > 1e-20f) {
                    const double diff = static_cast<double>(xi) - static_cast<double>(m);
                    const double dd = diff / static_cast<double>(m);
                    const double sq = dd * dd; // scalar=1.0 here
                    new_sq = sq;
                    new_valid = 1u;
                }
            }

            // Slide ring: drop old, add new
            if (valid_ring[ring_idx]) { sum -= sq_ring[ring_idx]; count -= 1; }
            if (new_valid) { sum += new_sq; count += 1; }
            sq_ring[ring_idx] = new_sq; valid_ring[ring_idx] = new_valid;
            ring_idx += 1; if (ring_idx == p) ring_idx = 0;

            // Emit after warmup end
            if (i >= warm_end) {
                if (count == p) {
                    double avg = sum / static_cast<double>(p);
                    if (avg < 0.0) avg = 0.0; // clamp tiny negatives
                    out[i] = static_cast<float>(sqrt(avg));
                } else {
                    out[i] = CUDART_NAN_F;
                }
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
        dst[i] = v * s; // NaN*scalar stays NaN
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
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || rows <= 0 || period <= 0) return;

    // dynamic shared memory for this block (single series): deq + sq_ring + valid ring
    extern __shared__ __align__(16) unsigned char shraw[];
    unsigned char* base = shraw;
    int* deq = reinterpret_cast<int*>(base);
    size_t off = static_cast<size_t>(period) * sizeof(int);
    const size_t a = sizeof(double) - 1;
    off = (off + a) & ~a;
    double* sq_ring = reinterpret_cast<double*>(base + off);
    unsigned char* valid_ring = reinterpret_cast<unsigned char*>(
        reinterpret_cast<unsigned char*>(sq_ring) + static_cast<size_t>(period) * sizeof(double));

    const int p = period;
    const int cap = p;
    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const int warm_end = fv + (2 * p - 2);
    for (int i = 0; i < p; ++i) { sq_ring[i] = 0.0; valid_ring[i] = 0u; }
    for (int t = 0; t < rows && t < warm_end; ++t) { out_tm[t * cols + s] = CUDART_NAN_F; }

    int head = 0, tail = 0, dsize = 0;
    int ring_idx = 0;
    double sum = 0.0;
    int count = 0;
    const float s_abs = fabsf(scalar);

    for (int t = fv; t < rows; ++t) {
        const int start = (t + 1 >= p) ? (t + 1 - p) : 0;
        // Expire stale
        while (dsize != 0) {
            const int j = deq[head];
            if (j < start) { head = head + 1; if (head == cap) head = 0; dsize -= 1; } else { break; }
        }
        const int idx = t * cols + s;
        const float xi = prices_tm[idx];
        const bool xi_finite = isfinite(xi);
        if (xi_finite) {
            while (dsize != 0) {
                int back = (tail == 0) ? (cap - 1) : (tail - 1);
                const int j = deq[back];
                const float xj = prices_tm[j * cols + s];
                if (xj <= xi) { tail = back; dsize -= 1; } else { break; }
            }
            deq[tail] = t; tail += 1; if (tail == cap) tail = 0; dsize += 1;
        }

        unsigned char new_valid = 0u; double new_sq = 0.0;
        if (t + 1 >= fv + p && dsize != 0) {
            const int jmax = deq[head];
            const float m = prices_tm[jmax * cols + s];
            if (xi_finite && isfinite(m) && fabsf(m) > 1e-20f) {
                const double diff = static_cast<double>(xi) - static_cast<double>(m);
                const double dd = diff / static_cast<double>(m);
                const double sq = dd * dd; // base scalar=1.0
                new_sq = sq;
                new_valid = 1u;
            }
        }
        if (valid_ring[ring_idx]) { sum -= sq_ring[ring_idx]; count -= 1; }
        if (new_valid) { sum += new_sq; count += 1; }
        sq_ring[ring_idx] = new_sq; valid_ring[ring_idx] = new_valid;
        ring_idx += 1; if (ring_idx == p) ring_idx = 0;

        if (t >= warm_end) {
            if (count == p) {
                double avg = sum / static_cast<double>(p);
                if (avg < 0.0) avg = 0.0;
                out_tm[idx] = static_cast<float>(sqrt(avg)) * s_abs;
            } else {
                out_tm[idx] = CUDART_NAN_F;
            }
        }
    }
}
