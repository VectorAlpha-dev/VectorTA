// Kaufmanstop CUDA kernels (FP32)
//
// Numeric contract matches the scalar implementation in src/indicators/kaufmanstop.rs:
// - warmup index per row = first_valid + period - 1; write NaN before warmup
// - base is low for long (stop below), high for short (stop above)
// - out = base + signed_mult * ma(range), where signed_mult = -mult for long, +mult for short
// - ignore NaNs exactly like scalar: NaN in base or MA propagates

#include <cuda_runtime.h>
#include <math_constants.h> // CUDART_NAN_F

extern "C" {

// -------------------- Single row (AXPY) --------------------
__global__ void kaufmanstop_axpy_row_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ ma_row,
    int len,
    float signed_mult,
    int warm,
    int base_is_low, // 1 = long (use low), 0 = short (use high)
    float* __restrict__ out_row
) {
    const float* __restrict__ base = base_is_low ? low : high; // hoist uniform branch

    // grid‑stride loop for flexible launch sizing and high occupancy
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        float out;
        if (i < warm) {
            out = CUDART_NAN_F; // canonical NaN
        } else {
            // fused multiply-add improves throughput and rounding
            out = fmaf(ma_row[i], signed_mult, base[i]);
        }
        out_row[i] = out;
    }
}

// -------------------- Many series × one param (time‑major) --------------------
__global__ void kaufmanstop_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,      // [rows * cols], time-major index = t*cols + s
    const float* __restrict__ low_tm,       // [rows * cols]
    const float* __restrict__ ma_tm,        // [rows * cols]
    const int*   __restrict__ first_valids, // [cols]
    int cols,
    int rows,
    float signed_mult,
    int base_is_low,
    int period,
    float* __restrict__ out_tm              // [rows * cols]
){
    const float* __restrict__ base_tm = base_is_low ? low_tm : high_tm;

    // Dual mapping: support legacy 1D launches from the host wrapper and
    // the optimized 2D launch (x=series, y=time tiles). This keeps the kernel
    // drop-in compatible while enabling future 2D dispatch without changes.
    if (gridDim.y == 1 && blockDim.y == 1) {
        // 1D path over flattened matrix (time-major)
        const int total = rows * cols;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
            const int s = i % cols;
            const int t = i / cols;
            const int warm = first_valids[s] + period - 1;
            float out;
            if (t < warm) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_tm[i], signed_mult, base_tm[i]);
            }
            out_tm[i] = out;
        }
    } else {
        // 2D path: x -> series, y -> time tiles
        int s = blockIdx.x * blockDim.x + threadIdx.x;      // series
        int t0 = blockIdx.y * blockDim.y + threadIdx.y;     // starting time for this thread
        int t_stride = blockDim.y * gridDim.y;

        if (s >= cols) return;
        const int warm = first_valids[s] + period - 1;

        for (int t = t0; t < rows; t += t_stride) {
            const int idx = t * cols + s; // MAD index
            float out;
            if (t < warm) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_tm[idx], signed_mult, base_tm[idx]);
            }
            out_tm[idx] = out;
        }
    }
}

// -------------------- One series × many params (time‑major inside each param) --------------------
// Shared memory broadcast of base[t] across parameter lanes in the block.
// Layout: ma_pm[p*rows + t] and out_pm[p*rows + t].
__global__ void kaufmanstop_one_series_many_params_time_major_f32(
    const float* __restrict__ high,          // [rows]
    const float* __restrict__ low,           // [rows]
    const float* __restrict__ ma_pm,         // [params * rows]
    const int*   __restrict__ warm_ps,       // [params]
    const float* __restrict__ signed_mults,  // [params]
    int rows,
    int params,
    int base_is_low,                         // 1 = long (use low), 0 = short (use high)
    float* __restrict__ out_pm               // [params * rows]
) {
    extern __shared__ float s_base[]; // size = blockDim.x
    const float* __restrict__ base = base_is_low ? low : high;

    // x -> time tile, y -> parameter lane
    int p  = blockIdx.y * blockDim.y + threadIdx.y;
    int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    int t_stride = blockDim.x * gridDim.x;

    for (int t = t0; t < rows; t += t_stride) {
        // single load of base[t] per x‑lane
        if (threadIdx.y == 0) {
            s_base[threadIdx.x] = base[t];
        }
        __syncthreads();

        if (p < params) {
            const int idx = p * rows + t;
            float out;
            if (t < warm_ps[p]) {
                out = CUDART_NAN_F;
            } else {
                out = fmaf(ma_pm[idx], signed_mults[p], s_base[threadIdx.x]);
            }
            out_pm[idx] = out;
        }
        __syncthreads();
    }
}

}

