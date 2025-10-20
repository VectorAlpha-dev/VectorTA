// CUDA kernels for Elder Ray Index (ERI)
//
// Math mirrors src/indicators/eri.rs exactly:
// bull[i] = high[i] - MA[i]
// bear[i] = low[i]  - MA[i]
// Warmup/NaN semantics:
//   - Outputs before warmup = first_valid + period - 1 are NaN regardless of MA availability.
//   - If any input at i is NaN, result at i becomes NaN by normal FP rules.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef ERI_TIME_TILE
// Small time tile reused across params. 16 keeps smem tiny (128 B per array).
#define ERI_TIME_TILE 16
#endif

// --- Utility ---
__device__ __forceinline__ float eri_qnan() {
    // Use standard NaN constructor; compiled to a constant on device.
    return nanf("");
}

// ============================================================================
// 1) Single-series, single-param (row) – improved: grid-stride + precomputed warm
//    Same symbol as before.
// ==========================================================================
extern "C" __global__ void eri_batch_f32(
    const float* __restrict__ high,   // [series_len]
    const float* __restrict__ low,    // [series_len]
    const float* __restrict__ ma,     // [series_len]
    int series_len,
    int first_valid,
    int period,
    float* __restrict__ bull,         // [series_len]
    float* __restrict__ bear          // [series_len]
) {
    const int stride = blockDim.x * gridDim.x;
    const int warm   = first_valid + period - 1;
    const float nanv = eri_qnan();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += stride) {
        if (i < warm) {
            if (bull) bull[i] = nanv;
            if (bear) bear[i] = nanv;
        } else {
            const float m = ma[i];
            if (bull) bull[i] = high[i] - m;
            if (bear) bear[i] = low[i]  - m;
        }
    }
}

// ============================================================================
// 2) Many-series, one-param (time-major) – optimized: grid-stride T tiling
//    Layout: time-major [rows x cols]; index = t*cols + s
//    Same symbol as before.
// ==========================================================================
extern "C" __global__ void eri_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ ma_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids, // per-series first valid
    int period,
    float* __restrict__ bull_tm,
    float* __restrict__ bear_tm
) {
    const int s  = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    const int warm   = first_valids[s] + period - 1;
    const float nanv = eri_qnan();

    // Tile over time via grid-stride on the y-dimension
    for (int t0 = blockIdx.y * ERI_TIME_TILE; t0 < rows; t0 += gridDim.y * ERI_TIME_TILE) {
        const int tlimit = (rows - t0 > ERI_TIME_TILE) ? ERI_TIME_TILE : (rows - t0);

        // 1) NaN prefix in this tile (t < warm)
        int prefix = warm - t0;
        if (prefix < 0) prefix = 0;
        if (prefix > tlimit) prefix = tlimit;
        if (prefix > 0) {
            for (int tt = 0; tt < prefix; ++tt) {
                const int idx = (t0 + tt) * cols + s;
                if (bull_tm) bull_tm[idx] = nanv;
                if (bear_tm) bear_tm[idx] = nanv;
            }
        }

        // 2) Valid region in this tile (t >= warm)
        if (prefix < tlimit) {
            for (int tt = prefix; tt < tlimit; ++tt) {
                const int idx = (t0 + tt) * cols + s;
                const float m = ma_tm[idx];
                if (bull_tm) bull_tm[idx] = high_tm[idx] - m;
                if (bear_tm) bear_tm[idx] = low_tm[idx]  - m;
            }
        }
    }
}

// ============================================================================
// 3) ***NEW PRIMARY KERNEL*** One price series, many params (time-major)
//    Layout for MA/bull/bear: [rows x P], time-major; index = t*P + p
//    high/low are single vectors of length [rows] shared by all params.
//    periods can be nullptr -> use `period` for all params.
// ==========================================================================
extern "C" __global__ void eri_one_series_many_params_time_major_f32(
    const float* __restrict__ high,      // [rows]
    const float* __restrict__ low,       // [rows]
    const float* __restrict__ ma_tm,     // [rows x P], time-major
    int P,                               
    int rows,
    int first_valid,
    const int* __restrict__ periods,     // [P] or nullptr
    int period,                           // fallback if periods == nullptr
    float* __restrict__ bull_out,        // [rows x P], time-major or row-major
    float* __restrict__ bear_out,        // [rows x P], time-major or row-major
    int out_row_major                    // 0: TM (t*P+p), 1: RM (p*rows+t)
) {
    __shared__ float sh_high[ERI_TIME_TILE];
    __shared__ float sh_low [ERI_TIME_TILE];

    const float nanv = eri_qnan();

    const int p0      = blockIdx.x * blockDim.x + threadIdx.x;
    const int pstride = gridDim.x  * blockDim.x;

    // Tile over time; broadcast high/low tile to all threads (params) in the block
    for (int t0 = blockIdx.y * ERI_TIME_TILE; t0 < rows; t0 += gridDim.y * ERI_TIME_TILE) {
        const int tlimit = (rows - t0 > ERI_TIME_TILE) ? ERI_TIME_TILE : (rows - t0);

        // Load the time tile once per block
        if (threadIdx.x < tlimit) {
            sh_high[threadIdx.x] = high[t0 + threadIdx.x];
            sh_low [threadIdx.x] = low [t0 + threadIdx.x];
        }
        __syncthreads();

        // Each thread handles a strided set of params
        for (int p = p0; p < P; p += pstride) {
            const int per   = (periods ? periods[p] : period);
            const int warm  = first_valid + per - 1;
            const int base  = t0 * P + p;

            // NaN prefix for this param in the current tile
            int prefix = warm - t0;
            if (prefix < 0) prefix = 0;
            if (prefix > tlimit) prefix = tlimit;
            if (prefix > 0) {
                if (out_row_major) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        const int t = t0 + tt;
                        if (bull_out) bull_out[p*rows + t] = nanv;
                        if (bear_out) bear_out[p*rows + t] = nanv;
                    }
                } else if (bull_out && bear_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        const int idx = base + tt * P;
                        bull_out[idx] = nanv;
                        bear_out[idx] = nanv;
                    }
                } else if (bull_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        bull_out[base + tt * P] = nanv;
                    }
                } else if (bear_out) {
                    for (int tt = 0; tt < prefix; ++tt) {
                        bear_out[base + tt * P] = nanv;
                    }
                }
            }

            // Valid region of the tile (t >= warm)
            if (prefix < tlimit) {
                if (out_row_major) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int t = t0 + tt;
                        const float m = ma_tm[base + tt * P];
                        if (bull_out) bull_out[p*rows + t] = sh_high[tt] - m;
                        if (bear_out) bear_out[p*rows + t] = sh_low [tt] - m;
                    }
                } else if (bull_out && bear_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        const float m = ma_tm[idx];
                        bull_out[idx] = sh_high[tt] - m;
                        bear_out[idx] = sh_low [tt] - m;
                    }
                } else if (bull_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        bull_out[idx] = sh_high[tt] - ma_tm[idx];
                    }
                } else if (bear_out) {
                    for (int tt = prefix; tt < tlimit; ++tt) {
                        const int idx = base + tt * P;
                        bear_out[idx] = sh_low[tt] - ma_tm[idx];
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// 4) Utility: bank-conflict-free tiled transpose (row-major R×C -> time-major C×R)
//    Input  index (RM): r*C + c
//    Output index (TM): c*R + r
// ----------------------------------------------------------------------------
extern "C" __global__ void transpose_rm_to_tm_32x32_pad_f32(
    const float* __restrict__ in,
    int R, int C,
    float* __restrict__ out
){
    __shared__ float tile[32][32+1]; // +1 padding to avoid bank conflicts

    int c0 = blockIdx.x * 32 + threadIdx.x; // column in 'in' (time index)
    int r0 = blockIdx.y * 32 + threadIdx.y; // row    in 'in' (param index)

    if (r0 < R && c0 < C) {
        tile[threadIdx.y][threadIdx.x] = in[r0 * C + c0];
    } else {
        tile[threadIdx.y][threadIdx.x] = eri_qnan();
    }
    __syncthreads();

    int r1 = blockIdx.y * 32 + threadIdx.x; // param index in output (RM fast dim)
    int c1 = blockIdx.x * 32 + threadIdx.y; // time index in output (TM slow dim)
    if (r1 < R && c1 < C) {
        out[c1 * R + r1] = tile[threadIdx.x][threadIdx.y];
    }
}

