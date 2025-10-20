// Keltner band combiners (optimized, FP32)
// - Vectorized float4 path when sizes permit
// - FMA for accuracy/perf
// - Grid-stride loops
// - Shared-memory broadcast of per-row scalars in the 1-series-many-params kernel

#include <cuda_runtime.h>

#ifndef __CUDACC_RTC__
// local fast NaN to avoid headers; identical bit pattern to CUDART_NAN_F
static __device__ __forceinline__ float fast_nan() {
    return __int_as_float(0x7fffffff);
}
#else
// NVRTC: fallback
static __device__ __forceinline__ float fast_nan() { return nanf(""); }
#endif

// -----------------------------------------------
// 1) One price series, many parameter rows (main focus)
// ma_rows, atr_rows : [Prows x len] row-major
// -----------------------------------------------
extern "C" __global__ __launch_bounds__(256, 2)
void keltner_batch_f32(
    const float* __restrict__ ma_rows,           // [Prows x len]
    const float* __restrict__ atr_rows,          // [Prows x len]
    const int*   __restrict__ row_period_idx,    // [R]
    const float* __restrict__ row_multipliers,   // [R]
    const int*   __restrict__ row_warms,         // [R] absolute warm index per row
    int len,
    int rows,                                    // R
    float* __restrict__ out_upper,               // [R x len]
    float* __restrict__ out_middle,              // [R x len]
    float* __restrict__ out_lower                // [R x len]
) {
    const int r = blockIdx.y;
    if (r >= rows) return;

    // Broadcast the row's scalar parameters once per block
    __shared__ int   s_pidx;
    __shared__ int   s_warm;
    __shared__ float s_mult;
    if (threadIdx.x == 0) {
        s_pidx = row_period_idx[r];
        s_warm = row_warms[r];
        s_mult = row_multipliers[r];
    }
    __syncthreads();

    const float neg_mult = -s_mult;
    const float nanv     = fast_nan();

    // Base pointers for this row/period row
    const size_t base_p = static_cast<size_t>(s_pidx) * static_cast<size_t>(len);
    const size_t base_r = static_cast<size_t>(r)      * static_cast<size_t>(len);

    const float* __restrict__ ma  = ma_rows  + base_p;
    const float* __restrict__ atr = atr_rows + base_p;
    float* __restrict__ outM = out_middle + base_r;
    float* __restrict__ outU = out_upper  + base_r;
    float* __restrict__ outL = out_lower  + base_r;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Vectorized path if len is multiple of 4 (cudaMalloc gives >=256B alignment)
    if ((len & 3) == 0) {
        const int len4 = len >> 2;
        for (int i4 = t0; i4 < len4; i4 += stride) {
            const int t = (i4 << 2);          // starting lane time index
            // If the whole 4-lane pack is before warm, write 4x NaN without reading inputs
            if (t + 3 < s_warm) {
                const float4 n4 = make_float4(nanv, nanv, nanv, nanv);
                reinterpret_cast<float4*>(outM)[i4] = n4;
                reinterpret_cast<float4*>(outU)[i4] = n4;
                reinterpret_cast<float4*>(outL)[i4] = n4;
                continue;
            }

            const float4 mid4 = reinterpret_cast<const float4*>(ma )[i4];
            const float4 a4   = reinterpret_cast<const float4*>(atr)[i4];

            const bool v0 = (t + 0) >= s_warm;
            const bool v1 = (t + 1) >= s_warm;
            const bool v2 = (t + 2) >= s_warm;
            const bool v3 = (t + 3) >= s_warm;

            const float m0 = v0 ? mid4.x : nanv;
            const float m1 = v1 ? mid4.y : nanv;
            const float m2 = v2 ? mid4.z : nanv;
            const float m3 = v3 ? mid4.w : nanv;

            const float u0 = v0 ? fmaf(s_mult, a4.x, mid4.x) : nanv;
            const float u1 = v1 ? fmaf(s_mult, a4.y, mid4.y) : nanv;
            const float u2 = v2 ? fmaf(s_mult, a4.z, mid4.z) : nanv;
            const float u3 = v3 ? fmaf(s_mult, a4.w, mid4.w) : nanv;

            const float l0 = v0 ? fmaf(neg_mult, a4.x, mid4.x) : nanv;
            const float l1 = v1 ? fmaf(neg_mult, a4.y, mid4.y) : nanv;
            const float l2 = v2 ? fmaf(neg_mult, a4.z, mid4.z) : nanv;
            const float l3 = v3 ? fmaf(neg_mult, a4.w, mid4.w) : nanv;

            reinterpret_cast<float4*>(outM)[i4] = make_float4(m0, m1, m2, m3);
            reinterpret_cast<float4*>(outU)[i4] = make_float4(u0, u1, u2, u3);
            reinterpret_cast<float4*>(outL)[i4] = make_float4(l0, l1, l2, l3);
        }
        return;
    }

    // Scalar fallback
    for (int t = t0; t < len; t += stride) {
        if (t < s_warm) {
            outM[t] = nanv; outU[t] = nanv; outL[t] = nanv;
            continue;
        }
        const float mid = ma[t];
        const float a   = atr[t];
        outM[t] = mid;
        outU[t] = fmaf(s_mult,  a, mid);
        outL[t] = fmaf(neg_mult, a, mid);
    }
}


// -----------------------------------------------
// 2) Many series, one param (time-major layout)
// ma_tm, atr_tm : [rows*cols] time-major (idx = t*cols + s)
// Includes both:
//   - fast 2-D launch path (grid.y = rows, x covers cols)
//   - 1-D fallback compatible with the original launch
// -----------------------------------------------
extern "C" __global__ __launch_bounds__(256, 2)
void keltner_many_series_one_param_f32(
    const float* __restrict__ ma_tm,        // [rows*cols] time-major
    const float* __restrict__ atr_tm,       // [rows*cols] time-major
    const int*   __restrict__ first_valids, // [cols]
    int period,
    int cols,
    int rows,
    int elems,                               // rows * cols (may be unused in 2-D path)
    float multiplier,
    float* __restrict__ out_upper_tm,       // [rows*cols] time-major
    float* __restrict__ out_middle_tm,      // [rows*cols] time-major
    float* __restrict__ out_lower_tm        // [rows*cols] time-major
) {
    const float nanv      = fast_nan();
    const float neg_mult  = -multiplier;

    // Fast 2-D path (recommended): grid.y == rows, grid.x covers columns
    if (gridDim.y > 1) {
        const int t = blockIdx.y;
        if (t >= rows) return;

        const int s0     = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        const size_t row_off = static_cast<size_t>(t) * static_cast<size_t>(cols);
        const float* __restrict__ ma_row  = ma_tm  + row_off;
        const float* __restrict__ atr_row = atr_tm + row_off;
        float* __restrict__ outM_row = out_middle_tm + row_off;
        float* __restrict__ outU_row = out_upper_tm  + row_off;
        float* __restrict__ outL_row = out_lower_tm  + row_off;

        if ((cols & 3) == 0) {
            const int cols4 = cols >> 2;
            for (int i4 = s0; i4 < cols4; i4 += stride) {
                const int s = (i4 << 2);

                // Load 4 first_valids; compute warms once
                const int4 fv4 = reinterpret_cast<const int4*>(first_valids)[i4];
                const int w0 = fv4.x + period - 1;
                const int w1 = fv4.y + period - 1;
                const int w2 = fv4.z + period - 1;
                const int w3 = fv4.w + period - 1;

                const bool v0 = t >= w0;
                const bool v1 = t >= w1;
                const bool v2 = t >= w2;
                const bool v3 = t >= w3;

                // If all invalid for this time t, write NaNs without touching MA/ATR
                if (!(v0 | v1 | v2 | v3)) {
                    const float4 n4 = make_float4(nanv, nanv, nanv, nanv);
                    reinterpret_cast<float4*>(outM_row)[i4] = n4;
                    reinterpret_cast<float4*>(outU_row)[i4] = n4;
                    reinterpret_cast<float4*>(outL_row)[i4] = n4;
                    continue;
                }

                const float4 mid4 = reinterpret_cast<const float4*>(ma_row )[i4];
                const float4 a4   = reinterpret_cast<const float4*>(atr_row)[i4];

                const float m0 = v0 ? mid4.x : nanv;
                const float m1 = v1 ? mid4.y : nanv;
                const float m2 = v2 ? mid4.z : nanv;
                const float m3 = v3 ? mid4.w : nanv;

                const float u0 = v0 ? fmaf(multiplier, a4.x, mid4.x) : nanv;
                const float u1 = v1 ? fmaf(multiplier, a4.y, mid4.y) : nanv;
                const float u2 = v2 ? fmaf(multiplier, a4.z, mid4.z) : nanv;
                const float u3 = v3 ? fmaf(multiplier, a4.w, mid4.w) : nanv;

                const float l0 = v0 ? fmaf(neg_mult, a4.x, mid4.x) : nanv;
                const float l1 = v1 ? fmaf(neg_mult, a4.y, mid4.y) : nanv;
                const float l2 = v2 ? fmaf(neg_mult, a4.z, mid4.z) : nanv;
                const float l3 = v3 ? fmaf(neg_mult, a4.w, mid4.w) : nanv;

                reinterpret_cast<float4*>(outM_row)[i4] = make_float4(m0, m1, m2, m3);
                reinterpret_cast<float4*>(outU_row)[i4] = make_float4(u0, u1, u2, u3);
                reinterpret_cast<float4*>(outL_row)[i4] = make_float4(l0, l1, l2, l3);
            }
            return;
        }

        // Scalar fallback across columns
        for (int s = s0; s < cols; s += stride) {
            const int warm = first_valids[s] + period - 1;
            if (t < warm) {
                outM_row[s] = nanv; outU_row[s] = nanv; outL_row[s] = nanv;
                continue;
            }
            const float mid = ma_row[s];
            const float a   = atr_row[s];
            outM_row[s] = mid;
            outU_row[s] = fmaf(multiplier, a, mid);
            outL_row[s] = fmaf(neg_mult,  a, mid);
        }
        return;
    }

    // 1-D fallback path (compatible with original launch model)
    {
        const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int idx = i0; idx < elems; idx += stride) {
            const int t = idx / cols;
            const int s = idx - t * cols;
            const int warm = first_valids[s] + period - 1;
            if (t < warm) {
                out_middle_tm[idx] = nanv;
                out_upper_tm [idx] = nanv;
                out_lower_tm [idx] = nanv;
                continue;
            }
            const float mid = ma_tm[idx];
            const float a   = atr_tm[idx];
            out_middle_tm[idx] = mid;
            out_upper_tm [idx] = fmaf(multiplier, a, mid);
            out_lower_tm [idx] = fmaf(-multiplier, a, mid); // local negate
        }
    }
}
