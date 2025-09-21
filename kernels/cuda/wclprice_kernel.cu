// CUDA kernel for WCLPRICE (Weighted Close Price).
//
// Each thread processes a subset of the price series using a grid-stride loop.
// The kernel expects FP32 inputs and writes FP32 outputs, mirroring the
// high+low+2*close divided by 4.0 calculation performed by the scalar Rust
// implementation. NaN handling matches the CPU path: any NaN in the inputs or
// indices before the first valid element produce NaN in the output.

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void wclprice_kernel_f32(const float* __restrict__ high,
                                               const float* __restrict__ low,
                                               const float* __restrict__ close,
                                               int len,
                                               int first_valid,
                                               float* __restrict__ out) {
    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (len <= 0) {
        return;
    }

    const int clamped_first = first_valid < 0 ? 0 : first_valid;
    for (int idx = thread_index; idx < len; idx += stride) {
        if (idx < clamped_first) {
            out[idx] = CUDART_NAN_F;
            continue;
        }

        const float h = high[idx];
        const float l = low[idx];
        const float c = close[idx];

        if (isnan(h) || isnan(l) || isnan(c)) {
            out[idx] = CUDART_NAN_F;
        } else {
            const float sum = h + l + c + c;
            out[idx] = sum * 0.25f;
        }
    }
}
