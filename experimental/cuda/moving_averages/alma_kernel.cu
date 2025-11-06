// CUDA kernel for ALMA (Arnaud Legoux Moving Average) batch operations
// This kernel processes multiple series with different ALMA parameters in parallel
//
// IMPORTANT: While this kernel uses double precision (f64), GPU computation may have
// different rounding behavior compared to CPU implementations. Users should validate
// that the precision meets their requirements, especially for financial calculations.
// CUDA is never auto-selected and must be explicitly requested via Kernel::CudaBatch.

extern "C" __global__
void alma_batch_f64(const double* __restrict__ prices,      // Input price data (SoA layout)
                    const double* __restrict__ weights_flat, // Flattened weights for all parameter combos
                    const int*    __restrict__ periods,      // Period for each parameter combo
                    const double* __restrict__ inv_norms,    // Precomputed 1/sum(weights) for each combo
                    int           max_period,                // Maximum period across all combos
                    int           series_len,                // Length of each series
                    int           n_combos,                  // Number of parameter combinations
                    int           first_valid,               // First valid index in price data
                    double*       __restrict__ out)          // Output array (SoA layout)
{
    // Dynamic shared memory for weights of current combo
    extern __shared__ double shared_weights[];
    
    // Grid-stride loop to handle arbitrary batch sizes
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = n_combos * series_len;
    
    for (int flat_idx = tid; flat_idx < total; flat_idx += stride) {
        const int combo_idx = flat_idx / series_len;
        const int t = flat_idx % series_len;
        
        // Bounds check
        if (combo_idx >= n_combos) continue;
        
        const int period = periods[combo_idx];
        const double inv_norm = inv_norms[combo_idx];
        
        // Check warmup period (matching CPU convention: first valid at period-1)
        if (t < first_valid + period - 1) {
            out[flat_idx] = NAN;
            continue;
        }
        
        // Load weights for this combo into shared memory (coalesced access)
        if (threadIdx.x < period) {
            shared_weights[threadIdx.x] = weights_flat[combo_idx * max_period + threadIdx.x];
        }
        __syncthreads();
        
        // Calculate ALMA value
        const int start_idx = t - period + 1;
        double sum = 0.0;
        
        // Unrolled dot product for better performance
        #pragma unroll 4
        for (int k = 0; k < period; ++k) {
            sum += prices[start_idx + k] * shared_weights[k];
        }
        
        out[flat_idx] = sum * inv_norm;
    }
}


// Helper kernel to compute Gaussian weights on GPU
extern "C" __global__
void compute_alma_weights_f64(double* __restrict__ weights_out,
                              const double* __restrict__ offsets,
                              const double* __restrict__ sigmas,
                              const int* __restrict__ periods,
                              double* __restrict__ inv_norms_out,
                              int n_combos,
                              int max_period)
{
    const int combo_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (combo_idx >= n_combos) return;
    
    const int period = periods[combo_idx];
    const double offset = offsets[combo_idx];
    const double sigma = sigmas[combo_idx];
    
    // Calculate Gaussian weights
    const double m = offset * (period - 1);
    const double s = period / sigma;
    const double s2 = 2.0 * s * s;
    
    extern __shared__ double local_weights[];
    
    // Each thread computes some weights
    for (int i = tid; i < period; i += blockDim.x) {
        const double diff = i - m;
        local_weights[i] = exp(-(diff * diff) / s2);
    }
    __syncthreads();
    
    // Parallel reduction to compute sum (for normalization)
    double sum = 0.0;
    for (int i = tid; i < period; i += blockDim.x) {
        sum += local_weights[i];
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Thread 0 writes the final results
    if (tid == 0) {
        inv_norms_out[combo_idx] = 1.0 / sum;
    }
    
    // Write weights to global memory
    const int base_idx = combo_idx * max_period;
    for (int i = tid; i < period; i += blockDim.x) {
        weights_out[base_idx + i] = local_weights[i];
    }
}