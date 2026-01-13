







extern "C" __global__
void alma_batch_f64(const double* __restrict__ prices,
                    const double* __restrict__ weights_flat,
                    const int*    __restrict__ periods,
                    const double* __restrict__ inv_norms,
                    int           max_period,
                    int           series_len,
                    int           n_combos,
                    int           first_valid,
                    double*       __restrict__ out)
{

    extern __shared__ double shared_weights[];


    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = n_combos * series_len;

    for (int flat_idx = tid; flat_idx < total; flat_idx += stride) {
        const int combo_idx = flat_idx / series_len;
        const int t = flat_idx % series_len;


        if (combo_idx >= n_combos) continue;

        const int period = periods[combo_idx];
        const double inv_norm = inv_norms[combo_idx];


        if (t < first_valid + period - 1) {
            out[flat_idx] = NAN;
            continue;
        }


        if (threadIdx.x < period) {
            shared_weights[threadIdx.x] = weights_flat[combo_idx * max_period + threadIdx.x];
        }
        __syncthreads();


        const int start_idx = t - period + 1;
        double sum = 0.0;


        #pragma unroll 4
        for (int k = 0; k < period; ++k) {
            sum += prices[start_idx + k] * shared_weights[k];
        }

        out[flat_idx] = sum * inv_norm;
    }
}



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


    const double m = offset * (period - 1);
    const double s = period / sigma;
    const double s2 = 2.0 * s * s;

    extern __shared__ double local_weights[];


    for (int i = tid; i < period; i += blockDim.x) {
        const double diff = i - m;
        local_weights[i] = exp(-(diff * diff) / s2);
    }
    __syncthreads();


    double sum = 0.0;
    for (int i = tid; i < period; i += blockDim.x) {
        sum += local_weights[i];
    }


    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }


    if (tid == 0) {
        inv_norms_out[combo_idx] = 1.0 / sum;
    }


    const int base_idx = combo_idx * max_period;
    for (int i = tid; i < period; i += blockDim.x) {
        weights_out[base_idx + i] = local_weights[i];
    }
}