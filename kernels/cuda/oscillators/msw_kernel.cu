











#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef MSW_BLOCK_X
#define MSW_BLOCK_X 256
#endif



#ifndef MSW_CHUNK_PER_THREAD
#define MSW_CHUNK_PER_THREAD 8
#endif


static __device__ __constant__ float MSW_TWO_PI_F    = 6.28318530717958647692f;
static __device__ __constant__ float MSW_SQRT_HALF_F = 0.70710678118654752440f;




static __device__ __forceinline__ float msw_phase_from_rp_ip_eps(float rp, float ip, float eps) {
    float phase;
    if (fabsf(rp) > eps) {
        phase = atanf(ip / rp);
    } else {
        phase = (ip < 0.0f ? -CUDART_PI_F : CUDART_PI_F);
    }
    if (rp < 0.0f) phase += CUDART_PI_F;
    phase += 0.5f * CUDART_PI_F;
    if (phase < 0.0f) phase += CUDART_PI_F * 2.0f;
    if (phase > CUDART_PI_F * 2.0f) phase -= CUDART_PI_F * 2.0f;
    return phase;
}



static __device__ __forceinline__ void
msw_build_weights_stride(float* __restrict__ cosw,
                         float* __restrict__ sinw,
                         int period)
{
    const float step = MSW_TWO_PI_F / (float)period;

    
    float s_stride, c_stride;
    __sincosf(step * (float)blockDim.x, &s_stride, &c_stride);

    const int lane = threadIdx.x;
    float s0, c0;
    __sincosf(step * (float)lane, &s0, &c0);

    
    for (int j = lane; j < period; j += blockDim.x) {
        sinw[j] = s0;
        cosw[j] = c0;

        
        
        
        float s_old = s0, c_old = c0;
        float s_new = fmaf(c_old, s_stride, s_old * c_stride);
        float c_new = fmaf(-s_old, s_stride, c_old * c_stride);
        s0 = s_new; c0 = c_new;
    }
}


static __device__ __forceinline__ void
msw_dot_weighted_window(const float* __restrict__ tile,
                        const float* __restrict__ cosw,
                        const float* __restrict__ sinw,
                        int start,
                        int period,
                        float &rp, float &ip)
{
    rp = 0.0f; ip = 0.0f;
    float cr = 0.0f, ci = 0.0f; 
    #pragma unroll 4
    for (int k = 0; k < period; ++k) {
        const float w = tile[start + (period - 1 - k)];
        const float tr = cosw[k] * w;
        const float ti = sinw[k] * w;
        float yr = tr - cr; float sr = rp + yr; cr = (sr - rp) - yr; rp = sr;
        float yi = ti - ci; float si = ip + yi; ci = (si - ip) - yi; ip = si;
    }
}


static __device__ __forceinline__ void
msw_dot_weighted_window_f64(const float* __restrict__ tile,
                            const float* __restrict__ cosw,
                            const float* __restrict__ sinw,
                            int start,
                            int period,
                            float &rp, float &ip)
{
    double dr = 0.0, di = 0.0;
    #pragma unroll 4
    for (int k = 0; k < period; ++k) {
        const double w = (double)tile[start + (period - 1 - k)];
        dr += (double)cosw[k] * w;
        di += (double)sinw[k] * w;
    }
    rp = (float)dr;
    ip = (float)di;
}


static __device__ __forceinline__ void
msw_dot_weighted_window_f64_drdi(const float* __restrict__ tile,
                                 const float* __restrict__ cosw,
                                 const float* __restrict__ sinw,
                                 int start,
                                 int period,
                                 double &dr, double &di)
{
    dr = 0.0; di = 0.0;
    #pragma unroll 4
    for (int k = 0; k < period; ++k) {
        const double w = (double)tile[start + (period - 1 - k)];
        dr += (double)cosw[k] * w;
        di += (double)sinw[k] * w;
    }
}


static __device__ __forceinline__ float msw_phase_batch_from_dr_di(double dr, double di)
{
    float phase;
    if (fabs(dr) > 1e-3) {
        
        phase = atanf((float)(di / dr));
    } else {
        phase = ((di < 0.0) ? -CUDART_PI_F : CUDART_PI_F);
    }
    if (dr < 0.0) phase += CUDART_PI_F;
    phase += 0.5f * CUDART_PI_F;
    if (phase < 0.0f) phase += CUDART_PI_F * 2.0f;
    if (phase > CUDART_PI_F * 2.0f) phase -= CUDART_PI_F * 2.0f;
    return phase;
}



static __device__ __forceinline__ void
msw_dot_window_rotate(const float* __restrict__ tile,
                      float c_step, float s_step,
                      int start, int period,
                      float &rp, float &ip)
{
    rp = 0.0f; ip = 0.0f;
    float cr = 0.0f, ci = 0.0f; 
    float c = 1.0f, s = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < period; ++k) {
        const float w = tile[start + (period - 1 - k)];
        const float tr = c * w;
        const float ti = s * w;
        float yr = tr - cr; float sr = rp + yr; cr = (sr - rp) - yr; rp = sr;
        float yi = ti - ci; float si = ip + yi; ci = (si - ip) - yi; ip = si;
        
        float c_old = c, s_old = s;
        float s_new = fmaf(c_old, s_step, s_old * c_step);
        float c_new = fmaf(-s_old, s_step, c_old * c_step);
        c = c_new; s = s_new;
    }
}


static __device__ __forceinline__ void
msw_dot_window_rotate_f64(const float* __restrict__ tile,
                          double c_step, double s_step,
                          int start, int period,
                          double &dr, double &di)
{
    dr = 0.0; di = 0.0;
    double c = 1.0, s = 0.0;
    #pragma unroll 4
    for (int k = 0; k < period; ++k) {
        const double w = (double)tile[start + (period - 1 - k)];
        dr += c * w;
        di += s * w;
        const double s_new = s * c_step + c * s_step;
        const double c_new = c * c_step - s * s_step;
        s = s_new; c = c_new;
    }
}


static __device__ __forceinline__ void
msw_rotate(float c_step, float s_step, float &rp, float &ip)
{
    float rp_old = rp, ip_old = ip;
    float rp_rot = fmaf(rp_old, c_step, -ip_old * s_step);
    float ip_rot = fmaf(rp_old, s_step,  ip_old * c_step);
    rp = rp_rot; ip = ip_rot;
}

static __device__ __forceinline__ void
msw_rotate_f64(double c_step, double s_step, double &rp, double &ip)
{
    double rp_old = rp, ip_old = ip;
    double rp_rot = rp_old * c_step - ip_old * s_step;
    double ip_rot = rp_old * s_step + ip_old * c_step;
    rp = rp_rot; ip = ip_rot;
}



extern "C" __global__
void msw_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) 
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_sine = combo * 2;
    const int row_lead = row_sine + 1;
    const int base_sine = row_sine * series_len;
    const int base_lead = row_lead * series_len;

    
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop = min(warm, series_len);
        for (; t < stop; t += stride) {
            out[base_sine + t] = NAN;
            out[base_lead + t] = NAN;
        }
    }

    
    
    extern __shared__ unsigned char shmem_raw[];
    double* __restrict__ cosw_d = reinterpret_cast<double*>(shmem_raw);
    double* __restrict__ sinw_d = cosw_d + period;
    float* __restrict__ tile = reinterpret_cast<float*>(sinw_d + period); 

    
    const double step_d = 6.2831852 / (double)period;
    for (int j = threadIdx.x; j < period; j += blockDim.x) {
        const double ang = step_d * (double)j;
        sinw_d[j] = sin(ang);
        cosw_d[j] = cos(ang);
    }
    __syncthreads();

    
    const int stride2 = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride2) {
        const int t_begin = max(base_t, warm);
        const int t_end = min(base_t + blockDim.x - 1, series_len - 1);
        if (t_begin > t_end) continue;

        const int tile_in_start = t_begin - (period - 1);
        const int tile_len = (t_end - t_begin + 1) + (period - 1);

        
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            tile[i] = prices[tile_in_start + i];
        }
        __syncthreads();

        const int t = base_t + threadIdx.x;
        if (t >= t_begin && t <= t_end) {
            const int start = t - t_begin; 
            double dr = 0.0, di = 0.0;
            #pragma unroll 1
            for (int j = 0; j < period; ++j) {
                const double w = (double)tile[start + (period - 1 - j)];
                dr += cosw_d[j] * w;
                di += sinw_d[j] * w;
            }
            const float phase = msw_phase_batch_from_dr_di(dr, di);
            float s, c;
            __sincosf(phase, &s, &c);
            out[base_sine + t] = s;
            out[base_lead + t] = (s + c) * MSW_SQRT_HALF_F;
        }
        __syncthreads();
    }
}



extern "C" __global__
void msw_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm, 
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids, 
    float* __restrict__ out_tm)           
{
    if (period <= 0) return;
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const int warm = first_valids[series_idx] + period - 1;
    const int col_sine = series_idx;
    const int col_lead = series_idx + num_series;

    
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop = min(warm, series_len);
        for (; t < stop; t += stride) {
            out_tm[t * (2 * num_series) + col_sine] = NAN;
            out_tm[t * (2 * num_series) + col_lead] = NAN;
        }
    }

    
    const bool use_lut = true;
    extern __shared__ float shmem[];
    float* __restrict__ cosw = shmem;
    float* __restrict__ sinw = cosw + (use_lut ? period : 0);
    const int T = blockDim.x * MSW_CHUNK_PER_THREAD;
    float* __restrict__ tile = sinw + (use_lut ? period : 0);

    float s_step, c_step;
    __sincosf(MSW_TWO_PI_F / (float)period, &s_step, &c_step);

    if (use_lut) {
        msw_build_weights_stride(cosw, sinw, period);
    }
    __syncthreads();

    const int grid_tile_stride = gridDim.x * T;
    for (int base_t = blockIdx.x * T; base_t < series_len; base_t += grid_tile_stride) {

        const int t_begin = max(base_t, warm);
        const int t_end   = min(base_t + T - 1, series_len - 1);
        if (t_begin > t_end) continue;

        const int tile_in_start = t_begin - (period - 1);
        const int tile_len      = (t_end - t_begin + 1) + (period - 1);

        
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            const int tt = tile_in_start + i;
            tile[i] = prices_tm[tt * num_series + series_idx];
        }
        __syncthreads();

        const int out0 = t_begin + threadIdx.x * MSW_CHUNK_PER_THREAD;
        if (out0 <= t_end) {
            const int emit  = min(MSW_CHUNK_PER_THREAD, t_end - out0 + 1);
            const int start = out0 - t_begin;

            float rp, ip;
            if (use_lut) {
                msw_dot_weighted_window_f64(tile, cosw, sinw, start, period, rp, ip);
            } else {
                msw_dot_window_rotate(tile, c_step, s_step, start, period, rp, ip);
            }

            for (int m = 0; m < emit; ++m) {
                const int t = out0 + m;
                const float phase = msw_phase_from_rp_ip_eps(rp, ip, 1e-3f);

                float s_val, c_val;
                __sincosf(phase, &s_val, &c_val);
                out_tm[t * (2 * num_series) + col_sine] = s_val;
                out_tm[t * (2 * num_series) + col_lead] = (s_val + c_val) * MSW_SQRT_HALF_F;

                if (m + 1 < emit) {
                    const float x_old = tile[start + m];
                    const float x_new = tile[start + m + period];
                    msw_rotate(c_step, s_step, rp, ip);
                    rp += (x_new - x_old);
                    if (fabsf(rp) <= 0.0015f) {
                        const int start_next = start + m + 1;
                        if (use_lut) { msw_dot_weighted_window(tile, cosw, sinw, start_next, period, rp, ip); }
                        else { msw_dot_window_rotate(tile, c_step, s_step, start_next, period, rp, ip); }
                    }
                }
            }
        }
        __syncthreads();
    }
}
