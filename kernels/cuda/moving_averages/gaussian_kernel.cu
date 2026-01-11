










#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

 #include <cuda_runtime.h>
 #include <math.h>

 
 
 
 #ifndef GAUSS_BLOCK_DIM
 #define GAUSS_BLOCK_DIM 256
 #endif

 #ifndef GAUSS_USE_STREAMING_STORES
 #define GAUSS_USE_STREAMING_STORES 0
 #endif

 
 
 
 static __device__ __forceinline__ float qnan_f() {
     return __int_as_float(0x7fffffff); 
 }

 static __device__ __forceinline__ void store_out(float* __restrict__ p, float v) {
 #if GAUSS_USE_STREAMING_STORES
     
     
     asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(p), "f"(v));
 #else
     *p = v;
 #endif
 }

 static __device__ __forceinline__ int clampi(int x, int lo, int hi) {
     return (x < lo) ? lo : (x > hi ? hi : x);
 }

 
 
 
 
 
 static __device__ __forceinline__ void gaussian_run_poles1(
     const float* __restrict__ prices,
     float* __restrict__ out,
     int series_len,
     int stride,
     int valid,
     float nan_f,
     double c0,
     double c1)
 {
     double y_prev = 0.0;
     int idx = 0;
     int t = 0;

     
     for (; t < valid && t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         y_prev = c1 * y_prev + c0 * x;
         store_out(out + idx, nan_f);
     }
     
     for (; t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         y_prev = c1 * y_prev + c0 * x;
         store_out(out + idx, static_cast<float>(y_prev));
     }
 }

 static __device__ __forceinline__ void gaussian_run_poles2(
     const float* __restrict__ prices,
     float* __restrict__ out,
     int series_len,
     int stride,
     int valid,
     float nan_f,
     double c0,
     double c1,
     double c2)
 {
     double p1 = 0.0; 
     double p0 = 0.0; 
     int idx = 0;
     int t = 0;

     for (; t < valid && t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = c2 * p0 + c1 * p1 + c0 * x;
         p0 = p1; p1 = y;
         store_out(out + idx, nan_f);
     }
     for (; t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = c2 * p0 + c1 * p1 + c0 * x;
         p0 = p1; p1 = y;
         store_out(out + idx, static_cast<float>(y));
     }
 }

 static __device__ __forceinline__ void gaussian_run_poles3(
     const float* __restrict__ prices,
     float* __restrict__ out,
     int series_len,
     int stride,
     int valid,
     float nan_f,
     double c0,
     double c1,
     double c2,
     double c3)
 {
     double p2 = 0.0; 
     double p1 = 0.0; 
     double p0 = 0.0; 
     int idx = 0;
     int t = 0;

     for (; t < valid && t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = c3 * p0 + c2 * p1 + c1 * p2 + c0 * x;
         p0 = p1; p1 = p2; p2 = y;
         store_out(out + idx, nan_f);
     }
     for (; t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = c3 * p0 + c2 * p1 + c1 * p2 + c0 * x;
         p0 = p1; p1 = p2; p2 = y;
         store_out(out + idx, static_cast<float>(y));
     }
 }

 static __device__ __forceinline__ void gaussian_run_poles4(
     const float* __restrict__ prices,
     float* __restrict__ out,
     int series_len,
     int stride,
     int valid,
     float nan_f,
     double c0,
     double c1,
     double c2,
     double c3,
     double c4)
 {
     double p3 = 0.0; 
     double p2 = 0.0; 
     double p1 = 0.0; 
     double p0 = 0.0; 
     int idx = 0;
     int t = 0;

     for (; t < valid && t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = (((c4 * p0) + (c3 * p1)) + (c2 * p2)) + (c1 * p3) + (c0 * x);
         p0 = p1; p1 = p2; p2 = p3; p3 = y;
         store_out(out + idx, nan_f);
     }
     for (; t < series_len; ++t, idx += stride) {
         const double x = static_cast<double>(prices[idx]);
         const double y = (((c4 * p0) + (c3 * p1)) + (c2 * p2)) + (c1 * p3) + (c0 * x);
         p0 = p1; p1 = p2; p2 = p3; p3 = y;
         store_out(out + idx, static_cast<float>(y));
     }
 }

 
 
 
 
 extern "C" __global__ void gaussian_batch_f32(
     const float* __restrict__ prices,
     const int* __restrict__ periods,
     const int* __restrict__ poles,
     const float* __restrict__ coeffs,
     int coeff_stride,
     int series_len,
     int n_combos,
     int first_valid,
     float* __restrict__ out)
 {
     const float nan_f = qnan_f();

     
     for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
          combo < n_combos;
          combo += gridDim.x * blockDim.x)
     {
         const int period = periods[combo];
         const int pole   = poles[combo];

         
         if (period < 2 || pole < 1 || pole > 4 || series_len <= 0) {
             
             
             
             continue;
         }

         float* out_row = out + combo * series_len;

         int start = first_valid;
         start = clampi(start, 0, series_len);
         
         int warm = first_valid + period;
         warm = clampi(warm, 0, series_len);

         
         int valid = warm > start ? warm : start;
         valid = clampi(valid, 0, series_len);

         const float* coeff = coeffs + combo * coeff_stride;
         const double c0 = static_cast<double>(coeff[0]);
         const double c1 = static_cast<double>(coeff[1]);
         const double c2 = static_cast<double>(coeff[2]);
         const double c3 = static_cast<double>(coeff[3]);
         const double c4 = static_cast<double>(coeff[4]);

         switch (pole) {
             case 1:
                 gaussian_run_poles1(prices, out_row, series_len, 1, valid, nan_f, c0, c1);
                 break;
             case 2:
                 gaussian_run_poles2(prices, out_row, series_len, 1, valid, nan_f, c0, c1, c2);
                 break;
             case 3:
                 gaussian_run_poles3(prices, out_row, series_len, 1, valid, nan_f, c0, c1, c2, c3);
                 break;
             case 4:
             default:
                 gaussian_run_poles4(prices, out_row, series_len, 1, valid, nan_f, c0, c1, c2, c3, c4);
                 break;
         }
     }
 }

 
 
 
 
 extern "C" __global__ void gaussian_many_series_one_param_f32(
     const float* __restrict__ prices_tm,
     const float* __restrict__ coeffs,
     int period,
     int poles,
     int num_series,
     int series_len,
     const int* __restrict__ first_valids,
     float* __restrict__ out_tm)
 {
     if (period < 2 || poles < 1 || poles > 4 || series_len <= 0) return;

     const double c0 = static_cast<double>(coeffs[0]);
     const double c1 = static_cast<double>(coeffs[1]);
     const double c2 = static_cast<double>(coeffs[2]);
     const double c3 = static_cast<double>(coeffs[3]);
     const double c4 = static_cast<double>(coeffs[4]);
     const float nan_f = qnan_f();

     
     for (int s = blockIdx.x * blockDim.x + threadIdx.x;
          s < num_series;
          s += gridDim.x * blockDim.x)
     {
         int start = first_valids[s];
         start = clampi(start, 0, series_len);

         
         int warm = first_valids[s] + period;
         warm = clampi(warm, 0, series_len);

         int valid = warm > start ? warm : start;
         valid = clampi(valid, 0, series_len);

         const float* price_series = prices_tm + s;
         float* out_series = out_tm + s;
         const int stride = num_series;

         switch (poles) {
             case 1:
                 gaussian_run_poles1(price_series, out_series, series_len, stride, valid, nan_f, c0, c1);
                 break;
             case 2:
                 gaussian_run_poles2(price_series, out_series, series_len, stride, valid, nan_f, c0, c1, c2);
                 break;
             case 3:
                 gaussian_run_poles3(price_series, out_series, series_len, stride, valid, nan_f, c0, c1, c2, c3);
                 break;
             case 4:
             default:
                 gaussian_run_poles4(price_series, out_series, series_len, stride, valid, nan_f, c0, c1, c2, c3, c4);
                 break;
         }
     }
 }
