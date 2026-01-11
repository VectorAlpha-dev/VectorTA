












extern "C" {

#if __CUDACC_VER_MAJOR__ >= 10
#define UO_LDG(x) __ldg(&(x))
#else
#define UO_LDG(x) (x)
#endif


__device__ __constant__ float UO_W1 = 100.0f * (4.0f / 7.0f);
__device__ __constant__ float UO_W2 = 100.0f * (2.0f / 7.0f);
__device__ __constant__ float UO_W3 = 100.0f * (1.0f / 7.0f);


__device__ __forceinline__ float2 ldg_float2(const float2* __restrict__ base, int idx)
{
    float2 v; v.x = __ldg(&base[idx].x); v.y = __ldg(&base[idx].y); return v;
}


__device__ __forceinline__ void d_to_ds(float& hi, float& lo, const double d)
{
    
    hi = (float)d;
    
    lo = (float)(d - (double)hi);
}


__device__ __forceinline__ float ds_diff_to_f(
    const float ah, const float al,
    const float bh, const float bl)
{
    
    float s  = ah - bh;
    float vb = s - ah;
    float e  = (ah - (s - vb)) - (bh + vb);

    
    float t   = (al - bl);
    float s2  = s + t;
    float vb2 = s2 - s;
    e += (t - vb2);

    
    float hi = s2 + e;
    float lo = (s2 - hi) + e;
    return hi + lo;
}


__device__ __forceinline__ float recip_nr1(float x)
{
    float r = __frcp_rn(x);
    r = r * (2.0f - x * r);
    r = r * (2.0f - x * r);
    return r;
}

__device__ __forceinline__ float uo_nan() { return __int_as_float(0x7fffffff); }












__global__ void ultosc_batch_f32(
    const float2* __restrict__ pcmtl,
    const float2* __restrict__ ptr,
    int len,
    int first,
    const int3* __restrict__ periods,
    int nrows,
    float* __restrict__ out)
{
    const int row = blockIdx.y;
    if (row >= nrows) return;

    
    __shared__ int sp1, sp2, sp3, sstart;
    if (threadIdx.x == 0) {
        const int3 p = periods[row];
        const int p1 = p.x;
        const int p2 = p.y;
        const int p3 = p.z;
        sp1 = p1; sp2 = p2; sp3 = p3;
        const int maxp = max(p1, max(p2, p3));
        sstart = first + maxp - 1;
    }
    __syncthreads();

    float* __restrict__ row_out = out + (size_t)row * (size_t)len;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += stride)
    {
        if (i < sstart) { row_out[i] = uo_nan(); continue; }

        const int a  = i + 1;
        const int i1 = a - sp1;
        const int i2 = a - sp2;
        const int i3 = a - sp3;

        
        const float2 c_now  = ldg_float2(pcmtl, a);
        const float2 c_p1   = ldg_float2(pcmtl, i1);
        const float2 c_p2   = ldg_float2(pcmtl, i2);
        const float2 c_p3   = ldg_float2(pcmtl, i3);
        const float2 tr_now = ldg_float2(ptr, a);
        const float2 tr_p1  = ldg_float2(ptr, i1);
        const float2 tr_p2  = ldg_float2(ptr, i2);
        const float2 tr_p3  = ldg_float2(ptr, i3);

        
        const float s1a = ds_diff_to_f(c_now.x,  c_now.y,  c_p1.x,  c_p1.y);
        const float s1b = ds_diff_to_f(tr_now.x, tr_now.y, tr_p1.x, tr_p1.y);
        const float s2a = ds_diff_to_f(c_now.x,  c_now.y,  c_p2.x,  c_p2.y);
        const float s2b = ds_diff_to_f(tr_now.x, tr_now.y, tr_p2.x, tr_p2.y);
        const float s3a = ds_diff_to_f(c_now.x,  c_now.y,  c_p3.x,  c_p3.y);
        const float s3b = ds_diff_to_f(tr_now.x, tr_now.y, tr_p3.x, tr_p3.y);

        const float t1 = (s1b != 0.0f) ? (s1a * recip_nr1(s1b)) : 0.0f;
        const float t2 = (s2b != 0.0f) ? (s2a * recip_nr1(s2b)) : 0.0f;
        const float t3 = (s3b != 0.0f) ? (s3a * recip_nr1(s3b)) : 0.0f;

        row_out[i] = fmaf(UO_W1, t1, fmaf(UO_W2, t2, UO_W3 * t3));
    }
}











__global__ void ultosc_many_series_one_param_f32(
    const float2* __restrict__ pcmtl_tm,
    const float2* __restrict__ ptr_tm,
    int cols,
    int rows,
    int p1,
    int p2,
    int p3,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= cols) return;

    const int maxp  = max(p1, max(p2, p3));
    const int first = UO_LDG(first_valids[s]);
    const int start = first + maxp - 1;

    const int t_stride = blockDim.x * gridDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < rows; t += t_stride)
    {
        float* out_row = out_tm + (size_t)t * (size_t)cols;
        if (t < start) { out_row[s] = uo_nan(); continue; }

        const int idx_now = (t + 1) * cols + s;
        const int idx_1   = idx_now - p1 * cols;
        const int idx_2   = idx_now - p2 * cols;
        const int idx_3   = idx_now - p3 * cols;

        const float2 c_now  = ldg_float2(pcmtl_tm, idx_now);
        const float2 c_1    = ldg_float2(pcmtl_tm, idx_1);
        const float2 c_2    = ldg_float2(pcmtl_tm, idx_2);
        const float2 c_3    = ldg_float2(pcmtl_tm, idx_3);
        const float2 tr_now = ldg_float2(ptr_tm, idx_now);
        const float2 tr_1   = ldg_float2(ptr_tm, idx_1);
        const float2 tr_2   = ldg_float2(ptr_tm, idx_2);
        const float2 tr_3   = ldg_float2(ptr_tm, idx_3);

        const float s1a = ds_diff_to_f(c_now.x,  c_now.y,  c_1.x,  c_1.y);
        const float s1b = ds_diff_to_f(tr_now.x, tr_now.y, tr_1.x, tr_1.y);
        const float s2a = ds_diff_to_f(c_now.x,  c_now.y,  c_2.x,  c_2.y);
        const float s2b = ds_diff_to_f(tr_now.x, tr_now.y, tr_2.x, tr_2.y);
        const float s3a = ds_diff_to_f(c_now.x,  c_now.y,  c_3.x,  c_3.y);
        const float s3b = ds_diff_to_f(tr_now.x, tr_now.y, tr_3.x, tr_3.y);

        const float t1 = (s1b != 0.0f) ? (s1a * recip_nr1(s1b)) : 0.0f;
        const float t2 = (s2b != 0.0f) ? (s2a * recip_nr1(s2b)) : 0.0f;
        const float t3 = (s3b != 0.0f) ? (s3a * recip_nr1(s3b)) : 0.0f;

        out_row[s] = fmaf(UO_W1, t1, fmaf(UO_W2, t2, UO_W3 * t3));
    }
}

} 

