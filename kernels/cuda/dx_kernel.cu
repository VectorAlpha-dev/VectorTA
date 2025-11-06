// CUDA kernels for Directional Movement Index (DX)
//
// Math mirrors src/indicators/dx.rs exactly:
// - Warmup: accumulate +DM, -DM, TR over (period-1) steps from first_valid+1.
// - Then Wilder-style smoothing per step using FP64 accumulations.
// - DX = 100 * |DI+ - DI-| / (DI+ + DI-), with division-by-zero -> carry
// - NaN handling: if carry[i]!=0 (input at i was NaN), write previous output.
// - Before warm threshold, outputs are NaN (host pre-fills NaN prefix).

#include <cuda_runtime.h>
#include <math.h>

__device__ inline void fill_nan_row(float* ptr, int len) {
    const float nanv = nanf("");
    for (int i = 0; i < len; ++i) ptr[i] = nanv;
}

// Batch kernel using host-precomputed per-bar terms shared across rows.
// Arguments:
//  - plus_dm, minus_dm, tr: length = series_len; valid from first_valid+1 ..
//  - carry: length = series_len; 1 => carry-forward this bar, 0 => normal
//  - periods: per-row period (length = n_combos)
//  - series_len, n_combos, first_valid: sizes and prefix index
//  - out: row-major [n_combos x series_len]
extern "C" __global__
void dx_batch_f32(const double* __restrict__ plus_dm,
                  const double* __restrict__ minus_dm,
                  const double* __restrict__ tr,
                  const unsigned char* __restrict__ carry,
                  const int* __restrict__ periods,
                  int series_len,
                  int n_combos,
                  int first_valid,
                  float* __restrict__ out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos) return;

    float* dst = out + row * series_len;
    fill_nan_row(dst, series_len);

    const int p = periods[row];
    if (p <= 0) return;
    if (first_valid < 0 || first_valid + 1 >= series_len) return;

    const int i0 = first_valid;
    const int warm_needed = p - 1; // number of accumulation steps needed

    double s_plus = 0.0;
    double s_minus = 0.0;
    double s_tr = 0.0;
    int init_count = 0;

    for (int i = i0 + 1; i < series_len; ++i) {
        if (carry[i] != 0) {
            // carry-forward last value (or NaN if none yet)
            dst[i] = (i > 0) ? dst[i - 1] : nanf("");
            continue;
        }

        const double pdm = plus_dm[i];
        const double mdm = minus_dm[i];
        const double t   = tr[i];

        if (init_count < warm_needed) {
            s_plus  += pdm;
            s_minus += mdm;
            s_tr    += t;
            init_count += 1;
            if (init_count == warm_needed) {
                const double plus_di  = (s_tr != 0.0) ? ((s_plus  / s_tr) * 100.0) : 0.0;
                const double minus_di = (s_tr != 0.0) ? ((s_minus / s_tr) * 100.0) : 0.0;
                const double sum_di = plus_di + minus_di;
                const double dx = (sum_di != 0.0) ? (fabs(plus_di - minus_di) / sum_di) * 100.0 : 0.0;
                dst[i] = (float)dx;
            }
            continue;
        }

        // Wilder recurrence
        const double rp = 1.0 / (double)p;
        s_plus  = s_plus  - (s_plus  * rp) + pdm;
        s_minus = s_minus - (s_minus * rp) + mdm;
        s_tr    = s_tr    - (s_tr    * rp) + t;

        const double plus_di  = (s_tr != 0.0) ? ((s_plus  / s_tr) * 100.0) : 0.0;
        const double minus_di = (s_tr != 0.0) ? ((s_minus / s_tr) * 100.0) : 0.0;
        const double sum_di = plus_di + minus_di;
        if (sum_di != 0.0) {
            const double dx = (fabs(plus_di - minus_di) / sum_di) * 100.0;
            dst[i] = (float)dx;
        } else {
            dst[i] = (i > 0) ? dst[i - 1] : nanf("");
        }
    }
}

// Many-series, one param (time-major): compute DM/TR on the fly per series.
extern "C" __global__
void dx_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    // Initialize column with NaNs
    for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nanf("");
    if (period <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv + 1 >= rows) return;

    auto at = [&](int t) { return t * cols + s; };

    const int warm_needed = period - 1;
    double s_plus = 0.0, s_minus = 0.0, s_tr = 0.0;
    int init_count = 0;

    double prev_h = (double)high_tm[at(fv)];
    double prev_l = (double)low_tm[at(fv)];
    double prev_c = (double)close_tm[at(fv)];

    for (int t = fv + 1; t < rows; ++t) {
        const double ch = (double)high_tm[at(t)];
        const double cl = (double)low_tm[at(t)];
        const double cc = (double)close_tm[at(t)];
        if (isnan(ch) || isnan(cl) || isnan(cc)) {
            out_tm[at(t)] = (t > 0) ? out_tm[at(t - 1)] : nanf("");
            prev_h = ch; prev_l = cl; prev_c = cc;
            continue;
        }
        // If previous bar was NaN (after a NaN carry), treat this bar as a new seed
        if (isnan(prev_h) || isnan(prev_l) || isnan(prev_c)) {
            prev_h = ch; prev_l = cl; prev_c = cc; // no output this step
            continue;
        }
        const double up = ch - prev_h;
        const double dn = prev_l - cl;
        const double pdm = (up > 0.0 && up > dn) ? up : 0.0;
        const double mdm = (dn > 0.0 && dn > up) ? dn : 0.0;
        const double tr1 = ch - cl;
        const double tr2 = fabs(ch - prev_c);
        const double tr3 = fabs(cl - prev_c);
        const double tmax = fmax(fmax(tr1, tr2), tr3);

        if (init_count < warm_needed) {
            s_plus  += pdm;
            s_minus += mdm;
            s_tr    += tmax;
            init_count += 1;
            if (init_count == warm_needed) {
                const double plus_di  = (s_tr != 0.0) ? ((s_plus  / s_tr) * 100.0) : 0.0;
                const double minus_di = (s_tr != 0.0) ? ((s_minus / s_tr) * 100.0) : 0.0;
                const double sum_di = plus_di + minus_di;
                const double dx = (sum_di != 0.0) ? (fabs(plus_di - minus_di) / sum_di) * 100.0 : 0.0;
                out_tm[at(t)] = (float)dx;
            }
        } else {
            const double rp = 1.0 / (double)period;
            s_plus  = s_plus  - (s_plus  * rp) + pdm;
            s_minus = s_minus - (s_minus * rp) + mdm;
            s_tr    = s_tr    - (s_tr    * rp) + tmax;
            const double plus_di  = (s_tr != 0.0) ? ((s_plus  / s_tr) * 100.0) : 0.0;
            const double minus_di = (s_tr != 0.0) ? ((s_minus / s_tr) * 100.0) : 0.0;
            const double sum_di = plus_di + minus_di;
            if (sum_di != 0.0) {
                const double dx = (fabs(plus_di - minus_di) / sum_di) * 100.0;
                out_tm[at(t)] = (float)dx;
            } else {
                out_tm[at(t)] = out_tm[at(t - 1)];
            }
        }

        prev_h = ch; prev_l = cl; prev_c = cc;
    }
}

// -------------------- Experimental fast variants (FP32 DS + TR-less DX) --------------------
// Note: These are provided for future integration. They are not invoked by the current
// Rust wrappers, so unit tests remain bound to the reference path above. The intent is to
// enable evaluation/benching without changing host code.

struct dsf32 { float hi, lo; };
__device__ __forceinline__ void two_sum_f(float a, float b, float& s, float& err) {
    s = a + b; float bb = s - a; err = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void renorm_f(float& hi, float& lo) {
    float t = hi + lo; lo = lo - (t - hi); hi = t;
}
__device__ __forceinline__ void ds_add_inplace_f(dsf32& s, float x) {
    float sum, e; two_sum_f(s.hi, x, sum, e); s.lo += e; renorm_f(s.hi, s.lo);
}
__device__ __forceinline__ void ds_scale_add_inplace_f(dsf32& s, float a, float x) {
    float p = s.hi * a; float pe = fmaf(a, s.hi, -p); pe += s.lo * a; float sum, e2; two_sum_f(p, x, sum, e2); s.hi = sum; s.lo = pe + e2; renorm_f(s.hi, s.lo);
}

extern "C" __global__
void dx_batch_f32_fast(const double* __restrict__ plus_dm,
                       const double* __restrict__ minus_dm,
                       const double* __restrict__ /*tr_unused*/,
                       const unsigned char* __restrict__ carry,
                       const int* __restrict__ periods,
                       int series_len,
                       int n_combos,
                       int first_valid,
                       float* __restrict__ out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos) return;

    float* dst = out + row * series_len;
    fill_nan_row(dst, series_len);

    const int p = periods[row];
    if (p <= 0 || first_valid < 0 || first_valid + 1 >= series_len) return;

    const float rp = 1.0f / (float)p; const float ap = 1.0f - rp;
    int warm_left = p - 1; dsf32 s_plus{0.f,0.f}, s_minus{0.f,0.f}; float last_out = nanf("");

    for (int i = first_valid + 1; i < series_len; ++i) {
        if (carry[i]) { dst[i] = last_out; continue; }
        const float pdm = (float)plus_dm[i]; const float mdm = (float)minus_dm[i];
        if (warm_left > 0) {
            ds_add_inplace_f(s_plus,  pdm); ds_add_inplace_f(s_minus, mdm); --warm_left;
            if (warm_left == 0) {
                const float sp = s_plus.hi + s_plus.lo; const float sm = s_minus.hi + s_minus.lo;
                const float denom = sp + sm; const float dx = (denom > 0.f) ? (fabsf(sp - sm) / denom) * 100.f : 0.f;
                last_out = dx; dst[i] = dx;
            }
            continue;
        }
        ds_scale_add_inplace_f(s_plus, ap, pdm); ds_scale_add_inplace_f(s_minus, ap, mdm);
        const float sp = s_plus.hi + s_plus.lo; const float sm = s_minus.hi + s_minus.lo; const float denom = sp + sm;
        if (denom > 0.f) { const float dx = (fabsf(sp - sm) / denom) * 100.f; last_out = dx; dst[i] = dx; } else { dst[i] = last_out; }
    }
}

extern "C" __global__
void dx_many_series_one_param_time_major_f32_fast(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= cols || period <= 0) return;
    const int fv = first_valids[s]; if (fv < 0 || fv + 1 >= rows) return;
    for (int t = 0; t < rows; ++t) out_tm[t*cols + s] = nanf("");
    auto idx = [&](int t){ return t*cols + s; };
    int warm_left = period - 1; dsf32 s_plus{0.f,0.f}, s_minus{0.f,0.f}; const float rp = 1.0f/(float)period, ap = 1.0f - rp; float last_out = nanf("");
    float ph = high_tm[idx(fv)], pl = low_tm[idx(fv)], pc = close_tm[idx(fv)];
    for (int t = fv + 1; t < rows; ++t) {
        const int k = idx(t); const float ch = high_tm[k], cl = low_tm[k], cc = close_tm[k];
        if (isnan(ch) || isnan(cl) || isnan(cc)) { out_tm[k] = last_out; ph = ch; pl = cl; pc = cc; continue; }
        if (isnan(ph) || isnan(pl) || isnan(pc)) { ph = ch; pl = cl; pc = cc; continue; }
        const float up = ch - ph, dn = pl - cl; const float pdm = (up > 0.f && up > dn) ? up : 0.f; const float mdm = (dn > 0.f && dn > up) ? dn : 0.f;
        if (warm_left > 0) { ds_add_inplace_f(s_plus, pdm); ds_add_inplace_f(s_minus, mdm); --warm_left; if (warm_left == 0) { const float sp = s_plus.hi + s_plus.lo; const float sm = s_minus.hi + s_minus.lo; const float denom = sp + sm; const float dx = (denom > 0.f) ? (fabsf(sp - sm)/denom)*100.f : 0.f; last_out = dx; out_tm[k] = dx; } }
        else { ds_scale_add_inplace_f(s_plus, ap, pdm); ds_scale_add_inplace_f(s_minus, ap, mdm); const float sp = s_plus.hi + s_plus.lo; const float sm = s_minus.hi + s_minus.lo; const float denom = sp + sm; if (denom > 0.f) { const float dx = (fabsf(sp - sm)/denom)*100.f; last_out = dx; out_tm[k] = dx; } else { out_tm[k] = last_out; } }
        ph = ch; pl = cl; pc = cc;
    }
}
