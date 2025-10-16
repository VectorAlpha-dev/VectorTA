// CUDA kernels for the Band-Pass indicator (Ehlers-style two-stage filter).
//
// Batch kernel consumes a precomputed High-Pass (HP) matrix (rows × len)
// and evaluates the band-pass recurrence per-parameter row, then performs
// peak-based normalization and a trigger high-pass on the normalized output.
//
// Many-series kernel consumes a time-major HP buffer (cols × rows) for a
// single (period, bandwidth) set shared across all series.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// Utility: check finite
static __forceinline__ __device__ bool is_finite_f32(float x) {
    return !isnan(x) && !isinf(x);
}

// High-pass coefficients helper (same form as highpass_kernel.cu)
static __forceinline__ __device__ void hpf_coeffs_from_period(int period, double &c, double &oma, bool &ok) {
    ok = false;
    if (period <= 0) return;
    double s, co;
    // theta = 2*pi/period => sincospi(2/period)
    sincospi(2.0 / static_cast<double>(period), &s, &co);
    if (fabs(co) < 1e-12) return;
    const double alpha = 1.0 + ((s - 1.0) / co);
    c   = 1.0 - 0.5 * alpha; // (1 - α/2)
    oma = 1.0 - alpha;       // (1 - α)
    ok = true;
}

// Batch kernel: one-series × many-params, consuming HP rows
extern "C" __global__ void bandpass_batch_from_hp_f32(
    const float* __restrict__ hp,      // shape: hp_rows × len (row-major)
    int hp_rows,
    int len,
    const int* __restrict__ hp_row_idx, // per-combo index into hp rows
    const float* __restrict__ alphas,   // per-combo alpha
    const float* __restrict__ betas,    // per-combo beta (cos term)
    const int* __restrict__ trig_periods, // per-combo trigger highpass period
    int n_combos,
    float* __restrict__ out_bp,   // shape: n_combos × len
    float* __restrict__ out_bpn,  // shape: n_combos × len
    float* __restrict__ out_sig,  // shape: n_combos × len
    float* __restrict__ out_trg   // shape: n_combos × len
) {
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = row0; row < n_combos; row += stride) {
        const int hp_idx = hp_row_idx[row];
        if (hp_idx < 0 || hp_idx >= hp_rows) continue;

        const float *hp_row = hp + static_cast<size_t>(hp_idx) * len;
        float *bp_row  = out_bp  + static_cast<size_t>(row) * len;
        float *bpn_row = out_bpn + static_cast<size_t>(row) * len;
        float *sig_row = out_sig + static_cast<size_t>(row) * len;
        float *trg_row = out_trg + static_cast<size_t>(row) * len;

        // Find first finite HP sample; warmup requires at least two hp samples
        int first_valid_hp = 0;
        while (first_valid_hp < len && !is_finite_f32(hp_row[first_valid_hp])) {
            ++first_valid_hp;
        }
        int warm_bp = first_valid_hp < len ? max(first_valid_hp, 2) : 2;

        // Coefficients for band-pass recurrence
        const double alpha = static_cast<double>(alphas[row]);
        const double beta  = static_cast<double>(betas[row]);
        const double a = 0.5 * (1.0 - alpha);
        const double c = beta * (1.0 + alpha);
        const double d = -alpha;

        // Compute bp sequentially
        if (len > 0) {
            bp_row[0] = hp_row[0];
        }
        if (len > 1) {
            bp_row[1] = hp_row[1];
        }
        if (len > 2) {
            double y_im2 = static_cast<double>(bp_row[0]);
            double y_im1 = static_cast<double>(bp_row[1]);
            for (int i = 2; i < len; ++i) {
                const double hi   = static_cast<double>(hp_row[i]);
                const double him2 = static_cast<double>(hp_row[i - 2]);
                const double delta = hi - him2;
                const double y = fma(d, y_im2, fma(c, y_im1, a * delta));
                bp_row[i] = static_cast<float>(y);
                y_im2 = y_im1;
                y_im1 = y;
            }
        }

        // Warmup NaNs for bp
        for (int i = 0; i < min(warm_bp, len); ++i) {
            bp_row[i] = CUDART_NAN_F;
        }

        // Peak-based normalization into bpn
        for (int i = 0; i < len; ++i) {
            bpn_row[i] = CUDART_NAN_F;
            trg_row[i] = CUDART_NAN_F;
            sig_row[i] = CUDART_NAN_F;
        }
        const float k = 0.991f;
        double peak = 0.0;
        for (int i = warm_bp; i < len; ++i) {
            peak *= static_cast<double>(k);
            const double v = static_cast<double>(bp_row[i]);
            const double av = fabs(v);
            if (av > peak) peak = av;
            bpn_row[i] = (peak != 0.0) ? static_cast<float>(v / peak) : 0.0f;
        }

        // Trigger: high-pass on bpn starting at warm_bp
        int tper = trig_periods[row];
        double hc = 0.0, homa = 0.0; bool ok;
        hpf_coeffs_from_period(tper, hc, homa, ok);
        if (ok && warm_bp < len) {
            double prev_x = static_cast<double>(bpn_row[warm_bp]);
            double prev_y = prev_x;
            trg_row[warm_bp] = static_cast<float>(prev_y);
            for (int i = warm_bp + 1; i < len; ++i) {
                const double x = static_cast<double>(bpn_row[i]);
                const double y = fma(homa, prev_y, hc * (x - prev_x));
                trg_row[i] = static_cast<float>(y);
                prev_x = x;
                prev_y = y;
            }
        }

        // Signal: compare bpn vs trigger after warm region
        int first_tr = warm_bp;
        while (first_tr < len && !is_finite_f32(trg_row[first_tr])) {
            ++first_tr;
        }
        const int warm_sig = max(warm_bp, first_tr);
        for (int i = warm_sig; i < len; ++i) {
            const float bn = bpn_row[i];
            const float tr = trg_row[i];
            float s = 0.0f;
            if (bn < tr) s = 1.0f; else if (bn > tr) s = -1.0f;
            sig_row[i] = s;
        }
    }
}

// Many-series × one-param, time-major HP input.
extern "C" __global__ void bandpass_many_series_one_param_time_major_from_hp_f32(
    const float* __restrict__ hp_tm, // shape: rows × cols, time-major (t-major): idx = t*cols + s
    int cols,                        // number of series (columns)
    int rows,                        // series length (rows)
    float alpha_f,
    float beta_f,
    int trig_period,
    float* __restrict__ out_bp_tm,
    float* __restrict__ out_bpn_tm,
    float* __restrict__ out_sig_tm,
    float* __restrict__ out_trg_tm
) {
    if (cols <= 0 || rows <= 0) return;

    const double alpha = static_cast<double>(alpha_f);
    const double beta  = static_cast<double>(beta_f);
    const double a = 0.5 * (1.0 - alpha);
    const double c = beta * (1.0 + alpha);
    const double d = -alpha;

    // Trigger coefficients
    double hc = 0.0, homa = 0.0; bool ok;
    hpf_coeffs_from_period(trig_period, hc, homa, ok);

    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += blockDim.x * gridDim.x) {
        // Compute over time-major layout for series s
        // Helper to index time-major buffer
        auto at = [&](const float* base, int t) -> float { return base[t * cols + s]; };
        auto out_at = [&](float* base, int t) -> float& { return base[t * cols + s]; };

        // Find first finite in hp for this series
        int first_valid_hp = 0;
        while (first_valid_hp < rows && !is_finite_f32(at(hp_tm, first_valid_hp))) {
            ++first_valid_hp;
        }
        int warm_bp = first_valid_hp < rows ? max(first_valid_hp, 2) : 2;

        // bp seeds
        if (rows > 0) out_at(out_bp_tm, 0) = at(hp_tm, 0);
        if (rows > 1) out_at(out_bp_tm, 1) = at(hp_tm, 1);
        if (rows > 2) {
            double y_im2 = static_cast<double>(at(out_bp_tm, 0));
            double y_im1 = static_cast<double>(at(out_bp_tm, 1));
            for (int t = 2; t < rows; ++t) {
                const double hi   = static_cast<double>(at(hp_tm, t));
                const double him2 = static_cast<double>(at(hp_tm, t - 2));
                const double y = fma(d, y_im2, fma(c, y_im1, a * (hi - him2)));
                out_at(out_bp_tm, t) = static_cast<float>(y);
                y_im2 = y_im1;
                y_im1 = y;
            }
        }

        for (int t = 0; t < min(warm_bp, rows); ++t) out_at(out_bp_tm, t) = CUDART_NAN_F;
        for (int t = 0; t < rows; ++t) {
            out_at(out_bpn_tm, t) = CUDART_NAN_F;
            out_at(out_trg_tm, t) = CUDART_NAN_F;
            out_at(out_sig_tm, t) = CUDART_NAN_F;
        }

        // Normalize
        const float k = 0.991f;
        double peak = 0.0;
        for (int t = warm_bp; t < rows; ++t) {
            peak *= static_cast<double>(k);
            const double v = static_cast<double>(at(out_bp_tm, t));
            const double av = fabs(v);
            if (av > peak) peak = av;
            out_at(out_bpn_tm, t) = (peak != 0.0) ? static_cast<float>(v / peak) : 0.0f;
        }

        // Trigger
        if (ok && warm_bp < rows) {
            double prev_x = static_cast<double>(at(out_bpn_tm, warm_bp));
            double prev_y = prev_x;
            out_at(out_trg_tm, warm_bp) = static_cast<float>(prev_y);
            for (int t = warm_bp + 1; t < rows; ++t) {
                const double x = static_cast<double>(at(out_bpn_tm, t));
                const double y = fma(homa, prev_y, hc * (x - prev_x));
                out_at(out_trg_tm, t) = static_cast<float>(y);
                prev_x = x;
                prev_y = y;
            }
        }

        int first_tr = warm_bp;
        while (first_tr < rows && !is_finite_f32(at(out_trg_tm, first_tr))) ++first_tr;
        const int warm_sig = max(warm_bp, first_tr);
        for (int t = warm_sig; t < rows; ++t) {
            const float bn = at(out_bpn_tm, t);
            const float tr = at(out_trg_tm, t);
            float sgn = 0.0f;
            if (bn < tr) sgn = 1.0f; else if (bn > tr) sgn = -1.0f;
            out_at(out_sig_tm, t) = sgn;
        }
    }
}

