// CUDA kernels for Modified God Mode (composite oscillator)
//
// Category: Recurrence/IIR. Each thread performs a sequential scan over time
// carrying state for EMA/RSI/TSI/Laguerre and small rings for RSI momentum and
// MFI when volume is enabled. Outputs are row-major with shape (rows, len).
//
// Warmup/NaN policy matches scalar reference:
//   warm = first_valid + max(n1, n2, n3) - 1
//   wavetrend[0..warm) = NaN
//   signal[0..(warm+6-1)) = NaN
//   histogram[0..(warm+6-1)) = NaN

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline bool is_finite(float x) { return !isnan(x) && !isinf(x); }

// Safe EMA update with fused multiply-add style
__device__ inline float ema_step(float x, float prev, float alpha, float beta) {
    return fmaf(beta, prev, alpha * x);
}

// Laguerre RSI one step update (alpha fixed = 0.7 as in scalar)
struct LaguerreRSI {
    float l0, l1, l2, l3;
    __device__ LaguerreRSI() : l0(0.f), l1(0.f), l2(0.f), l3(0.f) {}
    __device__ float update(float x) {
        const float alpha = 0.7f;
        const float one_m = 1.0f - alpha;
        float prev_l0 = l0;
        l0 = alpha * x + one_m * prev_l0;
        float prev_l1 = l1;
        l1 = -one_m * l0 + prev_l0 + one_m * prev_l1;
        float prev_l2 = l2;
        l2 = -one_m * l1 + prev_l1 + one_m * prev_l2;
        l3 = -one_m * l2 + prev_l2 + one_m * l3;
        float cu = fmaxf(l0 - l1, 0.f) + fmaxf(l1 - l2, 0.f) + fmaxf(l2 - l3, 0.f);
        float cd = fmaxf(l1 - l0, 0.f) + fmaxf(l2 - l1, 0.f) + fmaxf(l3 - l2, 0.f);
        float denom = cu + cd;
        if (denom != 0.f && isfinite(denom)) {
            return 100.f * (cu / denom);
        }
        return CUDART_NAN_F;
    }
};

// Close-only Williams %R over window win ending at idx (inclusive)
__device__ inline float willr_close_only(const float* __restrict__ close,
                                         int idx, int win) {
    if (win <= 0 || idx + 1 < win) return CUDART_NAN_F;
    int start = idx + 1 - win;
    float hi = -CUDART_INF_F;
    float lo = CUDART_INF_F;
    for (int j = start; j <= idx; ++j) {
        float v = close[j];
        if (v > hi) hi = v;
        if (v < lo) lo = v;
    }
    float rng = hi - lo;
    if (rng == 0.f) return CUDART_NAN_F;
    return 60.f * (close[idx] - hi) / rng + 80.f;
}
} // namespace

// Mapping matches Rust enum order:
// 0: Godmode, 1: Tradition, 2: GodmodeMg, 3: TraditionMg
extern "C" __global__ void mod_god_mode_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ volume, // may be null when unused
    int len,
    int first_valid,
    int n_rows,
    const int* __restrict__ n1s,
    const int* __restrict__ n2s,
    const int* __restrict__ n3s,
    const int* __restrict__ modes,
    const int use_volume_flag,
    float* __restrict__ wavetrend_out,
    float* __restrict__ signal_out,
    float* __restrict__ histogram_out) {
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Conservative ring bounds for CBCI (n2) and MFI (n3). Typical values are small (<= 64).
    const int MAX_RING = 2048; // safeguard; larger windows will gracefully mod into this size

    for (int row = row0; row < n_rows; row += stride) {
        const int n1 = n1s[row];
        const int n2 = n2s[row];
        const int n3 = n3s[row];
        const int mode = modes[row];

        float* wt_row = wavetrend_out + (size_t)row * len;
        float* sig_row = signal_out + (size_t)row * len;
        float* hist_row = histogram_out + (size_t)row * len;

        for (int i = 0; i < len; ++i) {
            wt_row[i] = CUDART_NAN_F;
            sig_row[i] = CUDART_NAN_F;
            hist_row[i] = CUDART_NAN_F;
        }

        const float a1 = 2.f / (float(n1) + 1.f);
        const float b1 = 1.f - a1;
        const float a2 = 2.f / (float(n2) + 1.f);
        const float b2 = 1.f - a2;
        const float a3 = 2.f / (float(n3) + 1.f);
        const float b3 = 1.f - a3;

        int warm = first_valid + max(n1, max(n2, n3)) - 1;
        if (warm < 0) warm = 0;
        if (warm > len) warm = len;
        const int sig_start = warm + 6 - 1;

        // TCI state
        float ema1_c = 0.f, ema2_abs = 0.f, ema3_ci = 0.f;
        bool seed_e1 = false, seed_e2 = false, seed_e3 = false;

        // RSI (Wilder)
        float rs_avg_gain = 0.f, rs_avg_loss = 0.f; bool rsi_seeded = false; int rs_init = 0;
        float prev_close = (first_valid < len) ? close[first_valid] : 0.f;

        // Laguerre RSI
        LaguerreRSI lrsi;

        // CBCI helpers
        float rsi_ring[MAX_RING];
        for (int i = 0; i < MAX_RING; ++i) rsi_ring[i] = CUDART_NAN_F;
        int rsi_head = 0;
        float rsi_ema = 0.f; bool rsi_ema_seed = false;

        // MFI (n3 ring)
        float mf_ring_mf[MAX_RING];
        signed char mf_ring_sgn[MAX_RING];
        for (int i = 0; i < MAX_RING; ++i) { mf_ring_mf[i] = 0.f; mf_ring_sgn[i] = 0; }
        float mf_pos_sum = 0.f, mf_neg_sum = 0.f; int mf_head = 0; bool tp_has_prev = false; float tp_prev = 0.f;

        // CSI/TSI and CSI_MG state (per thread)
        float tsi_m_s = 0.f, tsi_m_l = 0.f, tsi_a_s = 0.f, tsi_a_l = 0.f; bool tsi_seed_s = false, tsi_seed_l = false;
        float csi_num_e1 = 0.f, csi_num_e2 = 0.f, csi_den_e1 = 0.f, csi_den_e2 = 0.f; bool csi_seed_e1 = false, csi_seed_e2 = false;

        for (int i = first_valid; i < len; ++i) {
            const float c = close[i];

            // === TCI chain ===
            if (!seed_e1) { ema1_c = c; seed_e1 = true; }
            else { ema1_c = ema_step(c, ema1_c, a1, b1); }
            float abs_dev = fabsf(c - ema1_c);
            if (!seed_e2) { ema2_abs = abs_dev; seed_e2 = true; }
            else { ema2_abs = ema_step(abs_dev, ema2_abs, a1, b1); }
            float tci_val = CUDART_NAN_F;
            if (ema2_abs != 0.f && isfinite(ema2_abs)) {
                float ci = (c - ema1_c) / (0.025f * ema2_abs);
                if (!seed_e3) { ema3_ci = ci; seed_e3 = true; }
                else { ema3_ci = ema_step(ci, ema3_ci, a2, b2); }
                tci_val = ema3_ci + 50.f;
            }

            // === Wilder RSI ===
            float rsi_val = CUDART_NAN_F;
            if (i == first_valid) {
                rs_avg_gain = 0.f; rs_avg_loss = 0.f; rs_init = 0; rsi_seeded = false;
            } else {
                float ch = c - prev_close;
                float gain = fmaxf(ch, 0.f);
                float loss = fmaxf(-ch, 0.f);
                if (!rsi_seeded) {
                    rs_init += 1;
                    rs_avg_gain += gain; rs_avg_loss += loss;
                    if (rs_init >= n3) {
                        rs_avg_gain /= (float)n3; rs_avg_loss /= (float)n3;
                        rsi_seeded = true;
                        float rs = (rs_avg_loss == 0.f) ? CUDART_INF_F : (rs_avg_gain / rs_avg_loss);
                        rsi_val = 100.f - 100.f / (1.f + rs);
                    }
                } else {
                    rs_avg_gain = ((rs_avg_gain * (float)(n3 - 1)) + gain) / (float)n3;
                    rs_avg_loss = ((rs_avg_loss * (float)(n3 - 1)) + loss) / (float)n3;
                    float rs = (rs_avg_loss == 0.f) ? CUDART_INF_F : (rs_avg_gain / rs_avg_loss);
                    rsi_val = 100.f - 100.f / (1.f + rs);
                }
            }

            // Laguerre RSI
            float lrsi_val = lrsi.update(c);

            // MF (either MFI with volume or RSI-as-MF)
            float mf_val = CUDART_NAN_F;
            if (use_volume_flag && volume != nullptr) {
                float tp = (high[i] + low[i] + c) * (1.f / 3.f);
                if (tp_has_prev) {
                    signed char sign = (tp > tp_prev) ? 1 : ((tp < tp_prev) ? -1 : 0);
                    float mf_raw = tp * volume[i];
                    if (rsi_seeded) {
                        int old_idx = mf_head % MAX_RING;
                        float old_mf = mf_ring_mf[old_idx];
                        signed char old_sign = mf_ring_sgn[old_idx];
                        if (old_sign > 0) mf_pos_sum -= old_mf;
                        else if (old_sign < 0) mf_neg_sum -= old_mf;
                    }
                    int idx = mf_head % MAX_RING;
                    mf_ring_mf[idx] = mf_raw;
                    mf_ring_sgn[idx] = sign;
                    if (sign > 0) mf_pos_sum += mf_raw; else if (sign < 0) mf_neg_sum += mf_raw;
                    mf_head = (mf_head + 1);
                    if (rsi_seeded) {
                        mf_val = (mf_neg_sum == 0.f) ? 100.f : (100.f - 100.f / (1.f + (mf_pos_sum / mf_neg_sum)));
                    }
                }
                tp_prev = tp; tp_has_prev = true;
            } else {
                mf_val = rsi_val;
            }

            // CBCI = RSI momentum over n2 + EMA(RSI, n3)
            float cbci_val = CUDART_NAN_F;
            if (rsi_seeded) {
                int oldi = rsi_head % MAX_RING;
                float old_rsi = rsi_ring[oldi];
                rsi_ring[oldi] = rsi_val; // store current
                rsi_head = (rsi_head + 1);
                float mom = (is_finite(old_rsi) && is_finite(rsi_val)) ? (rsi_val - old_rsi) : CUDART_NAN_F;
                if (!rsi_ema_seed && is_finite(rsi_val)) { rsi_ema = rsi_val; rsi_ema_seed = true; }
                else if (is_finite(rsi_val)) { rsi_ema = ema_step(rsi_val, rsi_ema, a3, b3); }
                if (is_finite(mom) && rsi_ema_seed) cbci_val = mom + rsi_ema;
            }

            // CSI and CSI_MG
            float csi_val = CUDART_NAN_F;
            float csi_mg_val = CUDART_NAN_F;
            // --- CSI (TSI basic) ---
            if (i > first_valid) {
                float mom = c - prev_close; float am = fabsf(mom);
                if (!tsi_seed_s) { tsi_m_s = mom; tsi_a_s = am; tsi_seed_s = true; }
                else { tsi_m_s = ema_step(mom, tsi_m_s, a1, b1); tsi_a_s = ema_step(am, tsi_a_s, a1, b1); }
                if (!tsi_seed_l && tsi_seed_s) { tsi_m_l = tsi_m_s; tsi_a_l = tsi_a_s; tsi_seed_l = true; }
                else if (tsi_seed_l) { tsi_m_l = ema_step(tsi_m_s, tsi_m_l, a2, b2); tsi_a_l = ema_step(tsi_a_s, tsi_a_l, a2, b2); }
                if (tsi_seed_l && tsi_a_l != 0.f && isfinite(tsi_a_l) && is_finite(rsi_val)) {
                    float tsi = 100.f * (tsi_m_l / tsi_a_l);
                    csi_val = 0.5f * (rsi_val + (tsi * 0.5f + 50.f));
                }
            }

            // --- CSI_MG (normalized price change with EMA chains) ---
            if (i > first_valid) {
                float pc = c - prev_close; float apc = fabsf(pc);
                if (!csi_seed_e1) { csi_num_e1 = pc; csi_den_e1 = apc; csi_seed_e1 = true; }
                else { csi_num_e1 = ema_step(pc, csi_num_e1, a1, b1); csi_den_e1 = ema_step(apc, csi_den_e1, a1, b1); }
                if (!csi_seed_e2 && csi_seed_e1) { csi_num_e2 = csi_num_e1; csi_den_e2 = csi_den_e1; csi_seed_e2 = true; }
                else if (csi_seed_e2) { csi_num_e2 = ema_step(csi_num_e1, csi_num_e2, a2, b2); csi_den_e2 = ema_step(csi_den_e1, csi_den_e2, a2, b2); }
                if (csi_seed_e2 && csi_den_e2 != 0.f && isfinite(csi_den_e2) && is_finite(rsi_val)) {
                    float pc_norm = 50.f * (csi_num_e2 / csi_den_e2) + 50.f;
                    csi_mg_val = 0.5f * (rsi_val + pc_norm);
                }
            }

            // Compose wavetrend depending on mode; write only at/after warm
            if (i >= warm) {
                float sum = 0.f; int cnt = 0;
                if (mode == 0) { // Godmode: avg(tci, csi, mf, willy)
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(csi_val)) { sum += csi_val; cnt++; }
                    if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                    float wil = willr_close_only(close, i, n2);
                    if (is_finite(wil)) { sum += wil; cnt++; }
                } else if (mode == 1) { // Tradition: avg(tci, mf, rsi)
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                    if (is_finite(rsi_val)) { sum += rsi_val; cnt++; }
                } else if (mode == 2) { // GodmodeMg: avg(tci, csi_mg, mf, willy, cbci, lrsi)
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(csi_mg_val)) { sum += csi_mg_val; cnt++; }
                    if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                    float wil = willr_close_only(close, i, n2);
                    if (is_finite(wil)) { sum += wil; cnt++; }
                    if (is_finite(cbci_val)) { sum += cbci_val; cnt++; }
                    if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; }
                } else { // TraditionMg: avg(tci, mf, rsi, cbci, lrsi)
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                    if (is_finite(rsi_val)) { sum += rsi_val; cnt++; }
                    if (is_finite(cbci_val)) { sum += cbci_val; cnt++; }
                    if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; }
                }
                if (cnt > 0) {
                    float wt = sum / (float)cnt;
                    wt_row[i] = wt;
                // Signal SMA(6)
                if (i >= sig_start) {
                    float s = 0.f; int ready = 1;
                    for (int k = 0; k < 6; ++k) {
                        float x = wt_row[i - (6 - 1) + k];
                        if (!is_finite(x)) { ready = 0; break; }
                        s += x;
                    }
                    if (ready) {
                        float sig = s / 6.f;
                        sig_row[i] = sig;
                        // Histogram EMA over d = (wt - sig)*2 + 50
                        float d = (wt - sig) * 2.f + 50.f;
                        if (!is_finite(hist_row[i - 1])) {
                            hist_row[i] = d; // seed at first valid point
                        } else {
                            hist_row[i] = ema_step(d, hist_row[i - 1], a3, b3);
                        }
                    }
                }
            }
            prev_close = c;
        }
        // NaN prefixes already set by initialization
    }
}

// Many-series (time-major), one param
extern "C" __global__ void mod_god_mode_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols, // number of series
    int rows, // time length
    int n1, int n2, int n3, int mode, int use_volume_flag,
    float* __restrict__ wt_tm,
    float* __restrict__ sig_tm,
    float* __restrict__ hist_tm) {
    int s = blockIdx.x; // one block per series; thread 0 performs the scan
    if (s >= cols) return;
    if (threadIdx.x != 0) return;

    const int MAX_RING = 2048;

    auto idx = [cols](int t, int s) { return t * cols + s; };

    // Find first valid close
    int first_valid = 0; bool found = false;
    for (int t = 0; t < rows; ++t) {
        float v = close_tm[idx(t, s)];
        if (is_finite(v)) { first_valid = t; found = true; break; }
    }
    if (!found) return;

    const float a1 = 2.f / (float(n1) + 1.f);
    const float b1 = 1.f - a1;
    const float a2 = 2.f / (float(n2) + 1.f);
    const float b2 = 1.f - a2;
    const float a3 = 2.f / (float(n3) + 1.f);
    const float b3 = 1.f - a3;

    int warm = first_valid + max(n1, max(n2, n3)) - 1;
    if (warm < 0) warm = 0;
    if (warm > rows) warm = rows;
    const int sig_start = warm + 6 - 1;

    // Init outputs to NaN
    for (int t = 0; t < rows; ++t) {
        wt_tm[idx(t, s)] = CUDART_NAN_F;
        sig_tm[idx(t, s)] = CUDART_NAN_F;
        hist_tm[idx(t, s)] = CUDART_NAN_F;
    }

    float ema1_c = 0.f, ema2_abs = 0.f, ema3_ci = 0.f; bool seed_e1=false, seed_e2=false, seed_e3=false;
    float rs_avg_gain=0.f, rs_avg_loss=0.f; bool rsi_seeded=false; int rs_init=0;
    float prev_close = close_tm[idx(first_valid, s)];
    LaguerreRSI lrsi;

    float rsi_ring[MAX_RING]; for (int i=0;i<MAX_RING;++i) rsi_ring[i]=CUDART_NAN_F;
    int rsi_head=0; float rsi_ema=0.f; bool rsi_ema_seed=false;
    float mf_ring_mf[MAX_RING]; signed char mf_ring_sgn[MAX_RING];
    for (int i=0;i<MAX_RING;++i){ mf_ring_mf[i]=0.f; mf_ring_sgn[i]=0; }
    float mf_pos_sum=0.f, mf_neg_sum=0.f; int mf_head=0; bool tp_has_prev=false; float tp_prev=0.f;

    // CSI/TSI and CSI_MG state (per thread)
    float tsi_m_s=0.f, tsi_m_l=0.f, tsi_a_s=0.f, tsi_a_l=0.f; bool tsi_seed_s=false, tsi_seed_l=false;
    float csi_num_e1=0.f, csi_num_e2=0.f, csi_den_e1=0.f, csi_den_e2=0.f; bool csi_seed_e1=false, csi_seed_e2=false;

    for (int t = first_valid; t < rows; ++t) {
        float c = close_tm[idx(t, s)];
        // TCI
        if (!seed_e1) { ema1_c=c; seed_e1=true; } else { ema1_c = ema_step(c, ema1_c, a1, b1); }
        float abs_dev = fabsf(c - ema1_c);
        if (!seed_e2) { ema2_abs=abs_dev; seed_e2=true; } else { ema2_abs = ema_step(abs_dev, ema2_abs, a1, b1); }
        float tci_val = CUDART_NAN_F;
        if (ema2_abs != 0.f && isfinite(ema2_abs)) {
            float ci = (c - ema1_c) / (0.025f * ema2_abs);
            if (!seed_e3) { ema3_ci=ci; seed_e3=true; } else { ema3_ci = ema_step(ci, ema3_ci, a2, b2); }
            tci_val = ema3_ci + 50.f;
        }
        // RSI Wilder
        float rsi_val = CUDART_NAN_F;
        if (t == first_valid) { rs_avg_gain=0.f; rs_avg_loss=0.f; rs_init=0; }
        else {
            float ch = c - prev_close; float gain=fmaxf(ch,0.f), loss=fmaxf(-ch,0.f);
            if (!rsi_seeded) {
                rs_init++; rs_avg_gain+=gain; rs_avg_loss+=loss;
                if (rs_init >= n3) { rs_avg_gain/=(float)n3; rs_avg_loss/=(float)n3; rsi_seeded=true; float rs=(rs_avg_loss==0.f)?CUDART_INF_F:(rs_avg_gain/rs_avg_loss); rsi_val=100.f - 100.f/(1.f+rs); }
            } else {
                rs_avg_gain=((rs_avg_gain*(float)(n3-1))+gain)/(float)n3;
                rs_avg_loss=((rs_avg_loss*(float)(n3-1))+loss)/(float)n3;
                float rs=(rs_avg_loss==0.f)?CUDART_INF_F:(rs_avg_gain/rs_avg_loss); rsi_val=100.f - 100.f/(1.f+rs);
            }
        }
        float lrsi_val = lrsi.update(c);

        float mf_val = CUDART_NAN_F;
        if (use_volume_flag && volume_tm != nullptr) {
            float tp = (high_tm[idx(t,s)] + low_tm[idx(t,s)] + c) * (1.f/3.f);
            if (tp_has_prev) {
                signed char sign = (tp > tp_prev) ? 1 : ((tp < tp_prev) ? -1 : 0);
                float mf_raw = tp * volume_tm[idx(t,s)];
                if (rsi_seeded) {
                    int old = mf_head % MAX_RING; float old_mf = mf_ring_mf[old]; signed char old_s = mf_ring_sgn[old];
                    if (old_s > 0) mf_pos_sum -= old_mf; else if (old_s < 0) mf_neg_sum -= old_mf;
                }
                int cur = mf_head % MAX_RING; mf_ring_mf[cur]=mf_raw; mf_ring_sgn[cur]=sign; if (sign>0) mf_pos_sum += mf_raw; else if (sign<0) mf_neg_sum += mf_raw; mf_head++;
                if (rsi_seeded) mf_val = (mf_neg_sum == 0.f) ? 100.f : (100.f - 100.f/(1.f + (mf_pos_sum/mf_neg_sum)));
            }
            tp_prev = tp; tp_has_prev = true;
        } else { mf_val = rsi_val; }

        // CBCI
        float cbci_val = CUDART_NAN_F;
        if (rsi_seeded) {
            int old = rsi_head % MAX_RING; float old_r = rsi_ring[old]; rsi_ring[old] = rsi_val; rsi_head++;
            float mom = (is_finite(old_r) && is_finite(rsi_val)) ? (rsi_val - old_r) : CUDART_NAN_F;
            if (!rsi_ema_seed && is_finite(rsi_val)) { rsi_ema = rsi_val; rsi_ema_seed = true; }
            else if (is_finite(rsi_val)) { rsi_ema = ema_step(rsi_val, rsi_ema, a3, b3); }
            if (is_finite(mom) && rsi_ema_seed) cbci_val = mom + rsi_ema;
        }

        float csi_val = CUDART_NAN_F, csi_mg_val = CUDART_NAN_F;
        if (t > first_valid) {
            float mom = c - prev_close; float am = fabsf(mom);
            // TSI chains
            if (!tsi_seed_s) { tsi_m_s=mom; tsi_a_s=am; tsi_seed_s=true; }
            else { tsi_m_s = ema_step(mom, tsi_m_s, a1, b1); tsi_a_s = ema_step(am, tsi_a_s, a1, b1); }
            if (!tsi_seed_l && tsi_seed_s) { tsi_m_l=tsi_m_s; tsi_a_l=tsi_a_s; tsi_seed_l=true; }
            else if (tsi_seed_l) { tsi_m_l = ema_step(tsi_m_s, tsi_m_l, a2, b2); tsi_a_l = ema_step(tsi_a_s, tsi_a_l, a2, b2); }
            if (tsi_seed_l && tsi_a_l != 0.f && isfinite(tsi_a_l) && is_finite(rsi_val)) {
                float tsi = 100.f * (tsi_m_l / tsi_a_l); csi_val = 0.5f * (rsi_val + (tsi * 0.5f + 50.f));
            }
            // CSI_MG normalized PC
            if (!csi_seed_e1) { csi_num_e1=mom; csi_den_e1=am; csi_seed_e1=true; }
            else { csi_num_e1 = ema_step(mom, csi_num_e1, a1, b1); csi_den_e1 = ema_step(am, csi_den_e1, a1, b1); }
            if (!csi_seed_e2 && csi_seed_e1) { csi_num_e2=csi_num_e1; csi_den_e2=csi_den_e1; csi_seed_e2=true; }
            else if (csi_seed_e2) { csi_num_e2 = ema_step(csi_num_e1, csi_num_e2, a2, b2); csi_den_e2 = ema_step(csi_den_e1, csi_den_e2, a2, b2); }
            if (csi_seed_e2 && csi_den_e2 != 0.f && isfinite(csi_den_e2) && is_finite(rsi_val)) {
                float pc_norm = 50.f * (csi_num_e2 / csi_den_e2) + 50.f; csi_mg_val = 0.5f * (rsi_val + pc_norm);
            }
        }

        float sum = 0.f; int cnt = 0;
        if (t >= warm) {
            if (mode == 0) { // Godmode with time-major williams
                if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                if (is_finite(csi_val)) { sum += csi_val; cnt++; }
                if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                float hi = -CUDART_INF_F, lo = CUDART_INF_F; bool ok = (t + 1 >= n2);
                if (ok) {
                    int start = t + 1 - n2;
                    for (int j = start; j <= t; ++j) { float v = close_tm[idx(j, s)]; if (v > hi) hi = v; if (v < lo) lo = v; }
                    float rng = hi - lo; if (rng != 0.f) { float wil = 60.f * (c - hi) / rng + 80.f; if (is_finite(wil)) { sum += wil; cnt++; } }
                }
            }
            else if (mode == 1) { if (is_finite(tci_val)) { sum += tci_val; cnt++; } if (is_finite(mf_val)) { sum += mf_val; cnt++; } if (is_finite(rsi_val)) { sum += rsi_val; cnt++; } }
            else if (mode == 2) {
                if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                if (is_finite(csi_mg_val)) { sum += csi_mg_val; cnt++; }
                if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                float hi = -CUDART_INF_F, lo = CUDART_INF_F; bool ok = (t + 1 >= n2);
                if (ok) {
                    int start = t + 1 - n2; for (int j = start; j <= t; ++j) { float v = close_tm[idx(j, s)]; if (v > hi) hi = v; if (v < lo) lo = v; }
                    float rng = hi - lo; if (rng != 0.f) { float wil = 60.f * (c - hi) / rng + 80.f; if (is_finite(wil)) { sum += wil; cnt++; } }
                }
                if (is_finite(cbci_val)) { sum += cbci_val; cnt++; }
                if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; }
            }
            else { if (is_finite(tci_val)) { sum += tci_val; cnt++; } if (is_finite(mf_val)) { sum += mf_val; cnt++; } if (is_finite(rsi_val)) { sum += rsi_val; cnt++; } if (is_finite(cbci_val)) { sum += cbci_val; cnt++; } if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; } }
            if (cnt > 0) {
                float wt = sum / (float)cnt; wt_tm[idx(t, s)] = wt;
                if (t >= sig_start) {
                float ssum = 0.f; int ready = 1;
                for (int k=0;k<6;++k){ float x = wt_tm[idx(t - (6 - 1) + k, s)]; if (!is_finite(x)){ ready=0; break; } ssum += x; }
                if (ready) { float sig = ssum / 6.f; sig_tm[idx(t, s)] = sig; float d = (wt - sig)*2.f + 50.f; if (!is_finite(hist_tm[idx(t-1,s)])) hist_tm[idx(t,s)] = d; else hist_tm[idx(t,s)] = ema_step(d, hist_tm[idx(t-1,s)], a3, b3); }
            }
    }
    prev_close = c;
  }
}

}

// ==== begin: optimized small-window kernel (shared-memory rings + O(1) williams) ====

#ifndef MGM_RING_KCAP
// Capacity per thread-slice in shared memory, must be power-of-two.
// 64 works well for typical indicator windows; raise to 128 if workloads use bigger n2/n3.
#define MGM_RING_KCAP 64
#endif

namespace {

__device__ __forceinline__ int pow2_cap() { return MGM_RING_KCAP; }
__device__ __forceinline__ int pow2_mask() { return MGM_RING_KCAP - 1; }

// Light-weight compensated adder for sliding sums (Neumaier/Kahan style)
struct Kahan {
    float s, c;
    __device__ Kahan(): s(0.f), c(0.f) {}
    __device__ inline void add(float x){
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ inline void sub(float x){ add(-x); }
};

// Monotonic deque storing indices; max/min controlled at call site
struct MonoDeque {
    int *buf; int head, tail, mask;
    __device__ MonoDeque(): buf(nullptr), head(0), tail(0), mask(0) {}
    __device__ inline void init(int* storage, int cap_mask) {
        buf = storage; head = tail = 0; mask = cap_mask;
    }
    __device__ inline bool empty() const { return head == tail; }
    __device__ inline int size()  const { return tail - head; }
    __device__ inline int& slot(int i)   { return buf[i & mask]; }
    __device__ inline int  front() const { return buf[head & mask]; }
    __device__ inline void pop_front()   { head++; }
    __device__ inline void push_back_max(const float* __restrict__ close, int idx) {
        while (head != tail) {
            int last = buf[(tail - 1) & mask];
            if (close[last] >= close[idx]) break;
            tail--;
        }
        slot(tail) = idx; tail++;
    }
    __device__ inline void push_back_min(const float* __restrict__ close, int idx) {
        while (head != tail) {
            int last = buf[(tail - 1) & mask];
            if (close[last] <= close[idx]) break;
            tail--;
        }
        slot(tail) = idx; tail++;
    }
    __device__ inline void expire(int oldest_idx_allowed) {
        while (head != tail && front() < oldest_idx_allowed) head++;
    }
};

} // namespace

extern "C" __global__ void mod_god_mode_batch_f32_shared_fast(
    const float* __restrict__ high,   // single series
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ volume, // may be null when unused
    int len,
    int first_valid,
    int n_rows,
    const int* __restrict__ n1s,
    const int* __restrict__ n2s,
    const int* __restrict__ n3s,
    const int* __restrict__ modes,
    const int use_volume_flag,
    float* __restrict__ wavetrend_out,
    float* __restrict__ signal_out,
    float* __restrict__ histogram_out)
{
    // One thread processes one parameter row (sequential scan in time)
    const int tid    = threadIdx.x;
    const int row0   = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    // ---- Shared memory layout per block ----
    extern __shared__ unsigned char smem_raw[];
    const int  cap   = pow2_cap();          // = MGM_RING_KCAP (power-of-two)
    const int  mask  = pow2_mask();

    // Compute per-array bases packed in shared memory: [rsi_ring | mfi_mf | mfi_sgn | dq_max | dq_min]
    unsigned char* p = smem_raw;

    float* rsi_base = reinterpret_cast<float*>(p);
    p += sizeof(float) * cap * blockDim.x;

    float* mfi_mf_base = reinterpret_cast<float*>(p);
    p += sizeof(float) * cap * blockDim.x;

    // Align to 4 bytes before placing int arrays
    size_t off = reinterpret_cast<size_t>(p);
    off = (off + 3u) & ~size_t(3u);
    p = reinterpret_cast<unsigned char*>(off);

    signed char* mfi_sgn_base = reinterpret_cast<signed char*>(p);
    p += sizeof(signed char) * cap * blockDim.x;

    off = reinterpret_cast<size_t>(p);
    off = (off + 3u) & ~size_t(3u);
    p = reinterpret_cast<unsigned char*>(off);

    int* dq_max_base = reinterpret_cast<int*>(p);
    p += sizeof(int) * cap * blockDim.x;

    int* dq_min_base = reinterpret_cast<int*>(p);
    // p += sizeof(int) * cap * blockDim.x; // not used further

    // ---- per-thread slices in shared memory ----
    float*       rsi_ring  = rsi_base     + tid * cap;
    float*       mfi_mf    = mfi_mf_base  + tid * cap;
    signed char* mfi_sgn   = mfi_sgn_base + tid * cap;
    int*         dq_max    = dq_max_base  + tid * cap;
    int*         dq_min    = dq_min_base  + tid * cap;

    // Zero/NaN-initialize thread slices
    for (int k = 0; k < cap; ++k) {
        rsi_ring[k] = CUDART_NAN_F;
        mfi_mf[k]   = 0.f;
        mfi_sgn[k]  = 0;
        dq_max[k]   = 0;
        dq_min[k]   = 0;
    }
    __syncthreads(); // make sure slices are ready

    for (int row = row0; row < n_rows; row += stride) {
        const int n1   = n1s[row];
        const int n2   = n2s[row];
        const int n3   = n3s[row];
        const int mode = modes[row];

        // Fast-path guard: only use this kernel for n2,n3 that fit the cap
        if (n2 > cap || n3 > cap) {
            // Leave outputs untouched here; host wrapper will launch a fallback for these rows.
            continue;
        }

        float* wt_row   = wavetrend_out  + (size_t)row * len;
        float* sig_row  = signal_out     + (size_t)row * len;
        float* hist_row = histogram_out  + (size_t)row * len;

        // Init outputs to NaN
        for (int i = 0; i < len; ++i) {
            wt_row[i]   = CUDART_NAN_F;
            sig_row[i]  = CUDART_NAN_F;
            hist_row[i] = CUDART_NAN_F;
        }

        const float a1 = 2.f / (float(n1) + 1.f);
        const float b1 = 1.f - a1;
        const float a2 = 2.f / (float(n2) + 1.f);
        const float b2 = 1.f - a2;
        const float a3 = 2.f / (float(n3) + 1.f);
        const float b3 = 1.f - a3;
        const float inv_n3 = 1.f / float(n3);
        const float n3m1   = float(n3 - 1);

        int warm = first_valid + max(n1, max(n2, n3)) - 1;
        if (warm < 0) warm = 0;
        if (warm > len) warm = len;
        const int sig_start = warm + 6 - 1;

        // --- TCI state
        float ema1_c=0.f, ema2_abs=0.f, ema3_ci=0.f; bool seed_e1=false, seed_e2=false, seed_e3=false;
        // --- Wilder RSI state
        float rs_avg_gain=0.f, rs_avg_loss=0.f; bool rsi_seeded=false; int rs_init=0;
        float prev_close = (first_valid < len) ? close[first_valid] : 0.f;
        // --- Laguerre RSI
        LaguerreRSI lrsi;

        // --- CBCI helpers (RSI momentum and EMA)
        int   rsi_head = 0, rsi_count = 0;
        float rsi_ema  = 0.f; bool rsi_ema_seed = false;

        // --- MFI ring
        Kahan mf_pos_sum, mf_neg_sum;
        int   mf_head = 0, mf_count = 0; bool tp_has_prev=false; float tp_prev=0.f;

        // --- CSI/TSI and CSI_MG
        float tsi_m_s=0.f, tsi_m_l=0.f, tsi_a_s=0.f, tsi_a_l=0.f; bool tsi_seed_s=false, tsi_seed_l=false;
        float csi_num_e1=0.f, csi_num_e2=0.f, csi_den_e1=0.f, csi_den_e2=0.f; bool csi_seed_e1=false, csi_seed_e2=false;

        // --- Williams %R deques (O(1) amortized)
        MonoDeque dqHi, dqLo;
        dqHi.init(dq_max, mask);
        dqLo.init(dq_min, mask);

        // --- rolling sum for signal SMA(6)
        float sig_sum6 = 0.f; bool sig_seeded=false;

        // Sequential scan
        for (int i = first_valid; i < len; ++i) {
            const float c = close[i];

            // TCI
            if (!seed_e1) { ema1_c = c; seed_e1 = true; }
            else          { ema1_c = fmaf(b1, ema1_c, a1 * c); }
            float abs_dev = fabsf(c - ema1_c);
            if (!seed_e2) { ema2_abs = abs_dev; seed_e2 = true; }
            else          { ema2_abs = fmaf(b1, ema2_abs, a1 * abs_dev); }
            float tci_val = CUDART_NAN_F;
            if (ema2_abs != 0.f && isfinite(ema2_abs)) {
                float ci = (c - ema1_c) / (0.025f * ema2_abs);
                if (!seed_e3) { ema3_ci = ci; seed_e3 = true; }
                else          { ema3_ci = fmaf(b2, ema3_ci, a2 * ci); }
                tci_val = ema3_ci + 50.f;
            }

            // Wilder RSI
            float rsi_val = CUDART_NAN_F;
            if (i == first_valid) {
                rs_avg_gain=0.f; rs_avg_loss=0.f; rs_init=0; rsi_seeded=false;
            } else {
                float ch   = c - prev_close;
                float gain = fmaxf(ch, 0.f);
                float loss = fmaxf(-ch, 0.f);
                if (!rsi_seeded) {
                    rs_init += 1;
                    rs_avg_gain += gain; rs_avg_loss += loss;
                    if (rs_init >= n3) {
                        rs_avg_gain *= inv_n3; rs_avg_loss *= inv_n3;
                        rsi_seeded = true;
                        float rs = (rs_avg_loss == 0.f) ? CUDART_INF_F : (rs_avg_gain / rs_avg_loss);
                        rsi_val = 100.f - 100.f / (1.f + rs);
                    }
                } else {
                    rs_avg_gain = (rs_avg_gain * n3m1 + gain) * inv_n3;
                    rs_avg_loss = (rs_avg_loss * n3m1 + loss) * inv_n3;
                    float rs = (rs_avg_loss == 0.f) ? CUDART_INF_F : (rs_avg_gain / rs_avg_loss);
                    rsi_val = 100.f - 100.f / (1.f + rs);
                }
            }

            // Laguerre RSI
            float lrsi_val = lrsi.update(c);

            // MFI (or RSI-as-MF)
            float mf_val = CUDART_NAN_F;
            if (use_volume_flag && volume != nullptr) {
                float tp = (high[i] + low[i] + c) * (1.f/3.f);
                if (tp_has_prev) {
                    signed char sign = (tp > tp_prev) ? 1 : ((tp < tp_prev) ? -1 : 0);
                    float mf_raw = tp * volume[i];

                    // eviction occurs only when we already have a full window
                    if (rsi_seeded && mf_count >= n3) {
                        int ev = mf_head & mask;
                        float old_mf = mfi_mf[ev];
                        signed char old_s = mfi_sgn[ev];
                        if (old_s > 0) mf_pos_sum.sub(old_mf);
                        else if (old_s < 0) mf_neg_sum.sub(old_mf);
                    }

                    int idx = mf_head & mask;
                    mfi_mf[idx]  = mf_raw;
                    mfi_sgn[idx] = sign;
                    if (sign > 0)      mf_pos_sum.add(mf_raw);
                    else if (sign < 0) mf_neg_sum.add(mf_raw);
                    mf_head++; mf_count++;

                    if (rsi_seeded && mf_count >= n3) {
                        float pos = mf_pos_sum.s;
                        float neg = mf_neg_sum.s;
                        mf_val = (neg == 0.f) ? 100.f : (100.f - 100.f / (1.f + (pos / neg)));
                    }
                }
                tp_prev = tp; tp_has_prev = true;
            } else {
                mf_val = rsi_val; // original behavior when no volume
            }

            // CBCI = RSI momentum over n2 + EMA(RSI, n3)
            float cbci_val = CUDART_NAN_F;
            if (rsi_seeded) {
                // Store current RSI
                rsi_ring[rsi_head & mask] = rsi_val;
                rsi_head++; rsi_count++;

                float mom = CUDART_NAN_F;
                if (rsi_count > n2) {
                    float old_rsi = rsi_ring[(rsi_head - 1 - n2) & mask];
                    if (is_finite(old_rsi) && is_finite(rsi_val)) mom = rsi_val - old_rsi;
                }

                if (!rsi_ema_seed && is_finite(rsi_val)) { rsi_ema = rsi_val; rsi_ema_seed = true; }
                else if (is_finite(rsi_val))             { rsi_ema = fmaf(b3, rsi_ema, a3 * rsi_val); }

                if (is_finite(mom) && rsi_ema_seed) cbci_val = mom + rsi_ema;
            }

            // CSI (TSI basic)
            float csi_val = CUDART_NAN_F;
            if (i > first_valid) {
                float mom = c - prev_close; float am = fabsf(mom);
                if (!tsi_seed_s) { tsi_m_s=mom; tsi_a_s=am; tsi_seed_s=true; }
                else             { tsi_m_s = fmaf(b1, tsi_m_s, a1 * mom);
                                   tsi_a_s = fmaf(b1, tsi_a_s, a1 * am); }
                if (!tsi_seed_l) { tsi_m_l = tsi_m_s; tsi_a_l = tsi_a_s; tsi_seed_l = true; }
                else             { tsi_m_l = fmaf(b2, tsi_m_l, a2 * tsi_m_s);
                                   tsi_a_l = fmaf(b2, tsi_a_l, a2 * tsi_a_s); }
                if (tsi_seed_l && tsi_a_l != 0.f && isfinite(tsi_a_l) && is_finite(rsi_val)) {
                    float tsi = 100.f * (tsi_m_l / tsi_a_l);
                    csi_val = 0.5f * (rsi_val + (tsi * 0.5f + 50.f));
                }
            }

            // CSI_MG (normalized price change)
            float csi_mg_val = CUDART_NAN_F;
            if (i > first_valid) {
                float pc  = c - prev_close; float apc = fabsf(pc);
                if (!csi_seed_e1) { csi_num_e1=pc; csi_den_e1=apc; csi_seed_e1=true; }
                else              { csi_num_e1 = fmaf(b1, csi_num_e1, a1 * pc);
                                    csi_den_e1 = fmaf(b1, csi_den_e1, a1 * apc); }
                if (!csi_seed_e2) { csi_num_e2=csi_num_e1; csi_den_e2=csi_den_e1; csi_seed_e2=true; }
                else              { csi_num_e2 = fmaf(b2, csi_num_e2, a2 * csi_num_e1);
                                    csi_den_e2 = fmaf(b2, csi_den_e2, a2 * csi_den_e1); }
                if (csi_seed_e2 && csi_den_e2 != 0.f && isfinite(csi_den_e2) && is_finite(rsi_val)) {
                    float pc_norm = 50.f * (csi_num_e2 / csi_den_e2) + 50.f;
                    csi_mg_val = 0.5f * (rsi_val + pc_norm);
                }
            }

            // Williams %R via monotonic deques on close only (window n2)
            float wil = CUDART_NAN_F;
            // push current index
            dqHi.push_back_max(close, i);
            dqLo.push_back_min(close, i);
            // expire indices < (i - n2 + 1)
            int oldest = i - n2 + 1;
            if (oldest < 0) oldest = 0;
            dqHi.expire(oldest);
            dqLo.expire(oldest);
            if (i + 1 >= n2) {
                float hi = close[dqHi.front()];
                float lo = close[dqLo.front()];
                float rng = hi - lo;
                if (rng != 0.f) wil = 60.f * (c - hi) / rng + 80.f;
            }

            // Compose wavetrend depending on mode; write only at/after warm
            if (i >= warm) {
                float sum = 0.f; int cnt = 0;
                if (mode == 0) { // Godmode
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(csi_val)) { sum += csi_val; cnt++; }
                    if (is_finite(mf_val))  { sum += mf_val;  cnt++; }
                    if (is_finite(wil))     { sum += wil;     cnt++; }
                } else if (mode == 1) { // Tradition
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(mf_val))  { sum += mf_val;  cnt++; }
                    if (is_finite(rsi_val)) { sum += rsi_val; cnt++; }
                } else if (mode == 2) { // GodmodeMg
                    if (is_finite(tci_val))     { sum += tci_val;     cnt++; }
                    if (is_finite(csi_mg_val))  { sum += csi_mg_val;  cnt++; }
                    if (is_finite(mf_val))      { sum += mf_val;      cnt++; }
                    if (is_finite(wil))         { sum += wil;         cnt++; }
                    if (is_finite(cbci_val))    { sum += cbci_val;    cnt++; }
                    if (is_finite(lrsi_val))    { sum += lrsi_val;    cnt++; }
                } else { // TraditionMg
                    if (is_finite(tci_val))  { sum += tci_val;  cnt++; }
                    if (is_finite(mf_val))   { sum += mf_val;   cnt++; }
                    if (is_finite(rsi_val))  { sum += rsi_val;  cnt++; }
                    if (is_finite(cbci_val)) { sum += cbci_val; cnt++; }
                    if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; }
                }
                if (cnt > 0) {
                    float wt = sum / (float)cnt;
                    wt_row[i] = wt;

                // Signal SMA(6) with O(1) rolling update
                if (i >= sig_start) {
                    if (!sig_seeded) {
                        float s = 0.f; bool ok = true;
                        for (int k = 0; k < 6; ++k) {
                            float x = wt_row[i - (6 - 1) + k];
                            if (!is_finite(x)) { ok = false; break; }
                            s += x;
                        }
                        if (ok) { sig_sum6 = s; sig_seeded = true; }
                    } else {
                        float old_w = wt_row[i - 6];
                        if (is_finite(old_w)) sig_sum6 += wt - old_w;
                        else {
                            // rebuild if any non-finite sneaks in
                            float s = 0.f; bool ok = true;
                            for (int k = 0; k < 6; ++k) {
                                float x = wt_row[i - (6 - 1) + k];
                                if (!is_finite(x)) { ok = false; break; }
                                s += x;
                            }
                            if (ok) sig_sum6 = s; else sig_seeded = false;
                        }
                    }
                    if (sig_seeded) {
                        float sig = sig_sum6 / 6.f;
                        sig_row[i] = sig;
                        float d = (wt - sig) * 2.f + 50.f;
                        if (!is_finite(hist_row[i - 1])) hist_row[i] = d;
                        else                             hist_row[i] = fmaf(b3, hist_row[i - 1], a3 * d);
                    }
                }
            }
            prev_close = c;
        } // time loop
    } // row loop
}

}
// ==== end: optimized small-window kernel ====
}
