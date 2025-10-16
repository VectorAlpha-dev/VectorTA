// CUDA kernels for Range Filter (Donovan Wall style)
// Behavior parity with scalar range_filter.rs:
// - Warmup prefix is NaN for first_valid + max(range_period, smooth ? smooth_period : 0)
// - EMA of absolute change (ac_ema) and optional EMA smoothing of range
// - Clamp previous filter into [price - range, price + range]
// - Double precision for accumulators; FP32 inputs/outputs

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// ------------------- Batch: one series × many params -------------------
// Grid: (x=1, y=n_combos)
// Block: arbitrary; only thread 0 in each block performs the scan
extern "C" __global__
void range_filter_batch_f32(const float* __restrict__ prices,
                            const float* __restrict__ range_sizes,
                            const int*   __restrict__ range_periods,
                            const int*   __restrict__ smooth_flags,
                            const int*   __restrict__ smooth_periods,
                            int series_len,
                            int n_combos,
                            int first_valid,
                            float* __restrict__ filter_out,
                            float* __restrict__ high_out,
                            float* __restrict__ low_out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos || threadIdx.x != 0) { return; }

    const float rs_f   = range_sizes[combo];
    const double range_size = (double)rs_f;
    const int    rp    = range_periods[combo];
    const int    sflag = smooth_flags[combo];
    const int    sp    = smooth_periods[combo];
    if (series_len <= 0 || rp <= 0) { return; }

    // Output row base pointers
    float* __restrict__ f_row = filter_out + combo * series_len;
    float* __restrict__ h_row = high_out   + combo * series_len;
    float* __restrict__ l_row = low_out    + combo * series_len;

    // Warmup prefix length
    // Initialize entire row to NaN, then overwrite computed region
    const unsigned int nan_bits = 0x7FC00000u; // canonical quiet NaN
    const float qnan = __int_as_float((int)nan_bits);
    for (int i = 0; i < series_len; ++i) {
        f_row[i] = qnan; h_row[i] = qnan; l_row[i] = qnan;
    }

    if (first_valid >= series_len - 1) { return; }

    // Seed state
    double prev_filter = (double)prices[first_valid];
    double prev_price  = prev_filter;
    double ac_ema = 0.0; bool ac_initialized = false;
    double range_ema = 0.0; bool range_initialized = false;
    const double alpha_ac = 2.0 / (double(rp) + 1.0);
    const double one_minus_alpha_ac = 1.0 - alpha_ac;
    const double alpha_range = sflag ? (2.0 / (double(sp) + 1.0)) : 0.0;
    const double one_minus_alpha_range = 1.0 - alpha_range;

    // Scan forward from first_valid + 1
    for (int i = first_valid + 1; i < series_len; ++i) {
        const double price = (double)prices[i];
        const double d = price - prev_price;
        const double abs_change = fabs(d);
        if (!isnan(abs_change)) {
            if (!ac_initialized) { ac_ema = abs_change; ac_initialized = true; }
            else { ac_ema = fma(alpha_ac, abs_change, one_minus_alpha_ac * ac_ema); }
        }
        if (!ac_initialized) { prev_price = price; continue; }

        double range = ac_ema * range_size;
        if (sflag) {
            if (!range_initialized) { range_ema = range; range_initialized = true; }
            else { range_ema = fma(alpha_range, range, one_minus_alpha_range * range_ema); }
            range = range_ema;
        }

        const double min_b = price - range;
        const double max_b = price + range;
        const double current = fmin(fmax(prev_filter, min_b), max_b);

        // Batch CPU path writes values as soon as available (i >= first_valid+1)
        f_row[i] = (float)current;
        h_row[i] = (float)(current + range);
        l_row[i] = (float)(current - range);

        prev_filter = current;
        prev_price  = price;
    }
}

// --------- Many series (time-major), one param ---------
// Each block processes one series (column); only thread 0 does the scan.
extern "C" __global__
void range_filter_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                            float range_size_f,
                                            int   range_period,
                                            int   smooth_flag,
                                            int   smooth_period,
                                            int   num_series,
                                            int   series_len,
                                            const int* __restrict__ first_valids,
                                            float* __restrict__ filter_tm,
                                            float* __restrict__ high_tm,
                                            float* __restrict__ low_tm) {
    const int series = blockIdx.x;
    if (series >= num_series || threadIdx.x != 0) { return; }
    if (series_len <= 0 || range_period <= 0) { return; }

    const double range_size = (double)range_size_f;
    const int first_valid = first_valids[series];
    const int warm_extra = smooth_flag ? max(range_period, smooth_period) : range_period;
    const int warm_end = min(series_len, first_valid + warm_extra);

    const unsigned int nan_bits = 0x7FC00000u; // qNaN
    const float qnan = __int_as_float((int)nan_bits);
    for (int t = 0; t < series_len; ++t) {
        const int idx = t * num_series + series;
        filter_tm[idx] = qnan; high_tm[idx] = qnan; low_tm[idx] = qnan;
    }

    if (first_valid >= series_len - 1) { return; }

    double prev_filter = (double)prices_tm[first_valid * num_series + series];
    double prev_price  = prev_filter;
    double ac_ema = 0.0; bool ac_initialized = false;
    double range_ema = 0.0; bool range_initialized = false;
    const double alpha_ac = 2.0 / (double(range_period) + 1.0);
    const double one_minus_alpha_ac = 1.0 - alpha_ac;
    const double alpha_range = smooth_flag ? (2.0 / (double(smooth_period) + 1.0)) : 0.0;
    const double one_minus_alpha_range = 1.0 - alpha_range;

    for (int t = first_valid + 1; t < series_len; ++t) {
        const int idx = t * num_series + series;
        const double price = (double)prices_tm[idx];
        const double d = price - prev_price;
        const double abs_change = fabs(d);
        if (!isnan(abs_change)) {
            if (!ac_initialized) { ac_ema = abs_change; ac_initialized = true; }
            else { ac_ema = fma(alpha_ac, abs_change, one_minus_alpha_ac * ac_ema); }
        }
        if (!ac_initialized) { prev_price = price; continue; }

        double range = ac_ema * range_size;
        if (smooth_flag) {
            if (!range_initialized) { range_ema = range; range_initialized = true; }
            else { range_ema = fma(alpha_range, range, one_minus_alpha_range * range_ema); }
            range = range_ema;
        }

        const double min_b = price - range;
        const double max_b = price + range;
        const double current = fmin(fmax(prev_filter, min_b), max_b);
        if (t >= warm_end) {
            filter_tm[idx] = (float)current;
            high_tm[idx]   = (float)(current + range);
            low_tm[idx]    = (float)(current - range);
        }
        prev_filter = current;
        prev_price  = price;
    }
}
