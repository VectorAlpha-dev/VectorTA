#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        warp_sums[wid] = v;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (wid == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) >> 5;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum;
}

static __forceinline__ __device__ bool valid_ohlc(float o, float h, float l, float c) {
    return isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) &&
           o > 0.0f && h > 0.0f && l > 0.0f && c > 0.0f;
}

static __forceinline__ __device__ bool valid_bar(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int j
) {
    if (j <= 0) return false;
    const float o = open[j];
    const float h = high[j];
    const float l = low[j];
    const float c = close[j];
    const float pc = close[j - 1];
    return valid_ohlc(o, h, l, c) && isfinite(pc) && pc > 0.0f;
}

static __forceinline__ __device__ void compute_terms_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int j,
    float* rs,
    float* oret,
    float* cret
) {
    const float o = open[j];
    const float h = high[j];
    const float l = low[j];
    const float c = close[j];
    const float pc = close[j - 1];
    const float a = logf(h / c);
    const float b = logf(h / o);
    const float d = logf(l / c);
    const float e = logf(l / o);
    *rs = __fmaf_rn(d, e, a * b);
    *oret = logf(o / pc);
    *cret = logf(c / o);
}

extern "C" __global__ void yang_zhang_precompute_terms_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    int* __restrict__ valid_flags,
    float* __restrict__ rs_terms,
    float* __restrict__ oret_terms,
    float* __restrict__ cret_terms
) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < series_len;
         j += blockDim.x * gridDim.x) {
        int valid = 0;
        float rs = 0.0f;
        float oret = 0.0f;
        float cret = 0.0f;
        if (valid_bar(open, high, low, close, j)) {
            valid = 1;
            compute_terms_f32(open, high, low, close, j, &rs, &oret, &cret);
        }
        valid_flags[j] = valid;
        rs_terms[j] = rs;
        oret_terms[j] = oret;
        cret_terms[j] = cret;
    }
}

extern "C" __global__ void yang_zhang_prefix_terms_f32(
    const int* __restrict__ valid_flags,
    const float* __restrict__ rs_terms,
    const float* __restrict__ oret_terms,
    const float* __restrict__ cret_terms,
    int series_len,
    int* __restrict__ prefix_valid,
    float* __restrict__ prefix_rs,
    float* __restrict__ prefix_o,
    float* __restrict__ prefix_oo,
    float* __restrict__ prefix_c,
    float* __restrict__ prefix_cc
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    prefix_valid[0] = 0;
    prefix_rs[0] = 0.0f;
    prefix_o[0] = 0.0f;
    prefix_oo[0] = 0.0f;
    prefix_c[0] = 0.0f;
    prefix_cc[0] = 0.0f;

    int valid_acc = 0;
    float rs_acc = 0.0f;
    float o_acc = 0.0f;
    float oo_acc = 0.0f;
    float c_acc = 0.0f;
    float cc_acc = 0.0f;
    for (int j = 0; j < series_len; ++j) {
        const float o = oret_terms[j];
        const float c = cret_terms[j];
        valid_acc += valid_flags[j];
        rs_acc += rs_terms[j];
        o_acc += o;
        oo_acc += o * o;
        c_acc += c;
        cc_acc += c * c;

        const int out = j + 1;
        prefix_valid[out] = valid_acc;
        prefix_rs[out] = rs_acc;
        prefix_o[out] = o_acc;
        prefix_oo[out] = oo_acc;
        prefix_c[out] = c_acc;
        prefix_cc[out] = cc_acc;
    }
}

extern "C" __global__ void yang_zhang_volatility_batch_prefix_f32(
    const int* __restrict__ lookbacks,
    const int* __restrict__ k_overrides,
    const float* __restrict__ k_values,
    int series_len,
    int first_valid,
    int n_combos,
    const int* __restrict__ prefix_valid,
    const float* __restrict__ prefix_rs,
    const float* __restrict__ prefix_o,
    const float* __restrict__ prefix_oo,
    const float* __restrict__ prefix_c,
    const float* __restrict__ prefix_cc,
    float* __restrict__ out_yz,
    float* __restrict__ out_rs
) {
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    __shared__ int lookback_s;
    __shared__ int warmup_s;
    __shared__ int combo_valid_s;
    __shared__ float k_s;
    __shared__ float inv_lb_s;
    __shared__ float inv_denom_s;

    if (threadIdx.x == 0) {
        const int lookback = lookbacks[combo];
        int combo_valid = 1;
        float k = 0.0f;
        if (lookback <= 0 || lookback > series_len) {
            combo_valid = 0;
        } else if (k_overrides[combo] != 0) {
            k = k_values[combo];
            if (!isfinite(k) || k < 0.0f || k > 1.0f) {
                combo_valid = 0;
            }
        } else {
            k = (lookback <= 1)
                ? 0.0f
                : 0.34f / (1.34f + ((float)(lookback + 1) / (float)(lookback - 1)));
        }

        lookback_s = lookback;
        warmup_s = first_valid + lookback;
        combo_valid_s = combo_valid;
        k_s = k;
        inv_lb_s = combo_valid && lookback > 0 ? 1.0f / (float)lookback : 0.0f;
        inv_denom_s = combo_valid && lookback > 1 ? 1.0f / (float)(lookback - 1) : 0.0f;
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);
    const int base = combo * series_len;
    for (int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         t < series_len;
         t += (int)blockDim.x * (int)gridDim.x) {
        float yz_out = nan_f;
        float rs_out = nan_f;

        if (combo_valid_s != 0 && warmup_s < series_len && t >= warmup_s) {
            const int window_start = t + 1 - lookback_s;
            const int valid_count = prefix_valid[t + 1] - prefix_valid[window_start];
            if (valid_count == lookback_s) {
                float rs_var = (prefix_rs[t + 1] - prefix_rs[window_start]) * inv_lb_s;
                if (rs_var < 0.0f) {
                    rs_var = 0.0f;
                }

                float o_var = 0.0f;
                float c_var = 0.0f;
                if (lookback_s > 1) {
                    const float sum_o = prefix_o[t + 1] - prefix_o[window_start];
                    const float sum_oo = prefix_oo[t + 1] - prefix_oo[window_start];
                    const float sum_c = prefix_c[t + 1] - prefix_c[window_start];
                    const float sum_cc = prefix_cc[t + 1] - prefix_cc[window_start];
                    o_var = (sum_oo - sum_o * sum_o * inv_lb_s) * inv_denom_s;
                    c_var = (sum_cc - sum_c * sum_c * inv_lb_s) * inv_denom_s;
                    if (o_var < 0.0f) {
                        o_var = 0.0f;
                    }
                    if (c_var < 0.0f) {
                        c_var = 0.0f;
                    }
                }

                float yz_var = o_var + __fmaf_rn(1.0f - k_s, rs_var, k_s * c_var);
                if (yz_var < 0.0f) {
                    yz_var = 0.0f;
                }
                rs_out = sqrtf(rs_var);
                yz_out = sqrtf(yz_var);
            }
        }
        out_rs[base + t] = rs_out;
        out_yz[base + t] = yz_out;
    }
}

extern "C" __global__ void yang_zhang_volatility_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int* __restrict__ first_valids,
    int lookback,
    int k_override,
    float k_input,
    int cols,
    int rows,
    float* __restrict__ out_yz_tm,
    float* __restrict__ out_rs_tm
) {
    const int s = (int)blockIdx.x;
    if (s >= cols) {
        return;
    }

    const float nan_f = __int_as_float(0x7fffffff);
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        const int idx = t * cols + s;
        out_yz_tm[idx] = nan_f;
        out_rs_tm[idx] = nan_f;
    }
    __syncthreads();

    if (lookback <= 0 || lookback > rows) {
        return;
    }
    const int first_valid = first_valids[s];
    if (first_valid < 0 || first_valid >= rows) {
        return;
    }

    float k = 0.0f;
    if (k_override != 0) {
        k = k_input;
        if (!isfinite(k) || k < 0.0f || k > 1.0f) {
            return;
        }
    } else {
        k = (lookback <= 1)
            ? 0.0f
            : 0.34f / (1.34f + ((float)(lookback + 1) / (float)(lookback - 1)));
    }

    const int warmup = first_valid + lookback;
    if (warmup >= rows) {
        return;
    }

    const int start = warmup;
    const int win_start = start + 1 - lookback;
    const float inv_lb = 1.0f / (float)lookback;
    const float inv_denom = (lookback > 1) ? (1.0f / (float)(lookback - 1)) : 0.0f;

    float sum_rs_local = 0.0f;
    float sum_o_local = 0.0f;
    float sumsq_o_local = 0.0f;
    float sum_c_local = 0.0f;
    float sumsq_c_local = 0.0f;
    float invalid_local = 0.0f;

    for (int offset = threadIdx.x; offset < lookback; offset += blockDim.x) {
        const int j = win_start + offset;
        if (j <= 0) {
            invalid_local += 1.0f;
            continue;
        }
        const int idx = j * cols + s;
        const int prev = (j - 1) * cols + s;
        const float o = open_tm[idx];
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];
        const float pc = close_tm[prev];
        if (!(valid_ohlc(o, h, l, c) && isfinite(pc) && pc > 0.0f)) {
            invalid_local += 1.0f;
            continue;
        }

        const float a = logf(h / c);
        const float b = logf(h / o);
        const float d = logf(l / c);
        const float e = logf(l / o);
        const float rs = __fmaf_rn(d, e, a * b);
        const float oret = logf(o / pc);
        const float cret = logf(c / o);
        sum_rs_local += rs;
        sum_o_local += oret;
        sumsq_o_local += oret * oret;
        sum_c_local += cret;
        sumsq_c_local += cret * cret;
    }

    const float sum_rs = block_reduce_sum(sum_rs_local);
    const float sum_o = block_reduce_sum(sum_o_local);
    const float sumsq_o = block_reduce_sum(sumsq_o_local);
    const float sum_c = block_reduce_sum(sum_c_local);
    const float sumsq_c = block_reduce_sum(sumsq_c_local);
    const int invalid_count = (int)block_reduce_sum(invalid_local);

    if (threadIdx.x == 0) {
        float rolling_rs = sum_rs;
        float rolling_o = sum_o;
        float rolling_oo = sumsq_o;
        float rolling_c = sum_c;
        float rolling_cc = sumsq_c;
        int rolling_invalid = invalid_count;

        for (int t = start; t < rows; ++t) {
            const int out_idx = t * cols + s;
            if (rolling_invalid == 0) {
                float rs_var = rolling_rs * inv_lb;
                if (rs_var < 0.0f) {
                    rs_var = 0.0f;
                }

                float o_var = 0.0f;
                float c_var = 0.0f;
                if (lookback > 1) {
                    o_var = (rolling_oo - rolling_o * rolling_o * inv_lb) * inv_denom;
                    c_var = (rolling_cc - rolling_c * rolling_c * inv_lb) * inv_denom;
                    if (o_var < 0.0f) {
                        o_var = 0.0f;
                    }
                    if (c_var < 0.0f) {
                        c_var = 0.0f;
                    }
                }

                float yz_var = o_var + __fmaf_rn(1.0f - k, rs_var, k * c_var);
                if (yz_var < 0.0f) {
                    yz_var = 0.0f;
                }
                out_rs_tm[out_idx] = sqrtf(rs_var);
                out_yz_tm[out_idx] = sqrtf(yz_var);
            }

            if (t + 1 < rows) {
                const int add_idx = t + 1;
                const int sub_idx = add_idx - lookback;

                if (add_idx > 0) {
                    const int idx = add_idx * cols + s;
                    const int prev = (add_idx - 1) * cols + s;
                    const float o = open_tm[idx];
                    const float h = high_tm[idx];
                    const float l = low_tm[idx];
                    const float c = close_tm[idx];
                    const float pc = close_tm[prev];
                    if (valid_ohlc(o, h, l, c) && isfinite(pc) && pc > 0.0f) {
                        const float a = logf(h / c);
                        const float b = logf(h / o);
                        const float d = logf(l / c);
                        const float e = logf(l / o);
                        const float rs = __fmaf_rn(d, e, a * b);
                        const float oret = logf(o / pc);
                        const float cret = logf(c / o);
                        rolling_rs += rs;
                        rolling_o += oret;
                        rolling_oo += oret * oret;
                        rolling_c += cret;
                        rolling_cc += cret * cret;
                    } else {
                        ++rolling_invalid;
                    }
                }

                if (sub_idx > 0) {
                    const int idx = sub_idx * cols + s;
                    const int prev = (sub_idx - 1) * cols + s;
                    const float o = open_tm[idx];
                    const float h = high_tm[idx];
                    const float l = low_tm[idx];
                    const float c = close_tm[idx];
                    const float pc = close_tm[prev];
                    if (valid_ohlc(o, h, l, c) && isfinite(pc) && pc > 0.0f) {
                        const float a = logf(h / c);
                        const float b = logf(h / o);
                        const float d = logf(l / c);
                        const float e = logf(l / o);
                        const float rs = __fmaf_rn(d, e, a * b);
                        const float oret = logf(o / pc);
                        const float cret = logf(c / o);
                        rolling_rs -= rs;
                        rolling_o -= oret;
                        rolling_oo -= oret * oret;
                        rolling_c -= cret;
                        rolling_cc -= cret * cret;
                    } else {
                        --rolling_invalid;
                    }
                }
            }
        }
    }
}
