#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float pr_open(float body_low, float body_high, int8_t direction) {
    return (direction >= 0) ? body_low : body_high;
}

__device__ __forceinline__ float pr_close(float body_low, float body_high, int8_t direction) {
    return (direction >= 0) ? body_high : body_low;
}

__device__ __forceinline__ float pr_high(float body_high, float upper_shadow) {
    return body_high + upper_shadow;
}

__device__ __forceinline__ float pr_low(float body_low, float lower_shadow) {
    return body_low - lower_shadow;
}

extern "C" __global__ void pattern_features_kernel_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    float* __restrict__ body,
    float* __restrict__ body_low,
    float* __restrict__ body_high,
    float* __restrict__ range,
    float* __restrict__ upper_shadow,
    float* __restrict__ lower_shadow,
    int8_t* __restrict__ direction,
    uint8_t* __restrict__ body_gap_up,
    uint8_t* __restrict__ body_gap_down)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        const float o = open[i];
        const float h = high[i];
        const float l = low[i];
        const float c = close[i];
        const float lo = fminf(o, c);
        const float hi = fmaxf(o, c);

        body[i] = fabsf(c - o);
        body_low[i] = lo;
        body_high[i] = hi;
        range[i] = h - l;
        upper_shadow[i] = (c >= o) ? (h - c) : (h - o);
        lower_shadow[i] = (c >= o) ? (o - l) : (c - l);
        direction[i] = (c >= o) ? 1 : -1;

        if (i == 0) {
            body_gap_up[i] = 0;
            body_gap_down[i] = 0;
        } else {
            const float prev_lo = fminf(open[i - 1], close[i - 1]);
            const float prev_hi = fmaxf(open[i - 1], close[i - 1]);
            body_gap_up[i] = (lo > prev_hi) ? 1 : 0;
            body_gap_down[i] = (hi < prev_lo) ? 1 : 0;
        }
    }
}

extern "C" __global__ void pattern_doji_predicate_kernel_f32(
    const float* __restrict__ body,
    const float* __restrict__ range,
    int len,
    float threshold,
    uint8_t* __restrict__ out_mask)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        const float b = body[i];
        const float r = range[i];
        const bool finite_vals = isfinite(b) && isfinite(r);
        const bool hit = finite_vals && r > 0.0f && b <= (threshold * r);
        out_mask[i] = hit ? 1 : 0;
    }
}

extern "C" __global__ void pattern_rolling_stats_10_f32_kernel(
    const float* __restrict__ body,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    float* __restrict__ body_avg10,
    float* __restrict__ body_avg5,
    float* __restrict__ upper_avg10,
    float* __restrict__ lower_avg10,
    float* __restrict__ max_shadow_avg10,
    float* __restrict__ belt_shadow_avg10,
    float* __restrict__ closing_shadow_avg10)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        if (i < 10) {
            body_avg10[i] = 0.0f;
            upper_avg10[i] = 0.0f;
            lower_avg10[i] = 0.0f;
            max_shadow_avg10[i] = 0.0f;
            belt_shadow_avg10[i] = 0.0f;
            closing_shadow_avg10[i] = 0.0f;
        } else {
            float sum_body = 0.0f;
            float sum_upper = 0.0f;
            float sum_lower = 0.0f;
            float sum_max_shadow = 0.0f;
            float sum_belt_shadow = 0.0f;
            float sum_closing_shadow = 0.0f;
            for (int j = i - 10; j < i; ++j) {
                const float upper = upper_shadow[j];
                const float lower = lower_shadow[j];
                sum_body += body[j];
                sum_upper += upper;
                sum_lower += lower;
                sum_max_shadow += fmaxf(upper, lower);
                if (direction[j] >= 0) {
                    sum_belt_shadow += lower;
                    sum_closing_shadow += upper;
                } else {
                    sum_belt_shadow += upper;
                    sum_closing_shadow += lower;
                }
            }

            body_avg10[i] = sum_body * 0.1f;
            upper_avg10[i] = sum_upper * 0.1f;
            lower_avg10[i] = sum_lower * 0.1f;
            max_shadow_avg10[i] = sum_max_shadow * 0.1f;
            belt_shadow_avg10[i] = sum_belt_shadow * 0.1f;
            closing_shadow_avg10[i] = sum_closing_shadow * 0.1f;
        }

        if (i < 5) {
            body_avg5[i] = 0.0f;
        } else {
            float sum_body5 = 0.0f;
            for (int j = i - 5; j < i; ++j) {
                sum_body5 += body[j];
            }
            body_avg5[i] = sum_body5 * 0.2f;
        }
    }
}

extern "C" __global__ void pattern_rows_simple10_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ upper_avg10,
    const float* __restrict__ max_shadow_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row_cdldoji,
    int row_cdldragonflydoji,
    int row_cdlgravestonedoji,
    int row_cdllongleggeddoji,
    int row_cdlmarubozu,
    int row_cdlhighwave,
    int row_cdllongline,
    int row_cdlshortline,
    int row_cdlspinningtop)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        const bool warmup = i < 10;
        const float b = body[i];
        const float avg_body = body_avg10[i];
        const float upper = upper_shadow[i];
        const float lower = lower_shadow[i];
        const float avg_upper = upper_avg10[i];
        const float avg_shadow = max_shadow_avg10[i];

        if (row_cdldoji >= 0) {
            matrix[row_cdldoji * cols + i] = warmup ? 0u : ((b <= avg_body) ? 1u : 0u);
        }
        if (row_cdldragonflydoji >= 0) {
            const bool hit = !warmup && (b <= avg_body) && (upper < avg_shadow) && (lower > avg_shadow);
            matrix[row_cdldragonflydoji * cols + i] = hit ? 1u : 0u;
        }
        if (row_cdlgravestonedoji >= 0) {
            const bool hit = !warmup && (b <= avg_body) && (lower < avg_upper) && (upper > avg_upper);
            matrix[row_cdlgravestonedoji * cols + i] = hit ? 1u : 0u;
        }
        if (row_cdllongleggeddoji >= 0) {
            const bool hit = !warmup && (b <= avg_body) && ((lower > avg_upper) || (upper > avg_upper));
            matrix[row_cdllongleggeddoji * cols + i] = hit ? 1u : 0u;
        }
        if (row_cdlmarubozu >= 0) {
            const bool hit = !warmup && (b > avg_body) && (upper < avg_upper) && (lower < avg_upper);
            matrix[row_cdlmarubozu * cols + i] = hit ? 1u : 0u;
        }
        if (row_cdlhighwave >= 0) {
            const bool hit = !warmup && (b < avg_body) && (upper > avg_upper) && (lower > avg_upper);
            matrix[row_cdlhighwave * cols + i] = hit ? 1u : 0u;
        }
        if (row_cdllongline >= 0) {
            const bool hit = !warmup && (b > avg_body) && (upper < avg_upper) && (lower < avg_upper);
            matrix[row_cdllongline * cols + i] = hit ? 1u : 0u;
        }
        if (row_cdlshortline >= 0) {
            const bool hit = !warmup && (b < avg_body) && (upper < avg_upper) && (lower < avg_upper);
            matrix[row_cdlshortline * cols + i] = hit ? 1u : 0u;
        }
        if (row_cdlspinningtop >= 0) {
            const bool hit = !warmup && (b < avg_body) && (upper > b) && (lower > b);
            matrix[row_cdlspinningtop * cols + i] = hit ? 1u : 0u;
        }
    }
}

extern "C" __global__ void pattern_rows_two_bar_body10_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row_cdldojistar,
    int row_cdlharami,
    int row_cdlharamicross,
    int row_cdlhomingpigeon)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int lookback = 11;

    for (int i = tid; i < len; i += stride) {
        const bool warmup = i < lookback;
        if (row_cdldojistar >= 0) {
            bool hit = false;
            if (!warmup) {
                const float avg_long = body_avg10[i - 1];
                const float avg_doji = body_avg10[i];
                const int8_t prev_dir = direction[i - 1];
                hit = body[i - 1] > avg_long
                    && body[i] <= avg_doji
                    && ((prev_dir > 0 && body_gap_up[i] != 0u) || (prev_dir < 0 && body_gap_down[i] != 0u));
            }
            matrix[row_cdldojistar * cols + i] = hit ? 1u : 0u;
        }
        if (row_cdlharami >= 0 || row_cdlharamicross >= 0) {
            bool hit = false;
            if (!warmup) {
                const float avg_long = body_avg10[i - 1];
                const float avg_short = body_avg10[i];
                if (body[i - 1] > avg_long && body[i] <= avg_short) {
                    const float hi0 = body_high[i - 1];
                    const float lo0 = body_low[i - 1];
                    const float hi1 = body_high[i];
                    const float lo1 = body_low[i];
                    hit = (hi1 <= hi0 && lo1 >= lo0);
                }
            }
            if (row_cdlharami >= 0) {
                matrix[row_cdlharami * cols + i] = hit ? 1u : 0u;
            }
            if (row_cdlharamicross >= 0) {
                matrix[row_cdlharamicross * cols + i] = hit ? 1u : 0u;
            }
        }
        if (row_cdlhomingpigeon >= 0) {
            bool hit = false;
            if (!warmup) {
                const float avg_long = body_avg10[i - 1];
                const float avg_short = body_avg10[i];
                hit = direction[i - 1] < 0
                    && direction[i] < 0
                    && body[i - 1] > avg_long
                    && body[i] <= avg_short
                    && body_high[i] < body_high[i - 1]
                    && body_low[i] > body_low[i - 1];
            }
            matrix[row_cdlhomingpigeon * cols + i] = hit ? 1u : 0u;
        }
    }
}

extern "C" __global__ void pattern_rows_single_bar_shadow_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ body_avg10,
    const float* __restrict__ body_avg5,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ upper_avg10,
    const float* __restrict__ lower_avg10,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row_cdlhammer,
    int row_cdlhangingman,
    int row_cdlinvertedhammer,
    int row_cdlshootingstar,
    int row_cdltakuri,
    int row_cdlrickshawman)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        const bool warmup10 = i < 10;
        const bool warmup11 = i < 11;
        const float b = body[i];
        const float body_avg_i = body_avg10[i];
        const float upper_i = upper_shadow[i];
        const float lower_i = lower_shadow[i];
        const float upper_avg_i = upper_avg10[i];
        const float lower_avg_i = lower_avg10[i];

        if (row_cdlhammer >= 0) {
            bool hit = false;
            if (!warmup11) {
                const float prev_low = pr_low(body_low[i - 1], lower_shadow[i - 1]);
                hit = b < body_avg_i
                    && lower_i > lower_avg_i
                    && upper_i < upper_avg_i
                    && body_low[i] <= prev_low + body_avg10[i - 1];
            }
            matrix[row_cdlhammer * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdlhangingman >= 0) {
            bool hit = false;
            if (!warmup11) {
                const float prev_high = pr_high(body_high[i - 1], upper_shadow[i - 1]);
                hit = b < body_avg_i
                    && lower_i > lower_avg_i
                    && upper_i < upper_avg_i
                    && body_low[i] >= prev_high - body_avg10[i - 1];
            }
            matrix[row_cdlhangingman * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdlinvertedhammer >= 0) {
            bool hit = false;
            if (!warmup11) {
                hit = b < body_avg10[i - 1]
                    && upper_i > upper_avg10[i - 1]
                    && lower_i < lower_avg10[i - 1]
                    && body_gap_down[i] != 0u;
            }
            matrix[row_cdlinvertedhammer * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdlshootingstar >= 0) {
            bool hit = false;
            if (!warmup11) {
                hit = b < body_avg10[i - 1]
                    && upper_i > upper_avg10[i - 1]
                    && lower_i < lower_avg10[i - 1]
                    && body_gap_up[i] != 0u;
            }
            matrix[row_cdlshootingstar * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdltakuri >= 0) {
            const bool hit = !warmup10
                && (b <= body_avg_i)
                && (upper_i < upper_avg_i)
                && (lower_i > lower_avg_i);
            matrix[row_cdltakuri * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdlrickshawman >= 0) {
            bool hit = false;
            if (!warmup10) {
                const float high_i = pr_high(body_high[i], upper_i);
                const float low_i = pr_low(body_low[i], lower_i);
                const float mid = low_i + (high_i - low_i) * 0.5f;
                hit = b <= body_avg_i
                    && lower_i > upper_avg_i
                    && upper_i > upper_avg_i
                    && body_low[i] <= mid + body_avg5[i]
                    && body_high[i] >= mid - body_avg5[i];
            }
            matrix[row_cdlrickshawman * cols + i] = hit ? 1u : 0u;
        }
    }
}

extern "C" __global__ void pattern_rows_directional_shadow_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ max_shadow_avg10,
    const float* __restrict__ belt_shadow_avg10,
    const float* __restrict__ closing_shadow_avg10,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row_cdlbelthold,
    int row_cdlclosingmarubozu,
    int row_cdlkicking,
    int row_cdlkickingbylength)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        const bool warmup10 = i < 10;
        const bool warmup11 = i < 11;
        const float b = body[i];
        const int8_t dir_i = direction[i];
        const float upper_i = upper_shadow[i];
        const float lower_i = lower_shadow[i];
        const float max_shadow_i = fmaxf(upper_i, lower_i);

        if (row_cdlbelthold >= 0) {
            bool hit = false;
            if (!warmup10) {
                const float avg_shadow = belt_shadow_avg10[i];
                hit = b > body_avg10[i]
                    && ((dir_i >= 0 && lower_i < avg_shadow) || (dir_i < 0 && upper_i < avg_shadow));
            }
            matrix[row_cdlbelthold * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdlclosingmarubozu >= 0) {
            bool hit = false;
            if (!warmup10) {
                const float avg_shadow = closing_shadow_avg10[i];
                hit = b > body_avg10[i]
                    && ((dir_i >= 0 && upper_i < avg_shadow) || (dir_i < 0 && lower_i < avg_shadow));
            }
            matrix[row_cdlclosingmarubozu * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdlkicking >= 0 || row_cdlkickingbylength >= 0) {
            bool hit = false;
            if (!warmup11) {
                const int8_t dir_prev = direction[i - 1];
                const bool gap_cond = (dir_prev < 0 && body_gap_up[i] != 0u)
                    || (dir_prev >= 0 && body_gap_down[i] != 0u);
                hit = dir_prev == -dir_i
                    && body[i - 1] > body_avg10[i - 1]
                    && fmaxf(upper_shadow[i - 1], lower_shadow[i - 1]) < max_shadow_avg10[i - 1]
                    && b > body_avg10[i]
                    && max_shadow_i < max_shadow_avg10[i]
                    && gap_cond;
            }
            if (row_cdlkicking >= 0) {
                matrix[row_cdlkicking * cols + i] = hit ? 1u : 0u;
            }
            if (row_cdlkickingbylength >= 0) {
                matrix[row_cdlkickingbylength * cols + i] = hit ? 1u : 0u;
            }
        }
    }
}

extern "C" __global__ void pattern_rows_star3_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ body_avg10,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    float penetration,
    uint8_t* __restrict__ matrix,
    int cols,
    int row_cdleveningdojistar,
    int row_cdleveningstar,
    int row_cdlmorningdojistar,
    int row_cdlmorningstar)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int lookback = 12;

    for (int i = tid; i < len; i += stride) {
        const bool warmup = i < lookback;

        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float avg_long = body_avg10[i - 2];
        const float avg_mid = body_avg10[i - 1];
        const float avg_short = body_avg10[i];

        if (row_cdleveningdojistar >= 0) {
            bool hit = false;
            if (!warmup) {
                hit = body[i - 2] > avg_long
                    && direction[i - 2] >= 0
                    && body[i - 1] <= avg_mid
                    && body_gap_up[i - 1] != 0u
                    && body[i] > avg_short
                    && direction[i] < 0
                    && close_i < close_prev2 - body[i - 2] * penetration;
            }
            matrix[row_cdleveningdojistar * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdleveningstar >= 0) {
            bool hit = false;
            if (!warmup) {
                hit = body[i - 2] > avg_long
                    && direction[i - 2] >= 0
                    && body[i - 1] <= avg_mid
                    && body_gap_up[i - 1] != 0u
                    && body[i] > avg_short
                    && direction[i] < 0
                    && close_i < close_prev2 - body[i - 2] * penetration;
            }
            matrix[row_cdleveningstar * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdlmorningdojistar >= 0) {
            bool hit = false;
            if (!warmup) {
                hit = body[i - 2] > avg_long
                    && direction[i - 2] < 0
                    && body[i - 1] <= avg_mid
                    && body_gap_down[i - 1] != 0u
                    && body[i] > avg_short
                    && direction[i] >= 0
                    && close_i > close_prev2 + body[i - 2] * penetration;
            }
            matrix[row_cdlmorningdojistar * cols + i] = hit ? 1u : 0u;
        }

        if (row_cdlmorningstar >= 0) {
            bool hit = false;
            if (!warmup) {
                hit = body[i - 2] > avg_long
                    && direction[i - 2] < 0
                    && body[i - 1] <= avg_mid
                    && body_gap_down[i - 1] != 0u
                    && body[i] > avg_short
                    && direction[i] >= 0
                    && close_i > close_prev2 + body[i - 2] * penetration;
            }
            matrix[row_cdlmorningstar * cols + i] = hit ? 1u : 0u;
        }
    }
}

extern "C" __global__ void pattern_rolling_mean_f32_kernel(
    const float* __restrict__ input,
    int len,
    int period,
    float* __restrict__ out_avg)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        if (i < period) {
            out_avg[i] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        for (int j = i - period; j < i; ++j) {
            sum += input[j];
        }
        out_avg[i] = sum / (float)period;
    }
}

extern "C" __global__ void pattern_rolling_max_shadow_mean_f32_kernel(
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    int len,
    int period,
    float* __restrict__ out_avg)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < len; i += stride) {
        if (i < period) {
            out_avg[i] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        for (int j = i - period; j < i; ++j) {
            sum += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        out_avg[i] = sum / (float)period;
    }
}

extern "C" __global__ void pattern_matrix_zero_u8_kernel(
    uint8_t* __restrict__ matrix,
    int total)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < total; i += stride) {
        matrix[i] = 0;
    }
}

extern "C" __global__ void pattern_row_cdldoji_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 10) {
            matrix[base + i] = 0;
            continue;
        }
        matrix[base + i] = (body[i] <= body_avg10[i]) ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdldragonflydoji_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ max_shadow_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 10) {
            matrix[base + i] = 0;
            continue;
        }

        const float avg_body = body_avg10[i];
        const float avg_shadow = max_shadow_avg10[i];
        const bool hit = (body[i] <= avg_body)
            && (upper_shadow[i] < avg_shadow)
            && (lower_shadow[i] > avg_shadow);
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlgravestonedoji_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ upper_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 10) {
            matrix[base + i] = 0;
            continue;
        }

        const float avg_body = body_avg10[i];
        const float avg_upper = upper_avg10[i];
        const bool hit = (body[i] <= avg_body)
            && (lower_shadow[i] < avg_upper)
            && (upper_shadow[i] > avg_upper);
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdllongleggeddoji_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ upper_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 10) {
            matrix[base + i] = 0;
            continue;
        }

        const float avg_body = body_avg10[i];
        const float avg_upper = upper_avg10[i];
        const bool hit = (body[i] <= avg_body)
            && ((lower_shadow[i] > avg_upper) || (upper_shadow[i] > avg_upper));
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlmarubozu_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ upper_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 10) {
            matrix[base + i] = 0;
            continue;
        }

        const float avg_body = body_avg10[i];
        const float avg_upper = upper_avg10[i];
        const bool hit = (body[i] > avg_body)
            && (upper_shadow[i] < avg_upper)
            && (lower_shadow[i] < avg_upper);
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdldojistar_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 11;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }
        const float avg_long = body_avg10[i - 1];
        const float avg_doji = body_avg10[i];
        const int8_t prev_dir = direction[i - 1];
        const bool hit = body[i - 1] > avg_long
            && body[i] <= avg_doji
            && ((prev_dir > 0 && body_gap_up[i] != 0u) || (prev_dir < 0 && body_gap_down[i] != 0u));
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlengulfing_u8_kernel(
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 1) {
            matrix[base + i] = 0;
            continue;
        }

        const int8_t d1 = direction[i - 1];
        const int8_t d2 = direction[i];
        const bool opposite = ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0));
        const float hi0 = body_high[i - 1];
        const float lo0 = body_low[i - 1];
        const float hi1 = body_high[i];
        const float lo1 = body_low[i];
        const bool envelop = hi1 >= hi0 && lo1 <= lo0 && (hi1 > hi0 || lo1 < lo0);
        matrix[base + i] = (opposite && envelop) ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlharami_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 11;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }
        const float avg_long = body_avg10[i - 1];
        const float avg_short = body_avg10[i];

        bool hit = false;
        if (body[i - 1] > avg_long && body[i] <= avg_short) {
            const float hi0 = body_high[i - 1];
            const float lo0 = body_low[i - 1];
            const float hi1 = body_high[i];
            const float lo1 = body_low[i];
            hit = (hi1 <= hi0 && lo1 >= lo0);
        }

        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlhighwave_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ upper_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 10) {
            matrix[base + i] = 0;
            continue;
        }
        const float avg_body = body_avg10[i];
        const float avg_upper = upper_avg10[i];
        const bool hit = body[i] < avg_body && upper_shadow[i] > avg_upper && lower_shadow[i] > avg_upper;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlinvertedhammer_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ upper_avg10,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ lower_avg10,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 11;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }
        const float avg_body = body_avg10[i - 1];
        const float avg_upper = upper_avg10[i - 1];
        const float avg_lower = lower_avg10[i - 1];
        const bool hit = body[i] < avg_body
            && upper_shadow[i] > avg_upper
            && lower_shadow[i] < avg_lower
            && body_gap_down[i] != 0u;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdllongline_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ upper_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 10;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }
        const float avg_body = body_avg10[i];
        const float avg_upper = upper_avg10[i];
        const bool hit = body[i] > avg_body && upper_shadow[i] < avg_upper && lower_shadow[i] < avg_upper;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlhomingpigeon_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    int len,
    int period_long,
    int period_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 1 + ((period_long > period_short) ? period_long : period_short);

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        for (int j = i - period_long; j < i; ++j) {
            sum_long += body[j - 1];
        }
        float sum_short = 0.0f;
        for (int j = i - period_short; j < i; ++j) {
            sum_short += body[j];
        }

        const float avg_long = sum_long / (float)period_long;
        const float avg_short = sum_short / (float)period_short;
        const bool hit = direction[i - 1] < 0
            && direction[i] < 0
            && body[i - 1] > avg_long
            && body[i] <= avg_short
            && body_high[i] < body_high[i - 1]
            && body_low[i] > body_low[i - 1];
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlshootingstar_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const uint8_t* __restrict__ body_gap_up,
    int len,
    int period_body_short,
    int period_shadow_long,
    int period_shadow_very_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int max_p = period_body_short > period_shadow_long ? period_body_short : period_shadow_long;
    const int lookback = 1 + (max_p > period_shadow_very_short ? max_p : period_shadow_very_short);

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body = 0.0f;
        for (int j = i - 1 - period_body_short; j < i - 1; ++j) {
            sum_body += body[j];
        }
        float sum_upper = 0.0f;
        for (int j = i - 1 - period_shadow_long; j < i - 1; ++j) {
            sum_upper += upper_shadow[j];
        }
        float sum_lower = 0.0f;
        for (int j = i - 1 - period_shadow_very_short; j < i - 1; ++j) {
            sum_lower += lower_shadow[j];
        }

        const float avg_body = sum_body / (float)period_body_short;
        const float avg_upper = sum_upper / (float)period_shadow_long;
        const float avg_lower = sum_lower / (float)period_shadow_very_short;
        const bool hit = body[i] < avg_body
            && upper_shadow[i] > avg_upper
            && lower_shadow[i] < avg_lower
            && body_gap_up[i] != 0u;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlshortline_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const float* __restrict__ upper_avg10,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 10;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }
        const float avg_body = body_avg10[i];
        const float avg_upper = upper_avg10[i];
        const bool hit = body[i] < avg_body && upper_shadow[i] < avg_upper && lower_shadow[i] < avg_upper;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlspinningtop_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_avg10,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 10) {
            matrix[base + i] = 0;
            continue;
        }
        const float avg_body = body_avg10[i];
        const bool hit = body[i] < avg_body && upper_shadow[i] > body[i] && lower_shadow[i] > body[i];
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdltakuri_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    int len,
    int period_body_doji,
    int period_shadow_very_short,
    int period_shadow_very_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_body_doji;
    if (period_shadow_very_short > lookback) {
        lookback = period_shadow_very_short;
    }
    if (period_shadow_very_long > lookback) {
        lookback = period_shadow_very_long;
    }

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body = 0.0f;
        for (int j = i - period_body_doji; j < i; ++j) {
            sum_body += body[j];
        }
        float sum_upper = 0.0f;
        for (int j = i - period_shadow_very_short; j < i; ++j) {
            sum_upper += upper_shadow[j];
        }
        float sum_lower = 0.0f;
        for (int j = i - period_shadow_very_long; j < i; ++j) {
            sum_lower += lower_shadow[j];
        }

        const float avg_body = sum_body / (float)period_body_doji;
        const float avg_upper = sum_upper / (float)period_shadow_very_short;
        const float avg_lower = sum_lower / (float)period_shadow_very_long;
        const bool hit = body[i] <= avg_body && upper_shadow[i] < avg_upper && lower_shadow[i] > avg_lower;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlbelthold_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    int period_shadow_very_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_body_long > period_shadow_very_short ? period_body_long : period_shadow_very_short;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body = 0.0f;
        float sum_shadow = 0.0f;
        for (int j = i - lookback; j < i; ++j) {
            sum_body += body[j];
            sum_shadow += (direction[j] >= 0) ? lower_shadow[j] : upper_shadow[j];
        }

        const float avg_body = sum_body / (float)period_body_long;
        const float avg_shadow = sum_shadow / (float)period_shadow_very_short;
        const bool hit = body[i] > avg_body
            && ((direction[i] >= 0 && lower_shadow[i] < avg_shadow)
                || (direction[i] < 0 && upper_shadow[i] < avg_shadow));
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlclosingmarubozu_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    int period_shadow_very_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_body_long > period_shadow_very_short ? period_body_long : period_shadow_very_short;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body = 0.0f;
        float sum_shadow = 0.0f;
        for (int j = i - lookback; j < i; ++j) {
            sum_body += body[j];
            sum_shadow += (direction[j] >= 0) ? upper_shadow[j] : lower_shadow[j];
        }

        const float avg_body = sum_body / (float)period_body_long;
        const float avg_shadow = sum_shadow / (float)period_shadow_very_short;
        const bool hit = body[i] > avg_body
            && ((direction[i] >= 0 && upper_shadow[i] < avg_shadow)
                || (direction[i] < 0 && lower_shadow[i] < avg_shadow));
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlhammer_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    int len,
    int period_body_short,
    int period_shadow_long,
    int period_shadow_very_short,
    int period_near,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_body_short;
    if (period_shadow_long > lookback) {
        lookback = period_shadow_long;
    }
    if (period_shadow_very_short > lookback) {
        lookback = period_shadow_very_short;
    }
    if (period_near > lookback) {
        lookback = period_near;
    }
    lookback += 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body = 0.0f;
        float sum_ls = 0.0f;
        float sum_us = 0.0f;
        float sum_near = 0.0f;
        for (int j = i - period_body_short; j < i; ++j) {
            sum_body += body[j];
        }
        for (int j = i - period_shadow_long; j < i; ++j) {
            sum_ls += lower_shadow[j];
        }
        for (int j = i - period_shadow_very_short; j < i; ++j) {
            sum_us += upper_shadow[j];
        }
        for (int j = i - 1 - period_near; j < i - 1; ++j) {
            sum_near += body[j];
        }

        const float avg_body = sum_body / (float)period_body_short;
        const float avg_ls = sum_ls / (float)period_shadow_long;
        const float avg_us = sum_us / (float)period_shadow_very_short;
        const float avg_near = sum_near / (float)period_near;
        const float prev_low = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const bool hit = body[i] < avg_body
            && lower_shadow[i] > avg_ls
            && upper_shadow[i] < avg_us
            && body_low[i] <= prev_low + avg_near;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlhangingman_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    int len,
    int period_body_short,
    int period_shadow_long,
    int period_shadow_very_short,
    int period_near,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_body_short;
    if (period_shadow_long > lookback) {
        lookback = period_shadow_long;
    }
    if (period_shadow_very_short > lookback) {
        lookback = period_shadow_very_short;
    }
    if (period_near > lookback) {
        lookback = period_near;
    }
    lookback += 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body = 0.0f;
        float sum_ls = 0.0f;
        float sum_us = 0.0f;
        float sum_near = 0.0f;
        for (int j = i - period_body_short; j < i; ++j) {
            sum_body += body[j];
        }
        for (int j = i - period_shadow_long; j < i; ++j) {
            sum_ls += lower_shadow[j];
        }
        for (int j = i - period_shadow_very_short; j < i; ++j) {
            sum_us += upper_shadow[j];
        }
        for (int j = i - 1 - period_near; j < i - 1; ++j) {
            sum_near += body[j];
        }

        const float avg_body = sum_body / (float)period_body_short;
        const float avg_ls = sum_ls / (float)period_shadow_long;
        const float avg_us = sum_us / (float)period_shadow_very_short;
        const float avg_near = sum_near / (float)period_near;
        const float prev_high = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const bool hit = body[i] < avg_body
            && lower_shadow[i] > avg_ls
            && upper_shadow[i] < avg_us
            && body_low[i] >= prev_high - avg_near;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlrickshawman_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    int len,
    int period_body_doji,
    int period_shadow_long,
    int period_near,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_body_doji;
    if (period_shadow_long > lookback) {
        lookback = period_shadow_long;
    }
    if (period_near > lookback) {
        lookback = period_near;
    }

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body = 0.0f;
        float sum_upper = 0.0f;
        float sum_near = 0.0f;
        for (int j = i - period_body_doji; j < i; ++j) {
            sum_body += body[j];
        }
        for (int j = i - period_shadow_long; j < i; ++j) {
            sum_upper += upper_shadow[j];
        }
        for (int j = i - period_near; j < i; ++j) {
            sum_near += body[j];
        }

        const float avg_body = sum_body / (float)period_body_doji;
        const float avg_upper = sum_upper / (float)period_shadow_long;
        const float avg_near = sum_near / (float)period_near;
        const float high_i = pr_high(body_high[i], upper_shadow[i]);
        const float low_i = pr_low(body_low[i], lower_shadow[i]);
        const float mid = low_i + (high_i - low_i) * 0.5f;
        const bool hit = body[i] <= avg_body
            && lower_shadow[i] > avg_upper
            && upper_shadow[i] > avg_upper
            && body_low[i] <= mid + avg_near
            && body_high[i] >= mid - avg_near;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlmatchinglow_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    int len,
    int period_equal,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_equal + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_equal = 0.0f;
        for (int j = i - period_equal - 1; j < i - 1; ++j) {
            sum_equal += body[j];
        }

        const float avg_equal = sum_equal / (float)period_equal;
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const bool hit = direction[i - 1] < 0
            && direction[i] < 0
            && fabsf(close_i - close_prev) <= avg_equal;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlinneck_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_equal,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_equal > period_body_long ? period_equal : period_body_long) + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_equal = 0.0f;
        float sum_long = 0.0f;
        for (int j = i - period_equal - 1; j < i - 1; ++j) {
            sum_equal += body[j];
        }
        for (int j = i - period_body_long - 1; j < i - 1; ++j) {
            sum_long += body[j];
        }

        const float avg_equal = sum_equal / (float)period_equal;
        const float avg_long = sum_long / (float)period_body_long;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float low_prev = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const bool hit = direction[i - 1] < 0
            && body[i - 1] > avg_long
            && direction[i] >= 0
            && open_i < low_prev
            && close_i <= close_prev + avg_equal
            && close_i >= close_prev;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlonneck_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_equal,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_equal > period_body_long ? period_equal : period_body_long) + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_equal = 0.0f;
        float sum_long = 0.0f;
        for (int j = i - period_equal - 1; j < i - 1; ++j) {
            sum_equal += body[j];
        }
        for (int j = i - period_body_long - 1; j < i - 1; ++j) {
            sum_long += body[j];
        }

        const float avg_equal = sum_equal / (float)period_equal;
        const float avg_long = sum_long / (float)period_body_long;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float low_prev = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const bool hit = direction[i - 1] < 0
            && body[i - 1] > avg_long
            && direction[i] >= 0
            && open_i < low_prev
            && close_i <= low_prev + avg_equal
            && close_i >= low_prev - avg_equal;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlpiercing_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_body_long + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_prev = 0.0f;
        float sum_curr = 0.0f;
        for (int j = i - period_body_long; j < i; ++j) {
            sum_prev += body[j - 1];
            sum_curr += body[j];
        }

        const float avg_prev = sum_prev / (float)period_body_long;
        const float avg_curr = sum_curr / (float)period_body_long;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float open_prev = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float low_prev = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const bool hit = direction[i - 1] < 0
            && body[i - 1] > avg_prev
            && direction[i] >= 0
            && body[i] > avg_curr
            && open_i < low_prev
            && close_i < open_prev
            && close_i > close_prev + body[i - 1] * 0.5f;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlthrusting_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_equal,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_equal > period_body_long ? period_equal : period_body_long) + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_equal = 0.0f;
        float sum_long = 0.0f;
        for (int j = i - period_equal - 1; j < i - 1; ++j) {
            sum_equal += body[j];
        }
        for (int j = i - period_body_long - 1; j < i - 1; ++j) {
            sum_long += body[j];
        }

        const float avg_equal = sum_equal / (float)period_equal;
        const float avg_long = sum_long / (float)period_body_long;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float low_prev = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const bool hit = direction[i - 1] < 0
            && body[i - 1] > avg_long
            && direction[i] >= 0
            && open_i < low_prev
            && close_i > close_prev + avg_equal
            && close_i <= close_prev + body[i - 1] * 0.5f;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdleveningdojistar_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    int len,
    int period_long,
    int period_doji,
    int period_short,
    float penetration,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_long;
    if (period_doji > lookback) {
        lookback = period_doji;
    }
    if (period_short > lookback) {
        lookback = period_short;
    }
    lookback += 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        float sum_doji = 0.0f;
        float sum_short = 0.0f;
        for (int j = i - 2 - period_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        for (int j = i - 1 - period_doji; j < i - 1; ++j) {
            sum_doji += body[j];
        }
        for (int j = i - period_short; j < i; ++j) {
            sum_short += body[j];
        }

        const float avg_long = sum_long / (float)period_long;
        const float avg_doji = sum_doji / (float)period_doji;
        const float avg_short = sum_short / (float)period_short;
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const bool hit = body[i - 2] > avg_long
            && direction[i - 2] >= 0
            && body[i - 1] <= avg_doji
            && body_gap_up[i - 1] != 0u
            && body[i] > avg_short
            && direction[i] < 0
            && close_i < close_prev2 - body[i - 2] * penetration;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdleveningstar_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    int len,
    int period_long,
    int period_short1,
    int period_short0,
    float penetration,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_long;
    if (period_short1 > lookback) {
        lookback = period_short1;
    }
    if (period_short0 > lookback) {
        lookback = period_short0;
    }
    lookback += 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        float sum_short1 = 0.0f;
        float sum_short0 = 0.0f;
        for (int j = i - 2 - period_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        for (int j = i - 1 - period_short1; j < i - 1; ++j) {
            sum_short1 += body[j];
        }
        for (int j = i - period_short0; j < i; ++j) {
            sum_short0 += body[j];
        }

        const float avg_long = sum_long / (float)period_long;
        const float avg_short1 = sum_short1 / (float)period_short1;
        const float avg_short0 = sum_short0 / (float)period_short0;
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const bool hit = body[i - 2] > avg_long
            && direction[i - 2] >= 0
            && body[i - 1] <= avg_short1
            && body_gap_up[i - 1] != 0u
            && body[i] > avg_short0
            && direction[i] < 0
            && close_i < close_prev2 - body[i - 2] * penetration;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlmorningdojistar_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    int period_long,
    int period_doji,
    int period_short,
    float penetration,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_long;
    if (period_doji > lookback) {
        lookback = period_doji;
    }
    if (period_short > lookback) {
        lookback = period_short;
    }
    lookback += 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        float sum_doji = 0.0f;
        float sum_short = 0.0f;
        for (int j = i - 2 - period_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        for (int j = i - 1 - period_doji; j < i - 1; ++j) {
            sum_doji += body[j];
        }
        for (int j = i - period_short; j < i; ++j) {
            sum_short += body[j];
        }

        const float avg_long = sum_long / (float)period_long;
        const float avg_doji = sum_doji / (float)period_doji;
        const float avg_short = sum_short / (float)period_short;
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const bool hit = body[i - 2] > avg_long
            && direction[i - 2] < 0
            && body[i - 1] <= avg_doji
            && body_gap_down[i - 1] != 0u
            && body[i] > avg_short
            && direction[i] >= 0
            && close_i > close_prev2 + body[i - 2] * penetration;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlmorningstar_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    int period_long,
    int period_short1,
    int period_short0,
    float penetration,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_long;
    if (period_short1 > lookback) {
        lookback = period_short1;
    }
    if (period_short0 > lookback) {
        lookback = period_short0;
    }
    lookback += 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        float sum_short1 = 0.0f;
        float sum_short0 = 0.0f;
        for (int j = i - 2 - period_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        for (int j = i - 1 - period_short1; j < i - 1; ++j) {
            sum_short1 += body[j];
        }
        for (int j = i - period_short0; j < i; ++j) {
            sum_short0 += body[j];
        }

        const float avg_long = sum_long / (float)period_long;
        const float avg_short1 = sum_short1 / (float)period_short1;
        const float avg_short0 = sum_short0 / (float)period_short0;
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const bool hit = body[i - 2] > avg_long
            && direction[i - 2] < 0
            && body[i - 1] <= avg_short1
            && body_gap_down[i - 1] != 0u
            && body[i] > avg_short0
            && direction[i] >= 0
            && close_i > close_prev2 + body[i - 2] * penetration;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlgapsidesidewhite_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    int period_near,
    int period_equal,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_near > period_equal ? period_near : period_equal) + 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_near = 0.0f;
        float sum_equal = 0.0f;
        for (int j = i - period_near - 1; j < i - 1; ++j) {
            sum_near += body[j];
        }
        for (int j = i - period_equal - 1; j < i - 1; ++j) {
            sum_equal += body[j];
        }

        const float avg_near = sum_near / (float)period_near;
        const float avg_equal = sum_equal / (float)period_equal;
        const bool gap_up_1 = body_gap_up[i - 1] != 0u;
        const bool gap_down_1 = body_gap_down[i - 1] != 0u;
        const bool gap_up_0 = body_low[i] > body_high[i - 2];
        const bool gap_down_0 = body_high[i] < body_low[i - 2];
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float open_prev = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const bool hit = ((gap_up_1 && gap_up_0) || (gap_down_1 && gap_down_0))
            && direction[i - 1] >= 0
            && direction[i] >= 0
            && body[i] >= body[i - 1] - avg_near
            && body[i] <= body[i - 1] + avg_near
            && open_i >= open_prev - avg_equal
            && open_i <= open_prev + avg_equal;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlkicking_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    int period_shadow,
    int period_body,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_shadow > period_body ? period_shadow : period_body) + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_sh_prev = 0.0f;
        float sum_sh_curr = 0.0f;
        float sum_body_prev = 0.0f;
        float sum_body_curr = 0.0f;
        for (int j = i - 1 - period_shadow; j < i - 1; ++j) {
            sum_sh_prev += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        for (int j = i - period_shadow; j < i; ++j) {
            sum_sh_curr += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        for (int j = i - 1 - period_body; j < i - 1; ++j) {
            sum_body_prev += body[j];
        }
        for (int j = i - period_body; j < i; ++j) {
            sum_body_curr += body[j];
        }

        const float avg_sh_prev = sum_sh_prev / (float)period_shadow;
        const float avg_sh_curr = sum_sh_curr / (float)period_shadow;
        const float avg_body_prev = sum_body_prev / (float)period_body;
        const float avg_body_curr = sum_body_curr / (float)period_body;
        const bool gap_cond = (direction[i - 1] < 0 && body_gap_up[i] != 0u)
            || (direction[i - 1] >= 0 && body_gap_down[i] != 0u);
        const bool hit = direction[i - 1] == -direction[i]
            && body[i - 1] > avg_body_prev
            && fmaxf(upper_shadow[i - 1], lower_shadow[i - 1]) < avg_sh_prev
            && body[i] > avg_body_curr
            && fmaxf(upper_shadow[i], lower_shadow[i]) < avg_sh_curr
            && gap_cond;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlkickingbylength_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    int period_shadow,
    int period_body,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_shadow > period_body ? period_shadow : period_body) + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_sh_prev = 0.0f;
        float sum_sh_curr = 0.0f;
        float sum_body_prev = 0.0f;
        float sum_body_curr = 0.0f;
        for (int j = i - 1 - period_shadow; j < i - 1; ++j) {
            sum_sh_prev += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        for (int j = i - period_shadow; j < i; ++j) {
            sum_sh_curr += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        for (int j = i - 1 - period_body; j < i - 1; ++j) {
            sum_body_prev += body[j];
        }
        for (int j = i - period_body; j < i; ++j) {
            sum_body_curr += body[j];
        }

        const float avg_sh_prev = sum_sh_prev / (float)period_shadow;
        const float avg_sh_curr = sum_sh_curr / (float)period_shadow;
        const float avg_body_prev = sum_body_prev / (float)period_body;
        const float avg_body_curr = sum_body_curr / (float)period_body;
        const bool gap_cond = (direction[i - 1] < 0 && body_gap_up[i] != 0u)
            || (direction[i - 1] >= 0 && body_gap_down[i] != 0u);
        const bool hit = direction[i - 1] == -direction[i]
            && body[i - 1] > avg_body_prev
            && fmaxf(upper_shadow[i - 1], lower_shadow[i - 1]) < avg_sh_prev
            && body[i] > avg_body_curr
            && fmaxf(upper_shadow[i], lower_shadow[i]) < avg_sh_curr
            && gap_cond;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlidentical3crows_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_shadow,
    int period_equal,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_shadow > period_equal ? period_shadow : period_equal) + 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_sh2 = 0.0f;
        float sum_sh1 = 0.0f;
        float sum_sh0 = 0.0f;
        float sum_eq2 = 0.0f;
        float sum_eq1 = 0.0f;
        for (int j = i - 2 - period_shadow; j < i - 2; ++j) {
            sum_sh2 += lower_shadow[j];
        }
        for (int j = i - 1 - period_shadow; j < i - 1; ++j) {
            sum_sh1 += lower_shadow[j];
        }
        for (int j = i - period_shadow; j < i; ++j) {
            sum_sh0 += lower_shadow[j];
        }
        for (int j = i - 2 - period_equal; j < i - 2; ++j) {
            sum_eq2 += body[j];
        }
        for (int j = i - 1 - period_equal; j < i - 1; ++j) {
            sum_eq1 += body[j];
        }

        const float avg_sh2 = sum_sh2 / (float)period_shadow;
        const float avg_sh1 = sum_sh1 / (float)period_shadow;
        const float avg_sh0 = sum_sh0 / (float)period_shadow;
        const float avg_eq2 = sum_eq2 / (float)period_equal;
        const float avg_eq1 = sum_eq1 / (float)period_equal;
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);
        const float open1 = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open0 = pr_open(body_low[i], body_high[i], direction[i]);
        const bool hit = direction[i - 2] < 0
            && direction[i - 1] < 0
            && direction[i] < 0
            && lower_shadow[i - 2] < avg_sh2
            && lower_shadow[i - 1] < avg_sh1
            && lower_shadow[i] < avg_sh0
            && close2 > close1
            && close1 > close0
            && open1 <= close2 + avg_eq2
            && open1 >= close2 - avg_eq2
            && open0 <= close1 + avg_eq1
            && open0 >= close1 - avg_eq1;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlsticksandwich_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_equal,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_equal + 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_equal = 0.0f;
        for (int j = i - 2 - period_equal; j < i - 2; ++j) {
            sum_equal += body[j];
        }
        const float avg_equal = sum_equal / (float)period_equal;
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);
        const float low1 = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const bool hit = direction[i - 2] < 0
            && direction[i - 1] >= 0
            && direction[i] < 0
            && low1 > close2
            && close0 <= close2 + avg_equal
            && close0 >= close2 - avg_equal;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlseparatinglines_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_shadow,
    int period_body_long,
    int period_equal,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_shadow;
    if (period_body_long > lookback) {
        lookback = period_body_long;
    }
    if (period_equal > lookback) {
        lookback = period_equal;
    }
    lookback += 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_shadow = 0.0f;
        float sum_body = 0.0f;
        float sum_equal = 0.0f;
        for (int j = i - period_shadow; j < i; ++j) {
            sum_shadow += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        for (int j = i - period_body_long; j < i; ++j) {
            sum_body += body[j];
        }
        for (int j = i - 1 - period_equal; j < i - 1; ++j) {
            sum_equal += body[j];
        }

        const float avg_shadow = sum_shadow / (float)period_shadow;
        const float avg_body = sum_body / (float)period_body_long;
        const float avg_equal = sum_equal / (float)period_equal;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float open_prev = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const bool hit = direction[i - 1] == -direction[i]
            && open_i <= open_prev + avg_equal
            && open_i >= open_prev - avg_equal
            && body[i] > avg_body
            && ((direction[i] >= 0 && lower_shadow[i] < avg_shadow)
                || (direction[i] < 0 && upper_shadow[i] < avg_shadow));
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlcounterattack_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    int len,
    int period_equal,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_equal > period_body_long ? period_equal : period_body_long) + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_equal = 0.0f;
        float sum_long_prev = 0.0f;
        float sum_long_curr = 0.0f;
        for (int j = i - period_equal - 1; j < i - 1; ++j) {
            sum_equal += body[j];
        }
        for (int j = i - period_body_long - 1; j < i - 1; ++j) {
            sum_long_prev += body[j];
        }
        for (int j = i - period_body_long; j < i; ++j) {
            sum_long_curr += body[j];
        }

        const float avg_equal = sum_equal / (float)period_equal;
        const float avg_long_prev = sum_long_prev / (float)period_body_long;
        const float avg_long_curr = sum_long_curr / (float)period_body_long;
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const bool hit = direction[i] == -direction[i - 1]
            && body[i - 1] > avg_long_prev
            && body[i] > avg_long_curr
            && close_i <= close_prev + avg_equal
            && close_i >= close_prev - avg_equal;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdldarkcloudcover_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    float penetration,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_body_long + 1;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        for (int j = i - period_body_long - 1; j < i - 1; ++j) {
            sum_long += body[j];
        }

        const float avg_long = sum_long / (float)period_body_long;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float open_prev = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float high_prev = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const bool hit = direction[i - 1] >= 0
            && body[i - 1] > avg_long
            && direction[i] < 0
            && open_i > high_prev
            && close_i > open_prev
            && close_i < close_prev - body[i - 1] * penetration;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlxsidegap3methods_u8_kernel(
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const bool gap_ok = (direction[i - 2] >= 0 && body_gap_up[i - 1] != 0u)
            || (direction[i - 2] < 0 && body_gap_down[i - 1] != 0u);
        const bool hit = direction[i - 2] == direction[i - 1]
            && direction[i] == -direction[i - 1]
            && open_i < body_high[i - 1]
            && open_i > body_low[i - 1]
            && close_i < body_high[i - 2]
            && close_i > body_low[i - 2]
            && gap_ok;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlupsidegap2crows_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    int len,
    int period_body_short,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_body_short > period_body_long ? period_body_short : period_body_long) + 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        float sum_short = 0.0f;
        for (int j = i - 2 - period_body_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        for (int j = i - 1 - period_body_short; j < i - 1; ++j) {
            sum_short += body[j];
        }
        const float avg_long = sum_long / (float)period_body_long;
        const float avg_short = sum_short / (float)period_body_short;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float open_prev = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close_prev2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const bool hit = direction[i - 2] >= 0
            && body[i - 2] > avg_long
            && direction[i - 1] < 0
            && body[i - 1] <= avg_short
            && body_gap_up[i - 1] != 0u
            && direction[i] < 0
            && open_i > open_prev
            && close_i < close_prev
            && close_i > close_prev2;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlunique3river_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_short,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = (period_body_short > period_body_long ? period_body_short : period_body_long) + 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        float sum_short = 0.0f;
        for (int j = i - 2 - period_body_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        for (int j = i - period_body_short; j < i; ++j) {
            sum_short += body[j];
        }
        const float avg_long = sum_long / (float)period_body_long;
        const float avg_short = sum_short / (float)period_body_short;
        const float close_prev2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open_prev = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open_prev2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float low_prev = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const float low_prev2 = pr_low(body_low[i - 2], lower_shadow[i - 2]);
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const bool hit = body[i - 2] > avg_long
            && direction[i - 2] < 0
            && direction[i - 1] < 0
            && close_prev > close_prev2
            && open_prev <= open_prev2
            && low_prev < low_prev2
            && body[i] < avg_short
            && direction[i] >= 0
            && open_i > low_prev;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdltasukigap_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    int period_near,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_near + 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_near = 0.0f;
        for (int j = i - period_near - 1; j < i - 1; ++j) {
            sum_near += body[j];
        }
        const float avg_near = sum_near / (float)period_near;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float open_prev = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close_prev = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open_prev2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close_prev2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);

        const bool up_variant = body_gap_up[i - 1] != 0u
            && direction[i - 1] >= 0
            && direction[i] < 0
            && open_i < close_prev
            && open_i > open_prev
            && close_i < open_prev
            && close_i > fmaxf(close_prev2, open_prev2)
            && fabsf(body[i - 1] - body[i]) < avg_near;

        const bool down_variant = body_gap_down[i - 1] != 0u
            && direction[i - 1] < 0
            && direction[i] >= 0
            && open_i < open_prev
            && open_i > close_prev
            && close_i > open_prev
            && close_i < fminf(close_prev2, open_prev2)
            && fabsf(body[i - 1] - body[i]) < avg_near;

        matrix[base + i] = (up_variant || down_variant) ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlladderbottom_u8_kernel(
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_shadow_very_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_shadow_very_short + 4;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_shadow = 0.0f;
        for (int j = i - period_shadow_very_short - 1; j < i - 1; ++j) {
            sum_shadow += upper_shadow[j];
        }
        const float avg_shadow = sum_shadow / (float)period_shadow_very_short;
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const float open_prev = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float high_prev = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const float open4 = pr_open(body_low[i - 4], body_high[i - 4], direction[i - 4]);
        const float open3 = pr_open(body_low[i - 3], body_high[i - 3], direction[i - 3]);
        const float open2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close4 = pr_close(body_low[i - 4], body_high[i - 4], direction[i - 4]);
        const float close3 = pr_close(body_low[i - 3], body_high[i - 3], direction[i - 3]);
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const bool hit = direction[i - 4] < 0
            && direction[i - 3] < 0
            && direction[i - 2] < 0
            && open4 > open3
            && open3 > open2
            && close4 > close3
            && close3 > close2
            && direction[i - 1] < 0
            && upper_shadow[i - 1] > avg_shadow
            && direction[i] >= 0
            && open_i > open_prev
            && close_i > high_prev;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlstalledpattern_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    int period_body_short,
    int period_shadow_very_short,
    int period_near,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int lookback = period_body_long;
    if (period_body_short > lookback) {
        lookback = period_body_short;
    }
    if (period_shadow_very_short > lookback) {
        lookback = period_shadow_very_short;
    }
    if (period_near > lookback) {
        lookback = period_near;
    }
    lookback += 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long2 = 0.0f;
        float sum_long1 = 0.0f;
        float sum_short0 = 0.0f;
        float sum_shadow1 = 0.0f;
        float sum_near2 = 0.0f;
        float sum_near1 = 0.0f;
        for (int j = i - period_body_long; j < i; ++j) {
            sum_long2 += body[j - 2];
            sum_long1 += body[j - 1];
        }
        for (int j = i - period_body_short; j < i; ++j) {
            sum_short0 += body[j];
        }
        for (int j = i - period_shadow_very_short; j < i; ++j) {
            sum_shadow1 += upper_shadow[j - 1];
        }
        for (int j = i - period_near; j < i; ++j) {
            sum_near2 += body[j - 2];
            sum_near1 += body[j - 1];
        }

        const float avg_long2 = sum_long2 / (float)period_body_long;
        const float avg_long1 = sum_long1 / (float)period_body_long;
        const float avg_short0 = sum_short0 / (float)period_body_short;
        const float avg_shadow1 = sum_shadow1 / (float)period_shadow_very_short;
        const float avg_near2 = sum_near2 / (float)period_near;
        const float avg_near1 = sum_near1 / (float)period_near;
        const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
        const float close_1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close_2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float open_1 = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open_2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float open_i = pr_open(body_low[i], body_high[i], direction[i]);
        const bool hit = direction[i - 2] >= 0
            && direction[i - 1] >= 0
            && direction[i] >= 0
            && close_i > close_1
            && close_1 > close_2
            && body[i - 2] > avg_long2
            && body[i - 1] > avg_long1
            && upper_shadow[i - 1] < avg_shadow1
            && open_1 > open_2
            && open_1 <= close_2 + avg_near2
            && body[i] < avg_short0
            && open_i >= close_1 - body[i] - avg_near1;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlhikkake_u8_kernel(
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) {
        return;
    }
    const int base = row * cols;
    if (len <= 5) {
        return;
    }

    int pattern_idx = 0;
    int pattern_result = 0;
    for (int i = 5; i < len; ++i) {
        const float high_i = pr_high(body_high[i], upper_shadow[i]);
        const float low_i = pr_low(body_low[i], lower_shadow[i]);

        if (pattern_idx != 0 && i <= (pattern_idx + 3)) {
            const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
            const float high_p1 = pr_high(body_high[pattern_idx - 1], upper_shadow[pattern_idx - 1]);
            const float low_p1 = pr_low(body_low[pattern_idx - 1], lower_shadow[pattern_idx - 1]);
            if ((pattern_result > 0 && close_i > high_p1) || (pattern_result < 0 && close_i < low_p1)) {
                matrix[base + i] = 1;
                pattern_idx = 0;
            }
        }

        const float high_1 = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const float low_1 = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const float high_2 = pr_high(body_high[i - 2], upper_shadow[i - 2]);
        const float low_2 = pr_low(body_low[i - 2], lower_shadow[i - 2]);

        if (high_1 < high_2 && low_1 > low_2) {
            if (high_i < high_1 && low_i < low_1) {
                matrix[base + i] = 1;
                pattern_result = 1;
                pattern_idx = i;
            } else if (high_i > high_1 && low_i > low_1) {
                matrix[base + i] = 1;
                pattern_result = -1;
                pattern_idx = i;
            }
        }
    }
}

extern "C" __global__ void pattern_row_cdlhikkakemod_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_near,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) {
        return;
    }
    const int base = row * cols;
    const int lookback = period_near + 5;
    if (len <= lookback) {
        return;
    }

    int pattern_idx = 0;
    int pattern_result = 0;
    for (int i = lookback; i < len; ++i) {
        const float high_i = pr_high(body_high[i], upper_shadow[i]);
        const float low_i = pr_low(body_low[i], lower_shadow[i]);
        if (pattern_idx != 0 && i <= (pattern_idx + 3)) {
            const float close_i = pr_close(body_low[i], body_high[i], direction[i]);
            const float high_p1 = pr_high(body_high[pattern_idx - 1], upper_shadow[pattern_idx - 1]);
            const float low_p1 = pr_low(body_low[pattern_idx - 1], lower_shadow[pattern_idx - 1]);
            if ((pattern_result > 0 && close_i > high_p1) || (pattern_result < 0 && close_i < low_p1)) {
                matrix[base + i] = 1;
                pattern_idx = 0;
            }
        }

        float sum_near = 0.0f;
        for (int j = i - 2 - period_near; j < i - 2; ++j) {
            sum_near += body[j];
        }
        const float near_avg = sum_near / (float)period_near;
        const float high_3 = pr_high(body_high[i - 3], upper_shadow[i - 3]);
        const float low_3 = pr_low(body_low[i - 3], lower_shadow[i - 3]);
        const float high_2 = pr_high(body_high[i - 2], upper_shadow[i - 2]);
        const float low_2 = pr_low(body_low[i - 2], lower_shadow[i - 2]);
        const float high_1 = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const float low_1 = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const float close_2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);

        if (high_2 < high_3
            && low_2 > low_3
            && high_1 < high_2
            && low_1 > low_2)
        {
            if (high_i < high_1 && low_i < low_1 && close_2 <= low_2 + near_avg) {
                matrix[base + i] = 1;
                pattern_result = 1;
                pattern_idx = i;
            } else if (high_i > high_1 && low_i > low_1 && close_2 >= high_2 - near_avg) {
                matrix[base + i] = 1;
                pattern_result = -1;
                pattern_idx = i;
            }
        }
    }
}

extern "C" __global__ void pattern_row_cdl2crows_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 2 + period_body_long;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        for (int j = i - 2 - period_body_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        const float avg_long = sum_long / (float)period_body_long;

        const int8_t first_color = direction[i - 2];
        const int8_t second_color = direction[i - 1];
        const int8_t third_color = direction[i];

        const float first_open = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float first_close = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float second_open = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float second_close = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float third_open = pr_open(body_low[i], body_high[i], direction[i]);
        const float third_close = pr_close(body_low[i], body_high[i], direction[i]);

        const bool hit = first_color > 0
            && body[i - 2] > avg_long
            && second_color < 0
            && body_low[i - 1] > body_high[i - 2]
            && third_color < 0
            && third_open < second_open
            && third_open > second_close
            && third_close > first_open
            && third_close < first_close;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdl3blackcrows_u8_kernel(
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_shadow_very_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 3 + period_shadow_very_short;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum2 = 0.0f;
        float sum1 = 0.0f;
        float sum0 = 0.0f;
        for (int j = i - 3 - period_shadow_very_short; j < i - 3; ++j) {
            sum2 += lower_shadow[j];
        }
        for (int j = i - 2 - period_shadow_very_short; j < i - 2; ++j) {
            sum1 += lower_shadow[j];
        }
        for (int j = i - 1 - period_shadow_very_short; j < i - 1; ++j) {
            sum0 += lower_shadow[j];
        }
        const float avg2 = sum2 / (float)period_shadow_very_short;
        const float avg1 = sum1 / (float)period_shadow_very_short;
        const float avg0 = sum0 / (float)period_shadow_very_short;

        const float open2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float open1 = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open0 = pr_open(body_low[i], body_high[i], direction[i]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);
        const float high3 = pr_high(body_high[i - 3], upper_shadow[i - 3]);

        const bool hit = direction[i - 3] > 0
            && direction[i - 2] < 0
            && lower_shadow[i - 2] < avg2
            && direction[i - 1] < 0
            && lower_shadow[i - 1] < avg1
            && direction[i] < 0
            && lower_shadow[i] < avg0
            && open1 < open2
            && open1 > close2
            && open0 < open1
            && open0 > close1
            && high3 > close2
            && close2 > close1
            && close1 > close0;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdl3inside_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    int period_body_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 2 + ((period_body_long > period_body_short) ? period_body_long : period_body_short);

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        for (int j = i - 2 - period_body_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        float sum_short = 0.0f;
        for (int j = i - 1 - period_body_short; j < i - 1; ++j) {
            sum_short += body[j];
        }
        const float avg_long = sum_long / (float)period_body_long;
        const float avg_short = sum_short / (float)period_body_short;

        const float open2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);
        const bool reversal = (direction[i - 2] > 0 && direction[i] < 0 && close0 < open2)
            || (direction[i - 2] < 0 && direction[i] > 0 && close0 > open2);

        const bool hit = body[i - 2] > avg_long
            && body[i - 1] <= avg_short
            && body_high[i - 1] < body_high[i - 2]
            && body_low[i - 1] > body_low[i - 2]
            && reversal;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdl3linestrike_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    int len,
    int period_near,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 3 + period_near;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum3 = 0.0f;
        float sum2 = 0.0f;
        for (int j = i - 3 - period_near; j < i - 3; ++j) {
            sum3 += body[j];
        }
        for (int j = i - 2 - period_near; j < i - 2; ++j) {
            sum2 += body[j];
        }
        const float avg3 = sum3 / (float)period_near;
        const float avg2 = sum2 / (float)period_near;

        const float open3 = pr_open(body_low[i - 3], body_high[i - 3], direction[i - 3]);
        const float close3 = pr_close(body_low[i - 3], body_high[i - 3], direction[i - 3]);
        const float open2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float open1 = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open0 = pr_open(body_low[i], body_high[i], direction[i]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);

        const int8_t dir3 = direction[i - 3];
        const int8_t dir2 = direction[i - 2];
        const int8_t dir1 = direction[i - 1];
        const int8_t dir0 = direction[i];

        const bool directional = dir3 == dir2 && dir2 == dir1 && dir0 == -dir1;
        const bool near_opens = open2 >= body_low[i - 3] - avg3
            && open2 <= body_high[i - 3] + avg3
            && open1 >= body_low[i - 2] - avg2
            && open1 <= body_high[i - 2] + avg2;

        const bool bullish_case = dir1 > 0
            && close1 > close2
            && close2 > close3
            && open0 > close1
            && close0 < open3;
        const bool bearish_case = dir1 < 0
            && close1 < close2
            && close2 < close3
            && open0 < close1
            && close0 > open3;

        const bool hit = directional && near_opens && (bullish_case || bearish_case);
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdl3outside_u8_kernel(
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const int8_t* __restrict__ direction,
    int len,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;

    for (int i = tid; i < len; i += stride) {
        if (i < 2) {
            matrix[base + i] = 0;
            continue;
        }

        const float open2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float open1 = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);

        const bool white_engulfs_black = direction[i - 1] > 0
            && direction[i - 2] < 0
            && close1 > open2
            && open1 < close2
            && close0 > close1;
        const bool black_engulfs_white = direction[i - 1] < 0
            && direction[i - 2] > 0
            && open1 > close2
            && close1 < open2
            && close0 < close1;
        matrix[base + i] = (white_engulfs_black || black_engulfs_white) ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdl3starsinsouth_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    int period_shadow_long,
    int period_shadow_very_short,
    int period_body_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int max_period = period_body_long;
    if (period_shadow_long > max_period) {
        max_period = period_shadow_long;
    }
    if (period_shadow_very_short > max_period) {
        max_period = period_shadow_very_short;
    }
    if (period_body_short > max_period) {
        max_period = period_body_short;
    }
    const int lookback = 2 + max_period;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body_long = 0.0f;
        float sum_shadow_long = 0.0f;
        float sum_shadow_short_1 = 0.0f;
        float sum_shadow_short_0 = 0.0f;
        float sum_body_short = 0.0f;

        for (int j = i - 2 - period_body_long; j < i - 2; ++j) {
            sum_body_long += body[j];
        }
        for (int j = i - 2 - period_shadow_long; j < i - 2; ++j) {
            sum_shadow_long += body[j];
        }
        for (int j = i - 1 - period_shadow_very_short; j < i - 1; ++j) {
            sum_shadow_short_1 += lower_shadow[j];
        }
        for (int j = i - period_shadow_very_short; j < i; ++j) {
            sum_shadow_short_0 += lower_shadow[j];
        }
        for (int j = i - period_body_short; j < i; ++j) {
            sum_body_short += body[j];
        }

        const float avg_body_long = sum_body_long / (float)period_body_long;
        const float avg_shadow_long = sum_shadow_long / (float)period_shadow_long;
        const float avg_shadow_short_1 = sum_shadow_short_1 / (float)period_shadow_very_short;
        const float avg_shadow_short_0 = sum_shadow_short_0 / (float)period_shadow_very_short;
        const float avg_body_short = sum_body_short / (float)period_body_short;

        const float open1 = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float low2 = pr_low(body_low[i - 2], lower_shadow[i - 2]);
        const float low1 = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const float low0 = pr_low(body_low[i], lower_shadow[i]);
        const float high2 = pr_high(body_high[i - 2], upper_shadow[i - 2]);
        const float high1 = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const float high0 = pr_high(body_high[i], upper_shadow[i]);

        const bool hit = direction[i - 2] < 0
            && direction[i - 1] < 0
            && direction[i] < 0
            && body[i - 2] > avg_body_long
            && lower_shadow[i - 2] > avg_shadow_long
            && body[i - 1] < body[i - 2]
            && open1 > close2
            && open1 <= high2
            && low1 < close2
            && low1 >= low2
            && lower_shadow[i - 1] > avg_shadow_short_1
            && body[i] < avg_body_short
            && lower_shadow[i] < avg_shadow_short_0
            && upper_shadow[i] < avg_shadow_short_0
            && low0 > low1
            && high0 < high1;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdl3whitesoldiers_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_shadow_very_short,
    int period_near,
    int period_far,
    int period_body_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int max_period = period_shadow_very_short;
    if (period_near > max_period) {
        max_period = period_near;
    }
    if (period_far > max_period) {
        max_period = period_far;
    }
    if (period_body_short > max_period) {
        max_period = period_body_short;
    }
    const int lookback = 2 + max_period;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_sv2 = 0.0f;
        float sum_sv1 = 0.0f;
        float sum_sv0 = 0.0f;
        float sum_near2 = 0.0f;
        float sum_near1 = 0.0f;
        float sum_far2 = 0.0f;
        float sum_far1 = 0.0f;
        float sum_body_short = 0.0f;

        for (int j = i - 2 - period_shadow_very_short; j < i - 2; ++j) {
            sum_sv2 += upper_shadow[j];
        }
        for (int j = i - 1 - period_shadow_very_short; j < i - 1; ++j) {
            sum_sv1 += upper_shadow[j];
        }
        for (int j = i - period_shadow_very_short; j < i; ++j) {
            sum_sv0 += upper_shadow[j];
        }
        for (int j = i - 2 - period_near; j < i - 2; ++j) {
            sum_near2 += body[j];
        }
        for (int j = i - 1 - period_near; j < i - 1; ++j) {
            sum_near1 += body[j];
        }
        for (int j = i - 2 - period_far; j < i - 2; ++j) {
            sum_far2 += body[j];
        }
        for (int j = i - 1 - period_far; j < i - 1; ++j) {
            sum_far1 += body[j];
        }
        for (int j = i - period_body_short; j < i; ++j) {
            sum_body_short += body[j];
        }

        const float avg_sv2 = sum_sv2 / (float)period_shadow_very_short;
        const float avg_sv1 = sum_sv1 / (float)period_shadow_very_short;
        const float avg_sv0 = sum_sv0 / (float)period_shadow_very_short;
        const float avg_near2 = sum_near2 / (float)period_near;
        const float avg_near1 = sum_near1 / (float)period_near;
        const float avg_far2 = sum_far2 / (float)period_far;
        const float avg_far1 = sum_far1 / (float)period_far;
        const float avg_body_short = sum_body_short / (float)period_body_short;

        const float open2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float open1 = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open0 = pr_open(body_low[i], body_high[i], direction[i]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);

        const bool hit = direction[i - 2] > 0
            && upper_shadow[i - 2] < avg_sv2
            && direction[i - 1] > 0
            && upper_shadow[i - 1] < avg_sv1
            && direction[i] > 0
            && upper_shadow[i] < avg_sv0
            && close0 > close1
            && close1 > close2
            && open1 > open2
            && open1 <= close2 + avg_near2
            && open0 > open1
            && open0 <= close1 + avg_near1
            && body[i - 1] > body[i - 2] - avg_far2
            && body[i] > body[i - 1] - avg_far1
            && body[i] > avg_body_short;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlabandonedbaby_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    int period_body_doji,
    int period_body_short,
    float penetration,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int max_period = period_body_long;
    if (period_body_doji > max_period) {
        max_period = period_body_doji;
    }
    if (period_body_short > max_period) {
        max_period = period_body_short;
    }
    const int lookback = 2 + max_period;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        float sum_doji = 0.0f;
        float sum_short = 0.0f;
        for (int j = i - 2 - period_body_long; j < i - 2; ++j) {
            sum_long += body[j];
        }
        for (int j = i - 1 - period_body_doji; j < i - 1; ++j) {
            sum_doji += body[j];
        }
        for (int j = i - period_body_short; j < i; ++j) {
            sum_short += body[j];
        }
        const float avg_long = sum_long / (float)period_body_long;
        const float avg_doji = sum_doji / (float)period_body_doji;
        const float avg_short = sum_short / (float)period_body_short;

        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);
        const float high2 = pr_high(body_high[i - 2], upper_shadow[i - 2]);
        const float low2 = pr_low(body_low[i - 2], lower_shadow[i - 2]);
        const float high1 = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const float low1 = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const float high0 = pr_high(body_high[i], upper_shadow[i]);
        const float low0 = pr_low(body_low[i], lower_shadow[i]);

        const bool bearish = direction[i - 2] > 0
            && direction[i] < 0
            && close0 < close2 - body[i - 2] * penetration
            && low1 > high2
            && high0 < low1;
        const bool bullish = direction[i - 2] < 0
            && direction[i] > 0
            && close0 > close2 + body[i - 2] * penetration
            && high1 < low2
            && low0 > high1;
        const bool hit = body[i - 2] > avg_long
            && body[i - 1] <= avg_doji
            && body[i] > avg_short
            && (bearish || bullish);
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdladvanceblock_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_shadow_short,
    int period_shadow_long,
    int period_near,
    int period_far,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    int max_period = period_shadow_short;
    if (period_shadow_long > max_period) {
        max_period = period_shadow_long;
    }
    if (period_near > max_period) {
        max_period = period_near;
    }
    if (period_far > max_period) {
        max_period = period_far;
    }
    if (period_body_long > max_period) {
        max_period = period_body_long;
    }
    const int lookback = 2 + max_period;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_shadow_short_2 = 0.0f;
        float sum_shadow_short_1 = 0.0f;
        float sum_shadow_short_0 = 0.0f;
        float sum_shadow_long_0 = 0.0f;
        float sum_near_2 = 0.0f;
        float sum_near_1 = 0.0f;
        float sum_far_2 = 0.0f;
        float sum_far_1 = 0.0f;
        float sum_body_long = 0.0f;

        for (int j = i - 2 - period_shadow_short; j < i - 2; ++j) {
            sum_shadow_short_2 += upper_shadow[j];
        }
        for (int j = i - 1 - period_shadow_short; j < i - 1; ++j) {
            sum_shadow_short_1 += upper_shadow[j];
        }
        for (int j = i - period_shadow_short; j < i; ++j) {
            sum_shadow_short_0 += upper_shadow[j];
        }
        for (int j = i - period_shadow_long; j < i; ++j) {
            sum_shadow_long_0 += upper_shadow[j];
        }
        for (int j = i - 2 - period_near; j < i - 2; ++j) {
            sum_near_2 += body[j];
        }
        for (int j = i - 1 - period_near; j < i - 1; ++j) {
            sum_near_1 += body[j];
        }
        for (int j = i - 2 - period_far; j < i - 2; ++j) {
            sum_far_2 += body[j];
        }
        for (int j = i - 1 - period_far; j < i - 1; ++j) {
            sum_far_1 += body[j];
        }
        for (int j = i - 2 - period_body_long; j < i - 2; ++j) {
            sum_body_long += body[j];
        }

        const float avg_shadow_short_2 = sum_shadow_short_2 / (float)period_shadow_short;
        const float avg_shadow_short_1 = sum_shadow_short_1 / (float)period_shadow_short;
        const float avg_shadow_short_0 = sum_shadow_short_0 / (float)period_shadow_short;
        const float avg_shadow_long_0 = sum_shadow_long_0 / (float)period_shadow_long;
        const float avg_near_2 = sum_near_2 / (float)period_near;
        const float avg_near_1 = sum_near_1 / (float)period_near;
        const float avg_far_2 = sum_far_2 / (float)period_far;
        const float avg_far_1 = sum_far_1 / (float)period_far;
        const float avg_body_long = sum_body_long / (float)period_body_long;

        const float open2 = pr_open(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float open1 = pr_open(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float close1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open0 = pr_open(body_low[i], body_high[i], direction[i]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);

        const bool base_hit = direction[i - 2] > 0
            && direction[i - 1] > 0
            && direction[i] > 0
            && close0 > close1
            && close1 > close2
            && open1 > open2
            && open1 <= close2 + avg_near_2
            && open0 > open1
            && open0 <= close1 + avg_near_1
            && body[i - 2] > avg_body_long
            && upper_shadow[i - 2] < avg_shadow_short_2;

        const bool cond_a = body[i - 1] < body[i - 2] - avg_far_2
            && body[i] < body[i - 1] + avg_near_1;
        const bool cond_b = body[i] < body[i - 1] - avg_far_1;
        const bool cond_c = body[i] < body[i - 1]
            && body[i - 1] < body[i - 2]
            && (upper_shadow[i] > avg_shadow_short_0 || upper_shadow[i - 1] > avg_shadow_short_1);
        const bool cond_d = body[i] < body[i - 1]
            && upper_shadow[i] > avg_shadow_long_0;

        matrix[base + i] = (base_hit && (cond_a || cond_b || cond_c || cond_d)) ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlbreakaway_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 4 + period_body_long;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_long = 0.0f;
        for (int j = i - 4 - period_body_long; j < i - 4; ++j) {
            sum_long += body[j];
        }
        const float avg_long = sum_long / (float)period_body_long;

        const int8_t c1 = direction[i - 4];
        const int8_t c2 = direction[i - 3];
        const int8_t c4 = direction[i - 1];
        const int8_t c5 = direction[i];

        const float open4 = pr_open(body_low[i - 4], body_high[i - 4], direction[i - 4]);
        const float close4 = pr_close(body_low[i - 4], body_high[i - 4], direction[i - 4]);
        const float open3 = pr_open(body_low[i - 3], body_high[i - 3], direction[i - 3]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);

        const float high3 = pr_high(body_high[i - 3], upper_shadow[i - 3]);
        const float high2 = pr_high(body_high[i - 2], upper_shadow[i - 2]);
        const float high1 = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const float low3 = pr_low(body_low[i - 3], lower_shadow[i - 3]);
        const float low2 = pr_low(body_low[i - 2], lower_shadow[i - 2]);
        const float low1 = pr_low(body_low[i - 1], lower_shadow[i - 1]);

        bool trend_path = false;
        if (c1 < 0) {
            const bool gap_down = body_high[i - 3] < body_low[i - 4];
            trend_path = gap_down
                && high2 < high3
                && low2 < low3
                && high1 < high2
                && low1 < low2
                && close0 > open3
                && close0 < close4;
        } else if (c1 > 0) {
            const bool gap_up = body_low[i - 3] > body_high[i - 4];
            trend_path = gap_up
                && high2 > high3
                && low2 > low3
                && high1 > high2
                && low1 > low2
                && close0 < open3
                && close0 > close4;
        }

        const bool hit = body[i - 4] > avg_long
            && c1 == c2
            && c2 == c4
            && c4 == -c5
            && trend_path;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlconcealbabyswall_u8_kernel(
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_shadow_very_short,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = 3 + period_shadow_very_short;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum3 = 0.0f;
        float sum2 = 0.0f;
        float sum1 = 0.0f;
        for (int j = i - 3 - period_shadow_very_short; j < i - 3; ++j) {
            sum3 += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        for (int j = i - 2 - period_shadow_very_short; j < i - 2; ++j) {
            sum2 += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        for (int j = i - 1 - period_shadow_very_short; j < i - 1; ++j) {
            sum1 += fmaxf(upper_shadow[j], lower_shadow[j]);
        }
        const float avg3 = sum3 / (float)period_shadow_very_short;
        const float avg2 = sum2 / (float)period_shadow_very_short;
        const float avg1 = sum1 / (float)period_shadow_very_short;

        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float high1 = pr_high(body_high[i - 1], upper_shadow[i - 1]);
        const float low1 = pr_low(body_low[i - 1], lower_shadow[i - 1]);
        const float high0 = pr_high(body_high[i], upper_shadow[i]);
        const float low0 = pr_low(body_low[i], lower_shadow[i]);

        const bool hit = direction[i - 3] < 0
            && direction[i - 2] < 0
            && direction[i - 1] < 0
            && direction[i] < 0
            && lower_shadow[i - 3] < avg3
            && upper_shadow[i - 3] < avg3
            && lower_shadow[i - 2] < avg2
            && upper_shadow[i - 2] < avg2
            && body_high[i - 1] < body_low[i - 2]
            && upper_shadow[i - 1] > avg1
            && high1 > close2
            && high0 > high1
            && low0 < low1;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlmathold_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_short,
    int period_body_long,
    float penetration,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int max_period = (period_body_short > period_body_long) ? period_body_short : period_body_long;
    const int lookback = max_period + 4;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum4 = 0.0f;
        float sum3 = 0.0f;
        float sum2 = 0.0f;
        float sum1 = 0.0f;
        for (int j = i - 4 - period_body_long; j < i - 4; ++j) {
            sum4 += body[j];
        }
        for (int j = i - 3 - period_body_short; j < i - 3; ++j) {
            sum3 += body[j];
        }
        for (int j = i - 2 - period_body_short; j < i - 2; ++j) {
            sum2 += body[j];
        }
        for (int j = i - 1 - period_body_short; j < i - 1; ++j) {
            sum1 += body[j];
        }
        const float avg4 = sum4 / (float)period_body_long;
        const float avg3 = sum3 / (float)period_body_short;
        const float avg2 = sum2 / (float)period_body_short;
        const float avg1 = sum1 / (float)period_body_short;

        const float open3 = pr_open(body_low[i - 3], body_high[i - 3], direction[i - 3]);
        const float close4 = pr_close(body_low[i - 4], body_high[i - 4], direction[i - 4]);
        const float close1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open0 = pr_open(body_low[i], body_high[i], direction[i]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);
        const float high3 = pr_high(body_high[i - 3], upper_shadow[i - 3]);
        const float high2 = pr_high(body_high[i - 2], upper_shadow[i - 2]);
        const float high1 = pr_high(body_high[i - 1], upper_shadow[i - 1]);

        const bool hit = body[i - 4] > avg4
            && body[i - 3] < avg3
            && body[i - 2] < avg2
            && body[i - 1] < avg1
            && direction[i - 4] > 0
            && direction[i - 3] < 0
            && direction[i] > 0
            && body_low[i - 3] > body_high[i - 4]
            && body_low[i - 2] < close4
            && body_low[i - 1] < close4
            && body_low[i - 2] > close4 - body[i - 4] * penetration
            && body_low[i - 1] > close4 - body[i - 4] * penetration
            && body_high[i - 2] < open3
            && body_high[i - 1] < body_high[i - 2]
            && open0 > close1
            && close0 > fmaxf(high3, fmaxf(high2, high1));
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdlrisefall3methods_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const float* __restrict__ upper_shadow,
    const float* __restrict__ lower_shadow,
    const int8_t* __restrict__ direction,
    int len,
    int period_body_short,
    int period_body_long,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int max_period = (period_body_short > period_body_long) ? period_body_short : period_body_long;
    const int lookback = max_period + 4;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum4 = 0.0f;
        float sum3 = 0.0f;
        float sum2 = 0.0f;
        float sum1 = 0.0f;
        float sum0 = 0.0f;
        for (int j = i - 4 - period_body_long; j < i - 4; ++j) {
            sum4 += body[j];
        }
        for (int j = i - 3 - period_body_short; j < i - 3; ++j) {
            sum3 += body[j];
        }
        for (int j = i - 2 - period_body_short; j < i - 2; ++j) {
            sum2 += body[j];
        }
        for (int j = i - 1 - period_body_short; j < i - 1; ++j) {
            sum1 += body[j];
        }
        for (int j = i - period_body_long; j < i; ++j) {
            sum0 += body[j];
        }
        const float avg4 = sum4 / (float)period_body_long;
        const float avg3 = sum3 / (float)period_body_short;
        const float avg2 = sum2 / (float)period_body_short;
        const float avg1 = sum1 / (float)period_body_short;
        const float avg0 = sum0 / (float)period_body_long;

        const int8_t c4 = direction[i - 4];
        const float s = (float)c4;
        const float close4 = pr_close(body_low[i - 4], body_high[i - 4], direction[i - 4]);
        const float close3 = pr_close(body_low[i - 3], body_high[i - 3], direction[i - 3]);
        const float close2 = pr_close(body_low[i - 2], body_high[i - 2], direction[i - 2]);
        const float close1 = pr_close(body_low[i - 1], body_high[i - 1], direction[i - 1]);
        const float open0 = pr_open(body_low[i], body_high[i], direction[i]);
        const float close0 = pr_close(body_low[i], body_high[i], direction[i]);
        const float high4 = pr_high(body_high[i - 4], upper_shadow[i - 4]);
        const float low4 = pr_low(body_low[i - 4], lower_shadow[i - 4]);

        const bool hit = body[i - 4] > avg4
            && body[i - 3] < avg3
            && body[i - 2] < avg2
            && body[i - 1] < avg1
            && body[i] > avg0
            && c4 == -direction[i - 3]
            && direction[i - 3] == direction[i - 2]
            && direction[i - 2] == direction[i - 1]
            && direction[i - 1] == -direction[i]
            && body_low[i - 3] < high4
            && body_high[i - 3] > low4
            && body_low[i - 2] < high4
            && body_high[i - 2] > low4
            && body_low[i - 1] < high4
            && body_high[i - 1] > low4
            && close2 * s < close3 * s
            && close1 * s < close2 * s
            && open0 * s > close1 * s
            && close0 * s > close4 * s;
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_row_cdltristar_u8_kernel(
    const float* __restrict__ body,
    const float* __restrict__ body_low,
    const float* __restrict__ body_high,
    const uint8_t* __restrict__ body_gap_up,
    const uint8_t* __restrict__ body_gap_down,
    int len,
    int period_body_doji,
    uint8_t* __restrict__ matrix,
    int cols,
    int row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int base = row * cols;
    const int lookback = period_body_doji + 2;

    for (int i = tid; i < len; i += stride) {
        if (i < lookback) {
            matrix[base + i] = 0;
            continue;
        }

        float sum_body = 0.0f;
        for (int j = i - 2 - period_body_doji; j < i - 2; ++j) {
            sum_body += body[j];
        }
        const float avg_body = sum_body / (float)period_body_doji;

        bool hit = false;
        if (body[i - 2] <= avg_body && body[i - 1] <= avg_body && body[i] <= avg_body) {
            const bool bearish = body_gap_up[i - 1] != 0u && (body_high[i] < body_high[i - 1]);
            const bool bullish = body_gap_down[i - 1] != 0u && (body_low[i] > body_low[i - 1]);
            hit = bearish || bullish;
        }
        matrix[base + i] = hit ? 1u : 0u;
    }
}

extern "C" __global__ void pattern_pack_u8_to_u64_kernel(
    const uint8_t* __restrict__ matrix,
    int rows,
    int cols,
    int words_per_row,
    unsigned long long* __restrict__ out_words)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_words = rows * words_per_row;

    for (int idx = tid; idx < total_words; idx += stride) {
        const int row = idx / words_per_row;
        const int word = idx - row * words_per_row;
        const int col0 = word * 64;

        unsigned long long bits = 0ull;
        #pragma unroll
        for (int bit = 0; bit < 64; ++bit) {
            const int col = col0 + bit;
            if (col < cols) {
                const uint8_t v = matrix[row * cols + col];
                bits |= (static_cast<unsigned long long>(v != 0u) << bit);
            }
        }
        out_words[idx] = bits;
    }
}

extern "C" __global__ void pattern_u8_to_f32_kernel(
    const uint8_t* __restrict__ matrix_u8,
    float* __restrict__ matrix_f32,
    int total)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < total; i += stride) {
        matrix_f32[i] = matrix_u8[i] == 0u ? 0.0f : 1.0f;
    }
}
