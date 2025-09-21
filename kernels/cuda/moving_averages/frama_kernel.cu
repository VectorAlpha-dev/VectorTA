// CUDA kernels for the Fractal Adaptive Moving Average (FRAMA).
//
// Each parameter combination is evaluated by a single thread that maintains
// monotonic deques for the rolling high/low extrema, mirroring the scalar
// implementation used on the CPU. A companion kernel handles the many-series ×
// one-parameter scenario using time-major inputs so that both CUDA paths offer
// feature parity with the Rust ALMA integration pattern.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef FRAMA_NAN
#define FRAMA_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef FRAMA_MAX_WINDOW
#define FRAMA_MAX_WINDOW 1024
#endif

struct MonoDeque {
    int* buf;
    int head;
    int tail;
    const float* data;
    int stride;
    int offset;
};

__device__ __forceinline__ int md_inc(int idx) {
    ++idx;
    if (idx >= FRAMA_MAX_WINDOW) {
        idx = 0;
    }
    return idx;
}

__device__ __forceinline__ int md_dec(int idx) {
    if (idx == 0) {
        idx = FRAMA_MAX_WINDOW;
    }
    return idx - 1;
}

__device__ __forceinline__ MonoDeque make_deque(
    int* storage,
    const float* data,
    int stride,
    int offset) {
    MonoDeque dq;
    dq.buf = storage;
    dq.head = 0;
    dq.tail = 0;
    dq.data = data;
    dq.stride = stride;
    dq.offset = offset;
    return dq;
}

__device__ __forceinline__ void md_clear(MonoDeque* dq) {
    dq->head = 0;
    dq->tail = 0;
}

__device__ __forceinline__ bool md_empty(const MonoDeque* dq) {
    return dq->head == dq->tail;
}

__device__ __forceinline__ int md_front(const MonoDeque* dq) {
    return dq->buf[dq->head];
}

__device__ __forceinline__ float md_value(const MonoDeque* dq, int idx) {
    return dq->data[idx * dq->stride + dq->offset];
}

__device__ __forceinline__ void md_expire(MonoDeque* dq, int idx_out) {
    if (!md_empty(dq) && dq->buf[dq->head] == idx_out) {
        dq->head = md_inc(dq->head);
    }
}

__device__ __forceinline__ void md_push_max(MonoDeque* dq, int idx) {
    const float cur = md_value(dq, idx);
    while (!md_empty(dq)) {
        int last_slot = md_dec(dq->tail);
        int last_idx = dq->buf[last_slot];
        if (md_value(dq, last_idx) >= cur) {
            break;
        }
        dq->tail = last_slot;
        if (dq->tail == dq->head) {
            break;
        }
    }
    dq->buf[dq->tail] = idx;
    dq->tail = md_inc(dq->tail);
}

__device__ __forceinline__ void md_push_min(MonoDeque* dq, int idx) {
    const float cur = md_value(dq, idx);
    while (!md_empty(dq)) {
        int last_slot = md_dec(dq->tail);
        int last_idx = dq->buf[last_slot];
        if (md_value(dq, last_idx) <= cur) {
            break;
        }
        dq->tail = last_slot;
        if (dq->tail == dq->head) {
            break;
        }
    }
    dq->buf[dq->tail] = idx;
    dq->tail = md_inc(dq->tail);
}

struct ExtremesPair {
    float maxv;
    float minv;
};

__device__ __forceinline__ ExtremesPair frama_front_or(
    const MonoDeque* dq_max,
    const MonoDeque* dq_min,
    float* prev_max,
    float* prev_min) {
    float maxv = *prev_max;
    float minv = *prev_min;
    if (!md_empty(dq_max)) {
        maxv = md_value(dq_max, md_front(dq_max));
    }
    if (!md_empty(dq_min)) {
        minv = md_value(dq_min, md_front(dq_min));
    }
    *prev_max = maxv;
    *prev_min = minv;
    ExtremesPair out = {maxv, minv};
    return out;
}

__device__ __forceinline__ void md_swap(MonoDeque* a, MonoDeque* b) {
    MonoDeque tmp = *a;
    *a = *b;
    *b = tmp;
}

extern "C" __global__ void frama_batch_f32(const float* __restrict__ high,
                                            const float* __restrict__ low,
                                            const float* __restrict__ close,
                                            const int* __restrict__ windows,
                                            const int* __restrict__ scs,
                                            const int* __restrict__ fcs,
                                            int series_len,
                                            int n_combos,
                                            int first_valid,
                                            float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    int storage_full_max[FRAMA_MAX_WINDOW];
    int storage_full_min[FRAMA_MAX_WINDOW];
    int storage_left_max[FRAMA_MAX_WINDOW];
    int storage_left_min[FRAMA_MAX_WINDOW];
    int storage_right_max[FRAMA_MAX_WINDOW];
    int storage_right_min[FRAMA_MAX_WINDOW];

    MonoDeque d_full_max = make_deque(storage_full_max, high, 1, 0);
    MonoDeque d_full_min = make_deque(storage_full_min, low, 1, 0);
    MonoDeque d_left_max = make_deque(storage_left_max, high, 1, 0);
    MonoDeque d_left_min = make_deque(storage_left_min, low, 1, 0);
    MonoDeque d_right_max = make_deque(storage_right_max, high, 1, 0);
    MonoDeque d_right_min = make_deque(storage_right_min, low, 1, 0);

    float* row_out = out + combo * series_len;
    for (int i = 0; i < series_len; ++i) {
        row_out[i] = FRAMA_NAN;
    }

    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    int window = windows[combo];
    int sc = scs[combo];
    int fc = fcs[combo];
    if (window <= 0 || sc <= 0 || fc <= 0) {
        return;
    }

    int win = window;
    if (win & 1) {
        ++win;
    }
    if (win <= 1 || win > FRAMA_MAX_WINDOW) {
        return;
    }
    const int half = win / 2;
    if (half <= 0) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < win) {
        return;
    }

    float seed = 0.0f;
    for (int j = 0; j < win; ++j) {
        seed += close[first_valid + j];
    }
    seed /= static_cast<float>(win);
    const int warm = first_valid + win - 1;
    row_out[warm] = seed;

    md_clear(&d_full_max);
    md_clear(&d_full_min);
    md_clear(&d_left_max);
    md_clear(&d_left_min);
    md_clear(&d_right_max);
    md_clear(&d_right_min);

    const int win_end = first_valid + win;
    for (int idx = first_valid; idx < win_end; ++idx) {
        const float hi = high[idx];
        const float lo = low[idx];
        if (isnan(hi) || isnan(lo)) {
            continue;
        }
        md_push_max(&d_full_max, idx);
        md_push_min(&d_full_min, idx);
        if (idx < first_valid + half) {
            md_push_max(&d_left_max, idx);
            md_push_min(&d_left_min, idx);
        } else {
            md_push_max(&d_right_max, idx);
            md_push_min(&d_right_min, idx);
        }
    }

    const float sc_f = static_cast<float>(sc);
    const float fc_f = static_cast<float>(fc);
    const float w_ln = logf(2.0f / (sc_f + 1.0f));
    const float sc_lim = 2.0f / (sc_f + 1.0f);
    const float ln2 = logf(2.0f);

    float d_prev = 1.0f;

    float pm1 = FRAMA_NAN;
    float pm2 = FRAMA_NAN;
    float pm3 = FRAMA_NAN;
    float pn1 = FRAMA_NAN;
    float pn2 = FRAMA_NAN;
    float pn3 = FRAMA_NAN;

    int half_progress = 0;

    for (int i = warm + 1; i < series_len; ++i) {
        const int idx_out = i - win;
        md_expire(&d_full_max, idx_out);
        md_expire(&d_full_min, idx_out);
        md_expire(&d_left_max, idx_out);
        md_expire(&d_left_min, idx_out);
        md_expire(&d_right_max, idx_out + half);
        md_expire(&d_right_min, idx_out + half);

        const int newest = i - 1;
        const float hi = high[newest];
        const float lo = low[newest];
        if (!(isnan(hi) || isnan(lo))) {
            md_push_max(&d_full_max, newest);
            md_push_min(&d_full_min, newest);
            if (newest < idx_out + half) {
                md_push_max(&d_left_max, newest);
                md_push_min(&d_left_min, newest);
            } else {
                md_push_max(&d_right_max, newest);
                md_push_min(&d_right_min, newest);
            }
        }

        ExtremesPair right = frama_front_or(&d_right_max, &d_right_min, &pm1, &pn1);
        ExtremesPair left = frama_front_or(&d_left_max, &d_left_min, &pm2, &pn2);
        ExtremesPair full = frama_front_or(&d_full_max, &d_full_min, &pm3, &pn3);

        const float hi_i = high[i];
        const float lo_i = low[i];
        const float close_i = close[i];
        const float prev_out = row_out[i - 1];

        if (!(isnan(hi_i) || isnan(lo_i) || isnan(close_i) || isnan(prev_out))) {
            const float n1 = (right.maxv - right.minv) / static_cast<float>(half);
            const float n2 = (left.maxv - left.minv) / static_cast<float>(half);
            const float n3 = (full.maxv - full.minv) / static_cast<float>(win);

            float d_cur = d_prev;
            if (n1 > 0.0f && n2 > 0.0f && n3 > 0.0f) {
                d_cur = (logf(n1 + n2) - logf(n3)) / ln2;
            }
            d_prev = d_cur;

            float alpha0 = expf(w_ln * (d_cur - 1.0f));
            alpha0 = fmaxf(alpha0, 0.1f);
            alpha0 = fminf(alpha0, 1.0f);

            const float old_n = (2.0f - alpha0) / alpha0;
            float new_n = (sc_f - fc_f) * ((old_n - 1.0f) / (sc_f - 1.0f)) + fc_f;
            if (sc_f == 1.0f) {
                new_n = fc_f;
            }
            float alpha = 2.0f / (new_n + 1.0f);
            alpha = fmaxf(alpha, sc_lim);
            alpha = fminf(alpha, 1.0f);

            row_out[i] = alpha * close_i + (1.0f - alpha) * prev_out;
        } else {
            row_out[i] = prev_out;
        }

        ++half_progress;
        if (half_progress == half) {
            md_swap(&d_left_max, &d_right_max);
            md_swap(&d_left_min, &d_right_min);
            md_clear(&d_right_max);
            md_clear(&d_right_min);
            half_progress = 0;
        }
    }
}

extern "C" __global__ void frama_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int window,
    int sc,
    int fc,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    int storage_full_max[FRAMA_MAX_WINDOW];
    int storage_full_min[FRAMA_MAX_WINDOW];
    int storage_left_max[FRAMA_MAX_WINDOW];
    int storage_left_min[FRAMA_MAX_WINDOW];
    int storage_right_max[FRAMA_MAX_WINDOW];
    int storage_right_min[FRAMA_MAX_WINDOW];

    MonoDeque d_full_max = make_deque(storage_full_max, high_tm, num_series, series);
    MonoDeque d_full_min = make_deque(storage_full_min, low_tm, num_series, series);
    MonoDeque d_left_max = make_deque(storage_left_max, high_tm, num_series, series);
    MonoDeque d_left_min = make_deque(storage_left_min, low_tm, num_series, series);
    MonoDeque d_right_max = make_deque(storage_right_max, high_tm, num_series, series);
    MonoDeque d_right_min = make_deque(storage_right_min, low_tm, num_series, series);

    float* col_out = out_tm + series;
    for (int row = 0; row < series_len; ++row) {
        col_out[row * num_series] = FRAMA_NAN;
    }

    if (window <= 0 || sc <= 0 || fc <= 0) {
        return;
    }

    int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    int win = window;
    if (win & 1) {
        ++win;
    }
    if (win <= 1 || win > FRAMA_MAX_WINDOW) {
        return;
    }
    const int half = win / 2;
    if (half <= 0) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < win) {
        return;
    }

    float seed = 0.0f;
    for (int j = 0; j < win; ++j) {
        seed += close_tm[(first_valid + j) * num_series + series];
    }
    seed /= static_cast<float>(win);
    const int warm = first_valid + win - 1;
    col_out[warm * num_series] = seed;

    md_clear(&d_full_max);
    md_clear(&d_full_min);
    md_clear(&d_left_max);
    md_clear(&d_left_min);
    md_clear(&d_right_max);
    md_clear(&d_right_min);

    const int win_end = first_valid + win;
    for (int idx = first_valid; idx < win_end; ++idx) {
        const float hi = high_tm[idx * num_series + series];
        const float lo = low_tm[idx * num_series + series];
        if (isnan(hi) || isnan(lo)) {
            continue;
        }
        md_push_max(&d_full_max, idx);
        md_push_min(&d_full_min, idx);
        if (idx < first_valid + half) {
            md_push_max(&d_left_max, idx);
            md_push_min(&d_left_min, idx);
        } else {
            md_push_max(&d_right_max, idx);
            md_push_min(&d_right_min, idx);
        }
    }

    const float sc_f = static_cast<float>(sc);
    const float fc_f = static_cast<float>(fc);
    const float w_ln = logf(2.0f / (sc_f + 1.0f));
    const float sc_lim = 2.0f / (sc_f + 1.0f);
    const float ln2 = logf(2.0f);

    float d_prev = 1.0f;

    float pm1 = FRAMA_NAN;
    float pm2 = FRAMA_NAN;
    float pm3 = FRAMA_NAN;
    float pn1 = FRAMA_NAN;
    float pn2 = FRAMA_NAN;
    float pn3 = FRAMA_NAN;

    int half_progress = 0;

    for (int i = warm + 1; i < series_len; ++i) {
        const int idx_out = i - win;
        md_expire(&d_full_max, idx_out);
        md_expire(&d_full_min, idx_out);
        md_expire(&d_left_max, idx_out);
        md_expire(&d_left_min, idx_out);
        md_expire(&d_right_max, idx_out + half);
        md_expire(&d_right_min, idx_out + half);

        const int newest = i - 1;
        const float hi = high_tm[newest * num_series + series];
        const float lo = low_tm[newest * num_series + series];
        if (!(isnan(hi) || isnan(lo))) {
            md_push_max(&d_full_max, newest);
            md_push_min(&d_full_min, newest);
            if (newest < idx_out + half) {
                md_push_max(&d_left_max, newest);
                md_push_min(&d_left_min, newest);
            } else {
                md_push_max(&d_right_max, newest);
                md_push_min(&d_right_min, newest);
            }
        }

        ExtremesPair right = frama_front_or(&d_right_max, &d_right_min, &pm1, &pn1);
        ExtremesPair left = frama_front_or(&d_left_max, &d_left_min, &pm2, &pn2);
        ExtremesPair full = frama_front_or(&d_full_max, &d_full_min, &pm3, &pn3);

        const float hi_i = high_tm[i * num_series + series];
        const float lo_i = low_tm[i * num_series + series];
        const float close_i = close_tm[i * num_series + series];
        const float prev_out = col_out[(i - 1) * num_series];

        if (!(isnan(hi_i) || isnan(lo_i) || isnan(close_i) || isnan(prev_out))) {
            const float n1 = (right.maxv - right.minv) / static_cast<float>(half);
            const float n2 = (left.maxv - left.minv) / static_cast<float>(half);
            const float n3 = (full.maxv - full.minv) / static_cast<float>(win);

            float d_cur = d_prev;
            if (n1 > 0.0f && n2 > 0.0f && n3 > 0.0f) {
                d_cur = (logf(n1 + n2) - logf(n3)) / ln2;
            }
            d_prev = d_cur;

            float alpha0 = expf(w_ln * (d_cur - 1.0f));
            alpha0 = fmaxf(alpha0, 0.1f);
            alpha0 = fminf(alpha0, 1.0f);

            const float old_n = (2.0f - alpha0) / alpha0;
            float new_n = (sc_f - fc_f) * ((old_n - 1.0f) / (sc_f - 1.0f)) + fc_f;
            if (sc_f == 1.0f) {
                new_n = fc_f;
            }
            float alpha = 2.0f / (new_n + 1.0f);
            alpha = fmaxf(alpha, sc_lim);
            alpha = fminf(alpha, 1.0f);

            col_out[i * num_series] = alpha * close_i + (1.0f - alpha) * prev_out;
        } else {
            col_out[i * num_series] = prev_out;
        }

        ++half_progress;
        if (half_progress == half) {
            md_swap(&d_left_max, &d_right_max);
            md_swap(&d_left_min, &d_right_min);
            md_clear(&d_right_max);
            md_clear(&d_right_min);
            half_progress = 0;
        }
    }
}
