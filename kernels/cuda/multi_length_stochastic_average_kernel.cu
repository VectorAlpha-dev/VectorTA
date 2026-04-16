#include <cmath>
#include <cstddef>

namespace {
constexpr int METHOD_NONE = 0;
constexpr int METHOD_SMA = 1;
constexpr int METHOD_TMA = 2;
constexpr int METHOD_LSMA = 3;
constexpr int MIN_STOCH_LENGTH = 4;
constexpr double FLOAT_TOL = 1e-12;

struct SmaState {
    double* ring;
    int length;
    int head;
    int count;
    double sum;

    __device__ void init(double* storage, int len) {
        ring = storage;
        length = len;
        head = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ bool update(double value, double* out) {
        if (count == length) {
            sum -= ring[head];
        } else {
            count += 1;
        }
        ring[head] = value;
        sum += value;
        head += 1;
        if (head == length) {
            head = 0;
        }
        if (count == length) {
            *out = sum / static_cast<double>(length);
            return true;
        }
        *out = NAN;
        return false;
    }
};

struct LsmaState {
    double* ring;
    int length;
    int head;
    int count;
    double sum_y;
    double sum_xy;
    double x_sum;
    double denom;

    __device__ void init(double* storage, int len) {
        ring = storage;
        length = len;
        head = 0;
        count = 0;
        sum_y = 0.0;
        sum_xy = 0.0;
        const double n_f = static_cast<double>(len);
        x_sum = static_cast<double>((len * (len - 1)) / 2);
        const double x2_sum = static_cast<double>((len * (len - 1) * (2 * len - 1)) / 6);
        denom = n_f * x2_sum - x_sum * x_sum;
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum_y = 0.0;
        sum_xy = 0.0;
    }

    __device__ double endpoint() const {
        const double n = static_cast<double>(length);
        const double slope = (n * sum_xy - x_sum * sum_y) / denom;
        const double intercept = (sum_y - slope * x_sum) / n;
        return intercept + slope * static_cast<double>(length - 1);
    }

    __device__ bool update(double value, double* out) {
        if (count < length) {
            const int idx = count;
            ring[head] = value;
            head += 1;
            if (head == length) {
                head = 0;
            }
            count += 1;
            sum_y += value;
            sum_xy += static_cast<double>(idx) * value;
            if (count == length) {
                *out = endpoint();
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old = ring[head];
        const double old_sum_y = sum_y;
        ring[head] = value;
        head += 1;
        if (head == length) {
            head = 0;
        }
        sum_y = old_sum_y - old + value;
        sum_xy = sum_xy - (old_sum_y - old) + static_cast<double>(length - 1) * value;
        *out = endpoint();
        return true;
    }
};

struct SmoothingState {
    int method;
    SmaState sma_1;
    SmaState sma_2;
    LsmaState lsma;

    __device__ void init(int method_code, int length, double* storage) {
        method = method_code;
        if (method == METHOD_SMA) {
            sma_1.init(storage, length);
        } else if (method == METHOD_TMA) {
            sma_1.init(storage, length);
            sma_2.init(storage + length, length);
        } else if (method == METHOD_LSMA) {
            lsma.init(storage, length);
        }
    }

    __device__ void reset() {
        if (method == METHOD_SMA) {
            sma_1.reset();
        } else if (method == METHOD_TMA) {
            sma_1.reset();
            sma_2.reset();
        } else if (method == METHOD_LSMA) {
            lsma.reset();
        }
    }

    __device__ bool update(double value, double* out) {
        if (method == METHOD_NONE) {
            *out = value;
            return true;
        }
        if (method == METHOD_SMA) {
            return sma_1.update(value, out);
        }
        if (method == METHOD_TMA) {
            double inner = NAN;
            if (!sma_1.update(value, &inner)) {
                *out = NAN;
                return false;
            }
            return sma_2.update(inner, out);
        }
        return lsma.update(value, out);
    }
};
}

extern "C" __global__ void multi_length_stochastic_average_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ presmooths,
    const int* __restrict__ postsmooths,
    const int* __restrict__ premethods,
    const int* __restrict__ postmethods,
    int rows,
    int scratch_cap,
    double* __restrict__ scratch_buf,
    double* __restrict__ out_values
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const int presmooth = presmooths[row];
    const int postsmooth = postsmooths[row];
    const int premethod = premethods[row];
    const int postmethod = postmethods[row];

    double* row_out = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (length < MIN_STOCH_LENGTH || presmooth <= 0 || postsmooth <= 0) {
        return;
    }
    if (premethod < METHOD_NONE || premethod > METHOD_LSMA || postmethod < METHOD_NONE ||
        postmethod > METHOD_LSMA) {
        return;
    }

    double* row_scratch = scratch_buf + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap);
    int offset = 0;
    double* pre_storage = row_scratch + offset;
    if (premethod == METHOD_SMA || premethod == METHOD_LSMA) {
        offset += presmooth;
    } else if (premethod == METHOD_TMA) {
        offset += presmooth * 2;
    }
    double* post_storage = row_scratch + offset;
    if (postmethod == METHOD_SMA || postmethod == METHOD_LSMA) {
        offset += postsmooth;
    } else if (postmethod == METHOD_TMA) {
        offset += postsmooth * 2;
    }
    double* main_ring = row_scratch + offset;
    offset += length;
    if (offset > scratch_cap) {
        return;
    }

    SmoothingState pre_state;
    SmoothingState post_state;
    pre_state.init(premethod, presmooth, pre_storage);
    post_state.init(postmethod, postsmooth, post_storage);

    int main_head = 0;
    int main_count = 0;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            pre_state.reset();
            post_state.reset();
            main_head = 0;
            main_count = 0;
            continue;
        }

        double pre_value = NAN;
        if (!pre_state.update(value, &pre_value)) {
            continue;
        }

        main_ring[main_head] = pre_value;
        main_head += 1;
        if (main_head == length) {
            main_head = 0;
        }
        if (main_count < length) {
            main_count += 1;
        }
        if (main_count < length) {
            continue;
        }

        const int newest = (main_head + length - 1) % length;
        const double current = main_ring[newest];
        double min_value = current;
        double max_value = current;
        int idx = newest;
        double sum = 0.0;
        bool invalid = false;

        for (int window = 1; window <= length; ++window) {
            const double ring_value = main_ring[idx];
            if (ring_value < min_value) {
                min_value = ring_value;
            }
            if (ring_value > max_value) {
                max_value = ring_value;
            }
            if (window >= MIN_STOCH_LENGTH) {
                const double denom = max_value - min_value;
                if (fabs(denom) <= FLOAT_TOL) {
                    post_state.reset();
                    invalid = true;
                    break;
                }
                sum += (current - min_value) / denom;
            }
            idx = (idx == 0) ? (length - 1) : (idx - 1);
        }

        if (invalid) {
            continue;
        }

        const double norm =
            (sum / static_cast<double>(length - (MIN_STOCH_LENGTH - 1))) * 100.0;
        double post_value = NAN;
        if (!post_state.update(norm, &post_value)) {
            continue;
        }
        row_out[i] = post_value;
    }
}
