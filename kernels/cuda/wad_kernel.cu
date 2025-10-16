// CUDA kernels for Williams Accumulation/Distribution (WAD)
//
// Semantics mirror src/indicators/wad.rs (scalar path):
// - No parameters; cumulative running sum with warmup value 0.0 at index 0
// - NaNs propagate naturally through arithmetic (no special casing)
// - Use double for accumulations to better match CPU f64 reference

#include <cuda_runtime.h>
#include <math.h>

// One-series × many-params (batch): for WAD there are no parameters, but we
// keep a batch form for API parity. Each block handles one row (combo) and a
// single thread performs the sequential scan.
extern "C" __global__ void wad_batch_f32(
    const float* __restrict__ high,   // [series_len]
    const float* __restrict__ low,    // [series_len]
    const float* __restrict__ close,  // [series_len]
    int series_len,
    int n_combos,
    float* __restrict__ out) {        // [n_combos * series_len]
  const int combo = blockIdx.x;
  if (combo >= n_combos || series_len <= 0) return;

  // Row base for this combo
  const int base = combo * series_len;
  float* __restrict__ out_row = out + base;

  // Warmup
  out_row[0] = 0.0f;
  double acc = 0.0;
  double pc = static_cast<double>(close[0]);

  for (int i = 1; i < series_len; ++i) {
    const double hi = static_cast<double>(high[i]);
    const double lo = static_cast<double>(low[i]);
    const double c  = static_cast<double>(close[i]);

    const double trh = (pc > hi) ? pc : hi;
    const double trl = (pc < lo) ? pc : lo;

    // Branch-friendly accumulation identical to scalar semantics
    double ad = 0.0;
    if (c > pc)       ad = c - trl;
    else if (c < pc)  ad = c - trh;

    acc += ad;
    out_row[i] = static_cast<float>(acc);
    pc = c;
  }
}

// Many-series × one-param (time-major). WAD has no parameter; we keep the
// conventional name for parity. Input layout is time-major: [t][series].
extern "C" __global__ void wad_many_series_one_param_f32(
    const float* __restrict__ high_tm,   // [rows * cols], rows=time
    const float* __restrict__ low_tm,    // [rows * cols]
    const float* __restrict__ close_tm,  // [rows * cols]
    int cols,   // number of series
    int rows,   // series length
    float* __restrict__ out_tm) {        // [rows * cols]
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= cols || rows <= 0) return;

  // stride between consecutive timesteps for a given series
  const int stride = cols;
  // Warmup
  out_tm[0 * stride + s] = 0.0f;
  double acc = 0.0;
  double pc = static_cast<double>(close_tm[0 * stride + s]);

  for (int t = 1; t < rows; ++t) {
    const double hi = static_cast<double>(high_tm[t * stride + s]);
    const double lo = static_cast<double>(low_tm[t * stride + s]);
    const double c  = static_cast<double>(close_tm[t * stride + s]);

    const double trh = (pc > hi) ? pc : hi;
    const double trl = (pc < lo) ? pc : lo;

    double ad = 0.0;
    if (c > pc)       ad = c - trl;
    else if (c < pc)  ad = c - trh;

    acc += ad;
    out_tm[t * stride + s] = static_cast<float>(acc);
    pc = c;
  }
}

// Back-compat: series-major helper where each thread processes one contiguous
// series arranged row-major. Kept for existing tests; wrapper prefers batch API.
extern "C" __global__ void wad_series_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    int n_series,
    float* __restrict__ out) {
  const int series = blockIdx.x * blockDim.x + threadIdx.x;
  if (series >= n_series || len <= 0) return;

  const int offset = series * len;
  const float* high_row = high + offset;
  const float* low_row  = low + offset;
  const float* close_row= close + offset;
  float* out_row        = out + offset;

  out_row[0] = 0.0f;
  double acc = 0.0;
  double pc = static_cast<double>(close_row[0]);
  for (int i = 1; i < len; ++i) {
    const double hi = static_cast<double>(high_row[i]);
    const double lo = static_cast<double>(low_row[i]);
    const double c  = static_cast<double>(close_row[i]);
    const double trh = (pc > hi) ? pc : hi;
    const double trl = (pc < lo) ? pc : lo;
    double ad = 0.0;
    if (c > pc)       ad = c - trl;
    else if (c < pc)  ad = c - trh;
    acc += ad;
    out_row[i] = static_cast<float>(acc);
    pc = c;
  }
}
