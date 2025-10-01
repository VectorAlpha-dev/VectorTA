//! # Williams' %R (WILLR)
//!
//! Williams' %R is a momentum oscillator that shows the relationship between the current close
//! and the high-low range over a period, indicating overbought/oversold conditions (-20 to -80).
//!
//! ## Parameters
//! - **period**: Lookback period for range calculation. Defaults to 14.
//!
//! ## Returns
//! - **`Ok(WillrOutput)`** containing a `Vec<f64>` of %R values (-100 to 0) matching input length.
//! - **`Err(WillrError)`** on invalid parameters or insufficient data.
//!
//! ## Developer Notes
//! - **Scalar kernel**: Hybrid selection
//!   - Small periods (<= 64): naive O(period) scan (fastest for common WILLR periods like 14)
//!   - Large periods  (> 64): monotonic-deque O(1) sliding window for high/low
//! - **SIMD status**: AVX2/AVX512 stubs currently delegate to scalar. Branchy min/max windows did not
//!   show consistent wins in preliminary attempts; revisit with alternative layouts if needed.
//! - **Streaming Performance**: O(1) update API (ring buffers) available via `WillrStream`
//! - **Memory Optimization**: ✓ Uses `alloc_with_nan_prefix` and `make_uninit_matrix` for batching
//! - **Batch Support**: ✓ Row-specific optimized batch via sparse tables (shared across rows)
//! - **TODO**: Only enable SIMD selection once kernels beat scalar by >5% at 10k/100k.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::CudaWillr;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// --- Data types ---

#[derive(Debug, Clone)]
pub enum WillrData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct WillrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct WillrParams {
    pub period: Option<usize>,
}

impl Default for WillrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct WillrInput<'a> {
    pub data: WillrData<'a>,
    pub params: WillrParams,
}

impl<'a> WillrInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: WillrParams) -> Self {
        Self {
            data: WillrData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: WillrParams,
    ) -> Self {
        Self {
            data: WillrData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, WillrParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WillrBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for WillrBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WillrBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn apply(self, c: &Candles) -> Result<WillrOutput, WillrError> {
        let p = WillrParams {
            period: self.period,
        };
        let i = WillrInput::from_candles(c, p);
        willr_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WillrOutput, WillrError> {
        let p = WillrParams {
            period: self.period,
        };
        let i = WillrInput::from_slices(high, low, close, p);
        willr_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum WillrError {
    #[error("willr: All input values are NaN.")]
    AllValuesNaN,
    #[error("willr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("willr: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("willr: Data slices are empty or mismatched.")]
    EmptyOrMismatched,
    #[error("willr: Output length mismatch: dst = {dst}, data_len = {data_len}")]
    OutputLenMismatch { dst: usize, data_len: usize },
}

// --- Main entrypoints ---

#[inline(always)]
fn willr_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            // No dedicated SIMD128 kernel yet. Keep hook for parity.
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                willr_scalar(high, low, close, period, first_valid, out);
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                willr_scalar(high, low, close, period, first_valid, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                willr_avx2(high, low, close, period, first_valid, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                willr_avx512(high, low, close, period, first_valid, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                willr_scalar(high, low, close, period, first_valid, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn willr(input: &WillrInput) -> Result<WillrOutput, WillrError> {
    willr_with_kernel(input, Kernel::Auto)
}

pub fn willr_with_kernel(input: &WillrInput, kernel: Kernel) -> Result<WillrOutput, WillrError> {
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        WillrData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        WillrData::Slices { high, low, close } => (high, low, close),
    };

    let len = high.len();
    if low.len() != len || close.len() != len || len == 0 {
        return Err(WillrError::EmptyOrMismatched);
    }
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(WillrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(WillrError::AllValuesNaN)?;

    if (len - first_valid) < period {
        return Err(WillrError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, first_valid + period - 1);
    willr_compute_into(high, low, close, period, first_valid, chosen, &mut out);
    Ok(WillrOutput { values: out })
}

#[inline]
pub fn willr_into_slice(
    dst: &mut [f64],
    input: &WillrInput,
    kernel: Kernel,
) -> Result<(), WillrError> {
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        WillrData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        WillrData::Slices { high, low, close } => (high, low, close),
    };

    let len = high.len();
    if low.len() != len || close.len() != len || len == 0 {
        return Err(WillrError::EmptyOrMismatched);
    }

    if dst.len() != len {
        return Err(WillrError::OutputLenMismatch {
            dst: dst.len(),
            data_len: len,
        });
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(WillrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(WillrError::AllValuesNaN)?;

    if (len - first_valid) < period {
        return Err(WillrError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    willr_compute_into(high, low, close, period, first_valid, chosen, dst);

    // Fill warmup period with NaN
    let warmup_end = first_valid + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

// --- Scalar/AVX kernels ---

#[inline]
pub fn willr_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    // Heuristic: the naive O(period) scan is typically faster for small windows
    // (common WILLR periods ~14). Switch to deque only for larger windows.
    const DEQUE_SWITCH: usize = 32;
    if period <= DEQUE_SWITCH {
        return willr_scalar_naive(high, low, close, period, first_valid, out);
    }

    willr_scalar_deque(high, low, close, period, first_valid, out);
}

#[inline(always)]
fn willr_scalar_naive(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    for i in (first_valid + period - 1)..high.len() {
        if close[i].is_nan() {
            out[i] = f64::NAN;
            continue;
        }
        let start = i + 1 - period;
        let (mut h, mut l) = (f64::NEG_INFINITY, f64::INFINITY);
        let mut has_nan = false;
        for j in start..=i {
            let hj = high[j];
            let lj = low[j];
            if hj.is_nan() || lj.is_nan() {
                has_nan = true;
                break;
            }
            if hj > h {
                h = hj;
            }
            if lj < l {
                l = lj;
            }
        }
        if has_nan || h.is_infinite() || l.is_infinite() {
            out[i] = f64::NAN;
        } else {
            let denom = h - l;
            out[i] = if denom == 0.0 { 0.0 } else { (h - close[i]) / denom * -100.0 };
        }
    }
}

#[inline(always)]
fn willr_scalar_deque(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let n = high.len();
    if n == 0 {
        return;
    }

    // First index where the value is computable
    let start_i = first_valid + period - 1;
    if start_i >= n {
        return;
    }

    let cap = period; // capacity for ring buffers

    // Monotonic deques implemented as ring buffers (store indices)
    let mut dq_max = vec![0usize; cap];
    let mut head_max = 0usize;
    let mut len_max = 0usize;

    let mut dq_min = vec![0usize; cap];
    let mut head_min = 0usize;
    let mut len_min = 0usize;

    // Rolling NaN tracker for the (high|low) window
    let mut nan_ring = vec![0u8; cap];
    let mut ring_pos = 0usize;
    let mut nan_count: usize = 0;

    // Prime the structures with the first window: [l0 .. start_i]
    let l0 = start_i + 1 - period;
    for j in l0..=start_i {
        let hj = high[j];
        let lj = low[j];
        let is_nan = (hj.is_nan() || lj.is_nan()) as u8;

        nan_count += is_nan as usize;
        nan_ring[ring_pos] = is_nan;
        ring_pos += 1;
        if ring_pos == cap {
            ring_pos = 0;
        }

        if is_nan == 0 {
            // Maintain decreasing deque for highs (pop while back <= hj)
            while len_max != 0 {
                let back_pos = (head_max + len_max - 1) % cap;
                let back_idx = dq_max[back_pos];
                if high[back_idx] <= hj {
                    len_max -= 1;
                } else {
                    break;
                }
            }
            let ins_pos = (head_max + len_max) % cap;
            dq_max[ins_pos] = j;
            len_max += 1;

            // Maintain increasing deque for lows (pop while back >= lj)
            while len_min != 0 {
                let back_pos = (head_min + len_min - 1) % cap;
                let back_idx = dq_min[back_pos];
                if low[back_idx] >= lj {
                    len_min -= 1;
                } else {
                    break;
                }
            }
            let ins_pos = (head_min + len_min) % cap;
            dq_min[ins_pos] = j;
            len_min += 1;
        }
    }

    // Main pass
    for i in start_i..n {
        let c = close[i];

        if c.is_nan() || nan_count != 0 || len_max == 0 || len_min == 0 {
            out[i] = f64::NAN;
        } else {
            // Current window extrema are at deque fronts
            let h_idx = dq_max[head_max];
            let l_idx = dq_min[head_min];
            let h = high[h_idx];
            let l = low[l_idx];

            if !(h.is_finite() && l.is_finite()) {
                out[i] = f64::NAN;
            } else {
                let denom = h - l;
                out[i] = if denom == 0.0 { 0.0 } else { (h - c) / denom * -100.0 };
            }
        }

        // Slide the window by one: incorporate i+1 and drop (i+1 - period)
        let next = i + 1;
        if next < n {
            // Remove outdated indices (<= next - period) from fronts
            let cutoff = next - period;

            while len_max != 0 {
                let f_idx = dq_max[head_max];
                if f_idx <= cutoff {
                    head_max += 1;
                    if head_max == cap {
                        head_max = 0;
                    }
                    len_max -= 1;
                } else {
                    break;
                }
            }
            while len_min != 0 {
                let f_idx = dq_min[head_min];
                if f_idx <= cutoff {
                    head_min += 1;
                    if head_min == cap {
                        head_min = 0;
                    }
                    len_min -= 1;
                } else {
                    break;
                }
            }

            // Add new sample at `next`
            let hj = high[next];
            let lj = low[next];
            let new_nan = (hj.is_nan() || lj.is_nan()) as u8;

            // Update rolling NaN count via ring overwrite (subtract old, add new)
            let old_nan = nan_ring[ring_pos] as usize;
            nan_count = nan_count + (new_nan as usize) - old_nan;
            nan_ring[ring_pos] = new_nan;
            ring_pos += 1;
            if ring_pos == cap {
                ring_pos = 0;
            }

            if new_nan == 0 {
                // Push into max deque (highs)
                while len_max != 0 {
                    let back_pos = (head_max + len_max - 1) % cap;
                    let back_idx = dq_max[back_pos];
                    if high[back_idx] <= hj {
                        len_max -= 1;
                    } else {
                        break;
                    }
                }
                let ins_pos = (head_max + len_max) % cap;
                dq_max[ins_pos] = next;
                len_max += 1;

                // Push into min deque (lows)
                while len_min != 0 {
                    let back_pos = (head_min + len_min - 1) % cap;
                    let back_idx = dq_min[back_pos];
                    if low[back_idx] >= lj {
                        len_min -= 1;
                    } else {
                        break;
                    }
                }
                let ins_pos = (head_min + len_min) % cap;
                dq_min[ins_pos] = next;
                len_min += 1;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        willr_avx512_short(high, low, close, period, first_valid, out);
    } else {
        willr_avx512_long(high, low, close, period, first_valid, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

// --- Batch grid/params/output for batch evaluation ---

#[derive(Clone, Debug)]
pub struct WillrBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for WillrBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WillrBatchBuilder {
    range: WillrBatchRange,
    kernel: Kernel,
}

impl WillrBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WillrBatchOutput, WillrError> {
        willr_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<WillrBatchOutput, WillrError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

pub fn willr_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    k: Kernel,
) -> Result<WillrBatchOutput, WillrError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    willr_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct WillrBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WillrParams>,
    pub rows: usize,
    pub cols: usize,
}

impl WillrBatchOutput {
    pub fn row_for_params(&self, p: &WillrParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &WillrParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &WillrBatchRange) -> Vec<WillrParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    periods
        .into_iter()
        .map(|p| WillrParams { period: Some(p) })
        .collect()
}

#[derive(Debug)]
struct SharedWillrCtx {
    log2: Vec<usize>,
    st_max: Vec<Vec<f64>>,
    st_min: Vec<Vec<f64>>,
    nan_psum: Vec<u32>,
}

impl SharedWillrCtx {
    #[inline]
    fn build_log2(n: usize) -> Vec<usize> {
        let mut lg = vec![0usize; n + 1];
        for i in 2..=n {
            lg[i] = lg[i >> 1] + 1;
        }
        lg
    }

    #[inline]
    fn build_sparse(data: &[f64], is_max: bool, log2: &[usize]) -> Vec<Vec<f64>> {
        let n = data.len();
        if n == 0 {
            return vec![Vec::new()];
        }
        let max_k = log2[n];
        let mut st: Vec<Vec<f64>> = Vec::with_capacity(max_k + 1);
        st.push(data.to_vec());
        let mut k = 1usize;
        while k <= max_k {
            let window = 1usize << k;
            let len = n + 1 - window;
            let prev = &st[k - 1];
            let offset = 1usize << (k - 1);
            let mut row = Vec::with_capacity(len);
            for i in 0..len {
                let a = prev[i];
                let b = prev[i + offset];
                row.push(if is_max { a.max(b) } else { a.min(b) });
            }
            st.push(row);
            k += 1;
        }
        st
    }

    #[inline]
    fn new(high: &[f64], low: &[f64]) -> Self {
        debug_assert_eq!(high.len(), low.len());
        let n = high.len();
        let log2 = Self::build_log2(n);

        let mut high_clean = Vec::with_capacity(n);
        let mut low_clean = Vec::with_capacity(n);
        let mut nan_psum = vec![0u32; n + 1];

        for i in 0..n {
            let h = high[i];
            let l = low[i];
            let has_nan = h.is_nan() || l.is_nan();
            nan_psum[i + 1] = nan_psum[i] + (has_nan as u32);

            high_clean.push(if h.is_nan() { f64::NEG_INFINITY } else { h });
            low_clean.push(if l.is_nan() { f64::INFINITY } else { l });
        }

        let st_max = Self::build_sparse(&high_clean, true, &log2);
        let st_min = Self::build_sparse(&low_clean, false, &log2);

        Self {
            log2,
            st_max,
            st_min,
            nan_psum,
        }
    }

    #[inline]
    fn qmax(&self, l: usize, r: usize) -> f64 {
        let len = r - l + 1;
        let k = self.log2[len];
        let offset = 1usize << k;
        let a = self.st_max[k][l];
        let b = self.st_max[k][r + 1 - offset];
        a.max(b)
    }

    #[inline]
    fn qmin(&self, l: usize, r: usize) -> f64 {
        let len = r - l + 1;
        let k = self.log2[len];
        let offset = 1usize << k;
        let a = self.st_min[k][l];
        let b = self.st_min[k][r + 1 - offset];
        a.min(b)
    }

    #[inline]
    fn window_has_nan(&self, l: usize, r: usize) -> bool {
        (self.nan_psum[r + 1] - self.nan_psum[l]) != 0
    }

    #[cfg(feature = "cuda")]
    #[inline]
    fn flatten(&self) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let levels = self.st_max.len();
        let mut offsets = Vec::with_capacity(levels + 1);
        let mut total = 0usize;
        for level in &self.st_max {
            offsets.push(total);
            total += level.len();
        }
        offsets.push(total);

        let mut flat_max = Vec::with_capacity(total);
        for level in &self.st_max {
            flat_max.extend_from_slice(level);
        }

        let mut flat_min = Vec::with_capacity(total);
        for level in &self.st_min {
            flat_min.extend_from_slice(level);
        }

        (flat_max, flat_min, offsets)
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct WillrGpuTables {
    pub log2: Vec<i32>,
    pub level_offsets: Vec<i32>,
    pub st_max: Vec<f32>,
    pub st_min: Vec<f32>,
    pub nan_psum: Vec<i32>,
}

#[cfg(feature = "cuda")]
impl WillrGpuTables {
    #[inline]
    fn from_shared_ctx(ctx: &SharedWillrCtx) -> Self {
        let (flat_max_f64, flat_min_f64, offsets_usize) = ctx.flatten();
        let log2 = ctx.log2.iter().map(|&v| v as i32).collect();
        let level_offsets = offsets_usize.iter().map(|&v| v as i32).collect();
        let st_max = flat_max_f64.iter().map(|&v| v as f32).collect();
        let st_min = flat_min_f64.iter().map(|&v| v as f32).collect();
        let nan_psum = ctx.nan_psum.iter().map(|&v| v as i32).collect();

        Self {
            log2,
            level_offsets,
            st_max,
            st_min,
            nan_psum,
        }
    }
}

#[cfg(feature = "cuda")]
#[inline]
pub fn build_willr_gpu_tables(high: &[f32], low: &[f32]) -> WillrGpuTables {
    debug_assert_eq!(high.len(), low.len());
    let high_f64: Vec<f64> = high.iter().map(|&v| v as f64).collect();
    let low_f64: Vec<f64> = low.iter().map(|&v| v as f64).collect();
    let ctx = SharedWillrCtx::new(&high_f64, &low_f64);
    WillrGpuTables::from_shared_ctx(&ctx)
}

#[inline(always)]
fn willr_row_shared_core(
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
    ctx: &SharedWillrCtx,
) {
    let n = close.len();
    let start_i = first_valid + period - 1;
    if start_i >= n {
        return;
    }

    for i in start_i..n {
        let c = close[i];
        if c.is_nan() {
            out[i] = f64::NAN;
            continue;
        }

        let l_idx = i + 1 - period;
        if ctx.window_has_nan(l_idx, i) {
            out[i] = f64::NAN;
            continue;
        }

        let h = ctx.qmax(l_idx, i);
        let l = ctx.qmin(l_idx, i);

        if !h.is_finite() || !l.is_finite() {
            out[i] = f64::NAN;
            continue;
        }

        let denom = h - l;
        out[i] = if denom == 0.0 {
            0.0
        } else {
            (h - c) / denom * -100.0
        };
    }
}

#[inline(always)]
fn willr_row_scalar_with_ctx(
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
    ctx: &SharedWillrCtx,
) {
    willr_row_shared_core(close, first_valid, period, out, ctx);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn willr_row_avx2_with_ctx(
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
    ctx: &SharedWillrCtx,
) {
    willr_row_shared_core(close, first_valid, period, out, ctx);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn willr_row_avx512_with_ctx(
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
    ctx: &SharedWillrCtx,
) {
    willr_row_shared_core(close, first_valid, period, out, ctx);
}

#[inline(always)]
pub fn willr_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    kern: Kernel,
) -> Result<WillrBatchOutput, WillrError> {
    willr_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn willr_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    kern: Kernel,
) -> Result<WillrBatchOutput, WillrError> {
    willr_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn willr_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<WillrBatchOutput, WillrError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(WillrError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let len = high.len();
    if combos.iter().any(|c| c.period == Some(0)) {
        return Err(WillrError::InvalidPeriod {
            period: 0,
            data_len: len,
        });
    }
    if low.len() != len || close.len() != len || len == 0 {
        return Err(WillrError::EmptyOrMismatched);
    }

    let first_valid = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(WillrError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first_valid < max_p {
        return Err(WillrError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid,
        });
    }
    let rows = combos.len();
    let cols = len;

    // Create uninitialized matrix and set up warmup periods
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first_valid + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    // Convert to Vec<f64> using ManuallyDrop for safety
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let mut values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    let ctx = SharedWillrCtx::new(high, low);

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => willr_row_scalar_with_ctx(close, first_valid, period, out_row, &ctx),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => unsafe {
                willr_row_avx2_with_ctx(close, first_valid, period, out_row, &ctx)
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => unsafe {
                willr_row_avx512_with_ctx(close, first_valid, period, out_row, &ctx)
            },
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(WillrBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn willr_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        willr_row_avx512_short(high, low, close, first_valid, period, out);
    } else {
        willr_row_avx512_long(high, low, close, first_valid, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

// --- Streaming implementation ---

pub struct WillrStream {
    period: usize,
    high_buffer: Vec<f64>,
    low_buffer: Vec<f64>,
    count: usize,
}

impl WillrStream {
    pub fn try_new(params: WillrParams) -> Result<Self, WillrError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(WillrError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }

        Ok(Self {
            period,
            high_buffer: vec![f64::NAN; period],
            low_buffer: vec![f64::NAN; period],
            count: 0,
        })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        // Store values in ring buffer
        let idx = self.count % self.period;
        self.high_buffer[idx] = high;
        self.low_buffer[idx] = low;
        self.count += 1;

        // Need at least period values before we can calculate
        if self.count < self.period {
            return None;
        }

        // Calculate Williams %R
        let start_idx = if self.count > self.period {
            (self.count - self.period) % self.period
        } else {
            0
        };

        let mut highest_high = f64::NEG_INFINITY;
        let mut lowest_low = f64::INFINITY;
        let mut has_nan = false;

        // Find highest high and lowest low in the period
        for i in 0..self.period {
            let buffer_idx = (start_idx + i) % self.period;
            let h = self.high_buffer[buffer_idx];
            let l = self.low_buffer[buffer_idx];

            if h.is_nan() || l.is_nan() || close.is_nan() {
                has_nan = true;
                break;
            }

            if h > highest_high {
                highest_high = h;
            }
            if l < lowest_low {
                lowest_low = l;
            }
        }

        if has_nan || highest_high.is_infinite() || lowest_low.is_infinite() {
            return Some(f64::NAN);
        }

        let denom = highest_high - lowest_low;
        if denom == 0.0 {
            Some(0.0)
        } else {
            Some((highest_high - close) / denom * -100.0)
        }
    }
}

#[inline(always)]
fn willr_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<WillrParams>, WillrError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(WillrError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let len = high.len();
    if combos.iter().any(|c| c.period == Some(0)) {
        return Err(WillrError::InvalidPeriod {
            period: 0,
            data_len: len,
        });
    }
    if low.len() != len || close.len() != len || len == 0 {
        return Err(WillrError::EmptyOrMismatched);
    }

    let rows = combos.len();
    let cols = len;

    // Ensure output buffer has correct size
    if out.len() != rows * cols {
        return Err(WillrError::OutputLenMismatch {
            dst: out.len(),
            data_len: rows * cols,
        });
    }

    let first_valid = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(WillrError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first_valid < max_p {
        return Err(WillrError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid,
        });
    }

    let ctx = SharedWillrCtx::new(high, low);

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        // Initialize warmup period with NaN
        let warmup_end = first_valid + period - 1;
        for v in &mut out_row[..warmup_end] {
            *v = f64::NAN;
        }
        match kern {
            Kernel::Scalar => willr_row_scalar_with_ctx(close, first_valid, period, out_row, &ctx),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => unsafe {
                willr_row_avx2_with_ctx(close, first_valid, period, out_row, &ctx)
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => unsafe {
                willr_row_avx512_with_ctx(close, first_valid, period, out_row, &ctx)
            },
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_willr_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = WillrParams { period: None };
        let input = WillrInput::from_candles(&candles, params);
        let output = willr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_willr_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = WillrParams { period: Some(14) };
        let input = WillrInput::from_candles(&candles, params);
        let output = willr_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -58.72876391329818,
            -61.77504393673111,
            -65.93438781487991,
            -60.27950310559006,
            -65.00449236298293,
        ];
        let start = output.values.len() - 5;
        for (i, &val) in output.values[start..].iter().enumerate() {
            assert!(
                (val - expected_last_five[i]).abs() < 1e-8,
                "[{}] WILLR {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_willr_with_slice_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [1.0, 2.0, 3.0, 4.0];
        let low = [0.5, 1.5, 2.5, 3.5];
        let close = [0.75, 1.75, 2.75, 3.75];
        let params = WillrParams { period: Some(2) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let output = willr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), 4);
        Ok(())
    }

    fn check_willr_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [1.0, 2.0];
        let low = [0.8, 1.8];
        let close = [1.0, 2.0];
        let params = WillrParams { period: Some(0) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let res = willr_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WILLR should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_willr_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [1.0, 2.0, 3.0];
        let low = [0.5, 1.5, 2.5];
        let close = [1.0, 2.0, 3.0];
        let params = WillrParams { period: Some(10) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let res = willr_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WILLR should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_willr_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let params = WillrParams::default();
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let res = willr_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WILLR should fail with all NaN",
            test_name
        );
        Ok(())
    }

    fn check_willr_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 2.0];
        let low = [f64::NAN, 1.0];
        let close = [f64::NAN, 1.5];
        let params = WillrParams { period: Some(3) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let res = willr_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WILLR should fail with not enough valid data",
            test_name
        );
        Ok(())
    }

    macro_rules! generate_all_willr_tests {
        ($($test_fn:ident),*) => {
            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    #[cfg(debug_assertions)]
    fn check_willr_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            WillrParams::default(),            // period: 14
            WillrParams { period: Some(2) },   // minimum viable period
            WillrParams { period: Some(5) },   // small period
            WillrParams { period: Some(10) },  // medium-small period
            WillrParams { period: Some(20) },  // medium period
            WillrParams { period: Some(30) },  // medium-large period
            WillrParams { period: Some(50) },  // large period
            WillrParams { period: Some(100) }, // very large period
            WillrParams { period: Some(200) }, // edge case large period
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = WillrInput::from_candles(&candles, params.clone());
            let output = willr_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_willr_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_willr_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating test data
        // Period range: 2-50 for realistic testing
        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                // Generate high, low, close prices with proper constraints
                prop::collection::vec(
                    // Generate (low, high_offset, close_ratio) to ensure high >= low and close within range
                    (1.0f64..1000.0f64).prop_flat_map(|low| {
                        (
                            Just(low),
                            0.0f64..100.0f64, // high_offset to ensure high >= low
                            0.0f64..=1.0f64,  // close_ratio for close within [low, high]
                        )
                    }),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(price_data, period)| {
            // Construct high, low, close arrays from generated data
            let mut high = Vec::with_capacity(price_data.len());
            let mut low = Vec::with_capacity(price_data.len());
            let mut close = Vec::with_capacity(price_data.len());

            for (l, h_offset, c_ratio) in price_data {
                let h = l + h_offset;
                let c = l + (h - l) * c_ratio;
                high.push(h);
                low.push(l);
                close.push(c);
            }

            let params = WillrParams {
                period: Some(period),
            };
            let input = WillrInput::from_slices(&high, &low, &close, params.clone());

            // Get output from kernel under test
            let WillrOutput { values: out } = willr_with_kernel(&input, kernel).unwrap();

            // Get reference output from scalar kernel
            let WillrOutput { values: ref_out } =
                willr_with_kernel(&input, Kernel::Scalar).unwrap();

            // Property 1: Warmup period - first (period-1) values should be NaN
            for i in 0..(period - 1) {
                prop_assert!(
                    out[i].is_nan(),
                    "Warmup period violation at index {}: expected NaN, got {}",
                    i,
                    out[i]
                );
            }

            // Test properties for valid output values
            for i in (period - 1)..high.len() {
                let y = out[i];
                let r = ref_out[i];

                if y.is_nan() {
                    // If NaN, reference should also be NaN
                    prop_assert!(r.is_nan(), "NaN mismatch at index {}", i);
                    continue;
                }

                // Property 2: Output bounds - Williams %R should be between -100 and 0
                prop_assert!(
                    y >= -100.0 - 1e-9 && y <= 0.0 + 1e-9,
                    "Output bounds violation at index {}: {} not in [-100, 0]",
                    i,
                    y
                );

                // Property 3: Extreme values validation
                let window_start = i + 1 - period;
                let window_high = high[window_start..=i]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let window_low = low[window_start..=i]
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);

                // If close equals highest high, %R should be close to 0
                if (close[i] - window_high).abs() < 1e-10 {
                    prop_assert!(
                        y.abs() < 1e-6,
                        "When close = highest high, %R should be ~0, got {} at index {}",
                        y,
                        i
                    );
                }

                // If close equals lowest low, %R should be close to -100
                if (close[i] - window_low).abs() < 1e-10 {
                    prop_assert!(
                        (y + 100.0).abs() < 1e-6,
                        "When close = lowest low, %R should be ~-100, got {} at index {}",
                        y,
                        i
                    );
                }

                // Property 4: Single period case
                if period == 1 {
                    let expected = if high[i] == low[i] {
                        0.0 // When high = low, denominator is 0, so %R = 0
                    } else {
                        (high[i] - close[i]) / (high[i] - low[i]) * -100.0
                    };
                    prop_assert!(
                        (y - expected).abs() < 1e-9,
                        "Period=1 mismatch at index {}: expected {}, got {}",
                        i,
                        expected,
                        y
                    );
                }

                // Property 5: Constant prices
                let window_highs = &high[window_start..=i];
                let window_lows = &low[window_start..=i];
                let window_closes = &close[window_start..=i];

                let all_equal = window_highs.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && window_lows.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && window_closes
                        .windows(2)
                        .all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && (window_highs[0] - window_lows[0]).abs() < 1e-10;

                if all_equal {
                    prop_assert!(
                        y.abs() < 1e-6,
                        "With constant equal prices, %R should be ~0, got {} at index {}",
                        y,
                        i
                    );
                }

                // Property 6: Kernel consistency
                if !r.is_finite() {
                    prop_assert!(
                        y.to_bits() == r.to_bits(),
                        "NaN/Inf mismatch at index {}: {} vs {}",
                        i,
                        y,
                        r
                    );
                } else {
                    let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch at index {}: {} vs {} (diff: {}, ULP: {})",
                        i,
                        y,
                        r,
                        (y - r).abs(),
                        ulp_diff
                    );
                }

                // Property 7: Mathematical bounds based on window
                // %R formula: ((high - close) / (high - low)) * -100
                // Since close is within [low, high], the fraction is in [0, 1]
                // So %R is in [-100, 0]
                let range = window_high - window_low;
                if range > 1e-10 {
                    let theoretical_value = (window_high - close[i]) / range * -100.0;
                    // Allow for some numerical error
                    prop_assert!(
                        (y - theoretical_value).abs() < 1e-6,
                        "Mathematical formula mismatch at index {}: expected {}, got {}",
                        i,
                        theoretical_value,
                        y
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_willr_tests!(
        check_willr_partial_params,
        check_willr_accuracy,
        check_willr_with_slice_data,
        check_willr_zero_period,
        check_willr_period_exceeds_length,
        check_willr_all_nan,
        check_willr_not_enough_valid_data,
        check_willr_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_willr_tests!(check_willr_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = WillrBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = WillrParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -58.72876391329818,
            -61.77504393673111,
            -65.93438781487991,
            -60.27950310559006,
            -65.00449236298293,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            (2, 10, 2),     // Small periods with step 2
            (5, 25, 5),     // Medium periods with step 5
            (10, 50, 10),   // Large periods with step 10
            (2, 5, 1),      // Dense small range
            (14, 14, 0),    // Single period (default)
            (30, 60, 15),   // Large periods with large step
            (50, 100, 25),  // Very large periods
            (100, 200, 50), // Edge case large periods
        ];

        for (cfg_idx, &(period_start, period_end, period_step)) in test_configs.iter().enumerate() {
            let output = WillrBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_start, period_end, period_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

// --- Python bindings ---

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "willr")]
#[pyo3(signature = (high, low, close, period, kernel=None))]
pub fn willr_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = WillrParams {
        period: Some(period),
    };
    let input = WillrInput::from_slices(high_slice, low_slice, close_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| willr_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "WillrStream")]
pub struct WillrStreamPy {
    stream: WillrStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WillrStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = WillrParams {
            period: Some(period),
        };
        let stream =
            WillrStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(WillrStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "willr_batch")]
#[pyo3(signature = (high, low, close, period_range, kernel=None))]
pub fn willr_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    let sweep = WillrBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high_slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            willr_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                simd,
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "willr_cuda_batch_dev")]
#[pyo3(signature = (high, low, close, period_range, device_id=0))]
pub fn willr_cuda_batch_dev_py(
    py: Python<'_>,
    high: PyReadonlyArray1<'_, f32>,
    low: PyReadonlyArray1<'_, f32>,
    close: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }

    let sweep = WillrBatchRange {
        period: period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaWillr::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.willr_batch_dev(high_slice, low_slice, close_slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

// --- WASM bindings ---

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn willr_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, JsValue> {
    if high.len() == 0 || low.len() != high.len() || close.len() != high.len() {
        return Err(JsValue::from_str("mismatched input lengths"));
    }
    let params = WillrParams {
        period: Some(period),
    };
    let input = WillrInput::from_slices(high, low, close, params);
    let mut out = vec![0.0; high.len()];
    willr_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WillrBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WillrBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WillrParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = willr_batch)]
pub fn willr_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: WillrBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = WillrBatchRange {
        period: cfg.period_range,
    };
    let out = willr_batch_inner(high, low, close, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_out = WillrBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js_out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// raw buffer helpers
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn willr_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn willr_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

// in-place compute with separate H/L/C pointers
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn willr_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to willr_into"));
    }
    unsafe {
        let high = core::slice::from_raw_parts(high_ptr, len);
        let low = core::slice::from_raw_parts(low_ptr, len);
        let close = core::slice::from_raw_parts(close_ptr, len);
        let mut out = core::slice::from_raw_parts_mut(out_ptr, len);
        let params = WillrParams {
            period: Some(period),
        };
        let input = WillrInput::from_slices(high, low, close, params);
        willr_into_slice(&mut out, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = willr_batch_into)]
pub fn willr_batch_into_js(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to willr_batch_into"));
    }
    unsafe {
        let high = core::slice::from_raw_parts(high_ptr, len);
        let low = core::slice::from_raw_parts(low_ptr, len);
        let close = core::slice::from_raw_parts(close_ptr, len);

        let sweep = WillrBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = core::slice::from_raw_parts_mut(out_ptr, rows * cols);
        willr_batch_inner_into(high, low, close, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
