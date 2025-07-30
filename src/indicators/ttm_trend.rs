//! # TTM Trend
//!
//! TTM Trend is a boolean trend indicator that compares each close value to a rolling average
//! of a chosen source (e.g., "hl2", "close", etc.) over a fixed period. Returns `true` if
//! close > average, otherwise `false`.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default 5).
//!
//! ## Errors
//! - **AllValuesNaN**: ttm_trend: All input data values are `NaN`.
//! - **InvalidPeriod**: ttm_trend: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: ttm_trend: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(TtmTrendOutput)`** on success, containing a `Vec<bool>` of length matching the input.
//! - **`Err(TtmTrendError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, 
	init_matrix_prefixes, make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::{self, MaybeUninit};
use thiserror::Error;

// Boolean-specific helper functions for uninitialized memory operations
#[inline(always)]
pub fn alloc_with_false_prefix(len: usize, warm: usize) -> Vec<bool> {
	let warm = warm.min(len);
	
	// Allocate with capacity
	let mut buf: Vec<MaybeUninit<bool>> = Vec::with_capacity(len);
	
	// In release mode, just set length after filling warmup
	#[cfg(not(debug_assertions))]
	{
		unsafe {
			buf.set_len(len);
		}
		for i in 0..warm {
			buf[i].write(false);
		}
	}
	
	// In debug mode, initialize all values to avoid any UB window
	#[cfg(debug_assertions)]
	{
		// Extend to full length with poison values
		for _ in 0..warm {
			buf.push(MaybeUninit::new(false));
		}
		for _ in warm..len {
			buf.push(MaybeUninit::new(true)); // Use true as poison for debugging
		}
	}
	
	// Convert to Vec<bool>
	let ptr = buf.as_mut_ptr() as *mut bool;
	let cap = buf.capacity();
	mem::forget(buf);
	unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

#[inline]
pub fn make_uninit_bool_matrix(rows: usize, cols: usize) -> Vec<MaybeUninit<bool>> {
	let total = rows.checked_mul(cols).expect("rows * cols overflowed usize");
	
	let mut v: Vec<MaybeUninit<bool>> = Vec::new();
	v.try_reserve_exact(total).expect("OOM in make_uninit_bool_matrix");
	
	// In release mode, just set length
	#[cfg(not(debug_assertions))]
	{
		unsafe {
			v.set_len(total);
		}
	}
	
	// DEBUG ONLY: poison all cells
	#[cfg(debug_assertions)]
	{
		for _ in 0..total {
			v.push(MaybeUninit::new(true)); // Use true as poison
		}
	}
	v
}

#[inline]
pub fn init_bool_matrix_prefixes(buf: &mut [MaybeUninit<bool>], cols: usize, warm_prefixes: &[usize]) {
	assert!(
		cols != 0 && buf.len() % cols == 0,
		"`buf` length must be a multiple of `cols`"
	);
	let rows = buf.len() / cols;
	assert_eq!(
		rows,
		warm_prefixes.len(),
		"`warm_prefixes` length must equal number of rows"
	);
	
	// Write false values to warm prefixes
	buf.chunks_exact_mut(cols).zip(warm_prefixes).for_each(|(row, &warm)| {
		assert!(warm <= cols, "warm prefix exceeds row width");
		for cell in &mut row[..warm] {
			cell.write(false);
		}
	});
}

#[derive(Debug, Clone)]
pub enum TtmTrendData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slices { source: &'a [f64], close: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct TtmTrendOutput {
	pub values: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct TtmTrendParams {
	pub period: Option<usize>,
}

impl Default for TtmTrendParams {
	fn default() -> Self {
		Self { period: Some(5) }
	}
}

#[derive(Debug, Clone)]
pub struct TtmTrendInput<'a> {
	pub data: TtmTrendData<'a>,
	pub params: TtmTrendParams,
}

impl<'a> TtmTrendInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: TtmTrendParams) -> Self {
		Self {
			data: TtmTrendData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices(source: &'a [f64], close: &'a [f64], p: TtmTrendParams) -> Self {
		Self {
			data: TtmTrendData::Slices { source, close },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "hl2", TtmTrendParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(5)
	}
	#[inline(always)]
	pub fn as_slices(&self) -> (&[f64], &[f64]) {
		match &self.data {
			TtmTrendData::Slices { source, close } => (source, close),
			TtmTrendData::Candles { candles, source } => (source_type(candles, source), source_type(candles, "close")),
		}
	}
	#[inline(always)]
	pub fn as_ref(&self) -> (&[f64], &[f64]) {
		match &self.data {
			TtmTrendData::Slices { source, close } => (*source, *close),
			TtmTrendData::Candles { candles, source } => (source_type(candles, source), source_type(candles, "close")),
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct TtmTrendBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for TtmTrendBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl TtmTrendBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn period(mut self, n: usize) -> Self {
		self.period = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<TtmTrendOutput, TtmTrendError> {
		let p = TtmTrendParams { period: self.period };
		let i = TtmTrendInput::from_candles(c, "hl2", p);
		ttm_trend_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, src: &[f64], close: &[f64]) -> Result<TtmTrendOutput, TtmTrendError> {
		let p = TtmTrendParams { period: self.period };
		let i = TtmTrendInput::from_slices(src, close, p);
		ttm_trend_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<TtmTrendStream, TtmTrendError> {
		let p = TtmTrendParams { period: self.period };
		TtmTrendStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum TtmTrendError {
	#[error("ttm_trend: All values are NaN.")]
	AllValuesNaN,
	#[error("ttm_trend: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("ttm_trend: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn ttm_trend(input: &TtmTrendInput) -> Result<TtmTrendOutput, TtmTrendError> {
	ttm_trend_with_kernel(input, Kernel::Auto)
}

pub fn ttm_trend_with_kernel(input: &TtmTrendInput, kernel: Kernel) -> Result<TtmTrendOutput, TtmTrendError> {
	let (source, close) = match &input.data {
		TtmTrendData::Candles { candles, source } => {
			let s = source_type(candles, source);
			let c = source_type(candles, "close");
			(s, c)
		}
		TtmTrendData::Slices { source, close } => (*source, *close),
	};

	let len = source.len().min(close.len());
	let period = input.get_period();
	let first = source
		.iter()
		.zip(close.iter())
		.position(|(&s, &c)| !s.is_nan() && !c.is_nan())
		.ok_or(TtmTrendError::AllValuesNaN)?;

	if period == 0 || period > len {
		return Err(TtmTrendError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(TtmTrendError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = first + period - 1;
	let mut out = alloc_with_false_prefix(len, warmup_period);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => ttm_trend_scalar(source, close, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => ttm_trend_avx2(source, close, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => ttm_trend_avx512(source, close, period, first, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(TtmTrendOutput { values: out })
}

#[inline]
pub fn ttm_trend_scalar(source: &[f64], close: &[f64], period: usize, first: usize, out: &mut [bool]) {
	let mut sum = 0.0;
	for &v in &source[first..first + period] {
		sum += v;
	}
	let inv_period = 1.0 / (period as f64);
	let mut idx = first + period - 1;
	if close[idx] > sum * inv_period {
		out[idx] = true;
	}
	idx += 1;
	while idx < source.len().min(close.len()) {
		sum += source[idx] - source[idx - period];
		out[idx] = close[idx] > sum * inv_period;
		idx += 1;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ttm_trend_avx512(source: &[f64], close: &[f64], period: usize, first: usize, out: &mut [bool]) {
	unsafe { ttm_trend_avx512_short(source, close, period, first, out) }
}

#[inline]
pub fn ttm_trend_avx2(source: &[f64], close: &[f64], period: usize, first: usize, out: &mut [bool]) {
	ttm_trend_scalar(source, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ttm_trend_avx512_short(source: &[f64], close: &[f64], period: usize, first: usize, out: &mut [bool]) {
	ttm_trend_scalar(source, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ttm_trend_avx512_long(source: &[f64], close: &[f64], period: usize, first: usize, out: &mut [bool]) {
	ttm_trend_scalar(source, close, period, first, out)
}

#[derive(Debug, Clone)]
pub struct TtmTrendStream {
	period: usize,
	buffer: Vec<f64>,
	sum: f64,
	head: usize,
	filled: bool,
}

impl TtmTrendStream {
	pub fn try_new(params: TtmTrendParams) -> Result<Self, TtmTrendError> {
		let period = params.period.unwrap_or(5);
		if period == 0 {
			return Err(TtmTrendError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buffer: vec![0.0; period],
			sum: 0.0,
			head: 0,
			filled: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, src_val: f64, close_val: f64) -> Option<bool> {
		let old = self.buffer[self.head];
		self.buffer[self.head] = src_val;
		if self.filled {
			self.sum += src_val - old;
		} else {
			self.sum += src_val;
		}
		self.head = (self.head + 1) % self.period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some(close_val > self.sum / (self.period as f64))
	}
}

#[derive(Clone, Debug)]
pub struct TtmTrendBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for TtmTrendBatchRange {
	fn default() -> Self {
		Self { period: (5, 60, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct TtmTrendBatchBuilder {
	range: TtmTrendBatchRange,
	kernel: Kernel,
}

impl TtmTrendBatchBuilder {
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
	pub fn apply_slices(self, source: &[f64], close: &[f64]) -> Result<TtmTrendBatchOutput, TtmTrendError> {
		ttm_trend_batch_with_kernel(source, close, &self.range, self.kernel)
	}
	pub fn with_default_slices(source: &[f64], close: &[f64], k: Kernel) -> Result<TtmTrendBatchOutput, TtmTrendError> {
		TtmTrendBatchBuilder::new().kernel(k).apply_slices(source, close)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TtmTrendBatchOutput, TtmTrendError> {
		let source = source_type(c, src);
		let close = source_type(c, "close");
		self.apply_slices(source, close)
	}
	pub fn with_default_candles(c: &Candles) -> Result<TtmTrendBatchOutput, TtmTrendError> {
		TtmTrendBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "hl2")
	}
}

pub fn ttm_trend_batch_with_kernel(
	source: &[f64],
	close: &[f64],
	sweep: &TtmTrendBatchRange,
	k: Kernel,
) -> Result<TtmTrendBatchOutput, TtmTrendError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(TtmTrendError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	ttm_trend_batch_par_slice(source, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TtmTrendBatchOutput {
	pub values: Vec<bool>,
	pub combos: Vec<TtmTrendParams>,
	pub rows: usize,
	pub cols: usize,
}

impl TtmTrendBatchOutput {
	pub fn row_for_params(&self, p: &TtmTrendParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
	}
	pub fn values_for(&self, p: &TtmTrendParams) -> Option<&[bool]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &TtmTrendBatchRange) -> Vec<TtmTrendParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(TtmTrendParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn ttm_trend_batch_slice(
	source: &[f64],
	close: &[f64],
	sweep: &TtmTrendBatchRange,
	kern: Kernel,
) -> Result<TtmTrendBatchOutput, TtmTrendError> {
	ttm_trend_batch_inner(source, close, sweep, kern, false)
}

#[inline(always)]
pub fn ttm_trend_batch_par_slice(
	source: &[f64],
	close: &[f64],
	sweep: &TtmTrendBatchRange,
	kern: Kernel,
) -> Result<TtmTrendBatchOutput, TtmTrendError> {
	ttm_trend_batch_inner(source, close, sweep, kern, true)
}

#[inline(always)]
fn ttm_trend_batch_inner(
	source: &[f64],
	close: &[f64],
	sweep: &TtmTrendBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<TtmTrendBatchOutput, TtmTrendError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TtmTrendError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let len = source.len().min(close.len());
	let first = source
		.iter()
		.zip(close.iter())
		.position(|(&s, &c)| !s.is_nan() && !c.is_nan())
		.ok_or(TtmTrendError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(TtmTrendError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	
	// Use uninitialized memory
	let mut buf_mu = make_uninit_bool_matrix(rows, cols);
	
	// Calculate warmup periods for each row
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_bool_matrix_prefixes(&mut buf_mu, cols, &warm);
	
	// Convert to bool slice for computation
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let buf_ptr = buf_guard.as_mut_ptr() as *mut bool;
	let values = unsafe { std::slice::from_raw_parts_mut(buf_ptr, rows * cols) };

	let do_row = |row: usize, out_row: &mut [bool]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => ttm_trend_row_scalar(source, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => ttm_trend_row_avx2(source, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => ttm_trend_row_avx512(source, close, first, period, out_row),
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
	
	// Convert back to Vec<bool>
	let values = unsafe {
		let buf_mu = core::mem::ManuallyDrop::into_inner(buf_guard);
		let ptr = buf_mu.as_ptr() as *const bool;
		let len = buf_mu.len();
		let cap = buf_mu.capacity();
		core::mem::forget(buf_mu);
		Vec::from_raw_parts(ptr as *mut bool, len, cap)
	};

	Ok(TtmTrendBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn ttm_trend_batch_inner_into(
	source: &[f64],
	close: &[f64],
	sweep: &TtmTrendBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [bool],
) -> Result<Vec<TtmTrendParams>, TtmTrendError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TtmTrendError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let len = source.len().min(close.len());
	let first = source
		.iter()
		.zip(close.iter())
		.position(|(&s, &c)| !s.is_nan() && !c.is_nan())
		.ok_or(TtmTrendError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(TtmTrendError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	if out.len() != rows * cols {
		return Err(TtmTrendError::InvalidPeriod {
			period: rows * cols,
			data_len: out.len(),
		});
	}

	let do_row = |row: usize, out_row: &mut [bool]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => ttm_trend_row_scalar(source, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => ttm_trend_row_avx2(source, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => ttm_trend_row_avx512(source, close, first, period, out_row),
			_ => unreachable!(),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out
				.par_chunks_mut(cols)
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

#[inline(always)]
unsafe fn ttm_trend_row_scalar(source: &[f64], close: &[f64], first: usize, period: usize, out: &mut [bool]) {
	let mut sum = 0.0;
	for &v in &source[first..first + period] {
		sum += v;
	}
	let inv_p = 1.0 / (period as f64);
	let mut idx = first + period - 1;
	if close[idx] > sum * inv_p {
		out[idx] = true;
	}
	idx += 1;
	while idx < source.len().min(close.len()) {
		sum += source[idx] - source[idx - period];
		out[idx] = close[idx] > sum * inv_p;
		idx += 1;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ttm_trend_row_avx2(source: &[f64], close: &[f64], first: usize, period: usize, out: &mut [bool]) {
	ttm_trend_row_scalar(source, close, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ttm_trend_row_avx512(source: &[f64], close: &[f64], first: usize, period: usize, out: &mut [bool]) {
	if period <= 32 {
		ttm_trend_row_avx512_short(source, close, first, period, out);
	} else {
		ttm_trend_row_avx512_long(source, close, first, period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ttm_trend_row_avx512_short(source: &[f64], close: &[f64], first: usize, period: usize, out: &mut [bool]) {
	ttm_trend_row_scalar(source, close, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ttm_trend_row_avx512_long(source: &[f64], close: &[f64], first: usize, period: usize, out: &mut [bool]) {
	ttm_trend_row_scalar(source, close, first, period, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use paste::paste;

	fn check_ttm_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let default_params = TtmTrendParams { period: None };
		let input = TtmTrendInput::from_candles(&candles, "hl2", default_params);
		let output = ttm_trend_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_ttm_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let close = source_type(&candles, "close");
		let params = TtmTrendParams { period: Some(5) };
		let input = TtmTrendInput::from_candles(&candles, "hl2", params);
		let result = ttm_trend_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), close.len());
		let expected_last_five = [true, false, false, false, false];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			assert_eq!(val, expected_last_five[i]);
		}
		Ok(())
	}

	fn check_ttm_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let src = [10.0, 20.0, 30.0];
		let close = [12.0, 22.0, 32.0];
		let params = TtmTrendParams { period: Some(0) };
		let input = TtmTrendInput::from_slices(&src, &close, params);
		let res = ttm_trend_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}

	fn check_ttm_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let src = [1.0, 2.0, 3.0];
		let close = [1.0, 2.0, 3.0];
		let params = TtmTrendParams { period: Some(10) };
		let input = TtmTrendInput::from_slices(&src, &close, params);
		let res = ttm_trend_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}

	fn check_ttm_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let src = [42.0];
		let close = [43.0];
		let params = TtmTrendParams { period: Some(5) };
		let input = TtmTrendInput::from_slices(&src, &close, params);
		let res = ttm_trend_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}

	fn check_ttm_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let src = [f64::NAN, f64::NAN, f64::NAN];
		let close = [f64::NAN, f64::NAN, f64::NAN];
		let params = TtmTrendParams { period: Some(5) };
		let input = TtmTrendInput::from_slices(&src, &close, params);
		let res = ttm_trend_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}

	fn check_ttm_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let src = source_type(&candles, "hl2");
		let close = source_type(&candles, "close");
		let period = 5;
		let input = TtmTrendInput::from_slices(src, close, TtmTrendParams { period: Some(period) });
		let batch_output = ttm_trend_with_kernel(&input, kernel)?.values;
		let mut stream = TtmTrendStream::try_new(TtmTrendParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(close.len());
		for (&s, &c) in src.iter().zip(close.iter()) {
			match stream.update(s, c) {
				Some(v) => stream_values.push(v),
				None => stream_values.push(false),
			}
		}
		assert_eq!(batch_output, stream_values);
		Ok(())
	}

	macro_rules! generate_all_ttm_tests {
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

	generate_all_ttm_tests!(
		check_ttm_partial_params,
		check_ttm_accuracy,
		check_ttm_zero_period,
		check_ttm_period_exceeds_length,
		check_ttm_very_small_dataset,
		check_ttm_all_nan,
		check_ttm_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let src = source_type(&c, "hl2");
		let close = source_type(&c, "close");
		let output = TtmTrendBatchBuilder::new().kernel(kernel).apply_slices(src, close)?;
		let def = TtmTrendParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), close.len());
		let expected = [true, false, false, false, false];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert_eq!(v, expected[i]);
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
	gen_batch_tests!(check_batch_default_row);
}

// ============================
// ==== Python Bindings =======
// ============================

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "ttm_trend")]
#[pyo3(signature = (source, close, period, kernel=None))]
pub fn ttm_trend_py<'py>(
	py: Python<'py>,
	source: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let source_slice = source.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = TtmTrendParams { period: Some(period) };
	let input = TtmTrendInput::from_slices(source_slice, close_slice, params);

	let result_vec: Vec<bool> = py
		.allow_threads(|| ttm_trend_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TtmTrendStream")]
pub struct TtmTrendStreamPy {
	stream: TtmTrendStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TtmTrendStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = TtmTrendParams { period: Some(period) };
		let stream = TtmTrendStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(TtmTrendStreamPy { stream })
	}

	fn update(&mut self, source_val: f64, close_val: f64) -> Option<bool> {
		self.stream.update(source_val, close_val)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "ttm_trend_batch")]
#[pyo3(signature = (source, close, period_range, kernel=None))]
pub fn ttm_trend_batch_py<'py>(
	py: Python<'py>,
	source: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let source_slice = source.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = TtmTrendBatchRange { period: period_range };
	
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = source_slice.len().min(close_slice.len());
	
	// Pre-allocate output array
	let out_arr = unsafe { PyArray1::<bool>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };
	
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
				_ => kernel,
			};
			ttm_trend_batch_inner_into(source_slice, close_slice, &sweep, simd, true, slice_out)
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
	
	Ok(dict)
}

// ================================
// WASM Bindings
// ================================

/// Write TTM Trend values directly to output slice - no allocations
/// Note: This writes boolean values, but for WASM we'll convert to u8 (0/1)
#[inline]
pub fn ttm_trend_into_slice(
	dst: &mut [bool],
	input: &TtmTrendInput,
	kern: Kernel,
) -> Result<(), TtmTrendError> {
	let (source, close) = input.as_slices();
	let len = source.len().min(close.len());
	let period = input.get_period();
	
	if dst.len() != len {
		return Err(TtmTrendError::InvalidPeriod {
			period: dst.len(),
			data_len: len,
		});
	}
	
	let first = source
		.iter()
		.zip(close.iter())
		.position(|(&s, &c)| !s.is_nan() && !c.is_nan())
		.ok_or(TtmTrendError::AllValuesNaN)?;
		
	if period == 0 || period > len {
		return Err(TtmTrendError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(TtmTrendError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => ttm_trend_scalar(source, close, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => ttm_trend_avx2(source, close, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => ttm_trend_avx512(source, close, period, first, dst),
			_ => ttm_trend_scalar(source, close, period, first, dst),
		}
	}
	
	// The warmup should already be filled by alloc_with_false_prefix when called properly
	// But ensure it's false just in case this is called with a regular slice
	let warmup_end = first.saturating_add(period.saturating_sub(1));
	let dst_len = dst.len();
	for v in &mut dst[..warmup_end.min(dst_len)] {
		*v = false;
	}
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ttm_trend_js(
	source: &[f64],
	close: &[f64],
	period: usize,
) -> Result<Vec<u8>, JsValue> {
	let params = TtmTrendParams { period: Some(period) };
	let input = TtmTrendInput::from_slices(source, close, params);
	
	let len = source.len().min(close.len());
	
	// Find first valid index for warmup calculation
	let first = source
		.iter()
		.zip(close.iter())
		.position(|(&s, &c)| !s.is_nan() && !c.is_nan())
		.ok_or(JsValue::from_str("All values are NaN"))?;
	
	let warmup_period = first + period - 1;
	let mut bool_output = alloc_with_false_prefix(len, warmup_period);
	
	ttm_trend_into_slice(&mut bool_output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Convert bool array to u8 array for WASM - more efficient iteration
	let mut output = Vec::with_capacity(len);
	output.extend(bool_output.iter().map(|&b| if b { 1u8 } else { 0u8 }));
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ttm_trend_into(
	source_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut u8,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if source_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let source = std::slice::from_raw_parts(source_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}
		
		let params = TtmTrendParams { period: Some(period) };
		let input = TtmTrendInput::from_slices(source, close, params);
		
		// Check if output pointer aliases with either input
		let source_ptr_u8 = source_ptr as *const u8;
		let close_ptr_u8 = close_ptr as *const u8;
		let out_ptr_const = out_ptr as *const u8;
		let source_overlaps = (out_ptr_const >= source_ptr_u8 && out_ptr_const < source_ptr_u8.add(len * 8)) ||
							  (source_ptr_u8 >= out_ptr_const && source_ptr_u8 < out_ptr_const.add(len));
		let close_overlaps = (out_ptr_const >= close_ptr_u8 && out_ptr_const < close_ptr_u8.add(len * 8)) ||
							 (close_ptr_u8 >= out_ptr_const && close_ptr_u8 < out_ptr_const.add(len));
		
		if source_overlaps || close_overlaps {
			// Use temporary buffer if aliasing detected
			// Find first valid index
			let first_idx = source
				.iter()
				.zip(close.iter())
				.position(|(&s, &c)| !s.is_nan() && !c.is_nan())
				.ok_or(JsValue::from_str("All values are NaN"))?;
			
			let warmup_period = first_idx.saturating_add(period.saturating_sub(1));
			let mut temp_bool = alloc_with_false_prefix(len, warmup_period);
			ttm_trend_into_slice(&mut temp_bool, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			for (i, &b) in temp_bool.iter().enumerate() {
				out[i] = if b { 1 } else { 0 };
			}
		} else {
			// Direct computation without intermediate bool allocation
			// We need to compute directly into u8 output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			
			// First, find the first valid index
			let first = source
				.iter()
				.zip(close.iter())
				.position(|(&s, &c)| !s.is_nan() && !c.is_nan())
				.ok_or(JsValue::from_str("All values are NaN"))?;
			
			// Fill warmup with 0 (false)
			let warmup_end = first.saturating_add(period.saturating_sub(1));
			for v in &mut out[..warmup_end.min(len)] {
				*v = 0;
			}
			
			// Direct computation into u8 buffer
			let mut sum = 0.0;
			// Initialize sum with first period values
			for &val in &source[first..first + period] {
				sum += val;
			}
			
			// Compute TTM Trend directly into u8
			for i in first + period - 1..len {
				let avg = sum / period as f64;
				out[i] = if close[i] > avg { 1 } else { 0 };
				
				// Update rolling sum
				if i + 1 < len {
					sum += source[i + 1] - source[i + 1 - period];
				}
			}
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ttm_trend_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ttm_trend_alloc_u8(len: usize) -> *mut u8 {
	let mut vec = Vec::<u8>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ttm_trend_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ttm_trend_free_u8(ptr: *mut u8, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TtmTrendBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TtmTrendBatchJsOutput {
	pub values: Vec<u8>,  // Flattened boolean array as u8
	pub periods: Vec<usize>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ttm_trend_batch)]
pub fn ttm_trend_batch_js(
	source: &[f64],
	close: &[f64],
	config: JsValue,
) -> Result<JsValue, JsValue> {
	let config: TtmTrendBatchConfig = 
		serde_wasm_bindgen::from_value(config)
			.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = TtmTrendBatchRange { period: config.period_range };
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = source.len().min(close.len());
	
	// Find first valid index
	let first = source
		.iter()
		.zip(close.iter())
		.position(|(&s, &c)| !s.is_nan() && !c.is_nan())
		.ok_or(JsValue::from_str("All values are NaN"))?;
	
	// Use uninitialized memory
	let mut buf_mu = make_uninit_bool_matrix(rows, cols);
	
	// Calculate warmup periods for each row
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_bool_matrix_prefixes(&mut buf_mu, cols, &warm);
	
	// Convert to bool slice for computation
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let buf_ptr = buf_guard.as_mut_ptr() as *mut bool;
	let bool_buffer = unsafe { std::slice::from_raw_parts_mut(buf_ptr, rows * cols) };
	
	let kernel = detect_best_batch_kernel();
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => kernel,
	};
	
	let periods = ttm_trend_batch_inner_into(source, close, &sweep, simd, false, bool_buffer)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Convert to Vec<bool> and then to u8
	let bool_vec = unsafe {
		let buf_mu = core::mem::ManuallyDrop::into_inner(buf_guard);
		let ptr = buf_mu.as_ptr() as *const bool;
		let len = buf_mu.len();
		let cap = buf_mu.capacity();
		core::mem::forget(buf_mu);
		Vec::from_raw_parts(ptr as *mut bool, len, cap)
	};
	
	// Convert boolean buffer to u8 more efficiently
	let mut values = Vec::with_capacity(rows * cols);
	values.extend(bool_vec.iter().map(|&b| if b { 1u8 } else { 0u8 }));
	
	let output = TtmTrendBatchJsOutput {
		values,
		periods: periods.iter().map(|p| p.period.unwrap()).collect(),
		rows,
		cols,
	};
	
	serde_wasm_bindgen::to_value(&output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
