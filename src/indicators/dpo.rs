use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for DpoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DpoData::Slice(slice) => slice,
            DpoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DpoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DpoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DpoParams {
    pub period: Option<usize>,
}

impl Default for DpoParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct DpoInput<'a> {
    pub data: DpoData<'a>,
    pub params: DpoParams,
}

impl<'a> DpoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DpoParams) -> Self {
        Self {
            data: DpoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DpoParams) -> Self {
        Self {
            data: DpoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DpoParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DpoBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DpoBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DpoBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<DpoOutput, DpoError> {
        let p = DpoParams {
            period: self.period,
        };
        let i = DpoInput::from_candles(c, "close", p);
        dpo_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DpoOutput, DpoError> {
        let p = DpoParams {
            period: self.period,
        };
        let i = DpoInput::from_slice(d, p);
        dpo_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DpoStream, DpoError> {
        let p = DpoParams {
            period: self.period,
        };
        DpoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DpoError {
    #[error("dpo: Input data slice is empty.")]
    EmptyInputData,
    #[error("dpo: All values are NaN.")]
    AllValuesNaN,

    #[error("dpo: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("dpo: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl From<DpoError> for JsValue {
    fn from(err: DpoError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

#[inline]
pub fn dpo(input: &DpoInput) -> Result<DpoOutput, DpoError> {
    dpo_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn dpo_into_slice(dst: &mut [f64], input: &DpoInput, kern: Kernel) -> Result<(), DpoError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(DpoError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DpoError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(DpoError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(DpoError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if dst.len() != len {
        return Err(DpoError::InvalidPeriod {
            period: dst.len(),
            data_len: len,
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
            dpo_simd128(data, period, first, dst);
        } else {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => dpo_scalar(data, period, first, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => dpo_avx2(data, period, first, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => dpo_avx512(data, period, first, dst),
                _ => unreachable!(),
            }
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => dpo_scalar(data, period, first, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => dpo_avx2(data, period, first, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => dpo_avx512(data, period, first, dst),
                _ => unreachable!(),
            }
        }
    }

    let back = period / 2 + 1;
    let warm = (first + period - 1).max(back);

    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut dst[..warm] {
        *v = qnan;
    }

    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn dpo_into(input: &DpoInput, out: &mut [f64]) -> Result<(), DpoError> {
    dpo_into_slice(out, input, Kernel::Auto)
}

pub fn dpo_with_kernel(input: &DpoInput, kernel: Kernel) -> Result<DpoOutput, DpoError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(DpoError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DpoError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(DpoError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(DpoError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let back = period / 2 + 1;
    let warm = (first + period - 1).max(back);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, warm);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => dpo_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => dpo_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => dpo_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(DpoOutput { values: out })
}

#[inline(always)]
pub fn dpo_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    let len = data.len();
    if len == 0 {
        return;
    }

    let back = period / 2 + 1;
    let scale = 1.0f64 / (period as f64);

    unsafe {
        let base = first_val;
        let ptr_d = data.as_ptr();
        let ptr_o = out.as_mut_ptr();

        let mut sum = 0.0f64;
        let mut k = 0usize;
        while k < period {
            sum += *ptr_d.add(base + k);
            k += 1;
        }

        let mut i = base + period - 1;

        if i < back {
            let stop = back.min(len.saturating_sub(1));
            while i < stop {
                let next = i + 1;
                sum += *ptr_d.add(next);
                sum -= *ptr_d.add(next - period);
                i = next;
            }
        }

        while i + 3 < len {
            let p0 = *ptr_d.add(i - back);
            *ptr_o.add(i) = sum.mul_add(-scale, p0);

            let a1 = *ptr_d.add(i + 1);
            let r1 = *ptr_d.add(i + 1 - period);
            let s1 = (sum + a1) - r1;
            let p1 = *ptr_d.add(i + 1 - back);
            *ptr_o.add(i + 1) = s1.mul_add(-scale, p1);

            let a2 = *ptr_d.add(i + 2);
            let r2 = *ptr_d.add(i + 2 - period);
            let s2 = (s1 + a2) - r2;
            let p2 = *ptr_d.add(i + 2 - back);
            *ptr_o.add(i + 2) = s2.mul_add(-scale, p2);

            let a3 = *ptr_d.add(i + 3);
            let r3 = *ptr_d.add(i + 3 - period);
            let s3 = (s2 + a3) - r3;
            let p3 = *ptr_d.add(i + 3 - back);
            *ptr_o.add(i + 3) = s3.mul_add(-scale, p3);

            i += 4;
            if i >= len {
                return;
            }
            sum = s3 + *ptr_d.add(i);
            sum -= *ptr_d.add(i - period);
        }

        while i < len {
            if i >= back {
                let p = *ptr_d.add(i - back);
                *ptr_o.add(i) = sum.mul_add(-scale, p);
            }
            if i + 1 < len {
                let next = i + 1;
                sum += *ptr_d.add(next);
                sum -= *ptr_d.add(next - period);
            }
            i += 1;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn dpo_simd128(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    use core::arch::wasm32::*;

    let len = data.len();
    if len == 0 {
        return;
    }

    let back = period / 2 + 1;
    let start_idx = first_val + period - 1;
    let warm = start_idx.max(back);
    if warm >= len {
        return;
    }

    let mut sum = 0.0f64;
    for j in 0..period {
        sum += data[first_val + j];
    }

    let mut cur = start_idx;
    while cur < warm {
        let next = cur + 1;
        sum += data[next] - data[next - period];
        cur = next;
    }

    let scale = 1.0f64 / (period as f64);
    let scale_vec = f64x2_splat(scale);

    let mut i = warm;
    while i + 1 < len {
        let price_vec = v128_load(&data[i - back] as *const f64 as *const v128);

        let sum0 = sum;
        let sum1 = sum0 + data[i + 1] - data[i + 1 - period];
        let sum_vec = f64x2(sum0, sum1);

        let result = f64x2_sub(price_vec, f64x2_mul(sum_vec, scale_vec));
        v128_store(&mut out[i] as *mut f64 as *mut v128, result);

        if i + 2 < len {
            sum = sum1 + data[i + 2] - data[i + 2 - period];
        }
        i += 2;
    }

    if i < len {
        out[i] = data[i - back] - (sum * scale);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dpo_avx2(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    let len = data.len();
    if len == 0 {
        return;
    }
    let back = period / 2 + 1;
    let scale = 1.0f64 / (period as f64);
    unsafe {
        let mut sum = 0.0f64;
        let base = first_val;
        let mut k = 0usize;
        let ptr_d = data.as_ptr();
        while k < period {
            sum += *ptr_d.add(base + k);
            k += 1;
        }
        let mut i = base + period - 1;

        if i < back {
            let stop = back.min(len.saturating_sub(1));
            while i < stop {
                sum += *ptr_d.add(i + 1) - *ptr_d.add(i + 1 - period);
                i += 1;
            }
            if i >= len {
                return;
            }
        }

        let ptr_o = out.as_mut_ptr();
        while i + 3 < len {
            let p0 = *ptr_d.add(i - back);
            *ptr_o.add(i) = sum.mul_add(-scale, p0);

            let a1 = *ptr_d.add(i + 1);
            let r1 = *ptr_d.add(i + 1 - period);
            let s1 = sum + (a1 - r1);
            if i + 1 >= back {
                let p1 = *ptr_d.add(i + 1 - back);
                *ptr_o.add(i + 1) = s1.mul_add(-scale, p1);
            }

            let a2 = *ptr_d.add(i + 2);
            let r2 = *ptr_d.add(i + 2 - period);
            let s2 = s1 + (a2 - r2);
            if i + 2 >= back {
                let p2 = *ptr_d.add(i + 2 - back);
                *ptr_o.add(i + 2) = s2.mul_add(-scale, p2);
            }

            let a3 = *ptr_d.add(i + 3);
            let r3 = *ptr_d.add(i + 3 - period);
            let s3 = s2 + (a3 - r3);
            if i + 3 >= back {
                let p3 = *ptr_d.add(i + 3 - back);
                *ptr_o.add(i + 3) = s3.mul_add(-scale, p3);
            }

            i += 4;
            if i >= len {
                return;
            }
            let a4 = *ptr_d.add(i);
            let r4 = *ptr_d.add(i - period);
            sum = s3 + (a4 - r4);
        }

        while i < len {
            if i >= back {
                let p = *ptr_d.add(i - back);
                *ptr_o.add(i) = sum.mul_add(-scale, p);
            }
            if i + 1 < len {
                sum += *ptr_d.add(i + 1) - *ptr_d.add(i + 1 - period);
            }
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dpo_avx512(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    dpo_avx2(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dpo_avx512_short(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    dpo_avx2(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dpo_avx512_long(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    dpo_avx2(data, period, first_val, out)
}

#[inline]
pub fn dpo_batch_with_kernel(
    data: &[f64],
    sweep: &DpoBatchRange,
    k: Kernel,
) -> Result<DpoBatchOutput, DpoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(DpoError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    dpo_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DpoBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for DpoBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 254, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DpoBatchBuilder {
    range: DpoBatchRange,
    kernel: Kernel,
}

impl DpoBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<DpoBatchOutput, DpoError> {
        dpo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DpoBatchOutput, DpoError> {
        DpoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DpoBatchOutput, DpoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DpoBatchOutput, DpoError> {
        DpoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct DpoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DpoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DpoBatchOutput {
    pub fn row_for_params(&self, p: &DpoParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }

    pub fn values_for(&self, p: &DpoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DpoBatchRange) -> Vec<DpoParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(DpoParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn dpo_batch_slice(
    data: &[f64],
    sweep: &DpoBatchRange,
    kern: Kernel,
) -> Result<DpoBatchOutput, DpoError> {
    dpo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn dpo_batch_par_slice(
    data: &[f64],
    sweep: &DpoBatchRange,
    kern: Kernel,
) -> Result<DpoBatchOutput, DpoError> {
    dpo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub fn dpo_batch_inner_into(
    data: &[f64],
    sweep: &DpoBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DpoParams>, DpoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DpoError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let len = data.len();
    if len == 0 {
        return Err(DpoError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DpoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(DpoError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    debug_assert_eq!(out.len(), rows * cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let back = p / 2 + 1;
            (first + p - 1).max(back)
        })
        .collect();

    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    let mut pfx = vec![0.0f64; cols];
    if cols > 0 {
        let mut acc = 0.0f64;
        for i in 0..cols {
            if i < first {
                pfx[i] = 0.0;
            } else {
                acc += data[i];
                pfx[i] = acc;
            }
        }
    }

    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let back = period / 2 + 1;
        let warm = (first + period - 1).max(back);
        let dst = std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols);
        let scale = 1.0f64 / (period as f64);

        let mut i = warm;
        while i < cols {
            let prev = if i >= period { pfx[i - period] } else { 0.0 };
            let sum = pfx[i] - prev;
            let avg = sum * scale;
            let price = data[i - back];
            dst[i] = price - avg;
            i += 1;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(combos)
}

#[inline(always)]
fn dpo_batch_inner(
    data: &[f64],
    sweep: &DpoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DpoBatchOutput, DpoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DpoError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let len = data.len();
    if len == 0 {
        return Err(DpoError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DpoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(DpoError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let back = p / 2 + 1;
            (first + p - 1).max(back)
        })
        .collect();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut values = unsafe {
        let ptr = buf_mu.as_mut_ptr() as *mut f64;
        let len = buf_mu.len();
        std::mem::forget(buf_mu);
        Vec::from_raw_parts(ptr, len, len)
    };

    let mut pfx = vec![0.0f64; cols];
    if cols > 0 {
        let mut acc = 0.0f64;
        for i in 0..cols {
            if i < first {
                pfx[i] = 0.0;
            } else {
                acc += data[i];
                pfx[i] = acc;
            }
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let back = period / 2 + 1;
        let warm = (first + period - 1).max(back);
        let scale = 1.0f64 / (period as f64);

        let mut i = warm;
        while i < cols {
            let prev = if i >= period { pfx[i - period] } else { 0.0 };
            let sum = pfx[i] - prev;
            let avg = sum * scale;
            let price = data[i - back];
            out_row[i] = price - avg;
            i += 1;
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

    Ok(DpoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn dpo_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dpo_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dpo_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dpo_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dpo_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        dpo_row_avx512_short(data, first, period, out);
    } else {
        dpo_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dpo_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dpo_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dpo_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dpo_avx2(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct DpoStream {
    period: usize,
    back: usize,
    inv_period: f64,

    sma_buf: Vec<f64>,
    lag_buf: Vec<f64>,
    sum: f64,

    sma_head: usize,
    lag_head: usize,
    count: usize,
}

impl DpoStream {
    #[inline]
    pub fn try_new(params: DpoParams) -> Result<Self, DpoError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(DpoError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let back = period / 2 + 1;
        let inv_period = 1.0f64 / (period as f64);

        Ok(Self {
            period,
            back,
            inv_period,

            sma_buf: vec![f64::NAN; period],

            lag_buf: vec![f64::NAN; back + 1],
            sum: 0.0,

            sma_head: 0,
            lag_head: 0,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.lag_buf[self.lag_head] = value;
        self.lag_head += 1;
        if self.lag_head == self.lag_buf.len() {
            self.lag_head = 0;
        }

        let old = self.sma_buf[self.sma_head];
        self.sma_buf[self.sma_head] = value;
        self.sma_head += 1;
        if self.sma_head == self.period {
            self.sma_head = 0;
        }

        if old.is_nan() {
            self.sum += value;
        } else {
            self.sum += value - old;
        }

        self.count += 1;

        if self.count < self.period || self.count <= self.back {
            return None;
        }

        let lagged_value = self.lag_buf[self.lag_head];

        let dpo = (-self.inv_period).mul_add(self.sum, lagged_value);

        Some(dpo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_dpo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = DpoParams { period: None };
        let input = DpoInput::from_candles(&candles, "close", default_params);
        let output = dpo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dpo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DpoInput::from_candles(&candles, "close", DpoParams { period: Some(5) });
        let result = dpo_with_kernel(&input, kernel)?;
        let expected_last_five = [
            65.3999999999287,
            131.3999999999287,
            32.599999999925785,
            98.3999999999287,
            117.99999999992724,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] DPO {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_dpo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DpoInput::with_default_candles(&candles);
        match input.data {
            DpoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected DpoData::Candles"),
        }
        let output = dpo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dpo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = DpoParams { period: Some(0) };
        let input = DpoInput::from_slice(&input_data, params);
        let res = dpo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DPO should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_dpo_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = DpoParams { period: Some(10) };
        let input = DpoInput::from_slice(&data_small, params);
        let res = dpo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DPO should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_dpo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = DpoParams { period: Some(5) };
        let input = DpoInput::from_slice(&single_point, params);
        let res = dpo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DPO should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_dpo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = DpoParams { period: Some(5) };
        let first_input = DpoInput::from_candles(&candles, "close", first_params);
        let first_result = dpo_with_kernel(&first_input, kernel)?;
        let second_params = DpoParams { period: Some(5) };
        let second_input = DpoInput::from_slice(&first_result.values, second_params);
        let second_result = dpo_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_dpo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DpoInput::from_candles(&candles, "close", DpoParams { period: Some(5) });
        let res = dpo_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 20 {
            for (i, &val) in res.values[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    fn check_dpo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = DpoInput::from_candles(
            &candles,
            "close",
            DpoParams {
                period: Some(period),
            },
        );
        let batch_output = dpo_with_kernel(&input, kernel)?.values;
        let mut stream = DpoStream::try_new(DpoParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(dpo_val) => stream_values.push(dpo_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] DPO streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_dpo_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            DpoParams::default(),
            DpoParams { period: Some(1) },
            DpoParams { period: Some(2) },
            DpoParams { period: Some(3) },
            DpoParams { period: Some(5) },
            DpoParams { period: Some(10) },
            DpoParams { period: Some(15) },
            DpoParams { period: Some(20) },
            DpoParams { period: Some(30) },
            DpoParams { period: Some(50) },
            DpoParams { period: Some(75) },
            DpoParams { period: Some(100) },
            DpoParams { period: Some(200) },
            DpoParams { period: Some(500) },
            DpoParams { period: Some(1000) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DpoInput::from_candles(&candles, "close", params.clone());
            let output = dpo_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
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
                        params.period.unwrap_or(5),
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
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_dpo_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_dpo_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period)| {
            let params = DpoParams {
                period: Some(period),
            };
            let input = DpoInput::from_slice(&data, params);

            let DpoOutput { values: out } = dpo_with_kernel(&input, kernel).unwrap();
            let DpoOutput { values: ref_out } = dpo_with_kernel(&input, Kernel::Scalar).unwrap();

            for i in 0..period.min(data.len()) {
                prop_assert!(
                    out[i].is_nan(),
                    "[{}] Expected NaN during warmup at index {}, got {}",
                    test_name,
                    i,
                    out[i]
                );
            }

            for i in period..data.len() {
                prop_assert!(
                    out[i].is_finite(),
                    "[{}] Expected finite value at index {}, got {}",
                    test_name,
                    i,
                    out[i]
                );
            }

            for i in 0..data.len() {
                if out[i].is_nan() && ref_out[i].is_nan() {
                    continue;
                }
                let y_bits = out[i].to_bits();
                let r_bits = ref_out[i].to_bits();
                let ulp_diff = if y_bits > r_bits {
                    y_bits - r_bits
                } else {
                    r_bits - y_bits
                };
                prop_assert!(
                    ulp_diff <= 3,
                    "[{}] Kernel mismatch at idx {}: {} ({:016X}) vs {} ({:016X}), ULP diff: {}",
                    test_name,
                    i,
                    out[i],
                    y_bits,
                    ref_out[i],
                    r_bits,
                    ulp_diff
                );
            }

            let back = period / 2 + 1;
            for i in period..data.len() {
                if i >= back {
                    let sum: f64 = data[i + 1 - period..=i].iter().sum();
                    let avg = sum / period as f64;
                    let expected_dpo = data[i - back] - avg;

                    prop_assert!(
                        (out[i] - expected_dpo).abs() < 1e-9,
                        "[{}] DPO formula mismatch at idx {}: got {}, expected {}",
                        test_name,
                        i,
                        out[i],
                        expected_dpo
                    );
                }
            }

            if data.len() > period {
                let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max_val - min_val;

                for i in period..data.len() {
                    prop_assert!(
                        out[i].abs() <= 2.0 * range,
                        "[{}] DPO exceeds reasonable bounds at idx {}: {} (data range: {})",
                        test_name,
                        i,
                        out[i],
                        range
                    );
                }
            }

            if period == 1 {
                for i in 1..data.len() {
                    let expected = data[i - 1] - data[i];
                    prop_assert!(
                        (out[i] - expected).abs() < 1e-9,
                        "[{}] Period=1 special case failed at idx {}: got {}, expected {}",
                        test_name,
                        i,
                        out[i],
                        expected
                    );
                }
            }

            if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() > period {
                for i in period..data.len() {
                    prop_assert!(
                        out[i].abs() < 1e-9,
                        "[{}] Expected zero DPO for constant data at idx {}, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }
            }

            for i in 0..data.len() {
                if !out[i].is_nan() {
                    let bits = out[i].to_bits();
                    prop_assert!(
                        bits != 0x11111111_11111111
                            && bits != 0x22222222_22222222
                            && bits != 0x33333333_33333333,
                        "[{}] Found poison value at idx {}: {} ({:016X})",
                        test_name,
                        i,
                        out[i],
                        bits
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    macro_rules! generate_all_dpo_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    generate_all_dpo_tests!(
        check_dpo_partial_params,
        check_dpo_accuracy,
        check_dpo_default_candles,
        check_dpo_zero_period,
        check_dpo_period_exceeds_length,
        check_dpo_very_small_dataset,
        check_dpo_reinput,
        check_dpo_nan_handling,
        check_dpo_streaming,
        check_dpo_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_dpo_tests!(check_dpo_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = DpoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = DpoParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            65.3999999999287,
            131.3999999999287,
            32.599999999925785,
            98.3999999999287,
            117.99999999992724,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
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

        let test_configs = vec![
            (1, 10, 1),
            (5, 30, 5),
            (10, 50, 10),
            (50, 200, 50),
            (100, 500, 100),
            (5, 5, 0),
            (2, 20, 2),
            (25, 100, 25),
            (200, 1000, 200),
        ];

        for (cfg_idx, &(period_start, period_end, period_step)) in test_configs.iter().enumerate() {
            let output = DpoBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_start, period_end, period_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

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
                        combo.period.unwrap_or(5)
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
                        combo.period.unwrap_or(5)
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
                        combo.period.unwrap_or(5)
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

    #[test]
    fn test_dpo_into_matches_api() -> Result<(), Box<dyn Error>> {
        let mut data = Vec::with_capacity(512);
        for _ in 0..4 {
            data.push(f64::NAN);
        }
        for i in 0..508 {
            let x = i as f64;
            data.push((0.1 * x).sin() * 50.0 + 0.01 * x);
        }

        let params = DpoParams { period: Some(5) };
        let input = DpoInput::from_slice(&data, params);

        let baseline = dpo(&input)?.values;

        let mut out = vec![0.0; data.len()];

        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            dpo_into(&input, &mut out)?;
            assert_eq!(out.len(), baseline.len());
            for (idx, (a, b)) in out.iter().zip(baseline.iter()).enumerate() {
                let equal = (a.is_nan() && b.is_nan()) || (a == b);
                assert!(
                    equal,
                    "dpo_into parity mismatch at idx {}: got {}, expected {}",
                    idx, a, b
                );
            }
        }

        Ok(())
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
pub type DpoDeviceArrayF32Py = DeviceArrayF32Py;

#[cfg(feature = "python")]
#[pyfunction(name = "dpo")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn dpo_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = DpoParams {
        period: Some(period),
    };
    let input = DpoInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| dpo_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DpoStream")]
pub struct DpoStreamPy {
    stream: DpoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DpoStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = DpoParams {
            period: Some(period),
        };
        let stream =
            DpoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(DpoStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dpo_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn dpo_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = DpoBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
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
                _ => unreachable!(),
            };
            dpo_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dpo_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn dpo_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DpoDeviceArrayF32Py, Bound<'py, PyDict>)> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = DpoBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let inner = py.allow_threads(|| {
        let cuda = crate::cuda::oscillators::dpo_wrapper::CudaDpo::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.dpo_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    let handle = make_device_array_py(device_id, inner)?;
    Ok((handle, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dpo_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn dpo_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DpoDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D array"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;
    let params = DpoParams {
        period: Some(period),
    };
    let inner = py.allow_threads(|| {
        let cuda = crate::cuda::oscillators::dpo_wrapper::CudaDpo::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.dpo_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    make_device_array_py(device_id, inner)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dpo_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = DpoParams {
        period: Some(period),
    };
    let input = DpoInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    dpo_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dpo_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = DpoParams {
            period: Some(period),
        };
        let input = DpoInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            dpo_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            dpo_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dpo_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dpo_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DpoBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DpoBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DpoParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dpo_batch)]
pub fn dpo_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: DpoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = DpoBatchRange {
        period: config.period_range,
    };

    let output = dpo_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = DpoBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dpo_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = DpoBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        dpo_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
