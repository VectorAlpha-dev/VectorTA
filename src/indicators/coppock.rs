use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for CoppockInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CoppockData::Slice(slice) => slice,
            CoppockData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CoppockData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CoppockOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct CoppockParams {
    pub short_roc_period: Option<usize>,
    pub long_roc_period: Option<usize>,
    pub ma_period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for CoppockParams {
    fn default() -> Self {
        Self {
            short_roc_period: Some(11),
            long_roc_period: Some(14),
            ma_period: Some(10),
            ma_type: Some("wma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoppockInput<'a> {
    pub data: CoppockData<'a>,
    pub params: CoppockParams,
}

impl<'a> CoppockInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CoppockParams) -> Self {
        Self {
            data: CoppockData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CoppockParams) -> Self {
        Self {
            data: CoppockData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CoppockParams::default())
    }
    #[inline]
    pub fn get_short_roc_period(&self) -> usize {
        self.params.short_roc_period.unwrap_or(11)
    }
    #[inline]
    pub fn get_long_roc_period(&self) -> usize {
        self.params.long_roc_period.unwrap_or(14)
    }
    #[inline]
    pub fn get_ma_period(&self) -> usize {
        self.params.ma_period.unwrap_or(10)
    }
    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("wma")
    }
}

#[derive(Clone, Debug)]
pub struct CoppockBuilder {
    short: Option<usize>,
    long: Option<usize>,
    ma: Option<usize>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for CoppockBuilder {
    fn default() -> Self {
        Self {
            short: None,
            long: None,
            ma: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CoppockBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_roc_period(mut self, n: usize) -> Self {
        self.short = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_roc_period(mut self, n: usize) -> Self {
        self.long = Some(n);
        self
    }
    #[inline(always)]
    pub fn ma_period(mut self, n: usize) -> Self {
        self.ma = Some(n);
        self
    }
    #[inline(always)]
    pub fn ma_type<T: Into<String>>(mut self, t: T) -> Self {
        self.ma_type = Some(t.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CoppockOutput, CoppockError> {
        let p = CoppockParams {
            short_roc_period: self.short,
            long_roc_period: self.long,
            ma_period: self.ma,
            ma_type: self.ma_type,
        };
        let i = CoppockInput::from_candles(c, "close", p);
        coppock_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CoppockOutput, CoppockError> {
        let p = CoppockParams {
            short_roc_period: self.short,
            long_roc_period: self.long,
            ma_period: self.ma,
            ma_type: self.ma_type,
        };
        let i = CoppockInput::from_slice(d, p);
        coppock_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CoppockStream, CoppockError> {
        let p = CoppockParams {
            short_roc_period: self.short,
            long_roc_period: self.long,
            ma_period: self.ma,
            ma_type: self.ma_type,
        };
        CoppockStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CoppockError {
    #[error("coppock: Empty data provided.")]
    EmptyData,
    #[error("coppock: All values are NaN.")]
    AllValuesNaN,
    #[error("coppock: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "coppock: Invalid period usage => short={short}, long={long}, ma={ma}, data_len={data_len}"
    )]
    InvalidPeriod {
        short: usize,
        long: usize,
        ma: usize,
        data_len: usize,
    },
    #[error("coppock: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("coppock: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("coppock: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("coppock: Invalid input: {0}")]
    InvalidInput(String),
    #[error("coppock: Underlying MA error: {0}")]
    MaError(#[from] Box<dyn Error + Send + Sync>),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl From<CoppockError> for JsValue {
    fn from(err: CoppockError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

#[inline]
pub fn coppock(input: &CoppockInput) -> Result<CoppockOutput, CoppockError> {
    coppock_with_kernel(input, Kernel::Auto)
}

pub fn coppock_with_kernel(
    input: &CoppockInput,
    kernel: Kernel,
) -> Result<CoppockOutput, CoppockError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(CoppockError::EmptyData);
    }

    let short = input.get_short_roc_period();
    let long = input.get_long_roc_period();
    let ma_p = input.get_ma_period();
    let data_len = data.len();

    if short == 0
        || long == 0
        || ma_p == 0
        || short > data_len
        || long > data_len
        || ma_p > data_len
    {
        return Err(CoppockError::InvalidPeriod {
            short,
            long,
            ma: ma_p,
            data_len,
        });
    }

    let first = data
        .iter()
        .position(|&x| !x.is_nan())
        .ok_or(CoppockError::AllValuesNaN)?;
    let largest_roc = short.max(long);
    if (data_len - first) < largest_roc {
        return Err(CoppockError::NotEnoughValidData {
            needed: largest_roc,
            valid: data_len - first,
        });
    }

    let warmup_period = first + largest_roc;

    let mut sum_roc = alloc_with_nan_prefix(data_len, warmup_period);

    unsafe {
        match match kernel {
            Kernel::Auto => Kernel::Scalar,
            other => other,
        } {
            Kernel::Scalar | Kernel::ScalarBatch => {
                coppock_scalar(data, short, long, first, &mut sum_roc)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                coppock_avx2(data, short, long, first, &mut sum_roc)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                coppock_avx512(data, short, long, first, &mut sum_roc)
            }
            _ => coppock_scalar(data, short, long, first, &mut sum_roc),
        }
    }

    let ma_type = input.get_ma_type();
    let smoothed = ma(&ma_type, MaData::Slice(&sum_roc), ma_p).map_err(|e| {
        use std::fmt;
        #[derive(Debug)]
        struct MaErrorWrapper(String);
        impl fmt::Display for MaErrorWrapper {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl Error for MaErrorWrapper {}
        CoppockError::MaError(Box::new(MaErrorWrapper(e.to_string())))
    })?;

    Ok(CoppockOutput { values: smoothed })
}

#[inline]
pub fn coppock_into_slice(
    out: &mut [f64],
    input: &CoppockInput,
    kernel: Kernel,
) -> Result<(), CoppockError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(CoppockError::EmptyData);
    }
    if out.len() != data.len() {
        return Err(CoppockError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let short = input.get_short_roc_period();
    let long = input.get_long_roc_period();
    let ma_p = input.get_ma_period();
    let data_len = data.len();

    if short == 0
        || long == 0
        || ma_p == 0
        || short > data_len
        || long > data_len
        || ma_p > data_len
    {
        return Err(CoppockError::InvalidPeriod {
            short,
            long,
            ma: ma_p,
            data_len,
        });
    }

    let first = data
        .iter()
        .position(|&x| !x.is_nan())
        .ok_or(CoppockError::AllValuesNaN)?;
    let largest_roc = short.max(long);
    if (data_len - first) < largest_roc {
        return Err(CoppockError::NotEnoughValidData {
            needed: largest_roc,
            valid: data_len - first,
        });
    }

    let warmup_period = first + largest_roc;

    let mut sum_roc = alloc_with_nan_prefix(data_len, warmup_period);

    let resolved_kernel = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let ma_type = input.get_ma_type();

    if resolved_kernel == Kernel::Scalar
        && (ma_type == "wma" || ma_type == "sma" || ma_type == "ema")
    {
        unsafe {
            return coppock_scalar_classic(data, short, long, ma_p, ma_type, first, out);
        }
    }

    unsafe {
        match resolved_kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                coppock_scalar(data, short, long, first, &mut sum_roc)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                coppock_avx2(data, short, long, first, &mut sum_roc)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                coppock_avx512(data, short, long, first, &mut sum_roc)
            }
            _ => coppock_scalar(data, short, long, first, &mut sum_roc),
        }
    }

    let smoothed = ma(ma_type, MaData::Slice(&sum_roc), ma_p).map_err(|e| {
        use std::fmt;
        #[derive(Debug)]
        struct MaErrorWrapper(String);
        impl fmt::Display for MaErrorWrapper {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl Error for MaErrorWrapper {}
        CoppockError::MaError(Box::new(MaErrorWrapper(e.to_string())))
    })?;

    out.copy_from_slice(&smoothed);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn coppock_into(input: &CoppockInput, out: &mut [f64]) -> Result<(), CoppockError> {
    coppock_into_slice(out, input, Kernel::ScalarBatch)
}

pub unsafe fn coppock_scalar_classic(
    data: &[f64],
    short: usize,
    long: usize,
    ma_period: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) -> Result<(), CoppockError> {
    let len = data.len();
    let largest_roc = short.max(long);
    let warmup_period = first + largest_roc;

    let mut sum_roc = alloc_with_nan_prefix(len, warmup_period);
    let start_idx = first + largest_roc;

    for i in start_idx..len {
        let current = data[i];
        let prev_short = data[i - short];
        let short_val = ((current / prev_short) - 1.0) * 100.0;
        let prev_long = data[i - long];
        let long_val = ((current / prev_long) - 1.0) * 100.0;
        sum_roc[i] = short_val + long_val;
    }

    match ma_type {
        "wma" => {
            let warmup_final = warmup_period + ma_period - 1;
            for i in 0..warmup_final.min(len) {
                out[i] = f64::NAN;
            }

            let weight_sum = (ma_period * (ma_period + 1)) as f64 / 2.0;

            for i in warmup_final..len {
                let mut weighted_sum = 0.0;
                let mut has_nan = false;

                for j in 0..ma_period {
                    let idx = i - ma_period + 1 + j;
                    if sum_roc[idx].is_nan() {
                        has_nan = true;
                        break;
                    }
                    weighted_sum += sum_roc[idx] * (j + 1) as f64;
                }

                out[i] = if has_nan {
                    f64::NAN
                } else {
                    weighted_sum / weight_sum
                };
            }
        }
        "sma" => {
            let warmup_final = warmup_period + ma_period - 1;
            for i in 0..warmup_final.min(len) {
                out[i] = f64::NAN;
            }

            let mut sum = 0.0;
            for i in warmup_period..(warmup_period + ma_period.min(len - warmup_period)) {
                if !sum_roc[i].is_nan() {
                    sum += sum_roc[i];
                }
            }

            if warmup_final < len {
                out[warmup_final] = sum / ma_period as f64;

                for i in (warmup_final + 1)..len {
                    if !sum_roc[i].is_nan() && !sum_roc[i - ma_period].is_nan() {
                        sum += sum_roc[i] - sum_roc[i - ma_period];
                        out[i] = sum / ma_period as f64;
                    } else {
                        out[i] = f64::NAN;
                    }
                }
            }
        }
        "ema" => {
            let warmup_final = warmup_period + ma_period - 1;
            for i in 0..warmup_final.min(len) {
                out[i] = f64::NAN;
            }

            let alpha = 2.0 / (ma_period as f64 + 1.0);
            let mut ema_value = f64::NAN;

            for i in warmup_period..len {
                if !sum_roc[i].is_nan() {
                    ema_value = sum_roc[i];
                    out[i] = ema_value;

                    for j in (i + 1)..len {
                        if !sum_roc[j].is_nan() {
                            ema_value = alpha * sum_roc[j] + (1.0 - alpha) * ema_value;
                            out[j] = ema_value;
                        } else {
                            out[j] = f64::NAN;
                        }
                    }
                    break;
                }
            }
        }
        _ => {
            let smoothed = ma(ma_type, MaData::Slice(&sum_roc), ma_period).map_err(|e| {
                CoppockError::MaError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                )))
            })?;
            out.copy_from_slice(&smoothed);
        }
    }

    Ok(())
}

#[inline]
pub fn coppock_scalar(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    let largest = short.max(long);
    let start_idx = first + largest;
    for i in start_idx..data.len() {
        let current = data[i];
        let prev_short = data[i - short];
        let short_val = ((current / prev_short) - 1.0) * 100.0;
        let prev_long = data[i - long];
        let long_val = ((current / prev_long) - 1.0) * 100.0;
        out[i] = short_val + long_val;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn coppock_avx2(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    let largest = short.max(long);
    let start = first + largest;
    let n = data.len();
    if start >= n {
        return;
    }

    let mut p_cur = data.as_ptr().add(start);
    let mut p_ps = data.as_ptr().add(start - short);
    let mut p_pl = data.as_ptr().add(start - long);
    let mut p_out = out.as_mut_ptr().add(start);

    let remaining = n - start;
    let step = 4usize;
    let vec_len = remaining / step * step;

    let v_one = _mm256_set1_pd(1.0);
    let v_scale = _mm256_set1_pd(100.0);

    let end_vec = p_cur.add(vec_len);
    while p_cur < end_vec {
        let vc = _mm256_loadu_pd(p_cur);
        let vs = _mm256_loadu_pd(p_ps);
        let vl = _mm256_loadu_pd(p_pl);

        let r_s = _mm256_div_pd(vc, vs);
        let r_l = _mm256_div_pd(vc, vl);

        let t0 = _mm256_mul_pd(_mm256_sub_pd(r_s, v_one), v_scale);
        let t1 = _mm256_mul_pd(_mm256_sub_pd(r_l, v_one), v_scale);
        let res = _mm256_add_pd(t0, t1);

        _mm256_storeu_pd(p_out, res);

        p_cur = p_cur.add(step);
        p_ps = p_ps.add(step);
        p_pl = p_pl.add(step);
        p_out = p_out.add(step);
    }

    let tail = remaining - vec_len;
    for _ in 0..tail {
        let c = *p_cur;
        let s = *p_ps;
        let l = *p_pl;
        let rs = (c / s - 1.0) * 100.0;
        let rl = (c / l - 1.0) * 100.0;
        *p_out = rs + rl;

        p_cur = p_cur.add(1);
        p_ps = p_ps.add(1);
        p_pl = p_pl.add(1);
        p_out = p_out.add(1);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn coppock_avx512(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    if short.max(long) <= 32 {
        coppock_avx512_short(data, short, long, first, out)
    } else {
        coppock_avx512_long(data, short, long, first, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn coppock_avx512_short(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let largest = short.max(long);
    let start = first + largest;
    let n = data.len();
    if start >= n {
        return;
    }

    let mut p_cur = data.as_ptr().add(start);
    let mut p_ps = data.as_ptr().add(start - short);
    let mut p_pl = data.as_ptr().add(start - long);
    let mut p_out = out.as_mut_ptr().add(start);

    let remaining = n - start;
    let step = 8usize;
    let vec_len = remaining / step * step;

    let v_one = _mm512_set1_pd(1.0);
    let v_scale = _mm512_set1_pd(100.0);

    let end_vec = p_cur.add(vec_len);
    while p_cur < end_vec {
        let vc = _mm512_loadu_pd(p_cur);
        let vs = _mm512_loadu_pd(p_ps);
        let vl = _mm512_loadu_pd(p_pl);

        let r_s = _mm512_div_pd(vc, vs);
        let r_l = _mm512_div_pd(vc, vl);

        let t0 = _mm512_mul_pd(_mm512_sub_pd(r_s, v_one), v_scale);
        let t1 = _mm512_mul_pd(_mm512_sub_pd(r_l, v_one), v_scale);
        let res = _mm512_add_pd(t0, t1);

        _mm512_storeu_pd(p_out, res);

        p_cur = p_cur.add(step);
        p_ps = p_ps.add(step);
        p_pl = p_pl.add(step);
        p_out = p_out.add(step);
    }

    let tail = remaining - vec_len;
    for _ in 0..tail {
        let c = *p_cur;
        let s = *p_ps;
        let l = *p_pl;
        let rs = (c / s - 1.0) * 100.0;
        let rl = (c / l - 1.0) * 100.0;
        *p_out = rs + rl;

        p_cur = p_cur.add(1);
        p_ps = p_ps.add(1);
        p_pl = p_pl.add(1);
        p_out = p_out.add(1);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn coppock_avx512_long(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    coppock_avx512_short(data, short, long, first, out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaMode {
    Wma,
    Sma,
    Ema,
    Unsupported,
}

#[derive(Debug, Clone)]
pub struct CoppockStream {
    short: usize,
    long: usize,
    ma_period: usize,
    ma_type: String,
    mode: MaMode,

    price: Vec<f64>,
    inv_price: Vec<f64>,
    p_head: usize,
    p_filled: bool,

    roc: Vec<f64>,
    r_head: usize,
    r_filled: bool,

    ma_sum: f64,
    wma_num: f64,
    wma_denom: f64,

    ema_alpha: f64,
    ema_val: f64,
    ema_init: bool,
}

#[inline(always)]
fn parse_mode(s: &str) -> MaMode {
    match s {
        "wma" => MaMode::Wma,
        "sma" => MaMode::Sma,
        "ema" => MaMode::Ema,
        _ => MaMode::Unsupported,
    }
}

#[inline(always)]
fn bump(i: &mut usize, n: usize) {
    *i += 1;
    if *i == n {
        *i = 0;
    }
}

#[inline(always)]
fn wrap_sub(idx: usize, offset: usize, n: usize) -> usize {
    let j = idx + n - offset;
    if j >= n {
        j - n
    } else {
        j
    }
}

#[inline(always)]
fn safe_inv(x: f64) -> f64 {
    if x.is_finite() && x != 0.0 {
        1.0 / x
    } else {
        f64::NAN
    }
}

impl CoppockStream {
    pub fn try_new(params: CoppockParams) -> Result<Self, CoppockError> {
        let short = params.short_roc_period.unwrap_or(11);
        let long = params.long_roc_period.unwrap_or(14);
        let ma_period = params.ma_period.unwrap_or(10);
        let ma_type = params.ma_type.unwrap_or_else(|| "wma".to_string());
        if short == 0 || long == 0 || ma_period == 0 {
            return Err(CoppockError::InvalidPeriod {
                short,
                long,
                ma: ma_period,
                data_len: 0,
            });
        }

        let mode = parse_mode(&ma_type);

        let price_cap = long.max(short) + 1;
        let ma_cap = ma_period;

        Ok(Self {
            short,
            long,
            ma_period,
            ma_type,
            mode,

            price: vec![f64::NAN; price_cap],
            inv_price: vec![f64::NAN; price_cap],
            p_head: 0,
            p_filled: false,

            roc: vec![f64::NAN; ma_cap],
            r_head: 0,
            r_filled: false,

            ma_sum: 0.0,
            wma_num: 0.0,
            wma_denom: (ma_period * (ma_period + 1)) as f64 * 0.5,

            ema_alpha: 2.0 / (ma_period as f64 + 1.0),
            ema_val: f64::NAN,
            ema_init: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let p_n = self.price.len();
        let write_p = self.p_head;

        self.price[write_p] = value;
        self.inv_price[write_p] = safe_inv(value);

        bump(&mut self.p_head, p_n);
        if !self.p_filled && self.p_head == 0 {
            self.p_filled = true;
        }
        if !self.p_filled {
            return None;
        }

        let cur_idx = write_p;
        let prev_s_idx = wrap_sub(cur_idx, self.short, p_n);
        let prev_l_idx = wrap_sub(cur_idx, self.long, p_n);

        let cur = self.price[cur_idx];
        let invs = self.inv_price[prev_s_idx];
        let invl = self.inv_price[prev_l_idx];

        if !(cur.is_finite() && invs.is_finite() && invl.is_finite()) {
            return None;
        }

        let mut sum_roc = (cur * (invs + invl) - 2.0) * 100.0;

        let n = self.ma_period;
        let write_r = self.r_head;
        let old = self.roc[write_r];

        if !self.r_filled {
            self.ma_sum += sum_roc;
            self.wma_num += (write_r as f64 + 1.0) * sum_roc;

            self.roc[write_r] = sum_roc;
            bump(&mut self.r_head, n);

            if !self.r_filled && self.r_head == 0 {
                self.r_filled = true;
            }

            if !self.r_filled {
                return None;
            }

            return Some(match self.mode {
                MaMode::Wma => self.wma_num / self.wma_denom,
                MaMode::Sma => self.ma_sum / n as f64,
                MaMode::Ema => {
                    self.ema_val = sum_roc;
                    self.ema_init = true;
                    self.ema_val
                }
                MaMode::Unsupported => return None,
            });
        }

        let prev_sum = self.ma_sum;
        self.ma_sum = prev_sum - old + sum_roc;

        self.wma_num = self.wma_num + (n as f64) * sum_roc - prev_sum;

        self.roc[write_r] = sum_roc;
        bump(&mut self.r_head, n);

        Some(match self.mode {
            MaMode::Wma => self.wma_num / self.wma_denom,
            MaMode::Sma => self.ma_sum / n as f64,
            MaMode::Ema => {
                if !self.ema_init {
                    self.ema_val = sum_roc;
                    self.ema_init = true;
                } else {
                    self.ema_val = self.ema_alpha * sum_roc + (1.0 - self.ema_alpha) * self.ema_val;
                }
                self.ema_val
            }
            MaMode::Unsupported => return None,
        })
    }
}
#[derive(Clone, Debug)]
pub struct CoppockBatchRange {
    pub short: (usize, usize, usize),
    pub long: (usize, usize, usize),
    pub ma: (usize, usize, usize),
}

impl Default for CoppockBatchRange {
    fn default() -> Self {
        Self {
            short: (11, 11, 0),
            long: (14, 14, 0),
            ma: (10, 259, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CoppockBatchBuilder {
    range: CoppockBatchRange,
    kernel: Kernel,
}

impl CoppockBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short = (start, end, step);
        self
    }
    #[inline]
    pub fn short_static(mut self, n: usize) -> Self {
        self.range.short = (n, n, 0);
        self
    }
    #[inline]
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long = (start, end, step);
        self
    }
    #[inline]
    pub fn long_static(mut self, n: usize) -> Self {
        self.range.long = (n, n, 0);
        self
    }
    #[inline]
    pub fn ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ma = (start, end, step);
        self
    }
    #[inline]
    pub fn ma_static(mut self, n: usize) -> Self {
        self.range.ma = (n, n, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<CoppockBatchOutput, CoppockError> {
        coppock_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CoppockBatchOutput, CoppockError> {
        CoppockBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CoppockBatchOutput, CoppockError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CoppockBatchOutput, CoppockError> {
        CoppockBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn coppock_batch_with_kernel(
    data: &[f64],
    sweep: &CoppockBatchRange,
    k: Kernel,
) -> Result<CoppockBatchOutput, CoppockError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => return Err(CoppockError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    coppock_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CoppockBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CoppockParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CoppockBatchOutput {
    pub fn row_for_params(&self, p: &CoppockParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_roc_period.unwrap_or(11) == p.short_roc_period.unwrap_or(11)
                && c.long_roc_period.unwrap_or(14) == p.long_roc_period.unwrap_or(14)
                && c.ma_period.unwrap_or(10) == p.ma_period.unwrap_or(10)
        })
    }
    pub fn values_for(&self, p: &CoppockParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CoppockBatchRange) -> Result<Vec<CoppockParams>, CoppockError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CoppockError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            loop {
                v.push(cur);
                if cur == end {
                    break;
                }
                cur =
                    cur.checked_add(step)
                        .ok_or(CoppockError::InvalidRange { start, end, step })?;
                if cur > end {
                    break;
                }
            }
        } else {
            let mut cur = start;
            loop {
                v.push(cur);
                if cur == end {
                    break;
                }
                cur =
                    cur.checked_sub(step)
                        .ok_or(CoppockError::InvalidRange { start, end, step })?;
                if cur < end {
                    break;
                }
            }
        }
        if v.is_empty() {
            return Err(CoppockError::InvalidRange { start, end, step });
        }
        Ok(v)
    }
    let shorts = axis_usize(r.short)?;
    let longs = axis_usize(r.long)?;
    let mas = axis_usize(r.ma)?;
    if shorts.is_empty() || longs.is_empty() || mas.is_empty() {
        return Err(CoppockError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    let cap = shorts
        .len()
        .checked_mul(longs.len())
        .and_then(|x| x.checked_mul(mas.len()))
        .ok_or_else(|| CoppockError::InvalidInput("coppock: parameter grid too large".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &s in &shorts {
        for &l in &longs {
            for &m in &mas {
                out.push(CoppockParams {
                    short_roc_period: Some(s),
                    long_roc_period: Some(l),
                    ma_period: Some(m),
                    ma_type: Some("wma".to_string()),
                });
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn coppock_batch_slice(
    data: &[f64],
    sweep: &CoppockBatchRange,
    kern: Kernel,
) -> Result<CoppockBatchOutput, CoppockError> {
    coppock_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn coppock_batch_par_slice(
    data: &[f64],
    sweep: &CoppockBatchRange,
    kern: Kernel,
) -> Result<CoppockBatchOutput, CoppockError> {
    coppock_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn coppock_batch_inner(
    data: &[f64],
    sweep: &CoppockBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CoppockBatchOutput, CoppockError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(CoppockError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CoppockError::AllValuesNaN)?;
    let max_roc = combos
        .iter()
        .map(|c| c.short_roc_period.unwrap().max(c.long_roc_period.unwrap()))
        .max()
        .unwrap();
    let _ = combos.len().checked_mul(max_roc).ok_or_else(|| {
        CoppockError::InvalidInput("coppock: n_combos*max_period overflow".into())
    })?;
    if data.len() - first < max_roc {
        return Err(CoppockError::NotEnoughValidData {
            needed: max_roc,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let _total = rows
        .checked_mul(cols)
        .ok_or_else(|| CoppockError::InvalidInput("rows*cols overflow".into()))?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let short = c.short_roc_period.unwrap();
            let long = c.long_roc_period.unwrap();
            let ma_p = c.ma_period.unwrap();
            let largest = short.max(long);

            first + largest + (ma_p - 1)
        })
        .collect();

    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let inv: Vec<f64> = data.iter().map(|&x| 1.0f64 / x).collect();

    let do_row = |row: usize, out_row: &mut [f64]| {
        let c = &combos[row];
        let short = c.short_roc_period.unwrap();
        let long = c.long_roc_period.unwrap();
        let ma_p = c.ma_period.unwrap();
        let ma_type = c.ma_type.as_deref().unwrap_or("wma");
        let largest = short.max(long);

        let sum_roc_warmup = first + largest;

        let mut sum_roc = alloc_with_nan_prefix(cols, sum_roc_warmup);
        coppock_row_scalar_with_inv(data, first, short, long, &inv, &mut sum_roc);

        let smoothed = ma(&ma_type, MaData::Slice(&sum_roc), ma_p).expect("MA error inside batch");

        out_row.copy_from_slice(&smoothed);
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

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(CoppockBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn coppock_batch_inner_into(
    data: &[f64],
    sweep: &CoppockBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CoppockParams>, CoppockError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(CoppockError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CoppockError::AllValuesNaN)?;
    let max_roc = combos
        .iter()
        .map(|c| c.short_roc_period.unwrap().max(c.long_roc_period.unwrap()))
        .max()
        .unwrap();
    let _ = combos.len().checked_mul(max_roc).ok_or_else(|| {
        CoppockError::InvalidInput("coppock: n_combos*max_period overflow".into())
    })?;
    if data.len() - first < max_roc {
        return Err(CoppockError::NotEnoughValidData {
            needed: max_roc,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| CoppockError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(CoppockError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let short = c.short_roc_period.unwrap();
            let long = c.long_roc_period.unwrap();
            let ma_p = c.ma_period.unwrap();
            let largest = short.max(long);
            first + largest + (ma_p - 1)
        })
        .collect();

    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    let inv: Vec<f64> = data.iter().map(|&x| 1.0f64 / x).collect();

    let do_row = |row: usize, out_row: &mut [f64]| {
        let c = &combos[row];
        let short = c.short_roc_period.unwrap();
        let long = c.long_roc_period.unwrap();
        let ma_p = c.ma_period.unwrap();
        let ma_type = c.ma_type.as_deref().unwrap_or("wma");
        let largest = short.max(long);
        let sum_roc_warmup = first + largest;

        let mut sum_roc = alloc_with_nan_prefix(cols, sum_roc_warmup);

        coppock_row_scalar_with_inv(data, first, short, long, &inv, &mut sum_roc);

        let smoothed = ma(&ma_type, MaData::Slice(&sum_roc), ma_p).expect("MA error inside batch");

        out_row.copy_from_slice(&smoothed);
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

#[inline(always)]
pub fn coppock_row_scalar(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    let largest = short.max(long);
    for i in (first + largest)..data.len() {
        let current = data[i];
        let prev_short = data[i - short];
        let short_val = ((current / prev_short) - 1.0) * 100.0;
        let prev_long = data[i - long];
        let long_val = ((current / prev_long) - 1.0) * 100.0;
        out[i] = short_val + long_val;
    }
}

#[inline(always)]
fn coppock_row_scalar_with_inv(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    inv: &[f64],
    out: &mut [f64],
) {
    let largest = short.max(long);
    for i in (first + largest)..data.len() {
        let c = data[i];
        let is = inv[i - short];
        let il = inv[i - long];
        out[i] = (c * is + c * il - 2.0) * 100.0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn coppock_row_avx2(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    let _ = (stride, w_ptr, inv_n);
    coppock_avx2(data, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn coppock_row_avx512(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    if short.max(long) <= 32 {
        coppock_row_avx512_short(data, first, short, long, stride, w_ptr, inv_n, out)
    } else {
        coppock_row_avx512_long(data, first, short, long, stride, w_ptr, inv_n, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn coppock_row_avx512_short(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    let _ = (stride, w_ptr, inv_n);
    coppock_avx512_short(data, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn coppock_row_avx512_long(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    let _ = (stride, w_ptr, inv_n);
    coppock_avx512_long(data, short, long, first, out)
}

#[inline(always)]
fn expand_grid_coppock(_r: &CoppockBatchRange) -> Vec<CoppockParams> {
    vec![CoppockParams::default()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_coppock_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = CoppockParams::default();
        let input = CoppockInput::from_candles(&candles, "close", default_params);
        let output = coppock_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    #[test]
    fn test_coppock_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.25).collect();

        let input = CoppockInput::from_slice(&data, CoppockParams::default());

        let baseline = coppock(&input)?.values;

        let mut out = vec![0.0; n];
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            coppock_into(&input, &mut out)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            out.copy_from_slice(&baseline);
        }

        assert_eq!(baseline.len(), out.len());

        for i in 0..n {
            let a = baseline[i];
            let b = out[i];
            let eq = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(eq, "mismatch at index {i}: baseline={a}, into={b}");
        }

        Ok(())
    }
    fn check_coppock_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CoppockInput::with_default_candles(&candles);
        let result = coppock_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -1.4542764618985533,
            -1.3795224034983653,
            -1.614331648987457,
            -1.9179048338714915,
            -2.1096548435774625,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-7,
                "[{}] Coppock {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_coppock_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CoppockInput::with_default_candles(&candles);
        match input.data {
            CoppockData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CoppockData::Candles"),
        }
        let output = coppock_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_coppock_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CoppockParams {
            short_roc_period: Some(0),
            long_roc_period: Some(14),
            ma_period: Some(10),
            ma_type: Some("wma".to_string()),
        };
        let input = CoppockInput::from_slice(&input_data, params);
        let res = coppock_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Coppock should fail with zero short period",
            test_name
        );
        Ok(())
    }
    fn check_coppock_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CoppockParams {
            short_roc_period: Some(14),
            long_roc_period: Some(20),
            ma_period: Some(10),
            ma_type: Some("wma".to_string()),
        };
        let input = CoppockInput::from_slice(&data_small, params);
        let res = coppock_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Coppock should fail with short/long>data.len()",
            test_name
        );
        Ok(())
    }
    fn check_coppock_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CoppockParams {
            short_roc_period: Some(11),
            long_roc_period: Some(14),
            ma_period: Some(10),
            ma_type: Some("wma".to_string()),
        };
        let input = CoppockInput::from_slice(&single_point, params);
        let res = coppock_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Coppock should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_coppock_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = CoppockParams::default();
        let first_input = CoppockInput::from_candles(&candles, "close", default_params.clone());
        let first_result = coppock_with_kernel(&first_input, kernel)?;

        let second_params = CoppockParams {
            short_roc_period: Some(5),
            long_roc_period: Some(8),
            ma_period: Some(3),
            ma_type: Some("sma".to_string()),
        };
        let second_input = CoppockInput::from_slice(&first_result.values, second_params.clone());
        let second_result = coppock_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        let short1 = default_params.short_roc_period.unwrap();
        let long1 = default_params.long_roc_period.unwrap();
        let ma1 = default_params.ma_period.unwrap();
        let largest1 = short1.max(long1);
        let first_valid1 = largest1 + (ma1 - 1);

        let short2 = second_params.short_roc_period.unwrap();
        let long2 = second_params.long_roc_period.unwrap();
        let ma2 = second_params.ma_period.unwrap();
        let largest2 = short2.max(long2);
        let first_valid2 = first_valid1 + largest2 + (ma2 - 1);

        for i in first_valid2..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after index {}, found NaN at {}",
                test_name,
                first_valid2,
                i
            );
        }

        Ok(())
    }
    fn check_coppock_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CoppockInput::from_candles(
            &candles,
            "close",
            CoppockParams {
                short_roc_period: Some(11),
                long_roc_period: Some(14),
                ma_period: Some(10),
                ma_type: Some("wma".to_string()),
            },
        );
        let res = coppock_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 30 {
            for (i, &val) in res.values[30..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    30 + i
                );
            }
        }
        Ok(())
    }
    fn check_coppock_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let short = 11;
        let long = 14;
        let ma_period = 10;
        let ma_type = "wma".to_string();
        let input = CoppockInput::from_candles(
            &candles,
            "close",
            CoppockParams {
                short_roc_period: Some(short),
                long_roc_period: Some(long),
                ma_period: Some(ma_period),
                ma_type: Some(ma_type.clone()),
            },
        );
        let batch_output = coppock_with_kernel(&input, Kernel::Scalar)?.values;
        let mut stream = CoppockStream::try_new(CoppockParams {
            short_roc_period: Some(short),
            long_roc_period: Some(long),
            ma_period: Some(ma_period),
            ma_type: Some(ma_type),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(v) => stream_values.push(v),
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
                diff < 1e-8,
                "[{}] Coppock streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_coppock_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let param_combos = vec![
            CoppockParams {
                short_roc_period: Some(11),
                long_roc_period: Some(14),
                ma_period: Some(10),
                ma_type: Some("wma".to_string()),
            },
            CoppockParams {
                short_roc_period: Some(5),
                long_roc_period: Some(8),
                ma_period: Some(3),
                ma_type: Some("sma".to_string()),
            },
            CoppockParams {
                short_roc_period: Some(20),
                long_roc_period: Some(25),
                ma_period: Some(15),
                ma_type: Some("ema".to_string()),
            },
        ];

        for params in param_combos {
            let input = CoppockInput::from_candles(&candles, "close", params);
            let output = coppock_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_coppock_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_coppock_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
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
    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_coppock_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let random_data_strat =
            (2usize..=20, 5usize..=30, 2usize..=15).prop_flat_map(|(short, long, ma_period)| {
                let data_len = long.max(short) + ma_period + 50;
                (
                    prop::collection::vec(
                        (10.0f64..10000.0f64)
                            .prop_filter("positive finite", |x| x.is_finite() && *x > 0.0),
                        data_len..data_len + 100,
                    ),
                    Just(short),
                    Just(long),
                    Just(ma_period),
                    prop::sample::select(vec!["wma", "sma", "ema"]),
                )
            });

        let constant_data_strat =
            (2usize..=15, 5usize..=20, 2usize..=10).prop_flat_map(|(short, long, ma_period)| {
                let data_len = long.max(short) + ma_period + 30;
                (
                    (100.0f64..1000.0f64).prop_map(move |val| vec![val; data_len]),
                    Just(short),
                    Just(long),
                    Just(ma_period),
                    Just("wma"),
                )
            });

        let trending_data_strat =
            (2usize..=15, 5usize..=25, 2usize..=12).prop_flat_map(|(short, long, ma_period)| {
                let data_len = long.max(short) + ma_period + 40;
                (
                    prop::bool::ANY.prop_flat_map(move |increasing| {
                        if increasing {
                            Just(
                                (0..data_len)
                                    .map(|i| 100.0 + i as f64 * 2.0)
                                    .collect::<Vec<_>>(),
                            )
                        } else {
                            Just(
                                (0..data_len)
                                    .map(|i| 1000.0 - i as f64 * 2.0)
                                    .collect::<Vec<_>>(),
                            )
                        }
                    }),
                    Just(short),
                    Just(long),
                    Just(ma_period),
                    Just("sma"),
                )
            });

        let edge_case_strat =
            (2usize..=3, 3usize..=5, 2usize..=3).prop_flat_map(|(short, long, ma_period)| {
                let data_len = 20;
                (
                    prop::collection::vec(
                        (50.0f64..150.0f64).prop_filter("positive", |x| *x > 0.0),
                        data_len..data_len + 10,
                    ),
                    Just(short),
                    Just(long),
                    Just(ma_period),
                    Just("wma"),
                )
            });

        let equal_periods_strat =
            (5usize..=15, 2usize..=10).prop_flat_map(|(period, ma_period)| {
                let data_len = period + ma_period + 30;
                (
                    prop::collection::vec(
                        (50.0f64..500.0f64).prop_filter("positive", |x| *x > 0.0),
                        data_len..data_len + 20,
                    ),
                    Just(period),
                    Just(period),
                    Just(ma_period),
                    Just("wma"),
                )
            });

        let nan_prefix_strat = (2usize..=10, 5usize..=15, 2usize..=8, 1usize..=5).prop_flat_map(
            |(short, long, ma_period, nan_count)| {
                let data_len = nan_count + long.max(short) + ma_period + 20;
                (
                    prop::collection::vec(
                        (100.0f64..1000.0f64),
                        data_len - nan_count..data_len - nan_count + 10,
                    )
                    .prop_map(move |mut vals| {
                        let mut result = vec![f64::NAN; nan_count];
                        result.append(&mut vals);
                        result
                    }),
                    Just(short),
                    Just(long),
                    Just(ma_period),
                    Just("sma"),
                )
            },
        );

        let combined_strat = prop::strategy::Union::new(vec![
            random_data_strat.boxed(),
            constant_data_strat.boxed(),
            trending_data_strat.boxed(),
            edge_case_strat.boxed(),
            equal_periods_strat.boxed(),
            nan_prefix_strat.boxed(),
        ]);

        proptest::test_runner::TestRunner::default()
            .run(
                &combined_strat,
                |(data, short, long, ma_period, ma_type)| {
                    let params = CoppockParams {
                        short_roc_period: Some(short),
                        long_roc_period: Some(long),
                        ma_period: Some(ma_period),
                        ma_type: Some(ma_type.to_string()),
                    };
                    let input = CoppockInput::from_slice(&data, params.clone());

                    let result = coppock_with_kernel(&input, kernel);
                    prop_assert!(
                        result.is_ok(),
                        "Coppock computation failed: {:?}",
                        result.err()
                    );
                    let out = result.unwrap().values;

                    let ref_result = coppock_with_kernel(&input, Kernel::Scalar);
                    prop_assert!(ref_result.is_ok(), "Reference computation failed");
                    let ref_out = ref_result.unwrap().values;

                    let first = data.iter().position(|&x| !x.is_nan()).unwrap_or(0);
                    let largest_roc = short.max(long);
                    let warmup = first + largest_roc + (ma_period - 1);

                    for i in warmup..data.len() {
                        let y = out[i];
                        let r = ref_out[i];

                        if y.is_nan() != r.is_nan() {
                            prop_assert!(
                                false,
                                "NaN mismatch at index {}: kernel={:?}, ref={:?}",
                                i,
                                y,
                                r
                            );
                        }

                        if y.is_finite() && r.is_finite() {
                            let y_bits = y.to_bits();
                            let r_bits = r.to_bits();
                            let ulp_diff = y_bits.abs_diff(r_bits);

                            prop_assert!(
                                (y - r).abs() <= 1e-9 || ulp_diff <= 10,
                                "Value mismatch at index {}: kernel={}, ref={}, diff={}, ULP={}",
                                i,
                                y,
                                r,
                                (y - r).abs(),
                                ulp_diff
                            );
                        }
                    }

                    let is_constant = data.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON);
                    if is_constant && data.len() > warmup + 5 {
                        for i in (warmup + 5)..data.len() {
                            let val = out[i];
                            if val.is_finite() {
                                prop_assert!(
                                    val.abs() <= 1e-6,
                                    "Constant data should produce ~0, got {} at index {}",
                                    val,
                                    i
                                );
                            }
                        }
                    }

                    let is_increasing = data.windows(2).all(|w| w[1] >= w[0]);
                    let is_decreasing = data.windows(2).all(|w| w[1] <= w[0]);

                    if (is_increasing || is_decreasing) && data.len() > warmup + 10 {
                        let expected_positive = is_increasing;

                        for i in (warmup + 10)..(warmup + 15).min(data.len()) {
                            let val = out[i];
                            if val.is_finite() && val.abs() > 1e-10 {
                                if expected_positive {
                                    prop_assert!(
									val >= -1e-6,
									"Expected positive Coppock for increasing data, got {} at index {}",
									val, i
								);
                                } else {
                                    prop_assert!(
									val <= 1e-6,
									"Expected negative Coppock for decreasing data, got {} at index {}",
									val, i
								);
                                }
                            }
                        }
                    }

                    for i in warmup..data.len() {
                        let val = out[i];
                        prop_assert!(
                            val.is_finite() || val.is_nan(),
                            "Found non-finite value {} at index {}",
                            val,
                            i
                        );
                    }

                    for i in warmup..data.len() {
                        let val = out[i];
                        if val.is_finite() {
                            prop_assert!(
							val.abs() <= 100_000.0,
							"Unreasonably large Coppock value {} at index {} (exceeds 100,000%)",
							val, i
						);
                        }
                    }

                    for i in warmup..data.len() {
                        let val = out[i];
                        prop_assert!(
                            val.is_finite() || val.is_nan(),
                            "Found infinity at index {}: {}",
                            i,
                            val
                        );

                        if i >= largest_roc {
                            let current = data[i];
                            let prev_short = data[i - short];
                            let prev_long = data[i - long];

                            if current.is_finite()
                                && prev_short.is_finite()
                                && prev_long.is_finite()
                                && prev_short != 0.0
                                && prev_long != 0.0
                            {
                                if i >= warmup {
                                    prop_assert!(
									val.is_finite(),
									"Expected finite value but got {} at index {} with valid inputs",
									val, i
								);
                                }
                            }
                        }
                    }

                    if short == long && data.len() > warmup + 5 {
                        for i in (warmup + 1)..data.len().min(warmup + 6) {
                            let val = out[i];
                            if val.is_finite() {
                                prop_assert!(
                                    val.abs() <= 100_000.0,
                                    "Equal periods produced unreasonable value {} at index {}",
                                    val,
                                    i
                                );
                            }
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    generate_all_coppock_tests!(
        check_coppock_partial_params,
        check_coppock_accuracy,
        check_coppock_default_candles,
        check_coppock_zero_period,
        check_coppock_period_exceeds_length,
        check_coppock_very_small_dataset,
        check_coppock_reinput,
        check_coppock_nan_handling,
        check_coppock_streaming,
        check_coppock_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_coppock_tests!(check_coppock_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = CoppockBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = CoppockParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            -1.4542764618985533,
            -1.3795224034983653,
            -1.614331648987457,
            -1.9179048338714915,
            -2.1096548435774625,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = CoppockBatchBuilder::new()
            .kernel(kernel)
            .short_range(5, 15, 5)
            .long_range(10, 20, 5)
            .ma_range(3, 9, 3)
            .apply_candles(&c, "close")?;

        for (idx, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            if bits == 0x11111111_11111111 {
                panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }

            if bits == 0x22222222_22222222 {
                panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }

            if bits == 0x33333333_33333333 {
                panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
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
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn coppock_js(
    data: &[f64],
    short_roc: usize,
    long_roc: usize,
    ma_period: usize,
    ma_type: &str,
) -> Result<Vec<f64>, JsValue> {
    let params = CoppockParams {
        short_roc_period: Some(short_roc),
        long_roc_period: Some(long_roc),
        ma_period: Some(ma_period),
        ma_type: Some(ma_type.to_string()),
    };
    let input = CoppockInput::from_slice(data, params);
    let mut out = vec![0.0; data.len()];
    coppock_into_slice(&mut out, &input, detect_best_kernel()).map_err(JsValue::from)?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct CoppockBatchConfig {
    pub short_range: (usize, usize, usize),
    pub long_range: (usize, usize, usize),
    pub ma_range: (usize, usize, usize),
    pub ma_type: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct CoppockBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CoppockParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = coppock_batch)]
pub fn coppock_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: CoppockBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = CoppockBatchRange {
        short: cfg.short_range,
        long: cfg.long_range,
        ma: cfg.ma_range,
    };
    let out =
        coppock_batch_inner(data, &sweep, detect_best_kernel(), false).map_err(JsValue::from)?;
    let js = CoppockBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn coppock_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn coppock_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn coppock_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    short_roc: usize,
    long_roc: usize,
    ma_period: usize,
    ma_type: &str,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }

    if short_roc == 0 || long_roc == 0 || ma_period == 0 {
        return Err(JsValue::from_str("Invalid period"));
    }

    let max_period = short_roc.max(long_roc).max(ma_period);
    if max_period > len {
        return Err(JsValue::from_str("Period exceeds data length"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = CoppockParams {
            short_roc_period: Some(short_roc),
            long_roc_period: Some(long_roc),
            ma_period: Some(ma_period),
            ma_type: Some(ma_type.to_string()),
        };
        let input = CoppockInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            coppock_into_slice(&mut tmp, &input, detect_best_kernel()).map_err(JsValue::from)?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            coppock_into_slice(out, &input, detect_best_kernel()).map_err(JsValue::from)?;
        }
    }
    Ok(())
}

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "coppock")]
#[pyo3(signature = (data, short_roc_period, long_roc_period, ma_period, ma_type=None, kernel=None))]
pub fn coppock_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    short_roc_period: usize,
    long_roc_period: usize,
    ma_period: usize,
    ma_type: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = CoppockParams {
        short_roc_period: Some(short_roc_period),
        long_roc_period: Some(long_roc_period),
        ma_period: Some(ma_period),
        ma_type: ma_type
            .map(|s| s.to_string())
            .or_else(|| Some("wma".to_string())),
    };
    let input = CoppockInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| coppock_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CoppockStream")]
pub struct CoppockStreamPy {
    stream: CoppockStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CoppockStreamPy {
    #[new]
    fn new(
        short_roc_period: usize,
        long_roc_period: usize,
        ma_period: usize,
        ma_type: Option<&str>,
    ) -> PyResult<Self> {
        let params = CoppockParams {
            short_roc_period: Some(short_roc_period),
            long_roc_period: Some(long_roc_period),
            ma_period: Some(ma_period),
            ma_type: ma_type
                .map(|s| s.to_string())
                .or_else(|| Some("wma".to_string())),
        };
        let stream =
            CoppockStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CoppockStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "coppock_batch")]
#[pyo3(signature = (data, short_range, long_range, ma_range, ma_type=None, kernel=None))]
pub fn coppock_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    short_range: (usize, usize, usize),
    long_range: (usize, usize, usize),
    ma_range: (usize, usize, usize),
    ma_type: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = CoppockBatchRange {
        short: short_range,
        long: long_range,
        ma: ma_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
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
                _ => kernel,
            };

            let mut filled_combos = combos.clone();
            if let Some(mt) = ma_type {
                for combo in &mut filled_combos {
                    combo.ma_type = Some(mt.to_string());
                }
            }

            coppock_batch_inner_into(slice_in, &sweep, simd, true, slice_out)?;
            Ok::<Vec<CoppockParams>, CoppockError>(filled_combos)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "shorts",
        combos
            .iter()
            .map(|p| p.short_roc_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "longs",
        combos
            .iter()
            .map(|p| p.long_roc_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_periods",
        combos
            .iter()
            .map(|p| p.ma_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_types",
        combos
            .iter()
            .map(|p| p.ma_type.as_deref().unwrap_or("wma"))
            .collect::<Vec<_>>(),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::coppock_wrapper::{CudaCoppock, DeviceArrayF32Coppock};

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct CoppockDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32Coppock,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl CoppockDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.inner.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
        use cust::memory::DeviceBuffer;

        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }

        if let Some(s) = stream.as_ref() {
            if let Ok(i) = s.extract::<i64>(py) {
                if i == 0 {
                    return Err(PyValueError::new_err(
                        "__dlpack__: stream 0 is disallowed for CUDA",
                    ));
                }
            }
        }

        let dummy =
            DeviceBuffer::from_slice(&[]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx_clone = self.inner.ctx.clone();
        let device_id = self.inner.device_id;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32Coppock {
                buf: dummy,
                rows: 0,
                cols: 0,
                ctx: ctx_clone,
                device_id,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "coppock_cuda_batch_dev")]
#[pyo3(signature = (data, short_range, long_range, ma_range, device_id=0))]
pub fn coppock_cuda_batch_dev_py(
    py: Python<'_>,
    data: numpy::PyReadonlyArray1<'_, f64>,
    short_range: (usize, usize, usize),
    long_range: (usize, usize, usize),
    ma_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<CoppockDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let price = data.as_slice()?;
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let sweep = CoppockBatchRange {
        short: short_range,
        long: long_range,
        ma: ma_range,
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaCoppock::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.coppock_batch_dev(&price_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(CoppockDeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "coppock_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm, cols, rows, short_period, long_period, ma_period, device_id=0))]
pub fn coppock_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm: numpy::PyReadonlyArray1<'_, f64>,
    cols: usize,
    rows: usize,
    short_period: usize,
    long_period: usize,
    ma_period: usize,
    device_id: usize,
) -> PyResult<CoppockDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data_tm.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if slice.len() != expected {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }
    let price_f32: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
    let inner = py.allow_threads(|| {
        let cuda = CudaCoppock::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.coppock_many_series_one_param_time_major_dev(
            &price_f32,
            cols,
            rows,
            short_period,
            long_period,
            ma_period,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(CoppockDeviceArrayF32Py { inner })
}
