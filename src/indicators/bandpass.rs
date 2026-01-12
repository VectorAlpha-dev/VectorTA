#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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
use crate::cuda::bandpass_wrapper::CudaBandpass;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::highpass::{highpass, HighPassError, HighPassInput, HighPassParams};
use crate::utilities::data_loader::{source_type, Candles};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::f64::consts::PI;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
use thiserror::Error;

impl<'a> AsRef<[f64]> for BandPassInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            BandPassData::Slice(slice) => slice,
            BandPassData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BandPassData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct BandPassParams {
    pub period: Option<usize>,
    pub bandwidth: Option<f64>,
}

impl Default for BandPassParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            bandwidth: Some(0.3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BandPassInput<'a> {
    pub data: BandPassData<'a>,
    pub params: BandPassParams,
}

impl<'a> BandPassInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: BandPassParams) -> Self {
        Self {
            data: BandPassData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: BandPassParams) -> Self {
        Self {
            data: BandPassData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", BandPassParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_bandwidth(&self) -> f64 {
        self.params.bandwidth.unwrap_or(0.3)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BandPassBuilder {
    period: Option<usize>,
    bandwidth: Option<f64>,
    kernel: Kernel,
}

impl Default for BandPassBuilder {
    fn default() -> Self {
        Self {
            period: None,
            bandwidth: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BandPassBuilder {
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
    pub fn bandwidth(mut self, b: f64) -> Self {
        self.bandwidth = Some(b);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<BandPassOutput, BandPassError> {
        let p = BandPassParams {
            period: self.period,
            bandwidth: self.bandwidth,
        };
        let i = BandPassInput::from_candles(c, "close", p);
        bandpass_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<BandPassOutput, BandPassError> {
        let p = BandPassParams {
            period: self.period,
            bandwidth: self.bandwidth,
        };
        let i = BandPassInput::from_slice(d, p);
        bandpass_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<BandPassStream, BandPassError> {
        let p = BandPassParams {
            period: self.period,
            bandwidth: self.bandwidth,
        };
        BandPassStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum BandPassError {
    #[error("bandpass: Input data slice is empty.")]
    EmptyInputData,
    #[error("bandpass: All values are NaN.")]
    AllValuesNaN,
    #[error("bandpass: Invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("bandpass: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("bandpass: Invalid bandwidth={bandwidth}")]
    InvalidBandwidth { bandwidth: f64 },
    #[error("bandpass: hp_period too small ({hp_period})")]
    HpPeriodTooSmall { hp_period: usize },
    #[error("bandpass: trigger_period too small ({trigger_period})")]
    TriggerPeriodTooSmall { trigger_period: usize },
    #[error("bandpass: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("bandpass: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("bandpass: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(transparent)]
    HighPassError(#[from] HighPassError),
}

#[derive(Debug, Clone)]
pub struct BandPassOutput {
    pub bp: Vec<f64>,
    pub bp_normalized: Vec<f64>,
    pub signal: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[inline]
pub fn bandpass(input: &BandPassInput) -> Result<BandPassOutput, BandPassError> {
    bandpass_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn bandpass_prepare<'a>(
    input: &'a BandPassInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, f64, usize, usize, Kernel), BandPassError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    let period = input.get_period();
    let bandwidth = input.get_bandwidth();

    if len == 0 {
        return Err(BandPassError::EmptyInputData);
    }
    let first_valid = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(BandPassError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(BandPassError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first_valid < period {
        return Err(BandPassError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid,
        });
    }
    if !(0.0..=1.0).contains(&bandwidth) || !bandwidth.is_finite() {
        return Err(BandPassError::InvalidBandwidth { bandwidth });
    }

    let hp_period = (4.0 * period as f64 / bandwidth).round() as usize;
    if hp_period < 2 {
        return Err(BandPassError::HpPeriodTooSmall { hp_period });
    }
    let trigger_period = ((period as f64 / bandwidth) / 1.5).round() as usize;
    if trigger_period < 2 {
        return Err(BandPassError::TriggerPeriodTooSmall { trigger_period });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((
        data,
        len,
        period,
        bandwidth,
        hp_period,
        trigger_period,
        chosen,
    ))
}

pub fn bandpass_with_kernel(
    input: &BandPassInput,
    kernel: Kernel,
) -> Result<BandPassOutput, BandPassError> {
    let (data, len, period, bandwidth, hp_period, trigger_period, chosen) =
        bandpass_prepare(input, kernel)?;

    let mut hp_params = HighPassParams::default();
    hp_params.period = Some(hp_period);
    let hp = highpass(&HighPassInput::from_slice(data, hp_params))?.values;

    let first_valid_hp = hp.iter().position(|&x| !x.is_nan()).unwrap_or(len);
    let warmup_bp = first_valid_hp.saturating_add(2).max(2).min(len);

    let beta = (2.0 * std::f64::consts::PI / period as f64).cos();
    let gamma = (2.0 * std::f64::consts::PI * bandwidth / period as f64).cos();
    let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

    let mut bp = alloc_with_nan_prefix(len, warmup_bp);

    let bp_start = warmup_bp.saturating_sub(2);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                bandpass_scalar(&hp[bp_start..], period, alpha, beta, &mut bp[bp_start..])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                bandpass_avx2(&hp[bp_start..], period, alpha, beta, &mut bp[bp_start..])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                bandpass_avx512(&hp[bp_start..], period, alpha, beta, &mut bp[bp_start..])
            }
            _ => unreachable!(),
        }
    }

    for v in &mut bp[..warmup_bp] {
        *v = f64::NAN;
    }

    let mut bp_normalized = alloc_with_nan_prefix(len, warmup_bp);

    let k = 0.991;
    let mut peak_prev = 0.0f64;
    for i in warmup_bp..len {
        peak_prev *= k;
        let abs_bp = bp[i].abs();
        if abs_bp > peak_prev {
            peak_prev = abs_bp;
        }
        bp_normalized[i] = if peak_prev != 0.0 {
            bp[i] / peak_prev
        } else {
            0.0
        };
    }

    let mut trigger = alloc_with_nan_prefix(len, warmup_bp);
    if warmup_bp < len {
        let mut trigger_params = HighPassParams::default();
        trigger_params.period = Some(trigger_period);
        let trig_inp = HighPassInput::from_slice(&bp_normalized[warmup_bp..], trigger_params);
        crate::indicators::moving_averages::highpass::highpass_into_slice(
            &mut trigger[warmup_bp..],
            &trig_inp,
            chosen,
        )?;
    }

    let first_trig = trigger.iter().position(|x| !x.is_nan()).unwrap_or(len);
    let warm_sig = warmup_bp.max(first_trig);
    let mut signal = alloc_with_nan_prefix(len, warm_sig);
    for i in warm_sig..len {
        let bn = bp_normalized[i];
        let tr = trigger[i];
        signal[i] = if bn < tr {
            1.0
        } else if bn > tr {
            -1.0
        } else {
            0.0
        };
    }

    Ok(BandPassOutput {
        bp,
        bp_normalized,
        signal,
        trigger,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn bandpass_into(
    input: &BandPassInput,
    bp_out: &mut [f64],
    bp_normalized_out: &mut [f64],
    signal_out: &mut [f64],
    trigger_out: &mut [f64],
) -> Result<(), BandPassError> {
    bandpass_into_slice(
        bp_out,
        bp_normalized_out,
        signal_out,
        trigger_out,
        input,
        Kernel::Auto,
    )
}

#[inline]
pub fn bandpass_into_slice(
    bp_dst: &mut [f64],
    bpn_dst: &mut [f64],
    sig_dst: &mut [f64],
    trig_dst: &mut [f64],
    input: &BandPassInput,
    kernel: Kernel,
) -> Result<(), BandPassError> {
    let (data, len, period, bandwidth, hp_period, trigger_period, chosen) =
        bandpass_prepare(input, kernel)?;
    if bp_dst.len() != len || bpn_dst.len() != len || sig_dst.len() != len || trig_dst.len() != len
    {
        return Err(BandPassError::OutputLengthMismatch {
            expected: len,
            got: *[bp_dst.len(), bpn_dst.len(), sig_dst.len(), trig_dst.len()]
                .iter()
                .min()
                .unwrap_or(&0),
        });
    }

    let mut hp_params = HighPassParams::default();
    hp_params.period = Some(hp_period);
    let hp = highpass(&HighPassInput::from_slice(data, hp_params))?.values;

    let first_valid_hp = hp.iter().position(|&x| !x.is_nan()).unwrap_or(len);
    let warm_bp = first_valid_hp.saturating_add(2).max(2).min(len);

    let beta = (2.0 * std::f64::consts::PI / period as f64).cos();
    let gamma = (2.0 * std::f64::consts::PI * bandwidth / period as f64).cos();
    let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

    let bp_start = warm_bp.saturating_sub(2);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => bandpass_scalar(
                &hp[bp_start..],
                period,
                alpha,
                beta,
                &mut bp_dst[bp_start..],
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => bandpass_avx2(
                &hp[bp_start..],
                period,
                alpha,
                beta,
                &mut bp_dst[bp_start..],
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => bandpass_avx512(
                &hp[bp_start..],
                period,
                alpha,
                beta,
                &mut bp_dst[bp_start..],
            ),
            _ => unreachable!(),
        }
    }

    for v in &mut bp_dst[..warm_bp] {
        *v = f64::NAN;
    }

    for v in &mut bpn_dst[..warm_bp] {
        *v = f64::NAN;
    }
    let k = 0.991;
    let mut peak = 0.0f64;
    for i in warm_bp..len {
        peak *= k;
        let v = bp_dst[i];
        let av = v.abs();
        if av > peak {
            peak = av;
        }
        bpn_dst[i] = if peak != 0.0 { v / peak } else { 0.0 };
    }

    for v in trig_dst.iter_mut() {
        *v = f64::NAN;
    }
    if warm_bp < len {
        let mut trigger_params = HighPassParams::default();
        trigger_params.period = Some(trigger_period);
        let trig_inp = HighPassInput::from_slice(&bpn_dst[warm_bp..], trigger_params);

        crate::indicators::moving_averages::highpass::highpass_into_slice(
            &mut trig_dst[warm_bp..],
            &trig_inp,
            chosen,
        )?;
    }

    let first_trig = trig_dst.iter().position(|x| !x.is_nan()).unwrap_or(len);
    let warm_sig = warm_bp.max(first_trig);
    for v in &mut sig_dst[..warm_sig] {
        *v = f64::NAN;
    }
    for i in warm_sig..len {
        let bn = bpn_dst[i];
        let tr = trig_dst[i];
        sig_dst[i] = if bn < tr {
            1.0
        } else if bn > tr {
            -1.0
        } else {
            0.0
        };
    }

    Ok(())
}

#[inline(always)]
pub fn bandpass_scalar(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    let len = hp.len();
    if len == 0 {
        return;
    }

    out[0] = hp[0];
    if len == 1 {
        return;
    }
    out[1] = hp[1];
    if len == 2 {
        return;
    }

    let a = 0.5 * (1.0 - alpha);
    let c = beta * (1.0 + alpha);
    let d = -alpha;

    let mut y_im2 = out[0];
    let mut y_im1 = out[1];

    let mut i = 2usize;
    while i + 3 < len {
        let delta0 = hp[i] - hp[i - 2];
        let y0 = d.mul_add(y_im2, c.mul_add(y_im1, a * delta0));
        out[i] = y0;

        let delta1 = hp[i + 1] - hp[i - 1];
        let y1 = d.mul_add(y_im1, c.mul_add(y0, a * delta1));
        out[i + 1] = y1;

        let delta2 = hp[i + 2] - hp[i];
        let y2 = d.mul_add(y0, c.mul_add(y1, a * delta2));
        out[i + 2] = y2;

        let delta3 = hp[i + 3] - hp[i + 1];
        let y3 = d.mul_add(y1, c.mul_add(y2, a * delta3));
        out[i + 3] = y3;

        y_im2 = y2;
        y_im1 = y3;
        i += 4;
    }

    while i + 1 < len {
        let delta0 = hp[i] - hp[i - 2];
        let y0 = d.mul_add(y_im2, c.mul_add(y_im1, a * delta0));
        out[i] = y0;

        let delta1 = hp[i + 1] - hp[i - 1];
        let y1 = d.mul_add(y_im1, c.mul_add(y0, a * delta1));
        out[i + 1] = y1;

        y_im2 = y0;
        y_im1 = y1;
        i += 2;
    }

    if i < len {
        let delta = hp[i] - hp[i - 2];
        out[i] = d.mul_add(y_im2, c.mul_add(y_im1, a * delta));
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx2(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    unsafe { bandpass_scalar_unchecked(hp, alpha, beta, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    unsafe { bandpass_scalar_unchecked(hp, alpha, beta, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512_short(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    unsafe { bandpass_scalar_unchecked(hp, alpha, beta, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512_long(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    unsafe { bandpass_scalar_unchecked(hp, alpha, beta, out) }
}

#[derive(Debug, Clone)]
pub struct BandPassStream {
    period: usize,

    c0: f64,
    c1: f64,
    c2: f64,

    hp_stream: crate::indicators::highpass::HighPassStream,

    hp_z1: f64,
    hp_z2: f64,
    y_z1: f64,
    y_z2: f64,

    hp_valid: u8,
}

impl BandPassStream {
    #[inline]
    pub fn try_new(params: BandPassParams) -> Result<Self, BandPassError> {
        let period = params.period.unwrap_or(20);
        if period < 2 {
            return Err(BandPassError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let bandwidth = params.bandwidth.unwrap_or(0.3);
        if !(0.0..=1.0).contains(&bandwidth) || !bandwidth.is_finite() {
            return Err(BandPassError::InvalidBandwidth { bandwidth });
        }

        let hp_period = (4.0 * period as f64 / bandwidth).round() as usize;
        if hp_period < 2 {
            return Err(BandPassError::HpPeriodTooSmall { hp_period });
        }

        let mut hp_params = HighPassParams::default();
        hp_params.period = Some(hp_period);
        let hp_stream = crate::indicators::highpass::HighPassStream::try_new(hp_params)?;

        use std::f64::consts::PI;
        let beta = (2.0 * PI / period as f64).cos();
        let gamma = (2.0 * PI * bandwidth / period as f64).cos();

        #[inline(always)]
        fn alpha_from_gamma(gamma: f64) -> f64 {
            let g2 = gamma * gamma;
            let s = (1.0 - g2).sqrt();
            (1.0 - s) / gamma
        }
        let alpha = alpha_from_gamma(gamma);

        let c0 = 0.5 * (1.0 - alpha);
        let c1 = beta * (1.0 + alpha);
        let c2 = -alpha;

        Ok(Self {
            period,
            c0,
            c1,
            c2,
            hp_stream,
            hp_z1: 0.0,
            hp_z2: 0.0,
            y_z1: 0.0,
            y_z2: 0.0,
            hp_valid: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        let hp = self.hp_stream.update(value);

        if !hp.is_finite() {
            return f64::NAN;
        }

        if self.hp_valid < 2 {
            let y = hp;

            self.hp_z2 = self.hp_z1;
            self.hp_z1 = hp;

            self.y_z2 = self.y_z1;
            self.y_z1 = y;

            self.hp_valid += 1;
            return f64::NAN;
        }

        let delta = hp - self.hp_z2;
        let y = self
            .c2
            .mul_add(self.y_z2, self.c1.mul_add(self.y_z1, self.c0 * delta));

        self.hp_z2 = self.hp_z1;
        self.hp_z1 = hp;

        self.y_z2 = self.y_z1;
        self.y_z1 = y;

        y
    }
}

#[derive(Clone, Debug)]
pub struct BandPassBatchRange {
    pub period: (usize, usize, usize),
    pub bandwidth: (f64, f64, f64),
}

impl Default for BandPassBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 20, 0),
            bandwidth: (0.3, 0.549, 0.001),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BandPassBatchBuilder {
    range: BandPassBatchRange,
    kernel: Kernel,
}

impl BandPassBatchBuilder {
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
    #[inline]
    pub fn bandwidth_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.bandwidth = (start, end, step);
        self
    }
    #[inline]
    pub fn bandwidth_static(mut self, b: f64) -> Self {
        self.range.bandwidth = (b, b, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<BandPassBatchOutput, BandPassError> {
        bandpass_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<BandPassBatchOutput, BandPassError> {
        BandPassBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<BandPassBatchOutput, BandPassError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<BandPassBatchOutput, BandPassError> {
        BandPassBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct BandPassBatchOutput {
    pub bp: Vec<f64>,
    pub bp_normalized: Vec<f64>,
    pub signal: Vec<f64>,
    pub trigger: Vec<f64>,
    pub combos: Vec<BandPassParams>,
    pub rows: usize,
    pub cols: usize,
}

impl BandPassBatchOutput {
    #[inline]
    pub fn row_for_params(&self, p: &BandPassParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.bandwidth.unwrap_or(0.3) - p.bandwidth.unwrap_or(0.3)).abs() < 1e-12
        })
    }
    #[inline]
    pub fn row_slices(&self, row: usize) -> Option<(&[f64], &[f64], &[f64], &[f64])> {
        if row >= self.rows {
            return None;
        }
        let (r, c) = (row, self.cols);
        Some((
            &self.bp[r * self.cols..r * self.cols + c],
            &self.bp_normalized[r * self.cols..r * self.cols + c],
            &self.signal[r * self.cols..r * self.cols + c],
            &self.trigger[r * self.cols..r * self.cols + c],
        ))
    }
}

#[inline(always)]
fn expand_grid(r: &BandPassBatchRange) -> Vec<BandPassParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, BandPassError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                vals.push(v);
                match v.checked_add(step) {
                    Some(next) => {
                        if next == v {
                            break;
                        }
                        v = next;
                    }
                    None => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                vals.push(v);
                if v == end {
                    break;
                }
                let next = v.saturating_sub(step);
                if next == v {
                    break;
                }
                v = next;
                if v < end {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(BandPassError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, BandPassError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start <= end {
            let mut x = start;
            loop {
                vals.push(x);
                if x >= end {
                    break;
                }
                let next = x + step;
                if !next.is_finite() || next == x {
                    break;
                }
                x = next;
                if x > end + 1e-12 {
                    break;
                }
            }
        } else {
            let mut x = start;
            loop {
                vals.push(x);
                if x <= end {
                    break;
                }
                let next = x - step.abs();
                if !next.is_finite() || next == x {
                    break;
                }
                x = next;
                if x < end - 1e-12 {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(BandPassError::InvalidRange {
                start: start as usize,
                end: end as usize,
                step: step.abs() as usize,
            });
        }
        Ok(vals)
    }
    let periods = match axis_usize(r.period) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let bandwidths = match axis_f64(r.bandwidth) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::with_capacity(periods.len() * bandwidths.len());
    for &p in &periods {
        for &b in &bandwidths {
            out.push(BandPassParams {
                period: Some(p),
                bandwidth: Some(b),
            });
        }
    }
    out
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(
    module = "ta_indicators.cuda",
    name = "BandPassDeviceArrayF32",
    unsendable
)]
pub struct BandPassDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
    pub(crate) stream: usize,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl BandPassDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = &self.inner;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (inner.device_ptr() as usize, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        Ok((2, self.device_id as i32))
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
        let (kdl, alloc_dev) = self.__dlpack_device__()?;
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
        let _ = stream;

        let dummy =
            DeviceBuffer::from_slice(&[]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

pub fn bandpass_batch_with_kernel(
    data: &[f64],
    sweep: &BandPassBatchRange,
    k: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(BandPassError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    bandpass_batch_par_slice(data, sweep, simd)
}

pub fn bandpass_batch_slice(
    data: &[f64],
    sweep: &BandPassBatchRange,
    kern: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
    bandpass_batch_inner(data, sweep, kern, false)
}

pub fn bandpass_batch_par_slice(
    data: &[f64],
    sweep: &BandPassBatchRange,
    kern: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
    bandpass_batch_inner(data, sweep, kern, true)
}

fn bandpass_batch_inner(
    data: &[f64],
    sweep: &BandPassBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<BandPassBatchOutput, BandPassError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(BandPassError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let cols = data.len();
    if cols == 0 {
        return Err(BandPassError::EmptyInputData);
    }
    let rows = combos.len();
    rows.checked_mul(cols).ok_or(BandPassError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;

    let mut bp_mu = make_uninit_matrix(rows, cols);
    let mut bpn_mu = make_uninit_matrix(rows, cols);
    let mut sig_mu = make_uninit_matrix(rows, cols);
    let mut trg_mu = make_uninit_matrix(rows, cols);

    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let warms_bp: Vec<usize> = combos
        .iter()
        .map(|p| {
            let period = p.period.unwrap();
            let bandwidth = p.bandwidth.unwrap();
            let hp_p = (4.0 * period as f64 / bandwidth).round() as usize;
            let warm_hp = first + hp_p - 1;
            warm_hp.max(2)
        })
        .collect();
    let warms_trg: Vec<usize> = combos
        .iter()
        .map(|p| {
            let period = p.period.unwrap();
            let bandwidth = p.bandwidth.unwrap();
            let hp_p = (4.0 * period as f64 / bandwidth).round() as usize;
            let trig_p = ((period as f64 / bandwidth) / 1.5).round() as usize;
            let warm_hp = first + hp_p - 1;
            let warm_bp = warm_hp.max(2);
            warm_bp + trig_p - 1
        })
        .collect();

    init_matrix_prefixes(&mut bp_mu, cols, &warms_bp);
    init_matrix_prefixes(&mut bpn_mu, cols, &warms_bp);
    init_matrix_prefixes(&mut trg_mu, cols, &warms_trg);
    init_matrix_prefixes(&mut sig_mu, cols, &warms_trg);

    let mut bp_guard = core::mem::ManuallyDrop::new(bp_mu);
    let mut bpn_guard = core::mem::ManuallyDrop::new(bpn_mu);
    let mut sig_guard = core::mem::ManuallyDrop::new(sig_mu);
    let mut trg_guard = core::mem::ManuallyDrop::new(trg_mu);

    let bp_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(bp_guard.as_mut_ptr() as *mut f64, bp_guard.len())
    };
    let bpn_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(bpn_guard.as_mut_ptr() as *mut f64, bpn_guard.len())
    };
    let sig_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(sig_guard.as_mut_ptr() as *mut f64, sig_guard.len())
    };
    let trg_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(trg_guard.as_mut_ptr() as *mut f64, trg_guard.len())
    };

    bandpass_batch_inner_into(
        data, &combos, kern, parallel, bp_out, bpn_out, sig_out, trg_out,
    )?;

    let bp = unsafe {
        Vec::from_raw_parts(
            bp_guard.as_mut_ptr() as *mut f64,
            bp_guard.len(),
            bp_guard.capacity(),
        )
    };
    let bpn = unsafe {
        Vec::from_raw_parts(
            bpn_guard.as_mut_ptr() as *mut f64,
            bpn_guard.len(),
            bpn_guard.capacity(),
        )
    };
    let sig = unsafe {
        Vec::from_raw_parts(
            sig_guard.as_mut_ptr() as *mut f64,
            sig_guard.len(),
            sig_guard.capacity(),
        )
    };
    let trg = unsafe {
        Vec::from_raw_parts(
            trg_guard.as_mut_ptr() as *mut f64,
            trg_guard.len(),
            trg_guard.capacity(),
        )
    };

    Ok(BandPassBatchOutput {
        bp,
        bp_normalized: bpn,
        signal: sig,
        trigger: trg,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn bandpass_batch_inner_into(
    data: &[f64],
    combos: &[BandPassParams],
    kern: Kernel,
    parallel: bool,
    bp_out: &mut [f64],
    bpn_out: &mut [f64],
    sig_out: &mut [f64],
    trg_out: &mut [f64],
) -> Result<(), BandPassError> {
    let rows = combos.len();
    let cols = data.len();
    let chosen = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match chosen {
        Kernel::ScalarBatch => Kernel::Scalar,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        _ => Kernel::Scalar,
    };

    struct RowMeta {
        period: usize,
        bandwidth: f64,
        hp_p: usize,
        trig_p: usize,
        ksel: Kernel,
    }

    let mut metas = Vec::with_capacity(rows);
    for &p in combos.iter() {
        let period = p.period.unwrap();
        let bandwidth = p.bandwidth.unwrap();
        let (_d, _len, _per, _bw, hp_p, trig_p, ksel) =
            bandpass_prepare(&BandPassInput::from_slice(data, p), simd)?;
        metas.push(RowMeta {
            period,
            bandwidth,
            hp_p,
            trig_p,
            ksel,
        });
    }

    use std::collections::HashMap;
    let mut hp_cache: HashMap<usize, Vec<f64>> = HashMap::new();
    for meta in metas.iter() {
        if !hp_cache.contains_key(&meta.hp_p) {
            let mut hp_params = HighPassParams::default();
            hp_params.period = Some(meta.hp_p);
            let hp = highpass(&HighPassInput::from_slice(data, hp_params))?.values;
            hp_cache.insert(meta.hp_p, hp);
        }
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicPtr, Ordering};

            let bp_ptr = AtomicPtr::new(bp_out.as_mut_ptr());
            let bpn_ptr = AtomicPtr::new(bpn_out.as_mut_ptr());
            let sig_ptr = AtomicPtr::new(sig_out.as_mut_ptr());
            let trg_ptr = AtomicPtr::new(trg_out.as_mut_ptr());

            (0..rows)
                .into_par_iter()
                .try_for_each(|row| -> Result<(), BandPassError> {
                    let m = &metas[row];
                    let period = m.period;
                    let bandwidth = m.bandwidth;

                    let bp_r = unsafe {
                        std::slice::from_raw_parts_mut(
                            bp_ptr.load(Ordering::Relaxed).add(row * cols),
                            cols,
                        )
                    };
                    let bpn_r = unsafe {
                        std::slice::from_raw_parts_mut(
                            bpn_ptr.load(Ordering::Relaxed).add(row * cols),
                            cols,
                        )
                    };
                    let sig_r = unsafe {
                        std::slice::from_raw_parts_mut(
                            sig_ptr.load(Ordering::Relaxed).add(row * cols),
                            cols,
                        )
                    };
                    let trg_r = unsafe {
                        std::slice::from_raw_parts_mut(
                            trg_ptr.load(Ordering::Relaxed).add(row * cols),
                            cols,
                        )
                    };
                    let trig_p = m.trig_p;
                    let ksel = m.ksel;
                    let hp = hp_cache.get(&m.hp_p).expect("hp cache missing");

                    let first_valid_hp = hp.iter().position(|&x| !x.is_nan()).unwrap_or(cols);
                    let warm_bp = first_valid_hp.saturating_add(2).max(2).min(cols);
                    let bp_start = warm_bp.saturating_sub(2);

                    let beta = (2.0 * std::f64::consts::PI / period as f64).cos();
                    let gamma = (2.0 * std::f64::consts::PI * bandwidth / period as f64).cos();
                    let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

                    unsafe {
                        match ksel {
                            Kernel::Scalar => bandpass_scalar(
                                &hp[bp_start..],
                                period,
                                alpha,
                                beta,
                                &mut bp_r[bp_start..],
                            ),
                            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                            Kernel::Avx2 => bandpass_avx2(
                                &hp[bp_start..],
                                period,
                                alpha,
                                beta,
                                &mut bp_r[bp_start..],
                            ),
                            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                            Kernel::Avx512 => bandpass_avx512(
                                &hp[bp_start..],
                                period,
                                alpha,
                                beta,
                                &mut bp_r[bp_start..],
                            ),
                            _ => unreachable!(),
                        }
                    }

                    for v in &mut bp_r[..warm_bp] {
                        *v = f64::NAN;
                    }

                    let k = 0.991;
                    let mut peak = 0.0f64;
                    for i in warm_bp..cols {
                        peak *= k;
                        let v = bp_r[i];
                        let av = v.abs();
                        if av > peak {
                            peak = av;
                        }
                        bpn_r[i] = if peak != 0.0 { v / peak } else { 0.0 };
                    }

                    for v in trg_r.iter_mut() {
                        *v = f64::NAN;
                    }
                    if warm_bp < cols {
                        let mut trig_params = HighPassParams::default();
                        trig_params.period = Some(trig_p);
                        let trig_inp = HighPassInput::from_slice(&bpn_r[warm_bp..], trig_params);
                        crate::indicators::moving_averages::highpass::highpass_into_slice(
                            &mut trg_r[warm_bp..],
                            &trig_inp,
                            ksel,
                        )?;
                    }

                    let first_tr = trg_r.iter().position(|x| !x.is_nan()).unwrap_or(cols);
                    let warm_sig = warm_bp.max(first_tr);
                    for v in &mut bpn_r[..warm_bp] {
                        if !v.is_nan() {
                            *v = f64::NAN;
                        }
                    }
                    for v in &mut sig_r[..warm_sig] {
                        *v = f64::NAN;
                    }
                    for i in warm_sig..cols {
                        let bn = bpn_r[i];
                        let tr = trg_r[i];
                        sig_r[i] = if bn < tr {
                            1.0
                        } else if bn > tr {
                            -1.0
                        } else {
                            0.0
                        };
                    }

                    Ok(())
                })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let m = &metas[row];
                let period = m.period;
                let bandwidth = m.bandwidth;

                let bp_r = &mut bp_out[row * cols..(row + 1) * cols];
                let bpn_r = &mut bpn_out[row * cols..(row + 1) * cols];
                let sig_r = &mut sig_out[row * cols..(row + 1) * cols];
                let trg_r = &mut trg_out[row * cols..(row + 1) * cols];
                let trig_p = m.trig_p;
                let ksel = m.ksel;
                let hp = hp_cache.get(&m.hp_p).expect("hp cache missing");

                let first_valid_hp = hp.iter().position(|&x| !x.is_nan()).unwrap_or(cols);
                let warm_bp = first_valid_hp.saturating_add(2).max(2).min(cols);
                let bp_start = warm_bp.saturating_sub(2);

                let beta = (2.0 * std::f64::consts::PI / period as f64).cos();
                let gamma = (2.0 * std::f64::consts::PI * bandwidth / period as f64).cos();
                let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

                unsafe {
                    match ksel {
                        Kernel::Scalar => bandpass_scalar(
                            &hp[bp_start..],
                            period,
                            alpha,
                            beta,
                            &mut bp_r[bp_start..],
                        ),
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        Kernel::Avx2 => bandpass_avx2(
                            &hp[bp_start..],
                            period,
                            alpha,
                            beta,
                            &mut bp_r[bp_start..],
                        ),
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        Kernel::Avx512 => bandpass_avx512(
                            &hp[bp_start..],
                            period,
                            alpha,
                            beta,
                            &mut bp_r[bp_start..],
                        ),
                        _ => unreachable!(),
                    }
                }

                for v in &mut bp_r[..warm_bp] {
                    *v = f64::NAN;
                }

                let k = 0.991;
                let mut peak = 0.0f64;
                for i in warm_bp..cols {
                    peak *= k;
                    let v = bp_r[i];
                    let av = v.abs();
                    if av > peak {
                        peak = av;
                    }
                    bpn_r[i] = if peak != 0.0 { v / peak } else { 0.0 };
                }

                for v in trg_r.iter_mut() {
                    *v = f64::NAN;
                }
                if warm_bp < cols {
                    let mut trig_params = HighPassParams::default();
                    trig_params.period = Some(trig_p);
                    let trig_inp = HighPassInput::from_slice(&bpn_r[warm_bp..], trig_params);
                    crate::indicators::moving_averages::highpass::highpass_into_slice(
                        &mut trg_r[warm_bp..],
                        &trig_inp,
                        ksel,
                    )?;
                }

                let first_tr = trg_r.iter().position(|x| !x.is_nan()).unwrap_or(cols);
                let warm_sig = warm_bp.max(first_tr);
                for v in &mut bpn_r[..warm_bp] {
                    if !v.is_nan() {
                        *v = f64::NAN;
                    }
                }
                for v in &mut sig_r[..warm_sig] {
                    *v = f64::NAN;
                }
                for i in warm_sig..cols {
                    let bn = bpn_r[i];
                    let tr = trg_r[i];
                    sig_r[i] = if bn < tr {
                        1.0
                    } else if bn > tr {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
        }
    } else {
        for row in 0..rows {
            let p = combos[row];
            let period = p.period.unwrap();
            let bandwidth = p.bandwidth.unwrap();

            let bp_r = &mut bp_out[row * cols..(row + 1) * cols];
            let bpn_r = &mut bpn_out[row * cols..(row + 1) * cols];
            let sig_r = &mut sig_out[row * cols..(row + 1) * cols];
            let trg_r = &mut trg_out[row * cols..(row + 1) * cols];

            let (_d, _len, _per, _bw, hp_p, trig_p, ksel) =
                bandpass_prepare(&BandPassInput::from_slice(data, p), simd)?;

            let mut hp_params = HighPassParams::default();
            hp_params.period = Some(hp_p);
            let hp = highpass(&HighPassInput::from_slice(data, hp_params))?.values;

            let first_valid_hp = hp.iter().position(|&x| !x.is_nan()).unwrap_or(cols);
            let warm_bp = first_valid_hp.saturating_add(2).max(2).min(cols);
            let bp_start = warm_bp.saturating_sub(2);

            let beta = (2.0 * std::f64::consts::PI / period as f64).cos();
            let gamma = (2.0 * std::f64::consts::PI * bandwidth / period as f64).cos();
            let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

            unsafe {
                match ksel {
                    Kernel::Scalar => {
                        bandpass_scalar(&hp[bp_start..], period, alpha, beta, &mut bp_r[bp_start..])
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 => {
                        bandpass_avx2(&hp[bp_start..], period, alpha, beta, &mut bp_r[bp_start..])
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 => {
                        bandpass_avx512(&hp[bp_start..], period, alpha, beta, &mut bp_r[bp_start..])
                    }
                    _ => unreachable!(),
                }
            }

            for v in &mut bp_r[..warm_bp] {
                *v = f64::NAN;
            }

            let k = 0.991;
            let mut peak = 0.0f64;
            for i in warm_bp..cols {
                peak *= k;
                let v = bp_r[i];
                let av = v.abs();
                if av > peak {
                    peak = av;
                }
                bpn_r[i] = if peak != 0.0 { v / peak } else { 0.0 };
            }

            for v in trg_r.iter_mut() {
                *v = f64::NAN;
            }
            if warm_bp < cols {
                let mut trig_params = HighPassParams::default();
                trig_params.period = Some(trig_p);
                let trig_vec =
                    highpass(&HighPassInput::from_slice(&bpn_r[warm_bp..], trig_params))?.values;
                trg_r[warm_bp..].copy_from_slice(&trig_vec);
            }

            let first_tr = trg_r.iter().position(|x| !x.is_nan()).unwrap_or(cols);
            let warm_sig = warm_bp.max(first_tr);
            for v in &mut bpn_r[..warm_bp] {
                if !v.is_nan() {
                    *v = f64::NAN;
                }
            }
            for v in &mut sig_r[..warm_sig] {
                *v = f64::NAN;
            }
            for i in warm_sig..cols {
                let bn = bpn_r[i];
                let tr = trg_r[i];
                sig_r[i] = if bn < tr {
                    1.0
                } else if bn > tr {
                    -1.0
                } else {
                    0.0
                };
            }
        }
    }

    Ok(())
}

#[inline(always)]
pub fn bandpass_row_scalar(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx2(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    unsafe { bandpass_scalar_unchecked(hp, alpha, beta, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    unsafe { bandpass_scalar_unchecked(hp, alpha, beta, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512_short(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    unsafe { bandpass_scalar_unchecked(hp, alpha, beta, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512_long(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
    unsafe { bandpass_scalar_unchecked(hp, alpha, beta, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn bandpass_scalar_unchecked(hp: &[f64], alpha: f64, beta: f64, out: &mut [f64]) {
    debug_assert!(out.len() >= hp.len());
    let len = hp.len();
    if len == 0 {
        return;
    }

    let out_ptr = out.as_mut_ptr();
    let hp_ptr = hp.as_ptr();

    *out_ptr.add(0) = *hp_ptr.add(0);
    if len == 1 {
        return;
    }
    *out_ptr.add(1) = *hp_ptr.add(1);
    if len == 2 {
        return;
    }

    let a = 0.5 * (1.0 - alpha);
    let c = beta * (1.0 + alpha);
    let d = -alpha;

    let mut y_im2 = *out_ptr.add(0);
    let mut y_im1 = *out_ptr.add(1);

    let mut i = 2usize;
    while i + 3 < len {
        let delta0 = *hp_ptr.add(i) - *hp_ptr.add(i - 2);
        let y0 = d.mul_add(y_im2, c.mul_add(y_im1, a * delta0));
        *out_ptr.add(i) = y0;

        let delta1 = *hp_ptr.add(i + 1) - *hp_ptr.add(i - 1);
        let y1 = d.mul_add(y_im1, c.mul_add(y0, a * delta1));
        *out_ptr.add(i + 1) = y1;

        let delta2 = *hp_ptr.add(i + 2) - *hp_ptr.add(i);
        let y2 = d.mul_add(y0, c.mul_add(y1, a * delta2));
        *out_ptr.add(i + 2) = y2;

        let delta3 = *hp_ptr.add(i + 3) - *hp_ptr.add(i + 1);
        let y3 = d.mul_add(y1, c.mul_add(y2, a * delta3));
        *out_ptr.add(i + 3) = y3;

        y_im2 = y2;
        y_im1 = y3;
        i += 4;
    }

    while i + 1 < len {
        let delta0 = *hp_ptr.add(i) - *hp_ptr.add(i - 2);
        let y0 = d.mul_add(y_im2, c.mul_add(y_im1, a * delta0));
        *out_ptr.add(i) = y0;

        let delta1 = *hp_ptr.add(i + 1) - *hp_ptr.add(i - 1);
        let y1 = d.mul_add(y_im1, c.mul_add(y0, a * delta1));
        *out_ptr.add(i + 1) = y1;

        y_im2 = y0;
        y_im1 = y1;
        i += 2;
    }

    if i < len {
        let delta = *hp_ptr.add(i) - *hp_ptr.add(i - 2);
        let y = d.mul_add(y_im2, c.mul_add(y_im1, a * delta));
        *out_ptr.add(i) = y;
    }
}

#[inline(always)]
fn expand_grid_for_bandpass(r: &BandPassBatchRange) -> Vec<BandPassParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_bandpass_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let len = 512usize;
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let t = i as f64 * 0.07;

            data.push((t).sin() * 0.8 + (0.33 * t).sin() * 0.2 + 0.001 * i as f64);
        }

        let input = BandPassInput::from_slice(&data, BandPassParams::default());

        let base = bandpass(&input)?;

        let mut bp = vec![0.0; len];
        let mut bpn = vec![0.0; len];
        let mut sig = vec![0.0; len];
        let mut trg = vec![0.0; len];

        #[allow(unused_mut)]
        let mut _ok = bandpass_into(&input, &mut bp, &mut bpn, &mut sig, &mut trg)?;

        assert_eq!(bp.len(), base.bp.len());
        assert_eq!(bpn.len(), base.bp_normalized.len());
        assert_eq!(sig.len(), base.signal.len());
        assert_eq!(trg.len(), base.trigger.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(bp[i], base.bp[i]),
                "bp mismatch at {}: {} vs {}",
                i,
                bp[i],
                base.bp[i]
            );
            assert!(
                eq_or_both_nan(bpn[i], base.bp_normalized[i]),
                "bpn mismatch at {}: {} vs {}",
                i,
                bpn[i],
                base.bp_normalized[i]
            );
            assert!(
                eq_or_both_nan(sig[i], base.signal[i]),
                "sig mismatch at {}: {} vs {}",
                i,
                sig[i],
                base.signal[i]
            );
            assert!(
                eq_or_both_nan(trg[i], base.trigger[i]),
                "trg mismatch at {}: {} vs {}",
                i,
                trg[i],
                base.trigger[i]
            );
        }

        Ok(())
    }

    fn check_bandpass_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = BandPassParams::default();
        let input = BandPassInput::from_candles(&candles, "close", default_params);
        let output = bandpass_with_kernel(&input, kernel)?;
        assert_eq!(output.bp.len(), candles.close.len());
        Ok(())
    }
    fn check_bandpass_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BandPassInput::with_default_candles(&candles);
        let result = bandpass_with_kernel(&input, kernel)?;
        let expected_bp_last_five = [
            -236.23678021132827,
            -247.4846395608195,
            -242.3788746078502,
            -212.89589193350128,
            -179.97293838509464,
        ];
        let expected_bp_normalized_last_five = [
            -0.4399672555578846,
            -0.4651011734720517,
            -0.4596426251402882,
            -0.40739824942488945,
            -0.3475245023284841,
        ];
        let expected_signal_last_five = [-1.0, 1.0, 1.0, 1.0, 1.0];
        let expected_trigger_last_five = [
            -0.4746908356434579,
            -0.4353877348116954,
            -0.3727126131420441,
            -0.2746336628365846,
            -0.18240018384226137,
        ];
        let start = result.bp.len().saturating_sub(5);
        assert!(result.bp.len() >= 5);
        assert!(result.bp_normalized.len() >= 5);
        assert!(result.signal.len() >= 5);
        assert!(result.trigger.len() >= 5);

        let first_bp = result
            .bp
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(result.bp.len());
        let first_bpn = result
            .bp_normalized
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(result.bp_normalized.len());
        let first_sig = result
            .signal
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(result.signal.len());
        let first_trig = result
            .trigger
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(result.trigger.len());

        if first_sig >= start {
            panic!(
                "Signal values are all NaN in the last 5 indices. Debug info:\n\
				first_bp: {}, first_bpn: {}, first_sig: {}, first_trig: {}, start: {}, len: {}\n\
				bp[start]: {:?}, bpn[start]: {:?}, trig[start]: {:?}",
                first_bp,
                first_bpn,
                first_sig,
                first_trig,
                start,
                result.bp.len(),
                result.bp[start],
                result.bp_normalized[start],
                result.trigger[start]
            );
        }
        for (i, &value) in result.bp[start..].iter().enumerate() {
            assert!(
                (value - expected_bp_last_five[i]).abs() < 1e-1,
                "BP value mismatch at index {}: expected {}, got {}",
                i,
                expected_bp_last_five[i],
                value
            );
        }
        for (i, &value) in result.bp_normalized[start..].iter().enumerate() {
            assert!(
                (value - expected_bp_normalized_last_five[i]).abs() < 1e-1,
                "BP Normalized value mismatch at index {}: expected {}, got {}",
                i,
                expected_bp_normalized_last_five[i],
                value
            );
        }
        for (i, &value) in result.signal[start..].iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            assert!(
                (value - expected_signal_last_five[i]).abs() < 1e-1,
                "Signal value mismatch at index {}: expected {}, got {}",
                i,
                expected_signal_last_five[i],
                value
            );
        }
        for (i, &value) in result.trigger[start..].iter().enumerate() {
            assert!(
                (value - expected_trigger_last_five[i]).abs() < 1e-1,
                "Trigger value mismatch at index {}: expected {}, got {}",
                i,
                expected_trigger_last_five[i],
                value
            );
        }

        for val in &result.bp {
            if !val.is_nan() {
                assert!(val.is_finite(), "bp value not finite: {}", val);
            }
        }
        for val in &result.bp_normalized {
            if !val.is_nan() {
                assert!(val.is_finite(), "bp_normalized value not finite: {}", val);
            }
        }
        for val in &result.signal {
            if !val.is_nan() {
                assert!(val.is_finite(), "signal value not finite: {}", val);
            }
        }
        for val in &result.trigger {
            if !val.is_nan() {
                assert!(val.is_finite(), "trigger value not finite: {}", val);
            }
        }
        Ok(())
    }
    fn check_bandpass_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BandPassInput::with_default_candles(&candles);
        match input.data {
            BandPassData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected BandPassData::Candles"),
        }
        let output = bandpass_with_kernel(&input, kernel)?;
        assert_eq!(output.bp.len(), candles.close.len());
        Ok(())
    }
    fn check_bandpass_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = BandPassParams {
            period: Some(0),
            bandwidth: Some(0.3),
        };
        let input = BandPassInput::from_slice(&input_data, params);
        let res = bandpass_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }
    fn check_bandpass_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = BandPassParams {
            period: Some(10),
            bandwidth: Some(0.3),
        };
        let input = BandPassInput::from_slice(&data_small, params);
        let res = bandpass_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }
    fn check_bandpass_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = BandPassParams {
            period: Some(20),
            bandwidth: Some(0.3),
        };
        let input = BandPassInput::from_slice(&single_point, params);
        let res = bandpass_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }
    fn check_bandpass_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = BandPassParams {
            period: Some(20),
            bandwidth: Some(0.3),
        };
        let first_input = BandPassInput::from_candles(&candles, "close", first_params);
        let first_result = bandpass_with_kernel(&first_input, kernel)?;
        let second_params = BandPassParams {
            period: Some(30),
            bandwidth: Some(0.5),
        };
        let second_input = BandPassInput::from_slice(&first_result.bp, second_params);
        let second_result = bandpass_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.bp.len(), first_result.bp.len());
        Ok(())
    }
    fn check_bandpass_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BandPassInput::from_candles(
            &candles,
            "close",
            BandPassParams {
                period: Some(20),
                bandwidth: Some(0.3),
            },
        );
        let res = bandpass_with_kernel(&input, kernel)?;
        assert_eq!(res.bp.len(), candles.close.len());

        let first_bp = res
            .bp
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(res.bp.len());
        let first_bpn = res
            .bp_normalized
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(res.bp_normalized.len());
        let first_sig = res
            .signal
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(res.signal.len());
        let first_trig = res
            .trigger
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(res.trigger.len());

        let warmup_end = first_bp.max(first_bpn).max(first_sig).max(first_trig);
        if warmup_end < res.bp.len() {
            for i in warmup_end..res.bp.len() {
                assert!(!res.bp[i].is_nan(), "bp[{}] is NaN after warmup", i);
                assert!(
                    !res.bp_normalized[i].is_nan(),
                    "bp_normalized[{}] is NaN after warmup",
                    i
                );
                assert!(!res.signal[i].is_nan(), "signal[{}] is NaN after warmup", i);
                assert!(
                    !res.trigger[i].is_nan(),
                    "trigger[{}] is NaN after warmup",
                    i
                );
            }
        }
        Ok(())
    }

    macro_rules! generate_all_bandpass_tests {
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

    #[cfg(debug_assertions)]
    fn check_bandpass_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            BandPassParams {
                period: Some(10),
                bandwidth: Some(0.2),
            },
            BandPassParams {
                period: Some(20),
                bandwidth: Some(0.3),
            },
            BandPassParams {
                period: Some(30),
                bandwidth: Some(0.4),
            },
            BandPassParams {
                period: Some(50),
                bandwidth: Some(0.5),
            },
            BandPassParams {
                period: Some(5),
                bandwidth: Some(0.1),
            },
            BandPassParams {
                period: Some(100),
                bandwidth: Some(0.8),
            },
        ];

        for params in test_params {
            let input = BandPassInput::from_candles(&candles, "close", params.clone());
            let output = bandpass_with_kernel(&input, kernel)?;

            for (i, &val) in output.bp.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in bp at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in bp at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in bp at index {} with params {:?}",
						test_name, val, bits, i, params
					);
                }
            }

            for (i, &val) in output.bp_normalized.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in bp_normalized at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in bp_normalized at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in bp_normalized at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }
            }

            for (i, &val) in output.signal.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in signal at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in signal at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in signal at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }
            }

            for (i, &val) in output.trigger.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in trigger at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in trigger at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in trigger at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_bandpass_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn check_bandpass_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 20;
        let bandwidth = 0.3;

        let input = BandPassInput::from_candles(
            &candles,
            "close",
            BandPassParams {
                period: Some(period),
                bandwidth: Some(bandwidth),
            },
        );
        let batch_output = bandpass_with_kernel(&input, kernel)?;

        let mut stream = BandPassStream::try_new(BandPassParams {
            period: Some(period),
            bandwidth: Some(bandwidth),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            let bp_val = stream.update(price);
            stream_values.push(bp_val);
        }

        assert_eq!(batch_output.bp.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.bp.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            let tol = 1e-10 * b.abs().max(1.0);
            assert!(
                diff < tol,
                "[{}] Streaming vs batch mismatch at index {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }

        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_bandpass_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
                (0.1f64..=0.9f64),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period, bandwidth)| {
            let params = BandPassParams {
                period: Some(period),
                bandwidth: Some(bandwidth),
            };
            let input = BandPassInput::from_slice(&data, params);

            let result = bandpass_with_kernel(&input, kernel)?;

            let ref_result = bandpass_with_kernel(&input, Kernel::Scalar)?;

            prop_assert_eq!(result.bp.len(), data.len());
            prop_assert_eq!(result.bp_normalized.len(), data.len());
            prop_assert_eq!(result.signal.len(), data.len());
            prop_assert_eq!(result.trigger.len(), data.len());

            let hp_period = ((4.0 * period as f64) / bandwidth).round() as usize;
            let expected_warmup = hp_period.saturating_sub(1).max(2);

            if data.len() >= expected_warmup {
                for i in 0..(expected_warmup.saturating_sub(1)).min(data.len()) {
                    prop_assert!(
                        result.bp[i].is_nan(),
                        "bp[{}] should be NaN during warmup but got {}",
                        i,
                        result.bp[i]
                    );
                }
            }

            for i in 0..data.len() {
                if !result.bp[i].is_nan() {
                    prop_assert!(
                        result.bp[i].is_finite(),
                        "bp[{}] not finite: {}",
                        i,
                        result.bp[i]
                    );
                }
                if !result.bp_normalized[i].is_nan() {
                    prop_assert!(
                        result.bp_normalized[i].is_finite(),
                        "bp_normalized[{}] not finite: {}",
                        i,
                        result.bp_normalized[i]
                    );
                }
                if !result.signal[i].is_nan() {
                    prop_assert!(
                        result.signal[i].is_finite(),
                        "signal[{}] not finite: {}",
                        i,
                        result.signal[i]
                    );
                }
                if !result.trigger[i].is_nan() {
                    prop_assert!(
                        result.trigger[i].is_finite(),
                        "trigger[{}] not finite: {}",
                        i,
                        result.trigger[i]
                    );
                }
            }

            for i in 0..data.len() {
                let sig = result.signal[i];
                if !sig.is_nan() {
                    prop_assert!(
                        sig == -1.0 || sig == 0.0 || sig == 1.0,
                        "signal[{}] = {} not in {{-1, 0, 1}}",
                        i,
                        sig
                    );
                }
            }

            for i in 0..data.len() {
                let norm = result.bp_normalized[i];
                if !norm.is_nan() {
                    prop_assert!(
                        norm >= -1.0 - 1e-9 && norm <= 1.0 + 1e-9,
                        "bp_normalized[{}] = {} not in [-1, 1]",
                        i,
                        norm
                    );
                }
            }

            if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9)
                && data.len() > expected_warmup + 20
            {
                let check_start = (expected_warmup + 10).min(data.len() - 1);
                for i in check_start..data.len() {
                    if !result.bp[i].is_nan() {
                        let tolerance = 1e-6 * data[0].abs().max(1.0);
                        prop_assert!(
                            result.bp[i].abs() < tolerance,
                            "bp[{}] = {} not near 0 for constant data (tolerance={})",
                            i,
                            result.bp[i],
                            tolerance
                        );
                    }
                }
            }

            for i in 0..data.len() {
                let bn = result.bp_normalized[i];
                let tr = result.trigger[i];
                let sig = result.signal[i];

                if !bn.is_nan() && !tr.is_nan() && !sig.is_nan() {
                    if (bn - tr).abs() < 1e-12 {
                        prop_assert_eq!(
                            sig,
                            0.0,
                            "signal[{}] should be 0 when bp_normalized≈trigger",
                            i
                        );
                    } else if bn < tr {
                        prop_assert_eq!(
                            sig,
                            1.0,
                            "signal[{}] should be 1 when bp_normalized<trigger ({} < {})",
                            i,
                            bn,
                            tr
                        );
                    } else {
                        prop_assert_eq!(
                            sig,
                            -1.0,
                            "signal[{}] should be -1 when bp_normalized>trigger ({} > {})",
                            i,
                            bn,
                            tr
                        );
                    }
                }
            }

            if data.len() > 50 {
                let last_quarter_start = (data.len() * 3) / 4;
                let mut max_abs_bp = 0.0f64;
                let mut max_abs_input = 0.0f64;

                for i in last_quarter_start..data.len() {
                    if !result.bp[i].is_nan() {
                        max_abs_bp = max_abs_bp.max(result.bp[i].abs());
                    }
                    max_abs_input = max_abs_input.max(data[i].abs());
                }

                if max_abs_input > 0.0 {
                    let amplification = max_abs_bp / max_abs_input;
                    prop_assert!(
							amplification < 100.0,
							"Recursive filter may be unstable: amplification={} at period={}, bandwidth={}",
							amplification, period, bandwidth
						);
                }
            }

            if period == 2 {
                let non_nan_count = result.bp.iter().filter(|&&x| !x.is_nan()).count();
                prop_assert!(
                    non_nan_count >= data.len().saturating_sub(expected_warmup),
                    "period=2 should produce mostly valid output after warmup"
                );
            }

            for i in 0..data.len() {
                let bp = result.bp[i];
                let bp_ref = ref_result.bp[i];
                let bp_norm = result.bp_normalized[i];
                let bp_norm_ref = ref_result.bp_normalized[i];
                let sig = result.signal[i];
                let sig_ref = ref_result.signal[i];
                let trig = result.trigger[i];
                let trig_ref = ref_result.trigger[i];

                if !bp.is_finite() || !bp_ref.is_finite() {
                    prop_assert_eq!(
                        bp.to_bits(),
                        bp_ref.to_bits(),
                        "bp finite/NaN mismatch at {}",
                        i
                    );
                } else {
                    let ulp_diff = bp.to_bits().abs_diff(bp_ref.to_bits());
                    prop_assert!(
                        (bp - bp_ref).abs() <= 1e-9 || ulp_diff <= 4,
                        "bp mismatch at {}: {} vs {} (ULP={})",
                        i,
                        bp,
                        bp_ref,
                        ulp_diff
                    );
                }

                if !bp_norm.is_finite() || !bp_norm_ref.is_finite() {
                    prop_assert_eq!(
                        bp_norm.to_bits(),
                        bp_norm_ref.to_bits(),
                        "bp_normalized finite/NaN mismatch at {}",
                        i
                    );
                } else {
                    let ulp_diff = bp_norm.to_bits().abs_diff(bp_norm_ref.to_bits());
                    prop_assert!(
                        (bp_norm - bp_norm_ref).abs() <= 1e-9 || ulp_diff <= 4,
                        "bp_normalized mismatch at {}: {} vs {} (ULP={})",
                        i,
                        bp_norm,
                        bp_norm_ref,
                        ulp_diff
                    );
                }

                if !sig.is_nan() && !sig_ref.is_nan() {
                    prop_assert_eq!(
                        sig,
                        sig_ref,
                        "signal mismatch at {}: {} vs {}",
                        i,
                        sig,
                        sig_ref
                    );
                } else {
                    prop_assert_eq!(
                        sig.is_nan(),
                        sig_ref.is_nan(),
                        "signal NaN mismatch at {}",
                        i
                    );
                }

                if !trig.is_finite() || !trig_ref.is_finite() {
                    prop_assert_eq!(
                        trig.to_bits(),
                        trig_ref.to_bits(),
                        "trigger finite/NaN mismatch at {}",
                        i
                    );
                } else {
                    let ulp_diff = trig.to_bits().abs_diff(trig_ref.to_bits());
                    prop_assert!(
                        (trig - trig_ref).abs() <= 1e-9 || ulp_diff <= 4,
                        "trigger mismatch at {}: {} vs {} (ULP={})",
                        i,
                        trig,
                        trig_ref,
                        ulp_diff
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_bandpass_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    generate_all_bandpass_tests!(
        check_bandpass_partial_params,
        check_bandpass_accuracy,
        check_bandpass_default_candles,
        check_bandpass_zero_period,
        check_bandpass_period_exceeds_length,
        check_bandpass_very_small_dataset,
        check_bandpass_reinput,
        check_bandpass_nan_handling,
        check_bandpass_no_poison,
        check_bandpass_streaming,
        check_bandpass_property
    );
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = BandPassBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = BandPassParams::default();
        let row_idx = output.row_for_params(&def).expect("default row missing");
        let (bp_slice, _bpn_slice, _sig_slice, _trg_slice) =
            output.row_slices(row_idx).expect("row missing");

        assert_eq!(bp_slice.len(), c.close.len());

        let expected = [
            -236.23678021132827,
            -247.4846395608195,
            -242.3788746078502,
            -212.89589193350128,
            -179.97293838509464,
        ];
        let start = bp_slice.len() - 5;
        for (i, &v) in bp_slice[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = BandPassBatchBuilder::new()
            .kernel(kernel)
            .period_range(5, 50, 5)
            .bandwidth_range(0.1, 0.9, 0.1)
            .apply_candles(&c, "close")?;

        for row_idx in 0..output.rows {
            let params = &output.combos[row_idx];
            let (bp_row, bpn_row, sig_row, trg_row) = output.row_slices(row_idx).unwrap();

            for (col_idx, &val) in bp_row.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in bp at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in bp at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in bp at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }
            }

            for (col_idx, &val) in bpn_row.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in bp_normalized at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in bp_normalized at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in bp_normalized at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }
            }

            for (col_idx, &val) in sig_row.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in signal at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in signal at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in signal at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }
            }

            for (col_idx, &val) in trg_row.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in trigger at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in trigger at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in trigger at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "bandpass")]
#[pyo3(signature = (data, period=20, bandwidth=0.3, kernel=None))]
pub fn bandpass_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    bandwidth: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArrayMethods;
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = BandPassParams {
        period: Some(period),
        bandwidth: Some(bandwidth),
    };
    let inp = BandPassInput::from_slice(slice_in, params);

    let out = py
        .allow_threads(|| bandpass_with_kernel(&inp, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("bp", out.bp.into_pyarray(py))?;
    dict.set_item("bp_normalized", out.bp_normalized.into_pyarray(py))?;
    dict.set_item("signal", out.signal.into_pyarray(py))?;
    dict.set_item("trigger", out.trigger.into_pyarray(py))?;

    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "BandPassStream")]
pub struct BandPassStreamPy {
    stream: BandPassStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl BandPassStreamPy {
    #[new]
    fn new(period: usize, bandwidth: f64) -> PyResult<Self> {
        let params = BandPassParams {
            period: Some(period),
            bandwidth: Some(bandwidth),
        };
        let stream =
            BandPassStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(BandPassStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> f64 {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "bandpass_batch")]
#[pyo3(signature = (data, period_range, bandwidth_range, kernel=None))]
pub fn bandpass_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    bandwidth_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let sweep = BandPassBatchRange {
        period: period_range,
        bandwidth: bandwidth_range,
    };

    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| bandpass_batch_with_kernel(slice_in, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = output.rows;
    let cols = output.cols;

    unsafe {
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| PyValueError::new_err("bandpass_batch: rows*cols overflow"))?;
        let bp_arr = PyArray1::<f64>::new(py, [total], false);
        let bpn_arr = PyArray1::<f64>::new(py, [total], false);
        let sig_arr = PyArray1::<f64>::new(py, [total], false);
        let trg_arr = PyArray1::<f64>::new(py, [total], false);

        bp_arr.as_slice_mut()?.copy_from_slice(&output.bp);
        bpn_arr
            .as_slice_mut()?
            .copy_from_slice(&output.bp_normalized);
        sig_arr.as_slice_mut()?.copy_from_slice(&output.signal);
        trg_arr.as_slice_mut()?.copy_from_slice(&output.trigger);

        let d = PyDict::new(py);
        d.set_item("bp", bp_arr.reshape((rows, cols))?)?;
        d.set_item("bp_normalized", bpn_arr.reshape((rows, cols))?)?;
        d.set_item("signal", sig_arr.reshape((rows, cols))?)?;
        d.set_item("trigger", trg_arr.reshape((rows, cols))?)?;
        d.set_item(
            "periods",
            output
                .combos
                .iter()
                .map(|p| p.period.unwrap() as u64)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        d.set_item(
            "bandwidths",
            output
                .combos
                .iter()
                .map(|p| p.bandwidth.unwrap())
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        Ok(d)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "bandpass_cuda_batch_dev")]
#[pyo3(signature = (close_f32, period_range, bandwidth_range, device_id=0))]
pub fn bandpass_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    bandwidth_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = close_f32.as_slice()?;
    let sweep = BandPassBatchRange {
        period: period_range,
        bandwidth: bandwidth_range,
    };
    let (outputs, combos, dev_id, ctx) = py.allow_threads(|| -> PyResult<_> {
        let cuda =
            CudaBandpass::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.context_arc();
        let res = cuda
            .bandpass_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.stream()
            .synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((res.outputs, res.combos, dev_id, ctx))
    })?;

    let d = PyDict::new(py);
    d.set_item(
        "bp",
        Py::new(
            py,
            BandPassDeviceArrayF32Py {
                inner: outputs.first,
                _ctx: ctx.clone(),
                device_id: dev_id,
                stream: 0,
            },
        )?,
    )?;
    d.set_item(
        "bp_normalized",
        Py::new(
            py,
            BandPassDeviceArrayF32Py {
                inner: outputs.second,
                _ctx: ctx.clone(),
                device_id: dev_id,
                stream: 0,
            },
        )?,
    )?;
    d.set_item(
        "signal",
        Py::new(
            py,
            BandPassDeviceArrayF32Py {
                inner: outputs.third,
                _ctx: ctx.clone(),
                device_id: dev_id,
                stream: 0,
            },
        )?,
    )?;
    d.set_item(
        "trigger",
        Py::new(
            py,
            BandPassDeviceArrayF32Py {
                inner: outputs.fourth,
                _ctx: ctx,
                device_id: dev_id,
                stream: 0,
            },
        )?,
    )?;

    let periods: Vec<usize> = combos.iter().map(|p| p.period.unwrap()).collect();
    let bands: Vec<f64> = combos.iter().map(|p| p.bandwidth.unwrap()).collect();
    d.set_item("periods", periods.into_pyarray(py))?;
    d.set_item("bandwidths", bands.into_pyarray(py))?;
    d.set_item("rows", combos.len())?;
    d.set_item("cols", slice.len())?;
    Ok(d)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "bandpass_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, bandwidth, device_id=0))]
pub fn bandpass_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    period: usize,
    bandwidth: f64,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
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
    let params = BandPassParams {
        period: Some(period),
        bandwidth: Some(bandwidth),
    };

    let (outputs, dev_id, ctx) = py.allow_threads(|| -> PyResult<_> {
        let cuda =
            CudaBandpass::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.context_arc();
        let out = cuda
            .bandpass_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.stream()
            .synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((out, dev_id, ctx))
    })?;

    let d = PyDict::new(py);
    d.set_item(
        "bp",
        Py::new(
            py,
            BandPassDeviceArrayF32Py {
                inner: outputs.first,
                _ctx: ctx.clone(),
                device_id: dev_id,
                stream: 0,
            },
        )?,
    )?;
    d.set_item(
        "bp_normalized",
        Py::new(
            py,
            BandPassDeviceArrayF32Py {
                inner: outputs.second,
                _ctx: ctx.clone(),
                device_id: dev_id,
                stream: 0,
            },
        )?,
    )?;
    d.set_item(
        "signal",
        Py::new(
            py,
            BandPassDeviceArrayF32Py {
                inner: outputs.third,
                _ctx: ctx.clone(),
                device_id: dev_id,
                stream: 0,
            },
        )?,
    )?;
    d.set_item(
        "trigger",
        Py::new(
            py,
            BandPassDeviceArrayF32Py {
                inner: outputs.fourth,
                _ctx: ctx,
                device_id: dev_id,
                stream: 0,
            },
        )?,
    )?;
    d.set_item("rows", rows)?;
    d.set_item("cols", cols)?;
    d.set_item("period", period)?;
    d.set_item("bandwidth", bandwidth)?;
    Ok(d)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct BandPassJsResult {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "bandpass_js")]
pub fn bandpass_js(data: &[f64], period: usize, bandwidth: f64) -> Result<JsValue, JsValue> {
    let input = BandPassInput::from_slice(
        data,
        BandPassParams {
            period: Some(period),
            bandwidth: Some(bandwidth),
        },
    );
    let out = bandpass_with_kernel(&input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let cols = data.len();
    let mut values = Vec::with_capacity(4 * cols);
    values.extend_from_slice(&out.bp);
    values.extend_from_slice(&out.bp_normalized);
    values.extend_from_slice(&out.signal);
    values.extend_from_slice(&out.trigger);
    serde_wasm_bindgen::to_value(&BandPassJsResult {
        values,
        rows: 4,
        cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct BandPassBatchConfig {
    pub period_range: (usize, usize, usize),
    pub bandwidth_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct BandPassBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub combos: Vec<BandPassParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "bandpass_batch")]
pub fn bandpass_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: BandPassBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = BandPassBatchRange {
        period: cfg.period_range,
        bandwidth: cfg.bandwidth_range,
    };
    let out = bandpass_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let rows = out.rows;
    let cols = out.cols;

    let total = rows
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| JsValue::from_str("bandpass_batch_js: rows*cols overflow"))?;
    let mut values = Vec::with_capacity(total);
    values.extend_from_slice(&out.bp);
    values.extend_from_slice(&out.bp_normalized);
    values.extend_from_slice(&out.signal);
    values.extend_from_slice(&out.trigger);

    let js_output = js_sys::Object::new();
    js_sys::Reflect::set(
        &js_output,
        &JsValue::from_str("values"),
        &serde_wasm_bindgen::to_value(&values).unwrap(),
    )?;
    js_sys::Reflect::set(
        &js_output,
        &JsValue::from_str("combos"),
        &JsValue::from_f64(out.combos.len() as f64),
    )?;
    js_sys::Reflect::set(
        &js_output,
        &JsValue::from_str("outputs"),
        &JsValue::from_f64(4.0),
    )?;
    js_sys::Reflect::set(
        &js_output,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    Ok(JsValue::from(js_output))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "bandpass_batch_metadata")]
pub fn bandpass_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    bandwidth_start: f64,
    bandwidth_end: f64,
    bandwidth_step: f64,
) -> Result<JsValue, JsValue> {
    let sweep = BandPassBatchRange {
        period: (period_start, period_end, period_step),
        bandwidth: (bandwidth_start, bandwidth_end, bandwidth_step),
    };
    let combos = expand_grid(&sweep);

    let mut flat = Vec::with_capacity(combos.len() * 2);
    for combo in &combos {
        flat.push(combo.period.unwrap() as f64);
        flat.push(combo.bandwidth.unwrap());
    }
    serde_wasm_bindgen::to_value(&flat).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn bandpass_into(
    in_ptr: *const f64,
    len: usize,
    period: usize,
    bandwidth: f64,
    bp_ptr: *mut f64,
    bpn_ptr: *mut f64,
    sig_ptr: *mut f64,
    trg_ptr: *mut f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null()
        || bp_ptr.is_null()
        || bpn_ptr.is_null()
        || sig_ptr.is_null()
        || trg_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer in bandpass_into"));
    }

    if in_ptr == bp_ptr as *const f64
        || in_ptr == bpn_ptr as *const f64
        || in_ptr == sig_ptr as *const f64
        || in_ptr == trg_ptr as *const f64
    {
        return Err(JsValue::from_str(
            "input and output pointers must not alias",
        ));
    }

    let out_ptrs = [bp_ptr, bpn_ptr, sig_ptr, trg_ptr];
    for i in 0..out_ptrs.len() {
        for j in i + 1..out_ptrs.len() {
            if out_ptrs[i] == out_ptrs[j] {
                return Err(JsValue::from_str(
                    "output pointers must not alias with each other",
                ));
            }
        }
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let mut bp = std::slice::from_raw_parts_mut(bp_ptr, len);
        let mut bpn = std::slice::from_raw_parts_mut(bpn_ptr, len);
        let mut sig = std::slice::from_raw_parts_mut(sig_ptr, len);
        let mut trg = std::slice::from_raw_parts_mut(trg_ptr, len);
        let input = BandPassInput::from_slice(
            data,
            BandPassParams {
                period: Some(period),
                bandwidth: Some(bandwidth),
            },
        );
        bandpass_into_slice(
            &mut bp,
            &mut bpn,
            &mut sig,
            &mut trg,
            &input,
            detect_best_kernel(),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn bandpass_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn bandpass_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}
