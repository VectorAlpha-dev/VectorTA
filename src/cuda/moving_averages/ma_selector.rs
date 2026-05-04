#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::cuda::device_types::ensure_same_device;
use crate::cuda::moving_averages::*;
use crate::cuda::runtime::CudaSession;
use crate::cuda::{CudaDeviceOhlcRef, CudaDeviceOhlcvRef, CudaDeviceSliceF32Ref};
use crate::utilities::data_loader::{source_type, Candles};

use cust::context::Context;
use cust::memory::DevicePointer;
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaMaSelectorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("invalid range: start={start} end={end} step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[inline]
fn mem_check_enabled() -> bool {
    match std::env::var("CUDA_MEM_CHECK") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

#[inline]
fn device_mem_info() -> Option<(usize, usize)> {
    mem_get_info().ok()
}

#[inline]
fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMaSelectorError> {
    if !mem_check_enabled() {
        return Ok(());
    }
    if let Some((free, _total)) = device_mem_info() {
        if required_bytes.saturating_add(headroom_bytes) <= free {
            Ok(())
        } else {
            Err(CudaMaSelectorError::OutOfMemory {
                required: required_bytes,
                free,
                headroom: headroom_bytes,
            })
        }
    } else {
        Ok(())
    }
}

#[inline]
fn get_param_f64(
    params: Option<&HashMap<String, f64>>,
    ma_type: &str,
    key: &'static str,
) -> Result<Option<f64>, CudaMaSelectorError> {
    match params.and_then(|m| m.get(key).copied()) {
        None => Ok(None),
        Some(v) if v.is_finite() => Ok(Some(v)),
        Some(v) => Err(CudaMaSelectorError::InvalidInput(format!(
            "invalid param '{key}' for '{ma_type}': expected finite number, got {v}"
        ))),
    }
}

#[inline]
fn get_param_usize(
    params: Option<&HashMap<String, f64>>,
    ma_type: &str,
    key: &'static str,
) -> Result<Option<usize>, CudaMaSelectorError> {
    let Some(v) = get_param_f64(params, ma_type, key)? else {
        return Ok(None);
    };
    if v < 0.0 {
        return Err(CudaMaSelectorError::InvalidInput(format!(
            "invalid param '{key}' for '{ma_type}': expected >= 0, got {v}"
        )));
    }
    let r = v.round();
    if (v - r).abs() > 1e-9 {
        return Err(CudaMaSelectorError::InvalidInput(format!(
            "invalid param '{key}' for '{ma_type}': expected integer, got {v}"
        )));
    }
    if r > (usize::MAX as f64) {
        return Err(CudaMaSelectorError::InvalidInput(format!(
            "invalid param '{key}' for '{ma_type}': too large for usize: {v}"
        )));
    }
    Ok(Some(r as usize))
}

#[inline]
fn get_param_u32(
    params: Option<&HashMap<String, f64>>,
    ma_type: &str,
    key: &'static str,
) -> Result<Option<u32>, CudaMaSelectorError> {
    let Some(v) = get_param_usize(params, ma_type, key)? else {
        return Ok(None);
    };
    if v > (u32::MAX as usize) {
        return Err(CudaMaSelectorError::InvalidInput(format!(
            "invalid param '{key}' for '{ma_type}': too large for u32: {v}"
        )));
    }
    Ok(Some(v as u32))
}

#[inline]
fn get_param_bool01(
    params: Option<&HashMap<String, f64>>,
    ma_type: &str,
    key: &'static str,
) -> Result<Option<bool>, CudaMaSelectorError> {
    let Some(v) = get_param_f64(params, ma_type, key)? else {
        return Ok(None);
    };
    let r = v.round();
    if (v - r).abs() > 1e-9 {
        return Err(CudaMaSelectorError::InvalidInput(format!(
            "invalid param '{key}' for '{ma_type}': expected integer 0 or 1, got {v}"
        )));
    }
    match r as i32 {
        0 => Ok(Some(false)),
        1 => Ok(Some(true)),
        _ => Err(CudaMaSelectorError::InvalidInput(format!(
            "invalid param '{key}' for '{ma_type}': expected 0 or 1, got {v}"
        ))),
    }
}

#[inline]
fn get_param_str<'a>(
    params: Option<&'a HashMap<String, String>>,
    _ma_type: &str,
    key: &'static str,
) -> Option<&'a str> {
    params.and_then(|m| m.get(key).map(String::as_str))
}

fn typed_params_to_maps(
    ma_type: &str,
    params: &[CudaMaParamKV<'_>],
) -> Result<(HashMap<String, f64>, HashMap<String, String>), CudaMaSelectorError> {
    let mut numeric: HashMap<String, f64> = HashMap::with_capacity(params.len());
    let mut text: HashMap<String, String> = HashMap::new();
    for p in params {
        match p.value {
            CudaMaParamValue::Int(v) => {
                numeric.insert(p.key.to_string(), v as f64);
            }
            CudaMaParamValue::Float(v) => {
                if !v.is_finite() {
                    return Err(CudaMaSelectorError::InvalidInput(format!(
                        "invalid param '{}' for '{}': expected finite number, got {}",
                        p.key, ma_type, v
                    )));
                }
                numeric.insert(p.key.to_string(), v);
            }
            CudaMaParamValue::Bool(v) => {
                numeric.insert(p.key.to_string(), if v { 1.0 } else { 0.0 });
            }
            CudaMaParamValue::EnumString(v) => {
                text.insert(p.key.to_string(), v.to_string());
            }
        }
    }
    Ok((numeric, text))
}

fn build_sweep_periods(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaMaSelectorError> {
    let periods: Vec<usize> = if step == 0 || start == end {
        vec![start]
    } else if start < end {
        let s = step.max(1);
        let mut v = Vec::new();
        let mut cur = start;
        while cur <= end {
            v.push(cur);
            match cur.checked_add(s) {
                Some(next) if next > cur => {
                    cur = next;
                    if cur > end {
                        break;
                    }
                }
                _ => break,
            }
        }
        v
    } else {
        let s = step.max(1);
        let mut v = Vec::new();
        let mut cur = start;
        while cur >= end {
            v.push(cur);
            if cur == 0 {
                break;
            }
            let next = cur.saturating_sub(s);
            if next == cur {
                break;
            }
            cur = next;
            if cur < end {
                break;
            }
        }
        v
    };
    if periods.is_empty() {
        Err(CudaMaSelectorError::InvalidRange { start, end, step })
    } else {
        Ok(periods)
    }
}

fn periods_to_i32(periods: &[usize]) -> Result<Box<[i32]>, CudaMaSelectorError> {
    periods
        .iter()
        .map(|&period| {
            i32::try_from(period).map_err(|_| {
                CudaMaSelectorError::InvalidInput(format!(
                    "period exceeds CUDA i32 range: {}",
                    period
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

#[derive(Debug, Clone, Copy)]
pub enum CudaMaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },

    Slice(&'a [f64]),

    SliceF32(&'a [f32]),

    OhlcF32 {
        open: &'a [f32],
        high: &'a [f32],
        low: &'a [f32],
        close: &'a [f32],
        source: Option<&'a [f32]>,
    },

    OhlcvF32 {
        timestamp: Option<&'a [i64]>,
        open: &'a [f32],
        high: &'a [f32],
        low: &'a [f32],
        close: &'a [f32],
        volume: &'a [f32],
        source: Option<&'a [f32]>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum CudaMaDeviceDataRef<'a> {
    Slice(CudaDeviceSliceF32Ref),
    Ohlc(CudaDeviceOhlcRef),
    Ohlcv(CudaDeviceOhlcvRef),
    _Marker(std::marker::PhantomData<&'a ()>),
}

impl<'a> CudaMaDeviceDataRef<'a> {
    #[inline]
    fn prices(self) -> CudaDeviceSliceF32Ref {
        match self {
            Self::Slice(values) => values,
            Self::Ohlc(values) => values.source().unwrap_or(values.close()),
            Self::Ohlcv(values) => values.source().unwrap_or(values.close()),
            Self::_Marker(_) => unreachable!(),
        }
    }

    #[inline]
    fn close(self) -> CudaDeviceSliceF32Ref {
        match self {
            Self::Slice(values) => values,
            Self::Ohlc(values) => values.close(),
            Self::Ohlcv(values) => values.close(),
            Self::_Marker(_) => unreachable!(),
        }
    }

    #[inline]
    fn high(self) -> Option<CudaDeviceSliceF32Ref> {
        match self {
            Self::Slice(_) => None,
            Self::Ohlc(values) => Some(values.high()),
            Self::Ohlcv(values) => Some(values.high()),
            Self::_Marker(_) => unreachable!(),
        }
    }

    #[inline]
    fn low(self) -> Option<CudaDeviceSliceF32Ref> {
        match self {
            Self::Slice(_) => None,
            Self::Ohlc(values) => Some(values.low()),
            Self::Ohlcv(values) => Some(values.low()),
            Self::_Marker(_) => unreachable!(),
        }
    }

    #[inline]
    fn volume(self) -> Option<CudaDeviceSliceF32Ref> {
        match self {
            Self::Slice(_) | Self::Ohlc(_) => None,
            Self::Ohlcv(values) => Some(values.volume()),
            Self::_Marker(_) => unreachable!(),
        }
    }

    #[inline]
    fn prices_len(self) -> usize {
        self.prices().len()
    }

    #[inline]
    fn device_id(self) -> u32 {
        self.prices().device_id()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CudaMaParamValue<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    EnumString(&'a str),
}

#[derive(Debug, Clone, Copy)]
pub struct CudaMaParamKV<'a> {
    pub key: &'a str,
    pub value: CudaMaParamValue<'a>,
}

pub struct CudaMaSweepPlan {
    device_id: u32,
    periods: Box<[usize]>,
    periods_i32: Box<[i32]>,
    d_periods: DeviceBuffer<i32>,
}

impl CudaMaSweepPlan {
    pub fn periods(&self) -> &[usize] {
        &self.periods
    }

    pub fn len(&self) -> usize {
        self.periods.len()
    }

    pub fn is_empty(&self) -> bool {
        self.periods.is_empty()
    }
}

impl<'a> CudaMaData<'a> {
    #[inline]
    fn as_prices_f64(self) -> &'a [f64] {
        match self {
            CudaMaData::Slice(s) => s,
            CudaMaData::Candles { candles, source } => source_type(candles, source),
            CudaMaData::SliceF32(_) | CudaMaData::OhlcF32 { .. } | CudaMaData::OhlcvF32 { .. } => {
                panic!("as_prices_f64 called for f32 data")
            }
        }
    }

    #[inline]
    fn prices_len(self) -> usize {
        match self {
            CudaMaData::Slice(s) => s.len(),
            CudaMaData::SliceF32(s) => s.len(),
            CudaMaData::OhlcF32 { close, source, .. } => source.unwrap_or(close).len(),
            CudaMaData::OhlcvF32 { close, source, .. } => source.unwrap_or(close).len(),
            CudaMaData::Candles { candles, source } => source_type(candles, source).len(),
        }
    }

    #[inline]
    fn to_prices_f32(self) -> Vec<f32> {
        match self {
            CudaMaData::SliceF32(s) => s.to_vec(),
            CudaMaData::Slice(s) => s.iter().map(|&v| v as f32).collect(),
            CudaMaData::OhlcF32 { close, source, .. } => source.unwrap_or(close).to_vec(),
            CudaMaData::OhlcvF32 { close, source, .. } => source.unwrap_or(close).to_vec(),
            CudaMaData::Candles { candles, source } => {
                let src = source_type(candles, source);
                src.iter().map(|&v| v as f32).collect()
            }
        }
    }
}

struct BorrowedCudaDeviceSeries {
    buf: ManuallyDrop<DeviceBuffer<f32>>,
}

impl BorrowedCudaDeviceSeries {
    unsafe fn from_view(view: CudaDeviceSliceF32Ref) -> Self {
        Self {
            buf: ManuallyDrop::new(DeviceBuffer::from_raw_parts(
                DevicePointer::<f32>::from_raw(view.device_ptr()),
                view.len(),
            )),
        }
    }

    fn as_buffer(&self) -> &DeviceBuffer<f32> {
        unsafe {
            &*(&self.buf as *const ManuallyDrop<DeviceBuffer<f32>> as *const DeviceBuffer<f32>)
        }
    }
}

struct BorrowedCudaMaInputs {
    prices: BorrowedCudaDeviceSeries,
    close: BorrowedCudaDeviceSeries,
    high: Option<BorrowedCudaDeviceSeries>,
    low: Option<BorrowedCudaDeviceSeries>,
    volume: Option<BorrowedCudaDeviceSeries>,
}

impl BorrowedCudaMaInputs {
    unsafe fn from_data(data: CudaMaDeviceDataRef<'_>) -> Result<Self, CudaMaSelectorError> {
        let device_id = data.device_id();
        ensure_same_device("ma_selector.close", device_id, data.close().device_id())
            .map_err(map_device_view_err)?;
        if let Some(high) = data.high() {
            ensure_same_device("ma_selector.high", device_id, high.device_id())
                .map_err(map_device_view_err)?;
        }
        if let Some(low) = data.low() {
            ensure_same_device("ma_selector.low", device_id, low.device_id())
                .map_err(map_device_view_err)?;
        }
        if let Some(volume) = data.volume() {
            ensure_same_device("ma_selector.volume", device_id, volume.device_id())
                .map_err(map_device_view_err)?;
        }

        Ok(Self {
            prices: Self::borrow(data.prices()),
            close: Self::borrow(data.close()),
            high: data.high().map(Self::borrow),
            low: data.low().map(Self::borrow),
            volume: data.volume().map(Self::borrow),
        })
    }

    fn as_vram_inputs(&self) -> super::vram_ma::VramMaInputs<'_> {
        super::vram_ma::VramMaInputs {
            prices: self.prices.as_buffer(),
            close: self.close.as_buffer(),
            high: self.high.as_ref().map(BorrowedCudaDeviceSeries::as_buffer),
            low: self.low.as_ref().map(BorrowedCudaDeviceSeries::as_buffer),
            volume: self
                .volume
                .as_ref()
                .map(BorrowedCudaDeviceSeries::as_buffer),
        }
    }

    fn borrow(view: CudaDeviceSliceF32Ref) -> BorrowedCudaDeviceSeries {
        unsafe { BorrowedCudaDeviceSeries::from_view(view) }
    }
}

fn map_device_view_err(err: crate::cuda::CudaDeviceViewError) -> CudaMaSelectorError {
    match err {
        crate::cuda::CudaDeviceViewError::NullPointerWithNonZeroLength => {
            CudaMaSelectorError::InvalidInput("invalid device view: null pointer".into())
        }
        crate::cuda::CudaDeviceViewError::MatrixLenOverflow => {
            CudaMaSelectorError::InvalidInput("invalid device view: matrix overflow".into())
        }
        crate::cuda::CudaDeviceViewError::LengthMismatch {
            name,
            expected,
            actual,
        } => CudaMaSelectorError::InvalidInput(format!(
            "device view length mismatch for {name}: expected {expected}, actual {actual}"
        )),
        crate::cuda::CudaDeviceViewError::DeviceMismatch {
            expected, actual, ..
        } => CudaMaSelectorError::DeviceMismatch {
            buf: actual,
            current: expected,
        },
    }
}

pub struct CudaMaSelector {
    session: Arc<CudaSession>,
    device_id: usize,

    stream: Arc<cust::stream::Stream>,
    _context: Arc<Context>,
    vram_ma: RefCell<Option<super::vram_ma::VramMaComputer>>,
    cached_sweep_periods_i32: RefCell<Option<(Box<[i32]>, DeviceBuffer<i32>)>>,
    sma: RefCell<Option<CudaSma>>,
    ema: RefCell<Option<CudaEma>>,
    dema: RefCell<Option<CudaDema>>,
    wma: RefCell<Option<CudaWma>>,
    zlema: RefCell<Option<CudaZlema>>,
}

pub struct CudaMaDeviceSelector<'a> {
    selector: &'a CudaMaSelector,
}

impl CudaMaSelector {
    pub fn new(device_id: usize) -> Self {
        let session =
            Arc::new(CudaSession::new(device_id).expect("failed to create shared CUDA session"));
        Self::from_session(session)
    }

    pub fn from_session(session: Arc<CudaSession>) -> Self {
        Self {
            device_id: session.device_id() as usize,
            stream: session.stream_arc(),
            _context: session.context_arc(),
            session,
            vram_ma: RefCell::new(None),
            cached_sweep_periods_i32: RefCell::new(None),
            sma: RefCell::new(None),
            ema: RefCell::new(None),
            dema: RefCell::new(None),
            wma: RefCell::new(None),
            zlema: RefCell::new(None),
        }
    }

    fn with_sma<R>(
        &self,
        f: impl FnOnce(&CudaSma) -> Result<R, CudaMaSelectorError>,
    ) -> Result<R, CudaMaSelectorError> {
        if self.sma.borrow().is_none() {
            let cuda = CudaSma::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
            *self.sma.borrow_mut() = Some(cuda);
        }
        let sma = self.sma.borrow();
        let cuda = sma
            .as_ref()
            .ok_or_else(|| CudaMaSelectorError::Backend("failed to initialize sma".into()))?;
        f(cuda)
    }

    fn with_ema<R>(
        &self,
        f: impl FnOnce(&CudaEma) -> Result<R, CudaMaSelectorError>,
    ) -> Result<R, CudaMaSelectorError> {
        if self.ema.borrow().is_none() {
            let cuda = CudaEma::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
            *self.ema.borrow_mut() = Some(cuda);
        }
        let ema = self.ema.borrow();
        let cuda = ema
            .as_ref()
            .ok_or_else(|| CudaMaSelectorError::Backend("failed to initialize ema".into()))?;
        f(cuda)
    }

    fn with_dema<R>(
        &self,
        f: impl FnOnce(&CudaDema) -> Result<R, CudaMaSelectorError>,
    ) -> Result<R, CudaMaSelectorError> {
        if self.dema.borrow().is_none() {
            let cuda = CudaDema::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
            *self.dema.borrow_mut() = Some(cuda);
        }
        let dema = self.dema.borrow();
        let cuda = dema
            .as_ref()
            .ok_or_else(|| CudaMaSelectorError::Backend("failed to initialize dema".into()))?;
        f(cuda)
    }

    fn with_wma<R>(
        &self,
        f: impl FnOnce(&CudaWma) -> Result<R, CudaMaSelectorError>,
    ) -> Result<R, CudaMaSelectorError> {
        if self.wma.borrow().is_none() {
            let cuda = CudaWma::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
            *self.wma.borrow_mut() = Some(cuda);
        }
        let wma = self.wma.borrow();
        let cuda = wma
            .as_ref()
            .ok_or_else(|| CudaMaSelectorError::Backend("failed to initialize wma".into()))?;
        f(cuda)
    }

    fn with_zlema<R>(
        &self,
        f: impl FnOnce(&CudaZlema) -> Result<R, CudaMaSelectorError>,
    ) -> Result<R, CudaMaSelectorError> {
        if self.zlema.borrow().is_none() {
            let cuda = CudaZlema::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
            *self.zlema.borrow_mut() = Some(cuda);
        }
        let zlema = self.zlema.borrow();
        let cuda = zlema
            .as_ref()
            .ok_or_else(|| CudaMaSelectorError::Backend("failed to initialize zlema".into()))?;
        f(cuda)
    }

    fn with_vram_ma<R>(
        &self,
        f: impl FnOnce(&mut super::vram_ma::VramMaComputer) -> Result<R, CudaMaSelectorError>,
    ) -> Result<R, CudaMaSelectorError> {
        let mut vram_ma = self.vram_ma.borrow_mut();
        if vram_ma.is_none() {
            *vram_ma = Some(super::vram_ma::VramMaComputer::new(self.device_id as u32));
        }
        let computer = vram_ma
            .as_mut()
            .ok_or_else(|| CudaMaSelectorError::Backend("failed to initialize vram ma".into()))?;
        f(computer)
    }

    fn with_cached_sweep_periods_i32<R>(
        &self,
        periods_i32: &[i32],
        f: impl FnOnce(&DeviceBuffer<i32>) -> Result<R, CudaMaSelectorError>,
    ) -> Result<R, CudaMaSelectorError> {
        let mut cached = self.cached_sweep_periods_i32.borrow_mut();
        let needs_refresh = cached
            .as_ref()
            .map(|(cached_periods, _)| cached_periods.as_ref() != periods_i32)
            .unwrap_or(true);
        if needs_refresh {
            *cached = Some((
                periods_i32.to_vec().into_boxed_slice(),
                DeviceBuffer::from_slice(periods_i32)?,
            ));
        }
        let (_, d_periods) = cached
            .as_ref()
            .ok_or_else(|| CudaMaSelectorError::Backend("failed to cache sweep periods".into()))?;
        f(d_periods)
    }

    pub fn create_sweep_plan(
        &self,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<CudaMaSweepPlan, CudaMaSelectorError> {
        let periods = build_sweep_periods(start, end, step)?.into_boxed_slice();
        let periods_i32 = periods_to_i32(periods.as_ref())?;
        let d_periods = DeviceBuffer::from_slice(periods_i32.as_ref())?;
        Ok(CudaMaSweepPlan {
            device_id: self.device_id as u32,
            periods,
            periods_i32,
            d_periods,
        })
    }

    #[inline]
    pub fn device_native(&self) -> CudaMaDeviceSelector<'_> {
        CudaMaDeviceSelector { selector: self }
    }

    fn ma_to_device_impl(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
        params: Option<&HashMap<String, f64>>,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        let n = data.prices_len();
        if n == 0 {
            return Err(CudaMaSelectorError::InvalidInput(
                "empty price input".into(),
            ));
        }
        if period == 0 || period > n {
            return Err(CudaMaSelectorError::InvalidInput(format!(
                "invalid period: {} for length {}",
                period, n
            )));
        }

        let is = |s: &str| ma_type.eq_ignore_ascii_case(s);

        if is("vwma") {
            if let CudaMaData::Candles { candles, source } = data {
                let prices = source_type(candles, source);
                let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
                let volumes_f32: Vec<f32> = candles.volume.iter().map(|&v| v as f32).collect();
                let sweep = crate::indicators::moving_averages::vwma::VwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaVwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                return cuda
                    .vwma_batch_dev(&prices_f32, &volumes_f32, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()));
            } else {
                return Err(CudaMaSelectorError::Unsupported(
                    "vwma requires candles with volume; pass CudaMaData::Candles".into(),
                ));
            }
        }

        if is("vpwma") {
            if let CudaMaData::Candles { candles, source } = data {
                let prices = source_type(candles, source);
                let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
                let sweep = crate::indicators::moving_averages::vpwma::VpwmaBatchRange {
                    period: (period, period, 0),
                    power: {
                        let p = get_param_f64(params, ma_type, "power")?.unwrap_or(0.382);
                        (p, p, 0.0)
                    },
                };
                let cuda = CudaVpwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .vpwma_batch_dev(&prices_f32, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                return Ok(super::DeviceArrayF32 {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                });
            } else {
                return Err(CudaMaSelectorError::Unsupported(
                    "vpwma requires candles with volume; pass CudaMaData::Candles".into(),
                ));
            }
        }

        if is("vwap") {
            if let CudaMaData::Candles { candles, .. } = data {
                let sweep = crate::indicators::moving_averages::vwap::VwapBatchRange {
                    anchor: ("1d".to_string(), "1d".to_string(), 0),
                };

                let prices = &candles.hlc3;
                let cuda = CudaVwap::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                return cuda
                    .vwap_batch_dev(&candles.timestamp, &candles.volume, prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()));
            } else {
                return Err(CudaMaSelectorError::Unsupported(
                    "vwap requires OHLC + volume; pass CudaMaData::Candles".into(),
                ));
            }
        }

        let mut prices_f32_cache: Option<Vec<f32>> = None;
        macro_rules! ensure_prices {
            () => {{
                if prices_f32_cache.is_none() {
                    prices_f32_cache = Some(data.to_prices_f32());
                }
                prices_f32_cache.as_ref().unwrap().as_slice()
            }};
        }

        match ma_type.to_ascii_lowercase().as_str() {
            "sma" => {
                let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: (period, period, 0),
                };
                self.with_sma(|cuda| {
                    let (dev, _combos) = cuda
                        .sma_batch_dev(ensure_prices!(), &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                    Ok(dev)
                })
            }
            "ema" => {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: (period, period, 0),
                };
                self.with_ema(|cuda| {
                    cuda.ema_batch_dev(ensure_prices!(), &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
                })
            }
            "dema" => {
                let sweep = crate::indicators::moving_averages::dema::DemaBatchRange {
                    period: (period, period, 0),
                };
                self.with_dema(|cuda| {
                    cuda.dema_batch_dev(ensure_prices!(), &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
                })
            }
            "wma" => {
                let sweep = crate::indicators::moving_averages::wma::WmaBatchRange {
                    period: (period, period, 0),
                };
                self.with_wma(|cuda| {
                    cuda.wma_batch_dev(ensure_prices!(), &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
                })
            }
            "zlema" => {
                let sweep = crate::indicators::moving_averages::zlema::ZlemaBatchRange {
                    period: (period, period, 0),
                };
                self.with_zlema(|cuda| {
                    let (dev, _combos) = cuda
                        .zlema_batch_dev(ensure_prices!(), &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                    Ok(dev)
                })
            }
            "smma" => {
                let sweep = crate::indicators::moving_averages::smma::SmmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSmma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.smma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trima" => {
                let sweep = crate::indicators::moving_averages::trima::TrimaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTrima::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.trima_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tema" => {
                let sweep = crate::indicators::moving_averages::tema::TemaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.tema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tilson" => {
                let sweep = crate::indicators::moving_averages::tilson::TilsonBatchRange {
                    period: (period, period, 0),
                    volume_factor: {
                        let v = get_param_f64(params, ma_type, "volume_factor")?.unwrap_or(0.0);
                        (v, v, 0.0)
                    },
                };
                let cuda = CudaTilson::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.tilson_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "wilders" => {
                let sweep = crate::indicators::moving_averages::wilders::WildersBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaWilders::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.wilders_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "cwma" => {
                let sweep = crate::indicators::moving_averages::cwma::CwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaCwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.cwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "jsa" => {
                let sweep = crate::indicators::moving_averages::jsa::JsaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaJsa::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.jsa_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "fwma" => {
                let sweep = crate::indicators::moving_averages::fwma::FwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaFwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.fwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "hma" => {
                let sweep = crate::indicators::moving_averages::hma::HmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaHma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let cuda = CudaHma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .hma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "srwma" => {
                let sweep = crate::indicators::moving_averages::srwma::SrwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSrwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.srwma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sinwma" => {
                let sweep = crate::indicators::moving_averages::sinwma::SinWmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSinwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sinwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sqwma" => {
                let sweep = crate::indicators::moving_averages::sqwma::SqwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSqwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sqwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sgf" => {
                let poly_order = get_param_usize(params, ma_type, "poly_order")?.unwrap_or(2);
                let sweep = crate::indicators::moving_averages::sgf::SgfBatchRange {
                    period: (period, period, 0),
                    poly_order: (poly_order, poly_order, 0),
                };
                let cuda = CudaSgf::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sgf_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "swma" => {
                let sweep = crate::indicators::moving_averages::swma::SwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.swma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "linreg" => {
                let sweep = crate::indicators::moving_averages::linreg::LinRegBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaLinreg::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let cuda = CudaLinreg::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .linreg_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "hwma" => {
                let sweep = crate::indicators::moving_averages::hwma::HwmaBatchRange::default();
                let cuda = CudaHwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.hwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "edcf" => {
                let sweep = crate::indicators::moving_averages::edcf::EdcfBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEdcf::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.edcf_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "dma" => {
                let sweep = crate::indicators::moving_averages::dma::DmaBatchRange {
                    hull_length: (period, period, 0),
                    ema_length: {
                        let v = get_param_usize(params, ma_type, "ema_length")?.unwrap_or(20);
                        (v, v, 0)
                    },
                    ema_gain_limit: {
                        let v = get_param_usize(params, ma_type, "ema_gain_limit")?.unwrap_or(50);
                        (v, v, 0)
                    },
                    hull_ma_type: "WMA".to_string(),
                };
                let cuda = CudaDma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.dma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "highpass" => {
                let sweep = crate::indicators::moving_averages::highpass::HighPassBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaHighpass::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.highpass_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "highpass2" | "highpass_2_pole" => {
                let k = get_param_f64(params, ma_type, "k")?.unwrap_or(0.707);
                let sweep =
                    crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange {
                        period: (period, period, 0),
                        k: (k, k, 0.0),
                    };
                let cuda = CudaHighPass2::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.highpass2_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }

            "alma" => {
                let sweep = crate::indicators::moving_averages::alma::AlmaBatchRange {
                    period: (period, period, 0),

                    offset: {
                        let v = get_param_f64(params, ma_type, "offset")?.unwrap_or(0.85);
                        (v, v, 0.0)
                    },
                    sigma: {
                        let v = get_param_f64(params, ma_type, "sigma")?.unwrap_or(6.0);
                        (v, v, 0.0)
                    },
                };
                let cuda = CudaAlma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.alma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "epma" => {
                let sweep = crate::indicators::moving_averages::epma::EpmaBatchRange {
                    period: (period, period, 0),

                    offset: {
                        let v = get_param_usize(params, ma_type, "offset")?.unwrap_or(4);
                        (v, v, 0)
                    },
                };
                let cuda = CudaEpma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.epma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "gaussian" => {
                let sweep = crate::indicators::moving_averages::gaussian::GaussianBatchRange {
                    period: (period, period, 0),

                    poles: {
                        let v = get_param_usize(params, ma_type, "poles")?.unwrap_or(4);
                        (v, v, 0)
                    },
                };
                let cuda = CudaGaussian::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.gaussian_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "jma" => {
                let sweep = crate::indicators::moving_averages::jma::JmaBatchRange {
                    period: (period, period, 0),

                    phase: {
                        let v = get_param_f64(params, ma_type, "phase")?.unwrap_or(50.0);
                        (v, v, 0.0)
                    },
                    power: {
                        let v = get_param_u32(params, ma_type, "power")?.unwrap_or(2);
                        (v, v, 0)
                    },
                };
                let cuda = CudaJma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.jma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehma" => {
                let sweep = crate::indicators::moving_averages::ehma::EhmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEhma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "supersmoother" => {
                let sweep =
                    crate::indicators::moving_averages::supersmoother::SuperSmootherBatchRange {
                        period: (period, period, 0),
                    };
                let sweep =
                    crate::indicators::moving_averages::supersmoother::SuperSmootherBatchRange {
                        period: (period, period, 0),
                    };
                let cuda = CudaSuperSmoother::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .supersmoother_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "supersmoother_3_pole" => {
                let sweep = crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSupersmoother3Pole::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.supersmoother_3_pole_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "kama" => {
                let sweep = crate::indicators::moving_averages::kama::KamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.kama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sama" => {
                let mut sweep = crate::indicators::moving_averages::sama::SamaBatchRange {
                    length: (period, period, 0),
                    ..Default::default()
                };
                if let Some(v) = get_param_usize(params, ma_type, "maj_length")? {
                    sweep.maj_length = (v, v, 0);
                }
                if let Some(v) = get_param_usize(params, ma_type, "min_length")? {
                    sweep.min_length = (v, v, 0);
                }
                let cuda = CudaSama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_kama" => {
                let sweep = crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEhlersKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_kama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_itrend" => {
                let warmup = get_param_usize(params, ma_type, "warmup_bars")?.unwrap_or(20);
                let sweep =
                    crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange {
                        warmup_bars: (warmup, warmup, 0),
                        max_dc_period: (period, period, 0),
                    };
                let cuda = CudaEhlersITrend::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_itrend_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_ecema" => {
                let gain_limit = get_param_usize(params, ma_type, "gain_limit")?.unwrap_or(50);
                let sweep =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaBatchRange {
                        length: (period, period, 0),
                        gain_limit: (gain_limit, gain_limit, 0),
                    };
                let params =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaParams::default();
                let cuda = CudaEhlersEcema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_ecema_batch_dev(ensure_prices!(), &sweep, &params)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "nama" => {
                let sweep = crate::indicators::moving_averages::nama::NamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaNama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.nama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "nma" => {
                let sweep = crate::indicators::moving_averages::nma::NmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaNma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let cuda = CudaNma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .nma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "pwma" => {
                let sweep = crate::indicators::moving_averages::pwma::PwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaPwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.pwma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "maaq" => {
                let sweep = crate::indicators::moving_averages::maaq::MaaqBatchRange {
                    period: (period, period, 0),
                    fast_period: {
                        let v = get_param_usize(params, ma_type, "fast_period")?.unwrap_or(2);
                        (v, v, 0)
                    },
                    slow_period: {
                        let v = get_param_usize(params, ma_type, "slow_period")?.unwrap_or(30);
                        (v, v, 0)
                    },
                };
                let cuda = CudaMaaq::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.maaq_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "mwdx" => {
                let sweep = crate::indicators::moving_averages::mwdx::MwdxBatchRange {
                    factor: (0.2, 0.2, 0.0),
                };
                let cuda = CudaMwdx::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.mwdx_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "cora_wave" => {
                let r_multi = get_param_f64(params, ma_type, "r_multi")?.unwrap_or(2.0);
                if r_multi < 0.0 {
                    return Err(CudaMaSelectorError::InvalidInput(format!(
                        "invalid param 'r_multi' for '{ma_type}': expected >= 0, got {r_multi}"
                    )));
                }
                let smooth = get_param_bool01(params, ma_type, "smooth")?.unwrap_or(true);
                let sweep = crate::indicators::cora_wave::CoraWaveBatchRange {
                    period: (period, period, 0),
                    r_multi: (r_multi, r_multi, 0.0),
                    smooth,
                };
                let cuda = CudaCoraWave::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.cora_wave_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "reflex" => {
                let sweep = crate::indicators::moving_averages::reflex::ReflexBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaReflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.reflex_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "volatility_adjusted_ma" | "vama" => {
                let sweep =
                    crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
                        base_period: (period, period, 0),
                        vol_period: {
                            let v = get_param_usize(params, ma_type, "vol_period")?.unwrap_or(51);
                            (v, v, 0)
                        },
                    };
                let cuda = CudaVama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.vama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trendflex" => {
                let sweep = crate::indicators::moving_averages::trendflex::TrendFlexBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTrendflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .trendflex_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }

            "frama" => {
                let sc = get_param_usize(params, ma_type, "sc")?.unwrap_or(300);
                let fc = get_param_usize(params, ma_type, "fc")?.unwrap_or(1);
                let sweep = crate::indicators::moving_averages::frama::FramaBatchRange {
                    window: (period, period, 0),
                    sc: (sc, sc, 0),
                    fc: (fc, fc, 0),
                };
                let cuda = CudaFrama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                match data {
                    CudaMaData::Candles { candles, .. } => {
                        let high_f32: Vec<f32> = candles.high.iter().map(|&v| v as f32).collect();
                        let low_f32: Vec<f32> = candles.low.iter().map(|&v| v as f32).collect();
                        let close_f32: Vec<f32> = candles.close.iter().map(|&v| v as f32).collect();
                        let (dev, _combos) = cuda
                            .frama_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                    CudaMaData::SliceF32(s) => {
                        let (dev, _combos) = cuda
                            .frama_batch_dev(s, s, s, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                    CudaMaData::Slice(s) => {
                        let prices_f32: Vec<f32> = s.iter().map(|&v| v as f32).collect();
                        let (dev, _combos) = cuda
                            .frama_batch_dev(&prices_f32, &prices_f32, &prices_f32, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                    CudaMaData::OhlcF32 {
                        high, low, close, ..
                    } => {
                        let (dev, _combos) = cuda
                            .frama_batch_dev(high, low, close, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                    CudaMaData::OhlcvF32 {
                        high, low, close, ..
                    } => {
                        let (dev, _combos) = cuda
                            .frama_batch_dev(high, low, close, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                }
            }

            "vwap" | "vwma" | "vpwma" => Err(CudaMaSelectorError::Unsupported(
                "requires candles; pass CudaMaData::Candles for VWAP/VWMA/VPWMA".into(),
            )),
            "volume_adjusted_ma" => Err(CudaMaSelectorError::Unsupported(
                "volume_adjusted_ma requires volume; use CudaVolumeAdjustedMa".into(),
            )),
            "tradjema" => Err(CudaMaSelectorError::Unsupported(
                "tradjema requires high/low/close; use CudaTradjema directly".into(),
            )),
            "uma" => Err(CudaMaSelectorError::Unsupported(
                "uma requires volume; use CudaUma directly".into(),
            )),
            "mama" => Err(CudaMaSelectorError::Unsupported(
                "mama returns dual outputs; use CudaMama and pick the series".into(),
            )),
            "ehlers_pma" => Err(CudaMaSelectorError::Unsupported(
                "ehlers_pma returns dual outputs; use CudaEhlersPma".into(),
            )),
            "buff_averages" => Err(CudaMaSelectorError::Unsupported(
                "buff_averages returns dual outputs and requires volume; use ma_sweep_to_device_with_typed_params with output=fast|slow".into(),
            )),

            other => Err(CudaMaSelectorError::InvalidInput(format!(
                "unknown moving average type: {}",
                other
            ))),
        }
    }

    pub fn ma_to_device(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.ma_to_device_impl(ma_type, data, period, None)
    }

    pub fn ma_to_device_with_params(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
        params: &HashMap<String, f64>,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.ma_to_device_impl(ma_type, data, period, Some(params))
    }

    pub fn ma_to_host_f32(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<Vec<f32>, CudaMaSelectorError> {
        let dev = self.ma_to_device(ma_type, data, period)?;
        debug_assert_eq!(dev.rows, 1);

        let total = dev
            .rows
            .checked_mul(dev.cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total) }?;

        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaMaSelectorError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaMaSelectorError::Cuda)?;
        Ok(pinned.to_vec())
    }

    pub fn ma_to_host_f32_with_params(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
        params: &HashMap<String, f64>,
    ) -> Result<Vec<f32>, CudaMaSelectorError> {
        let dev = self.ma_to_device_impl(ma_type, data, period, Some(params))?;
        debug_assert_eq!(dev.rows, 1);

        let total = dev
            .rows
            .checked_mul(dev.cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total) }?;

        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaMaSelectorError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaMaSelectorError::Cuda)?;
        Ok(pinned.to_vec())
    }

    pub fn ma_to_host_f64(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<Vec<f64>, CudaMaSelectorError> {
        let out32 = self.ma_to_host_f32(ma_type, data, period)?;
        Ok(out32.into_iter().map(|v| v as f64).collect())
    }

    pub fn ma_to_host_f64_with_params(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
        params: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, CudaMaSelectorError> {
        let out32 = self.ma_to_host_f32_with_params(ma_type, data, period, params)?;
        Ok(out32.into_iter().map(|v| v as f64).collect())
    }

    fn ma_sweep_to_device_impl(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
        params: Option<&HashMap<String, f64>>,
        text_params: Option<&HashMap<String, String>>,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else if start < end {
            let s = step.max(1);
            (start..=end).step_by(s).collect()
        } else {
            let s = step.max(1);
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur == 0 {
                    break;
                }
                let next = cur.saturating_sub(s);
                if next == cur {
                    break;
                }
                cur = next;
                if cur < end {
                    break;
                }
            }
            v
        };
        if periods.is_empty() {
            return Err(CudaMaSelectorError::InvalidRange { start, end, step });
        }

        let ma_lc = ma_type.to_ascii_lowercase();

        let mut prices_owned: Option<Vec<f32>> = None;
        let prices: &[f32] = match data {
            CudaMaData::SliceF32(s) => s,
            CudaMaData::Slice(s) => {
                prices_owned = Some(s.iter().map(|&v| v as f32).collect());
                prices_owned.as_deref().unwrap()
            }
            CudaMaData::OhlcF32 { close, source, .. } => source.unwrap_or(close),
            CudaMaData::OhlcvF32 { close, source, .. } => source.unwrap_or(close),
            CudaMaData::Candles { candles, source } => {
                let src = source_type(candles, source);
                prices_owned = Some(src.iter().map(|&v| v as f32).collect());
                prices_owned.as_deref().unwrap()
            }
        };
        if prices.is_empty() {
            return Err(CudaMaSelectorError::InvalidInput(
                "empty price input".into(),
            ));
        }
        let period_range = (start, end, step);

        let rows = periods.len();
        let cols = prices.len();
        let elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_out = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("byte size overflow".into()))?;
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        will_fit(bytes_out, headroom)?;

        match ma_lc.as_str() {
            "vwma" => {
                let volumes = match data {
                    CudaMaData::Candles { candles, .. } => candles
                        .volume
                        .iter()
                        .map(|&v| v as f32)
                        .collect::<Vec<f32>>(),
                    CudaMaData::OhlcvF32 { volume, .. } => volume.to_vec(),
                    _ => {
                        return Err(CudaMaSelectorError::Unsupported(
                            "vwma requires volume input".into(),
                        ));
                    }
                };
                let sweep = crate::indicators::moving_averages::vwma::VwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaVwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.vwma_batch_dev(&prices, &volumes, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "vpwma" => {
                let sweep = crate::indicators::moving_averages::vpwma::VpwmaBatchRange {
                    period: period_range,
                    power: {
                        let p = get_param_f64(params, ma_type, "power")?.unwrap_or(0.382);
                        (p, p, 0.0)
                    },
                };
                let cuda = CudaVpwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .vpwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(DeviceArrayF32 {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                })
            }

            "sma" => {
                let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: period_range,
                };
                self.with_sma(|cuda| {
                    let (dev, _combos) = cuda
                        .sma_batch_dev(&prices, &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                    Ok(dev)
                })
            }
            "ema" => {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: period_range,
                };
                self.with_ema(|cuda| {
                    cuda.ema_batch_dev(&prices, &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
                })
            }
            "dema" => {
                let sweep = crate::indicators::moving_averages::dema::DemaBatchRange {
                    period: period_range,
                };
                self.with_dema(|cuda| {
                    cuda.dema_batch_dev(&prices, &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
                })
            }
            "wma" => {
                let sweep = crate::indicators::moving_averages::wma::WmaBatchRange {
                    period: period_range,
                };
                self.with_wma(|cuda| {
                    cuda.wma_batch_dev(&prices, &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
                })
            }
            "zlema" => {
                let sweep = crate::indicators::moving_averages::zlema::ZlemaBatchRange {
                    period: period_range,
                };
                self.with_zlema(|cuda| {
                    let (dev, _combos) = cuda
                        .zlema_batch_dev(&prices, &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                    Ok(dev)
                })
            }
            "smma" => {
                let sweep = crate::indicators::moving_averages::smma::SmmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSmma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.smma_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trima" => {
                let sweep = crate::indicators::moving_averages::trima::TrimaBatchRange {
                    period: period_range,
                };
                let cuda = CudaTrima::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.trima_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tema" => {
                let sweep = crate::indicators::moving_averages::tema::TemaBatchRange {
                    period: period_range,
                };
                let cuda = CudaTema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.tema_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tilson" => {
                let sweep = crate::indicators::moving_averages::tilson::TilsonBatchRange {
                    period: period_range,
                    volume_factor: {
                        let v = get_param_f64(params, ma_type, "volume_factor")?.unwrap_or(0.0);
                        (v, v, 0.0)
                    },
                };
                let cuda = CudaTilson::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.tilson_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "wilders" => {
                let sweep = crate::indicators::moving_averages::wilders::WildersBatchRange {
                    period: period_range,
                };
                let cuda = CudaWilders::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.wilders_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "cwma" => {
                let sweep = crate::indicators::moving_averages::cwma::CwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaCwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.cwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "jsa" => {
                let sweep = crate::indicators::moving_averages::jsa::JsaBatchRange {
                    period: period_range,
                };
                let cuda = CudaJsa::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.jsa_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "fwma" => {
                let sweep = crate::indicators::moving_averages::fwma::FwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaFwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.fwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "hma" => {
                let sweep = crate::indicators::moving_averages::hma::HmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaHma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .hma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "srwma" => {
                let sweep = crate::indicators::moving_averages::srwma::SrwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSrwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.srwma_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sinwma" => {
                let sweep = crate::indicators::moving_averages::sinwma::SinWmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSinwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sinwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sqwma" => {
                let sweep = crate::indicators::moving_averages::sqwma::SqwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSqwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sqwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sgf" => {
                let poly_order = get_param_usize(params, ma_type, "poly_order")?.unwrap_or(2);
                let sweep = crate::indicators::moving_averages::sgf::SgfBatchRange {
                    period: period_range,
                    poly_order: (poly_order, poly_order, 0),
                };
                let cuda = CudaSgf::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sgf_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "swma" => {
                let sweep = crate::indicators::moving_averages::swma::SwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.swma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "linreg" => {
                let sweep = crate::indicators::moving_averages::linreg::LinRegBatchRange {
                    period: period_range,
                };
                let cuda = CudaLinreg::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .linreg_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "edcf" => {
                let sweep = crate::indicators::moving_averages::edcf::EdcfBatchRange {
                    period: period_range,
                };
                let cuda = CudaEdcf::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.edcf_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "dma" => {
                let sweep = crate::indicators::moving_averages::dma::DmaBatchRange {
                    hull_length: period_range,
                    ema_length: {
                        let v = get_param_usize(params, ma_type, "ema_length")?.unwrap_or(20);
                        (v, v, 0)
                    },
                    ema_gain_limit: {
                        let v = get_param_usize(params, ma_type, "ema_gain_limit")?.unwrap_or(50);
                        (v, v, 0)
                    },
                    hull_ma_type: get_param_str(text_params, ma_type, "hull_ma_type")
                        .unwrap_or("WMA")
                        .to_string(),
                };
                let cuda = CudaDma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.dma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "highpass" => {
                let sweep = crate::indicators::moving_averages::highpass::HighPassBatchRange {
                    period: period_range,
                };
                let cuda = CudaHighpass::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.highpass_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "highpass2" | "highpass_2_pole" => {
                let k = get_param_f64(params, ma_type, "k")?.unwrap_or(0.707);
                let sweep =
                    crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange {
                        period: period_range,
                        k: (k, k, 0.0),
                    };
                let cuda = CudaHighPass2::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.highpass2_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "alma" => {
                let sweep = crate::indicators::moving_averages::alma::AlmaBatchRange {
                    period: period_range,
                    offset: {
                        let v = get_param_f64(params, ma_type, "offset")?.unwrap_or(0.85);
                        (v, v, 0.0)
                    },
                    sigma: {
                        let v = get_param_f64(params, ma_type, "sigma")?.unwrap_or(6.0);
                        (v, v, 0.0)
                    },
                };
                let cuda = CudaAlma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.alma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "epma" => {
                let sweep = crate::indicators::moving_averages::epma::EpmaBatchRange {
                    period: period_range,
                    offset: {
                        let v = get_param_usize(params, ma_type, "offset")?.unwrap_or(4);
                        (v, v, 0)
                    },
                };
                let cuda = CudaEpma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.epma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "gaussian" => {
                let sweep = crate::indicators::moving_averages::gaussian::GaussianBatchRange {
                    period: period_range,
                    poles: {
                        let v = get_param_usize(params, ma_type, "poles")?.unwrap_or(4);
                        (v, v, 0)
                    },
                };
                let cuda = CudaGaussian::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.gaussian_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "jma" => {
                let sweep = crate::indicators::moving_averages::jma::JmaBatchRange {
                    period: period_range,
                    phase: {
                        let v = get_param_f64(params, ma_type, "phase")?.unwrap_or(50.0);
                        (v, v, 0.0)
                    },
                    power: {
                        let v = get_param_u32(params, ma_type, "power")?.unwrap_or(2);
                        (v, v, 0)
                    },
                };
                let cuda = CudaJma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.jma_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehma" => {
                let sweep = crate::indicators::moving_averages::ehma::EhmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaEhma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "supersmoother" => {
                let sweep =
                    crate::indicators::moving_averages::supersmoother::SuperSmootherBatchRange {
                        period: period_range,
                    };
                let cuda = CudaSuperSmoother::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .supersmoother_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "supersmoother_3_pole" => {
                let sweep = crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange { period: period_range };
                let cuda = CudaSupersmoother3Pole::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.supersmoother_3_pole_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "kama" => {
                let sweep = crate::indicators::moving_averages::kama::KamaBatchRange {
                    period: period_range,
                };
                let cuda = CudaKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.kama_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sama" => {
                let mut sweep = crate::indicators::moving_averages::sama::SamaBatchRange {
                    length: period_range,
                    ..Default::default()
                };
                if let Some(v) = get_param_usize(params, ma_type, "maj_length")? {
                    sweep.maj_length = (v, v, 0);
                }
                if let Some(v) = get_param_usize(params, ma_type, "min_length")? {
                    sweep.min_length = (v, v, 0);
                }
                let cuda = CudaSama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sama_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_kama" => {
                let sweep = crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange {
                    period: period_range,
                };
                let cuda = CudaEhlersKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_kama_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_itrend" => {
                let warmup = get_param_usize(params, ma_type, "warmup_bars")?.unwrap_or(20);
                let sweep =
                    crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange {
                        warmup_bars: (warmup, warmup, 0),
                        max_dc_period: period_range,
                    };
                let cuda = CudaEhlersITrend::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_itrend_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_ecema" => {
                let gain_limit = get_param_usize(params, ma_type, "gain_limit")?.unwrap_or(50);
                let sweep =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaBatchRange {
                        length: period_range,
                        gain_limit: (gain_limit, gain_limit, 0),
                    };
                let params =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaParams::default();
                let cuda = CudaEhlersEcema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_ecema_batch_dev(&prices, &sweep, &params)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "nama" => {
                let sweep = crate::indicators::moving_averages::nama::NamaBatchRange {
                    period: period_range,
                };
                let cuda = CudaNama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.nama_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "nma" => {
                let sweep = crate::indicators::moving_averages::nma::NmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaNma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .nma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "pwma" => {
                let sweep = crate::indicators::moving_averages::pwma::PwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaPwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.pwma_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "maaq" => {
                let sweep = crate::indicators::moving_averages::maaq::MaaqBatchRange {
                    period: period_range,
                    fast_period: {
                        let v = get_param_usize(params, ma_type, "fast_period")?.unwrap_or(2);
                        (v, v, 0)
                    },
                    slow_period: {
                        let v = get_param_usize(params, ma_type, "slow_period")?.unwrap_or(30);
                        (v, v, 0)
                    },
                };
                let cuda = CudaMaaq::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.maaq_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "cora_wave" => {
                let r_multi = get_param_f64(params, ma_type, "r_multi")?.unwrap_or(2.0);
                if r_multi < 0.0 {
                    return Err(CudaMaSelectorError::InvalidInput(format!(
                        "invalid param 'r_multi' for '{ma_type}': expected >= 0, got {r_multi}"
                    )));
                }
                let smooth = get_param_bool01(params, ma_type, "smooth")?.unwrap_or(true);
                let sweep = crate::indicators::cora_wave::CoraWaveBatchRange {
                    period: period_range,
                    r_multi: (r_multi, r_multi, 0.0),
                    smooth,
                };
                let cuda = CudaCoraWave::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.cora_wave_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "mwdx" => {
                let first_valid = prices.iter().position(|x| !x.is_nan()).ok_or_else(|| {
                    CudaMaSelectorError::InvalidInput("all values are NaN".into())
                })?;
                let factors: Vec<f32> = if let Some(f) = get_param_f64(params, ma_type, "factor")? {
                    vec![f as f32; periods.len()]
                } else {
                    periods
                        .iter()
                        .map(|&p| 2.0f32 / (p as f32 + 1.0f32))
                        .collect()
                };
                let d_prices = DeviceBuffer::from_slice(&prices)?;
                let d_factors = DeviceBuffer::from_slice(&factors)?;
                let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };
                let cuda = CudaMwdx::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.mwdx_batch_device(&d_prices, &d_factors, cols, first_valid, rows, &mut d_out)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.synchronize()
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(DeviceArrayF32 {
                    buf: d_out,
                    rows,
                    cols,
                })
            }
            "reflex" => {
                let sweep = crate::indicators::moving_averages::reflex::ReflexBatchRange {
                    period: period_range,
                };
                let cuda = CudaReflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.reflex_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "volatility_adjusted_ma" | "vama" => {
                let sweep =
                    crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
                        base_period: period_range,
                        vol_period: {
                            let v = get_param_usize(params, ma_type, "vol_period")?.unwrap_or(51);
                            (v, v, 0)
                        },
                    };
                let cuda = CudaVama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.vama_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trendflex" => {
                let sweep = crate::indicators::moving_averages::trendflex::TrendFlexBatchRange {
                    period: period_range,
                };
                let cuda = CudaTrendflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .trendflex_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "hwma" => {
                let mut sweep = crate::indicators::moving_averages::hwma::HwmaBatchRange::default();
                if let Some(v) = get_param_f64(params, ma_type, "na")? {
                    sweep.na = (v, v, 0.0);
                }
                if let Some(v) = get_param_f64(params, ma_type, "nb")? {
                    sweep.nb = (v, v, 0.0);
                }
                if let Some(v) = get_param_f64(params, ma_type, "nc")? {
                    sweep.nc = (v, v, 0.0);
                }
                let cuda = CudaHwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.hwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tradjema" => {
                let mut sweep =
                    crate::indicators::moving_averages::tradjema::TradjemaBatchRange::default();
                sweep.length = period_range;
                if let Some(v) = get_param_f64(params, ma_type, "mult")? {
                    sweep.mult = (v, v, 0.0);
                }
                let cuda = CudaTradjema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                match data {
                    CudaMaData::Candles { candles, .. } => {
                        let high: Vec<f32> = candles.high.iter().map(|&v| v as f32).collect();
                        let low: Vec<f32> = candles.low.iter().map(|&v| v as f32).collect();
                        let close: Vec<f32> = candles.close.iter().map(|&v| v as f32).collect();
                        cuda.tradjema_batch_dev(&high, &low, &close, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
                    }
                    CudaMaData::OhlcF32 {
                        high, low, close, ..
                    } => cuda
                        .tradjema_batch_dev(high, low, close, &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string())),
                    CudaMaData::OhlcvF32 {
                        high, low, close, ..
                    } => cuda
                        .tradjema_batch_dev(high, low, close, &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string())),
                    _ => Err(CudaMaSelectorError::Unsupported(
                        "tradjema requires high/low/close input".into(),
                    )),
                }
            }
            "uma" => {
                let mut sweep = crate::indicators::moving_averages::uma::UmaBatchRange::default();
                sweep.max_length = period_range;
                if let Some(v) = get_param_f64(params, ma_type, "accelerator")? {
                    sweep.accelerator = (v, v, 0.0);
                }
                if let Some(v) = get_param_usize(params, ma_type, "min_length")? {
                    sweep.min_length = (v, v, 0);
                }
                if let Some(v) = get_param_usize(params, ma_type, "max_length")? {
                    sweep.max_length = (v, v, 0);
                }
                if let Some(v) = get_param_usize(params, ma_type, "smooth_length")? {
                    sweep.smooth_length = (v, v, 0);
                }
                let volumes = match data {
                    CudaMaData::Candles { candles, .. } => Some(
                        candles
                            .volume
                            .iter()
                            .map(|&v| v as f32)
                            .collect::<Vec<f32>>(),
                    ),
                    CudaMaData::OhlcvF32 { volume, .. } => Some(volume.to_vec()),
                    _ => None,
                };
                let volumes_ref = volumes.as_deref();
                let cuda = CudaUma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.uma_batch_dev(&prices, volumes_ref, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "volume_adjusted_ma" => {
                let volumes = match data {
                    CudaMaData::Candles { candles, .. } => candles
                        .volume
                        .iter()
                        .map(|&v| v as f32)
                        .collect::<Vec<f32>>(),
                    CudaMaData::OhlcvF32 { volume, .. } => volume.to_vec(),
                    _ => {
                        return Err(CudaMaSelectorError::Unsupported(
                            "volume_adjusted_ma requires volume input".into(),
                        ));
                    }
                };
                let mut sweep =
                    crate::indicators::moving_averages::volume_adjusted_ma::VolumeAdjustedMaBatchRange::default();
                sweep.length = period_range;
                if let Some(v) = get_param_f64(params, ma_type, "vi_factor")? {
                    sweep.vi_factor = (v, v, 0.0);
                }
                if let Some(v) = get_param_usize(params, ma_type, "sample_period")? {
                    sweep.sample_period = (v, v, 0);
                }
                if let Some(v) = get_param_bool01(params, ma_type, "strict")? {
                    sweep.strict = Some(v);
                }
                let cuda = CudaVolumeAdjustedMa::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.volume_adjusted_ma_batch_dev(&prices, &volumes, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "vwap" => {
                let single_anchor = get_param_str(text_params, ma_type, "anchor");
                let anchor_start = get_param_str(text_params, ma_type, "anchor_start")
                    .or(single_anchor)
                    .unwrap_or("1d")
                    .to_string();
                let anchor_end = get_param_str(text_params, ma_type, "anchor_end")
                    .or(single_anchor)
                    .unwrap_or(anchor_start.as_str())
                    .to_string();
                let anchor_step = get_param_usize(params, ma_type, "anchor_step")?
                    .map(|v| v as u32)
                    .unwrap_or_else(|| if anchor_start == anchor_end { 0 } else { 1 });
                let sweep = crate::indicators::moving_averages::vwap::VwapBatchRange {
                    anchor: (anchor_start, anchor_end, anchor_step),
                };
                let cuda = CudaVwap::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                match data {
                    CudaMaData::Candles { candles, .. } => {
                        let prices_f64: Vec<f64> = prices.iter().map(|&v| v as f64).collect();
                        cuda.vwap_batch_dev(
                            &candles.timestamp,
                            &candles.volume,
                            &prices_f64,
                            &sweep,
                        )
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
                    }
                    CudaMaData::OhlcvF32 {
                        timestamp: Some(timestamp),
                        volume,
                        ..
                    } => cuda
                        .vwap_batch_dev_f32(timestamp, volume, prices, &sweep)
                        .map_err(|e| CudaMaSelectorError::Backend(e.to_string())),
                    _ => {
                        return Err(CudaMaSelectorError::Unsupported(
                            "vwap requires timestamp + volume + source input".into(),
                        ));
                    }
                }
            }
            "mama" => {
                let mut sweep = crate::indicators::moving_averages::mama::MamaBatchRange::default();
                let has_fast_override = params
                    .and_then(|p| {
                        p.get("fast_limit")
                            .or_else(|| p.get("fast_limit_start"))
                            .or_else(|| p.get("fast_limit_end"))
                            .or_else(|| p.get("fast_limit_step"))
                    })
                    .is_some();

                if let Some(v) = get_param_f64(params, ma_type, "fast_limit")? {
                    sweep.fast_limit = (v, v, 0.0);
                } else if has_fast_override {
                    if let Some(v) = get_param_f64(params, ma_type, "fast_limit_start")? {
                        sweep.fast_limit.0 = v;
                    }
                    if let Some(v) = get_param_f64(params, ma_type, "fast_limit_end")? {
                        sweep.fast_limit.1 = v;
                    }
                    if let Some(v) = get_param_f64(params, ma_type, "fast_limit_step")? {
                        sweep.fast_limit.2 = v;
                    }
                } else {
                    let fast_start = 2.0 / (period_range.0 as f64 + 1.0);
                    let fast_end = 2.0 / (period_range.1 as f64 + 1.0);
                    let next_period = if period_range.2 == 0 || period_range.0 == period_range.1 {
                        period_range.0
                    } else if period_range.0 < period_range.1 {
                        period_range.0.saturating_add(period_range.2)
                    } else {
                        period_range.0.saturating_sub(period_range.2)
                    };
                    let fast_next = 2.0 / (next_period as f64 + 1.0);
                    let fast_step = (fast_next - fast_start).abs();
                    sweep.fast_limit = (fast_start, fast_end, fast_step);
                }

                if let Some(v) = get_param_f64(params, ma_type, "slow_limit")? {
                    sweep.slow_limit = (v, v, 0.0);
                } else {
                    if let Some(v) = get_param_f64(params, ma_type, "slow_limit_start")? {
                        sweep.slow_limit.0 = v;
                    }
                    if let Some(v) = get_param_f64(params, ma_type, "slow_limit_end")? {
                        sweep.slow_limit.1 = v;
                    }
                    if let Some(v) = get_param_f64(params, ma_type, "slow_limit_step")? {
                        sweep.slow_limit.2 = v;
                    }
                }

                let output = get_param_str(text_params, ma_type, "output")
                    .unwrap_or("mama")
                    .to_ascii_lowercase();
                let cuda = CudaMama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let pair = cuda
                    .mama_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                match output.as_str() {
                    "mama" => Ok(pair.mama),
                    "fama" => Ok(pair.fama),
                    _ => Err(CudaMaSelectorError::InvalidInput(format!(
                        "invalid param 'output' for '{ma_type}': expected 'mama' or 'fama'"
                    ))),
                }
            }

            "frama" => {
                let sc = get_param_usize(params, ma_type, "sc")?.unwrap_or(300);
                let fc = get_param_usize(params, ma_type, "fc")?.unwrap_or(1);
                let sweep = crate::indicators::moving_averages::frama::FramaBatchRange {
                    window: period_range,
                    sc: (sc, sc, 0),
                    fc: (fc, fc, 0),
                };
                let cuda = CudaFrama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                match data {
                    CudaMaData::Candles { candles, .. } => {
                        let high_f32: Vec<f32> = candles.high.iter().map(|&v| v as f32).collect();
                        let low_f32: Vec<f32> = candles.low.iter().map(|&v| v as f32).collect();
                        let close_f32: Vec<f32> = candles.close.iter().map(|&v| v as f32).collect();
                        let (dev, _combos) = cuda
                            .frama_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                    CudaMaData::SliceF32(s) => {
                        let (dev, _combos) = cuda
                            .frama_batch_dev(s, s, s, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                    CudaMaData::Slice(s) => {
                        let prices_f32: Vec<f32> = s.iter().map(|&v| v as f32).collect();
                        let (dev, _combos) = cuda
                            .frama_batch_dev(&prices_f32, &prices_f32, &prices_f32, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                    CudaMaData::OhlcF32 {
                        high, low, close, ..
                    } => {
                        let (dev, _combos) = cuda
                            .frama_batch_dev(high, low, close, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                    CudaMaData::OhlcvF32 {
                        high, low, close, ..
                    } => {
                        let (dev, _combos) = cuda
                            .frama_batch_dev(high, low, close, &sweep)
                            .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                        Ok(dev)
                    }
                }
            }
            "buff_averages" => {
                let volumes = match data {
                    CudaMaData::Candles { candles, .. } => candles
                        .volume
                        .iter()
                        .map(|&v| v as f32)
                        .collect::<Vec<f32>>(),
                    CudaMaData::OhlcvF32 { volume, .. } => volume.to_vec(),
                    _ => {
                        return Err(CudaMaSelectorError::Unsupported(
                            "buff_averages requires volume input".into(),
                        ));
                    }
                };
                let mut sweep =
                    crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange::default();
                sweep.slow_period = period_range;
                if let Some(v) = get_param_usize(params, ma_type, "fast_period")? {
                    sweep.fast_period = (v, v, 0);
                }
                if let Some(v) = get_param_usize(params, ma_type, "slow_period")? {
                    sweep.slow_period = (v, v, 0);
                }
                if let Some(v) = get_param_usize(params, ma_type, "fast_period_start")? {
                    sweep.fast_period.0 = v;
                }
                if let Some(v) = get_param_usize(params, ma_type, "fast_period_end")? {
                    sweep.fast_period.1 = v;
                }
                if let Some(v) = get_param_usize(params, ma_type, "fast_period_step")? {
                    sweep.fast_period.2 = v;
                }
                if let Some(v) = get_param_usize(params, ma_type, "slow_period_start")? {
                    sweep.slow_period.0 = v;
                }
                if let Some(v) = get_param_usize(params, ma_type, "slow_period_end")? {
                    sweep.slow_period.1 = v;
                }
                if let Some(v) = get_param_usize(params, ma_type, "slow_period_step")? {
                    sweep.slow_period.2 = v;
                }

                let output = get_param_str(text_params, ma_type, "output")
                    .unwrap_or("fast")
                    .to_ascii_lowercase();
                let cuda = CudaBuffAverages::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let pair = cuda
                    .buff_averages_batch_dev(&prices, &volumes, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                match output.as_str() {
                    "fast" | "fast_buff" => Ok(pair.0),
                    "slow" | "slow_buff" => Ok(pair.1),
                    _ => Err(CudaMaSelectorError::InvalidInput(format!(
                        "invalid param 'output' for '{ma_type}': expected 'fast' or 'slow'"
                    ))),
                }
            }
            "ehlers_pma" => {
                let sweep = crate::indicators::moving_averages::ehlers_pma::EhlersPmaBatchRange {
                    combos: periods.len(),
                };
                let output = get_param_str(text_params, ma_type, "output")
                    .unwrap_or("predict")
                    .to_ascii_lowercase();
                let cuda = CudaEhlersPma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let pair = cuda
                    .ehlers_pma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                match output.as_str() {
                    "predict" => Ok(pair.predict),
                    "trigger" => Ok(pair.trigger),
                    _ => Err(CudaMaSelectorError::InvalidInput(format!(
                        "invalid param 'output' for '{ma_type}': expected 'predict' or 'trigger'"
                    ))),
                }
            }

            other => Err(CudaMaSelectorError::InvalidInput(format!(
                "ma_sweep_to_device unsupported for {}",
                other
            ))),
        }
    }

    fn ma_sweep_to_device_ref_impl(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        start: usize,
        end: usize,
        step: usize,
        params: Option<&HashMap<String, f64>>,
        _text_params: Option<&HashMap<String, String>>,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        let periods = build_sweep_periods(start, end, step)?;
        let periods_i32 = periods_to_i32(periods.as_slice())?;
        self.with_cached_sweep_periods_i32(periods_i32.as_ref(), |d_periods| {
            self.ma_sweep_periods_to_device_ref_impl(
                ma_type,
                data,
                first_valid,
                periods.as_slice(),
                periods_i32.as_ref(),
                d_periods,
                params,
            )
        })
    }

    fn ma_sweep_periods_to_device_ref_impl(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        periods: &[usize],
        periods_i32: &[i32],
        d_periods: &DeviceBuffer<i32>,
        params: Option<&HashMap<String, f64>>,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        let series_len = data.prices_len();
        if series_len == 0 {
            return Err(CudaMaSelectorError::InvalidInput(
                "empty price input".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaMaSelectorError::InvalidInput(format!(
                "invalid first_valid: {} for length {}",
                first_valid, series_len
            )));
        }
        for &period in periods {
            if period == 0 || period > series_len {
                return Err(CudaMaSelectorError::InvalidInput(format!(
                    "invalid period: {} for length {}",
                    period, series_len
                )));
            }
        }

        let device_id = data.device_id();
        if device_id != self.device_id as u32 {
            return Err(CudaMaSelectorError::DeviceMismatch {
                buf: device_id,
                current: self.device_id as u32,
            });
        }

        let ma_lc = ma_type.trim().to_ascii_lowercase();
        if !super::vram_ma::supports_vram_kernel_ma(&ma_lc) {
            return Err(CudaMaSelectorError::Unsupported(format!(
                "device-native selector path is not available for '{}'",
                ma_type
            )));
        }

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(periods.len() * series_len) }?;
        let borrowed = unsafe { BorrowedCudaMaInputs::from_data(data)? };
        let inputs = borrowed.as_vram_inputs();
        self.with_vram_ma(|computer| {
            if ma_lc.eq_ignore_ascii_case("sma") {
                computer
                    .ensure_sma_prefix_f64(inputs.prices, series_len, first_valid)
                    .map_err(CudaMaSelectorError::Backend)?;
            }
            if ma_lc.eq_ignore_ascii_case("vwma") {
                let volume = inputs.volume.ok_or_else(|| {
                    CudaMaSelectorError::Unsupported(
                        "vwma requires volume input for device-native selector path".into(),
                    )
                })?;
                computer
                    .ensure_vwma_prefix_pv_vol_f64(inputs.prices, volume, series_len, first_valid)
                    .map_err(CudaMaSelectorError::Backend)?;
            }

            computer
                .compute_period_ma_into(
                    ma_type,
                    params,
                    &inputs,
                    series_len,
                    first_valid,
                    periods_i32,
                    d_periods,
                    &mut d_out,
                )
                .map_err(CudaMaSelectorError::Backend)
        })?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: periods.len(),
            cols: series_len,
        })
    }

    pub fn ma_sweep_to_device(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.ma_sweep_to_device_impl(ma_type, data, start, end, step, None, None)
    }

    pub fn ma_sweep_to_device_with_params(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
        params: &HashMap<String, f64>,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.ma_sweep_to_device_impl(ma_type, data, start, end, step, Some(params), None)
    }

    pub fn ma_sweep_to_device_with_typed_params(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
        params: &[CudaMaParamKV<'_>],
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        let (numeric, text) = typed_params_to_maps(ma_type, params)?;
        self.ma_sweep_to_device_impl(ma_type, data, start, end, step, Some(&numeric), Some(&text))
    }

    pub fn ma_to_device_ref(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.ma_sweep_to_device_ref_impl(ma_type, data, first_valid, period, period, 0, None, None)
    }

    pub fn ma_sweep_to_device_ref(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.ma_sweep_to_device_ref_impl(ma_type, data, first_valid, start, end, step, None, None)
    }

    pub fn ma_sweep_to_device_ref_with_typed_params(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        start: usize,
        end: usize,
        step: usize,
        params: &[CudaMaParamKV<'_>],
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        let (numeric, text) = typed_params_to_maps(ma_type, params)?;
        self.ma_sweep_to_device_ref_impl(
            ma_type,
            data,
            first_valid,
            start,
            end,
            step,
            Some(&numeric),
            Some(&text),
        )
    }

    pub fn ma_sweep_plan_to_device_ref(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        plan: &CudaMaSweepPlan,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.ma_sweep_plan_to_device_ref_with_params(ma_type, data, first_valid, plan, None)
    }

    pub fn ma_sweep_plan_to_device_ref_with_params(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        plan: &CudaMaSweepPlan,
        params: Option<&HashMap<String, f64>>,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        if plan.device_id != self.device_id as u32 {
            return Err(CudaMaSelectorError::DeviceMismatch {
                buf: plan.device_id,
                current: self.device_id as u32,
            });
        }
        self.ma_sweep_periods_to_device_ref_impl(
            ma_type,
            data,
            first_valid,
            plan.periods.as_ref(),
            plan.periods_i32.as_ref(),
            &plan.d_periods,
            params,
        )
    }

    pub fn ma_sweep_plan_to_device_ref_with_typed_params(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        plan: &CudaMaSweepPlan,
        params: &[CudaMaParamKV<'_>],
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        let (numeric, _text) = typed_params_to_maps(ma_type, params)?;
        self.ma_sweep_plan_to_device_ref_with_params(
            ma_type,
            data,
            first_valid,
            plan,
            Some(&numeric),
        )
    }

    pub fn ma_sweep_to_host_f32(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<(Vec<f32>, usize, usize), CudaMaSelectorError> {
        let dev = self.ma_sweep_to_device(ma_type, data, start, end, step)?;
        let total = dev
            .rows
            .checked_mul(dev.cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total) }?;
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaMaSelectorError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaMaSelectorError::Cuda)?;
        Ok((pinned.to_vec(), dev.rows, dev.cols))
    }

    pub fn ma_sweep_to_host_f32_with_params(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
        params: &HashMap<String, f64>,
    ) -> Result<(Vec<f32>, usize, usize), CudaMaSelectorError> {
        let dev =
            self.ma_sweep_to_device_impl(ma_type, data, start, end, step, Some(params), None)?;
        let total = dev
            .rows
            .checked_mul(dev.cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total) }?;
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaMaSelectorError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaMaSelectorError::Cuda)?;
        Ok((pinned.to_vec(), dev.rows, dev.cols))
    }

    pub fn ma_sweep_to_host_f64(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<(Vec<f64>, usize, usize), CudaMaSelectorError> {
        let (out32, rows, cols) = self.ma_sweep_to_host_f32(ma_type, data, start, end, step)?;
        Ok((out32.into_iter().map(|v| v as f64).collect(), rows, cols))
    }

    pub fn ma_sweep_to_host_f64_with_params(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
        params: &HashMap<String, f64>,
    ) -> Result<(Vec<f64>, usize, usize), CudaMaSelectorError> {
        let (out32, rows, cols) =
            self.ma_sweep_to_host_f32_with_params(ma_type, data, start, end, step, params)?;
        Ok((out32.into_iter().map(|v| v as f64).collect(), rows, cols))
    }
}

impl<'a> CudaMaDeviceSelector<'a> {
    #[inline]
    pub fn create_sweep_plan(
        &self,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<CudaMaSweepPlan, CudaMaSelectorError> {
        self.selector.create_sweep_plan(start, end, step)
    }

    #[inline]
    pub fn ma_to_device_ref(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.selector
            .ma_to_device_ref(ma_type, data, first_valid, period)
    }

    #[inline]
    pub fn ma_sweep_to_device_ref(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.selector
            .ma_sweep_to_device_ref(ma_type, data, first_valid, start, end, step)
    }

    #[inline]
    pub fn ma_sweep_to_device_ref_with_typed_params(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        start: usize,
        end: usize,
        step: usize,
        params: &[CudaMaParamKV<'_>],
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.selector.ma_sweep_to_device_ref_with_typed_params(
            ma_type,
            data,
            first_valid,
            start,
            end,
            step,
            params,
        )
    }

    #[inline]
    pub fn ma_sweep_plan_to_device_ref(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        plan: &CudaMaSweepPlan,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.selector
            .ma_sweep_plan_to_device_ref(ma_type, data, first_valid, plan)
    }

    #[inline]
    pub fn ma_sweep_plan_to_device_ref_with_typed_params(
        &self,
        ma_type: &str,
        data: CudaMaDeviceDataRef<'_>,
        first_valid: usize,
        plan: &CudaMaSweepPlan,
        params: &[CudaMaParamKV<'_>],
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        self.selector.ma_sweep_plan_to_device_ref_with_typed_params(
            ma_type,
            data,
            first_valid,
            plan,
            params,
        )
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;

#[cfg(all(feature = "python", feature = "cuda"))]
pub struct DeviceArrayF32Sel {
    pub buf: cust::memory::DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl DeviceArrayF32Sel {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "vector_ta", unsendable)]
pub struct DeviceArrayF32PySel {
    inner: Option<DeviceArrayF32Sel>,
    device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32PySel {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;

        let row_stride = inner
            .cols
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| PyValueError::new_err("byte stride overflow"))?;
        d.set_item("strides", (row_stride, std::mem::size_of::<f32>()))?;
        d.set_item("data", (inner.device_ptr() as usize, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

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

        let _ = stream;

        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
fn not_empty_f32(arr: PyReadonlyArray1<'_, f32>) -> PyResult<Vec<f32>> {
    let s = arr.as_slice()?;
    if s.is_empty() {
        Err(PyValueError::new_err("empty data"))
    } else {
        Ok(s.to_vec())
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ma_selector_cuda_to_device")]
#[pyo3(signature = (ma_type, data, period, device_id=0))]
pub fn ma_selector_cuda_to_device_py(
    py: Python<'_>,
    ma_type: &str,
    data: PyReadonlyArray1<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32PySel> {
    let prices = not_empty_f32(data)?;
    let is = |s: &str| ma_type.eq_ignore_ascii_case(s);
    let inner = py
        .allow_threads(|| -> Result<DeviceArrayF32Sel, String> {
            if is("sma") {
                let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSma::new(device_id).map_err(|e| e.to_string())?;
                let ctx = cuda.context_arc_clone();
                let dev_id = cuda.device_id();
                let (dev, _c) = cuda
                    .sma_batch_dev(&prices, &sweep)
                    .map_err(|e| e.to_string())?;
                return Ok(DeviceArrayF32Sel {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                    ctx,
                    device_id: dev_id,
                });
            }
            if is("ema") {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEma::new(device_id).map_err(|e| e.to_string())?;
                let ctx = cuda.context_arc();
                let dev = cuda
                    .ema_batch_dev(&prices, &sweep)
                    .map_err(|e| e.to_string())?;
                let dev_id = device_id as u32;
                return Ok(DeviceArrayF32Sel {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                    ctx,
                    device_id: dev_id,
                });
            }
            Err(format!("unsupported MA type: {}", ma_type))
        })
        .map_err(PyValueError::new_err)?;
    let device_id = inner.device_id;
    Ok(DeviceArrayF32PySel {
        inner: Some(inner),
        device_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ma_selector_cuda_sweep_to_device")]
#[pyo3(signature = (ma_type, data, period_range, device_id=0))]
pub fn ma_selector_cuda_sweep_to_device_py(
    py: Python<'_>,
    ma_type: &str,
    data: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32PySel> {
    let prices = not_empty_f32(data)?;
    let is = |s: &str| ma_type.eq_ignore_ascii_case(s);
    let inner = py
        .allow_threads(|| -> Result<DeviceArrayF32Sel, String> {
            if is("sma") {
                let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSma::new(device_id).map_err(|e| e.to_string())?;
                let ctx = cuda.context_arc_clone();
                let dev_id = cuda.device_id();
                let (dev, _c) = cuda
                    .sma_batch_dev(&prices, &sweep)
                    .map_err(|e| e.to_string())?;
                return Ok(DeviceArrayF32Sel {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                    ctx,
                    device_id: dev_id,
                });
            }
            if is("ema") {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaEma::new(device_id).map_err(|e| e.to_string())?;
                let ctx = cuda.context_arc();
                let dev = cuda
                    .ema_batch_dev(&prices, &sweep)
                    .map_err(|e| e.to_string())?;
                let dev_id = device_id as u32;
                return Ok(DeviceArrayF32Sel {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                    ctx,
                    device_id: dev_id,
                });
            }
            Err(format!("ma_sweep_to_device unsupported for {}", ma_type))
        })
        .map_err(PyValueError::new_err)?;
    let device_id = inner.device_id;
    Ok(DeviceArrayF32PySel {
        inner: Some(inner),
        device_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::Candles;
    use cust::memory::CopyDestination;

    fn sample_prices(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| ((i as f64) * 0.1).sin() + (i as f64) * 0.001 + 100.0)
            .collect()
    }

    fn sample_candles(len: usize) -> Candles {
        let timestamp: Vec<i64> = (0..len)
            .map(|i| 1_700_000_000_000_i64 + (i as i64) * 60_000)
            .collect();
        let close = sample_prices(len);
        let open: Vec<f64> = close.iter().map(|v| v - 0.1).collect();
        let high: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.35 + ((i as f64) * 0.01).sin().abs())
            .collect();
        let low: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, v)| v - 0.35 - ((i as f64) * 0.01).sin().abs())
            .collect();
        let volume: Vec<f64> = (0..len)
            .map(|i| 1000.0 + ((i % 31) as f64) * 7.0 + (i as f64) * 0.1)
            .collect();
        Candles::new(timestamp, open, high, low, close, volume)
    }

    fn assert_series_eq_f32_f64(a: &[f32], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            if av.is_nan() && bv.is_nan() {
                continue;
            }
            let d = (av as f64 - bv).abs();
            assert!(
                d <= tol,
                "series mismatch at index {i}: left={av}, right={bv}, abs_diff={d}"
            );
        }
    }

    fn assert_series_eq_f32(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            if av.is_nan() && bv.is_nan() {
                continue;
            }
            let d = (av - bv).abs();
            assert!(
                d <= tol,
                "series mismatch at index {i}: left={av}, right={bv}, abs_diff={d}"
            );
        }
    }

    fn err_string<T, E: ToString>(res: Result<T, E>) -> String {
        match res {
            Ok(_) => panic!("expected error result"),
            Err(e) => e.to_string(),
        }
    }

    fn repo_source(path: &str) -> String {
        let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        std::fs::read_to_string(&full)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", full.display(), e))
    }

    fn fn_body<'a>(source: &'a str, fn_name: &str) -> &'a str {
        let marker = format!("fn {fn_name}");
        let start = source
            .find(&marker)
            .unwrap_or_else(|| panic!("function {fn_name} not found"));
        let body_start = source[start..]
            .find('{')
            .map(|offset| start + offset)
            .unwrap_or_else(|| panic!("function {fn_name} has no body"));
        let mut depth = 0usize;
        for (offset, ch) in source[body_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &source[body_start..=body_start + offset];
                    }
                }
                _ => {}
            }
        }
        panic!("function {fn_name} body did not terminate");
    }

    fn assert_fn_body_avoids(path: &str, fn_name: &str, needle: &str) {
        let source = repo_source(path);
        let body = fn_body(&source, fn_name);
        assert!(
            !body.contains(needle),
            "found `{}` inside {}:{}",
            needle,
            path,
            fn_name
        );
    }

    fn assert_fn_body_contains(path: &str, fn_name: &str, needle: &str) {
        let source = repo_source(path);
        let body = fn_body(&source, fn_name);
        assert!(
            body.contains(needle),
            "expected `{}` inside {}:{}",
            needle,
            path,
            fn_name
        );
    }

    #[test]
    fn cuda_selector_borrowed_device_ref_path_avoids_host_materialization() {
        let path = "src/cuda/moving_averages/ma_selector.rs";
        let fn_name = "ma_sweep_to_device_ref_impl";
        for needle in [
            "to_prices_f32",
            "to_vec()",
            "copy_to(",
            "async_copy_to(",
            "async_copy_from(",
            "LockedBuffer::from_slice",
        ] {
            assert_fn_body_avoids(path, fn_name, needle);
        }
    }

    #[test]
    fn cuda_selector_borrowed_device_ref_path_avoids_rebuilding_vram_ma_per_call() {
        let path = "src/cuda/moving_averages/ma_selector.rs";
        let fn_name = "ma_sweep_to_device_ref_impl";
        assert_fn_body_avoids(path, fn_name, "VramMaComputer::new(");
    }

    #[test]
    fn cuda_selector_borrowed_device_ref_path_avoids_direct_period_reupload() {
        let path = "src/cuda/moving_averages/ma_selector.rs";
        let fn_name = "ma_sweep_to_device_ref_impl";
        assert_fn_body_avoids(
            path,
            fn_name,
            "DeviceBuffer::from_slice(periods_i32.as_slice())",
        );
    }

    #[test]
    fn borrowed_device_selector_consumers_avoid_host_shaped_selector_entrypoints() {
        for (path, fn_name) in [
            (
                "src/cuda/eri_wrapper.rs",
                "eri_batch_dev_from_device_inputs",
            ),
            (
                "src/cuda/kaufmanstop_wrapper.rs",
                "kaufmanstop_batch_dev_from_device_inputs",
            ),
            (
                "src/cuda/stoch_wrapper.rs",
                "stoch_batch_dev_from_device_ptrs",
            ),
            (
                "src/cuda/moving_averages/ott_wrapper.rs",
                "ott_batch_dev_from_device_prices",
            ),
            (
                "src/cuda/moving_averages/mab_wrapper.rs",
                "mab_batch_dev_from_device_prices",
            ),
        ] {
            assert_fn_body_avoids(path, fn_name, "ma_to_device(");
            assert_fn_body_avoids(path, fn_name, "ma_sweep_to_device(");
        }
    }

    #[test]
    fn borrowed_device_selector_consumers_use_device_native_selector_surface() {
        for (path, fn_name) in [
            (
                "src/cuda/eri_wrapper.rs",
                "eri_batch_dev_from_device_inputs",
            ),
            (
                "src/cuda/kaufmanstop_wrapper.rs",
                "kaufmanstop_batch_dev_from_device_inputs",
            ),
            (
                "src/cuda/moving_averages/ott_wrapper.rs",
                "ott_batch_dev_from_device_prices",
            ),
            (
                "src/cuda/moving_averages/mab_wrapper.rs",
                "mab_batch_dev_from_device_prices",
            ),
            (
                "src/cuda/stoch_wrapper.rs",
                "stoch_batch_dev_from_device_ptrs",
            ),
        ] {
            assert_fn_body_contains(path, fn_name, "device_native()");
        }
    }

    #[test]
    fn cuda_mama_typed_output_selection_matches_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(256);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "fast_limit",
                value: CudaMaParamValue::Float(0.35),
            },
            CudaMaParamKV {
                key: "slow_limit",
                value: CudaMaParamValue::Float(0.06),
            },
            CudaMaParamKV {
                key: "output",
                value: CudaMaParamValue::EnumString("fama"),
            },
        ];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "mama",
                CudaMaData::Slice(&prices),
                10,
                10,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct = crate::indicators::moving_averages::mama::mama_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::mama::MamaBatchRange {
                fast_limit: (0.35, 0.35, 0.0),
                slow_limit: (0.06, 0.06, 0.0),
            },
            crate::utilities::enums::Kernel::Auto,
        )
        .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.fama_values, 1e-3);
    }

    #[test]
    fn cuda_ehlers_pma_typed_output_selection_matches_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(300);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "output",
            value: CudaMaParamValue::EnumString("trigger"),
        }];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "ehlers_pma",
                CudaMaData::Slice(&prices),
                8,
                10,
                1,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let input = crate::indicators::moving_averages::ehlers_pma::EhlersPmaInput::from_slice(
            &prices,
            crate::indicators::moving_averages::ehlers_pma::EhlersPmaParams::default(),
        );
        let direct = crate::indicators::moving_averages::ehlers_pma::ehlers_pma_with_kernel(
            &input,
            crate::utilities::enums::Kernel::Auto,
        )
        .unwrap();

        assert_eq!(dev.rows, 3);
        assert_eq!(dev.cols, prices.len());
        for row in 0..dev.rows {
            let start = row * dev.cols;
            let end = start + dev.cols;
            assert_series_eq_f32_f64(&got[start..end], &direct.trigger, 1e-3);
        }
    }

    #[test]
    fn cuda_hwma_typed_params_match_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(256);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "na",
                value: CudaMaParamValue::Float(0.23),
            },
            CudaMaParamKV {
                key: "nb",
                value: CudaMaParamValue::Float(0.11),
            },
            CudaMaParamKV {
                key: "nc",
                value: CudaMaParamValue::Float(0.17),
            },
        ];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "hwma",
                CudaMaData::Slice(&prices),
                10,
                10,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct = crate::indicators::moving_averages::hwma::hwma_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::hwma::HwmaBatchRange {
                na: (0.23, 0.23, 0.0),
                nb: (0.11, 0.11, 0.0),
                nc: (0.17, 0.17, 0.0),
            },
            crate::utilities::enums::Kernel::Auto,
        )
        .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.values, 1e-3);
    }

    #[test]
    fn cuda_mwdx_typed_factor_matches_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(256);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "factor",
            value: CudaMaParamValue::Float(2.0 / 11.0),
        }];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "mwdx",
                CudaMaData::Slice(&prices),
                10,
                10,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct = crate::indicators::moving_averages::mwdx::mwdx_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::mwdx::MwdxBatchRange {
                factor: (2.0 / 11.0, 2.0 / 11.0, 0.0),
            },
            crate::utilities::enums::Kernel::Auto,
        )
        .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.values, 1e-3);
    }

    #[test]
    fn cuda_uma_typed_params_match_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(256);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "accelerator",
                value: CudaMaParamValue::Float(1.0),
            },
            CudaMaParamKV {
                key: "min_length",
                value: CudaMaParamValue::Int(5),
            },
            CudaMaParamKV {
                key: "max_length",
                value: CudaMaParamValue::Int(35),
            },
            CudaMaParamKV {
                key: "smooth_length",
                value: CudaMaParamValue::Int(4),
            },
        ];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "uma",
                CudaMaData::Slice(&prices),
                35,
                35,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct_sweep = crate::indicators::moving_averages::uma::UmaBatchRange {
            accelerator: (1.0, 1.0, 0.0),
            min_length: (5, 5, 0),
            max_length: (35, 35, 0),
            smooth_length: (4, 4, 0),
        };
        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let cuda = CudaUma::new(0).unwrap();
        let direct_dev = cuda
            .uma_batch_dev(&prices_f32, None, &direct_sweep)
            .unwrap();
        let mut direct = vec![0f32; direct_dev.rows * direct_dev.cols];
        direct_dev.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, direct_dev.rows);
        assert_eq!(dev.cols, direct_dev.cols);
        assert_series_eq_f32(&got, &direct, 1e-5);
    }

    #[test]
    fn cuda_tradjema_typed_params_match_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "mult",
            value: CudaMaParamValue::Float(2.3),
        }];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "tradjema",
                CudaMaData::Candles {
                    candles: &candles,
                    source: "close",
                },
                40,
                40,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct = crate::indicators::moving_averages::tradjema::tradjema_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            &crate::indicators::moving_averages::tradjema::TradjemaBatchRange {
                length: (40, 40, 0),
                mult: (2.3, 2.3, 0.0),
            },
            crate::utilities::enums::Kernel::Auto,
        )
        .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.values, 5e-2);
    }

    #[test]
    fn cuda_volume_adjusted_ma_typed_params_match_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "vi_factor",
                value: CudaMaParamValue::Float(2.0),
            },
            CudaMaParamKV {
                key: "sample_period",
                value: CudaMaParamValue::Int(30),
            },
            CudaMaParamKV {
                key: "strict",
                value: CudaMaParamValue::Bool(true),
            },
        ];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "volume_adjusted_ma",
                CudaMaData::Candles {
                    candles: &candles,
                    source: "close",
                },
                20,
                20,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct =
            crate::indicators::moving_averages::volume_adjusted_ma::VolumeAdjustedMa_batch_with_kernel(
                &candles.close,
                &candles.volume,
                &crate::indicators::moving_averages::volume_adjusted_ma::VolumeAdjustedMaBatchRange {
                    length: (20, 20, 0),
                    vi_factor: (2.0, 2.0, 0.0),
                    sample_period: (30, 30, 0),
                    strict: Some(true),
                },
                crate::utilities::enums::Kernel::Auto,
            )
            .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.values, 5e-2);
    }

    #[test]
    fn cuda_vwap_typed_anchor_matches_cpu_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "anchor",
            value: CudaMaParamValue::EnumString("1d"),
        }];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "vwap",
                CudaMaData::Candles {
                    candles: &candles,
                    source: "close",
                },
                10,
                10,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct = crate::indicators::moving_averages::vwap::vwap_batch_with_kernel(
            &candles.timestamp,
            &candles.volume,
            &candles.close,
            &crate::indicators::moving_averages::vwap::VwapBatchRange {
                anchor: ("1d".to_string(), "1d".to_string(), 0),
            },
            crate::utilities::enums::Kernel::Auto,
        )
        .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.values, 1e-2);
    }

    #[test]
    fn cuda_dma_typed_hull_ma_type_matches_direct_cuda_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(256);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "ema_length",
                value: CudaMaParamValue::Int(20),
            },
            CudaMaParamKV {
                key: "ema_gain_limit",
                value: CudaMaParamValue::Int(50),
            },
            CudaMaParamKV {
                key: "hull_ma_type",
                value: CudaMaParamValue::EnumString("EMA"),
            },
        ];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "dma",
                CudaMaData::Slice(&prices),
                14,
                14,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let direct_sweep = crate::indicators::moving_averages::dma::DmaBatchRange {
            hull_length: (14, 14, 0),
            ema_length: (20, 20, 0),
            ema_gain_limit: (50, 50, 0),
            hull_ma_type: "EMA".to_string(),
        };
        let cuda = CudaDma::new(0).unwrap();
        let direct_dev = cuda.dma_batch_dev(&prices_f32, &direct_sweep).unwrap();
        let mut direct = vec![0f32; direct_dev.rows * direct_dev.cols];
        direct_dev.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, direct_dev.rows);
        assert_eq!(dev.cols, direct_dev.cols);
        assert_series_eq_f32(&got, &direct, 1e-5);
    }

    #[test]
    fn cuda_ehlers_itrend_typed_params_match_direct_cuda_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(320);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "warmup_bars",
            value: CudaMaParamValue::Int(30),
        }];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "ehlers_itrend",
                CudaMaData::Slice(&prices),
                48,
                48,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let sweep = crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange {
            warmup_bars: (30, 30, 0),
            max_dc_period: (48, 48, 0),
        };
        let cuda = CudaEhlersITrend::new(0).unwrap();
        let direct_dev = cuda.ehlers_itrend_batch_dev(&prices_f32, &sweep).unwrap();
        let mut direct = vec![0f32; direct_dev.rows * direct_dev.cols];
        direct_dev.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, direct_dev.rows);
        assert_eq!(dev.cols, direct_dev.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_vama_typed_params_match_direct_cuda_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(320);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "vol_period",
            value: CudaMaParamValue::Int(51),
        }];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "vama",
                CudaMaData::Slice(&prices),
                18,
                22,
                2,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let sweep = crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
            base_period: (18, 22, 2),
            vol_period: (51, 51, 0),
        };
        let cuda = CudaVama::new(0).unwrap();
        let direct_dev = cuda.vama_batch_dev(&prices_f32, &sweep).unwrap();
        let mut direct = vec![0f32; direct_dev.rows * direct_dev.cols];
        direct_dev.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, direct_dev.rows);
        assert_eq!(dev.cols, direct_dev.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_maaq_typed_params_match_direct_cuda_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(320);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "fast_period",
                value: CudaMaParamValue::Int(2),
            },
            CudaMaParamKV {
                key: "slow_period",
                value: CudaMaParamValue::Int(30),
            },
        ];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "maaq",
                CudaMaData::Slice(&prices),
                18,
                22,
                2,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let sweep = crate::indicators::moving_averages::maaq::MaaqBatchRange {
            period: (18, 22, 2),
            fast_period: (2, 2, 0),
            slow_period: (30, 30, 0),
        };
        let cuda = CudaMaaq::new(0).unwrap();
        let direct_dev = cuda.maaq_batch_dev(&prices_f32, &sweep).unwrap();
        let mut direct = vec![0f32; direct_dev.rows * direct_dev.cols];
        direct_dev.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, direct_dev.rows);
        assert_eq!(dev.cols, direct_dev.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_tradjema_requires_candles_error() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let prices = sample_prices(128);
        let selector = CudaMaSelector::new(0);
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "tradjema",
            CudaMaData::Slice(&prices),
            40,
            40,
            0,
            &[],
        ));
        assert!(err.contains("high/low/close input"));
    }

    #[test]
    fn cuda_vwap_requires_candles_error() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let prices = sample_prices(128);
        let selector = CudaMaSelector::new(0);
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "vwap",
            CudaMaData::Slice(&prices),
            10,
            10,
            0,
            &[],
        ));
        assert!(err.contains("timestamp + volume + source input"));
    }

    #[test]
    fn cuda_volume_adjusted_ma_requires_candles_error() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let prices = sample_prices(128);
        let selector = CudaMaSelector::new(0);
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "volume_adjusted_ma",
            CudaMaData::Slice(&prices),
            20,
            20,
            0,
            &[],
        ));
        assert!(err.contains("requires volume input"));
    }

    #[test]
    fn cuda_mama_invalid_output_error() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let prices = sample_prices(256);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "output",
            value: CudaMaParamValue::EnumString("bad_line"),
        }];
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "mama",
            CudaMaData::Slice(&prices),
            10,
            10,
            0,
            &params,
        ));
        assert!(err.contains("expected 'mama' or 'fama'"));
    }

    #[test]
    fn cuda_ehlers_pma_invalid_output_error() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let prices = sample_prices(256);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "output",
            value: CudaMaParamValue::EnumString("bad_line"),
        }];
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "ehlers_pma",
            CudaMaData::Slice(&prices),
            10,
            10,
            0,
            &params,
        ));
        assert!(err.contains("expected 'predict' or 'trigger'"));
    }

    #[test]
    fn cuda_vwap_invalid_anchor_step_error() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "anchor",
                value: CudaMaParamValue::EnumString("1d"),
            },
            CudaMaParamKV {
                key: "anchor_step",
                value: CudaMaParamValue::Float(-1.0),
            },
        ];
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "vwap",
            CudaMaData::Candles {
                candles: &candles,
                source: "close",
            },
            10,
            10,
            0,
            &params,
        ));
        assert!(err.contains("expected >= 0"));
    }

    #[test]
    fn cuda_dma_invalid_hull_ma_type_error() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let prices = sample_prices(256);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "hull_ma_type",
            value: CudaMaParamValue::EnumString("BAD"),
        }];
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "dma",
            CudaMaData::Slice(&prices),
            14,
            14,
            0,
            &params,
        ))
        .to_ascii_lowercase();
        assert!(err.contains("hull"));
        assert!(err.contains("unsupported") || err.contains("invalid"));
    }

    #[test]
    fn cuda_buff_averages_typed_output_selection_matches_direct_cuda_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "fast_period",
                value: CudaMaParamValue::Int(5),
            },
            CudaMaParamKV {
                key: "output",
                value: CudaMaParamValue::EnumString("slow"),
            },
        ];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "buff_averages",
                CudaMaData::Candles {
                    candles: &candles,
                    source: "close",
                },
                20,
                20,
                0,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let prices_f32: Vec<f32> = candles.close.iter().map(|&v| v as f32).collect();
        let volumes_f32: Vec<f32> = candles.volume.iter().map(|&v| v as f32).collect();
        let sweep = crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange {
            fast_period: (5, 5, 0),
            slow_period: (20, 20, 0),
        };
        let cuda = CudaBuffAverages::new(0).unwrap();
        let (_fast, slow) = cuda
            .buff_averages_batch_dev(&prices_f32, &volumes_f32, &sweep)
            .unwrap();
        let mut direct = vec![0f32; slow.rows * slow.cols];
        slow.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, slow.rows);
        assert_eq!(dev.cols, slow.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_buff_averages_requires_candles_error() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let prices = sample_prices(128);
        let selector = CudaMaSelector::new(0);
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "buff_averages",
            CudaMaData::Slice(&prices),
            20,
            20,
            0,
            &[],
        ));
        assert!(err.contains("requires volume input"));
    }

    #[test]
    fn cuda_buff_averages_invalid_output_error() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "output",
            value: CudaMaParamValue::EnumString("bad_line"),
        }];
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "buff_averages",
            CudaMaData::Candles {
                candles: &candles,
                source: "close",
            },
            20,
            20,
            0,
            &params,
        ));
        assert!(err.contains("expected 'fast' or 'slow'"));
    }

    #[test]
    fn cuda_buff_averages_numeric_params_match_direct_fast_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let mut params = std::collections::HashMap::new();
        params.insert("fast_period_start".to_string(), 5.0);
        params.insert("fast_period_end".to_string(), 5.0);
        params.insert("fast_period_step".to_string(), 0.0);
        params.insert("slow_period_start".to_string(), 20.0);
        params.insert("slow_period_end".to_string(), 22.0);
        params.insert("slow_period_step".to_string(), 1.0);
        let dev = selector
            .ma_sweep_to_device_with_params(
                "buff_averages",
                CudaMaData::Candles {
                    candles: &candles,
                    source: "close",
                },
                20,
                22,
                1,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let prices_f32: Vec<f32> = candles.close.iter().map(|&v| v as f32).collect();
        let volumes_f32: Vec<f32> = candles.volume.iter().map(|&v| v as f32).collect();
        let sweep = crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange {
            fast_period: (5, 5, 0),
            slow_period: (20, 22, 1),
        };
        let cuda = CudaBuffAverages::new(0).unwrap();
        let (fast, _slow) = cuda
            .buff_averages_batch_dev(&prices_f32, &volumes_f32, &sweep)
            .unwrap();
        let mut direct = vec![0f32; fast.rows * fast.cols];
        fast.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, fast.rows);
        assert_eq!(dev.cols, fast.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_buff_averages_many_params_match_cpu_reference_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let mut params = std::collections::HashMap::new();
        params.insert("fast_period_start".to_string(), 5.0);
        params.insert("fast_period_end".to_string(), 5.0);
        params.insert("fast_period_step".to_string(), 0.0);
        params.insert("slow_period_start".to_string(), 20.0);
        params.insert("slow_period_end".to_string(), 24.0);
        params.insert("slow_period_step".to_string(), 2.0);
        let dev = selector
            .ma_sweep_to_device_with_params(
                "buff_averages",
                CudaMaData::Candles {
                    candles: &candles,
                    source: "close",
                },
                20,
                24,
                2,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct =
            crate::indicators::moving_averages::buff_averages::buff_averages_batch_with_kernel(
                &candles.close,
                &candles.volume,
                &crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange {
                    fast_period: (5, 5, 0),
                    slow_period: (20, 24, 2),
                },
                crate::utilities::enums::Kernel::Auto,
            )
            .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.fast, 5e-2);
    }

    #[test]
    fn cuda_vama_many_params_match_cpu_reference_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(320);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "vol_period",
            value: CudaMaParamValue::Int(51),
        }];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "vama",
                CudaMaData::Slice(&prices),
                18,
                24,
                2,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct =
            crate::indicators::moving_averages::volatility_adjusted_ma::vama_batch_with_kernel(
                &prices,
                &crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
                    base_period: (18, 24, 2),
                    vol_period: (51, 51, 0),
                },
                crate::utilities::enums::Kernel::Auto,
            )
            .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.values, 5e-2);
    }

    #[test]
    fn cuda_maaq_many_params_match_cpu_reference_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(320);
        let selector = CudaMaSelector::new(0);
        let params = [
            CudaMaParamKV {
                key: "fast_period",
                value: CudaMaParamValue::Int(2),
            },
            CudaMaParamKV {
                key: "slow_period",
                value: CudaMaParamValue::Int(30),
            },
        ];
        let dev = selector
            .ma_sweep_to_device_with_typed_params(
                "maaq",
                CudaMaData::Slice(&prices),
                18,
                24,
                2,
                &params,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let direct = crate::indicators::moving_averages::maaq::maaq_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::maaq::MaaqBatchRange {
                period: (18, 24, 2),
                fast_period: (2, 2, 0),
                slow_period: (30, 30, 0),
            },
            crate::utilities::enums::Kernel::Auto,
        )
        .unwrap();

        assert_eq!(dev.rows, direct.rows);
        assert_eq!(dev.cols, direct.cols);
        assert_series_eq_f32_f64(&got, &direct.values, 5e-2);
    }

    #[test]
    fn cuda_buff_averages_fractional_fast_period_error() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let selector = CudaMaSelector::new(0);
        let params = [CudaMaParamKV {
            key: "fast_period",
            value: CudaMaParamValue::Float(5.5),
        }];
        let err = err_string(selector.ma_sweep_to_device_with_typed_params(
            "buff_averages",
            CudaMaData::Candles {
                candles: &candles,
                source: "close",
            },
            20,
            20,
            0,
            &params,
        ));
        assert!(err.contains("expected integer"));
    }

    #[test]
    fn cuda_selector_device_ref_ema_matches_host_selector_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(256);
        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let runtime = crate::cuda::CudaRuntime::new(0).unwrap();
        let d_prices = runtime.upload_f32(&prices_f32).unwrap();
        let selector = CudaMaSelector::new(0);

        let dev = selector
            .ma_sweep_to_device_ref(
                "ema",
                CudaMaDeviceDataRef::Slice(d_prices.as_view()),
                0,
                12,
                18,
                3,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let host = selector
            .ma_sweep_to_device("ema", CudaMaData::SliceF32(&prices_f32), 12, 18, 3)
            .unwrap();
        let mut direct = vec![0f32; host.rows * host.cols];
        host.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, host.rows);
        assert_eq!(dev.cols, host.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_selector_device_ref_vwma_matches_host_selector_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(300);
        let timestamps: Vec<i64> = candles.timestamp.clone();
        let open: Vec<f32> = candles.open.iter().map(|&v| v as f32).collect();
        let high: Vec<f32> = candles.high.iter().map(|&v| v as f32).collect();
        let low: Vec<f32> = candles.low.iter().map(|&v| v as f32).collect();
        let close: Vec<f32> = candles.close.iter().map(|&v| v as f32).collect();
        let volume: Vec<f32> = candles.volume.iter().map(|&v| v as f32).collect();
        let runtime = crate::cuda::CudaRuntime::new(0).unwrap();
        let d_ohlcv = runtime
            .upload_ohlcv(Some(&timestamps), &open, &high, &low, &close, &volume, None)
            .unwrap();
        let selector = CudaMaSelector::new(0);

        let dev = selector
            .ma_sweep_to_device_ref(
                "vwma",
                CudaMaDeviceDataRef::Ohlcv(d_ohlcv.as_view()),
                0,
                10,
                14,
                2,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let host = selector
            .ma_sweep_to_device(
                "vwma",
                CudaMaData::OhlcvF32 {
                    timestamp: Some(&timestamps),
                    open: &open,
                    high: &high,
                    low: &low,
                    close: &close,
                    volume: &volume,
                    source: None,
                },
                10,
                14,
                2,
            )
            .unwrap();
        let mut direct = vec![0f32; host.rows * host.cols];
        host.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, host.rows);
        assert_eq!(dev.cols, host.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_selector_device_ref_frama_matches_host_selector_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let candles = sample_candles(320);
        let open: Vec<f32> = candles.open.iter().map(|&v| v as f32).collect();
        let high: Vec<f32> = candles.high.iter().map(|&v| v as f32).collect();
        let low: Vec<f32> = candles.low.iter().map(|&v| v as f32).collect();
        let close: Vec<f32> = candles.close.iter().map(|&v| v as f32).collect();
        let runtime = crate::cuda::CudaRuntime::new(0).unwrap();
        let d_ohlc = runtime
            .upload_ohlc(&open, &high, &low, &close, None)
            .unwrap();
        let selector = CudaMaSelector::new(0);

        let dev = selector
            .ma_sweep_to_device_ref(
                "frama",
                CudaMaDeviceDataRef::Ohlc(d_ohlc.as_view()),
                0,
                16,
                20,
                2,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let host = selector
            .ma_sweep_to_device(
                "frama",
                CudaMaData::OhlcF32 {
                    open: &open,
                    high: &high,
                    low: &low,
                    close: &close,
                    source: None,
                },
                16,
                20,
                2,
            )
            .unwrap();
        let mut direct = vec![0f32; host.rows * host.cols];
        host.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, host.rows);
        assert_eq!(dev.cols, host.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_selector_device_ref_first_valid_matches_host_selector_when_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let mut prices = sample_prices(240);
        let first_valid = 4usize;
        for value in prices.iter_mut().take(first_valid) {
            *value = f64::NAN;
        }
        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let runtime = crate::cuda::CudaRuntime::new(0).unwrap();
        let d_prices = runtime.upload_f32(&prices_f32).unwrap();
        let selector = CudaMaSelector::new(0);

        let dev = selector
            .ma_sweep_to_device_ref(
                "sma",
                CudaMaDeviceDataRef::Slice(d_prices.as_view()),
                first_valid,
                8,
                12,
                2,
            )
            .unwrap();
        let mut got = vec![0f32; dev.rows * dev.cols];
        dev.buf.copy_to(&mut got).unwrap();

        let host = selector
            .ma_sweep_to_device("sma", CudaMaData::SliceF32(&prices_f32), 8, 12, 2)
            .unwrap();
        let mut direct = vec![0f32; host.rows * host.cols];
        host.buf.copy_to(&mut direct).unwrap();

        assert_eq!(dev.rows, host.rows);
        assert_eq!(dev.cols, host.cols);
        assert_series_eq_f32(&got, &direct, 5e-5);
    }

    #[test]
    fn cuda_selector_device_ref_reuses_cached_vram_ma_without_output_drift() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(256);
        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let runtime = crate::cuda::CudaRuntime::new(0).unwrap();
        let d_prices = runtime.upload_f32(&prices_f32).unwrap();
        let selector = CudaMaSelector::new(0);

        let first_dev = selector
            .ma_sweep_to_device_ref(
                "alma",
                CudaMaDeviceDataRef::Slice(d_prices.as_view()),
                0,
                8,
                14,
                2,
            )
            .unwrap();
        let mut first_got = vec![0f32; first_dev.rows * first_dev.cols];
        first_dev.buf.copy_to(&mut first_got).unwrap();

        let second_dev = selector
            .ma_sweep_to_device_ref(
                "alma",
                CudaMaDeviceDataRef::Slice(d_prices.as_view()),
                0,
                16,
                22,
                2,
            )
            .unwrap();
        let mut second_got = vec![0f32; second_dev.rows * second_dev.cols];
        second_dev.buf.copy_to(&mut second_got).unwrap();

        let first_host = selector
            .ma_sweep_to_device("alma", CudaMaData::SliceF32(&prices_f32), 8, 14, 2)
            .unwrap();
        let mut first_direct = vec![0f32; first_host.rows * first_host.cols];
        first_host.buf.copy_to(&mut first_direct).unwrap();

        let second_host = selector
            .ma_sweep_to_device("alma", CudaMaData::SliceF32(&prices_f32), 16, 22, 2)
            .unwrap();
        let mut second_direct = vec![0f32; second_host.rows * second_host.cols];
        second_host.buf.copy_to(&mut second_direct).unwrap();

        assert_eq!(first_dev.rows, first_host.rows);
        assert_eq!(first_dev.cols, first_host.cols);
        assert_series_eq_f32(&first_got, &first_direct, 5e-5);

        assert_eq!(second_dev.rows, second_host.rows);
        assert_eq!(second_dev.cols, second_host.cols);
        assert_series_eq_f32(&second_got, &second_direct, 5e-5);
    }

    #[test]
    fn cuda_selector_device_ref_reuses_cached_period_buffer_without_output_drift() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices = sample_prices(256);
        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let runtime = crate::cuda::CudaRuntime::new(0).unwrap();
        let d_prices = runtime.upload_f32(&prices_f32).unwrap();
        let selector = CudaMaSelector::new(0);

        let first_dev = selector
            .ma_sweep_to_device_ref(
                "alma",
                CudaMaDeviceDataRef::Slice(d_prices.as_view()),
                0,
                8,
                14,
                2,
            )
            .unwrap();
        let mut first_got = vec![0f32; first_dev.rows * first_dev.cols];
        first_dev.buf.copy_to(&mut first_got).unwrap();

        let second_dev = selector
            .ma_sweep_to_device_ref(
                "alma",
                CudaMaDeviceDataRef::Slice(d_prices.as_view()),
                0,
                8,
                14,
                2,
            )
            .unwrap();
        let mut second_got = vec![0f32; second_dev.rows * second_dev.cols];
        second_dev.buf.copy_to(&mut second_got).unwrap();

        let host = selector
            .ma_sweep_to_device("alma", CudaMaData::SliceF32(&prices_f32), 8, 14, 2)
            .unwrap();
        let mut direct = vec![0f32; host.rows * host.cols];
        host.buf.copy_to(&mut direct).unwrap();

        assert_eq!(first_dev.rows, host.rows);
        assert_eq!(first_dev.cols, host.cols);
        assert_eq!(second_dev.rows, host.rows);
        assert_eq!(second_dev.cols, host.cols);
        assert_series_eq_f32(&first_got, &direct, 5e-5);
        assert_series_eq_f32(&second_got, &direct, 5e-5);
    }

    #[test]
    fn cuda_selector_device_ref_explicit_sweep_plan_reuses_device_periods_without_output_drift() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let prices_a = sample_prices(256);
        let prices_b: Vec<f64> = sample_prices(256)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + (i as f64 * 0.0005))
            .collect();
        let prices_a_f32: Vec<f32> = prices_a.iter().map(|&v| v as f32).collect();
        let prices_b_f32: Vec<f32> = prices_b.iter().map(|&v| v as f32).collect();
        let runtime = crate::cuda::CudaRuntime::new(0).unwrap();
        let d_prices_a = runtime.upload_f32(&prices_a_f32).unwrap();
        let d_prices_b = runtime.upload_f32(&prices_b_f32).unwrap();
        let selector = CudaMaSelector::new(0);
        let plan = selector.create_sweep_plan(8, 14, 2).unwrap();

        let dev_a = selector
            .ma_sweep_plan_to_device_ref(
                "alma",
                CudaMaDeviceDataRef::Slice(d_prices_a.as_view()),
                0,
                &plan,
            )
            .unwrap();
        let mut got_a = vec![0f32; dev_a.rows * dev_a.cols];
        dev_a.buf.copy_to(&mut got_a).unwrap();

        let dev_b = selector
            .ma_sweep_plan_to_device_ref(
                "alma",
                CudaMaDeviceDataRef::Slice(d_prices_b.as_view()),
                0,
                &plan,
            )
            .unwrap();
        let mut got_b = vec![0f32; dev_b.rows * dev_b.cols];
        dev_b.buf.copy_to(&mut got_b).unwrap();

        let host_a = selector
            .ma_sweep_to_device("alma", CudaMaData::SliceF32(&prices_a_f32), 8, 14, 2)
            .unwrap();
        let mut direct_a = vec![0f32; host_a.rows * host_a.cols];
        host_a.buf.copy_to(&mut direct_a).unwrap();

        let host_b = selector
            .ma_sweep_to_device("alma", CudaMaData::SliceF32(&prices_b_f32), 8, 14, 2)
            .unwrap();
        let mut direct_b = vec![0f32; host_b.rows * host_b.cols];
        host_b.buf.copy_to(&mut direct_b).unwrap();

        assert_eq!(dev_a.rows, host_a.rows);
        assert_eq!(dev_a.cols, host_a.cols);
        assert_eq!(dev_b.rows, host_b.rows);
        assert_eq!(dev_b.cols, host_b.cols);
        assert_series_eq_f32(&got_a, &direct_a, 5e-5);
        assert_series_eq_f32(&got_b, &direct_b, 5e-5);
    }
}
