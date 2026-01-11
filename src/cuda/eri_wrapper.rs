#![cfg(feature = "cuda")]

//! CUDA wrapper for ERI (Elder Ray Index)
//!
//! Parity goals (aligned with ALMA/CWMA wrappers):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/eri_kernel.ptx")) with DetermineTargetFromContext and O2.
//! - Stream NON_BLOCKING.
//! - Warmup/NaN semantics match scalar eri.rs exactly (triple-validity; warmup = first_valid + period - 1).
//! - Batch (one-series × many-params) computes MA per row via the CUDA MA selector and fuses a subtract kernel.
//! - Many-series × one-param uses time-major layout; MA computed by the chosen CUDA MA wrapper.

use crate::cuda::moving_averages::ma_selector::{CudaMaData, CudaMaSelector};
use crate::cuda::moving_averages::{CudaEmaError, CudaSmaError, CudaWmaError, CudaZlemaError};
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::eri::{EriBatchRange, EriParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

const ERI_TIME_TILE: u32 = 16; 
const ERI_SMALL_P_NO_TRANSPOSE_THRESHOLD: usize = 64;
#[inline]
fn ceil_div(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaEriPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaEriPolicy {
        fn default() -> Self {
            Self {
                batch: BatchKernelPolicy::Auto,
                many_series: ManySeriesKernelPolicy::Auto,
            }
        }
}

#[derive(thiserror::Error, Debug)]
pub enum CudaEriError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error(transparent)]
    Ema(#[from] CudaEmaError),
    #[error(transparent)]
    Sma(#[from] CudaSmaError),
    #[error(transparent)]
    Wma(#[from] CudaWmaError),
    #[error(transparent)]
    Zlema(#[from] CudaZlemaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaEri {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaEriPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEri {
    pub fn new(device_id: usize) -> Result<Self, CudaEriError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/eri_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaEriPolicy::default(),
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaEriPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaEriPolicy {
        &self.policy
    }
    pub fn synchronize(&self) -> Result<(), CudaEriError> { self.stream.synchronize().map_err(Into::into) }
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }
    

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else { true }
    }

    fn expand_periods(sweep: &EriBatchRange) -> Result<Vec<usize>, CudaEriError> {
        let (start, end, step) = sweep.period;
        if step == 0 { return Ok(vec![start]); }
        if start == end { return Ok(vec![start]); }
        let mut out = Vec::new();
        if start < end {
            let mut p = start;
            while p <= end {
                out.push(p);
                p = p.checked_add(step).ok_or_else(|| CudaEriError::InvalidInput("range overflow".into()))?;
                if p == 0 { break; }
            }
        } else {
            let mut p = start;
            while p >= end {
                out.push(p);
                if p < step { break; }
                p -= step;
            }
        }
        if out.is_empty() { return Err(CudaEriError::InvalidInput("empty period sweep".into())); }
        Ok(out)
    }

    
    fn validate_and_first_valid(
        high: &[f32],
        low: &[f32],
        src: &[f32],
        max_period: usize,
    ) -> Result<usize, CudaEriError> {
        if high.is_empty() || low.is_empty() || src.is_empty() {
            return Err(CudaEriError::InvalidInput("empty input".into()));
        }
        let n = high.len().min(low.len()).min(src.len());
        let first = (0..n)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !src[i].is_nan())
            .ok_or_else(|| CudaEriError::InvalidInput("all values are NaN".into()))?;
        if n - first < max_period {
            return Err(CudaEriError::InvalidInput("not enough valid data".into()));
        }
        Ok(first)
    }

    
    pub fn eri_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        source: &[f32],
        sweep: &EriBatchRange,
    ) -> Result<((DeviceArrayF32, DeviceArrayF32), Vec<EriParams>), CudaEriError> {
        
        let periods = Self::expand_periods(sweep)?;
        let max_p = *periods.iter().max().unwrap();
        let first_valid = Self::validate_and_first_valid(high, low, source, max_p)?;
        let len = source.len().min(high.len()).min(low.len());

        
        let combos = periods.len();
        let pl = combos
            .checked_mul(len)
            .ok_or_else(|| CudaEriError::InvalidInput("P*len overflow".into()))?;
        let el = std::mem::size_of::<f32>();
        let two_pl = pl
            .checked_mul(2)
            .ok_or_else(|| CudaEriError::InvalidInput("size overflow".into()))?;
        let base = len
            .checked_mul(3)
            .and_then(|x| x.checked_add(two_pl))
            .ok_or_else(|| CudaEriError::InvalidInput("size overflow".into()))?;
        let req = base
            .checked_mul(el)
            .ok_or_else(|| CudaEriError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; 
        if !Self::will_fit(req, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaEriError::OutOfMemory { required: req, free, headroom });
            } else {
                return Err(CudaEriError::InvalidInput("insufficient device memory".into()));
            }
        }

        
        let d_high = unsafe { DeviceBuffer::from_slice_async(high, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low, &self.stream) }?;

        
        let mut d_bull: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(pl, &self.stream) }?;
        let mut d_bear: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(pl, &self.stream) }?;

        
        let ma_type_lc = sweep.ma_type.to_ascii_lowercase();
        let maybe_ma_batch: Option<DeviceArrayF32> = match ma_type_lc.as_str() {
            "ema" => {
                let range = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::ema_wrapper::CudaEma::new(self.device_id as usize)?;
                Some(cuda.ema_batch_dev(source, &range)?)
            }
            "sma" => {
                let range = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::sma_wrapper::CudaSma::new(self.device_id as usize)?;
                let (dev, _combos) = cuda.sma_batch_dev(source, &range)?;
                Some(dev)
            }
            "wma" => {
                let range = crate::indicators::moving_averages::wma::WmaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::wma_wrapper::CudaWma::new(self.device_id as usize)?;
                Some(cuda.wma_batch_dev(source, &range)?)
            }
            "zlema" => {
                let range = crate::indicators::moving_averages::zlema::ZlemaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::zlema_wrapper::CudaZlema::new(self.device_id as usize)?;
                let (dev, _combos) = cuda.zlema_batch_dev(source, &range)?;
                Some(dev)
            }
            _ => None,
        };

        let mut combos = Vec::with_capacity(periods.len());
        if let Some(ma_rm) = maybe_ma_batch {
            
            debug_assert_eq!(ma_rm.rows, periods.len());
            debug_assert_eq!(ma_rm.cols, len);

            if periods.len() <= ERI_SMALL_P_NO_TRANSPOSE_THRESHOLD {
                
                let func = self
                    .module
                    .get_function("eri_batch_f32")
                    .map_err(|_| CudaEriError::MissingKernelSymbol { name: "eri_batch_f32" })?;
                let block_x = match self.policy.batch {
                    BatchKernelPolicy::Auto => 256,
                    BatchKernelPolicy::Plain { block_x } => block_x.max(32),
                };
                let block: BlockSize = (block_x, 1, 1).into();
                let grid: GridSize = (((len as u32 + block_x - 1) / block_x).max(1), 1, 1).into();

                for (row_idx, &p) in periods.iter().enumerate() {
                    let row_bytes = match row_idx
                        .checked_mul(len)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    {
                        Some(v) => v as u64,
                        None => return Err(CudaEriError::InvalidInput("row offset overflow".into())),
                    };
                    unsafe {
                        let mut h = d_high.as_device_ptr().as_raw();
                        let mut l = d_low.as_device_ptr().as_raw();
                        let mut m = ma_rm.buf.as_device_ptr().as_raw() + row_bytes;
                        let mut n = len as i32;
                        let mut fv = first_valid as i32;
                        let mut per = p as i32;
                        let mut bo = d_bull.as_device_ptr().as_raw() + row_bytes;
                        let mut ro = d_bear.as_device_ptr().as_raw() + row_bytes;
                        let mut args: [*mut c_void; 8] = [
                            &mut h as *mut _ as *mut c_void,
                            &mut l as *mut _ as *mut c_void,
                            &mut m as *mut _ as *mut c_void,
                            &mut n as *mut _ as *mut c_void,
                            &mut fv as *mut _ as *mut c_void,
                            &mut per as *mut _ as *mut c_void,
                            &mut bo as *mut _ as *mut c_void,
                            &mut ro as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, &mut args)?;
                    }
                }

                combos.extend(periods.iter().map(|&p| EriParams {
                    period: Some(p),
                    ma_type: Some(sweep.ma_type.clone()),
                }));

                self.stream.synchronize()?;
                let bull = DeviceArrayF32 { buf: d_bull, rows: periods.len(), cols: len };
                let bear = DeviceArrayF32 { buf: d_bear, rows: periods.len(), cols: len };
                return Ok(((bull, bear), combos));
            }

            
            let func_tr = self
                .module
                .get_function("transpose_rm_to_tm_32x32_pad_f32")
                .map_err(|_| CudaEriError::MissingKernelSymbol { name: "transpose_rm_to_tm_32x32_pad_f32" })?;
            let tm_len = periods.len().checked_mul(len).ok_or_else(|| CudaEriError::InvalidInput("P*len overflow".into()))?;
            let mut d_ma_tm: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(tm_len, &self.stream) }?;
            unsafe {
                let mut in_ptr = ma_rm.buf.as_device_ptr().as_raw();
                let mut R = periods.len() as i32; 
                let mut C = len as i32; 
                let mut out_ptr = d_ma_tm.as_device_ptr().as_raw();
                let block_tr: BlockSize = (32, 32, 1).into();
                let grid_tr: GridSize = (ceil_div(C as u32, 32), ceil_div(R as u32, 32), 1).into();
                let mut args: [*mut c_void; 4] = [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut R as *mut _ as *mut c_void,
                    &mut C as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func_tr, grid_tr, block_tr, 0, &mut args)?;
            }
            

            
            let func = self
                .module
                .get_function("eri_one_series_many_params_time_major_f32")
                .map_err(|_| CudaEriError::MissingKernelSymbol { name: "eri_one_series_many_params_time_major_f32" })?;
            let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
            let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }?;

            let block_x = match self.policy.batch {
                BatchKernelPolicy::Auto => {
                    let p = periods.len() as u32;
                    if p <= 32 { 32 } else if p <= 64 { 64 } else if p <= 128 { 128 } else { 256 }
                }
                BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            };
            let block: BlockSize = (block_x, 1, 1).into();
            let grid: GridSize = (
                ceil_div(periods.len() as u32, block_x),
                ceil_div(len as u32, ERI_TIME_TILE),
                1,
            )
                .into();

            if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged
            {
                eprintln!(
                    "[eri] batch kernel (one-series→many-params): block_x={} P={} rows={} ma_type={} first_valid={}",
                    block_x, periods.len(), len, sweep.ma_type, first_valid
                );
                unsafe {
                    (*(self as *const _ as *mut CudaEri)).debug_batch_logged = true;
                }
            }

            unsafe {
                let mut h = d_high.as_device_ptr().as_raw();
                let mut l = d_low.as_device_ptr().as_raw();
                let mut m_tm = d_ma_tm.as_device_ptr().as_raw();
                let mut P_i = periods.len() as i32;
                let mut rows_i = len as i32;
                let mut fv_i = first_valid as i32;
                let mut per_ptr = d_periods.as_device_ptr().as_raw(); 
                let mut per_fallback = 0i32; 
                let mut bo = d_bull.as_device_ptr().as_raw();
                let mut ro = d_bear.as_device_ptr().as_raw();
                let mut out_rm = 1i32; 
                let mut args: [*mut c_void; 11] = [
                    &mut h as *mut _ as *mut c_void,
                    &mut l as *mut _ as *mut c_void,
                    &mut m_tm as *mut _ as *mut c_void,
                    &mut P_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut per_fallback as *mut _ as *mut c_void,
                    &mut bo as *mut _ as *mut c_void,
                    &mut ro as *mut _ as *mut c_void,
                    &mut out_rm as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, &mut args)?;
            }

            combos.extend(periods.iter().map(|&p| EriParams {
                period: Some(p),
                ma_type: Some(sweep.ma_type.clone()),
            }));
        } else {
            
            let func = self
                .module
                .get_function("eri_batch_f32")
                .map_err(|_| CudaEriError::MissingKernelSymbol { name: "eri_batch_f32" })?;
            let block_x = match self.policy.batch {
                BatchKernelPolicy::Auto => 256,
                BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            };
            let grid: GridSize = (((len as u32 + block_x - 1) / block_x).max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged
            {
                eprintln!("[eri] batch kernel (fallback per-row): block_x={} rows={} len={} ma_type={} first_valid={}", block_x, periods.len(), len, sweep.ma_type, first_valid);
                unsafe {
                    (*(self as *const _ as *mut CudaEri)).debug_batch_logged = true;
                }
            }
            let selector = CudaMaSelector::new(self.device_id as usize);
            for (row_idx, &p) in periods.iter().enumerate() {
                let ma_dev = selector
                    .ma_to_device(&sweep.ma_type, CudaMaData::SliceF32(source), p)
                    .map_err(|e| CudaEriError::InvalidInput(e.to_string()))?;
                debug_assert_eq!(ma_dev.rows, 1);
                debug_assert_eq!(ma_dev.cols, len);
                unsafe {
                    let mut h = d_high.as_device_ptr().as_raw();
                    let mut l = d_low.as_device_ptr().as_raw();
                    let mut m = ma_dev.buf.as_device_ptr().as_raw();
                    let mut n = len as i32;
                    let mut fv = first_valid as i32;
                    let mut per = p as i32;
                    let row_bytes = match row_idx
                        .checked_mul(len)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    {
                        Some(v) => v,
                        None => {
                            return Err(CudaEriError::InvalidInput(
                                "row offset overflow".into(),
                            ))
                        }
                    };
                    let row_off_bytes = row_bytes as u64;
                    let mut bo = d_bull.as_device_ptr().as_raw() + row_off_bytes;
                    let mut ro = d_bear.as_device_ptr().as_raw() + row_off_bytes;
                    let mut args: [*mut c_void; 8] = [
                        &mut h as *mut _ as *mut c_void,
                        &mut l as *mut _ as *mut c_void,
                        &mut m as *mut _ as *mut c_void,
                        &mut n as *mut _ as *mut c_void,
                        &mut fv as *mut _ as *mut c_void,
                        &mut per as *mut _ as *mut c_void,
                        &mut bo as *mut _ as *mut c_void,
                        &mut ro as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, &mut args)?;
                }
                combos.push(EriParams { period: Some(p), ma_type: Some(sweep.ma_type.clone()) });
            }
        }

        self.stream.synchronize()?;
        let bull = DeviceArrayF32 { buf: d_bull, rows: periods.len(), cols: len };
        let bear = DeviceArrayF32 { buf: d_bear, rows: periods.len(), cols: len };
        Ok(((bull, bear), combos))
    }

    
    pub fn eri_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        source_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        ma_type: &str,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaEriError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEriError::InvalidInput("empty matrix".into()));
        }
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || source_tm.len() != cols * rows
        {
            return Err(CudaEriError::InvalidInput("matrix shape mismatch".into()));
        }

        
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !source_tm[idx].is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            
            if (first_valids[s] as usize) + period - 1 >= rows {
                return Err(CudaEriError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        
        let cr = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaEriError::InvalidInput("cols*rows overflow".into()))?;
        let el = std::mem::size_of::<f32>();
        let three_cr = cr
            .checked_mul(3)
            .ok_or_else(|| CudaEriError::InvalidInput("size overflow".into()))?;
        let two_cr = cr
            .checked_mul(2)
            .ok_or_else(|| CudaEriError::InvalidInput("size overflow".into()))?;
        let base = three_cr
            .checked_add(cols)
            .and_then(|x| x.checked_add(two_cr))
            .ok_or_else(|| CudaEriError::InvalidInput("size overflow".into()))?;
        let req = base
            .checked_mul(el)
            .ok_or_else(|| CudaEriError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; 
        if !Self::will_fit(req, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaEriError::OutOfMemory { required: req, free, headroom });
            } else {
                return Err(CudaEriError::InvalidInput("insufficient device memory".into()));
            }
        }

        
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;

        
        let ma_dev =
            self.ma_many_series_one_param_time_major_dev(source_tm, cols, rows, period, ma_type)?;

        
        let total = cr;
        let mut d_bull: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;
        let mut d_bear: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;

        
        let func = self
            .module
            .get_function("eri_many_series_one_param_time_major_f32")
            .map_err(|_| CudaEriError::MissingKernelSymbol { name: "eri_many_series_one_param_time_major_f32" })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        
        let grid: GridSize = (
            ceil_div(cols as u32, block_x),
            ceil_div(rows as u32, ERI_TIME_TILE),
            1,
        )
            .into();
        let block: BlockSize = (block_x, 1, 1).into();
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_many_logged {
            eprintln!(
                "[eri] many-series kernel: block_x={} cols={} rows={} period={} ma_type={} ",
                block_x, cols, rows, period, ma_type
            );
            unsafe {
                (*(self as *const _ as *mut CudaEri)).debug_many_logged = true;
            }
        }
        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut m = ma_dev.buf.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv = d_first.as_device_ptr().as_raw();
            let mut p = period as i32;
            let mut bo = d_bull.as_device_ptr().as_raw();
            let mut ro = d_bear.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 9] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut m as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut bo as *mut _ as *mut c_void,
                &mut ro as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }

        self.stream.synchronize()?;
        let bull = DeviceArrayF32 { buf: d_bull, rows, cols };
        let bear = DeviceArrayF32 { buf: d_bear, rows, cols };
        Ok((bull, bear))
    }

    fn ma_many_series_one_param_time_major_dev(
        &self,
        source_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        ma_type: &str,
    ) -> Result<DeviceArrayF32, CudaEriError> {
        use crate::cuda::moving_averages;
        let t = ma_type.to_ascii_lowercase();
        
        match t.as_str() {
            "ema" => {
                let params = crate::indicators::moving_averages::ema::EmaParams { period: Some(period) };
                let cuda = crate::cuda::moving_averages::ema_wrapper::CudaEma::new(self.device_id as usize)?;
                cuda.ema_many_series_one_param_time_major_dev(source_tm, cols, rows, &params).map_err(Into::into)
            }
            "sma" => {
                let params = crate::indicators::moving_averages::sma::SmaParams {
                    period: Some(period),
                };
                let cuda = crate::cuda::moving_averages::sma_wrapper::CudaSma::new(self.device_id as usize)?;
                let dev = cuda.sma_multi_series_one_param_time_major_dev(source_tm, cols, rows, &params)?;
                Ok(dev)
            }
            "wma" => {
                let params = crate::indicators::moving_averages::wma::WmaParams {
                    period: Some(period),
                };
                let cuda = crate::cuda::moving_averages::wma_wrapper::CudaWma::new(self.device_id as usize)?;
                cuda.wma_multi_series_one_param_time_major_dev(source_tm, cols, rows, &params).map_err(Into::into)
            }
            "zlema" => {
                let params = crate::indicators::moving_averages::zlema::ZlemaParams {
                    period: Some(period),
                };
                let cuda = crate::cuda::moving_averages::zlema_wrapper::CudaZlema::new(self.device_id as usize)?;
                cuda.zlema_many_series_one_param_time_major_dev(source_tm, cols, rows, &params).map_err(Into::into)
            }
            _ => Err(CudaEriError::InvalidInput(format!("unsupported MA: {}", t))),
        }
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const LEN_1M: usize = 1_000_000;
    const COLS_512: usize = 512;
    const ROWS_16K: usize = 16_384;

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct BatchDeviceState {
        cuda: CudaEri,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_ma_rm: DeviceBuffer<f32>,
        d_bull: DeviceBuffer<f32>,
        d_bear: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        periods: Vec<i32>,
        grid: GridSize,
        block: BlockSize,
    }
    impl CudaBenchState for BatchDeviceState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("eri_batch_f32")
                .expect("eri_batch_f32");

            for (row_idx, &p) in self.periods.iter().enumerate() {
                let row_bytes = (row_idx * self.len * std::mem::size_of::<f32>()) as u64;
                unsafe {
                    let mut h = self.d_high.as_device_ptr().as_raw();
                    let mut l = self.d_low.as_device_ptr().as_raw();
                    let mut m = self.d_ma_rm.as_device_ptr().as_raw() + row_bytes;
                    let mut n = self.len as i32;
                    let mut fv = self.first_valid as i32;
                    let mut per = p;
                    let mut bo = self.d_bull.as_device_ptr().as_raw() + row_bytes;
                    let mut ro = self.d_bear.as_device_ptr().as_raw() + row_bytes;
                    let mut args: [*mut c_void; 8] = [
                        &mut h as *mut _ as *mut c_void,
                        &mut l as *mut _ as *mut c_void,
                        &mut m as *mut _ as *mut c_void,
                        &mut n as *mut _ as *mut c_void,
                        &mut fv as *mut _ as *mut c_void,
                        &mut per as *mut _ as *mut c_void,
                        &mut bo as *mut _ as *mut c_void,
                        &mut ro as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&func, self.grid, self.block, 0, &mut args)
                        .expect("eri_batch launch");
                }
            }
            self.cuda.stream.synchronize().expect("eri batch sync");
        }
    }

    struct ManySeriesDeviceState {
        cuda: CudaEri,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_ma_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_bull: DeviceBuffer<f32>,
        d_bear: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: i32,
        grid: GridSize,
        block: BlockSize,
    }
    impl CudaBenchState for ManySeriesDeviceState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("eri_many_series_one_param_time_major_f32")
                .expect("eri_many_series_one_param_time_major_f32");
            unsafe {
                let mut h = self.d_high_tm.as_device_ptr().as_raw();
                let mut l = self.d_low_tm.as_device_ptr().as_raw();
                let mut m = self.d_ma_tm.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut fv = self.d_first.as_device_ptr().as_raw();
                let mut p = self.period;
                let mut bo = self.d_bull.as_device_ptr().as_raw();
                let mut ro = self.d_bear.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 9] = [
                    &mut h as *mut _ as *mut c_void,
                    &mut l as *mut _ as *mut c_void,
                    &mut m as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut fv as *mut _ as *mut c_void,
                    &mut p as *mut _ as *mut c_void,
                    &mut bo as *mut _ as *mut c_void,
                    &mut ro as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, &mut args)
                    .expect("eri many launch");
            }
            self.cuda.stream.synchronize().expect("eri many sync");
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaEri::new(0).expect("cuda eri");
        let close = gen_series(LEN_1M);
        let (high, low) = synth_hl_from_close(&close);
        let sweep = EriBatchRange {
            period: (8, 64, 8),
            ma_type: "ema".to_string(),
        };
        let periods = CudaEri::expand_periods(&sweep).expect("expand_periods");
        let len = close.len();
        let first_valid = close.iter().position(|v| v.is_finite()).unwrap_or(0);

        let range = crate::indicators::moving_averages::ema::EmaBatchRange {
            period: sweep.period,
        };
        let cuda_ma = crate::cuda::moving_averages::ema_wrapper::CudaEma::new(0).expect("cuda ema");
        let ma_rm = cuda_ma.ema_batch_dev(&close, &range).expect("ema_batch_dev");
        let d_ma_rm = ma_rm.buf;

        let d_high = DeviceBuffer::from_slice(&high).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&low).expect("d_low");
        let rows = periods.len();
        let d_bull: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * len) }.expect("d_bull");
        let d_bear: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * len) }.expect("d_bear");

        let block_x = 256u32;
        let grid: GridSize = (((len as u32 + block_x - 1) / block_x).max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(BatchDeviceState {
            cuda,
            d_high,
            d_low,
            d_ma_rm,
            d_bull,
            d_bear,
            len,
            first_valid,
            periods: periods.into_iter().map(|p| p as i32).collect(),
            grid,
            block,
        })
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaEri::new(0).expect("cuda eri");
        let cols = COLS_512;
        let rows = ROWS_16K;
        let close_tm = {
            let mut v = vec![f32::NAN; cols * rows];
            for s in 0..cols {
                for t in s..rows {
                    let x = (t as f32) + (s as f32) * 0.2;
                    v[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                }
            }
            v
        };
        let (high_tm, low_tm) = synth_hl_from_close(&close_tm);
        let period = 14usize;
        let ma_type = "ema";

        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let ma_dev = cuda
            .ma_many_series_one_param_time_major_dev(&close_tm, cols, rows, period, ma_type)
            .expect("ma tm");
        let d_ma_tm = ma_dev.buf;

        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let total = cols * rows;
        let d_bull: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }.expect("d_bull");
        let d_bear: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }.expect("d_bear");

        let block_x = 256u32;
        let grid: GridSize = (
            ceil_div(cols as u32, block_x),
            ceil_div(rows as u32, ERI_TIME_TILE),
            1,
        )
            .into();
        let block: BlockSize = (block_x, 1, 1).into();
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(ManySeriesDeviceState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_ma_tm,
            d_first,
            d_bull,
            d_bear,
            cols,
            rows,
            period: period as i32,
            grid,
            block,
        })
    }

    
    fn bytes_many() -> usize {
        (3 * COLS_512 * ROWS_16K + COLS_512 + 2 * COLS_512 * ROWS_16K) * std::mem::size_of::<f32>()
            + 64 * 1024 * 1024
    }
    fn bytes_batch() -> usize {
        (3 * LEN_1M + 2 * ((64 - 8) / 8 + 1) * LEN_1M) * std::mem::size_of::<f32>()
            + 64 * 1024 * 1024
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("eri", "batch", "eri_cuda_batch", "1m", prep_batch)
                .with_mem_required(bytes_batch()),
            
            CudaBenchScenario::new(
                "eri",
                "many_series_one_param",
                "eri_cuda_many_series",
                "16k x 512",
                prep_many,
            )
            .with_mem_required(bytes_many()),
        ]
    }
}
