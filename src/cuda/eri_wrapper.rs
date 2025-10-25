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

const ERI_TIME_TILE: u32 = 16; // must match kernels/cuda/eri_kernel.cu
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

#[derive(Debug)]
pub enum CudaEriError {
    Cuda(String),
    InvalidInput(String),
    UnsupportedMa(String),
}
impl fmt::Display for CudaEriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEriError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEriError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaEriError::UnsupportedMa(e) => write!(f, "Unsupported MA: {}", e),
        }
    }
}
impl std::error::Error for CudaEriError {}

pub struct CudaEri {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaEriPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEri {
    pub fn new(device_id: usize) -> Result<Self, CudaEriError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/eri_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaEriError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
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
    pub fn synchronize(&self) -> Result<(), CudaEriError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaEriError::Cuda(e.to_string()))
    }
    

    #[inline]
    fn device_mem_ok(bytes: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => bytes.saturating_add(64 * 1024 * 1024) <= free,
            Err(_) => true,
        }
    }

    fn expand_periods(sweep: &EriBatchRange) -> Vec<usize> {
        let (start, end, step) = sweep.period;
        if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        }
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

    // ---------- Batch: one series × many params ----------
    pub fn eri_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        source: &[f32],
        sweep: &EriBatchRange,
    ) -> Result<((DeviceArrayF32, DeviceArrayF32), Vec<EriParams>), CudaEriError> {
        // Validate and build combos
        let periods = Self::expand_periods(sweep);
        if periods.is_empty() {
            return Err(CudaEriError::InvalidInput("empty period sweep".into()));
        }
        if periods.is_empty() {
            return Err(CudaEriError::InvalidInput("empty period sweep".into()));
        }
        let max_p = *periods.iter().max().unwrap();
        let first_valid = Self::validate_and_first_valid(high, low, source, max_p)?;
        let len = source.len().min(high.len()).min(low.len());

        // VRAM estimate (approx): inputs + outputs + P*len MA + small temporaries
        let req = (3 * len + 2 * periods.len() * len) * std::mem::size_of::<f32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaEriError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }
        if !Self::device_mem_ok(req) {
            return Err(CudaEriError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload shared inputs
        let d_high = unsafe { DeviceBuffer::from_slice_async(high, &self.stream) }
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low, &self.stream) }
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let d_high = unsafe { DeviceBuffer::from_slice_async(high, &self.stream) }
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low, &self.stream) }
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;

        // Outputs (row-major [P x len])
        let mut d_bull: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(periods.len() * len, &self.stream) }
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let mut d_bear: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(periods.len() * len, &self.stream) }
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;

        // Try to compute MA as a batch on device.
        let ma_type_lc = sweep.ma_type.to_ascii_lowercase();
        let maybe_ma_batch: Option<DeviceArrayF32> = match ma_type_lc.as_str() {
            "ema" => {
                let range = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: sweep.period,
                };
                let range = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::ema_wrapper::CudaEma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                Some(
                    cuda.ema_batch_dev(source, &range)
                        .map_err(|e| CudaEriError::Cuda(e.to_string()))?,
                )
            }
            "sma" => {
                let range = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::sma_wrapper::CudaSma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .sma_batch_dev(source, &range)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let range = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::sma_wrapper::CudaSma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .sma_batch_dev(source, &range)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                Some(dev)
            }
            "wma" => {
                let range = crate::indicators::moving_averages::wma::WmaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::wma_wrapper::CudaWma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let range = crate::indicators::moving_averages::wma::WmaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::wma_wrapper::CudaWma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                Some(
                    cuda.wma_batch_dev(source, &range)
                        .map_err(|e| CudaEriError::Cuda(e.to_string()))?,
                )
            }
            "zlema" => {
                let range = crate::indicators::moving_averages::zlema::ZlemaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::zlema_wrapper::CudaZlema::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .zlema_batch_dev(source, &range)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let range = crate::indicators::moving_averages::zlema::ZlemaBatchRange {
                    period: sweep.period,
                };
                let cuda = crate::cuda::moving_averages::zlema_wrapper::CudaZlema::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .zlema_batch_dev(source, &range)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                Some(dev)
            }
            _ => None,
        };

        let mut combos = Vec::with_capacity(periods.len());
        if let Some(ma_rm) = maybe_ma_batch {
            // Expect row-major [P x len]
            debug_assert_eq!(ma_rm.rows, periods.len());
            debug_assert_eq!(ma_rm.cols, len);

            // 1) Transpose MA to time-major [len x P] on device
            let func_tr = self
                .module
                .get_function("transpose_rm_to_tm_32x32_pad_f32")
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
            let mut d_ma_tm: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(periods.len() * len, &self.stream) }
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
            unsafe {
                let mut in_ptr = ma_rm.buf.as_device_ptr().as_raw();
                let mut R = periods.len() as i32; // rows in RM (params)
                let mut C = len as i32; // cols in RM (time)
                let mut out_ptr = d_ma_tm.as_device_ptr().as_raw();
                let block_tr: BlockSize = (32, 32, 1).into();
                let grid_tr: GridSize = (ceil_div(C as u32, 32), ceil_div(R as u32, 32), 1).into();
                let mut args: [*mut c_void; 4] = [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut R as *mut _ as *mut c_void,
                    &mut C as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func_tr, grid_tr, block_tr, 0, &mut args)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
            }
            // Debug logging removed: no indicator-specific env flags.
            drop(ma_rm);

            // 2) Launch optimized ERI kernel once (write row-major outputs)
            let func = self
                .module
                .get_function("eri_one_series_many_params_time_major_f32")
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
            let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
            let d_periods = DeviceBuffer::from_slice(&periods_i32)
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;

            let block_x = match self.policy.batch {
                BatchKernelPolicy::Auto => 256,
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
                let mut per_ptr = d_periods.as_device_ptr().as_raw(); // [P]
                let mut per_fallback = 0i32; // ignored because per_ptr != nullptr
                let mut bo = d_bull.as_device_ptr().as_raw();
                let mut ro = d_bear.as_device_ptr().as_raw();
                let mut out_rm = 1i32; // write outputs as row-major
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
                self.stream
                    .launch(&func, grid, block, 0, &mut args)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
            }

            combos.extend(periods.iter().map(|&p| EriParams {
                period: Some(p),
                ma_type: Some(sweep.ma_type.clone()),
            }));
        } else {
            // Fallback: per-row MA via selector (inputs already uploaded)
            let func = self
                .module
                .get_function("eri_batch_f32")
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
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
            let selector = CudaMaSelector::new(0);
            for (row_idx, &p) in periods.iter().enumerate() {
                let ma_dev = selector
                    .ma_to_device(&sweep.ma_type, CudaMaData::SliceF32(source), p)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                debug_assert_eq!(ma_dev.rows, 1);
                debug_assert_eq!(ma_dev.cols, len);
                unsafe {
                    let mut h = d_high.as_device_ptr().as_raw();
                    let mut l = d_low.as_device_ptr().as_raw();
                    let mut m = ma_dev.buf.as_device_ptr().as_raw();
                    let mut n = len as i32;
                    let mut fv = first_valid as i32;
                    let mut per = p as i32;
                    let row_off_bytes = (row_idx * len * std::mem::size_of::<f32>()) as u64;
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
                    self.stream
                        .launch(&func, grid, block, 0, &mut args)
                        .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                    self.stream
                        .launch(&func, grid, block, 0, &mut args)
                        .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                }
                combos.push(EriParams { period: Some(p), ma_type: Some(sweep.ma_type.clone()) });
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let bull = DeviceArrayF32 { buf: d_bull, rows: periods.len(), cols: len };
        let bear = DeviceArrayF32 { buf: d_bear, rows: periods.len(), cols: len };
        Ok(((bull, bear), combos))
    }

    // ---------- Many-series × one param (time-major) ----------
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

        // Per-series triple-valid first_valids
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !source_tm[idx].is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            // Validate warmup
            if (first_valids[s] as usize) + period - 1 >= rows {
                return Err(CudaEriError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
                return Err(CudaEriError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        // VRAM: inputs + first_valids + outputs + MA
        let req = (3 * cols * rows + cols + 2 * cols * rows) * std::mem::size_of::<f32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaEriError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }
        if !Self::device_mem_ok(req) {
            return Err(CudaEriError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload inputs
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;

        // Compute MA TM on device using specific wrappers
        let ma_dev =
            self.ma_many_series_one_param_time_major_dev(source_tm, cols, rows, period, ma_type)?;
        let ma_dev =
            self.ma_many_series_one_param_time_major_dev(source_tm, cols, rows, period, ma_type)?;

        // Allocate outputs
        let mut d_bull: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let mut d_bear: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let mut d_bull: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let mut d_bear: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;

        // Launch kernel
        let func = self
            .module
            .get_function("eri_many_series_one_param_time_major_f32")
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        // 2D grid: x over series (cols), y over time tiles (rows / ERI_TIME_TILE)
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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
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
        // Map common MA types; extend as needed.
        match t.as_str() {
            "ema" => {
                let cuda = crate::cuda::moving_averages::ema_wrapper::CudaEma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let params = crate::indicators::moving_averages::ema::EmaParams { period: Some(period) };
                let cuda = crate::cuda::moving_averages::ema_wrapper::CudaEma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                cuda.ema_many_series_one_param_time_major_dev(source_tm, cols, rows, &params)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))
            }
            "sma" => {
                let params = crate::indicators::moving_averages::sma::SmaParams {
                    period: Some(period),
                };
                let cuda = crate::cuda::moving_averages::sma_wrapper::CudaSma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let dev = cuda
                    .sma_multi_series_one_param_time_major_dev(source_tm, cols, rows, &params)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                Ok(dev)
            }
            "wma" => {
                let params = crate::indicators::moving_averages::wma::WmaParams {
                    period: Some(period),
                };
                let cuda = crate::cuda::moving_averages::wma_wrapper::CudaWma::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                cuda.wma_multi_series_one_param_time_major_dev(source_tm, cols, rows, &params)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))
            }
            "zlema" => {
                let params = crate::indicators::moving_averages::zlema::ZlemaParams {
                    period: Some(period),
                };
                let cuda = crate::cuda::moving_averages::zlema_wrapper::CudaZlema::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                let params = crate::indicators::moving_averages::zlema::ZlemaParams {
                    period: Some(period),
                };
                let cuda = crate::cuda::moving_averages::zlema_wrapper::CudaZlema::new(0)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))?;
                cuda.zlema_many_series_one_param_time_major_dev(source_tm, cols, rows, &params)
                    .map_err(|e| CudaEriError::Cuda(e.to_string()))
            }
            _ => Err(CudaEriError::UnsupportedMa(t)),
        }
    }
}

// ---------- Benches ----------
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

    struct BatchState {
        cuda: CudaEri,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: EriBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .eri_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .unwrap();
        }
    }

    struct ManySeriesState {
        cuda: CudaEri,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        ma_type: &'static str,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .eri_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    self.period,
                    self.ma_type,
                )
                .unwrap();
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
        Box::new(BatchState {
            cuda,
            high,
            low,
            close,
            sweep,
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
        Box::new(ManySeriesState { cuda, high_tm, low_tm, close_tm, cols, rows, period, ma_type })
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
