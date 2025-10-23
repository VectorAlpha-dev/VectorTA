//! CUDA support for the WaveTrend Oscillator (WTO).
//!
//! This module mirrors the CUDA integration style used by the ALMA and CWMA
//! indicators: kernels execute in single precision, results remain on the
//! device until staged out by the caller, and memory safety checks are kept
//! conservative to avoid VRAM over-commitment.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::wto::{WtoBatchRange, WtoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::sync::atomic::{AtomicBool, Ordering};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaWtoError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWtoError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWtoError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for CudaWtoError {}

pub struct DeviceArrayF32Triplet {
    pub wt1: DeviceArrayF32,
    pub wt2: DeviceArrayF32,
    pub hist: DeviceArrayF32,
}

impl DeviceArrayF32Triplet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.wt1.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.wt1.cols
    }
}

pub struct CudaWtoBatchResult {
    pub outputs: DeviceArrayF32Triplet,
    pub combos: Vec<WtoParams>,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaWtoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaWto {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaWtoPolicy,
    debug_batch_logged: AtomicBool,
    debug_many_logged: AtomicBool,
}

impl CudaWto {
    pub fn new(device_id: usize) -> Result<Self, CudaWtoError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wto_kernel.ptx"));
        // Prefer context-targeted JIT with highest opt level, fallback to simpler modes
        let module = match Module::from_ptx(
            ptx,
            &[ModuleJitOption::DetermineTargetFromContext, ModuleJitOption::OptLevel(OptLevel::O4)],
        ) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaWtoError::Cuda(e.to_string()))?
                }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaWtoPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto },
            debug_batch_logged: AtomicBool::new(false),
            debug_many_logged: AtomicBool::new(false),
        })
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline] fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }

    // Prefill outputs with canonical qNaN via driver memset (async, ordered on the stream)
    #[inline]
    fn prefill_nan_triplet(
        &self,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWtoError> {
        const QNAN_BITS: u32 = 0x7FC0_0000u32;
        unsafe {
            let st: cu::CUstream = self.stream.as_inner();
            let p1: cu::CUdeviceptr = d_wt1.as_device_ptr().as_raw();
            let p2: cu::CUdeviceptr = d_wt2.as_device_ptr().as_raw();
            let p3: cu::CUdeviceptr = d_hist.as_device_ptr().as_raw();
            let n1 = d_wt1.len();
            let n2 = d_wt2.len();
            let n3 = d_hist.len();
            let r1 = cu::cuMemsetD32Async(p1, QNAN_BITS, n1, st);
            if r1 != cu::CUresult::CUDA_SUCCESS { return Err(CudaWtoError::Cuda(format!("cuMemsetD32Async wt1 failed: {:?}", r1))); }
            let r2 = cu::cuMemsetD32Async(p2, QNAN_BITS, n2, st);
            if r2 != cu::CUresult::CUDA_SUCCESS { return Err(CudaWtoError::Cuda(format!("cuMemsetD32Async wt2 failed: {:?}", r2))); }
            let r3 = cu::cuMemsetD32Async(p3, QNAN_BITS, n3, st);
            if r3 != cu::CUresult::CUDA_SUCCESS { return Err(CudaWtoError::Cuda(format!("cuMemsetD32Async hist failed: {:?}", r3))); }
        }
        Ok(())
    }

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    pub fn wto_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &WtoBatchRange,
    ) -> Result<CudaWtoBatchResult, CudaWtoError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let mut channel_i32 = Vec::with_capacity(n_combos);
        let mut average_i32 = Vec::with_capacity(n_combos);
        for params in &combos {
            channel_i32.push(params.channel_length.unwrap() as i32);
            average_i32.push(params.average_length.unwrap() as i32);
        }

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let params_bytes = 2 * n_combos * std::mem::size_of::<i32>();
        let out_bytes = 3 * n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaWtoError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let d_channel = DeviceBuffer::from_slice(&channel_i32)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let d_average = DeviceBuffer::from_slice(&average_i32)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let mut d_wt1: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let mut d_wt2: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let mut d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;

        // Prefill outputs with NaN (async)
        self.prefill_nan_triplet(&mut d_wt1, &mut d_wt2, &mut d_hist)?;

        self.launch_batch_kernel(
            &d_prices,
            &d_channel,
            &d_average,
            series_len,
            n_combos,
            first_valid,
            &mut d_wt1,
            &mut d_wt2,
            &mut d_hist,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;

        let outputs = DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 {
                buf: d_wt1,
                rows: n_combos,
                cols: series_len,
            },
            wt2: DeviceArrayF32 {
                buf: d_wt2,
                rows: n_combos,
                cols: series_len,
            },
            hist: DeviceArrayF32 {
                buf: d_hist,
                rows: n_combos,
                cols: series_len,
            },
        };

        Ok(CudaWtoBatchResult { outputs, combos })
    }

    pub fn wto_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &WtoBatchRange,
        wt1_host: &mut [f32],
        wt2_host: &mut [f32],
        hist_host: &mut [f32],
    ) -> Result<(usize, usize, Vec<WtoParams>), CudaWtoError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if wt1_host.len() != expected || wt2_host.len() != expected || hist_host.len() != expected {
            return Err(CudaWtoError::InvalidInput(format!(
                "output slices must be len {}",
                expected
            )));
        }
        let CudaWtoBatchResult { outputs, combos } = self.wto_batch_dev(data_f32, sweep)?;
        let DeviceArrayF32Triplet { wt1, wt2, hist } = outputs;
        wt1.buf
            .copy_to(wt1_host)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        wt2.buf
            .copy_to(wt2_host)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        hist.buf
            .copy_to(hist_host)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        Ok((combos.len(), series_len, combos))
    }

    pub fn wto_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WtoParams,
    ) -> Result<DeviceArrayF32Triplet, CudaWtoError> {
        let (first_valids, channel, average) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = 3 * cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaWtoError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let mut d_wt1: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let mut d_wt2: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        let mut d_hist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;

        // Prefill outputs with NaN (async)
        self.prefill_nan_triplet(&mut d_wt1, &mut d_wt2, &mut d_hist)?;

        self.launch_many_series_kernel(
            &d_prices,
            cols,
            rows,
            channel,
            average,
            &d_first,
            &mut d_wt1,
            &mut d_wt2,
            &mut d_hist,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 {
                buf: d_wt1,
                rows,
                cols,
            },
            wt2: DeviceArrayF32 {
                buf: d_wt2,
                rows,
                cols,
            },
            hist: DeviceArrayF32 {
                buf: d_hist,
                rows,
                cols,
            },
        })
    }

    pub fn wto_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WtoParams,
        wt1_tm: &mut [f32],
        wt2_tm: &mut [f32],
        hist_tm: &mut [f32],
    ) -> Result<(), CudaWtoError> {
        let expected = cols * rows;
        if wt1_tm.len() != expected || wt2_tm.len() != expected || hist_tm.len() != expected {
            return Err(CudaWtoError::InvalidInput(format!(
                "output slices must be len {}",
                expected
            )));
        }
        let DeviceArrayF32Triplet { wt1, wt2, hist } =
            self.wto_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        wt1.buf
            .copy_to(wt1_tm)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        wt2.buf
            .copy_to(wt2_tm)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        hist.buf
            .copy_to(hist_tm)
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        Ok(())
    }

    pub fn wto_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_channel: &DeviceBuffer<i32>,
        d_average: &DeviceBuffer<i32>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWtoError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaWtoError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        // Ensure outputs are prefilled in device-only path as well
        self.prefill_nan_triplet(d_wt1, d_wt2, d_hist)?;
        self.launch_batch_kernel(
            d_prices,
            d_channel,
            d_average,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            d_wt1,
            d_wt2,
            d_hist,
        )
    }

    pub fn wto_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: i32,
        rows: i32,
        channel_length: i32,
        average_length: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWtoError> {
        if cols <= 0 || rows <= 0 {
            return Err(CudaWtoError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if channel_length <= 0 || average_length <= 0 {
            return Err(CudaWtoError::InvalidInput(
                "channel_length and average_length must be positive".into(),
            ));
        }
        // Ensure outputs are prefilled in device-only path as well
        self.prefill_nan_triplet(d_wt1, d_wt2, d_hist)?;
        self.launch_many_series_kernel(
            d_prices_tm,
            cols as usize,
            rows as usize,
            channel_length as usize,
            average_length as usize,
            d_first_valids,
            d_wt1,
            d_wt2,
            d_hist,
        )
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &WtoBatchRange,
    ) -> Result<(Vec<WtoParams>, usize, usize), CudaWtoError> {
        if data_f32.is_empty() {
            return Err(CudaWtoError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaWtoError::InvalidInput("all values are NaN".into()))?;
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaWtoError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let len = data_f32.len();
        for params in &combos {
            let ch = params.channel_length.unwrap();
            let av = params.average_length.unwrap();
            if ch == 0 || ch > len {
                return Err(CudaWtoError::InvalidInput(format!(
                    "channel_length {} invalid for data length {}",
                    ch, len
                )));
            }
            if av == 0 || av > len {
                return Err(CudaWtoError::InvalidInput(format!(
                    "average_length {} invalid for data length {}",
                    av, len
                )));
            }
            let needed = av + 3;
            if len - first_valid < needed {
                return Err(CudaWtoError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    needed,
                    len - first_valid
                )));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WtoParams,
    ) -> Result<(Vec<i32>, usize, usize), CudaWtoError> {
        if cols == 0 || rows == 0 {
            return Err(CudaWtoError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaWtoError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let channel = params.channel_length.unwrap_or(0);
        let average = params.average_length.unwrap_or(0);
        if channel == 0 || average == 0 {
            return Err(CudaWtoError::InvalidInput(
                "channel_length and average_length must be > 0".into(),
            ));
        }
        if channel > rows || average > rows {
            return Err(CudaWtoError::InvalidInput(format!(
                "parameters exceed series length: channel={}, average={}, len={}",
                channel, average, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + series];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaWtoError::InvalidInput(format!("series {} consists entirely of NaNs", series))
            })?;
            let needed = average + 3;
            if rows - fv < needed {
                return Err(CudaWtoError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series,
                    needed,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, channel, average))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_channel: &DeviceBuffer<i32>,
        d_average: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWtoError> {
        // Kernel block policy (parity with ALMA’s explicit selection knob)
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1")
            && !self.debug_batch_logged.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[wto] batch kernel: block_x={}, grid_x={}, device={}",
                block_x, grid_x, self.device_id
            );
        }

        let func = self
            .module
            .get_function("wto_batch_f32")
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut channel_ptr = d_channel.as_device_ptr().as_raw();
            let mut average_ptr = d_average.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut wt1_ptr = d_wt1.as_device_ptr().as_raw();
            let mut wt2_ptr = d_wt2.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut channel_ptr as *mut _ as *mut c_void,
                &mut average_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut wt1_ptr as *mut _ as *mut c_void,
                &mut wt2_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        channel_length: usize,
        average_length: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWtoError> {
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1")
            && !self.debug_many_logged.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[wto] many-series kernel: block_x={}, grid_x={}, device={}",
                block_x, grid_x, self.device_id
            );
        }

        let func = self
            .module
            .get_function("wto_many_series_one_param_time_major_f32")
            .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut channel_i = channel_length as i32;
            let mut average_i = average_length as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut wt1_ptr = d_wt1.as_device_ptr().as_raw();
            let mut wt2_ptr = d_wt2.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut channel_i as *mut _ as *mut c_void,
                &mut average_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut wt1_ptr as *mut _ as *mut c_void,
                &mut wt2_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWtoError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        // 3 outputs
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = 3 * elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct WtoBatchState {
        cuda: CudaWto,
        price: Vec<f32>,
        sweep: WtoBatchRange,
    }
    impl CudaBenchState for WtoBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .wto_batch_dev(&self.price, &self.sweep)
                .expect("wto batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaWto::new(0).expect("cuda wto");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = WtoBatchRange {
            channel: (10, 10 + PARAM_SWEEP - 1, 1),
            average: (21, 21, 0),
        };
        Box::new(WtoBatchState { cuda, price, sweep })
    }

    struct WtoManyState {
        cuda: CudaWto,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: WtoParams,
    }
    impl CudaBenchState for WtoManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .wto_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("wto many-series");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaWto::new(0).expect("cuda wto");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = WtoParams {
            channel_length: Some(10),
            average_length: Some(21),
        };
        Box::new(WtoManyState {
            cuda,
            data_tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "wto",
                "one_series_many_params",
                "wto_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "wto",
                "many_series_one_param",
                "wto_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

fn expand_grid(r: &WtoBatchRange) -> Vec<WtoParams> {
    fn axis_u(range: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = range;
        if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect::<Vec<_>>()
        }
    }
    let channels = axis_u(r.channel);
    let averages = axis_u(r.average);
    let mut out = Vec::with_capacity(channels.len() * averages.len());
    for &ch in &channels {
        for &av in &averages {
            out.push(WtoParams {
                channel_length: Some(ch),
                average_length: Some(av),
            });
        }
    }
    out
}
