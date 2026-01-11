//! CUDA scaffolding for ROCR (Rate of Change Ratio).
//!
//! Mirrors ALMA wrapper conventions: NON_BLOCKING stream, PTX load with
//! DetermineTargetFromContext + O2 (with fallbacks), simple policies, VRAM
//! checks + grid.y chunking, and public device entry points for:
//! - One-series × many-params (batch)
//! - Many-series × one-param (time-major)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32; 
use crate::indicators::rocr::RocrBatchRange;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaRocrError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
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

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaRocrPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaRocr {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaRocrPolicy,
    debug_once: AtomicBool,
    sm_count: u32,
}

impl CudaRocr {
    pub fn new(device_id: usize) -> Result<Self, CudaRocrError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/rocr_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) =
                    Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
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
            policy: CudaRocrPolicy::default(),
            debug_once: AtomicBool::new(false),
            sm_count,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaRocrPolicy) -> Result<Self, CudaRocrError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaRocrError> {
        Ok(self.stream.synchronize()?)
    }

    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    
    pub fn prepare_inv_device(
        &self,
        d_data: &DeviceBuffer<f32>,
        len: usize,
        d_inv_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRocrError> {
        let func = self
            .module
            .get_function("rocr_prepare_inv_f32")
            .map_err(|_| CudaRocrError::MissingKernelSymbol {
                name: "rocr_prepare_inv_f32",
            })?;
        let block_x: u32 = 256;
        let grid_x: u32 = ((len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut data_ptr = d_data.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut inv_ptr = d_inv_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }
    

    fn expand_grid(range: &RocrBatchRange) -> Result<Vec<usize>, CudaRocrError> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            let mut v = Vec::new();
            let mut x = start;
            while x <= end {
                v.push(x);
                x = match x.checked_add(st) {
                    Some(next) => next,
                    None => break,
                };
            }
            if v.is_empty() {
                return Err(CudaRocrError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            return Ok(v);
        }
        let st = step.max(1) as isize;
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaRocrError::InvalidInput(format!(
                "Invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(v)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
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

    fn first_valid(data: &[f32]) -> Option<usize> {
        data.iter().position(|v| !v.is_nan())
    }

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaRocrError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaRocrError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    #[inline]
    fn assert_current_device(&self) -> Result<(), CudaRocrError> {
        unsafe {
            let mut dev: i32 = -1;
            cust::sys::cuCtxGetDevice(&mut dev);
            if dev < 0 {
                return Ok(());
            }
            let cur = dev as u32;
            if cur != self.device_id {
                return Err(CudaRocrError::DeviceMismatch {
                    buf: self.device_id,
                    current: cur,
                });
            }
        }
        Ok(())
    }

    pub fn rocr_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &RocrBatchRange,
    ) -> Result<DeviceArrayF32, CudaRocrError> {
        if data_f32.is_empty() {
            return Err(CudaRocrError::InvalidInput("empty data".into()));
        }
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaRocrError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let len = data_f32.len();
        let first = Self::first_valid(data_f32)
            .ok_or_else(|| CudaRocrError::InvalidInput("all values are NaN".into()))?;
        let max_period = *combos.iter().max().unwrap();
        if max_period == 0 || len - first < max_period {
            return Err(CudaRocrError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                len - first
            )));
        }

        
        
        let use_inv = combos.len() >= 3 && len >= 4096;

        
        let elem_f32 = std::mem::size_of::<f32>();
        let prices_bytes = len
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaRocrError::InvalidInput("byte size overflow".into()))?;
        let total_elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaRocrError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = total_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaRocrError::InvalidInput("byte size overflow".into()))?;
        let inv_bytes = if use_inv { prices_bytes } else { 0 };
        let periods_bytes = combos
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaRocrError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(inv_bytes)
            .and_then(|x| x.checked_add(periods_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaRocrError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaRocrError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaRocrError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        
        let d_data = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream)? };
        let periods_i32: Vec<i32> = combos.iter().map(|&p| p as i32).collect();
        let d_periods =
            unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream)? };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total_elems, &self.stream)? };
        
        let mut d_inv_opt: Option<DeviceBuffer<f32>> = None;
        if use_inv {
            let mut d_inv: DeviceBuffer<f32> = unsafe {
                DeviceBuffer::uninitialized_async(len, &self.stream)?
            };
            self.prepare_inv_device(&d_data, len, &mut d_inv)?;
            d_inv_opt = Some(d_inv);
        }

        self.rocr_batch_device(
            &d_data,
            d_inv_opt.as_ref(),
            &d_periods,
            len,
            first,
            combos.len(),
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rocr_batch_device(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_inv_opt: Option<&DeviceBuffer<f32>>, 
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRocrError> {
        if d_periods.len() != n_combos {
            return Err(CudaRocrError::InvalidInput(
                "period buffer length must match n_combos".into(),
            ));
        }
        let expected_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaRocrError::InvalidInput("rows*cols overflow".into()))?;
        if d_out.len() != expected_elems {
            return Err(CudaRocrError::InvalidInput("output length mismatch".into()));
        }

        let func = self
            .module
            .get_function("rocr_batch_f32")
            .map_err(|_| CudaRocrError::MissingKernelSymbol {
                name: "rocr_batch_f32",
            })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256u32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let need_blocks = ((len as u32) + block_x - 1) / block_x;
        let cap_blocks = self.sm_count.saturating_mul(32).max(1);
        let grid_x = need_blocks.min(cap_blocks);

        self.assert_current_device()?;

        
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self
                .debug_once
                .swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                eprintln!(
                    "[DEBUG] ROCR batch kernel: rocr_batch_f32, block_x={}, grid_x={}, n_combos={}",
                    block_x,
                    grid_x.max(1),
                    n_combos
                );
            }
        }

        
        let mut launched = 0usize;
            while launched < n_combos {
            let remain = n_combos - launched;
            let chunk = remain.min(65_535);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid_x.max(1), chunk as u32, 1, block_x, 1, 1)?;

            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let null_f32: u64 = 0;
                let mut inv_ptr = d_inv_opt
                    .map(|b| b.as_device_ptr().as_raw())
                    .unwrap_or(null_f32);
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut periods_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                let mut n_i = chunk as i32;
                let mut out_ptr = d_out.as_device_ptr().add(launched * len).as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut inv_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += chunk;
        }
        Ok(())
    }

    

    pub fn rocr_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaRocrError> {
        if cols == 0 || rows == 0 {
            return Err(CudaRocrError::InvalidInput("cols/rows must be > 0".into()));
        }
        let total_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaRocrError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != total_elems {
            return Err(CudaRocrError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                total_elems
            )));
        }
        if period == 0 {
            return Err(CudaRocrError::InvalidInput("period must be > 0".into()));
        }

        
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaRocrError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (fv as usize) < period {
                return Err(CudaRocrError::InvalidInput(format!(
                    "series {} not enough valid data (needed>={}, valid={})",
                    s,
                    period,
                    rows - (fv as usize)
                )));
            }
            first_valids[s] = fv;
        }

        
        let elem_f32 = std::mem::size_of::<f32>();
        let series_bytes = total_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaRocrError::InvalidInput("byte size overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaRocrError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = series_bytes;
        let required = series_bytes
            .checked_add(first_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaRocrError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaRocrError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaRocrError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_data_tm =
            unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)? };
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(total_elems, &self.stream)?
        };

        self.rocr_many_series_one_param_device(
            &d_data_tm,
            cols,
            rows,
            period,
            &d_first,
            &mut d_out_tm,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out_tm, rows, cols })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rocr_many_series_one_param_device(
        &self,
        d_data_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRocrError> {
        let func = self
            .module
            .get_function("rocr_many_series_one_param_f32")
            .map_err(|_| CudaRocrError::MissingKernelSymbol {
                name: "rocr_many_series_one_param_f32",
            })?;

        
        let (block_x, block_y) = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => (128u32, 4u32),
            ManySeriesKernelPolicy::OneD { block_x } => (block_x.max(64), 4u32),
        };
        let mut grid_x = ((cols as u32) + block_x - 1) / block_x;
        let mut grid_y = ((rows as u32) + block_y - 1) / block_y;
        if grid_y > 65_535 { grid_y = 65_535; } 
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self
                .debug_once
                .swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                eprintln!(
                    "[DEBUG] ROCR many-series kernel: rocr_many_series_one_param_f32, block=({},{}), grid=({},{}), cols={}, rows={}, period={}",
                    block_x, block_y, grid_x.max(1), grid_y.max(1), cols, rows, period
                );
            }
        }
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (block_x, block_y, 1).into();
        self.validate_launch(grid_x.max(1), grid_y.max(1), 1, block_x, block_y, 1)?;
        self.assert_current_device()?;

        unsafe {
            let mut data_ptr = d_data_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "rocr",
                "batch_dev",
                "rocr_60k_x_49combos",
                "rocr_cuda_batch_dev",
                prep_rocr_batch,
            )
            .with_inner_iters(8),
            CudaBenchScenario::new(
                "rocr",
                "many_series_one_param",
                "rocr_250x1m",
                "rocr_cuda_many_series_one_param",
                prep_rocr_many_series,
            )
            .with_inner_iters(2),
        ]
    }

    struct RocrBatchState {
        cuda: CudaRocr,
        d_prices: DeviceBuffer<f32>,
        d_inv: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first: usize,
        n_combos: usize,
    }
    impl CudaBenchState for RocrBatchState {
        fn launch(&mut self) {
            self.cuda
                .rocr_batch_device(
                    &self.d_prices,
                    Some(&self.d_inv),
                    &self.d_periods,
                    self.len,
                    self.first,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("rocr_batch_device");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_rocr_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaRocr::new(0).expect("cuda rocr");
        let len = 60_000usize;
        let mut prices = vec![f32::NAN; len];
        for i in 16..len {
            let x = i as f32;
            prices[i] = (x * 0.00123).sin() + 0.00017 * x;
        }
        let first = prices.iter().position(|v| !v.is_nan()).unwrap_or(0);
        let sweep = RocrBatchRange { period: (5, 245, 5) };
        let combos = super::CudaRocr::expand_grid(&sweep).expect("valid sweep");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|&p| p as i32).collect();
        let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
        
        let mut d_inv: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.expect("d_inv");
        cuda.prepare_inv_device(&d_prices, len, &mut d_inv).expect("prepare_inv_device");
        cuda.synchronize().expect("sync");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let total_elems = n_combos.checked_mul(len).expect("rows*cols overflow");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }.expect("d_out");
        Box::new(RocrBatchState {
            cuda,
            d_prices,
            d_inv,
            d_periods,
            d_out,
            len,
            first,
            n_combos,
        })
    }

    struct RocrManySeriesState {
        cuda: CudaRocr,
        d_data_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for RocrManySeriesState {
        fn launch(&mut self) {
            self.cuda
                .rocr_many_series_one_param_device(
                    &self.d_data_tm,
                    self.cols,
                    self.rows,
                    self.period,
                    &self.d_first,
                    &mut self.d_out_tm,
                )
                .expect("rocr_many_series_one_param_device");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_rocr_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaRocr::new(0).expect("cuda rocr");
        let cols = 250usize;
        let rows = 1_000_000usize;
        let total_elems = cols.checked_mul(rows).expect("cols*rows overflow");
        let mut tm = vec![f32::NAN; total_elems];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.3;
                tm[t * cols + s] = (x * 0.0023).sin() + 0.00011 * x;
            }
        }
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.3;
                tm[t * cols + s] = (x * 0.0023).sin() + 0.00011 * x;
            }
        }
        let period = 21usize;
        let mut first = vec![0i32; cols];
        for s in 0..cols {
            first[s] = s as i32;
        }
        let d_data_tm = DeviceBuffer::from_slice(&tm).expect("d_data_tm");
        let d_first = DeviceBuffer::from_slice(&first).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }.expect("d_out_tm");
        Box::new(RocrManySeriesState {
            cuda,
            d_data_tm,
            d_first,
            d_out_tm,
            cols,
            rows,
            period,
        })
    }
}
