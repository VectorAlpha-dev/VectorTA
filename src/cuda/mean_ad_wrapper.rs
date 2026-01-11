//! CUDA wrapper for Mean Absolute Deviation (MeanAd).
//!
//! Matches ALMA-style API for policy, PTX load, stream, VRAM checks, and
//! public entrypoints. Math pattern is a recurrence per time step:
//! - Rolling SMA via window sum
//! - Rolling mean(|x - SMA|) via a period-length ring buffer
//!
//! Kernels expected:
//! - "mean_ad_batch_f32"
//! - "mean_ad_many_series_one_param_f32"

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32; 
use crate::indicators::mean_ad::{MeanAdBatchRange, MeanAdParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu_sys; 
use std::ffi::c_void;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaMeanAdError {
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
pub struct CudaMeanAdPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaMeanAdPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaMeanAd {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaMeanAdPolicy,
    sm_count: i32,           
    max_smem_per_block: i32, 
}

impl CudaMeanAd {
    pub fn new(device_id: usize) -> Result<Self, CudaMeanAdError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)?;
        let max_smem_per_block = device
            .get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mean_ad_kernel.ptx"));
        
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaMeanAdPolicy::default(),
            sm_count,
            max_smem_per_block,
        })
    }

    pub fn set_policy(&mut self, policy: CudaMeanAdPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaMeanAdPolicy { &self.policy }
    pub fn synchronize(&self) -> Result<(), CudaMeanAdError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMeanAdError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaMeanAdError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
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
    ) -> Result<(), CudaMeanAdError> {
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
            return Err(CudaMeanAdError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MeanAdBatchRange,
    ) -> Result<(Vec<MeanAdParams>, usize, usize, usize), CudaMeanAdError> {
        if data_f32.is_empty() {
            return Err(CudaMeanAdError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("all values are NaN".into()))?;

        
        
        let combos: Vec<MeanAdParams> = {
            let (start, end, step) = sweep.period;
            if step == 0 || start == end {
                vec![MeanAdParams { period: Some(start) }]
            } else if start < end {
                let st = step.max(1);
                let mut v = Vec::new();
                let mut p = start;
                loop {
                    if p > end {
                        break;
                    }
                    v.push(MeanAdParams { period: Some(p) });
                    let next = match p.checked_add(st) {
                        Some(n) => n,
                        None => break,
                    };
                    if next == p {
                        break;
                    }
                    p = next;
                }
                if v.is_empty() {
                    return Err(CudaMeanAdError::InvalidInput(
                        "invalid period range (empty expansion)".into(),
                    ));
                }
                v
            } else {
                
                let st = step.max(1);
                let mut v = Vec::new();
                let mut x = start as isize;
                let end_i = end as isize;
                let st_i = st as isize;
                while x >= end_i {
                    v.push(MeanAdParams { period: Some(x as usize) });
                    x -= st_i;
                }
                if v.is_empty() {
                    return Err(CudaMeanAdError::InvalidInput(
                        "invalid period range (empty expansion)".into(),
                    ));
                }
                v
            }
        };

        let len = data_f32.len();
        let mut max_period = 0usize;
        let valid = len
            .checked_sub(first_valid)
            .ok_or_else(|| CudaMeanAdError::InvalidInput("first_valid out of range".into()))?;
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaMeanAdError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaMeanAdError::InvalidInput(
                    "period exceeds data length".into(),
                ));
            }
            if valid < p {
                return Err(CudaMeanAdError::InvalidInput(
                    "not enough valid data for period".into(),
                ));
            }
            max_period = max_period.max(p);
        }
        Ok((combos, first_valid, len, max_period))
    }

    pub fn mean_ad_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MeanAdBatchRange,
    ) -> Result<DeviceArrayF32, CudaMeanAdError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        
        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let warms_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let total_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let out_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;

        let required = prices_bytes + periods_bytes + warms_bytes + out_bytes;
        let headroom = 64usize * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warms_i32 = Vec::with_capacity(n_combos);
        for prm in &combos {
            let p = prm.period.unwrap();
            periods_i32.push(p as i32);
            let warm = first_valid + 2 * p - 2;
            warms_i32.push(warm as i32);
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;

        self.mean_ad_batch_device(
            &d_prices,
            &d_periods,
            &d_warms,
            series_len,
            first_valid,
            max_period,
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn mean_ad_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMeanAdError> {
        if series_len == 0 {
            return Err(CudaMeanAdError::InvalidInput("empty input".into()));
        }
        if d_prices.len() != series_len {
            return Err(CudaMeanAdError::InvalidInput(
                "device prices length mismatch".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaMeanAdError::InvalidInput(
                "first_valid out of range".into(),
            ));
        }
        let n_combos = d_periods.len();
        if n_combos == 0 {
            return Err(CudaMeanAdError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if d_warms.len() != n_combos {
            return Err(CudaMeanAdError::InvalidInput(
                "warm_indices length mismatch".into(),
            ));
        }
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaMeanAdError::InvalidInput("output size overflow".into()))?;
        if d_out.len() != out_elems {
            return Err(CudaMeanAdError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }
        if max_period == 0 {
            return Err(CudaMeanAdError::InvalidInput(
                "max_period must be > 0".into(),
            ));
        }

        let mut func = self
            .module
            .get_function("mean_ad_batch_f32")
            .map_err(|_| CudaMeanAdError::MissingKernelSymbol {
                name: "mean_ad_batch_f32",
            })?;

        
        
        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x >= 32 => (block_x / 32) * 32,
            _ => 128u32,
        }
        .max(32);
        let mut warps_per_block = (block_x / 32) as usize;

        
        let mut shared_bytes = (max_period * warps_per_block * std::mem::size_of::<f32>()) as u32;
        let smem_limit = (self.max_smem_per_block as usize).max(1);
        if (shared_bytes as usize) > smem_limit {
            warps_per_block = (smem_limit / (max_period * std::mem::size_of::<f32>())).max(1);
            block_x = (warps_per_block * 32) as u32;
            shared_bytes = (max_period * warps_per_block * std::mem::size_of::<f32>()) as u32;
        }

        
        let ceil_div = |a: usize, b: usize| -> usize { (a + b - 1) / b };
        let base_blocks = ceil_div(n_combos, warps_per_block) as u32;
        let min_busy = (self.sm_count.max(1) as u32) * 2; 
        let grid_x = base_blocks.max(min_busy).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;

        
        unsafe {
            let _ = cu_sys::cuFuncSetAttribute(
                func.to_raw(),
                cu_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            );
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warms_ptr = d_warms.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut max_period_i = max_period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warms_ptr as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut max_period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&mut func, grid, block, shared_bytes, &mut args)?;
        }
        Ok(())
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MeanAdParams,
    ) -> Result<(Vec<i32>, usize), CudaMeanAdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMeanAdError::InvalidInput("empty grid".into()));
        }
        let expected_len = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        if data_tm_f32.len() != expected_len {
            return Err(CudaMeanAdError::InvalidInput("data length mismatch".into()));
        }
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaMeanAdError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaMeanAdError::InvalidInput(
                "period exceeds series length".into(),
            ));
        }
        
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut f = -1;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    f = t as i32;
                    break;
                }
            }
            firsts[s] = f;
        }
        Ok((firsts, period))
    }

    pub fn mean_ad_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MeanAdParams,
    ) -> Result<DeviceArrayF32, CudaMeanAdError> {
        let (firsts, period) = Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let total_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;

        
        let max_shmem: usize = 48 * 1024; 
        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x as usize,
            _ => 128,
        };
        
        let small_period_max: usize = 64;
        if period > small_period_max {
            block_x = block_x
                .min(max_shmem / (period * std::mem::size_of::<f32>()))
                .max(1);
        }
        let grid_x = ((cols + block_x - 1) / block_x) as u32;
        let block: BlockSize = (block_x as u32, 1, 1).into();
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let shared_bytes = if period > small_period_max {
            (period * block_x * std::mem::size_of::<f32>()) as u32
        } else {
            0
        };
        self.validate_launch(grid_x, 1, 1, block_x as u32, 1, 1)?;

        
        let prices_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let firsts_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let out_bytes = prices_bytes;
        let required = prices_bytes + firsts_bytes + out_bytes;
        let headroom = 64usize * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_firsts = DeviceBuffer::from_slice(&firsts)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;

        let mut func = self
            .module
            .get_function("mean_ad_many_series_one_param_f32")
            .map_err(|_| CudaMeanAdError::MissingKernelSymbol {
                name: "mean_ad_many_series_one_param_f32",
            })?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&mut func, grid, block, shared_bytes, &mut args)?;
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    fn mean_ad_many_series_one_param_time_major_device_inplace(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_firsts: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMeanAdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMeanAdError::InvalidInput("empty grid".into()));
        }
        if period == 0 {
            return Err(CudaMeanAdError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaMeanAdError::InvalidInput(
                "period exceeds series length".into(),
            ));
        }
        let total_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        if d_prices_tm.len() != total_elems {
            return Err(CudaMeanAdError::InvalidInput(
                "device prices buffer wrong length".into(),
            ));
        }
        if d_firsts.len() != cols {
            return Err(CudaMeanAdError::InvalidInput(
                "device first_valids buffer wrong length".into(),
            ));
        }
        if d_out_tm.len() != total_elems {
            return Err(CudaMeanAdError::InvalidInput(
                "device output buffer wrong length".into(),
            ));
        }

        
        let max_shmem: usize = 48 * 1024; 
        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x as usize,
            _ => 128,
        };
        
        let small_period_max: usize = 64;
        if period > small_period_max {
            block_x = block_x
                .min(max_shmem / (period * std::mem::size_of::<f32>()))
                .max(1);
        }
        let grid_x = ((cols + block_x - 1) / block_x) as u32;
        let block: BlockSize = (block_x as u32, 1, 1).into();
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let shared_bytes = if period > small_period_max {
            (period * block_x * std::mem::size_of::<f32>()) as u32
        } else {
            0
        };
        self.validate_launch(grid_x, 1, 1, block_x as u32, 1, 1)?;

        let mut func = self
            .module
            .get_function("mean_ad_many_series_one_param_f32")
            .map_err(|_| CudaMeanAdError::MissingKernelSymbol {
                name: "mean_ad_many_series_one_param_f32",
            })?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&mut func, grid, block, shared_bytes, &mut args)?;
        }

        Ok(())
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct MeanAdBatchDeviceState {
        cuda: CudaMeanAd,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_warms: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        max_period: usize,
    }

    impl CudaBenchState for MeanAdBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .mean_ad_batch_device(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_warms,
                    self.series_len,
                    self.first_valid,
                    self.max_period,
                    &mut self.d_out,
                )
                .expect("mean_ad_batch_device");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMeanAd::new(0).expect("cuda mean_ad");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = MeanAdBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (combos, first_valid, series_len, max_period) =
            CudaMeanAd::prepare_batch_inputs(&price, &sweep).expect("prep");

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut warms_i32 = Vec::with_capacity(combos.len());
        for prm in &combos {
            let p = prm.period.unwrap() as usize;
            periods_i32.push(p as i32);
            let warm = first_valid + 2 * p - 2;
            warms_i32.push(warm as i32);
        }

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("d_warms");
        let out_elems = combos.len() * series_len;
        let d_out = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }.expect("d_out");

        Box::new(MeanAdBatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            d_warms,
            d_out,
            series_len,
            first_valid,
            max_period,
        })
    }

    struct MeanAdManySeriesState {
        cuda: CudaMeanAd,
        d_prices_tm: DeviceBuffer<f32>,
        d_firsts: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MeanAdManySeriesState {
        fn launch(&mut self) {
            self.cuda
                .mean_ad_many_series_one_param_time_major_device_inplace(
                    &self.d_prices_tm,
                    &self.d_firsts,
                    self.cols,
                    self.rows,
                    self.period,
                    &mut self.d_out_tm,
                )
                .expect("mean_ad many-series launch");
            self.cuda.synchronize().expect("mean_ad many-series sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cols = 250usize;
        let rows = 1_000_000usize;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = MeanAdParams { period: Some(32) };
        let (firsts, period) =
            CudaMeanAd::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("mean_ad prepare many-series inputs");

        let cuda = CudaMeanAd::new(0).expect("cuda mean_ad");
        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_firsts = DeviceBuffer::from_slice(&firsts).expect("d_firsts");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.synchronize().expect("mean_ad many prep sync");

        Box::new(MeanAdManySeriesState {
            cuda,
            d_prices_tm,
            d_firsts,
            cols,
            rows,
            period,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "mean_ad",
                "one_series_many_params",
                "mean_ad_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "mean_ad",
                "many_series_one_param",
                "mean_ad_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(
                (2 * 250usize * 1_000_000usize) * std::mem::size_of::<f32>()
                    + 250usize * std::mem::size_of::<i32>()
                    + 64 * 1024 * 1024,
            ),
        ]
    }
}
