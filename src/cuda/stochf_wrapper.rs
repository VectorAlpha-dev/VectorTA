//! CUDA wrapper for StochF (Fast Stochastic Oscillator: %K and %D).
//!
//! Mirrors ALMA/KDJ integration patterns:
//! - PTX loaded via include_str!(concat!(env!("OUT_DIR"), "/stochf_kernel.ptx"))
//! - Stream NON_BLOCKING; JIT options: DetermineTargetFromContext + O2 with fallbacks
//! - VRAM checks with ~64MB headroom; chunk rows to <= 65_535
//! - Batch (one series × many-params) and many-series (time-major, one-param)
//! - Batch uses WILLR sparse tables for HH/LL queries (shared precompute across rows)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::stochf::{StochfBatchRange, StochfParams};
use crate::indicators::willr::build_willr_gpu_tables;
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaStochfError {
    #[error("CUDA: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

pub struct DeviceArrayF32Pair {
    pub a: DeviceArrayF32,
    pub b: DeviceArrayF32,
}
impl DeviceArrayF32Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.a.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.a.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self { Self::Auto }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self { Self::Auto }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaStochfPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaStochf {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaStochfPolicy,
}

impl CudaStochf {
    pub fn new(device_id: usize) -> Result<Self, CudaStochfError> {
        Self::new_with_policy(device_id, CudaStochfPolicy::default())
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaStochfPolicy,
    ) -> Result<Self, CudaStochfError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/stochf_kernel.ptx"));
        let jit = [
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, &jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy,
        })
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
    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaStochfError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaStochfError::OutOfMemory { required, free, headroom });
            }
            Ok(())
        } else {
            Err(CudaStochfError::InvalidInput(
                "unable to query device memory via mem_get_info()".into(),
            ))
        }
    }

    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    
    pub fn stochf_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &StochfBatchRange,
    ) -> Result<(DeviceArrayF32Pair, Vec<StochfParams>), CudaStochfError> {
        if high_f32.len() != low_f32.len() || high_f32.len() != close_f32.len() {
            return Err(CudaStochfError::InvalidInput("length mismatch".into()));
        }
        let len = high_f32.len();
        if len == 0 {
            return Err(CudaStochfError::InvalidInput("empty input".into()));
        }

        
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaStochfError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            if start < end {
                let mut v = Vec::new();
                let st = step.max(1);
                let mut x = start;
                while x <= end {
                    v.push(x);
                    x = match x.checked_add(st) {
                        Some(next) => next,
                        None => break,
                    };
                }
                if v.is_empty() {
                    return Err(CudaStochfError::InvalidInput(format!(
                        "invalid fastk/fastd range: start={start}, end={end}, step={step}"
                    )));
                }
                return Ok(v);
            }
            
            let mut v = Vec::new();
            let st = step.max(1) as isize;
            let mut x = start as isize;
            let end_i = end as isize;
            while x >= end_i {
                v.push(x as usize);
                x -= st;
            }
            if v.is_empty() {
                return Err(CudaStochfError::InvalidInput(format!(
                    "invalid fastk/fastd range: start={start}, end={end}, step={step}"
                )));
            }
            Ok(v)
        }
        let fastks = axis_usize(sweep.fastk_period).unwrap_or_else(|_| Vec::new());
        let fastds = axis_usize(sweep.fastd_period).unwrap_or_else(|_| Vec::new());
        let mut combos = Vec::<StochfParams>::with_capacity(fastks.len().saturating_mul(fastds.len()));
        for &k in &fastks {
            for &d in &fastds {
                combos.push(StochfParams {
                    fastk_period: Some(k),
                    fastd_period: Some(d),
                    fastd_matype: Some(0),
                });
            }
        }
        if combos.is_empty() {
            return Err(CudaStochfError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        
        let first_valid = (0..len)
            .find(|&i| {
                high_f32[i].is_finite() && low_f32[i].is_finite() && close_f32[i].is_finite()
            })
            .ok_or_else(|| CudaStochfError::InvalidInput("all values NaN".into()))?;
        
        let max_fk = combos
            .iter()
            .map(|p| p.fastk_period.unwrap())
            .max()
            .unwrap();
        if len - first_valid < max_fk {
            return Err(CudaStochfError::InvalidInput(
                "insufficient data after first_valid".into(),
            ));
        }
        let max_fk = combos
            .iter()
            .map(|p| p.fastk_period.unwrap())
            .max()
            .unwrap();
        if len - first_valid < max_fk {
            return Err(CudaStochfError::InvalidInput(
                "insufficient data after first_valid".into(),
            ));
        }

        
        
        let rows = combos.len();
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let in_bytes_close = len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaStochfError::InvalidInput("len*4 overflow".into()))?;
        let params_elems = 3usize
            .checked_mul(rows)
            .ok_or_else(|| CudaStochfError::InvalidInput("rows*3 overflow".into()))?;
        let params_bytes = params_elems
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaStochfError::InvalidInput("params_bytes overflow".into()))?;
        let out_elems = rows
            .checked_mul(len)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| CudaStochfError::InvalidInput("rows*len*2 overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaStochfError::InvalidInput("out_bytes overflow".into()))?;
        
        let tables_overhead = len
            .checked_mul(8)
            .and_then(|n| n.checked_mul(sz_f32))
            .ok_or_else(|| CudaStochfError::InvalidInput("tables_overhead overflow".into()))?;
        let required = in_bytes_close
            .checked_add(tables_overhead)
            .and_then(|v| v.checked_add(params_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaStochfError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        
        let tables = build_willr_gpu_tables(high_f32, low_f32);

        
        let tables_bytes =
            tables.log2.len().saturating_mul(sz_i32) +
            tables.level_offsets.len().saturating_mul(sz_i32) +
            tables.st_max.len().saturating_mul(sz_f32) +
            tables.st_min.len().saturating_mul(sz_f32) +
            tables.nan_psum.len().saturating_mul(sz_i32);
        let required_exact = in_bytes_close
            .saturating_add(tables_bytes)
            .saturating_add(params_bytes)
            .saturating_add(out_bytes);
        let _ = Self::will_fit(required_exact, 64 * 1024 * 1024);

        
        let use_pinned = len >= 131_072;
        let mut pinned_close: Option<LockedBuffer<f32>> = None;
        let d_close = if use_pinned {
            let host = LockedBuffer::from_slice(close_f32)?;
            let mut dc = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
                ?;
            unsafe { dc.async_copy_from(&host, &self.stream) }?;
            pinned_close = Some(host);
            dc
        } else {
            DeviceBuffer::from_slice(close_f32)?
        };

        
        let d_log2 = DeviceBuffer::from_slice(&tables.log2)?;
        let d_offs = DeviceBuffer::from_slice(&tables.level_offsets)?;
        let d_st_max = DeviceBuffer::from_slice(&tables.st_max)?;
        let d_st_min = DeviceBuffer::from_slice(&tables.st_min)?;
        let d_nan_ps = DeviceBuffer::from_slice(&tables.nan_psum)?;

        
        let out_len = rows
            .checked_mul(len)
            .ok_or_else(|| CudaStochfError::InvalidInput("rows*len overflow".into()))?;
        let mut d_k = unsafe { DeviceBuffer::<f32>::uninitialized(out_len) }?;
        let mut d_d = unsafe { DeviceBuffer::<f32>::uninitialized(out_len) }?;

        
        let fk_host: Vec<i32> = combos.iter().map(|p| p.fastk_period.unwrap() as i32).collect();
        let fd_host: Vec<i32> = combos.iter().map(|p| p.fastd_period.unwrap() as i32).collect();
        let mt_host: Vec<i32> = combos.iter().map(|p| p.fastd_matype.unwrap_or(0) as i32).collect();

        let d_fk_all = DeviceBuffer::from_slice(&fk_host)?;
        let d_fd_all = DeviceBuffer::from_slice(&fd_host)?;
        let d_mt_all = DeviceBuffer::from_slice(&mt_host)?;

        
        let mut func: Function = self.module
            .get_function("stochf_batch_f32")
            .map_err(|_| CudaStochfError::MissingKernelSymbol { name: "stochf_batch_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        let combos_per_launch = 65_535usize; 
        let mut row0 = 0usize;
        while row0 < rows {
            let n = (rows - row0).min(combos_per_launch);
            let gx = n as u32;
            let grid: GridSize = (gx, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            if block_x > 1024 || gx == 0 {
                return Err(CudaStochfError::LaunchConfigTooLarge {
                    gx,
                    gy: 1,
                    gz: 1,
                    bx: block_x,
                    by: 1,
                    bz: 1,
                });
            }

            unsafe {
                
                let mut high_ptr: u64 = 0;
                let mut low_ptr:  u64 = 0;
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut log2_ptr = d_log2.as_device_ptr().as_raw();
                let mut offs_ptr = d_offs.as_device_ptr().as_raw();
                let mut stmax_ptr = d_st_max.as_device_ptr().as_raw();
                let mut stmin_ptr = d_st_min.as_device_ptr().as_raw();
                let mut npsum_ptr = d_nan_ps.as_device_ptr().as_raw();
                let mut fk_ptr    = d_fk_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut fd_ptr    = d_fd_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut mt_ptr    = d_mt_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut len_i     = len as i32;
                let mut first_i   = first_valid as i32;
                let mut levels_i  = (tables.level_offsets.len() as i32);
                let mut n_i       = n as i32;
                let mut k_out_ptr = d_k.as_device_ptr().offset((row0 * len) as isize).as_raw();
                let mut d_out_ptr = d_d.as_device_ptr().offset((row0 * len) as isize).as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut log2_ptr as *mut _ as *mut c_void,
                    &mut offs_ptr as *mut _ as *mut c_void,
                    &mut stmax_ptr as *mut _ as *mut c_void,
                    &mut stmin_ptr as *mut _ as *mut c_void,
                    &mut npsum_ptr as *mut _ as *mut c_void,
                    &mut fk_ptr as *mut _ as *mut c_void,
                    &mut fd_ptr as *mut _ as *mut c_void,
                    &mut mt_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut levels_i as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut k_out_ptr as *mut _ as *mut c_void,
                    &mut d_out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    ?;
            }
            row0 += n;
        }

        self.stream
            .synchronize()?;

        Ok((
            DeviceArrayF32Pair {
                a: DeviceArrayF32 { buf: d_k, rows, cols: len },
                b: DeviceArrayF32 { buf: d_d, rows, cols: len },
            },
            combos,
        ))
    }

    
    pub fn stochf_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &StochfParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaStochfError> {
        if cols == 0 || rows == 0 {
            return Err(CudaStochfError::InvalidInput(
                "series dims must be positive".into(),
            ));
        }
        let tm_len = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaStochfError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm_f32.len() != tm_len
            || low_tm_f32.len() != tm_len
            || close_tm_f32.len() != tm_len
        {
            return Err(CudaStochfError::InvalidInput(
                "time-major slices mismatch dims".into(),
            ));
        }
        let fk = params.fastk_period.unwrap_or(5);
        let fd = params.fastd_period.unwrap_or(3);
        let mt = params.fastd_matype.unwrap_or(0);

        
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm_f32[idx].is_finite()
                    && low_tm_f32[idx].is_finite()
                    && close_tm_f32[idx].is_finite()
                {
                    fv = Some(t as i32);
                    break;
                }
            }
            let f =
                fv.ok_or_else(|| CudaStochfError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (f as usize) < fk {
                return Err(CudaStochfError::InvalidInput(format!(
                    "series {} insufficient data for fk {}",
                    s, fk
                )));
            }
            first_valids[s] = f;
        }

        
        let mut pinned_h: Option<LockedBuffer<f32>> = None;
        let mut pinned_l: Option<LockedBuffer<f32>> = None;
        let mut pinned_c: Option<LockedBuffer<f32>> = None;
        let (d_h, d_l, d_c) = if tm_len >= 131_072 {
            let h_h = LockedBuffer::from_slice(high_tm_f32)?;
            let h_l = LockedBuffer::from_slice(low_tm_f32 )?;
            let h_c = LockedBuffer::from_slice(close_tm_f32 )?;
            let mut d_h = unsafe { DeviceBuffer::<f32>::uninitialized_async(tm_len, &self.stream) }?;
            let mut d_l = unsafe { DeviceBuffer::<f32>::uninitialized_async(tm_len, &self.stream) }?;
            let mut d_c = unsafe { DeviceBuffer::<f32>::uninitialized_async(tm_len, &self.stream) }?;
            unsafe { d_h.async_copy_from(&h_h, &self.stream) }?;
            unsafe { d_l.async_copy_from(&h_l, &self.stream) }?;
            unsafe { d_c.async_copy_from(&h_c, &self.stream) }?;
            pinned_h = Some(h_h);
            pinned_l = Some(h_l);
            pinned_c = Some(h_c);
            (d_h, d_l, d_c)
        } else {
            (
                DeviceBuffer::from_slice(high_tm_f32)?,
                DeviceBuffer::from_slice(low_tm_f32 )?,
                DeviceBuffer::from_slice(close_tm_f32 )?,
            )
        };
        let d_fv= DeviceBuffer::from_slice(&first_valids )?;

        let out_len = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaStochfError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_k = unsafe { DeviceBuffer::<f32>::uninitialized(out_len) }?;
        let mut d_d = unsafe { DeviceBuffer::<f32>::uninitialized(out_len) }?;

        
        let mut func: Function = self.module
            .get_function("stochf_many_series_one_param_f32")
            .map_err(|_| CudaStochfError::MissingKernelSymbol { name: "stochf_many_series_one_param_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| CudaStochfError::LaunchConfigTooLarge {
                gx: u32::MAX,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            })?;
        let grid_x = (cols_u32 + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if block_x > 1024 || grid_x == 0 {
            return Err(CudaStochfError::LaunchConfigTooLarge {
                gx: grid_x.max(1),
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut h_ptr = d_h.as_device_ptr().as_raw();
            let mut l_ptr = d_l.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut fk_i = fk as i32;
            let mut fd_i = fd as i32;
            let mut mt_i = mt as i32;
            let mut ko_ptr = d_k.as_device_ptr().as_raw();
            let mut do_ptr = d_d.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut fk_i as *mut _ as *mut c_void,
                &mut fd_i as *mut _ as *mut c_void,
                &mut mt_i as *mut _ as *mut c_void,
                &mut ko_ptr as *mut _ as *mut c_void,
                &mut do_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)
                ?;
        }

        
        self.stream.synchronize()?;
        drop(pinned_h); drop(pinned_l); drop(pinned_c);

        Ok((
            DeviceArrayF32 { buf: d_k, rows, cols },
            DeviceArrayF32 { buf: d_d, rows, cols },
        ))
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::stochf::StochfBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 256; 

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() { continue; }
            let x = i as f32 * 0.0019;
            let off = (0.0031 * x.sin()).abs() + 0.08;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    fn bytes_one_series_many_params() -> usize {
        
        let in_bytes = 1 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct StochfBatchDeviceState {
        cuda: CudaStochf,
        func: Function<'static>,
        d_close: DeviceBuffer<f32>,
        d_log2: DeviceBuffer<i32>,
        d_offs: DeviceBuffer<i32>,
        d_st_max: DeviceBuffer<f32>,
        d_st_min: DeviceBuffer<f32>,
        d_nan_ps: DeviceBuffer<i32>,
        d_fk_all: DeviceBuffer<i32>,
        d_fd_all: DeviceBuffer<i32>,
        d_mt_all: DeviceBuffer<i32>,
        d_k: DeviceBuffer<f32>,
        d_d: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        levels: i32,
        rows: usize,
        block_x: u32,
    }
    impl CudaBenchState for StochfBatchDeviceState {
        fn launch(&mut self) {
            let combos_per_launch = 65_535usize;
            let mut row0 = 0usize;
            while row0 < self.rows {
                let n = (self.rows - row0).min(combos_per_launch);
                let gx = n as u32;
                let grid: GridSize = (gx, 1, 1).into();
                let block: BlockSize = (self.block_x, 1, 1).into();
                unsafe {
                    // Kernel ignores high/low; pass null device pointers
                    let mut high_ptr: u64 = 0;
                    let mut low_ptr: u64 = 0;
                    let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                    let mut log2_ptr = self.d_log2.as_device_ptr().as_raw();
                    let mut offs_ptr = self.d_offs.as_device_ptr().as_raw();
                    let mut stmax_ptr = self.d_st_max.as_device_ptr().as_raw();
                    let mut stmin_ptr = self.d_st_min.as_device_ptr().as_raw();
                    let mut npsum_ptr = self.d_nan_ps.as_device_ptr().as_raw();
                    let mut fk_ptr = self.d_fk_all.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut fd_ptr = self.d_fd_all.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut mt_ptr = self.d_mt_all.as_device_ptr().offset(row0 as isize).as_raw();
                    let mut len_i = self.len as i32;
                    let mut first_i = self.first_valid as i32;
                    let mut levels_i = self.levels;
                    let mut n_i = n as i32;
                    let mut k_out_ptr = self.d_k.as_device_ptr().offset((row0 * self.len) as isize).as_raw();
                    let mut d_out_ptr = self.d_d.as_device_ptr().offset((row0 * self.len) as isize).as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
                        &mut close_ptr as *mut _ as *mut c_void,
                        &mut log2_ptr as *mut _ as *mut c_void,
                        &mut offs_ptr as *mut _ as *mut c_void,
                        &mut stmax_ptr as *mut _ as *mut c_void,
                        &mut stmin_ptr as *mut _ as *mut c_void,
                        &mut npsum_ptr as *mut _ as *mut c_void,
                        &mut fk_ptr as *mut _ as *mut c_void,
                        &mut fd_ptr as *mut _ as *mut c_void,
                        &mut mt_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut levels_i as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut k_out_ptr as *mut _ as *mut c_void,
                        &mut d_out_ptr as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&self.func, grid, block, 0, args)
                        .expect("stochf launch");
                }
                row0 += n;
            }
            self.cuda.stream.synchronize().expect("stochf sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaStochf::new(0).expect("cuda stochf");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        // Sweep fast_k only; keep fast_d fixed to 3
        let sweep = StochfBatchRange {
            fastk_period: (5, 5 + PARAM_SWEEP - 1, 1),
            fastd_period: (3, 3, 0),
        };
        // Build WILLR sparse tables once on host
        let tables = build_willr_gpu_tables(&high, &low);
        let levels = tables.level_offsets.len() as i32;
        let first_valid = (0..ONE_SERIES_LEN)
            .find(|&i| close[i].is_finite())
            .unwrap_or(0);

        // Precompute parameter arrays (same semantics as wrapper)
        let mut fk_host = Vec::with_capacity(PARAM_SWEEP);
        let mut fd_host = Vec::with_capacity(PARAM_SWEEP);
        let mut mt_host = Vec::with_capacity(PARAM_SWEEP);
        for k in sweep.fastk_period.0..=sweep.fastk_period.1 {
            fk_host.push(k as i32);
            fd_host.push(sweep.fastd_period.0 as i32);
            mt_host.push(0i32);
        }
        let rows = fk_host.len();

        // Upload buffers once
        let d_close =
            unsafe { DeviceBuffer::from_slice_async(&close, &cuda.stream) }.expect("d_close");
        let d_log2 = DeviceBuffer::from_slice(&tables.log2).expect("d_log2");
        let d_offs = DeviceBuffer::from_slice(&tables.level_offsets).expect("d_offs");
        let d_st_max = DeviceBuffer::from_slice(&tables.st_max).expect("d_st_max");
        let d_st_min = DeviceBuffer::from_slice(&tables.st_min).expect("d_st_min");
        let d_nan_ps = DeviceBuffer::from_slice(&tables.nan_psum).expect("d_nan_ps");
        let d_fk_all = DeviceBuffer::from_slice(&fk_host).expect("d_fk_all");
        let d_fd_all = DeviceBuffer::from_slice(&fd_host).expect("d_fd_all");
        let d_mt_all = DeviceBuffer::from_slice(&mt_host).expect("d_mt_all");
        let out_len = rows * ONE_SERIES_LEN;
        let d_k: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_len) }.expect("d_k");
        let d_d: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_len) }.expect("d_d");

        let func = cuda
            .module
            .get_function("stochf_batch_f32")
            .expect("stochf_batch_f32");
        let mut func: Function<'static> = unsafe { std::mem::transmute(func) };
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x = match cuda.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(StochfBatchDeviceState {
            cuda,
            func,
            d_close,
            d_log2,
            d_offs,
            d_st_max,
            d_st_min,
            d_nan_ps,
            d_fk_all,
            d_fd_all,
            d_mt_all,
            d_k,
            d_d,
            len: ONE_SERIES_LEN,
            first_valid,
            levels,
            rows,
            block_x,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "stochf",
            "one_series_many_params",
            "stochf_cuda_batch_dev",
            "1m_x_256",
            prep_one_series_many_params,
        )
        .with_mem_required(bytes_one_series_many_params())
        .with_sample_size(10)]
    }
}
