#![cfg(feature = "cuda")]

//! CUDA wrapper for TTM Trend (close > SMA(source, period)).
//!
//! Parity with ALMA wrapper policy:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/ttm_trend_kernel.ptx"))
//!   with DetermineTargetFromContext + OptLevel O2, and simpler fallbacks.
//! - NON_BLOCKING stream.
//! - VRAM check with ~64MB headroom and grid.y chunking kept under 65_535.
//! - Public device entry points:
//!   - `ttm_trend_batch_dev(&[f32], &[f32], &TtmTrendBatchRange) -> DeviceArrayF32`
//!   - `ttm_trend_many_series_one_param_time_major_dev(&[f32], &[f32], cols, rows, period)`

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::ttm_trend::TtmTrendBatchRange;
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaTtmTrendError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
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

#[derive(Clone, Debug)]
struct Combo {
    period: i32,
    warm: i32,
}

pub struct CudaTtmTrend {
    pub(crate) module: Module,
    pub(crate) stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaTtmTrend {
    pub fn new(device_id: usize) -> Result<Self, CudaTtmTrendError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ttm_trend_kernel.ptx"));
        
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> Result<(), CudaTtmTrendError> {
        match mem_get_info() {
            Ok((free, _)) => {
                if required_bytes.saturating_add(headroom) <= free {
                    Ok(())
                } else {
                    Err(CudaTtmTrendError::OutOfMemory {
                        required: required_bytes,
                        free,
                        headroom,
                    })
                }
            }
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn validate_launch_dims(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaTtmTrendError> {
        let dev = Device::get_device(self.device_id)?;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let max_threads = dev.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaTtmTrendError::InvalidInput(
                "zero-sized grid or block".into(),
            ));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz {
            return Err(CudaTtmTrendError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        let threads = bx
            .checked_mul(by)
            .and_then(|v| v.checked_mul(bz))
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("block size overflow".into()))?;
        if threads > max_threads {
            return Err(CudaTtmTrendError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    fn expand_grid(range: &TtmTrendBatchRange) -> Result<Vec<i32>, CudaTtmTrendError> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return Ok(vec![start as i32]);
        }
        if start < end {
            let st = step.max(1);
            let v: Vec<i32> = (start..=end).step_by(st).map(|p| p as i32).collect();
            if v.is_empty() {
                return Err(CudaTtmTrendError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as i32);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaTtmTrendError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(v)
    }

    fn prepare_batch_inputs(
        source_f32: &[f32],
        close_f32: &[f32],
        sweep: &TtmTrendBatchRange,
    ) -> Result<(Vec<Combo>, usize, usize), CudaTtmTrendError> {
        if source_f32.len() != close_f32.len() {
            return Err(CudaTtmTrendError::InvalidInput(
                "source/close length mismatch".into(),
            ));
        }
        let len = source_f32.len();
        if len == 0 {
            return Err(CudaTtmTrendError::InvalidInput("empty inputs".into()));
        }
        let first = source_f32
            .iter()
            .zip(close_f32)
            .position(|(&s, &c)| !s.is_nan() && !c.is_nan())
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("all values are NaN".into()))?;
        let periods = Self::expand_grid(sweep)?;
        let mut combos = Vec::with_capacity(periods.len());
        for &p in &periods {
            let pu = p as usize;
            if pu == 0 || pu > len {
                return Err(CudaTtmTrendError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    pu, len
                )));
            }
            if len - first < pu {
                return Err(CudaTtmTrendError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    pu,
                    len - first
                )));
            }
            let warm = (first + pu - 1) as i32;
            combos.push(Combo { period: p, warm });
        }
        Ok((combos, first, len))
    }

    
    
    fn build_prefix_source_ff2(source_f32: &[f32], first_valid: usize) -> Vec<[f32; 2]> {
        #[inline]
        fn two_sum(a: f32, b: f32) -> (f32, f32) {
            let s = a + b;
            let bb = s - a;
            let e = (a - (s - bb)) + (b - bb);
            (s, e)
        }

        let n = source_f32.len();
        let mut pref = vec![[0.0f32, 0.0f32]; n];
        if first_valid < n {
            
            let mut hi: f32 = 0.0;
            let mut lo: f32 = 0.0;
            for i in first_valid..n {
                let (s, mut e) = two_sum(hi, source_f32[i]);
                e += lo;
                let (rhi, rlo) = two_sum(s, e);
                hi = rhi; lo = rlo;
                pref[i] = [hi, lo];
            }
        }
        pref
    }

    #[inline]
    fn chunk_rows(n_rows: usize) -> usize {
        
        let max_grid_y = 65_000usize;
        max_grid_y.min(n_rows).max(1)
    }

    pub fn ttm_trend_batch_dev(
        &self,
        source_f32: &[f32],
        close_f32: &[f32],
        sweep: &TtmTrendBatchRange,
    ) -> Result<DeviceArrayF32, CudaTtmTrendError> {
        let (combos, first, len) = Self::prepare_batch_inputs(source_f32, close_f32, sweep)?;
        let n_combos = combos.len();
        let elem_ff2 = std::mem::size_of::<[f32; 2]>();
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();

        
        let prefix_bytes = len
            .checked_mul(elem_ff2)
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("size overflow in prefix bytes".into()))?;
        let close_bytes = len
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("size overflow in close bytes".into()))?;
        let params_bytes = n_combos
            .checked_mul(elem_i32)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("size overflow in params bytes".into()))?;
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("size overflow in output bytes".into()))?;
        let logical = prefix_bytes
            .checked_add(close_bytes)
            .and_then(|x| x.checked_add(params_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64usize << 20;
        Self::will_fit(logical, headroom)?;

        
        let prefix_ff2 = Self::build_prefix_source_ff2(source_f32, first);

        
        let d_prefix: DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&prefix_ff2)?;
        let d_close  = DeviceBuffer::from_slice(close_f32)?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        let warms: Vec<i32> = combos.iter().map(|c| c.warm).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_warms = DeviceBuffer::from_slice(&warms)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        
        let mut func = self
            .module
            .get_function("ttm_trend_batch_prefix_ff2_tiled")
            .map_err(|_| CudaTtmTrendError::MissingKernelSymbol {
                name: "ttm_trend_batch_prefix_ff2_tiled",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        
        const TTM_TILE_TIME: u32 = 256;
        const TTM_TILE_PARAMS: u32 = 4;
        let grid_x: u32 = ((len as u32) + TTM_TILE_TIME - 1) / TTM_TILE_TIME;
        let grid_y: u32 = ((n_combos as u32) + TTM_TILE_PARAMS - 1) / TTM_TILE_PARAMS;
        let gx = grid_x.max(1);
        let gy = grid_y.max(1);
        let grid: GridSize = (gx, gy, 1).into();
        let block: BlockSize = (TTM_TILE_TIME, TTM_TILE_PARAMS, 1).into();
        self.validate_launch_dims((gx, gy, 1), (TTM_TILE_TIME, TTM_TILE_PARAMS, 1))?;
        unsafe {
            let mut pref_ptr = d_prefix.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut per_ptr   = d_periods.as_device_ptr().as_raw();
            let mut warm_ptr  = d_warms.as_device_ptr().as_raw();
            let mut len_i     = len as i32;
            let mut ncomb_i   = n_combos as i32;
            let mut out_ptr   = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pref_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut per_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut ncomb_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    pub fn ttm_trend_many_series_one_param_time_major_dev(
        &self,
        source_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTtmTrendError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTtmTrendError::InvalidInput(
                "cols/rows must be > 0".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("rows*cols overflow".into()))?;
        if source_tm_f32.len() != expected || close_tm_f32.len() != expected {
            return Err(CudaTtmTrendError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaTtmTrendError::InvalidInput("invalid period".into()));
        }

        
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let si = source_tm_f32[t * cols + s];
                let ci = close_tm_f32[t * cols + s];
                if !si.is_nan() && !ci.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaTtmTrendError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (fv as usize) < period {
                return Err(CudaTtmTrendError::InvalidInput(format!(
                    "series {} not enough valid data: needed >= {}, valid = {}",
                    s,
                    period,
                    rows - fv as usize
                )));
            }
            first_valids[s] = fv;
        }

        
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let series_elems = expected;
        let in_bytes = series_elems
            .checked_mul(elem_f32)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("size overflow in input bytes".into()))?;
        let fv_bytes = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("size overflow in first_valid bytes".into()))?;
        let out_bytes = series_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("size overflow in output bytes".into()))?;
        let logical = in_bytes
            .checked_add(fv_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64usize << 20;
        Self::will_fit(logical, headroom)?;

        
        let d_src = DeviceBuffer::from_slice(source_tm_f32)?;
        let d_close = DeviceBuffer::from_slice(close_tm_f32)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }?;
        
        unsafe { d_out.set_zero_async(&self.stream) }?;

        let mut func = self
            .module
            .get_function("ttm_trend_many_series_one_param_time_major_f32")
            .map_err(|_| CudaTtmTrendError::MissingKernelSymbol {
                name: "ttm_trend_many_series_one_param_time_major_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        
        let auto_block = || -> u32 {
            let (_min_grid, bs) = func
                .suggested_launch_configuration(0, (0, 0, 0).into())
                .unwrap_or((0, 128));
            let bs = ((bs + 31) / 32) * 32;
            bs.max(32).min(256)
        };
        let block_x: u32 = auto_block();
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let gx = grid_x.max(1);
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch_dims((gx, 1, 1), (block_x, 1, 1))?;
        unsafe {
            let mut src_ptr = d_src.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p_i = period as i32;
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p_i = period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut src_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaTtmTrendError> {
        self.stream.synchronize().map_err(Into::into)
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    fn gen_series(n: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; n];
        for i in 8..n {
            let x = i as f32;
            v[i] = (x * 0.00123).sin() + 0.00031 * x;
        }
        v
    }

    struct TtmBatchState {
        cuda: CudaTtmTrend,
        d_prefix: DeviceBuffer<[f32; 2]>,
        d_close: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_warms: DeviceBuffer<i32>,
        len: usize,
        n_combos: usize,
        grid: GridSize,
        block: BlockSize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TtmBatchState {
        fn launch(&mut self) {
            let mut func = self
                .cuda
                .module
                .get_function("ttm_trend_batch_prefix_ff2_tiled")
                .expect("ttm_trend_batch_prefix_ff2_tiled");
            let _ = func.set_cache_config(CacheConfig::PreferL1);
            unsafe {
                let mut pref_ptr = self.d_prefix.as_device_ptr().as_raw();
                let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                let mut per_ptr = self.d_periods.as_device_ptr().as_raw();
                let mut warm_ptr = self.d_warms.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut ncomb_i = self.n_combos as i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut pref_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut warm_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, args)
                    .expect("ttm_trend batch launch");
            }
            self.cuda.stream.synchronize().expect("ttm_trend batch sync");
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaTtmTrend::new(0).expect("cuda ttm");
        let len = 1_000_000usize; 
        let src = gen_series(len);
        
        let mut close = vec![f32::NAN; len];
        for i in 8..len {
            let x = i as f32;
            close[i] = src[i] + 0.05 * (x * 0.00071).cos();
        }
        let sweep = TtmTrendBatchRange {
            period: (5, 254, 1),
        };

        let (combos, first, len) = CudaTtmTrend::prepare_batch_inputs(&src, &close, &sweep)
            .expect("prepare_batch_inputs");
        let n_combos = combos.len();
        let prefix_ff2 = CudaTtmTrend::build_prefix_source_ff2(&src, first);

        let d_prefix: DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&prefix_ff2).expect("d_prefix");
        let d_close = DeviceBuffer::from_slice(&close).expect("d_close");
        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        let warms: Vec<i32> = combos.iter().map(|c| c.warm).collect();
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_warms = DeviceBuffer::from_slice(&warms).expect("d_warms");
        let out_elems = n_combos.checked_mul(len).expect("rows*cols overflow");
        let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("d_out");

        
        const TTM_TILE_TIME: u32 = 256;
        const TTM_TILE_PARAMS: u32 = 4;
        let grid_x: u32 = ((len as u32) + TTM_TILE_TIME - 1) / TTM_TILE_TIME;
        let grid_y: u32 = ((n_combos as u32) + TTM_TILE_PARAMS - 1) / TTM_TILE_PARAMS;
        let gx = grid_x.max(1);
        let gy = grid_y.max(1);
        let grid: GridSize = (gx, gy, 1).into();
        let block: BlockSize = (TTM_TILE_TIME, TTM_TILE_PARAMS, 1).into();

        cuda.stream.synchronize().expect("ttm_trend prep sync");
        Box::new(TtmBatchState {
            cuda,
            d_prefix,
            d_close,
            d_periods,
            d_warms,
            len,
            n_combos,
            grid,
            block,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "ttm_trend",
            "one_series_many_params",
            "ttm_trend_cuda_batch_dev",
            "1m_x_250",
            prep_batch,
        )]
    }
}
