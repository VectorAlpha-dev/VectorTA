#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::{
    BatchKernelPolicy, BatchKernelSelected, ManySeriesKernelPolicy, ManySeriesKernelSelected,
};
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::tsi::{TsiBatchRange, TsiParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaTsiError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
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

#[derive(Clone, Copy, Debug)]
pub struct CudaTsiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaTsiPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaTsi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaTsiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,

    scratch: Option<TsiScratch>,
}

impl CudaTsi {
    pub fn new(device_id: usize) -> Result<Self, CudaTsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/tsi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
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
            policy: CudaTsiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            scratch: None,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaTsiPolicy) {
        self.policy = p;
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
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn validate_launch_dims(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaTsiError> {
        let dev = Device::get_device(self.device_id)?;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaTsiError::InvalidInput(
                "zero-sized grid or block".into(),
            ));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz {
            return Err(CudaTsiError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        Ok(())
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaTsiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn tsi_batch_dev(
        &mut self,
        prices_f32: &[f32],
        sweep: &TsiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<TsiParams>), CudaTsiError> {
        if prices_f32.is_empty() {
            return Err(CudaTsiError::InvalidInput("empty input".into()));
        }
        let len = prices_f32.len();
        let first_valid = prices_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaTsiError::InvalidInput("all values are NaN/inf".into()))?;

        let combos = expand_grid(sweep)?;

        let mut longs_i32 = Vec::<i32>::with_capacity(combos.len());
        let mut shorts_i32 = Vec::<i32>::with_capacity(combos.len());
        for p in &combos {
            let l = p.long_period.unwrap_or(25);
            let s = p.short_period.unwrap_or(13);
            if l == 0 || s == 0 || l > len || s > len {
                return Err(CudaTsiError::InvalidInput("invalid period in combo".into()));
            }
            let needed = 1 + l + s;
            if len - first_valid < needed {
                return Err(CudaTsiError::InvalidInput(format!(
                    "not enough valid data: need {}, have {}",
                    needed,
                    len - first_valid
                )));
            }
            longs_i32.push(l as i32);
            shorts_i32.push(s as i32);
        }

        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let in_bytes = len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let params_elems = combos
            .len()
            .checked_mul(2usize)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let params_bytes = params_elems
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let out_elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let plain_required = in_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let fast_extra_terms = len
            .checked_mul(2)
            .and_then(|v| v.checked_add(out_elems))
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let fast_extra = fast_extra_terms
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let fast_required = plain_required
            .checked_add(fast_extra)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let head = 64usize * 1024 * 1024;
        let mut free_ok_plain = CudaTsi::will_fit(plain_required, head);
        let mut free_ok_fast = CudaTsi::will_fit(fast_required, head);
        if !free_ok_plain {
            if let Some((free, _)) = CudaTsi::device_mem_info() {
                return Err(CudaTsiError::OutOfMemory {
                    required: plain_required,
                    free,
                    headroom: head,
                });
            } else {
                return Err(CudaTsiError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_prices = DeviceBuffer::from_slice(prices_f32)?;
        let d_longs = DeviceBuffer::from_slice(&longs_i32)?;
        let d_shorts = DeviceBuffer::from_slice(&shorts_i32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems)? };

        let prefer_fast = combos.len() >= 32 && len >= 4_096 && free_ok_fast;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };

        if prefer_fast {
            self.ensure_scratch(len, combos.len())?;
            if self.scratch.is_some() {
                let mut s = self.scratch.take().unwrap();

                self.launch_prepare_momentum(&d_prices, len, first_valid, &mut s.mom, &mut s.amom)?;

                self.launch_param_parallel_tm(
                    &s.mom,
                    &s.amom,
                    &d_longs,
                    &d_shorts,
                    len,
                    first_valid,
                    combos.len(),
                    &mut s.out_tm,
                    block_x,
                )?;

                self.launch_transpose_tm_to_rm(&s.out_tm, len, combos.len(), &mut d_out)?;

                self.scratch = Some(s);
            }
            self.last_batch = Some(BatchKernelSelected::Plain { block_x });
            self.maybe_log_batch_debug();
            self.synchronize()?;
        } else {
            self.launch_batch_kernel(
                &d_prices,
                &d_longs,
                &d_shorts,
                len,
                first_valid,
                combos.len(),
                &mut d_out,
            )?;
            self.synchronize()?;
        }
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    fn launch_batch_kernel(
        &mut self,
        d_prices: &DeviceBuffer<f32>,
        d_longs: &DeviceBuffer<i32>,
        d_shorts: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsiError> {
        if len == 0 || n_combos == 0 {
            return Ok(());
        }
        if len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || first_valid > i32::MAX as usize
        {
            return Err(CudaTsiError::InvalidInput(
                "inputs exceed kernel limits".into(),
            ));
        }
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let gx = n_combos as u32;
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch_dims((gx, 1, 1), (block_x, 1, 1))?;
        self.last_batch = Some(BatchKernelSelected::Plain { block_x });
        self.maybe_log_batch_debug();

        let func = self.module.get_function("tsi_batch_f32").map_err(|_| {
            CudaTsiError::MissingKernelSymbol {
                name: "tsi_batch_f32",
            }
        })?;

        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut l_ptr = d_longs.as_device_ptr().as_raw();
            let mut s_ptr = d_shorts.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn maybe_log_batch_debug(&mut self) {
        if !self.debug_batch_logged && env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[TSI CUDA] batch kernel selected: {:?}", sel);
                self.debug_batch_logged = true;
            }
        }
    }

    fn maybe_log_many_debug(&mut self) {
        if !self.debug_many_logged && env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[TSI CUDA] many-series kernel selected: {:?}", sel);
                self.debug_many_logged = true;
            }
        }
    }

    pub fn tsi_many_series_one_param_time_major_dev(
        &mut self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        long_period: usize,
        short_period: usize,
    ) -> Result<DeviceArrayF32, CudaTsiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTsiError::InvalidInput("cols/rows zero".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaTsiError::InvalidInput("matrix size mismatch".into()));
        }
        if long_period == 0 || short_period == 0 {
            return Err(CudaTsiError::InvalidInput("periods must be > 0".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let v = prices_tm_f32[t * cols + s];
                if v.is_finite() {
                    fv = t as i32;
                    break;
                }
            }
            if (rows as i32) - fv < (1 + long_period + short_period) as i32 {
                return Err(CudaTsiError::InvalidInput(format!(
                    "series {} insufficient data for long+short={}, have {}",
                    s,
                    long_period + short_period + 1,
                    (rows as i32) - fv
                )));
            }
            first_valids[s] = fv;
        }

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let in_bytes = elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let fv_bytes = cols
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let head = 64usize * 1024 * 1024;
        let required = in_bytes
            .checked_add(fv_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        if !CudaTsi::will_fit(required, head) {
            if let Some((free, _)) = CudaTsi::device_mem_info() {
                return Err(CudaTsiError::OutOfMemory {
                    required,
                    free,
                    headroom: head,
                });
            } else {
                return Err(CudaTsiError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };

        self.launch_many_series_kernel(
            &d_prices_tm,
            cols,
            rows,
            long_period,
            short_period,
            &d_first,
            &mut d_out_tm,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    fn launch_many_series_kernel(
        &mut self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        long_period: usize,
        short_period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsiError> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }
        if cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaTsiError::InvalidInput(
                "inputs exceed kernel limits".into(),
            ));
        }
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 128,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch_dims((grid_x.max(1), 1, 1), (block_x, 1, 1))?;
        self.last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        self.maybe_log_many_debug();

        let func = self
            .module
            .get_function("tsi_many_series_one_param_f32")
            .map_err(|_| CudaTsiError::MissingKernelSymbol {
                name: "tsi_many_series_one_param_f32",
            })?;

        unsafe {
            let mut p_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut l_i = long_period as i32;
            let mut s_i = short_period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut l_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }
}

struct TsiScratch {
    mom: DeviceBuffer<f32>,
    amom: DeviceBuffer<f32>,
    out_tm: DeviceBuffer<f32>,
    len_cap: usize,
    combos_cap: usize,
}

impl CudaTsi {
    fn ensure_scratch(&mut self, len: usize, combos: usize) -> Result<(), CudaTsiError> {
        let need_new = match &self.scratch {
            None => true,
            Some(s) => s.len_cap < len || s.combos_cap < combos,
        };
        if !need_new {
            return Ok(());
        }
        let elems_tm = len
            .checked_mul(combos)
            .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
        let mom = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let amom = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let out_tm = unsafe { DeviceBuffer::<f32>::uninitialized(elems_tm) }?;
        self.scratch = Some(TsiScratch {
            mom,
            amom,
            out_tm,
            len_cap: len,
            combos_cap: combos,
        });
        Ok(())
    }

    fn launch_prepare_momentum(
        &mut self,
        d_prices: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_mom: &mut DeviceBuffer<f32>,
        d_amom: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsiError> {
        let func = self
            .module
            .get_function("tsi_prepare_momentum_f32")
            .map_err(|_| CudaTsiError::MissingKernelSymbol {
                name: "tsi_prepare_momentum_f32",
            })?;
        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut mom_ptr = d_mom.as_device_ptr().as_raw();
            let mut amom_ptr = d_amom.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut mom_ptr as *mut _ as *mut c_void,
                &mut amom_ptr as *mut _ as *mut c_void,
            ];
            let block_x: u32 = 256;
            let grid_x = ((len as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1u32, 1u32).into();
            let block: BlockSize = (block_x, 1u32, 1u32).into();
            self.validate_launch_dims((grid_x.max(1), 1, 1), (block_x, 1, 1))?;
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_param_parallel_tm(
        &mut self,
        d_mom: &DeviceBuffer<f32>,
        d_amom: &DeviceBuffer<f32>,
        d_longs: &DeviceBuffer<i32>,
        d_shorts: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
        block_x: u32,
    ) -> Result<(), CudaTsiError> {
        let func = self
            .module
            .get_function("tsi_one_series_many_params_tm_f32")
            .map_err(|_| CudaTsiError::MissingKernelSymbol {
                name: "tsi_one_series_many_params_tm_f32",
            })?;
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        unsafe {
            let mut mom_ptr = d_mom.as_device_ptr().as_raw();
            let mut amom_ptr = d_amom.as_device_ptr().as_raw();
            let mut l_ptr = d_longs.as_device_ptr().as_raw();
            let mut s_ptr = d_shorts.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut mom_ptr as *mut _ as *mut c_void,
                &mut amom_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x.max(1), 1u32, 1u32).into();
            let block: BlockSize = (block_x, 1u32, 1u32).into();
            self.validate_launch_dims((grid_x.max(1), 1, 1), (block_x, 1, 1))?;
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_transpose_tm_to_rm(
        &mut self,
        d_in_tm: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        d_out_rm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsiError> {
        let func = self
            .module
            .get_function("transpose_tm_to_rm_f32")
            .map_err(|_| CudaTsiError::MissingKernelSymbol {
                name: "transpose_tm_to_rm_f32",
            })?;
        let grid_x = ((cols as u32) + 31) / 32;
        let grid_y = ((rows as u32) + 31) / 32;
        let block: BlockSize = (32u32, 8u32, 1u32).into();
        unsafe {
            let mut in_ptr = d_in_tm.as_device_ptr().as_raw();
            let mut r_i = rows as i32;
            let mut c_i = cols as i32;
            let mut out_ptr = d_out_rm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut c_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1u32).into();
            self.validate_launch_dims((grid_x.max(1), grid_y.max(1), 1), (32, 8, 1))?;
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }
}

fn expand_grid(r: &TsiBatchRange) -> Result<Vec<TsiParams>, CudaTsiError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaTsiError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let vals: Vec<usize> = (start..=end).step_by(step).collect();
            if vals.is_empty() {
                return Err(CudaTsiError::InvalidInput(format!(
                    "invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            return Ok(vals);
        }
        let mut v = start;
        let mut out = Vec::new();
        loop {
            out.push(v);
            let guard = end.saturating_add(step);
            if v <= guard {
                break;
            }
            v = v.saturating_sub(step);
        }
        if out.is_empty() {
            return Err(CudaTsiError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        if *out.last().unwrap() != end {
            out.push(end);
        }
        Ok(out)
    }
    let longs = axis_usize(r.long_period)?;
    let shorts = axis_usize(r.short_period)?;
    if longs.is_empty() || shorts.is_empty() {
        return Err(CudaTsiError::InvalidInput(
            "no parameter combinations".into(),
        ));
    }
    let total = longs
        .len()
        .checked_mul(shorts.len())
        .ok_or_else(|| CudaTsiError::InvalidInput("size overflow".into()))?;
    let mut out = Vec::with_capacity(total);
    for &l in &longs {
        for &s in &shorts {
            out.push(TsiParams {
                long_period: Some(l),
                short_period: Some(s),
            });
        }
    }
    Ok(out)
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const LEN: usize = 1_000_000;
    const ROWS: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = LEN * std::mem::size_of::<f32>();
        let params_bytes = ROWS * 2 * std::mem::size_of::<i32>();
        let scratch_bytes = 2 * LEN * std::mem::size_of::<f32>();
        let tm_bytes = ROWS * LEN * std::mem::size_of::<f32>();
        let out_bytes = ROWS * LEN * std::mem::size_of::<f32>();
        in_bytes + params_bytes + scratch_bytes + tm_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct TsiBatchDeviceState {
        cuda: CudaTsi,
        d_prices: DeviceBuffer<f32>,
        d_longs: DeviceBuffer<i32>,
        d_shorts: DeviceBuffer<i32>,
        d_mom: DeviceBuffer<f32>,
        d_amom: DeviceBuffer<f32>,
        d_out_tm: DeviceBuffer<f32>,
        d_out_rm: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        block_x: u32,
    }
    impl CudaBenchState for TsiBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_prepare_momentum(
                    &self.d_prices,
                    self.len,
                    self.first_valid,
                    &mut self.d_mom,
                    &mut self.d_amom,
                )
                .expect("tsi launch_prepare_momentum");
            self.cuda
                .launch_param_parallel_tm(
                    &self.d_mom,
                    &self.d_amom,
                    &self.d_longs,
                    &self.d_shorts,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out_tm,
                    self.block_x,
                )
                .expect("tsi launch_param_parallel_tm");
            self.cuda
                .launch_transpose_tm_to_rm(
                    &self.d_out_tm,
                    self.len,
                    self.n_combos,
                    &mut self.d_out_rm,
                )
                .expect("tsi launch_transpose_tm_to_rm");
            self.cuda.synchronize().expect("tsi sync");
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaTsi::new(0).expect("cuda tsi");
        cuda.set_policy(CudaTsiPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });

        let price = gen_series(LEN);
        let first_valid = price.iter().position(|v| v.is_finite()).unwrap_or(0);

        let sweep = TsiBatchRange {
            long_period: (25, 25 + ROWS - 1, 1),
            short_period: (13, 13, 0),
        };
        let combos = expand_grid(&sweep).expect("expand_grid");
        let n_combos = combos.len();

        let mut longs_i32 = Vec::<i32>::with_capacity(n_combos);
        let mut shorts_i32 = Vec::<i32>::with_capacity(n_combos);
        for p in &combos {
            longs_i32.push(p.long_period.unwrap_or(25) as i32);
            shorts_i32.push(p.short_period.unwrap_or(13) as i32);
        }

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_longs = DeviceBuffer::from_slice(&longs_i32).expect("d_longs");
        let d_shorts = DeviceBuffer::from_slice(&shorts_i32).expect("d_shorts");
        let d_mom: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(LEN) }.expect("d_mom");
        let d_amom: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(LEN) }.expect("d_amom");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(LEN * n_combos) }.expect("d_out_tm");
        let d_out_rm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(LEN * n_combos) }.expect("d_out_rm");
        cuda.synchronize().expect("sync after prep");

        Box::new(TsiBatchDeviceState {
            cuda,
            d_prices,
            d_longs,
            d_shorts,
            d_mom,
            d_amom,
            d_out_tm,
            d_out_rm,
            len: LEN,
            first_valid,
            n_combos,
            block_x: 256,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "tsi",
            "batch_dev",
            "tsi_cuda_batch_dev",
            "1m_x_128",
            prep_batch,
        )
        .with_inner_iters(4)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
