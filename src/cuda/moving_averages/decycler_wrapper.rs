//! CUDA wrapper for Ehlers Decycler (input minus two‑pole high‑pass).
//!
//! Parity goals with ALMA wrapper:
//! - Stream NON_BLOCKING, PTX loaded from OUT_DIR
//! - Policy enums for kernel selection (1D variants)
//! - VRAM estimation and basic checks, chunking kept simple (grid 1D)
//! - Bench registration via define_ma_period_benches!

#![cfg(feature = "cuda")]


use crate::indicators::decycler::{DecyclerBatchRange, DecyclerParams};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Error)]
pub enum CudaDecyclerError {
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
pub struct CudaDecyclerPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaDecyclerPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    WarpScan { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaDecycler {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaDecyclerPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDecycler {
    pub fn new(device_id: usize) -> Result<Self, CudaDecyclerError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/decycler_kernel.ptx"));
        
        let module = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        
        if let Ok(mut f) = module.get_function("decycler_batch_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }
        if let Ok(mut f) = module.get_function("decycler_many_series_one_param_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaDecyclerPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaDecyclerPolicy,
    ) -> Result<Self, CudaDecyclerError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaDecyclerPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaDecyclerPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaDecyclerError> { self.stream.synchronize().map_err(Into::into) }
    pub fn context_arc(&self) -> Arc<Context> { self._context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn calc_launch_1d(
        &self,
        func: &Function,
        n_items: usize,
        override_block: Option<u32>,
    ) -> (BlockSize, GridSize) {
        if let Some(bx) = override_block {
            let bx = bx.max(32);
            let gx = ((n_items as u32 + bx - 1) / bx).max(1);
            return ((bx, 1, 1).into(), (gx, 1, 1).into());
        }
        let (min_grid, block_x) = func
            .suggested_launch_configuration(0usize, BlockSize::xyz(0, 0, 0))
            .unwrap_or((1, 256));
        let block_x = block_x.max(128);
        let mut grid_x = ((n_items as u32 + block_x - 1) / block_x).max(min_grid.max(1));
        if let Ok(dev) = Device::get_device(self.device_id) {
            if let Ok(max_gx) = dev.get_attribute(DeviceAttribute::MaxGridDimX) {
                grid_x = grid_x.min(max_gx as u32);
            }
        }
        ((block_x, 1, 1).into(), (grid_x, 1, 1).into())
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
    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DECYCLER batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDecycler)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DECYCLER many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDecycler)).debug_many_logged = true;
                }
            }
        }
    }

    

    pub fn decycler_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &DecyclerBatchRange,
    ) -> Result<DeviceArrayF32Decycler, CudaDecyclerError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n = prepared.combos.len();

        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let prices_bytes = prepared
            .series_len
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaDecyclerError::InvalidInput("series_len bytes overflow".into()))?;
        let per_combo_bytes = elem_i32
            .checked_add(
                3usize
                    .checked_mul(elem_f32)
                    .ok_or_else(|| {
                        CudaDecyclerError::InvalidInput("param bytes overflow".into())
                    })?,
            )
            .ok_or_else(|| CudaDecyclerError::InvalidInput("param bytes overflow".into()))?;
        let params_bytes = n
            .checked_mul(per_combo_bytes)
            .ok_or_else(|| CudaDecyclerError::InvalidInput("params bytes overflow".into()))?;
        let diff_bytes = prepared
            .series_len
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaDecyclerError::InvalidInput("diff bytes overflow".into()))?;
        let out_elems = prepared
            .series_len
            .checked_mul(n)
            .ok_or_else(|| {
                CudaDecyclerError::InvalidInput("output elements overflow".into())
            })?;
        let out_bytes = out_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaDecyclerError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(diff_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaDecyclerError::InvalidInput("total bytes overflow".into()))?;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            let (free, _) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaDecyclerError::OutOfMemory { required, free, headroom: 64 * 1024 * 1024 });
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;
        let d_periods =
            unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream) }?;
        let d_c = unsafe { DeviceBuffer::from_slice_async(&prepared.c_vals, &self.stream) }?;
        let d_two = unsafe { DeviceBuffer::from_slice_async(&prepared.two_1m_vals, &self.stream) }?;
        let d_neg = unsafe { DeviceBuffer::from_slice_async(&prepared.neg_oma_sq_vals, &self.stream) }?;
        let d_diff = unsafe { DeviceBuffer::from_slice_async(&prepared.diff, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream)? };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_c,
            &d_two,
            &d_neg,
            &d_diff,
            prepared.series_len,
            n,
            prepared.first_valid,
            &mut d_out,
        )?;

        
        self.synchronize()?;
        Ok(DeviceArrayF32Decycler { buf: d_out, rows: n, cols: prepared.series_len, ctx: self._context.clone(), device_id: self.device_id })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_c: &DeviceBuffer<f32>,
        d_two_1m: &DeviceBuffer<f32>,
        d_neg_oma_sq: &DeviceBuffer<f32>,
        d_diff: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDecyclerError> {
        if matches!(self.policy.batch, BatchKernelPolicy::Auto) {
            if let Ok(mut func) = self.module.get_function("decycler_batch_warp_scan_f32") {
                let _ = func.set_cache_config(CacheConfig::PreferL1);

                const MAX_GRID_X: usize = 65_535;
                let block: BlockSize = (32u32, 1, 1).into();

                unsafe {
                    (*(self as *const _ as *mut CudaDecycler)).last_batch =
                        Some(BatchKernelSelected::WarpScan { block_x: 32 });
                }

                let mut launched = 0usize;
                while launched < n_combos {
                    let rows = (n_combos - launched).min(MAX_GRID_X);
                    let grid: GridSize = (rows as u32, 1, 1).into();

                    unsafe {
                        let mut p_ptr = d_prices.as_device_ptr().as_raw();
                        let mut per_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                        let mut c_ptr = d_c.as_device_ptr().add(launched).as_raw();
                        let mut two_ptr = d_two_1m.as_device_ptr().add(launched).as_raw();
                        let mut neg_ptr = d_neg_oma_sq.as_device_ptr().add(launched).as_raw();
                        let mut diff_ptr = d_diff.as_device_ptr().as_raw();
                        let mut len_i = series_len as i32;
                        let mut n_i = rows as i32;
                        let mut first_i = first_valid as i32;
                        let mut out_ptr = d_out
                            .as_device_ptr()
                            .add(launched * series_len)
                            .as_raw();
                        let mut args: [*mut c_void; 10] = [
                            &mut p_ptr as *mut _ as *mut c_void,
                            &mut per_ptr as *mut _ as *mut c_void,
                            &mut c_ptr as *mut _ as *mut c_void,
                            &mut two_ptr as *mut _ as *mut c_void,
                            &mut neg_ptr as *mut _ as *mut c_void,
                            &mut diff_ptr as *mut _ as *mut c_void,
                            &mut len_i as *mut _ as *mut c_void,
                            &mut n_i as *mut _ as *mut c_void,
                            &mut first_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, &mut args)?;
                    }

                    launched += rows;
                }

                self.maybe_log_batch_debug();
                return Ok(());
            }
        }

        let func = self
            .module
            .get_function("decycler_batch_f32")
            .map_err(|_| CudaDecyclerError::MissingKernelSymbol { name: "decycler_batch_f32" })?;
        let (block, grid) = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => (
                (block_x, 1, 1).into(),
                (((n_combos as u32 + block_x - 1) / block_x).max(1), 1, 1).into(),
            ),
            BatchKernelPolicy::Auto => self.calc_launch_1d(&func, n_combos, None),
        };
        
        if let Ok(dev) = Device::get_device(self.device_id) {
            let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX).unwrap_or(2_147_483_647) as u32;
            let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX).unwrap_or(1024) as u32;
            if grid.x > max_gx || block.x > max_bx {
                return Err(CudaDecyclerError::LaunchConfigTooLarge { gx: grid.x, gy: grid.y, gz: grid.z, bx: block.x, by: block.y, bz: block.z });
            }
        }
        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut per_ptr = d_periods.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut two_ptr = d_two_1m.as_device_ptr().as_raw();
            let mut neg_ptr = d_neg_oma_sq.as_device_ptr().as_raw();
            let mut diff_ptr = d_diff.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut n_i = n_combos as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut per_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut two_ptr as *mut _ as *mut c_void,
                &mut neg_ptr as *mut _ as *mut c_void,
                &mut diff_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)?;
        }
        unsafe {
            let bx = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } => block_x,
                BatchKernelPolicy::Auto => block.x,
            };
            (*(self as *const _ as *mut CudaDecycler)).last_batch =
                Some(BatchKernelSelected::Plain { block_x: bx });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    

    pub fn decycler_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DecyclerParams,
    ) -> Result<DeviceArrayF32Decycler, CudaDecyclerError> {
        let prepared = Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDecyclerError::InvalidInput("cols*rows overflow".into()))?;
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let prices_bytes = elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaDecyclerError::InvalidInput("prices bytes overflow".into()))?;
        let first_bytes = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaDecyclerError::InvalidInput("first_valid bytes overflow".into()))?;
        let out_bytes = elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaDecyclerError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaDecyclerError::InvalidInput("total bytes overflow".into()))?;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            let (free, _) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaDecyclerError::OutOfMemory { required, free, headroom: 64 * 1024 * 1024 });
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream)? };
        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.period,
            prepared.c,
            prepared.two_1m,
            prepared.neg_oma_sq,
            cols,
            rows,
            &mut d_out,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32Decycler { buf: d_out, rows, cols, ctx: self._context.clone(), device_id: self.device_id })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        c: f32,
        two_1m: f32,
        neg_oma_sq: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDecyclerError> {
        let func = self
            .module
            .get_function("decycler_many_series_one_param_f32")
            .map_err(|_| CudaDecyclerError::MissingKernelSymbol { name: "decycler_many_series_one_param_f32" })?;
        let (block, grid) = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => (
                (block_x, 1, 1).into(),
                (((num_series as u32 + block_x - 1) / block_x).max(1), 1, 1).into(),
            ),
            ManySeriesKernelPolicy::Auto => self.calc_launch_1d(&func, num_series, None),
        };
        if let Ok(dev) = Device::get_device(self.device_id) {
            let max_gx = dev
                .get_attribute(DeviceAttribute::MaxGridDimX)
                .unwrap_or(2_147_483_647) as u32;
            let max_bx = dev
                .get_attribute(DeviceAttribute::MaxBlockDimX)
                .unwrap_or(1024) as u32;
            if grid.x > max_gx || block.x > max_bx {
                return Err(CudaDecyclerError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx: block.x,
                    by: block.y,
                    bz: block.z,
                });
            }
        }
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut c_val = c;
            let mut two_val = two_1m;
            let mut neg_val = neg_oma_sq;
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut c_val as *mut _ as *mut c_void,
                &mut two_val as *mut _ as *mut c_void,
                &mut neg_val as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                std::ptr::null_mut(), 
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        unsafe {
            let bx = match self.policy.many_series {
                ManySeriesKernelPolicy::OneD { block_x } => block_x,
                ManySeriesKernelPolicy::Auto => block.x,
            };
            (*(self as *const _ as *mut CudaDecycler)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: bx });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DecyclerBatchRange,
    ) -> Result<PreparedDecyclerBatch, CudaDecyclerError> {
        let series_len = data_f32.len();
        if series_len == 0 {
            return Err(CudaDecyclerError::InvalidInput("empty series".into()));
        }
        let combos = expand_grid(sweep)?;

        
        let mut first_valid: Option<usize> = None;
        for i in 0..series_len {
            if data_f32[i].is_finite() {
                first_valid = Some(i);
                break;
            }
        }
        let fv = first_valid
            .ok_or_else(|| CudaDecyclerError::InvalidInput("all values are NaN".into()))?;
        let max_p = combos.iter().map(|c| c.hp_period.unwrap()).max().unwrap();
        if series_len - fv < max_p {
            return Err(CudaDecyclerError::InvalidInput(format!(
                "not enough valid data: needed >= {}, valid = {}",
                max_p,
                series_len - fv
            )));
        }

        
        let mut diff = vec![0f32; series_len];
        for i in (fv + 2)..series_len {
            let x = data_f32[i];
            let x1 = data_f32[i - 1];
            let x2 = data_f32[i - 2];
            diff[i] = x - 2.0 * x1 + x2;
        }

        
        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut c_vals = Vec::with_capacity(combos.len());
        let mut two_1m_vals = Vec::with_capacity(combos.len());
        let mut neg_oma_sq_vals = Vec::with_capacity(combos.len());
        for p in &combos {
            let period = p.hp_period.unwrap();
            let k = p.k.unwrap();
            let coeffs = compute_coefficients(period, k);
            periods_i32.push(period as i32);
            c_vals.push(coeffs.c);
            two_1m_vals.push(coeffs.two_1m);
            neg_oma_sq_vals.push(coeffs.neg_oma_sq);
        }

        Ok(PreparedDecyclerBatch {
            combos,
            first_valid: fv,
            series_len,
            periods_i32,
            c_vals,
            two_1m_vals,
            neg_oma_sq_vals,
            diff,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DecyclerParams,
    ) -> Result<PreparedDecyclerMany, CudaDecyclerError> {
        if cols == 0 || rows == 0 {
            return Err(CudaDecyclerError::InvalidInput("empty matrix".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaDecyclerError::InvalidInput(
                "data shape mismatch".into(),
            ));
        }
        let period = params.hp_period.unwrap_or(125);
        let k = params.k.unwrap_or(0.707);
        if period < 2 {
            return Err(CudaDecyclerError::InvalidInput(
                "hp_period must be >= 2".into(),
            ));
        }
        if !(k.is_finite()) || k <= 0.0 {
            return Err(CudaDecyclerError::InvalidInput(
                "k must be positive and finite".into(),
            ));
        }

        
        let needed = period;
        let mut first_valids = Vec::with_capacity(cols);
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if v.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fvu =
                fv.ok_or_else(|| CudaDecyclerError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fvu < needed {
                return Err(CudaDecyclerError::InvalidInput(format!(
                    "series {} not enough valid data: needed >= {}, valid = {}",
                    s,
                    needed,
                    rows - fvu
                )));
            }
            first_valids.push(fvu as i32);
        }

        let coeffs = compute_coefficients(period, k);
        Ok(PreparedDecyclerMany {
            first_valids,
            period: period as i32,
            c: coeffs.c,
            two_1m: coeffs.two_1m,
            neg_oma_sq: coeffs.neg_oma_sq,
        })
    }
}


struct PreparedDecyclerBatch {
    combos: Vec<DecyclerParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    c_vals: Vec<f32>,
    two_1m_vals: Vec<f32>,
    neg_oma_sq_vals: Vec<f32>,
    diff: Vec<f32>,
}
struct PreparedDecyclerMany {
    first_valids: Vec<i32>,
    period: i32,
    c: f32,
    two_1m: f32,
    neg_oma_sq: f32,
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::decycler::DecyclerParams;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct DecyclerBatchDevState {
        cuda: CudaDecycler,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_c: DeviceBuffer<f32>,
        d_two_1m: DeviceBuffer<f32>,
        d_neg_oma_sq: DeviceBuffer<f32>,
        d_diff: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for DecyclerBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_c,
                    &self.d_two_1m,
                    &self.d_neg_oma_sq,
                    &self.d_diff,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("decycler batch kernel");
            self.cuda.synchronize().expect("decycler sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaDecycler::new(0).expect("cuda decycler");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = DecyclerBatchRange {
            hp_period: (10, 10 + PARAM_SWEEP - 1, 1),
            k: (0.5, 0.5, 0.0),
        };
        let prep =
            CudaDecycler::prepare_batch_inputs(&price, &sweep).expect("decycler prepare batch");
        let n_combos = prep.combos.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&prep.periods_i32).expect("d_periods");
        let d_c = DeviceBuffer::from_slice(&prep.c_vals).expect("d_c");
        let d_two_1m = DeviceBuffer::from_slice(&prep.two_1m_vals).expect("d_two_1m");
        let d_neg_oma_sq = DeviceBuffer::from_slice(&prep.neg_oma_sq_vals).expect("d_neg_oma_sq");
        let d_diff = DeviceBuffer::from_slice(&prep.diff).expect("d_diff");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prep.series_len * n_combos) }.expect("d_out");
        cuda.synchronize().expect("sync after prep");

        Box::new(DecyclerBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_c,
            d_two_1m,
            d_neg_oma_sq,
            d_diff,
            series_len: prep.series_len,
            n_combos,
            first_valid: prep.first_valid,
            d_out,
        })
    }

    struct DecyclerManyDevState {
        cuda: CudaDecycler,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: i32,
        c: f32,
        two_1m: f32,
        neg_oma_sq: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for DecyclerManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.period,
                    self.c,
                    self.two_1m,
                    self.neg_oma_sq,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("decycler many-series kernel");
            self.cuda.synchronize().expect("decycler sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaDecycler::new(0).expect("cuda decycler");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = DecyclerParams {
            hp_period: Some(64),
            k: Some(0.5),
        };
        let prep = CudaDecycler::prepare_many_series_inputs(&data_tm, cols, rows, &params)
            .expect("decycler prepare many-series");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&prep.first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.synchronize().expect("sync after prep");

        Box::new(DecyclerManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period: prep.period,
            c: prep.c,
            two_1m: prep.two_1m,
            neg_oma_sq: prep.neg_oma_sq,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "decycler",
                "one_series_many_params",
                "decycler_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "decycler",
                "many_series_one_param",
                "decycler_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}


struct Coefficients {
    c: f32,
    two_1m: f32,
    neg_oma_sq: f32,
}
fn compute_coefficients(period: usize, k: f64) -> Coefficients {
    use std::f64::consts::PI;
    let theta = 2.0 * PI * k / period as f64;
    let sin_v = theta.sin();
    let cos_v = theta.cos();
    let alpha = 1.0 + ((sin_v - 1.0) / cos_v);
    let c = (1.0 - 0.5 * alpha).powi(2);
    let oma = 1.0 - alpha;
    let two_1m = 2.0 * oma;
    let neg_oma_sq = -(oma * oma);
    Coefficients {
        c: c as f32,
        two_1m: two_1m as f32,
        neg_oma_sq: neg_oma_sq as f32,
    }
}

fn expand_grid(range: &DecyclerBatchRange) -> Result<Vec<DecyclerParams>, CudaDecyclerError> {
    fn axis_usize(a: (usize, usize, usize)) -> Result<Vec<usize>, CudaDecyclerError> {
        let (s, e, st) = a;
        if st == 0 || s == e { return Ok(vec![s]); }
        if s < e { return Ok((s..=e).step_by(st).collect()); }
        let mut v = Vec::new();
        let mut cur: isize = s as isize;
        let end_i: isize = e as isize;
        let step_i: isize = st as isize;
        while cur >= end_i {
            v.push(cur as usize);
            if step_i == 0 { break; }
            cur = cur.saturating_sub(step_i);
            if cur == std::isize::MIN { break; }
        }
        if v.is_empty() {
            return Err(CudaDecyclerError::InvalidInput("invalid usize range".into()));
        }
        Ok(v)
    }
    fn axis_f64(a: (f64, f64, f64)) -> Result<Vec<f64>, CudaDecyclerError> {
        let (s, e, st) = a;
        if st.abs() < 1e-12 || (s - e).abs() < 1e-12 { return Ok(vec![s]); }
        let mut v = Vec::new();
        if s < e && st > 0.0 {
            let mut cur = s;
            while cur <= e + 1e-12 { v.push(cur); cur += st; }
        } else if s > e && st < 0.0 {
            let mut cur = s;
            while cur >= e - 1e-12 { v.push(cur); cur += st; }
        }
        if v.is_empty() { return Err(CudaDecyclerError::InvalidInput("invalid f64 range".into())); }
        Ok(v)
    }
    let ps = axis_usize(range.hp_period)?;
    let ks = axis_f64(range.k)?;
    let mut out = Vec::with_capacity(ps.len().saturating_mul(ks.len()));
    for &p in &ps { for &k in &ks { out.push(DecyclerParams { hp_period: Some(p), k: Some(k) }); } }
    Ok(out)
}

/// VRAM-backed array handle for decycler with context/device info for interop
pub struct DeviceArrayF32Decycler {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Decycler {
    #[inline] pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline] pub fn len(&self) -> usize { self.rows * self.cols }
}
