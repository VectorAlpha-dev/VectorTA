#![cfg(feature = "cuda")]

use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::pivot::{PivotBatchRange, PivotParams};

#[derive(Debug, Error)]
pub enum CudaPivotError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error(
        "Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes"
    )]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
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
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaPivot {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    batch_policy: BatchKernelPolicy,
    many_policy: ManySeriesKernelPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaPivot {
    pub fn new(device_id: usize) -> Result<Self, CudaPivotError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pivot_kernel.ptx"));
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
            _context: context,
            device_id: device_id as u32,
            batch_policy: BatchKernelPolicy::Auto,
            many_policy: ManySeriesKernelPolicy::Auto,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaPivotError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaPivotError::OutOfMemory {
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
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] pivot batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaPivot)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] pivot many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaPivot)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn validate_launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaPivotError> {
        let dev = Device::get_device(self.device_id).map_err(CudaPivotError::Cuda)?;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .map_err(CudaPivotError::Cuda)? as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .map_err(CudaPivotError::Cuda)? as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .map_err(CudaPivotError::Cuda)? as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaPivotError::Cuda)? as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .map_err(CudaPivotError::Cuda)? as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
            .map_err(CudaPivotError::Cuda)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if bx > max_bx || by > max_by || bz > max_bz || gx > max_gx || gy > max_gy || gz > max_gz {
            return Err(CudaPivotError::LaunchConfigTooLarge {
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

    fn expand_grid(range: &PivotBatchRange) -> Result<Vec<PivotParams>, CudaPivotError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaPivotError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            let mut vals = Vec::new();
            if start < end {
                let mut cur = start;
                while cur <= end {
                    vals.push(cur);
                    cur = cur.checked_add(step).ok_or_else(|| {
                        CudaPivotError::InvalidInput("mode sweep overflow".into())
                    })?;
                }
            } else {
                let mut cur = start;
                while cur >= end {
                    vals.push(cur);
                    cur = cur.checked_sub(step).ok_or_else(|| {
                        CudaPivotError::InvalidInput("mode sweep overflow".into())
                    })?;
                    if cur == 0 && end > 0 {
                        break;
                    }
                    if let Some(&last) = vals.last() {
                        if last == cur {
                            break;
                        }
                    }
                }
                if let Some(&last) = vals.last() {
                    if last < end {
                        vals.pop();
                    }
                }
            }
            if vals.is_empty() {
                return Err(CudaPivotError::InvalidInput(
                    "invalid mode sweep: produced no values".into(),
                ));
            }
            Ok(vals)
        }

        let modes = axis_usize(range.mode)?;
        let mut out = Vec::with_capacity(modes.len());
        for m in modes {
            out.push(PivotParams { mode: Some(m) });
        }
        Ok(out)
    }

    #[inline]
    fn first_valid_ohlc_f32(high: &[f32], low: &[f32], close: &[f32]) -> Option<usize> {
        let len = high.len().min(low.len()).min(close.len());
        for i in 0..len {
            if !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()) {
                return Some(i);
            }
        }
        None
    }

    pub fn pivot_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        open: &[f32],
        sweep: &PivotBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<PivotParams>), CudaPivotError> {
        if high.is_empty() || low.is_empty() || close.is_empty() || open.is_empty() {
            return Err(CudaPivotError::InvalidInput("empty input".into()));
        }
        let n = high.len();
        if low.len() != n || close.len() != n || open.len() != n {
            return Err(CudaPivotError::InvalidInput(
                "input arrays must have same length".into(),
            ));
        }
        let first_valid = Self::first_valid_ohlc_f32(high, low, close)
            .ok_or_else(|| CudaPivotError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaPivotError::InvalidInput("empty mode sweep".into()));
        }
        let n_combos = combos.len();

        let need_o_any = combos.iter().any(|p| matches!(p.mode.unwrap_or(3), 2 | 4));

        let inputs_arrays: usize = 3 + if need_o_any { 1 } else { 0 };
        let bytes_inputs = inputs_arrays
            .checked_mul(n)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let bytes_modes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(9)
            .and_then(|x| x.checked_mul(n))
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let bytes_out = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_modes)
            .and_then(|a| a.checked_add(bytes_out))
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_high = DeviceBuffer::from_slice(high).map_err(CudaPivotError::Cuda)?;
        let d_low = DeviceBuffer::from_slice(low).map_err(CudaPivotError::Cuda)?;
        let d_close = DeviceBuffer::from_slice(close).map_err(CudaPivotError::Cuda)?;

        let d_open_opt = if need_o_any {
            Some(DeviceBuffer::from_slice(open).map_err(CudaPivotError::Cuda)?)
        } else {
            None
        };
        let d_open_ref: &DeviceBuffer<f32> = d_open_opt.as_ref().unwrap_or(&d_close);
        let mut modes_i32 = Vec::with_capacity(n_combos);
        for p in &combos {
            modes_i32.push(p.mode.unwrap_or(3) as i32);
        }
        let d_modes = DeviceBuffer::from_slice(&modes_i32).map_err(CudaPivotError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.map_err(CudaPivotError::Cuda)?;

        self.launch_pivot_batch(
            &d_high,
            &d_low,
            &d_close,
            d_open_ref,
            n,
            first_valid,
            &d_modes,
            n_combos,
            &mut d_out,
        )?;
        self.stream.synchronize().map_err(CudaPivotError::Cuda)?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: 9 * n_combos,
                cols: n,
            },
            combos,
        ))
    }

    fn launch_pivot_batch(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_open: &DeviceBuffer<f32>,
        n: usize,
        first_valid: usize,
        d_modes: &DeviceBuffer<i32>,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPivotError> {
        let func = self.module.get_function("pivot_batch_f32").map_err(|_| {
            CudaPivotError::MissingKernelSymbol {
                name: "pivot_batch_f32",
            }
        })?;
        let block_x = match self.batch_policy {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        let grid_x = ((n as u32) + block_x - 1) / block_x;

        let gx = grid_x.max(1);
        let grid_dims = (gx, 1, 1);
        let block_dims = (block_x, 1, 1);
        self.validate_launch(grid_dims, block_dims)?;
        let grid: GridSize = grid_dims.into();
        let block: BlockSize = block_dims.into();
        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = d_close.as_device_ptr().as_raw();
            let mut o = d_open.as_device_ptr().as_raw();
            let mut m = d_modes.as_device_ptr().as_raw();
            let mut n_i = n as i32;
            let mut fv_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
                &mut m as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaPivotError::Cuda)?;
        }

        unsafe {
            (*(self as *const _ as *mut CudaPivot)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaPivot)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn pivot_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        open_tm: &[f32],
        cols: usize,
        rows: usize,
        mode: usize,
    ) -> Result<DeviceArrayF32, CudaPivotError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPivotError::InvalidInput("empty dims".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        if high_tm.len() != elems
            || low_tm.len() != elems
            || close_tm.len() != elems
            || open_tm.len() != elems
        {
            return Err(CudaPivotError::InvalidInput(
                "time-major inputs must all be cols*rows".into(),
            ));
        }

        let need_o = mode == 2 || mode == 4;
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let idx = t * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if need_o {
                    let o = open_tm[idx];
                    if !(h.is_nan() || l.is_nan() || c.is_nan() || o.is_nan()) {
                        fv = t as i32;
                        break;
                    }
                } else if !(h.is_nan() || l.is_nan() || c.is_nan()) {
                    fv = t as i32;
                    break;
                }
            }
            if fv == rows as i32 {
                return Err(CudaPivotError::InvalidInput(format!(
                    "series {}: all values are NaN",
                    s
                )));
            }
            first_valids[s] = fv;
        }

        let inputs_arrays: usize = 3 + if need_o { 1 } else { 0 };
        let inputs_elems = inputs_arrays
            .checked_mul(elems)
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let out_elems = 9usize
            .checked_mul(elems)
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let flops_bytes = inputs_elems
            .checked_add(out_elems)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let fv_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let required = flops_bytes
            .checked_add(fv_bytes)
            .ok_or_else(|| CudaPivotError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_high = DeviceBuffer::from_slice(high_tm).map_err(CudaPivotError::Cuda)?;
        let d_low = DeviceBuffer::from_slice(low_tm).map_err(CudaPivotError::Cuda)?;
        let d_close = DeviceBuffer::from_slice(close_tm).map_err(CudaPivotError::Cuda)?;

        let d_open_opt = if need_o {
            Some(DeviceBuffer::from_slice(open_tm).map_err(CudaPivotError::Cuda)?)
        } else {
            None
        };
        let d_open_ref: &DeviceBuffer<f32> = d_open_opt.as_ref().unwrap_or(&d_close);
        let d_fv = DeviceBuffer::from_slice(&first_valids).map_err(CudaPivotError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.map_err(CudaPivotError::Cuda)?;

        self.launch_pivot_many_series_tm(
            &d_high, &d_low, &d_close, d_open_ref, &d_fv, cols, rows, mode, &mut d_out,
        )?;
        self.stream.synchronize().map_err(CudaPivotError::Cuda)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 9 * rows,
            cols,
        })
    }

    fn launch_pivot_many_series_tm(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        d_open_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        mode: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPivotError> {
        let func = self
            .module
            .get_function("pivot_many_series_one_param_time_major_f32")
            .map_err(|_| CudaPivotError::MissingKernelSymbol {
                name: "pivot_many_series_one_param_time_major_f32",
            })?;
        let block_x = match self.many_policy {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let gx = grid_x.max(1);
        let grid_dims = (gx, 1, 1);
        let block_dims = (block_x, 1, 1);
        self.validate_launch(grid_dims, block_dims)?;
        let grid: GridSize = grid_dims.into();
        let block: BlockSize = block_dims.into();
        unsafe {
            let mut hp = d_high_tm.as_device_ptr().as_raw();
            let mut lp = d_low_tm.as_device_ptr().as_raw();
            let mut cp = d_close_tm.as_device_ptr().as_raw();
            let mut op = d_open_tm.as_device_ptr().as_raw();
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut mode_i = mode as i32;
            let mut outp = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp as *mut _ as *mut c_void,
                &mut lp as *mut _ as *mut c_void,
                &mut cp as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut mode_i as *mut _ as *mut c_void,
                &mut outp as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaPivotError::Cuda)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaPivot)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_LEN: usize = 1_000_000;
    const MANY_ROWS: usize = 200_000;
    const MANY_COLS: usize = 128;

    fn bytes_batch(n_combos: usize) -> usize {
        let in_bytes = 4 * ONE_LEN * std::mem::size_of::<f32>();
        let out_bytes = 9 * n_combos * ONE_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many() -> usize {
        let elems = MANY_ROWS * MANY_COLS;
        (4 * elems + 9 * elems) * std::mem::size_of::<f32>()
            + MANY_COLS * std::mem::size_of::<i32>()
            + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaPivot,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_open: DeviceBuffer<f32>,
        d_modes: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        n: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_pivot_batch(
                    &self.d_high,
                    &self.d_low,
                    &self.d_close,
                    &self.d_open,
                    self.n,
                    self.first_valid,
                    &self.d_modes,
                    self.n_combos,
                    &mut self.d_out,
                )
                .unwrap();
            self.cuda.stream.synchronize().unwrap();
        }
    }

    struct ManyState {
        cuda: CudaPivot,
        d_h_tm: DeviceBuffer<f32>,
        d_l_tm: DeviceBuffer<f32>,
        d_c_tm: DeviceBuffer<f32>,
        d_o_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        mode: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            self.cuda
                .launch_pivot_many_series_tm(
                    &self.d_h_tm,
                    &self.d_l_tm,
                    &self.d_c_tm,
                    &self.d_o_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.mode,
                    &mut self.d_out_tm,
                )
                .unwrap();
            self.cuda.stream.synchronize().unwrap();
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaPivot::new(0).expect("cuda pivot");
        let mut h = vec![f32::NAN; ONE_LEN];
        let mut l = vec![f32::NAN; ONE_LEN];
        let mut c = vec![f32::NAN; ONE_LEN];
        let mut o = vec![f32::NAN; ONE_LEN];
        for i in 5..ONE_LEN {
            let x = i as f32 * 0.0015;
            let base = (x * 0.9).sin() + 0.001 * x;
            let range = 0.2 + 0.03 * (x * 0.37).cos().abs();
            c[i] = base;
            o[i] = base + 0.01 * (x * 0.23).sin();
            l[i] = base - range;
            h[i] = base + range;
        }
        let first_valid = h.iter().position(|v| !v.is_nan()).unwrap_or(0);
        let modes: Vec<i32> = (0..=4).map(|m| m as i32).collect();
        let n_combos = modes.len();

        let d_high = DeviceBuffer::from_slice(&h).unwrap();
        let d_low = DeviceBuffer::from_slice(&l).unwrap();
        let d_close = DeviceBuffer::from_slice(&c).unwrap();
        let d_open = DeviceBuffer::from_slice(&o).unwrap();
        let d_modes = DeviceBuffer::from_slice(&modes).unwrap();
        let out_elems = 9usize * n_combos * ONE_LEN;
        let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems) }.unwrap();
        Box::new(BatchState {
            cuda,
            d_high,
            d_low,
            d_close,
            d_open,
            d_modes,
            d_out,
            n: ONE_LEN,
            first_valid,
            n_combos,
        })
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaPivot::new(0).expect("cuda pivot");

        let mut h_tm = vec![f32::NAN; MANY_ROWS * MANY_COLS];
        let mut l_tm = vec![f32::NAN; MANY_ROWS * MANY_COLS];
        let mut c_tm = vec![f32::NAN; MANY_ROWS * MANY_COLS];
        let mut o_tm = vec![f32::NAN; MANY_ROWS * MANY_COLS];
        for s in 0..MANY_COLS {
            for t in 0..MANY_ROWS {
                let idx = t * MANY_COLS + s;
                let x = (t as f32) * 0.001 + (s as f32) * 0.01;
                let base = (x * 0.77).sin() + 0.002 * x;
                let rng = 0.1 + 0.05 * (x * 0.21).cos().abs();
                c_tm[idx] = base;
                o_tm[idx] = base + 0.01 * (x * 0.33).sin();
                l_tm[idx] = base - rng;
                h_tm[idx] = base + rng;
            }
        }
        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let mode = 3usize;
        let first_valids = vec![0i32; cols];
        let d_h_tm = DeviceBuffer::from_slice(&h_tm).unwrap();
        let d_l_tm = DeviceBuffer::from_slice(&l_tm).unwrap();
        let d_c_tm = DeviceBuffer::from_slice(&c_tm).unwrap();
        let d_o_tm = DeviceBuffer::from_slice(&o_tm).unwrap();
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
        let out_elems = 9usize * cols * rows;
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.unwrap();
        cuda.stream.synchronize().unwrap();
        Box::new(ManyState {
            cuda,
            d_h_tm,
            d_l_tm,
            d_c_tm,
            d_o_tm,
            d_first_valids,
            cols,
            rows,
            mode,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let combos = 5usize;
        vec![
            CudaBenchScenario::new(
                "pivot",
                "batch",
                "pivot_cuda_batch",
                "1m × 5 modes",
                prep_batch,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_batch(combos)),
            CudaBenchScenario::new(
                "pivot",
                "many_series",
                "pivot_cuda_many_series_tm",
                "200k × 128",
                prep_many,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many()),
        ]
    }
}
