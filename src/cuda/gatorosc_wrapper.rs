//! CUDA wrapper for Gator Oscillator (GATOR): upper/lower histograms and their 1-bar changes.
//!
//! Pattern classification: Recurrence/IIR per-parameter (sequential over time).
//! We implement:
//! - gatorosc_batch_f32: one series × many params (one block per row; shared rings)
//! - gatorosc_many_series_one_param_f32: many series × one param (one thread per series)
//!
//! Parity items (mirrors ALMA/CWMA wrappers):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/gatorosc_kernel.ptx"))
//! - Stream NON_BLOCKING
//! - JIT options: DetermineTargetFromContext + OptLevel O2 with fallbacks
//! - VRAM check with ~64MB headroom and grid.y (or chunk count) <= 65_535
//! - Logging of selected policy when BENCH_DEBUG=1

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::gatorosc::{GatorOscBatchRange, GatorOscParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaGatorOscError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaGatorOscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaGatorOscError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaGatorOscError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaGatorOscError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaGatorOscPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct DeviceGatorOscQuad {
    pub upper: DeviceArrayF32,
    pub lower: DeviceArrayF32,
    pub upper_change: DeviceArrayF32,
    pub lower_change: DeviceArrayF32,
}

pub struct CudaGatorOsc {
    module: Module,
    stream: Stream,
    _ctx: Context,
    max_grid_x: usize,
    max_smem_per_block: usize,
    policy: CudaGatorOscPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaGatorOsc {
    pub fn new(device_id: usize) -> Result<Self, CudaGatorOscError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        // Query limits we’ll use for chunking & shared-mem capping
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(65_535) as usize;
        let max_smem_per_block = device
            .get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
            .unwrap_or(48 * 1024) as usize;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/gatorosc_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _ctx: ctx,
            policy: CudaGatorOscPolicy::default(),
            max_grid_x,
            max_smem_per_block,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaGatorOscPolicy) { self.policy = p; }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] GATOR batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaGatorOsc)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] GATOR many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaGatorOsc)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaGatorOsc)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------- Batch (one series × many params) ----------
    pub fn gatorosc_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &GatorOscBatchRange,
    ) -> Result<DeviceGatorOscQuad, CudaGatorOscError> {
        let len = data_f32.len();
        if len == 0 {
            return Err(CudaGatorOscError::InvalidInput("empty series".into()));
        }
        if len == 0 {
            return Err(CudaGatorOscError::InvalidInput("empty series".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaGatorOscError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaGatorOscError::InvalidInput("empty sweep".into()));
        }
        if combos.is_empty() {
            return Err(CudaGatorOscError::InvalidInput("empty sweep".into()));
        }

        // Flatten params and validate
        let mut jl: Vec<i32> = Vec::with_capacity(combos.len());
        let mut js: Vec<i32> = Vec::with_capacity(combos.len());
        let mut tl: Vec<i32> = Vec::with_capacity(combos.len());
        let mut ts_: Vec<i32> = Vec::with_capacity(combos.len());
        let mut ll: Vec<i32> = Vec::with_capacity(combos.len());
        let mut ls: Vec<i32> = Vec::with_capacity(combos.len());
        let mut needed_max: usize = 0;
        for p in &combos {
            let jlen = p.jaws_length.unwrap_or(13) as i32;
            let jsh = p.jaws_shift.unwrap_or(8) as i32;
            let tlen = p.teeth_length.unwrap_or(8) as i32;
            let tsh = p.teeth_shift.unwrap_or(5) as i32;
            let llen = p.lips_length.unwrap_or(5) as i32;
            let lsh = p.lips_shift.unwrap_or(3) as i32;
            if jlen <= 0 || tlen <= 0 || llen <= 0 {
                return Err(CudaGatorOscError::InvalidInput(
                    "non-positive length".into(),
                ));
            }
            let upper_needed =
                (jlen as usize).max(tlen as usize) + (jsh as usize).max(tsh as usize);
            let lower_needed =
                (tlen as usize).max(llen as usize) + (tsh as usize).max(lsh as usize);
            needed_max = needed_max.max(upper_needed.max(lower_needed));
            jl.push(jlen); js.push(jsh); tl.push(tlen); ts_.push(tsh); ll.push(llen); ls.push(lsh);
        }
        let valid_tail = len - first_valid;
        if valid_tail < needed_max {
            return Err(CudaGatorOscError::InvalidInput(format!(
                "not enough valid data: needed >= {}, valid = {}",
                needed_max, valid_tail
            )));
        }

        // VRAM estimation for outputs + prices (params sliced per chunk)
        let rows = combos.len();
        let bytes_prices = len * std::mem::size_of::<f32>();
        let bytes_out = 4 * rows * len * std::mem::size_of::<f32>();
        let headroom = 64usize * 1024 * 1024;
        let free_now = mem_get_info().ok().map(|(f, _)| f).unwrap_or(usize::MAX);
        if free_now.saturating_sub(headroom) < bytes_out + bytes_prices {
            return Err(CudaGatorOscError::Cuda(format!(
                "Insufficient VRAM for outputs+prices (need ~{:.1} MiB). Reduce sweep size or run CPU.",
                ((bytes_out + bytes_prices + headroom) as f64) / (1024.0 * 1024.0)
            )));
        }

        // Upload prices once
        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;

        // Final device outputs allocated once
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let mut d_uchn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let mut d_lchn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;

        // Launcher: write directly into final outputs at row offset = start
        let launch_chunk = |this: &CudaGatorOsc,
                            start: usize,
                            chunk: usize,
                            ring_len_i: i32|
         -> Result<(), CudaGatorOscError> {
            let d_jl = DeviceBuffer::from_slice(&jl[start..start + chunk])
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
            let d_js = DeviceBuffer::from_slice(&js[start..start + chunk])
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
            let d_tl = DeviceBuffer::from_slice(&tl[start..start + chunk])
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
            let d_ts = DeviceBuffer::from_slice(&ts_[start..start + chunk])
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
            let d_ll = DeviceBuffer::from_slice(&ll[start..start + chunk])
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
            let d_ls = DeviceBuffer::from_slice(&ls[start..start + chunk])
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;

            let func = this
                .module
                .get_function("gatorosc_batch_f32")
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
            let block_x = this.policy.batch_block_x.unwrap_or(256);
            let grid: GridSize = (chunk as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                (*(this as *const _ as *mut CudaGatorOsc)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
                (*(this as *const _ as *mut CudaGatorOsc)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
                let mut p_ptr = d_prices.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut jl_ptr = d_jl.as_device_ptr().as_raw();
                let mut js_ptr = d_js.as_device_ptr().as_raw();
                let mut tl_ptr = d_tl.as_device_ptr().as_raw();
                let mut ts_ptr = d_ts.as_device_ptr().as_raw();
                let mut ll_ptr = d_ll.as_device_ptr().as_raw();
                let mut ls_ptr = d_ls.as_device_ptr().as_raw();
                let mut ncomb_i = chunk as i32;
                let mut ring_len_param = ring_len_i;

                let row_off_bytes = (start * len * std::mem::size_of::<f32>()) as u64;
                let mut u_ptr = d_upper.as_device_ptr().as_raw().wrapping_add(row_off_bytes);
                let mut l_ptr = d_lower.as_device_ptr().as_raw().wrapping_add(row_off_bytes);
                let mut uc_ptr = d_uchn.as_device_ptr().as_raw().wrapping_add(row_off_bytes);
                let mut lc_ptr = d_lchn.as_device_ptr().as_raw().wrapping_add(row_off_bytes);

                let mut args: [*mut c_void; 15] = [
                    &mut p_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut jl_ptr as *mut _ as *mut c_void,
                    &mut js_ptr as *mut _ as *mut c_void,
                    &mut tl_ptr as *mut _ as *mut c_void,
                    &mut ts_ptr as *mut _ as *mut c_void,
                    &mut ll_ptr as *mut _ as *mut c_void,
                    &mut ls_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut ring_len_param as *mut _ as *mut c_void,
                    &mut u_ptr as *mut _ as *mut c_void,
                    &mut l_ptr as *mut _ as *mut c_void,
                    &mut uc_ptr as *mut _ as *mut c_void,
                    &mut lc_ptr as *mut _ as *mut c_void,
                ];
                let dyn_shmem = (ring_len_i as usize) * 3 * std::mem::size_of::<f32>();
                this.stream
                    .launch(&func, grid, block, dyn_shmem.try_into().unwrap(), &mut args)
                    .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
            }
            this.stream
                .synchronize()
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))
        };

        // Chunk by device grid.x and shrink shared memory per chunk
        let mut launched = 0usize;
        while launched < rows {
            let chunk = (rows - launched).min(self.max_grid_x);
            let chunk_max_shift = max_shift_in_range(&js, &ts_, &ls, launched, chunk);
            let ring_len_i = (chunk_max_shift + 1) as i32;
            launch_chunk(self, launched, chunk, ring_len_i)?;
            launched += chunk;
        }
        self.maybe_log_batch_debug();

        Ok(DeviceGatorOscQuad {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols: len,
            },
            upper_change: DeviceArrayF32 {
                buf: d_uchn,
                rows,
                cols: len,
            },
            lower_change: DeviceArrayF32 {
                buf: d_lchn,
                rows,
                cols: len,
            },
        })
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn gatorosc_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        jaws_length: usize,
        jaws_shift: usize,
        teeth_length: usize,
        teeth_shift: usize,
        lips_length: usize,
        lips_shift: usize,
    ) -> Result<DeviceGatorOscQuad, CudaGatorOscError> {
        if cols == 0 || rows == 0 {
            return Err(CudaGatorOscError::InvalidInput("invalid dims".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaGatorOscError::InvalidInput("invalid dims".into()));
        }
        if prices_tm.len() != cols * rows {
            return Err(CudaGatorOscError::InvalidInput(
                "time-major length mismatch".into(),
            ));
            return Err(CudaGatorOscError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        if jaws_length == 0 || teeth_length == 0 || lips_length == 0 {
            return Err(CudaGatorOscError::InvalidInput(
                "non-positive length".into(),
            ));
            return Err(CudaGatorOscError::InvalidInput(
                "non-positive length".into(),
            ));
        }
        // Per-series first_valid
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                if !prices_tm[t * cols + s].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
                if !prices_tm[t * cols + s].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            first_valids[s] =
                fv.ok_or_else(|| CudaGatorOscError::InvalidInput(format!("series {} all NaN", s)))?;
            first_valids[s] =
                fv.ok_or_else(|| CudaGatorOscError::InvalidInput(format!("series {} all NaN", s)))?;
        }
        let needed_upper = jaws_length.max(teeth_length) + jaws_shift.max(teeth_shift);
        let needed_lower = teeth_length.max(lips_length) + teeth_shift.max(lips_shift);
        let needed = needed_upper.max(needed_lower);
        for s in 0..cols {
            let tail = rows - (first_valids[s] as usize);
            if tail < needed {
                return Err(CudaGatorOscError::InvalidInput(format!(
                    "series {} not enough valid data: needed >= {}, valid = {}",
                    s, needed, tail
                )));
            }
        }

        let d_prices = DeviceBuffer::from_slice(prices_tm)
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let mut d_uchn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        let mut d_lchn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("gatorosc_many_series_one_param_f32")
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        // Ring len drives per-thread shared memory
        let ring_len = (jaws_shift.max(teeth_shift).max(lips_shift) + 1) as i32;
        let per_thread_smem = (ring_len as usize) * 3 * std::mem::size_of::<f32>();
        let smem_budget = self.max_smem_per_block.saturating_sub(1024);
        let requested_block_x = self.policy.many_block_x.unwrap_or(128);
        let max_by_smem = if per_thread_smem == 0 {
            requested_block_x as usize
        } else {
            smem_budget / per_thread_smem
        };
        let mut block_x = requested_block_x.min((max_by_smem as u32).max(32));
        block_x -= block_x % 32; if block_x == 0 { block_x = 32; }
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaGatorOsc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
            (*(self as *const _ as *mut CudaGatorOsc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut fv_ptr = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut jl_i = jaws_length as i32;
            let mut js_i = jaws_shift as i32;
            let mut tl_i = teeth_length as i32;
            let mut ts_i = teeth_shift as i32;
            let mut ll_i = lips_length as i32;
            let mut ls_i = lips_shift as i32;
            let mut ring_i = ring_len;
            let mut u_ptr = d_upper.as_device_ptr().as_raw();
            let mut l_ptr = d_lower.as_device_ptr().as_raw();
            let mut uc_ptr = d_uchn.as_device_ptr().as_raw();
            let mut lc_ptr = d_lchn.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 15] = [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut jl_i as *mut _ as *mut c_void,
                &mut js_i as *mut _ as *mut c_void,
                &mut tl_i as *mut _ as *mut c_void,
                &mut ts_i as *mut _ as *mut c_void,
                &mut ll_i as *mut _ as *mut c_void,
                &mut ls_i as *mut _ as *mut c_void,
                &mut ring_i as *mut _ as *mut c_void,
                &mut u_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut uc_ptr as *mut _ as *mut c_void,
                &mut lc_ptr as *mut _ as *mut c_void,
            ];
            let dyn_shmem = per_thread_smem * (block_x as usize);
            self.stream
                .launch(&func, grid, block, dyn_shmem.try_into().unwrap(), &mut args)
                .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGatorOscError::Cuda(e.to_string()))?;

        self.maybe_log_many_debug();
        Ok(DeviceGatorOscQuad {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols,
            },
            upper_change: DeviceArrayF32 {
                buf: d_uchn,
                rows,
                cols,
            },
            lower_change: DeviceArrayF32 {
                buf: d_lchn,
                rows,
                cols,
            },
        })
    }
}

// ---- Local helpers ----
fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        vec![start]
    } else {
        (start..=end).step_by(step).collect()
    }
}
fn expand_grid(r: &GatorOscBatchRange) -> Vec<GatorOscParams> {
    let jl = axis(r.jaws_length);
    let js = axis(r.jaws_shift);
    let tl = axis(r.teeth_length);
    let ts = axis(r.teeth_shift);
    let ll = axis(r.lips_length);
    let ls = axis(r.lips_shift);
    let mut out =
        Vec::with_capacity(jl.len() * js.len() * tl.len() * ts.len() * ll.len() * ls.len());
    for &a in &jl {
        for &b in &js {
            for &c in &tl {
                for &d in &ts {
                    for &e in &ll {
                        for &f in &ls {
                            out.push(GatorOscParams {
                                jaws_length: Some(a),
                                jaws_shift: Some(b),
                                teeth_length: Some(c),
                                teeth_shift: Some(d),
                                lips_length: Some(e),
                                lips_shift: Some(f),
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

#[inline]
fn max_shift_in_range(js: &[i32], ts: &[i32], ls: &[i32], start: usize, count: usize) -> usize {
    let mut m = 0usize;
    for i in start..start + count {
        m = m.max(js[i] as usize).max(ts[i] as usize).max(ls[i] as usize);
    }
    m
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 96; // modest grid to bound VRAM (e.g., 4x4x6)

    fn mem_required() -> usize {
        let out_bytes = 4 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + (64 << 20)
    }

    struct GatorBatchState {
        cuda: CudaGatorOsc,
        data: Vec<f32>,
        sweep: GatorOscBatchRange,
    }
    impl CudaBenchState for GatorBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .gatorosc_batch_dev(&self.data, &self.sweep)
                .expect("gator batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaGatorOsc::new(0).expect("cuda gator");
        let data = gen_series(ONE_SERIES_LEN);
        // 3×(length) × 2×(shift) grid ~ 96 rows
        let sweep = GatorOscBatchRange {
            jaws_length: (8, 14, 2),
            jaws_shift: (2, 6, 2),
            teeth_length: (6, 10, 2),
            teeth_shift: (1, 5, 2),
            lips_length: (4, 8, 2),
            lips_shift: (0, 4, 2),
        };
        Box::new(GatorBatchState { cuda, data, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "gatorosc",
            "one_series_many_params",
            "gatorosc_cuda_batch_dev",
            "1m_x_96",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(mem_required())]
    }
}
