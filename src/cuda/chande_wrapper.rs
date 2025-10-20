//! CUDA wrapper for Chande (Chandelier Exit) kernels.
//!
//! Parity targets:
//! - ALMA-like API returning `DeviceArrayF32` VRAM handles.
//! - Batch (one-series × many-params) and Many-series × one-param (time-major).
//! - Warmup/NaN identical to scalar: warm = first_valid + period - 1.
//! - Simple VRAM estimate + chunked launches (grid.x) to respect limits.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::chande::ChandeBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaChandeError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaChandeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaChandeError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaChandeError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaChandeError {}

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
pub struct CudaChandePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaChandePolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

pub struct CudaChande {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaChandePolicy,

    // Persistent monotone-deque workspaces for one-series × many-params path
    dq_idx: Option<DeviceBuffer<i32>>,
    dq_val: Option<DeviceBuffer<f32>>,
    dq_combo_cap: usize,
    dq_cap: usize,
}

impl CudaChande {
    pub fn new(device_id: usize) -> Result<Self, CudaChandeError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/chande_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                { m } else { Module::from_ptx(ptx, &[]).map_err(|e| CudaChandeError::Cuda(e.to_string()))? }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaChandePolicy::default(),
            // NEW: ring-buffer workspace cache
            dq_idx: None,
            dq_val: None,
            dq_combo_cap: 0,
            dq_cap: 0,
        })
    }

    pub fn set_policy(&mut self, policy: CudaChandePolicy) { self.policy = policy; }
    pub fn synchronize(&self) -> Result<(), CudaChandeError> { self.stream.synchronize().map_err(|e| CudaChandeError::Cuda(e.to_string())) }

    fn first_valid_hlc(high: &[f32], low: &[f32], close: &[f32]) -> Result<usize, CudaChandeError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaChandeError::InvalidInput("empty input".into()));
        }
        let n = high.len().min(low.len()).min(close.len());
        for i in 0..n {
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() { return Ok(i); }
        }
        Err(CudaChandeError::InvalidInput("all values are NaN".into()))
    }

    fn device_will_fit(bytes: usize, headroom: usize) -> bool {
        let check = match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && v.to_lowercase() != "false", Err(_) => true };
        if !check { return true; }
        if let Ok((free, _)) = mem_get_info() { bytes.saturating_add(headroom) <= free } else { true }
    }

    #[inline]
    fn next_pow2_usize(x: usize) -> usize { (x.max(1)).next_power_of_two() }

    fn ensure_workspace(&mut self, combos: usize, queue_cap: usize) -> Result<(), CudaChandeError> {
        if self.dq_idx.is_some() && self.dq_combo_cap >= combos && self.dq_cap >= queue_cap {
            return Ok(());
        }
        let need = combos
            .checked_mul(queue_cap)
            .ok_or_else(|| CudaChandeError::InvalidInput("dq size overflow".into()))?;
        self.dq_idx = Some(DeviceBuffer::<i32>::zeroed(need).map_err(|e| CudaChandeError::Cuda(e.to_string()))?);
        self.dq_val = Some(DeviceBuffer::<f32>::zeroed(need).map_err(|e| CudaChandeError::Cuda(e.to_string()))?);
        self.dq_combo_cap = combos;
        self.dq_cap = queue_cap;
        Ok(())
    }

    fn will_fit_full_output_one_series(&self, n_combos: usize, len: usize, queue_cap: usize) -> bool {
        let in_bytes = 3usize * len * std::mem::size_of::<f32>();
        let params_bytes = n_combos * (3 * std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>());
        let out_bytes = n_combos * len * std::mem::size_of::<f32>();
        let dq_bytes = n_combos * queue_cap * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let headroom = 64usize * 1024 * 1024;
        Self::device_will_fit(in_bytes + params_bytes + out_bytes + dq_bytes, headroom)
    }

    fn chunk_size_for_batch(n_combos: usize, len: usize) -> usize {
        // Inputs: 3×len f32; params: periods/mults/dirs/alphas/warm per combo; outputs: combos×len f32.
        let in_bytes = 3 * len * std::mem::size_of::<f32>();
        let params_bytes = n_combos * (std::mem::size_of::<i32>() * 3 + std::mem::size_of::<f32>() * 2);
        let out_per_combo = len * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024;
        let mut chunk = n_combos.max(1);
        while chunk > 1 {
            let need = in_bytes + params_bytes + chunk * out_per_combo + headroom;
            if Self::device_will_fit(need, 0) { break; }
            chunk = (chunk + 1) / 2;
        }
        chunk.max(1)
    }

    pub fn chande_batch_dev(
        &mut self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &ChandeBatchRange,
        direction: &str,
    ) -> Result<DeviceArrayF32, CudaChandeError> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaChandeError::InvalidInput("input length mismatch".into()));
        }
        let len = high.len();
        let first_valid = Self::first_valid_hlc(high, low, close)?;
        // Expand params on host
        let (ps, pe, pst) = sweep.period;
        let (ms, me, mst) = sweep.mult;
        if ps == 0 { return Err(CudaChandeError::InvalidInput("period must be > 0".into())); }
        if !(direction.eq_ignore_ascii_case("long") || direction.eq_ignore_ascii_case("short")) {
            return Err(CudaChandeError::InvalidInput("direction must be 'long' or 'short'".into()));
        }
        let dir_flag = if direction.eq_ignore_ascii_case("long") { 1i32 } else { 0i32 };
        let periods: Vec<usize> = if pst == 0 || ps == pe { vec![ps] } else { (ps..=pe).step_by(pst).collect() };
        let mults_host: Vec<f32> = if mst.abs() < f64::EPSILON || (ms - me).abs() < f64::EPSILON {
            vec![ms as f32]
        } else {
            let mut v = Vec::new(); let mut x = ms; while x <= me + 1e-12 { v.push(x as f32); x += mst; } v
        };
        if periods.is_empty() || mults_host.is_empty() {
            return Err(CudaChandeError::InvalidInput("no parameter combos".into()));
        }
        let mut h_periods = Vec::<i32>::new();
        let mut h_alphas = Vec::<f32>::new();
        let mut h_warms = Vec::<i32>::new();
        let mut h_mults = Vec::<f32>::new();
        let mut h_dirs  = Vec::<i32>::new();
        let mut max_p = 0usize;
        for &p in &periods {
            if p == 0 || p > len || (len - first_valid) < p {
                return Err(CudaChandeError::InvalidInput(format!(
                    "invalid period {} for data length {} (valid after {}: {})",
                    p, len, first_valid, len - first_valid
                )));
            }
            if p > max_p { max_p = p; }
            for &m in &mults_host {
                h_periods.push(p as i32);
                h_alphas.push(1.0f32 / (p as f32));
                h_warms.push((first_valid + p - 1) as i32);
                h_mults.push(m);
                h_dirs.push(dir_flag);
            }
        }
        let n_combos = h_periods.len();

        // Device buffers (inputs & params)
        let d_high  = DeviceBuffer::from_slice(high).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_low   = DeviceBuffer::from_slice(low).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&h_periods).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_mults   = DeviceBuffer::from_slice(&h_mults).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_dirs    = DeviceBuffer::from_slice(&h_dirs).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_alphas  = DeviceBuffer::from_slice(&h_alphas).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;

        // Check for new kernels and compute queue_cap for deque buffers
        let have_oneseries = self.module.get_function("chande_one_series_many_params_f32").is_ok();
        let have_oneseries_tr = self.module.get_function("chande_one_series_many_params_from_tr_f32").is_ok();
        let queue_cap = Self::next_pow2_usize(max_p + 1);

        // VRAM checks and workspace ensure
        if have_oneseries || have_oneseries_tr {
            if !self.will_fit_full_output_one_series(n_combos, len, queue_cap) {
                return Err(CudaChandeError::Cuda("insufficient device memory for output+workspace".into()));
            }
            self.ensure_workspace(n_combos, queue_cap)?;
        } else {
            let in_bytes = 3 * len * std::mem::size_of::<f32>();
            let params_bytes = n_combos * (3 * std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>());
            let out_bytes = n_combos * len * std::mem::size_of::<f32>();
            let headroom = 64 * 1024 * 1024;
            if !Self::device_will_fit(in_bytes + params_bytes + out_bytes, headroom) {
                return Err(CudaChandeError::Cuda("insufficient device memory for output".into()));
            }
        }

        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * len) }
            .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;

        if have_oneseries {
            // Fast path: on-the-fly TR, no extra copy
            let func = self.module.get_function("chande_one_series_many_params_f32")
                .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
            let warps_needed = ((n_combos + 31) / 32) as u32;
            let warps_per_block = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } => (block_x.max(32) / 32),
                BatchKernelPolicy::Auto => 4,
            }
            .max(1);
            let block_x = warps_per_block * 32;
            let grid_x = ((warps_needed + warps_per_block - 1) / warps_per_block).max(1);
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut high_ptr   = d_high.as_device_ptr().as_raw();
                let mut low_ptr    = d_low.as_device_ptr().as_raw();
                let mut close_ptr  = d_close.as_device_ptr().as_raw();
                let mut periods_ptr= d_periods.as_device_ptr().as_raw();
                let mut mults_ptr  = d_mults.as_device_ptr().as_raw();
                let mut dirs_ptr   = d_dirs.as_device_ptr().as_raw();
                let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
                let mut first_i    = first_valid as i32;
                let mut len_i      = len as i32;
                let mut combos_i   = n_combos as i32;
                let mut qcap_i     = queue_cap as i32;
                let dq_idx_ref = self.dq_idx.as_ref().unwrap();
                let dq_val_ref = self.dq_val.as_ref().unwrap();
                let mut dq_idx_ptr = dq_idx_ref.as_device_ptr().as_raw();
                let mut dq_val_ptr = dq_val_ref.as_device_ptr().as_raw();
                let mut out_ptr   = d_out.as_device_ptr().as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut mults_ptr as *mut _ as *mut c_void,
                    &mut dirs_ptr as *mut _ as *mut c_void,
                    &mut alphas_ptr as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut qcap_i as *mut _ as *mut c_void,
                    &mut dq_idx_ptr as *mut _ as *mut c_void,
                    &mut dq_val_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
            }
        } else {
            // Fallback to legacy batch implementation
            let use_tr = self.module.get_function("chande_batch_from_tr_f32").is_ok();
            let func = if use_tr {
                self.module.get_function("chande_batch_from_tr_f32").unwrap()
            } else {
                self.module
                    .get_function("chande_batch_f32")
                    .map_err(|e| CudaChandeError::Cuda(e.to_string()))?
            };

            let d_tr: Option<DeviceBuffer<f32>> = if use_tr {
                let mut tr = vec![0f32; len];
                let mut prev_c = close[first_valid];
                for t in first_valid..len {
                    let hi = high[t];
                    let lo = low[t];
                    if t == first_valid {
                        tr[t] = hi - lo;
                    } else {
                        let mut tri = hi - lo;
                        let hc = (hi - prev_c).abs();
                        if hc > tri { tri = hc; }
                        let lc = (lo - prev_c).abs();
                        if lc > tri { tri = lc; }
                        tr[t] = tri;
                    }
                    prev_c = close[t];
                }
                Some(DeviceBuffer::from_slice(&tr).map_err(|e| CudaChandeError::Cuda(e.to_string()))?)
            } else { None };

            // No VRAM-driven chunking anymore; we validated full output.
            let block_x = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => block_x, BatchKernelPolicy::Auto => 256 };
            let chunk = n_combos.max(1);
            let mut launched = 0usize;
            // Upload warms per chunk to simplify pointer arithmetic
            while launched < n_combos {
                let cur = (n_combos - launched).min(chunk);
                let grid: GridSize = (cur as u32, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                unsafe {
                    let mut len_i   = len as i32;
                    let mut first_i = first_valid as i32;
                    let mut cur_i   = cur as i32;
                    let mut out_ptr = d_out.as_device_ptr().as_raw()
                        .wrapping_add(((launched * len) * std::mem::size_of::<f32>()) as u64);

                    if use_tr {
                        let d_tr_ref = d_tr.as_ref().unwrap();
                        let mut high_ptr = d_high.as_device_ptr().as_raw();
                        let mut low_ptr  = d_low.as_device_ptr().as_raw();
                        let mut tr_ptr   = d_tr_ref.as_device_ptr().as_raw();
                        let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                        let mut mults_ptr   = d_mults.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                        let mut dirs_ptr    = d_dirs.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                        let mut alphas_ptr  = d_alphas.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                        let warms_slice = &h_warms[launched..(launched + cur)];
                        let d_warms   = DeviceBuffer::from_slice(warms_slice).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
                        let mut warms_ptr = d_warms.as_device_ptr().as_raw();

                        let args: &mut [*mut c_void] = &mut [
                            &mut high_ptr as *mut _ as *mut c_void,
                            &mut low_ptr as *mut _ as *mut c_void,
                            &mut tr_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut mults_ptr as *mut _ as *mut c_void,
                            &mut dirs_ptr as *mut _ as *mut c_void,
                            &mut alphas_ptr as *mut _ as *mut c_void,
                            &mut warms_ptr as *mut _ as *mut c_void,
                            &mut len_i as *mut _ as *mut c_void,
                            &mut first_i as *mut _ as *mut c_void,
                            &mut cur_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, args)
                            .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
                    } else {
                        let mut high_ptr = d_high.as_device_ptr().as_raw();
                        let mut low_ptr  = d_low.as_device_ptr().as_raw();
                        let mut close_ptr= d_close.as_device_ptr().as_raw();
                        let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                        let mut mults_ptr   = d_mults.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                        let mut dirs_ptr    = d_dirs.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                        let mut alphas_ptr  = d_alphas.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                        let warms_slice = &h_warms[launched..(launched + cur)];
                        let d_warms   = DeviceBuffer::from_slice(warms_slice).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
                        let mut warms_ptr = d_warms.as_device_ptr().as_raw();

                        let args: &mut [*mut c_void] = &mut [
                            &mut high_ptr as *mut _ as *mut c_void,
                            &mut low_ptr as *mut _ as *mut c_void,
                            &mut close_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut mults_ptr as *mut _ as *mut c_void,
                            &mut dirs_ptr as *mut _ as *mut c_void,
                            &mut alphas_ptr as *mut _ as *mut c_void,
                            &mut warms_ptr as *mut _ as *mut c_void,
                            &mut len_i as *mut _ as *mut c_void,
                            &mut first_i as *mut _ as *mut c_void,
                            &mut cur_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, args)
                            .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
                    }
                }
                launched += cur;
            }
        }

        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: len })
    }

    fn first_valids_time_major(high_tm: &[f32], low_tm: &[f32], close_tm: &[f32], cols: usize, rows: usize) -> Result<Vec<i32>, CudaChandeError> {
        let n = cols.checked_mul(rows).ok_or_else(|| CudaChandeError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != n || low_tm.len() != n || close_tm.len() != n {
            return Err(CudaChandeError::InvalidInput("time-major input length mismatch".into()));
        }
        let mut out = vec![-1i32; cols];
        for s in 0..cols { for t in 0..rows {
            let idx = t * cols + s;
            let h = high_tm[idx]; let l = low_tm[idx]; let c = close_tm[idx];
            if !h.is_nan() && !l.is_nan() && !c.is_nan() { out[s] = t as i32; break; }
        }}
        Ok(out)
    }

    pub fn chande_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        direction: &str,
    ) -> Result<DeviceArrayF32, CudaChandeError> {
        if period == 0 { return Err(CudaChandeError::InvalidInput("period must be > 0".into())); }
        if !(direction.eq_ignore_ascii_case("long") || direction.eq_ignore_ascii_case("short")) {
            return Err(CudaChandeError::InvalidInput("direction must be 'long' or 'short'".into()));
        }
        let first_valids = Self::first_valids_time_major(high_tm, low_tm, close_tm, cols, rows)?;
        if rows < period { return Err(CudaChandeError::InvalidInput("not enough rows for period".into())); }

        let d_high  = DeviceBuffer::from_slice(high_tm).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_low   = DeviceBuffer::from_slice(low_tm).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close_tm).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let d_fv    = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;

        let func = self.module.get_function("chande_many_series_one_param_f32")
            .map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => block_x, ManySeriesKernelPolicy::Auto => 256 };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let dir_flag: i32 = if direction.eq_ignore_ascii_case("long") { 1 } else { 0 };
        let alpha = 1.0f32 / (period as f32);
        unsafe {
            let mut high_ptr   = d_high.as_device_ptr().as_raw();
            let mut low_ptr    = d_low.as_device_ptr().as_raw();
            let mut close_ptr  = d_close.as_device_ptr().as_raw();
            let mut fv_ptr     = d_fv.as_device_ptr().as_raw();
            let mut period_i   = period as i32;
            let mut mult_f     = mult;
            let mut dir_i      = dir_flag;
            let mut alpha_f    = alpha;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr    = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut mult_f as *mut _ as *mut c_void,
                &mut dir_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaChandeError::Cuda(e.to_string()))?;
        }

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ---------------- Bench profiles ----------------
#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 512_000; // moderate, naive window scan is O(period)

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i]; if v.is_nan() { continue; }
            let x = i as f32 * 0.002f32;
            let off = (0.004 * x.sin()).abs() + 0.12;
            high[i] = v + off; low[i] = v - off;
        }
        (high, low)
    }

    struct ChandeBatchState { cuda: CudaChande, high: Vec<f32>, low: Vec<f32>, close: Vec<f32>, sweep: ChandeBatchRange, dir: String }
    impl CudaBenchState for ChandeBatchState { fn launch(&mut self) { let _ = self.cuda.chande_batch_dev(&self.high, &self.low, &self.close, &self.sweep, &self.dir).unwrap(); } }

    struct ChandeManyState { cuda: CudaChande, high_tm: Vec<f32>, low_tm: Vec<f32>, close_tm: Vec<f32>, cols: usize, rows: usize, period: usize, mult: f32, dir: String }
    impl CudaBenchState for ChandeManyState { fn launch(&mut self) { let _ = self.cuda.chande_many_series_one_param_time_major_dev(&self.high_tm, &self.low_tm, &self.close_tm, self.cols, self.rows, self.period, self.mult, &self.dir).unwrap(); } }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let len = ONE_SERIES_LEN; let close = gen_series(len); let (high, low) = synth_hlc_from_close(&close);
        let sweep = ChandeBatchRange { period: (10, 40, 5), mult: (2.0, 4.0, 1.0) };
        Box::new(ChandeBatchState { cuda: CudaChande::new(0).unwrap(), high, low, close, sweep, dir: "long".into() })
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let (cols, rows, period, mult) = (128usize, 262_144usize, 22usize, 3.0f32);
        let mut close_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols { for t in s..rows { let x = (t as f32) + (s as f32) * 0.2; close_tm[t * cols + s] = (x * 0.0017).sin() + 0.00015 * x; } }
        let (mut high_tm, mut low_tm) = (close_tm.clone(), close_tm.clone());
        for s in 0..cols { for t in 0..rows { let v = close_tm[t * cols + s]; if v.is_nan() { continue; } let x = (t as f32) * 0.002; let off = (0.004 * x.cos()).abs() + 0.11; high_tm[t * cols + s] = v + off; low_tm[t * cols + s] = v - off; } }
        Box::new(ChandeManyState { cuda: CudaChande::new(0).unwrap(), high_tm, low_tm, close_tm, cols, rows, period, mult, dir: "long".into() })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let scen_batch = CudaBenchScenario::new(
            "chande",
            "one_series_many_params",
            "chande_cuda_batch_dev",
            "512k_x_params",
            prep_one_series_many_params,
        );
        let scen_many = CudaBenchScenario::new(
            "chande",
            "many_series_one_param",
            "chande_cuda_many_series_one_param_dev",
            "128x262k",
            prep_many_series_one_param,
        );
        vec![scen_batch, scen_many]
    }
}
