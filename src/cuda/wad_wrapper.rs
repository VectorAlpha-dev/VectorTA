#![cfg(feature = "cuda")]

//! CUDA wrapper for WAD aligned to ALMA parity:
//! - Policy enums and light introspection for batch and many-series
//! - JIT options: DetermineTargetFromContext + OptLevel O2, with fallbacks
//! - NON_BLOCKING stream
//! - VRAM estimation + grid chunking (<= 65_535 rows) for batch

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaWadError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWadError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWadError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}
impl std::error::Error for CudaWadError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self { BatchKernelPolicy::Auto }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy { fn default() -> Self { ManySeriesKernelPolicy::Auto } }

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaWadPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaWad {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaWadPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaWad {
    pub fn new(device_id: usize) -> Result<Self, CudaWadError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wad_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaWadPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, p: CudaWadPolicy) { self.policy = p; }
    pub fn policy(&self) -> &CudaWadPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaWadError> {
        self.stream.synchronize().map_err(|e| CudaWadError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[DEBUG] WAD batch selected kernel: {:?}", sel);
                unsafe { (*(self as *const _ as *mut CudaWad)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[DEBUG] WAD many-series selected kernel: {:?}", sel);
                unsafe { (*(self as *const _ as *mut CudaWad)).debug_many_logged = true; }
            }
        }
    }

    // -------- Utilities --------
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Ok((free, _)) = mem_get_info() { required_bytes.saturating_add(headroom) <= free } else { true }
    }

    // -------- Batch (one-series × many-params) --------
    fn prepare_batch_inputs(
        high: &[f32], low: &[f32], close: &[f32],
    ) -> Result<usize, CudaWadError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaWadError::InvalidInput("empty input slices".into()));
        }
        let len = high.len();
        if low.len() != len || close.len() != len {
            return Err(CudaWadError::InvalidInput("input slice length mismatch".into()));
        }
        if high.iter().all(|x| x.is_nan()) || low.iter().all(|x| x.is_nan()) || close.iter().all(|x| x.is_nan()) {
            return Err(CudaWadError::InvalidInput("all values are NaN".into()));
        }
        Ok(len)
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWadError> {
        // Grid-stride batch kernel: choose a robust grid/block for occupancy
        let func = self
            .module
            .get_function("wad_batch_f32")
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        let block_x = self.default_block_x("WAD_BLOCK_X", 256);
        let grid_x = self.choose_grid_1d(n_combos, block_x)?;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr  = d_high.as_device_ptr().as_raw();
            let mut low_ptr   = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut len_i     = series_len as i32;
            let mut combos_i  = n_combos as i32;
            let mut out_ptr   = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr  as *mut _ as *mut c_void,
                &mut low_ptr   as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut len_i     as *mut _ as *mut c_void,
                &mut combos_i  as *mut _ as *mut c_void,
                &mut out_ptr   as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        }

        unsafe { (*(self as *const _ as *mut CudaWad)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn run_batch(
        &self,
        high: &[f32], low: &[f32], close: &[f32],
        n_combos: usize,
    ) -> Result<DeviceArrayF32, CudaWadError> {
        let series_len = Self::prepare_batch_inputs(high, low, close)?;

        // VRAM check: 3 inputs + output (with overflow checks)
        let required_cells_inputs = 3usize
            .checked_mul(series_len)
            .ok_or_else(|| CudaWadError::InvalidInput("size overflow".into()))?;
        let required_cells_output = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaWadError::InvalidInput("size overflow".into()))?;
        let required = (required_cells_inputs + required_cells_output) * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(required, headroom) {
            return Err(CudaWadError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Upload inputs asynchronously
        let d_high  = unsafe { DeviceBuffer::from_slice_async(high,  &self.stream) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let d_low   = unsafe { DeviceBuffer::from_slice_async(low,   &self.stream) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close, &self.stream) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        // Output buffer
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream)
        }.map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        if n_combos > 1 {
            // Compute once then broadcast to all rows (WAD rows are identical)
            let mut d_row: DeviceBuffer<f32> = unsafe {
                DeviceBuffer::uninitialized_async(series_len, &self.stream)
            }.map_err(|e| CudaWadError::Cuda(e.to_string()))?;

            self.launch_compute_single_row(&d_high, &d_low, &d_close, series_len, &mut d_row)?;
            self.launch_broadcast_row(&d_row, series_len, n_combos, &mut d_out)?;
        } else {
            // Regular batch kernel (grid-stride over combos)
            self.launch_batch_kernel(&d_high, &d_low, &d_close, series_len, n_combos, &mut d_out)?;
        }

        // Ensure completion before returning
        self.stream.synchronize().map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    pub fn wad_batch_dev(
        &self,
        high: &[f32], low: &[f32], close: &[f32],
    ) -> Result<DeviceArrayF32, CudaWadError> {
        // WAD has no parameters; batch rows=1 for parity
        self.run_batch(high, low, close, 1)
    }

    pub fn wad_batch_into_host_f32(
        &self,
        high: &[f32], low: &[f32], close: &[f32], out: &mut [f32],
    ) -> Result<(usize, usize), CudaWadError> {
        let arr = self.wad_batch_dev(high, low, close)?;
        if out.len() != arr.cols * arr.rows {
            return Err(CudaWadError::InvalidInput(format!(
                "out slice length {} != expected {}",
                out.len(), arr.cols * arr.rows
            )));
        }
        unsafe { arr.buf.async_copy_to(out, &self.stream) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        self.stream.synchronize().map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols))
    }

    // Back-compat convenience used by existing tests
    pub fn wad_series_dev(
        &self,
        high: &[f32], low: &[f32], close: &[f32],
    ) -> Result<DeviceArrayF32, CudaWadError> {
        self.wad_batch_dev(high, low, close)
    }

    // Back-compat: original API expected length on success
    pub fn wad_into_host_f32(
        &self,
        high: &[f32], low: &[f32], close: &[f32], out: &mut [f32],
    ) -> Result<usize, CudaWadError> {
        let (_rows, cols) = self.wad_batch_into_host_f32(high, low, close, out)?;
        Ok(cols)
    }

    // -------- Many-series × one-param (time-major) --------
    fn prepare_many_series_inputs(
        high_tm: &[f32], low_tm: &[f32], close_tm: &[f32], cols: usize, rows: usize,
    ) -> Result<(), CudaWadError> {
        if cols == 0 || rows == 0 { return Err(CudaWadError::InvalidInput("cols/rows must be > 0".into())); }
        if high_tm.len() != cols * rows || low_tm.len() != cols * rows || close_tm.len() != cols * rows {
            return Err(CudaWadError::InvalidInput("input length != cols*rows".into()));
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        cols: usize, rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWadError> {
        let func = self
            .module
            .get_function("wad_many_series_one_param_f32")
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        // Many-series: one thread per series; grid-stride over series with robust defaults
        let block_x = self.default_block_x("WAD_MS_BLOCK_X", 256);
        let grid_x = self.choose_grid_1d(cols, block_x)?;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x.max(1), 1, 1).into();
        unsafe { (*(self as *const _ as *mut CudaWad)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }

        unsafe {
            let mut high_ptr  = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr   = d_low_tm.as_device_ptr().as_raw();
            let mut close_ptr = d_close_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    // ----- Helpers: device SM count and launch heuristics -----
    #[inline]
    fn sm_count(&self) -> Result<u32, CudaWadError> {
        let dev = Device::get_device(self.device_id)
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        dev.get_attribute(DeviceAttribute::MultiprocessorCount)
            .map(|v| v as u32)
            .map_err(|e| CudaWadError::Cuda(e.to_string()))
    }

    /// Pick a default block size unless user overrides via env. Defaults to 256.
    #[inline]
    fn default_block_x(&self, env_key: &str, fallback: u32) -> u32 {
        env::var(env_key)
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&bx| bx != 0)
            .unwrap_or(fallback)
    }

    /// Given problem size N and threads-per-block, choose a grid that is large enough
    /// to saturate the GPU but not absurdly large. Heuristic: min(ceil_div(N, block_x), SMs * 32).
    #[inline]
    fn choose_grid_1d(&self, n: usize, block_x: u32) -> Result<u32, CudaWadError> {
        let sm = self.sm_count()?;
        let target_blocks = sm.saturating_mul(32);
        let need = ((n as u64 + block_x as u64 - 1) / block_x as u64) as u32;
        Ok(need.max(1).min(target_blocks.max(1)))
    }

    // ----- Optional helpers: compute-once + broadcast for WAD -----
    fn launch_compute_single_row(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        series_len: usize,
        d_row_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWadError> {
        let func = self
            .module
            .get_function("wad_compute_single_row_f32")
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut high_ptr  = d_high.as_device_ptr().as_raw();
            let mut low_ptr   = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut len_i     = series_len as i32;
            let mut out_ptr   = d_row_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr  as *mut _ as *mut c_void,
                &mut low_ptr   as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut len_i     as *mut _ as *mut c_void,
                &mut out_ptr   as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn launch_broadcast_row(
        &self,
        d_row: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWadError> {
        let func = self
            .module
            .get_function("broadcast_row_f32")
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        let total = series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaWadError::InvalidInput("overflow in broadcast size".into()))?;

        // Reuse the main batch block size default; no indicator-specific extra env knob.
        let block_x = self.default_block_x("WAD_BLOCK_X", 256);
        let grid_x = self.choose_grid_1d(total, block_x)?;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut row_ptr  = d_row.as_device_ptr().as_raw();
            let mut len_i    = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr  = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut row_ptr  as *mut _ as *mut c_void,
                &mut len_i    as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr  as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn wad_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32], low_tm: &[f32], close_tm: &[f32],
        cols: usize, rows: usize,
    ) -> Result<DeviceArrayF32, CudaWadError> {
        Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows)?;

        let required = (3 * cols * rows + cols * rows) * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaWadError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_high  = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let d_low   = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_high, &d_low, &d_close, cols, rows, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    fn bytes_one_series() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 32 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i]; if v.is_nan() { continue; }
            let x = i as f32 * 0.0027; let off = (0.0031 * x.cos()).abs() + 0.12;
            high[i] = v + off; low[i] = v - off;
        }
        (high, low)
    }

    struct WadState { cuda: CudaWad, high: Vec<f32>, low: Vec<f32>, close: Vec<f32> }
    impl CudaBenchState for WadState {
        fn launch(&mut self) {
            let _ = self.cuda.wad_batch_dev(&self.high, &self.low, &self.close).expect("wad kernel");
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaWad::new(0).expect("cuda wad");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        Box::new(WadState { cuda, high, low, close })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new("wad", "one_series", "wad_cuda_series", "1m", prep_one_series)
            .with_sample_size(10)
            .with_mem_required(bytes_one_series())]
    }
}
