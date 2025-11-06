//! CUDA wrapper for IFT RSI (Inverse Fisher Transform of RSI).
//!
//! Parity goals with ALMA wrapper:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/ift_rsi_kernel.ptx"))
//! - Non-blocking stream; DetermineTargetFromContext + O2 with fallbacks
//! - VRAM checks (with ~64 MB headroom) and simple grid policies
//! - Public device entry points:
//!     - `ift_rsi_batch_dev(&[f32], &IftRsiBatchRange) -> (DeviceArrayF32, Vec<IftRsiParams>)`
//!     - `ift_rsi_many_series_one_param_time_major_dev(&[f32], cols, rows, &IftRsiParams)`

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::ift_rsi::{IftRsiBatchRange, IftRsiParams};
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
pub enum CudaIftRsiError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaIftRsiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaIftRsiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaIftRsiError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaIftRsiError {}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    /// One block per combo; thread 0 performs the scan. `block_x` kept small.
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    /// 1D launch over series; dynamic shared memory sized to `block_x * wma_period * 4`.
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaIftRsiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32, shmem_bytes: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32, shmem_bytes: u32 },
}

pub struct CudaIftRsi {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaIftRsiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // --- cached device caps for launch heuristics ---
    sm_count: u32,
    max_smem_per_block: usize,
    warp_size: u32,
    max_threads_per_block: u32,
}

impl CudaIftRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaIftRsiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;

        // Query device caps for heuristics and safety checks
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))? as u32;
        let max_smem_per_block = device
            .get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))? as usize;
        let warp_size = device
            .get_attribute(DeviceAttribute::WarpSize)
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))? as u32;
        let max_threads_per_block = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))? as u32;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ift_rsi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaIftRsiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
            max_smem_per_block,
            warp_size,
            max_threads_per_block,
        })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaIftRsiError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaIftRsiPolicy) { self.policy = policy; }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn expand_grid(r: &IftRsiBatchRange) -> Vec<IftRsiParams> {
        fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end { return vec![start]; }
            (start..=end).step_by(step).collect()
        }
        let rsi = axis(r.rsi_period);
        let wma = axis(r.wma_period);
        let mut out = Vec::with_capacity(rsi.len() * wma.len());
        for &rp in &rsi {
            for &wp in &wma {
                out.push(IftRsiParams { rsi_period: Some(rp), wma_period: Some(wp) });
            }
        }
        for &rp in &rsi {
            for &wp in &wma {
                out.push(IftRsiParams {
                    rsi_period: Some(rp),
                    wma_period: Some(wp),
                });
            }
        }
        out
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &IftRsiBatchRange,
    ) -> Result<(Vec<IftRsiParams>, usize, usize, usize), CudaIftRsiError> {
        let len = data_f32.len();
        if len == 0 {
            return Err(CudaIftRsiError::InvalidInput("empty input".into()));
        }
        if len == 0 {
            return Err(CudaIftRsiError::InvalidInput("empty input".into()));
        }
        let mut first_valid: Option<usize> = None;
        for i in 0..len {
            let v = data_f32[i];
            if v == v {
                first_valid = Some(i);
                break;
            }
        }
        let first_valid = first_valid
            .ok_or_else(|| CudaIftRsiError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaIftRsiError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if combos.is_empty() {
            return Err(CudaIftRsiError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_rp = combos.iter().map(|c| c.rsi_period.unwrap()).max().unwrap();
        let max_wp = combos.iter().map(|c| c.wma_period.unwrap()).max().unwrap();
        let need = core::cmp::max(max_rp, max_wp);
        if need == 0 || need > len {
            return Err(CudaIftRsiError::InvalidInput("invalid period".into()));
        }
        if len - first_valid < need {
            return Err(CudaIftRsiError::InvalidInput(
                "not enough valid data".into(),
            ));
        }
        if need == 0 || need > len {
            return Err(CudaIftRsiError::InvalidInput("invalid period".into()));
        }
        if len - first_valid < need {
            return Err(CudaIftRsiError::InvalidInput(
                "not enough valid data".into(),
            ));
        }
        Ok((combos, first_valid, len, max_wp))
    }

    pub fn ift_rsi_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &IftRsiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<IftRsiParams>), CudaIftRsiError> {
        let (combos, first_valid, len, max_wp) = Self::prepare_batch_inputs(data_f32, sweep)?;

        // VRAM estimate: input + two params + output
        let bytes_in = len * core::mem::size_of::<f32>();
        let bytes_params = combos.len() * 2 * core::mem::size_of::<i32>();
        let bytes_out = combos.len() * len * core::mem::size_of::<f32>();
        let required = bytes_in + bytes_params + bytes_out;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaIftRsiError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        let d_in = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let rsi_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.rsi_period.unwrap() as i32)
            .collect();
        let wma_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.wma_period.unwrap() as i32)
            .collect();
        let rsi_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.rsi_period.unwrap() as i32)
            .collect();
        let wma_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.wma_period.unwrap() as i32)
            .collect();
        let d_rp = unsafe { DeviceBuffer::from_slice_async(&rsi_i32, &self.stream) }
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let d_wp = unsafe { DeviceBuffer::from_slice_async(&wma_i32, &self.stream) }
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }
                .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("ift_rsi_batch_f32")
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;

        // One warp per block by default; grid capped to ~8x SMs per guide.
        // Safety check for shared memory ring used by each block
        let shmem_bytes_usize: usize = max_wp
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaIftRsiError::InvalidInput("wma_period too large".into()))?;
        if shmem_bytes_usize > self.max_smem_per_block {
            return Err(CudaIftRsiError::InvalidInput(format!(
                "wma_period={} requires {}B shared memory but device allows {}B per block",
                max_wp, shmem_bytes_usize, self.max_smem_per_block
            )));
        }

        // One warp per block by default; kernel grid-strides over combos
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => self.warp_size.max(32),
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        let target_blocks = self.sm_count.saturating_mul(8).max(1);
        let grid_x = (combos.len() as u32).min(target_blocks);
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shmem_bytes: u32 = shmem_bytes_usize as u32;
        unsafe { (*(self as *const _ as *mut CudaIftRsi)).last_batch = Some(BatchKernelSelected::Plain { block_x, shmem_bytes }); }
        unsafe {
            let mut in_ptr = d_in.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut n_combos_i = combos.len() as i32;
            let mut first_i = first_valid as i32;
            let mut rp_ptr = d_rp.as_device_ptr().as_raw();
            let mut wp_ptr = d_wp.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut rp_ptr as *mut _ as *mut c_void,
                &mut wp_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shmem_bytes, args)
                .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        }

        self.maybe_log_batch_debug();

        Ok((
            DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len },
            combos,
        ))
    }

    pub fn ift_rsi_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &IftRsiParams,
    ) -> Result<DeviceArrayF32, CudaIftRsiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaIftRsiError::InvalidInput("empty matrix".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaIftRsiError::InvalidInput("bad shape".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaIftRsiError::InvalidInput("empty matrix".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaIftRsiError::InvalidInput("bad shape".into()));
        }
        let rsi_p = params.rsi_period.unwrap_or(5);
        let wma_p = params.wma_period.unwrap_or(9);
        if rsi_p == 0 || wma_p == 0 || rsi_p > rows || wma_p > rows {
            return Err(CudaIftRsiError::InvalidInput("invalid periods".into()));
        }

        // Per-series first_valid (first non-NaN row)
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for r in 0..rows {
                let v = data_tm_f32[r * cols + s];
                if v == v {
                    fv = r as i32;
                    break;
                }
                if v == v {
                    fv = r as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        // VRAM estimate: input + first + out
        let bytes_in = cols * rows * core::mem::size_of::<f32>();
        let bytes_first = cols * core::mem::size_of::<i32>();
        let bytes_out = cols * rows * core::mem::size_of::<f32>();
        let required = bytes_in + bytes_first + bytes_out;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaIftRsiError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        let d_in = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("ift_rsi_many_series_one_param_f32")
            .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;

        // Each thread needs wp*sizeof(f32) shared memory; clamp block by SMEM and HW limits
        let bytes_per_thread = wma_p
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaIftRsiError::InvalidInput("wma_period too large".into()))?;
        let max_threads_by_smem = (self.max_smem_per_block / bytes_per_thread).max(1);
        let mut block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };
        block_x = block_x
            .min(max_threads_by_smem as u32)
            .min(self.max_threads_per_block)
            .max(1);

        let shmem_bytes_usize = (block_x as usize)
            .checked_mul(bytes_per_thread)
            .ok_or_else(|| CudaIftRsiError::InvalidInput("shared memory size overflow".into()))?;
        if shmem_bytes_usize > self.max_smem_per_block {
            return Err(CudaIftRsiError::InvalidInput(format!(
                "block_x={} with wma_period={} needs {}B shared memory; device allows {}B",
                block_x, wma_p, shmem_bytes_usize, self.max_smem_per_block
            )));
        }

        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shmem_bytes: u32 = shmem_bytes_usize as u32;
        unsafe { (*(self as *const _ as *mut CudaIftRsi)).last_many = Some(ManySeriesKernelSelected::OneD { block_x, shmem_bytes }); }
        unsafe {
            let mut in_ptr = d_in.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut rsi_i = rsi_p as i32;
            let mut wma_i = wma_p as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut rsi_i as *mut _ as *mut c_void,
                &mut wma_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shmem_bytes, args)
                .map_err(|e| CudaIftRsiError::Cuda(e.to_string()))?;
        }
        self.maybe_log_many_debug();
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() != Some("1") {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() != Some("1") {
            return;
        }
        if let Some(sel) = self.last_batch {
            eprintln!("[CudaIftRsi] batch kernel selected: {:?}", sel);
        }
        unsafe {
            (*(self as *const _ as *mut CudaIftRsi)).debug_batch_logged = true;
        }
        unsafe {
            (*(self as *const _ as *mut CudaIftRsi)).debug_batch_logged = true;
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() != Some("1") {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() != Some("1") {
            return;
        }
        if let Some(sel) = self.last_many {
            eprintln!("[CudaIftRsi] many-series kernel selected: {:?}", sel);
        }
        unsafe {
            (*(self as *const _ as *mut CudaIftRsi)).debug_many_logged = true;
        }
        unsafe {
            (*(self as *const _ as *mut CudaIftRsi)).debug_many_logged = true;
        }
    }
}

// ------------------------ Benches ------------------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::ift_rsi::IftRsiBatchRange;

    const ONE_SERIES_LEN: usize = 200_000;
    const PARAM_SWEEP: usize = 64;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * core::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * core::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct IftRsiBatchState {
        cuda: CudaIftRsi,
        data: Vec<f32>,
        sweep: IftRsiBatchRange,
    }
    impl CudaBenchState for IftRsiBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .ift_rsi_batch_dev(&self.data, &self.sweep)
                .expect("ift_rsi batch");
            // Explicit sync for deterministic benchmark timing since the API is non-blocking
            self.cuda.synchronize().expect("stream sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaIftRsi::new(0).expect("CudaIftRsi");
        let mut data = gen_series(ONE_SERIES_LEN);
        // NaNs at start for warmup semantics
        for i in 0..16 {
            data[i] = f32::NAN;
        }
        for i in 0..16 {
            data[i] = f32::NAN;
        }
        // Sweep rp in [5..(5+PARAM_SWEEP/2)], wp in [9..(9+PARAM_SWEEP/2)]
        let sweep = IftRsiBatchRange {
            rsi_period: (5, 5 + PARAM_SWEEP / 2 - 1, 1),
            wma_period: (9, 9 + PARAM_SWEEP / 2 - 1, 1),
        };
        let sweep = IftRsiBatchRange {
            rsi_period: (5, 5 + PARAM_SWEEP / 2 - 1, 1),
            wma_period: (9, 9 + PARAM_SWEEP / 2 - 1, 1),
        };
        Box::new(IftRsiBatchState { cuda, data, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "ift_rsi",
            "one_series_many_params",
            "ift_rsi_cuda_batch_dev",
            "200k_x_64",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
