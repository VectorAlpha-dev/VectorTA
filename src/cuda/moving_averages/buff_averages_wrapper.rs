//! CUDA scaffolding for Buff Averages kernels.
//!
//! Upgraded to follow the ALMA "gold standard" conventions:
//! - Policy-based kernel selection (plain vs tiled) with introspection
//! - VRAM-awareness (optional mem_get_info) and deterministic launches
//! - Bench profiles that preallocate and only launch kernels in the hot path
//! - Host-side prefix sums (default) with a device-prefix launch path

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{DeviceBuffer, mem_get_info};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaBuffAveragesError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaBuffAveragesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaBuffAveragesError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaBuffAveragesError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::{gen_time_major_prices, gen_time_major_volumes};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "buff_averages",
                "batch_dev",
                "buff_averages_cuda_batch_dev",
                "60k_x_49combos",
                prep_buff_averages_batch_box,
            )
            .with_inner_iters(8),
            CudaBenchScenario::new(
                "buff_averages",
                "many_series_one_param",
                "buff_averages_cuda_many_series_one_param",
                "250x1m",
                prep_buff_averages_many_series_box,
            )
            .with_inner_iters(4),
        ]
    }

    struct BuffAveragesBatchState {
        cuda: CudaBuffAverages,
        // Pre-staged device buffers
        d_prefix_pv: DeviceBuffer<f32>,
        d_prefix_vv: DeviceBuffer<f32>,
        d_fast: DeviceBuffer<i32>,
        d_slow: DeviceBuffer<i32>,
        d_fast_out: DeviceBuffer<f32>,
        d_slow_out: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
    }

    impl CudaBenchState for BuffAveragesBatchState {
        fn launch(&mut self) {
            self.cuda
                .buff_averages_batch_from_device_prefixes(
                    &self.d_prefix_pv,
                    &self.d_prefix_vv,
                    &self.d_fast,
                    &self.d_slow,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_fast_out,
                    &mut self.d_slow_out,
                )
                .expect("launch buff averages (device prefixes)");
            // Deterministic timing for benches
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_buff_averages_batch() -> BuffAveragesBatchState {
        let mut cuda = CudaBuffAverages::new(0).expect("cuda buff averages");
        cuda.set_policy(CudaBuffPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto });

        let len = 60_000usize;
        let mut price = vec![f32::NAN; len];
        let mut volume = vec![f32::NAN; len];
        for i in 3..len {
            let x = i as f32;
            price[i] = (x * 0.001).sin() + 0.0001 * x;
            volume[i] = (x * 0.0007).cos().abs() + 0.6;
        }
        let sweep = BuffAveragesBatchRange {
            fast_period: (4, 28, 4),
            slow_period: (32, 128, 16),
        };

        // Expand combos and build host prefix once
        let combos = CudaBuffAverages::expand_grid(&sweep);
        let (prefix_pv, prefix_vv) = CudaBuffAverages::build_prefix_sums(&price, &volume);
        let fast_periods: Vec<i32> = combos.iter().map(|&(f, _)| f as i32).collect();
        let slow_periods: Vec<i32> = combos.iter().map(|&(_, s)| s as i32).collect();
        let first_valid = price.iter().position(|v| !v.is_nan()).unwrap_or(0);

        // Upload device buffers once; preallocate outputs
        let d_prefix_pv = DeviceBuffer::from_slice(&prefix_pv).expect("d_prefix_pv");
        let d_prefix_vv = DeviceBuffer::from_slice(&prefix_vv).expect("d_prefix_vv");
        let d_fast = DeviceBuffer::from_slice(&fast_periods).expect("d_fast");
        let d_slow = DeviceBuffer::from_slice(&slow_periods).expect("d_slow");
        let elems = combos.len() * len;
        let d_fast_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_fast_out");
        let d_slow_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_slow_out");

        BuffAveragesBatchState {
            cuda,
            d_prefix_pv,
            d_prefix_vv,
            d_fast,
            d_slow,
            d_fast_out,
            d_slow_out,
            len,
            n_combos: combos.len(),
            first_valid,
        }
    }

    fn prep_buff_averages_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_buff_averages_batch())
    }

    struct BuffAveragesManySeriesState {
        cuda: CudaBuffAverages,
        d_pv_tm: DeviceBuffer<f32>,
        d_vv_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_fast_out_tm: DeviceBuffer<f32>,
        d_slow_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        fast: usize,
        slow: usize,
    }
    impl CudaBenchState for BuffAveragesManySeriesState {
        fn launch(&mut self) {
            self.cuda
                .buff_averages_many_series_one_param_device(
                    &self.d_pv_tm,
                    &self.d_vv_tm,
                    self.fast,
                    self.slow,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_fast_out_tm,
                    &mut self.d_slow_out_tm,
                )
                .expect("buff_averages many-series device-precomputed");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_buff_averages_many_series() -> BuffAveragesManySeriesState {
        let mut cuda = CudaBuffAverages::new(0).expect("cuda buff averages");
        cuda.set_policy(CudaBuffPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 4 } });

        let cols = 250usize;
        let rows = 1_000_000usize;
        let price_tm = gen_time_major_prices(cols, rows);
        let volume_tm = gen_time_major_volumes(cols, rows);
        let fast = 16usize;
        let slow = 64usize;

        // Host prep for prefixes and first_valids
        let prep = CudaBuffAverages::prepare_many_series_inputs(&price_tm, &volume_tm, cols, rows, fast, slow).expect("prep ms");
        let d_pv_tm = DeviceBuffer::from_slice(&prep.pv_prefix_tm).expect("d_pv_tm");
        let d_vv_tm = DeviceBuffer::from_slice(&prep.vv_prefix_tm).expect("d_vv_tm");
        let d_first_valids = DeviceBuffer::from_slice(&prep.first_valids).expect("d_first_valids");
        let d_fast_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_fast_out_tm");
        let d_slow_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_slow_out_tm");

        BuffAveragesManySeriesState { cuda, d_pv_tm, d_vv_tm, d_first_valids, d_fast_out_tm, d_slow_out_tm, cols, rows, fast, slow }
    }

    fn prep_buff_averages_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_buff_averages_many_series())
    }
}

impl std::error::Error for CudaBuffAveragesError {}

pub struct CudaBuffAverages {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaBuffPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy { Auto, Plain { block_x: u32 }, Tiled { tile: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy { Auto, OneD { block_x: u32 }, Tiled2D { tx: u32, ty: u32 } }

#[derive(Clone, Copy, Debug)]
pub struct CudaBuffPolicy { pub batch: BatchKernelPolicy, pub many_series: ManySeriesKernelPolicy }

impl Default for CudaBuffPolicy {
    fn default() -> Self { Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto } }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 }, Tiled1x { tile: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 }, Tiled2D { tx: u32, ty: u32 } }

impl CudaBuffAverages {
    pub fn new(device_id: usize) -> Result<Self, CudaBuffAveragesError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/buff_averages_kernel.ptx"));
        // Align with ALMA JIT policy: prefer DetermineTargetFromContext + O2, then progressively relax
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[])
                        .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaBuffPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with an explicit policy (mirrors ALMA convenience).
    pub fn new_with_policy(device_id: usize, policy: CudaBuffPolicy) -> Result<Self, CudaBuffAveragesError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaBuffPolicy) { self.policy = policy; }
    #[inline]
    pub fn policy(&self) -> &CudaBuffPolicy { &self.policy }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaBuffAveragesError> {
        self.stream.synchronize().map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] BUFF_AVG batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaBuffAverages)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] BUFF_AVG many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaBuffAverages)).debug_many_logged = true; }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && v.to_lowercase() != "false", Err(_) => true }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { required_bytes.saturating_add(headroom_bytes) <= free } else { true }
    }

    #[inline]
    fn pick_tiled_block(&self, series_len: usize) -> u32 {
        if let Ok(v) = std::env::var("BUFF_TILE") { if let Ok(tile) = v.parse::<u32>() { if tile == 128 || tile == 256 { return tile; } } }
        if series_len < 8192 { 128 } else { 256 }
    }

    pub fn expand_grid(range: &BuffAveragesBatchRange) -> Vec<(usize, usize)> {
        fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }

        let fasts = axis(range.fast_period);
        let slows = axis(range.slow_period);
        let mut combos = Vec::with_capacity(fasts.len() * slows.len());
        for &fast in &fasts {
            for &slow in &slows {
                combos.push((fast, slow));
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
    ) -> Result<(Vec<(usize, usize)>, usize, usize), CudaBuffAveragesError> {
        if price_f32.is_empty() {
            return Err(CudaBuffAveragesError::InvalidInput(
                "empty price data".into(),
            ));
        }
        if price_f32.len() != volume_f32.len() {
            return Err(CudaBuffAveragesError::InvalidInput(format!(
                "price/volume length mismatch ({} vs {})",
                price_f32.len(),
                volume_f32.len()
            )));
        }

        let len = price_f32.len();
        let first_valid = price_f32.iter().position(|v| !v.is_nan()).ok_or_else(|| {
            CudaBuffAveragesError::InvalidInput("all price values are NaN".into())
        })?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaBuffAveragesError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        for &(fast, slow) in &combos {
            if fast == 0 || slow == 0 {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "invalid periods (fast={}, slow={})",
                    fast, slow
                )));
            }
            if fast > len || slow > len {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "period exceeds length (len={}, fast={}, slow={})",
                    len, fast, slow
                )));
            }
            if len - first_valid < slow {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "not enough valid data for slow={} (valid after first={}): {}",
                    slow,
                    first_valid,
                    len - first_valid
                )));
            }
            if fast > slow {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "fast period {} must be <= slow period {}",
                    fast, slow
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

pub fn build_prefix_sums(price: &[f32], volume: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let len = price.len();
        let mut prefix_pv = vec![0.0f32; len + 1];
        let mut prefix_vv = vec![0.0f32; len + 1];
        let mut acc_pv = 0.0f64;
        let mut acc_vv = 0.0f64;
        for i in 0..len {
            let p = price[i];
            let v = volume[i];
            let (pv, vv) = if p.is_nan() || v.is_nan() {
                (0.0f64, 0.0f64)
            } else {
                let pf = p as f64;
                let vf = v as f64;
                (pf * vf, vf)
            };
            acc_pv += pv;
            acc_vv += vv;
            prefix_pv[i + 1] = acc_pv as f32;
            prefix_vv[i + 1] = acc_vv as f32;
        }
        (prefix_pv, prefix_vv)
}

    fn launch_batch_kernel(
        &self,
        d_prefix_pv: &DeviceBuffer<f32>,
        d_prefix_vv: &DeviceBuffer<f32>,
        d_fast: &DeviceBuffer<i32>,
        d_slow: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_fast_out: &mut DeviceBuffer<f32>,
        d_slow_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBuffAveragesError> {
        // Decide kernel per policy
        let mut use_tiled = len > 8192;
        let mut block_x: u32 = 256;
        let mut tile_choice: Option<u32> = None;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { block_x: bx } => { use_tiled = false; block_x = bx; }
            BatchKernelPolicy::Tiled { tile } => { use_tiled = true; tile_choice = Some(tile); }
        }

        if use_tiled {
            block_x = tile_choice.unwrap_or_else(|| self.pick_tiled_block(len));
            let func_name = match block_x { 128 => "buff_averages_batch_prefix_tiled_f32_tile128", _ => "buff_averages_batch_prefix_tiled_f32_tile256" };
            let func = self.module.get_function(func_name).or_else(|_| self.module.get_function("buff_averages_batch_prefix_f32")).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

            // Introspection
            unsafe { (*(self as *const _ as *mut CudaBuffAverages)).last_batch = Some(BatchKernelSelected::Tiled1x { tile: block_x }); }
            self.maybe_log_batch_debug();

            let grid_x = ((len as u32) + block_x - 1) / block_x;
            let block: BlockSize = (block_x, 1, 1).into();
            const MAX_GRID_Y: usize = 65_535;
            let mut start = 0usize;
            while start < n_combos {
                let chunk = (n_combos - start).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
                unsafe {
                    let mut prefix_pv_ptr = d_prefix_pv.as_device_ptr().as_raw();
                    let mut prefix_vv_ptr = d_prefix_vv.as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut first_valid_i = first_valid as i32;
                    let mut fast_ptr = d_fast.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut slow_ptr = d_slow.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut combos_i = chunk as i32;
                    let mut fast_out_ptr = d_fast_out.as_device_ptr().as_raw().wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let mut slow_out_ptr = d_slow_out.as_device_ptr().as_raw().wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let args: &mut [*mut c_void] = &mut [
                        &mut prefix_pv_ptr as *mut _ as *mut c_void,
                        &mut prefix_vv_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut fast_ptr as *mut _ as *mut c_void,
                        &mut slow_ptr as *mut _ as *mut c_void,
                        &mut combos_i as *mut _ as *mut c_void,
                        &mut fast_out_ptr as *mut _ as *mut c_void,
                        &mut slow_out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
                }
                start += chunk;
            }
        } else {
            let func = self.module.get_function("buff_averages_batch_prefix_f32").map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
            // Introspection
            unsafe { (*(self as *const _ as *mut CudaBuffAverages)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
            self.maybe_log_batch_debug();

            let grid_x = ((len as u32) + block_x - 1) / block_x;
            let block: BlockSize = (block_x, 1, 1).into();
            const MAX_GRID_Y: usize = 65_535;
            let mut start = 0usize;
            while start < n_combos {
                let chunk = (n_combos - start).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
                unsafe {
                    let mut prefix_pv_ptr = d_prefix_pv.as_device_ptr().as_raw();
                    let mut prefix_vv_ptr = d_prefix_vv.as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut first_valid_i = first_valid as i32;
                    let mut fast_ptr = d_fast.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut slow_ptr = d_slow.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut combos_i = chunk as i32;
                    let mut fast_out_ptr = d_fast_out.as_device_ptr().as_raw().wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let mut slow_out_ptr = d_slow_out.as_device_ptr().as_raw().wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let args: &mut [*mut c_void] = &mut [
                        &mut prefix_pv_ptr as *mut _ as *mut c_void,
                        &mut prefix_vv_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut fast_ptr as *mut _ as *mut c_void,
                        &mut slow_ptr as *mut _ as *mut c_void,
                        &mut combos_i as *mut _ as *mut c_void,
                        &mut fast_out_ptr as *mut _ as *mut c_void,
                        &mut slow_out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
                }
                start += chunk;
            }
        }

        Ok(())
    }

    // ---------------- Many-series host-side helpers ----------------
    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_period: usize,
        slow_period: usize,
    ) -> Result<PreparedManySeries, CudaBuffAveragesError> {
        if prices_tm_f32.len() != volumes_tm_f32.len() {
            return Err(CudaBuffAveragesError::InvalidInput("price/volume matrix length mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaBuffAveragesError::InvalidInput("matrix dims must be positive".into()));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaBuffAveragesError::InvalidInput("matrix shape mismatch".into()));
        }
        if fast_period == 0 || slow_period == 0 {
            return Err(CudaBuffAveragesError::InvalidInput("periods must be positive".into()));
        }
        if fast_period > slow_period {
            return Err(CudaBuffAveragesError::InvalidInput("fast_period must be <= slow_period".into()));
        }

        // Find first valid row per series where both price & volume are finite
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let p = prices_tm_f32[t * cols + s];
                let v = volumes_tm_f32[t * cols + s];
                if !p.is_nan() && !v.is_nan() { fv = Some(t); break; }
            }
            let val = fv.ok_or_else(|| CudaBuffAveragesError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - val < slow_period {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    s, slow_period, rows - val
                )));
            }
            first_valids[s] = val as i32;
        }

        let (pv_prefix_tm, vv_prefix_tm) = build_prefix_sums_time_major(
            prices_tm_f32,
            volumes_tm_f32,
            cols,
            rows,
            &first_valids,
        );
        Ok(PreparedManySeries { first_valids, pv_prefix_tm, vv_prefix_tm })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_pv_prefix_tm: &DeviceBuffer<f32>,
        d_vv_prefix_tm: &DeviceBuffer<f32>,
        fast_period: usize,
        slow_period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_fast_out_tm: &mut DeviceBuffer<f32>,
        d_slow_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBuffAveragesError> {
        if num_series == 0 || series_len == 0 { return Ok(()); }
        if fast_period == 0 || slow_period == 0 {
            return Err(CudaBuffAveragesError::InvalidInput("periods must be positive".into()));
        }

        // Prefer 2D tiles when available
        let try_2d = |tx: u32, ty: u32| -> Option<()> {
            let fname = match (tx, ty) {
                (128, 4) => "buff_averages_many_series_one_param_tiled2d_f32_tx128_ty4",
                (128, 2) => "buff_averages_many_series_one_param_tiled2d_f32_tx128_ty2",
                _ => return None,
            };
            let func = match self.module.get_function(fname) { Ok(f) => f, Err(_) => return None };
            let grid_x = ((series_len as u32) + tx - 1) / tx;
            let grid_y = ((num_series as u32) + ty - 1) / ty;
            let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
            let block: BlockSize = (tx, ty, 1).into();
            unsafe {
                let mut pv_ptr = d_pv_prefix_tm.as_device_ptr().as_raw();
                let mut vv_ptr = d_vv_prefix_tm.as_device_ptr().as_raw();
                let mut f = fast_period as i32;
                let mut s = slow_period as i32;
                let mut cols_i = num_series as i32;
                let mut rows_i = series_len as i32;
                let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut outf_ptr = d_fast_out_tm.as_device_ptr().as_raw();
                let mut outs_ptr = d_slow_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut pv_ptr as *mut _ as *mut c_void,
                    &mut vv_ptr as *mut _ as *mut c_void,
                    &mut f as *mut _ as *mut c_void,
                    &mut s as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut fv_ptr as *mut _ as *mut c_void,
                    &mut outf_ptr as *mut _ as *mut c_void,
                    &mut outs_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))
                    .ok()?;
            }
            unsafe { (*(self as *const _ as *mut CudaBuffAverages)).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty }); }
            self.maybe_log_many_debug();
            Some(())
        };

        match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                if try_2d(tx, ty).is_some() { return Ok(()); }
            }
            ManySeriesKernelPolicy::Auto => {
                if num_series >= 128 {
                    if try_2d(128, 4).is_some() { return Ok(()); }
                    if try_2d(128, 2).is_some() { return Ok(()); }
                } else {
                    if try_2d(128, 2).is_some() { return Ok(()); }
                    if try_2d(128, 4).is_some() { return Ok(()); }
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => {}
        }

        // Fallback 1D
        let func = self.module
            .get_function("buff_averages_many_series_one_param_f32")
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => block_x, _ => 128 };
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut pv_ptr = d_pv_prefix_tm.as_device_ptr().as_raw();
            let mut vv_ptr = d_vv_prefix_tm.as_device_ptr().as_raw();
            let mut f = fast_period as i32;
            let mut s = slow_period as i32;
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut outf_ptr = d_fast_out_tm.as_device_ptr().as_raw();
            let mut outs_ptr = d_slow_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut vv_ptr as *mut _ as *mut c_void,
                &mut f as *mut _ as *mut c_void,
                &mut s as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut outf_ptr as *mut _ as *mut c_void,
                &mut outs_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        }
        unsafe { (*(self as *const _ as *mut CudaBuffAverages)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        combos: &[(usize, usize)],
        first_valid: usize,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let len = price_f32.len();
        let (prefix_pv, prefix_vv) = Self::build_prefix_sums(price_f32, volume_f32);

        // Optional VRAM check (rough estimate). Headroom default: 64MB.
        let rows = combos.len();
        let bytes_required = (len + 1) * 4 * 2  // prefixes
            + rows * 4 * 2                      // period arrays
            + rows * len * 4 * 2;               // outputs
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes_required, headroom) {
            return Err(CudaBuffAveragesError::InvalidInput(format!(
                "insufficient VRAM: need ~{} MB (incl headroom)", (bytes_required + headroom) / (1024*1024)
            )));
        }

        let d_prefix_pv = DeviceBuffer::from_slice(&prefix_pv).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_prefix_vv = DeviceBuffer::from_slice(&prefix_vv).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        let fast_periods: Vec<i32> = combos.iter().map(|&(f, _)| f as i32).collect();
        let slow_periods: Vec<i32> = combos.iter().map(|&(_, s)| s as i32).collect();
        let d_fast = DeviceBuffer::from_slice(&fast_periods).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_slow = DeviceBuffer::from_slice(&slow_periods).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_fast_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let mut d_slow_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prefix_pv,
            &d_prefix_vv,
            &d_fast,
            &d_slow,
            len,
            first_valid,
            combos.len(),
            &mut d_fast_out,
            &mut d_slow_out,
        )?;

        self.stream.synchronize().map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_fast_out,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_slow_out,
                rows: combos.len(),
                cols: len,
            },
        ))
    }

    /// Launch using pre-staged device buffers (prefix arrays, period arrays, outputs).
    /// Does not allocate or copy; intended for deterministic benches and integrations.
    pub fn buff_averages_batch_from_device_prefixes(
        &self,
        d_prefix_pv: &DeviceBuffer<f32>,
        d_prefix_vv: &DeviceBuffer<f32>,
        d_fast: &DeviceBuffer<i32>,
        d_slow: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_fast_out: &mut DeviceBuffer<f32>,
        d_slow_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBuffAveragesError> {
        self.launch_batch_kernel(
            d_prefix_pv,
            d_prefix_vv,
            d_fast,
            d_slow,
            len,
            first_valid,
            n_combos,
            d_fast_out,
            d_slow_out,
        )
    }

    pub fn buff_averages_batch_dev(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let (combos, first_valid, _len) = Self::prepare_batch_inputs(price_f32, volume_f32, sweep)?;
        self.run_batch_kernel(price_f32, volume_f32, &combos, first_valid)
    }

    /// Expansion-based (two-float) variant: builds PV/VV prefix sums as (hi, lo) and
    /// launches the corresponding FP32-only kernels. Improves numeric accuracy without FP64.
    pub fn buff_averages_batch_dev_exp2(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(price_f32, volume_f32, sweep)?;
        // Build expansion prefixes on host
        let (pv_hi, pv_lo, vv_hi, vv_lo) = build_prefix_sums_exp2(price_f32, volume_f32);

        // VRAM estimate
        let rows = combos.len();
        let bytes_required = (len + 1) * 4 * 4  // four prefix arrays
            + rows * 4 * 2                      // period arrays
            + rows * len * 4 * 2;               // outputs
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes_required, headroom) {
            return Err(CudaBuffAveragesError::InvalidInput(format!(
                "insufficient VRAM: need ~{} MB (incl headroom)", (bytes_required + headroom) / (1024*1024)
            )));
        }

        // Upload device buffers
        let d_pv_hi = DeviceBuffer::from_slice(&pv_hi).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_pv_lo = DeviceBuffer::from_slice(&pv_lo).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_vv_hi = DeviceBuffer::from_slice(&vv_hi).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_vv_lo = DeviceBuffer::from_slice(&vv_lo).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let fast_periods: Vec<i32> = combos.iter().map(|&(f, _)| f as i32).collect();
        let slow_periods: Vec<i32> = combos.iter().map(|&(_, s)| s as i32).collect();
        let d_fast = DeviceBuffer::from_slice(&fast_periods).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_slow = DeviceBuffer::from_slice(&slow_periods).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        // Outputs
        let elems = combos.len() * len;
        let mut d_fast_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let mut d_slow_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        // Launch plain 1D expansion kernel
        let func = self.module
            .get_function("buff_averages_batch_prefix_exp2_f32")
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let block_x: u32 = 256;
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();
        const MAX_GRID_Y: usize = 65_535;
        let mut start = 0usize;
        while start < combos.len() {
            let chunk = (combos.len() - start).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            unsafe {
                let mut pv_hi_ptr = d_pv_hi.as_device_ptr().as_raw();
                let mut pv_lo_ptr = d_pv_lo.as_device_ptr().as_raw();
                let mut vv_hi_ptr = d_vv_hi.as_device_ptr().as_raw();
                let mut vv_lo_ptr = d_vv_lo.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut fast_ptr = d_fast.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                let mut slow_ptr = d_slow.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                let mut combos_i = chunk as i32;
                let mut fast_out_ptr = d_fast_out.as_device_ptr().as_raw().wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                let mut slow_out_ptr = d_slow_out.as_device_ptr().as_raw().wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut pv_hi_ptr as *mut _ as *mut c_void,
                    &mut pv_lo_ptr as *mut _ as *mut c_void,
                    &mut vv_hi_ptr as *mut _ as *mut c_void,
                    &mut vv_lo_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut fast_out_ptr as *mut _ as *mut c_void,
                    &mut slow_out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
            }
            start += chunk;
        }
        self.stream.synchronize().map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 { buf: d_fast_out, rows: combos.len(), cols: len },
            DeviceArrayF32 { buf: d_slow_out, rows: combos.len(), cols: len },
        ))
    }

    /// Many-series × one-parameter (time-major). Returns VRAM-backed (fast, slow).
    pub fn buff_averages_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_period: usize,
        slow_period: usize,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let prep = Self::prepare_many_series_inputs(prices_tm_f32, volumes_tm_f32, cols, rows, fast_period, slow_period)?;

        // VRAM estimate
        let elems = cols * rows;
        let required = ((rows + 1) * cols * 2 + elems * 2) * std::mem::size_of::<f32>() + cols * std::mem::size_of::<i32>();
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(required, headroom) {
            return Err(CudaBuffAveragesError::InvalidInput("insufficient VRAM for many-series run".into()));
        }

        let d_pv = DeviceBuffer::from_slice(&prep.pv_prefix_tm).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_vv = DeviceBuffer::from_slice(&prep.vv_prefix_tm).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&prep.first_valids).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let mut d_fast_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let mut d_slow_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_pv,
            &d_vv,
            fast_period,
            slow_period,
            cols,
            rows,
            &d_fv,
            &mut d_fast_out,
            &mut d_slow_out,
        )?;

        self.stream.synchronize().map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        Ok((
            DeviceArrayF32 { buf: d_fast_out, rows, cols },
            DeviceArrayF32 { buf: d_slow_out, rows, cols },
        ))
    }

    /// Expansion-based many-series (time-major) launcher.
    pub fn buff_averages_many_series_one_param_dev_exp2(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_period: usize,
        slow_period: usize,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let prep = Self::prepare_many_series_inputs(prices_tm_f32, volumes_tm_f32, cols, rows, fast_period, slow_period)?;
        let (pv_hi_tm, pv_lo_tm, vv_hi_tm, vv_lo_tm) = build_prefix_sums_time_major_exp2(
            prices_tm_f32, volumes_tm_f32, cols, rows, &prep.first_valids,
        );

        // VRAM check (four prefix arrays)
        let elems = cols * rows;
        let required = ((rows + 1) * cols * 4 + elems * 2) * std::mem::size_of::<f32>() + cols * std::mem::size_of::<i32>();
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(required, headroom) { return Err(CudaBuffAveragesError::InvalidInput("insufficient VRAM for many-series run (exp2)".into())); }

        let d_pv_hi = DeviceBuffer::from_slice(&pv_hi_tm).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_pv_lo = DeviceBuffer::from_slice(&pv_lo_tm).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_vv_hi = DeviceBuffer::from_slice(&vv_hi_tm).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_vv_lo = DeviceBuffer::from_slice(&vv_lo_tm).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&prep.first_valids).map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let mut d_fast_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let mut d_slow_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        let func = self.module
            .get_function("buff_averages_many_series_one_param_exp2_f32")
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        let block_x: u32 = 128;
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut pv_hi_ptr = d_pv_hi.as_device_ptr().as_raw();
            let mut pv_lo_ptr = d_pv_lo.as_device_ptr().as_raw();
            let mut vv_hi_ptr = d_vv_hi.as_device_ptr().as_raw();
            let mut vv_lo_ptr = d_vv_lo.as_device_ptr().as_raw();
            let mut f = fast_period as i32;
            let mut s = slow_period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut outf_ptr = d_fast_out.as_device_ptr().as_raw();
            let mut outs_ptr = d_slow_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pv_hi_ptr as *mut _ as *mut c_void,
                &mut pv_lo_ptr as *mut _ as *mut c_void,
                &mut vv_hi_ptr as *mut _ as *mut c_void,
                &mut vv_lo_ptr as *mut _ as *mut c_void,
                &mut f as *mut _ as *mut c_void,
                &mut s as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut outf_ptr as *mut _ as *mut c_void,
                &mut outs_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        }
        self.stream.synchronize().map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        Ok((
            DeviceArrayF32 { buf: d_fast_out, rows, cols },
            DeviceArrayF32 { buf: d_slow_out, rows, cols },
        ))
    }

    /// Many-series device-precomputed variant (time-major prefixes on device).
    #[allow(clippy::too_many_arguments)]
    pub fn buff_averages_many_series_one_param_device(
        &self,
        d_pv_prefix_tm: &DeviceBuffer<f32>,
        d_vv_prefix_tm: &DeviceBuffer<f32>,
        fast_period: usize,
        slow_period: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_fast_out_tm: &mut DeviceBuffer<f32>,
        d_slow_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBuffAveragesError> {
        self.launch_many_series_kernel(
            d_pv_prefix_tm,
            d_vv_prefix_tm,
            fast_period,
            slow_period,
            cols,
            rows,
            d_first_valids,
            d_fast_out_tm,
            d_slow_out_tm,
        )
    }

    pub fn buff_averages_batch_into_host_f32(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
        fast_out: &mut [f32],
        slow_out: &mut [f32],
    ) -> Result<(usize, usize, Vec<(usize, usize)>), CudaBuffAveragesError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(price_f32, volume_f32, sweep)?;
        let expected = combos.len() * len;
        if fast_out.len() != expected || slow_out.len() != expected {
            return Err(CudaBuffAveragesError::InvalidInput(format!(
                "output slice mismatch (expected {}, fast={}, slow={})",
                expected,
                fast_out.len(),
                slow_out.len()
            )));
        }

        let (fast_dev, slow_dev) =
            self.run_batch_kernel(price_f32, volume_f32, &combos, first_valid)?;
        fast_dev
            .buf
            .copy_to(fast_out)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;
        slow_dev
            .buf
            .copy_to(slow_out)
            .map_err(|e| CudaBuffAveragesError::Cuda(e.to_string()))?;

        Ok((combos.len(), len, combos))
    }
}

// -------- Many-series structs + prefix builder --------

struct PreparedManySeries {
    first_valids: Vec<i32>,
    pv_prefix_tm: Vec<f32>,
    vv_prefix_tm: Vec<f32>,
}

fn build_prefix_sums_time_major(
    prices_tm: &[f32],
    volumes_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    // (rows+1) x cols layout so index (t+1, s) exists without bounds checks
    let mut pv_prefix = vec![0.0f32; (rows + 1) * cols];
    let mut vv_prefix = vec![0.0f32; (rows + 1) * cols];
    for s in 0..cols {
        let fv = first_valids[s].max(0) as usize;
        let mut acc_pv = 0.0f64;
        let mut acc_vv = 0.0f64;
        for t in 0..rows {
            if t >= fv {
                let idx = t * cols + s;
                let p = prices_tm[idx];
                let v = volumes_tm[idx];
                if !(p.is_nan() || v.is_nan()) {
                    acc_pv += (p as f64) * (v as f64);
                    acc_vv += (v as f64);
                }
            }
            let widx = (t + 1) * cols + s;
            pv_prefix[widx] = acc_pv as f32;
            vv_prefix[widx] = acc_vv as f32;
        }
    }
    (pv_prefix, vv_prefix)
}

// -------- Expansion prefix builders (host, FP32 EFT) --------

fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bp = s - a;
    let e = (a - (s - bp)) + (b - bp);
    (s, e)
}

#[inline]
fn prefix_step_f2(x: f32, hi: &mut f32, lo: &mut f32) {
    let (s_hi, s_lo) = two_sum_f32(*hi, x);
    let (r_hi, r_lo) = two_sum_f32(s_hi, s_lo + *lo);
    *hi = r_hi;
    *lo = r_lo;
}

pub fn build_prefix_sums_exp2(price: &[f32], volume: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = price.len();
    let mut pv_hi = vec![0.0f32; len + 1];
    let mut pv_lo = vec![0.0f32; len + 1];
    let mut vv_hi = vec![0.0f32; len + 1];
    let mut vv_lo = vec![0.0f32; len + 1];
    let mut sh = 0.0f32; let mut sl = 0.0f32;
    let mut th = 0.0f32; let mut tl = 0.0f32;
    pv_hi[0] = 0.0; pv_lo[0] = 0.0; vv_hi[0] = 0.0; vv_lo[0] = 0.0;
    for i in 0..len {
        let p = price[i];
        let v = volume[i];
        let v_ok = v.is_finite();
        let p_ok = p.is_finite();
        let vol = if v_ok { v } else { 0.0 };
        let pv = if v_ok && p_ok { p.mul_add(v, 0.0) } else { 0.0 };
        prefix_step_f2(pv, &mut sh, &mut sl);
        prefix_step_f2(vol, &mut th, &mut tl);
        pv_hi[i + 1] = sh; pv_lo[i + 1] = sl;
        vv_hi[i + 1] = th; vv_lo[i + 1] = tl;
    }
    (pv_hi, pv_lo, vv_hi, vv_lo)
}

fn build_prefix_sums_time_major_exp2(
    prices_tm: &[f32],
    volumes_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut pv_hi = vec![0.0f32; (rows + 1) * cols];
    let mut pv_lo = vec![0.0f32; (rows + 1) * cols];
    let mut vv_hi = vec![0.0f32; (rows + 1) * cols];
    let mut vv_lo = vec![0.0f32; (rows + 1) * cols];
    for s in 0..cols {
        let fv = first_valids[s].max(0) as usize;
        let mut sh = 0.0f32; let mut sl = 0.0f32;
        let mut th = 0.0f32; let mut tl = 0.0f32;
        pv_hi[0 * cols + s] = 0.0; pv_lo[0 * cols + s] = 0.0;
        vv_hi[0 * cols + s] = 0.0; vv_lo[0 * cols + s] = 0.0;
        for t in 0..rows {
            if t >= fv {
                let idx = t * cols + s;
                let p = prices_tm[idx];
                let v = volumes_tm[idx];
                let v_ok = v.is_finite();
                let p_ok = p.is_finite();
                let vol = if v_ok { v } else { 0.0 };
                let pv = if v_ok && p_ok { p.mul_add(v, 0.0) } else { 0.0 };
                prefix_step_f2(pv, &mut sh, &mut sl);
                prefix_step_f2(vol, &mut th, &mut tl);
            }
            let w = (t + 1) * cols + s;
            pv_hi[w] = sh; pv_lo[w] = sl;
            vv_hi[w] = th; vv_lo[w] = tl;
        }
    }
    (pv_hi, pv_lo, vv_hi, vv_lo)
}
