#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange;
use cust::context::CacheConfig;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaBuffAveragesError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("out of memory on device: required ≈{required} bytes (incl headroom {headroom}), free={free}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("arithmetic overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device/context mismatch: buffer on device {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_time_major_prices, gen_time_major_volumes};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

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

            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_buff_averages_batch() -> BuffAveragesBatchState {
        let mut cuda = CudaBuffAverages::new(0).expect("cuda buff averages");
        cuda.set_policy(CudaBuffPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });

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

        let combos = CudaBuffAverages::expand_grid(&sweep);
        let (prefix_pv, prefix_vv) = CudaBuffAverages::build_prefix_sums(&price, &volume);
        let fast_periods: Vec<i32> = combos.iter().map(|&(f, _)| f as i32).collect();
        let slow_periods: Vec<i32> = combos.iter().map(|&(_, s)| s as i32).collect();
        let first_valid = price.iter().position(|v| !v.is_nan()).unwrap_or(0);

        let d_prefix_pv = DeviceBuffer::from_slice(&prefix_pv).expect("d_prefix_pv");
        let d_prefix_vv = DeviceBuffer::from_slice(&prefix_vv).expect("d_prefix_vv");
        let d_fast = DeviceBuffer::from_slice(&fast_periods).expect("d_fast");
        let d_slow = DeviceBuffer::from_slice(&slow_periods).expect("d_slow");
        let elems = combos.len() * len;
        let d_fast_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_fast_out");
        let d_slow_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_slow_out");

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
        cuda.set_policy(CudaBuffPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 4 },
        });

        let cols = 250usize;
        let rows = 1_000_000usize;
        let price_tm = gen_time_major_prices(cols, rows);
        let volume_tm = gen_time_major_volumes(cols, rows);
        let fast = 16usize;
        let slow = 64usize;

        let prep = CudaBuffAverages::prepare_many_series_inputs(
            &price_tm, &volume_tm, cols, rows, fast, slow,
        )
        .expect("prep ms");
        let d_pv_tm = DeviceBuffer::from_slice(&prep.pv_prefix_tm).expect("d_pv_tm");
        let d_vv_tm = DeviceBuffer::from_slice(&prep.vv_prefix_tm).expect("d_vv_tm");
        let d_first_valids = DeviceBuffer::from_slice(&prep.first_valids).expect("d_first_valids");
        let d_fast_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_fast_out_tm");
        let d_slow_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_slow_out_tm");

        BuffAveragesManySeriesState {
            cuda,
            d_pv_tm,
            d_vv_tm,
            d_first_valids,
            d_fast_out_tm,
            d_slow_out_tm,
            cols,
            rows,
            fast,
            slow,
        }
    }

    fn prep_buff_averages_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_buff_averages_many_series())
    }
}

pub struct CudaBuffAverages {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaBuffPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaBuffPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaBuffPolicy {
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
    Tiled1x { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

impl CudaBuffAverages {
    pub fn new(device_id: usize) -> Result<Self, CudaBuffAveragesError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/buff_averages_kernel.ptx"));

        let mut jit_vec: Vec<ModuleJitOption> = vec![
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        if let Ok(rs) = std::env::var("BUFF_MAXREG") {
            if let Ok(v) = rs.parse::<u32>() {
                jit_vec.push(ModuleJitOption::MaxRegisters(v));
            }
        }
        let module = match Module::from_ptx(ptx, &jit_vec) {
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
            policy: CudaBuffPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaBuffPolicy,
    ) -> Result<Self, CudaBuffAveragesError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaBuffPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaBuffPolicy {
        &self.policy
    }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaBuffAveragesError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaBuffAveragesError::Cuda(e))
    }

    #[inline]
    fn prefer_l1(&self, func: &mut cust::function::Function) {
        let _ = func.set_cache_config(CacheConfig::PreferL1);
    }

    unsafe fn set_l2_persist_window(&self, base: u64, bytes: usize, hit_ratio: f32) {
        if bytes == 0 {
            return;
        }

        let device = Device::get_device(self.device_id).ok();
        let mut max_win: i32 = 0;
        if let Some(dev) = device {
            let _ = cu::cuDeviceGetAttribute(
                &mut max_win as *mut _,
                cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                dev.as_raw(),
            );
        }
        let win_bytes = if max_win > 0 {
            bytes.min(max_win as usize)
        } else {
            bytes
        };

        let mut max_persist: i32 = 0;
        if let Some(dev) = device {
            let _ = cu::cuDeviceGetAttribute(
                &mut max_persist as *mut _,
                cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
                dev.as_raw(),
            );
        }
        if max_persist > 0 {
            let want = (win_bytes as u64).min(
                std::env::var("BUFF_APW_SETASIDE_BYTES")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(win_bytes as u64),
            );
            let _ = cu::cuCtxSetLimit(
                cu::CUlimit_enum::CU_LIMIT_PERSISTING_L2_CACHE_SIZE,
                want.min(max_persist as u64) as usize,
            );
        }

        let mut val: cu::CUstreamAttrValue = std::mem::zeroed();
        let apw = cu::CUaccessPolicyWindow_v1 {
            base_ptr: base as *mut std::ffi::c_void,
            num_bytes: win_bytes,
            hitRatio: std::env::var("BUFF_APW_RATIO")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(hit_ratio),
            hitProp: cu::CUaccessProperty_enum::CU_ACCESS_PROPERTY_PERSISTING,
            missProp: cu::CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL,
        };

        *(&mut val.accessPolicyWindow) = apw;

        let _ = cu::cuStreamSetAttribute(
            self.stream.as_inner(),
            cu::CUstreamAttrID_enum::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &val,
        );
    }

    #[inline]
    unsafe fn set_l2_window_for_pair(
        &self,
        a_ptr: u64,
        a_bytes: usize,
        b_ptr: u64,
        b_bytes: usize,
    ) {
        if std::env::var("BUFF_APW").ok().as_deref() == Some("0") {
            return;
        }
        let start = a_ptr.min(b_ptr);
        let end = (a_ptr + a_bytes as u64).max(b_ptr + b_bytes as u64);
        let span = (end - start) as usize;

        let device = Device::get_device(self.device_id).ok();
        let mut max_win: i32 = 0;
        if let Some(dev) = device {
            let _ = cu::cuDeviceGetAttribute(
                &mut max_win as *mut _,
                cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                dev.as_raw(),
            );
        }
        if max_win > 0 && span > max_win as usize {
            if a_bytes >= b_bytes {
                self.set_l2_persist_window(a_ptr, a_bytes, 0.70);
            } else {
                self.set_l2_persist_window(b_ptr, b_bytes, 0.70);
            }
        } else {
            self.set_l2_persist_window(start, span, 0.70);
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
                    eprintln!("[DEBUG] BUFF_AVG batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBuffAverages)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] BUFF_AVG many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBuffAverages)).debug_many_logged = true;
                }
            }
        }
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
    fn will_fit_checked(
        required_bytes: usize,
        headroom_bytes: usize,
    ) -> Result<(), CudaBuffAveragesError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        let sum = required_bytes.checked_add(headroom_bytes).ok_or(
            CudaBuffAveragesError::ArithmeticOverflow {
                context: "required_bytes + headroom",
            },
        )?;
        if let Some((free, _)) = Self::device_mem_info() {
            if sum <= free {
                Ok(())
            } else {
                Err(CudaBuffAveragesError::OutOfMemory {
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
    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaBuffAveragesError> {
        unsafe {
            let dev = Device::get_device(self.device_id).ok();
            let mut max_threads_per_block: i32 = 1024;
            if let Some(d) = dev {
                let _ = cu::cuDeviceGetAttribute(
                    &mut max_threads_per_block as *mut _,
                    cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    d.as_raw(),
                );
            }
            let (bx, by, bz) = (block.x, block.y, block.z);
            let threads = (bx as u64) * (by as u64) * (bz as u64);
            if threads > (max_threads_per_block as u64) {
                return Err(CudaBuffAveragesError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx,
                    by,
                    bz,
                });
            }
        }
        Ok(())
    }

    #[inline]
    fn pick_tiled_block(&self, series_len: usize) -> u32 {
        if let Ok(v) = std::env::var("BUFF_TILE") {
            if let Ok(tile) = v.parse::<u32>() {
                if tile == 128 || tile == 256 {
                    return tile;
                }
            }
        }
        if series_len < 8192 {
            128
        } else {
            256
        }
    }

    pub fn expand_grid(range: &BuffAveragesBatchRange) -> Vec<(usize, usize)> {
        fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            let (lo, hi) = if start <= end {
                (start, end)
            } else {
                (end, start)
            };
            (lo..=hi).step_by(step).collect()
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
        let mut use_tiled = len > 8192;
        let mut block_x: u32 = 256;
        let mut tile_choice: Option<u32> = None;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { block_x: bx } => {
                use_tiled = false;
                block_x = bx;
            }
            BatchKernelPolicy::Tiled { tile } => {
                use_tiled = true;
                tile_choice = Some(tile);
            }
        }

        unsafe {
            let pv_ptr = d_prefix_pv.as_device_ptr().as_raw();
            let vv_ptr = d_prefix_vv.as_device_ptr().as_raw();
            let pv_bytes = (len + 1) * std::mem::size_of::<f32>();
            let vv_bytes = (len + 1) * std::mem::size_of::<f32>();
            self.set_l2_window_for_pair(pv_ptr, pv_bytes, vv_ptr, vv_bytes);
        }

        if use_tiled {
            block_x = tile_choice.unwrap_or_else(|| self.pick_tiled_block(len));
            let func_name = match block_x {
                128 => "buff_averages_batch_prefix_tiled_f32_tile128",
                _ => "buff_averages_batch_prefix_tiled_f32_tile256",
            };
            let mut func = match self.module.get_function(func_name) {
                Ok(f) => f,
                Err(_) => self
                    .module
                    .get_function("buff_averages_batch_prefix_f32")
                    .map_err(|_| CudaBuffAveragesError::MissingKernelSymbol { name: func_name })?,
            };
            self.prefer_l1(&mut func);

            unsafe {
                (*(self as *const _ as *mut CudaBuffAverages)).last_batch =
                    Some(BatchKernelSelected::Tiled1x { tile: block_x });
            }
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
                    let mut fast_ptr = d_fast
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut slow_ptr = d_slow
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut combos_i = chunk as i32;
                    let mut fast_out_ptr = d_fast_out
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let mut slow_out_ptr = d_slow_out
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
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
                    self.validate_launch(grid, block)?;
                    self.stream.launch(&func, grid, block, 0, args)?;
                }
                start += chunk;
            }
        } else {
            let mut func = self
                .module
                .get_function("buff_averages_batch_prefix_f32")
                .map_err(|_| CudaBuffAveragesError::MissingKernelSymbol {
                    name: "buff_averages_batch_prefix_f32",
                })?;
            self.prefer_l1(&mut func);

            unsafe {
                (*(self as *const _ as *mut CudaBuffAverages)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
            }
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
                    let mut fast_ptr = d_fast
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut slow_ptr = d_slow
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut combos_i = chunk as i32;
                    let mut fast_out_ptr = d_fast_out
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let mut slow_out_ptr = d_slow_out
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
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
                    self.validate_launch(grid, block)?;
                    self.stream.launch(&func, grid, block, 0, args)?;
                }
                start += chunk;
            }
        }

        Ok(())
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_period: usize,
        slow_period: usize,
    ) -> Result<PreparedManySeries, CudaBuffAveragesError> {
        if prices_tm_f32.len() != volumes_tm_f32.len() {
            return Err(CudaBuffAveragesError::InvalidInput(
                "price/volume matrix length mismatch".into(),
            ));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaBuffAveragesError::InvalidInput(
                "matrix dims must be positive".into(),
            ));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaBuffAveragesError::InvalidInput(
                "matrix shape mismatch".into(),
            ));
        }
        if fast_period == 0 || slow_period == 0 {
            return Err(CudaBuffAveragesError::InvalidInput(
                "periods must be positive".into(),
            ));
        }
        if fast_period > slow_period {
            return Err(CudaBuffAveragesError::InvalidInput(
                "fast_period must be <= slow_period".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let p = prices_tm_f32[t * cols + s];
                let v = volumes_tm_f32[t * cols + s];
                if !p.is_nan() && !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let val = fv.ok_or_else(|| {
                CudaBuffAveragesError::InvalidInput(format!("series {} all NaN", s))
            })?;
            if rows - val < slow_period {
                return Err(CudaBuffAveragesError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    s,
                    slow_period,
                    rows - val
                )));
            }
            first_valids[s] = val as i32;
        }

        let (pv_prefix_tm, vv_prefix_tm) =
            build_prefix_sums_time_major(prices_tm_f32, volumes_tm_f32, cols, rows, &first_valids);
        Ok(PreparedManySeries {
            first_valids,
            pv_prefix_tm,
            vv_prefix_tm,
        })
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
        if num_series == 0 || series_len == 0 {
            return Ok(());
        }
        if fast_period == 0 || slow_period == 0 {
            return Err(CudaBuffAveragesError::InvalidInput(
                "periods must be positive".into(),
            ));
        }

        unsafe {
            let pv_ptr = d_pv_prefix_tm.as_device_ptr().as_raw();
            let vv_ptr = d_vv_prefix_tm.as_device_ptr().as_raw();
            let bytes = (series_len + 1) * num_series * std::mem::size_of::<f32>();
            self.set_l2_window_for_pair(pv_ptr, bytes, vv_ptr, bytes);
        }

        let try_2d = |tx: u32, ty: u32| -> Option<()> {
            let fname_candidates: &[&str] = match (tx, ty) {
                (128, 4) => &[
                    "buff_averages_many_series_one_param_tiled2d_f32_sx128_ty4",
                    "buff_averages_many_series_one_param_tiled2d_f32_tx128_ty4",
                ],
                (128, 2) => &[
                    "buff_averages_many_series_one_param_tiled2d_f32_sx128_ty2",
                    "buff_averages_many_series_one_param_tiled2d_f32_tx128_ty2",
                ],
                (128, 1) => &["buff_averages_many_series_one_param_tiled2d_f32_sx128_ty1"],
                _ => return None,
            };
            let (mut func, picked): (cust::function::Function, &str) = {
                let mut sel: Option<(cust::function::Function, &str)> = None;
                for &name in fname_candidates {
                    if let Ok(f) = self.module.get_function(name) {
                        sel = Some((f, name));
                        break;
                    }
                }
                sel?
            };

            let (grid_x, grid_y) = if picked.contains("_sx") {
                let gx = ((series_len as u32) + ty - 1) / ty;
                let gy = ((num_series as u32) + 128 - 1) / 128;
                (gx, gy)
            } else {
                let gx = ((series_len as u32) + tx - 1) / tx;
                let gy = ((num_series as u32) + ty - 1) / ty;
                (gx, gy)
            };
            let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
            let block: BlockSize = (128, ty, 1).into();
            self.prefer_l1(&mut func);
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

                self.validate_launch(grid, block).ok()?;
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaBuffAveragesError::Cuda(e))
                    .ok()?;
            }
            unsafe {
                (*(self as *const _ as *mut CudaBuffAverages)).last_many =
                    Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
            }
            self.maybe_log_many_debug();
            Some(())
        };

        match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                if try_2d(tx, ty).is_some() {
                    return Ok(());
                }
            }
            ManySeriesKernelPolicy::Auto => {
                if num_series >= 128 {
                    if try_2d(128, 4).is_some() {
                        return Ok(());
                    }
                    if try_2d(128, 2).is_some() {
                        return Ok(());
                    }
                } else {
                    if try_2d(128, 2).is_some() {
                        return Ok(());
                    }
                    if try_2d(128, 4).is_some() {
                        return Ok(());
                    }
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => {}
        }

        let mut func = self
            .module
            .get_function("buff_averages_many_series_one_param_f32")
            .map_err(|_| CudaBuffAveragesError::MissingKernelSymbol {
                name: "buff_averages_many_series_one_param_f32",
            })?;
        self.prefer_l1(&mut func);
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
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
            self.validate_launch(grid, block)?;
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaBuffAverages)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
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

        let rows = combos.len();
        let prefix_bytes =
            (len + 1)
                .checked_mul(4 * 2)
                .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                    context: "prefix bytes",
                })?;
        let period_bytes =
            rows.checked_mul(4 * 2)
                .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                    context: "period bytes",
                })?;
        let out_elems = rows
            .checked_mul(len)
            .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                context: "rows * len",
            })?;
        let output_bytes =
            out_elems
                .checked_mul(4 * 2)
                .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                    context: "output bytes",
                })?;
        let bytes_required = prefix_bytes
            .checked_add(period_bytes)
            .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                context: "prefix+period",
            })?
            .checked_add(output_bytes)
            .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                context: "total bytes",
            })?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit_checked(bytes_required, headroom)?;

        let d_prefix_pv = DeviceBuffer::from_slice(&prefix_pv)?;
        let d_prefix_vv = DeviceBuffer::from_slice(&prefix_vv)?;

        let fast_periods: Vec<i32> = combos.iter().map(|&(f, _)| f as i32).collect();
        let slow_periods: Vec<i32> = combos.iter().map(|&(_, s)| s as i32).collect();
        let d_fast = DeviceBuffer::from_slice(&fast_periods)?;
        let d_slow = DeviceBuffer::from_slice(&slow_periods)?;

        let elems = combos.len() * len;
        let mut d_fast_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;
        let mut d_slow_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;

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

        self.stream.synchronize()?;

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

    pub fn buff_averages_batch_dev_exp2(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(price_f32, volume_f32, sweep)?;

        let (pv_hi, pv_lo, vv_hi, vv_lo) = build_prefix_sums_exp2(price_f32, volume_f32);

        let rows = combos.len();
        let prefix_bytes =
            (len + 1)
                .checked_mul(4 * 4)
                .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                    context: "prefix bytes (exp2)",
                })?;
        let period_bytes =
            rows.checked_mul(4 * 2)
                .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                    context: "period bytes (exp2)",
                })?;
        let out_elems = rows
            .checked_mul(len)
            .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                context: "rows * len (exp2)",
            })?;
        let output_bytes =
            out_elems
                .checked_mul(4 * 2)
                .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                    context: "output bytes (exp2)",
                })?;
        let bytes_required = prefix_bytes
            .checked_add(period_bytes)
            .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                context: "prefix+period (exp2)",
            })?
            .checked_add(output_bytes)
            .ok_or(CudaBuffAveragesError::ArithmeticOverflow {
                context: "total bytes (exp2)",
            })?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit_checked(bytes_required, headroom)?;

        let d_pv_hi = DeviceBuffer::from_slice(&pv_hi)?;
        let d_pv_lo = DeviceBuffer::from_slice(&pv_lo)?;
        let d_vv_hi = DeviceBuffer::from_slice(&vv_hi)?;
        let d_vv_lo = DeviceBuffer::from_slice(&vv_lo)?;
        let fast_periods: Vec<i32> = combos.iter().map(|&(f, _)| f as i32).collect();
        let slow_periods: Vec<i32> = combos.iter().map(|&(_, s)| s as i32).collect();
        let d_fast = DeviceBuffer::from_slice(&fast_periods)?;
        let d_slow = DeviceBuffer::from_slice(&slow_periods)?;

        let elems = combos.len() * len;
        let mut d_fast_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;
        let mut d_slow_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;

        let mut func = self
            .module
            .get_function("buff_averages_batch_prefix_exp2_f32")
            .map_err(|_| CudaBuffAveragesError::MissingKernelSymbol {
                name: "buff_averages_batch_prefix_exp2_f32",
            })?;
        self.prefer_l1(&mut func);
        let block_x: u32 = 256;
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();
        const MAX_GRID_Y: usize = 65_535;
        let mut start = 0usize;

        unsafe {
            let pv_ptr = d_pv_hi.as_device_ptr().as_raw();
            let vv_ptr = d_vv_hi.as_device_ptr().as_raw();
            let bytes = (len + 1) * std::mem::size_of::<f32>();
            self.set_l2_window_for_pair(pv_ptr, bytes, vv_ptr, bytes);
        }

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
                let mut fast_ptr = d_fast
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                let mut slow_ptr = d_slow
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                let mut combos_i = chunk as i32;
                let mut fast_out_ptr = d_fast_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                let mut slow_out_ptr = d_slow_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
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
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            start += chunk;
        }
        self.stream.synchronize()?;

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

    pub fn buff_averages_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_period: usize,
        slow_period: usize,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let prep = Self::prepare_many_series_inputs(
            prices_tm_f32,
            volumes_tm_f32,
            cols,
            rows,
            fast_period,
            slow_period,
        )?;

        let elems = cols * rows;
        let required = ((rows + 1) * cols * 2 + elems * 2) * std::mem::size_of::<f32>()
            + cols * std::mem::size_of::<i32>();
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::mem_check_enabled() {
        } else if let Some((free, _)) = Self::device_mem_info() {
            if required > free.saturating_sub(headroom) {
                return Err(CudaBuffAveragesError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        } else {
        }

        let d_pv = DeviceBuffer::from_slice(&prep.pv_prefix_tm)?;
        let d_vv = DeviceBuffer::from_slice(&prep.vv_prefix_tm)?;
        let d_fv = DeviceBuffer::from_slice(&prep.first_valids)?;
        let mut d_fast_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_slow_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

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

        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_fast_out,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_slow_out,
                rows,
                cols,
            },
        ))
    }

    pub fn buff_averages_many_series_one_param_dev_exp2(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_period: usize,
        slow_period: usize,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaBuffAveragesError> {
        let prep = Self::prepare_many_series_inputs(
            prices_tm_f32,
            volumes_tm_f32,
            cols,
            rows,
            fast_period,
            slow_period,
        )?;
        let (pv_hi_tm, pv_lo_tm, vv_hi_tm, vv_lo_tm) = build_prefix_sums_time_major_exp2(
            prices_tm_f32,
            volumes_tm_f32,
            cols,
            rows,
            &prep.first_valids,
        );

        let elems = cols * rows;
        let required = ((rows + 1) * cols * 4 + elems * 2) * std::mem::size_of::<f32>()
            + cols * std::mem::size_of::<i32>();
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::mem_check_enabled() {
        } else if let Some((free, _)) = Self::device_mem_info() {
            if required > free.saturating_sub(headroom) {
                return Err(CudaBuffAveragesError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }

        let d_pv_hi = DeviceBuffer::from_slice(&pv_hi_tm)?;
        let d_pv_lo = DeviceBuffer::from_slice(&pv_lo_tm)?;
        let d_vv_hi = DeviceBuffer::from_slice(&vv_hi_tm)?;
        let d_vv_lo = DeviceBuffer::from_slice(&vv_lo_tm)?;
        let d_fv = DeviceBuffer::from_slice(&prep.first_valids)?;
        let mut d_fast_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_slow_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        let mut func = self
            .module
            .get_function("buff_averages_many_series_one_param_exp2_f32")
            .map_err(|_| CudaBuffAveragesError::MissingKernelSymbol {
                name: "buff_averages_many_series_one_param_exp2_f32",
            })?;
        self.prefer_l1(&mut func);
        let block_x: u32 = 128;
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let pv_ptr = d_pv_hi.as_device_ptr().as_raw();
            let vv_ptr = d_vv_hi.as_device_ptr().as_raw();
            let bytes = (rows + 1) * cols * std::mem::size_of::<f32>();
            self.set_l2_window_for_pair(pv_ptr, bytes, vv_ptr, bytes);
        }
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
            self.validate_launch(grid, block)?;
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_fast_out,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_slow_out,
                rows,
                cols,
            },
        ))
    }

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
        fast_dev.buf.copy_to(fast_out)?;
        slow_dev.buf.copy_to(slow_out)?;

        Ok((combos.len(), len, combos))
    }

    pub fn buff_averages_batch_into_pinned_host_f32(
        &self,
        price_f32: &[f32],
        volume_f32: &[f32],
        sweep: &BuffAveragesBatchRange,
        fast_out_pinned: &mut LockedBuffer<f32>,
        slow_out_pinned: &mut LockedBuffer<f32>,
    ) -> Result<(usize, usize, Vec<(usize, usize)>), CudaBuffAveragesError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(price_f32, volume_f32, sweep)?;
        let expected = combos.len() * len;
        if fast_out_pinned.len() != expected || slow_out_pinned.len() != expected {
            return Err(CudaBuffAveragesError::InvalidInput(format!(
                "output slice mismatch (expected {}, fast={}, slow={})",
                expected,
                fast_out_pinned.len(),
                slow_out_pinned.len()
            )));
        }

        let (fast_dev, slow_dev) =
            self.run_batch_kernel(price_f32, volume_f32, &combos, first_valid)?;

        unsafe {
            fast_dev
                .buf
                .async_copy_to(fast_out_pinned.as_mut_slice(), &self.stream)?;
            slow_dev
                .buf
                .async_copy_to(slow_out_pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;

        Ok((combos.len(), len, combos))
    }
}

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

pub fn build_prefix_sums_exp2(
    price: &[f32],
    volume: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = price.len();
    let mut pv_hi = vec![0.0f32; len + 1];
    let mut pv_lo = vec![0.0f32; len + 1];
    let mut vv_hi = vec![0.0f32; len + 1];
    let mut vv_lo = vec![0.0f32; len + 1];
    let mut sh = 0.0f32;
    let mut sl = 0.0f32;
    let mut th = 0.0f32;
    let mut tl = 0.0f32;
    pv_hi[0] = 0.0;
    pv_lo[0] = 0.0;
    vv_hi[0] = 0.0;
    vv_lo[0] = 0.0;
    for i in 0..len {
        let p = price[i];
        let v = volume[i];
        let v_ok = v.is_finite();
        let p_ok = p.is_finite();
        let vol = if v_ok { v } else { 0.0 };
        let pv = if v_ok && p_ok { p.mul_add(v, 0.0) } else { 0.0 };
        prefix_step_f2(pv, &mut sh, &mut sl);
        prefix_step_f2(vol, &mut th, &mut tl);
        pv_hi[i + 1] = sh;
        pv_lo[i + 1] = sl;
        vv_hi[i + 1] = th;
        vv_lo[i + 1] = tl;
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
        let mut sh = 0.0f32;
        let mut sl = 0.0f32;
        let mut th = 0.0f32;
        let mut tl = 0.0f32;
        pv_hi[0 * cols + s] = 0.0;
        pv_lo[0 * cols + s] = 0.0;
        vv_hi[0 * cols + s] = 0.0;
        vv_lo[0 * cols + s] = 0.0;
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
            pv_hi[w] = sh;
            pv_lo[w] = sl;
            vv_hi[w] = th;
            vv_lo[w] = tl;
        }
    }
    (pv_hi, pv_lo, vv_hi, vv_lo)
}
