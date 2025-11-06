#![cfg(feature = "cuda")]

//! CUDA wrapper for Money Flow Index (MFI)
//!
//! Parity goals (Agents Guide):
//! - ALMA-style PTX load with DetermineTargetFromContext + O2 fallback
//! - NON_BLOCKING stream
//! - VRAM estimation + ~64MB headroom; grid-y chunking <= 65_535
//! - Batch: device-built double-single (float2) prefixes (3-stage scan) + O(1) per-period via prefix diffs
//! - Many-series×one-param: time-major sequential scan with double-single (float2) ring buffer via dynamic shared mem

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::mfi::{MfiBatchRange, MfiParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaMfiError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaMfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMfiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMfiError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaMfiError {}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaMfiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaMfi {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaMfiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaMfi {
    pub fn new(device_id: usize) -> Result<Self, CudaMfiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMfiError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mfi_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaMfiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaMfiPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaMfiPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

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
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] mfi batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaMfi)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] mfi many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaMfi)).debug_many_logged = true; }
            }
        }
    }

    fn expand_grid(r: &MfiBatchRange) -> Vec<MfiParams> {
        let (start, end, step) = r.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        };
        let mut out = Vec::with_capacity(periods.len());
        for &p in &periods { out.push(MfiParams { period: Some(p) }); }
        out
    }

    fn prepare_batch_inputs(
        typical: &[f32],
        volume: &[f32],
        sweep: &MfiBatchRange,
    ) -> Result<(Vec<MfiParams>, usize, usize), CudaMfiError> {
        if typical.len() != volume.len() {
            return Err(CudaMfiError::InvalidInput("length mismatch".into()));
        }
        if typical.is_empty() {
            return Err(CudaMfiError::InvalidInput("empty input".into()));
        }
        let len = typical.len();
        let first_valid = typical
            .iter()
            .zip(volume.iter())
            .position(|(p, v)| p.is_finite() && v.is_finite())
            .ok_or_else(|| CudaMfiError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaMfiError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaMfiError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaMfiError::InvalidInput(
                    "period exceeds data length".into(),
                ));
            }
            if len - first_valid < p {
                return Err(CudaMfiError::InvalidInput("not enough valid data".into()));
            }
        }
        Ok((combos, first_valid, len))
    }

    pub fn mfi_batch_dev(
        &self,
        typical_f32: &[f32],
        volume_f32: &[f32],
        sweep: &MfiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<MfiParams>), CudaMfiError> {
        use std::mem::size_of;

        let (combos, first_valid, len) = Self::prepare_batch_inputs(typical_f32, volume_f32, sweep)?;

        // Block size for Stages 1 & 3
        let block_x_scan: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let nb = ((len as u32 + block_x_scan - 1) / block_x_scan) as usize;

        // VRAM estimate: inputs + prefixes (float2) + block totals/offsets + periods + outputs
        let bytes_inputs  = 2 * len * size_of::<f32>();
        let bytes_prefix  = 2 * len * size_of::<[f32; 2]>();
        let bytes_blk     = 4 * nb  * size_of::<[f32; 2]>();
        let bytes_periods = combos.len() * size_of::<i32>();
        let bytes_out     = combos.len() * len * size_of::<f32>();
        let required      = bytes_inputs + bytes_prefix + bytes_blk + bytes_periods + bytes_out;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaMfiError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Device buffers
        let d_tp  = unsafe { DeviceBuffer::from_slice_async(typical_f32, &self.stream) }
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        let d_vol = unsafe { DeviceBuffer::from_slice_async(volume_f32, &self.stream) }
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;

        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(14) as i32)
            .collect();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(14) as i32)
            .collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;

        // Prefix workspace (float2)
        let mut d_pos_ps: DeviceBuffer<[f32; 2]> = unsafe {
            DeviceBuffer::uninitialized_async(len, &self.stream)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?
        };
        let mut d_neg_ps: DeviceBuffer<[f32; 2]> = unsafe {
            DeviceBuffer::uninitialized_async(len, &self.stream)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?
        };
        let mut d_blk_tot_pos: DeviceBuffer<[f32; 2]> = unsafe {
            DeviceBuffer::uninitialized_async(nb, &self.stream)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?
        };
        let mut d_blk_tot_neg: DeviceBuffer<[f32; 2]> = unsafe {
            DeviceBuffer::uninitialized_async(nb, &self.stream)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?
        };
        let mut d_blk_off_pos: DeviceBuffer<[f32; 2]> = unsafe {
            DeviceBuffer::uninitialized_async(nb, &self.stream)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?
        };
        let mut d_blk_off_neg: DeviceBuffer<[f32; 2]> = unsafe {
            DeviceBuffer::uninitialized_async(nb, &self.stream)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?
        };

        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(len * combos.len(), &self.stream)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?
        };

        // Kernel symbols
        let k1 = self.module.get_function("mfi_prefix_stage1_transform_scan_ds")
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        let k2 = self.module.get_function("mfi_prefix_stage2_scan_block_totals")
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        let k3 = self.module.get_function("mfi_prefix_stage3_add_offsets")
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        let k4 = self.module.get_function("mfi_batch_from_prefix_ds_f32")
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;

        unsafe { (*(self as *const _ as *mut CudaMfi)).last_batch = Some(BatchKernelSelected::Plain { block_x: block_x_scan }); }

        // Stage 1
        {
            let grid: GridSize = ((nb as u32).max(1), 1, 1).into();
            let block: BlockSize = (block_x_scan, 1, 1).into();
            unsafe {
                let mut tp_ptr    = d_tp.as_device_ptr().as_raw();
                let mut vol_ptr   = d_vol.as_device_ptr().as_raw();
                let mut len_i     = len as i32;
                let mut first_i   = first_valid as i32;
                let mut pos_ps    = d_pos_ps.as_device_ptr().as_raw();
                let mut neg_ps    = d_neg_ps.as_device_ptr().as_raw();
                let mut blk_tot_p = d_blk_tot_pos.as_device_ptr().as_raw();
                let mut blk_tot_n = d_blk_tot_neg.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut tp_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut pos_ps as *mut _ as *mut c_void,
                    &mut neg_ps as *mut _ as *mut c_void,
                    &mut blk_tot_p as *mut _ as *mut c_void,
                    &mut blk_tot_n as *mut _ as *mut c_void,
                ];
                self.stream.launch(&k1, grid, block, 0, args)
                    .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
            }
        }

        // Stage 2
        {
            let grid: GridSize = (1, 1, 1).into();
            let block: BlockSize = (1, 1, 1).into();
            unsafe {
                let mut blk_tot_p = d_blk_tot_pos.as_device_ptr().as_raw();
                let mut blk_tot_n = d_blk_tot_neg.as_device_ptr().as_raw();
                let mut blk_off_p = d_blk_off_pos.as_device_ptr().as_raw();
                let mut blk_off_n = d_blk_off_neg.as_device_ptr().as_raw();
                let mut nb_i      = nb as i32;
                let args: &mut [*mut c_void] = &mut [
                    &mut blk_tot_p as *mut _ as *mut c_void,
                    &mut blk_tot_n as *mut _ as *mut c_void,
                    &mut blk_off_p as *mut _ as *mut c_void,
                    &mut blk_off_n as *mut _ as *mut c_void,
                    &mut nb_i as *mut _ as *mut c_void,
                ];
                self.stream.launch(&k2, grid, block, 0, args)
                    .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
            }
        }

        // Stage 3
        {
            let grid: GridSize = ((nb as u32).max(1), 1, 1).into();
            let block: BlockSize = (block_x_scan, 1, 1).into();
            unsafe {
                let mut pos_ps    = d_pos_ps.as_device_ptr().as_raw();
                let mut neg_ps    = d_neg_ps.as_device_ptr().as_raw();
                let mut blk_off_p = d_blk_off_pos.as_device_ptr().as_raw();
                let mut blk_off_n = d_blk_off_neg.as_device_ptr().as_raw();
                let mut len_i     = len as i32;
                let args: &mut [*mut c_void] = &mut [
                    &mut pos_ps as *mut _ as *mut c_void,
                    &mut neg_ps as *mut _ as *mut c_void,
                    &mut blk_off_p as *mut _ as *mut c_void,
                    &mut blk_off_n as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                ];
                self.stream.launch(&k3, grid, block, 0, args)
                    .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
            }
        }

        // Stage 4 (chunked over grid.y)
        let block_x_out: u32 = 128;
        let grid_x_out: u32 = ((len as u32) + block_x_out - 1) / block_x_out;
        let mut launched = 0usize;
        while launched < combos.len() {
            let chunk = (combos.len() - launched).min(65_535);
            let periods_off = launched * size_of::<i32>();
            let out_off = (launched * len) * size_of::<f32>();
            let grid: GridSize = (grid_x_out.max(1), chunk as u32, 1).into();
            let block: BlockSize = (block_x_out, 1, 1).into();
            unsafe {
                let mut pos_ps  = d_pos_ps.as_device_ptr().as_raw();
                let mut neg_ps  = d_neg_ps.as_device_ptr().as_raw();
                let mut len_i   = len as i32;
                let mut first_i = first_valid as i32;
                let mut periods_ptr = d_periods.as_device_ptr().as_raw() + (periods_off as u64);
                let mut n_chunk_i   = chunk as i32; // kernel expects n_combos but uses grid.y index
                let mut out_ptr     = d_out.as_device_ptr().as_raw() + (out_off as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut pos_ps as *mut _ as *mut c_void,
                    &mut neg_ps as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut n_chunk_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&k4, grid, block, 0, args)
                    .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
            }
            launched += chunk;
        }

        self.stream.synchronize().map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        self.maybe_log_batch_debug();
        Ok((
            DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len },
            combos,
        ))
    }

    pub fn mfi_many_series_one_param_time_major_dev(
        &self,
        typical_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMfiError> {
        if typical_tm_f32.len() != volume_tm_f32.len() {
            return Err(CudaMfiError::InvalidInput("length mismatch".into()));
        }
        if typical_tm_f32.len() != cols * rows {
            return Err(CudaMfiError::InvalidInput("unexpected matrix size".into()));
        }
        if period == 0 {
            return Err(CudaMfiError::InvalidInput("period must be > 0".into()));
        }

        // First-valid per series
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let tp = typical_tm_f32[t * cols + s];
                let v = volume_tm_f32[t * cols + s];
                if tp.is_finite() && v.is_finite() {
                    fv = t as i32;
                    break;
                }
                if tp.is_finite() && v.is_finite() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        let d_tp = unsafe { DeviceBuffer::from_slice_async(typical_tm_f32, &self.stream) }
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        let d_vol = unsafe { DeviceBuffer::from_slice_async(volume_tm_f32, &self.stream) }
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;

        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(rows * cols, &self.stream)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?
        };

        let func = self
            .module
            .get_function("mfi_many_series_one_param_f32")
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = (2 * period * std::mem::size_of::<[f32; 2]>()) as u32; // ring buffers (float2)
        unsafe { (*(self as *const _ as *mut CudaMfi)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        unsafe {
            let mut tp_ptr = d_tp.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut tp_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaMfiError::Cuda(e.to_string()))?;
        self.maybe_log_many_debug();
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ------------------------ Benches ------------------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_volume};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::mfi::MfiBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        use std::mem::size_of;
        const BS: usize = 256;
        let nb = (ONE_SERIES_LEN + BS - 1) / BS;
        let in_bytes     = 2 * ONE_SERIES_LEN * size_of::<f32>();
        let prefix_bytes = 2 * ONE_SERIES_LEN * size_of::<[f32; 2]>(); // pos_ps + neg_ps
        let blk_bytes    = 4 * nb * size_of::<[f32; 2]>();              // blk_tot_* + blk_off_*
        let out_bytes    = ONE_SERIES_LEN * PARAM_SWEEP * size_of::<f32>();
        in_bytes + prefix_bytes + blk_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct MfiBatchState {
        cuda: CudaMfi,
        tp: Vec<f32>,
        vol: Vec<f32>,
        sweep: MfiBatchRange,
    }
    impl CudaBenchState for MfiBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .mfi_batch_dev(&self.tp, &self.vol, &self.sweep)
                .expect("mfi batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMfi::new(0).expect("CudaMfi");
        let mut tp = gen_series(ONE_SERIES_LEN);
        let mut vol = gen_volume(ONE_SERIES_LEN);
        // Early NaNs (warmup region)
        for i in 0..16 {
            tp[i] = f32::NAN;
            vol[i] = f32::NAN;
        }
        let sweep = MfiBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(MfiBatchState { cuda, tp, vol, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "mfi",
            "one_series_many_params",
            "mfi_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
