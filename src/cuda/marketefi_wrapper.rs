//! CUDA wrapper for Market Facilitation Index (marketefi)
//!
//! Parity with ALMA/CWMA wrappers:
//! - PTX load via include_str!(... "/marketefi_kernel.ptx") with DetermineTargetFromContext + O2
//!   and graceful fallbacks.
//! - NON_BLOCKING stream.
//! - Simple policy enums and one-dimensional launches.
//! - VRAM estimation with ~64MB headroom before async copies.
//! - Device entry points:
//!     - `marketefi_dev(&[f32], &[f32], &[f32]) -> DeviceArrayF32` (one-series)
//!     - `marketefi_many_series_one_param_time_major_dev(&[f32], ... ) -> DeviceArrayF32`.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaMarketefiError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error(
        "launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
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
pub struct CudaMarketefiPolicy {
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

pub struct CudaMarketefi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaMarketefiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaMarketefi {
    pub fn new(device_id: usize) -> Result<Self, CudaMarketefiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/marketefi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaMarketefiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn set_policy(&mut self, policy: CudaMarketefiPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaMarketefiPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
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
        if let Some((free, _)) = Self::device_mem_info() {
            match required_bytes.checked_add(headroom_bytes) {
                Some(needed) => needed <= free,
                None => false,
            }
        } else {
            true
        }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] marketefi batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMarketefi)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] marketefi many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMarketefi)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------- Public device entry points ----------

    /// One-series compute on device (paramless): returns a single row DeviceArrayF32 (1 x len)
    pub fn marketefi_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        volume_f32: &[f32],
    ) -> Result<DeviceArrayF32, CudaMarketefiError> {
        if high_f32.len() != low_f32.len() || low_f32.len() != volume_f32.len() {
            return Err(CudaMarketefiError::InvalidInput("length mismatch".into()));
        }
        let len = high_f32.len();
        if len == 0 {
            return Err(CudaMarketefiError::InvalidInput("empty input".into()));
        }
        let first = (0..len)
            .find(|&i| {
                high_f32[i].is_finite() && low_f32[i].is_finite() && volume_f32[i].is_finite()
            })
            .ok_or_else(|| CudaMarketefiError::InvalidInput("all values are NaN".into()))?;

        // VRAM estimate: 3 inputs + 1 output (FP32), ~64MB headroom
        let elem_bytes = std::mem::size_of::<f32>();
        let per_vec = len
            .checked_mul(elem_bytes)
            .ok_or_else(|| CudaMarketefiError::InvalidInput("size overflow (len * elem_size)".into()))?;
        let bytes = 4usize
            .checked_mul(per_vec)
            .ok_or_else(|| CudaMarketefiError::InvalidInput("size overflow (4 * len * elem_size)".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            let free = Self::device_mem_info().map(|(free, _)| free).unwrap_or(0);
            return Err(CudaMarketefiError::OutOfMemory {
                required: bytes,
                free,
                headroom,
            });
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_f32, &self.stream) }?;
        let d_vol = unsafe { DeviceBuffer::from_slice_async(volume_f32, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;

        self.marketefi_device(&d_high, &d_low, &d_vol, len, first, &mut d_out)?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    pub fn marketefi_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_vol: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMarketefiError> {
        if d_high.len() != len || d_low.len() != len || d_vol.len() != len || d_out.len() != len {
            return Err(CudaMarketefiError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }

        let func = self
            .module
            .get_function("marketefi_kernel_f32")
            .map_err(|_| CudaMarketefiError::MissingKernelSymbol {
                name: "marketefi_kernel_f32",
            })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256u32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = (first_valid.min(len)) as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaMarketefi)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    /// Many-series × one-param (paramless). Time-major layout (cols=num_series, rows=series_len).
    pub fn marketefi_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaMarketefiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMarketefiError::InvalidInput("empty input".into()));
        }
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMarketefiError::InvalidInput("size overflow (cols * rows)".into()))?;
        if high_tm_f32.len() != n || low_tm_f32.len() != n || volume_tm_f32.len() != n {
            return Err(CudaMarketefiError::InvalidInput("length mismatch".into()));
        }

        // Compute first-valid per series: first index t where H,L,V are finite
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0i32;
            let mut found = false;
            for t in 0..rows {
                let idx = t * cols + s;
                let h = high_tm_f32[idx];
                let l = low_tm_f32[idx];
                let v = volume_tm_f32[idx];
                if h.is_finite() && l.is_finite() && v.is_finite() {
                    fv = t as i32;
                    found = true;
                    break;
                }
            }
            first_valids[s] = if found { fv } else { rows as i32 };
        }

        // VRAM check: 3 inputs + output (all f32) + first_valids (i32)
        let f_bytes = n
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|b| b.checked_mul(4))
            .ok_or_else(|| CudaMarketefiError::InvalidInput("size overflow for fp32 buffers".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMarketefiError::InvalidInput("size overflow for first_valids".into()))?;
        let bytes = f_bytes
            .checked_add(first_bytes)
            .ok_or_else(|| CudaMarketefiError::InvalidInput("size overflow (total bytes)".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            let free = Self::device_mem_info().map(|(free, _)| free).unwrap_or(0);
            return Err(CudaMarketefiError::OutOfMemory {
                required: bytes,
                free,
                headroom,
            });
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }?;
        let d_vol = unsafe { DeviceBuffer::from_slice_async(volume_tm_f32, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }?;

        self.marketefi_many_series_one_param_device(
            &d_high, &d_low, &d_vol, &d_first, cols, rows, &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn marketefi_many_series_one_param_device(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_vol_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMarketefiError> {
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMarketefiError::InvalidInput("size overflow (cols * rows)".into()))?;
        if d_high_tm.len() != n || d_low_tm.len() != n || d_vol_tm.len() != n || d_out_tm.len() != n
        {
            return Err(CudaMarketefiError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != cols {
            return Err(CudaMarketefiError::InvalidInput("first_valids length mismatch".into()));
        }

        // --- Kernel entry ---
        let func = self
            .module
            .get_function("marketefi_many_series_one_param_f32")
            .map_err(|_| CudaMarketefiError::MissingKernelSymbol {
                name: "marketefi_many_series_one_param_f32",
            })?;

        // --- Launch policy (2D grid: time tiles × series tiles) ---
        // block.x spans series; grid.x tiles time; grid.y tiles series.
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };

        // Must match the CUDA kernel's MKT_T_TILE (default 128)
        const T_TILE: u32 = 128;

        // Host-side precheck for vectorization: need cols % 4 == 0 and 16B alignment.
        let mut h_ptr = d_high_tm.as_device_ptr().as_raw();
        let mut l_ptr = d_low_tm.as_device_ptr().as_raw();
        let mut v_ptr = d_vol_tm.as_device_ptr().as_raw();
        let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
        let mut o_ptr = d_out_tm.as_device_ptr().as_raw();

        let aligned16 = ((h_ptr | l_ptr | v_ptr | o_ptr | fv_ptr) & 0xF) == 0;
        let host_vec4_ok = aligned16 && ((cols & 3) == 0);

        // Series groups: 1 per series in scalar path, 1 per 4 series in vector path.
        let series_groups: u32 = if host_vec4_ok { (cols as u32) >> 2 } else { cols as u32 };

        // grid.x: number of time tiles; grid.y: number of series tiles
        let grid_x = ((rows as u32) + T_TILE - 1) / T_TILE;
        let grid_y = (series_groups + block_x - 1) / block_x;

        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut args: [*mut c_void; 7] = [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }

        // Keep the API the same; record chosen block size
        unsafe {
            (*(self as *const _ as *mut CudaMarketefi)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    // Back-compat alias for API parity with other wrappers
    pub fn marketefi_dev(
        &self,
        h: &[f32],
        l: &[f32],
        v: &[f32],
    ) -> Result<DeviceArrayF32, CudaMarketefiError> {
        self.marketefi_batch_dev(h, l, v)
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        // One-series (100k)
        v.push(CudaBenchScenario::new(
            "marketefi",
            "one_series",
            "marketefi_cuda_batch_dev",
            "100k",
            || {
                struct State {
                    cuda: CudaMarketefi,
                    h: Vec<f32>,
                    l: Vec<f32>,
                    v: Vec<f32>,
                }
                impl CudaBenchState for State {
                    fn launch(&mut self) {
                        let _ = self.cuda.marketefi_dev(&self.h, &self.l, &self.v);
                    }
                }
                let n = 100_000usize;
                let mut h = vec![f32::NAN; n];
                let mut l = vec![f32::NAN; n];
                let mut vv = vec![f32::NAN; n];
                for i in 0..n {
                    let x = i as f32;
                    h[i] = (x * 0.001).sin() + 1.0;
                    l[i] = h[i] - 0.5f32.abs();
                    vv[i] = (x * 0.002).cos().abs() + 0.1;
                }
                let cuda = CudaMarketefi::new(0).unwrap();
                Box::new(State { cuda, h, l, v: vv })
            },
        ));
        // Many-series (64 x 16k)
        v.push(CudaBenchScenario::new(
            "marketefi",
            "many_series_one_param",
            "marketefi_cuda_many_series_one_param_dev",
            "64x16k",
            || {
                struct State {
                    cuda: CudaMarketefi,
                    h: Vec<f32>,
                    l: Vec<f32>,
                    v: Vec<f32>,
                    cols: usize,
                    rows: usize,
                }
                impl CudaBenchState for State {
                    fn launch(&mut self) {
                        let _ = self.cuda.marketefi_many_series_one_param_time_major_dev(
                            &self.h, &self.l, &self.v, self.cols, self.rows,
                        );
                    }
                }
                let cols = 64usize;
                let rows = 16_384usize;
                let mut h = vec![f32::NAN; cols * rows];
                let mut l = vec![f32::NAN; cols * rows];
                let mut vv = vec![f32::NAN; cols * rows];
                for s in 0..cols {
                    for t in 0..rows {
                        let x = (t as f32) + (s as f32) * 0.25;
                        h[t * cols + s] = (x * 0.001).sin() + 1.0;
                        l[t * cols + s] = h[t * cols + s] - 0.3f32.abs();
                        vv[t * cols + s] = (x * 0.002).cos().abs() + 0.1;
                    }
                }
                let cuda = CudaMarketefi::new(0).unwrap();
                Box::new(State {
                    cuda,
                    h,
                    l,
                    v: vv,
                    cols,
                    rows,
                })
            },
        ));
        v
    }
}
