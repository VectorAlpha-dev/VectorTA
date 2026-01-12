#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::cuda::DeviceArrayF32Triplet;
use crate::indicators::mod_god_mode::{ModGodModeBatchRange, ModGodModeMode, ModGodModeParams};
use cust::context::Context;
use cust::device::Device;
use cust::device::DeviceAttribute as DevAttr;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

const MGM_RING_KCAP: i32 = 64;

#[derive(Debug, Error)]
pub enum CudaModGodModeError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaModGodModeBatchResult {
    pub outputs: DeviceArrayF32Triplet,
    pub combos: Vec<ModGodModeParams>,
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
pub struct CudaModGodModePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaModGodModePolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaModGodMode {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaModGodModePolicy,
    debug_logged: bool,
}

impl CudaModGodMode {
    pub fn new(device_id: usize) -> Result<Self, CudaModGodModeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mod_god_mode_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                .or_else(|_| Module::from_ptx(ptx, &[]))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaModGodModePolicy::default(),
            debug_logged: false,
        })
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
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaModGodModeError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaModGodModeError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    fn axis_usize_cuda(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaModGodModeError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<_> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(CudaModGodModeError::InvalidInput(format!(
                    "invalid range expansion: start={start}, end={end}, step={step}"
                )));
            }
            Ok(v)
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur - end < step {
                    break;
                }
                cur -= step;
            }
            if v.is_empty() {
                return Err(CudaModGodModeError::InvalidInput(format!(
                    "invalid range expansion: start={start}, end={end}, step={step}"
                )));
            }
            Ok(v)
        }
    }

    fn expand_range(
        r: &ModGodModeBatchRange,
    ) -> Result<Vec<ModGodModeParams>, CudaModGodModeError> {
        let n1s = Self::axis_usize_cuda(r.n1)?;
        let n2s = Self::axis_usize_cuda(r.n2)?;
        let n3s = Self::axis_usize_cuda(r.n3)?;
        let cap = n1s
            .len()
            .checked_mul(n2s.len())
            .and_then(|v| v.checked_mul(n3s.len()))
            .ok_or_else(|| CudaModGodModeError::InvalidInput("batch grid size overflow".into()))?;
        let mut v = Vec::with_capacity(cap);
        for &a in &n1s {
            for &b in &n2s {
                for &c in &n3s {
                    v.push(ModGodModeParams {
                        n1: Some(a),
                        n2: Some(b),
                        n3: Some(c),
                        mode: Some(r.mode),
                        use_volume: Some(false),
                    });
                }
            }
        }
        if v.is_empty() {
            return Err(CudaModGodModeError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        Ok(v)
    }

    #[inline]
    fn fast_cap() -> i32 {
        MGM_RING_KCAP
    }

    #[inline]
    fn fast_block_x() -> u32 {
        64
    }

    #[inline]
    fn fast_shared_bytes(block_x: u32) -> usize {
        let cap = Self::fast_cap() as usize;
        let per_thread = (2 * std::mem::size_of::<f32>()
            + 2 * std::mem::size_of::<i32>()
            + std::mem::size_of::<i8>())
            * cap;
        per_thread * (block_x as usize)
    }

    #[inline]
    fn validate_launch(&self, grid: GridSize, block: BlockSize) -> Result<(), CudaModGodModeError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev.get_attribute(DevAttr::MaxThreadsPerBlock)? as u32;
        let block_threads = block.x * block.y * block.z;
        if block_threads > max_threads {
            return Err(CudaModGodModeError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }
        let max_grid_x = dev.get_attribute(DevAttr::MaxGridDimX)? as u32;
        if grid.x > max_grid_x {
            return Err(CudaModGodModeError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }
        Ok(())
    }

    pub fn mod_god_mode_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: Option<&[f32]>,
        sweep: &ModGodModeBatchRange,
    ) -> Result<CudaModGodModeBatchResult, CudaModGodModeError> {
        let n = close.len();
        if n == 0 {
            return Err(CudaModGodModeError::InvalidInput("empty inputs".into()));
        }
        if high.len() != n || low.len() != n {
            return Err(CudaModGodModeError::InvalidInput(
                "H/L/C length mismatch".into(),
            ));
        }
        if let Some(v) = volume {
            if v.len() != n {
                return Err(CudaModGodModeError::InvalidInput(
                    "volume length mismatch".into(),
                ));
            }
        }
        if n == 0 {
            return Err(CudaModGodModeError::InvalidInput("empty inputs".into()));
        }
        if high.len() != n || low.len() != n {
            return Err(CudaModGodModeError::InvalidInput(
                "H/L/C length mismatch".into(),
            ));
        }
        if let Some(v) = volume {
            if v.len() != n {
                return Err(CudaModGodModeError::InvalidInput(
                    "volume length mismatch".into(),
                ));
            }
        }
        let combos = Self::expand_range(sweep)?;
        let rows = combos.len();

        let mut first_valid_opt = None;
        for (i, &v) in close.iter().enumerate() {
            if v.is_finite() {
                first_valid_opt = Some(i);
                break;
            }
        }
        let first_valid = first_valid_opt
            .ok_or_else(|| CudaModGodModeError::InvalidInput("all values are NaN".into()))?;

        let mut n1s: Vec<i32> = Vec::with_capacity(rows);
        let mut n2s: Vec<i32> = Vec::with_capacity(rows);
        let mut n3s: Vec<i32> = Vec::with_capacity(rows);
        let mut modes: Vec<i32> = Vec::with_capacity(rows);
        for p in &combos {
            n1s.push(p.n1.unwrap() as i32);
            n2s.push(p.n2.unwrap() as i32);
            n3s.push(p.n3.unwrap() as i32);
            let m = match p.mode.unwrap() {
                ModGodModeMode::Godmode => 0,
                ModGodModeMode::Tradition => 1,
                ModGodModeMode::GodmodeMg => 2,
                ModGodModeMode::TraditionMg => 3,
            };
            modes.push(m);
        }

        let cap = Self::fast_cap();
        let mut large_idxs: Vec<usize> = Vec::new();
        for i in 0..rows {
            let b = n2s[i];
            let c = n3s[i];
            if b > cap || c > cap {
                large_idxs.push(i);
            }
        }

        let use_vol = volume.is_some();
        let elem_f32 = std::mem::size_of::<f32>();
        let base_scalars = if use_vol {
            n.checked_mul(3)
                .ok_or_else(|| CudaModGodModeError::InvalidInput("input size overflow".into()))?
        } else {
            n
        };
        let in_bytes_base = base_scalars
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaModGodModeError::InvalidInput("input size overflow".into()))?;
        let vol_bytes = if volume.is_some() {
            n.checked_mul(elem_f32)
                .ok_or_else(|| CudaModGodModeError::InvalidInput("volume size overflow".into()))?
        } else {
            0
        };
        let in_bytes = in_bytes_base
            .checked_add(vol_bytes)
            .ok_or_else(|| CudaModGodModeError::InvalidInput("input size overflow".into()))?;
        let param_bytes = rows
            .checked_mul(4)
            .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| CudaModGodModeError::InvalidInput("param size overflow".into()))?;
        let out_elems = rows
            .checked_mul(n)
            .ok_or_else(|| CudaModGodModeError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(3)
            .and_then(|v| v.checked_mul(elem_f32))
            .ok_or_else(|| CudaModGodModeError::InvalidInput("output size overflow".into()))?;
        let required = in_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaModGodModeError::InvalidInput("total size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if let Err(e @ CudaModGodModeError::OutOfMemory { .. }) = Self::will_fit(required, headroom)
        {
            return Err(e);
        }

        let d_close = DeviceBuffer::from_slice(close)?;
        let d_high = if use_vol {
            Some(DeviceBuffer::from_slice(high)?)
        } else {
            None
        };
        let d_low = if use_vol {
            Some(DeviceBuffer::from_slice(low)?)
        } else {
            None
        };
        let d_volume = if let Some(v) = volume {
            Some(DeviceBuffer::from_slice(v)?)
        } else {
            None
        };
        let d_n1s = DeviceBuffer::from_slice(&n1s)?;
        let d_n2s = DeviceBuffer::from_slice(&n2s)?;
        let d_n3s = DeviceBuffer::from_slice(&n3s)?;
        let d_modes = DeviceBuffer::from_slice(&modes)?;

        let mut d_wt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems)? };
        let mut d_sig: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems)? };
        let mut d_hist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems)? };

        if large_idxs.len() < rows {
            let func_fast = self
                .module
                .get_function("mod_god_mode_batch_f32_shared_fast")
                .map_err(|_| CudaModGodModeError::MissingKernelSymbol {
                    name: "mod_god_mode_batch_f32_shared_fast",
                })?;
            let mut block_x = Self::fast_block_x();
            let mut shmem_bytes = Self::fast_shared_bytes(block_x);
            let max_dyn_default: usize = 48 * 1024;
            while shmem_bytes > max_dyn_default && block_x > 1 {
                block_x /= 2;
                shmem_bytes = Self::fast_shared_bytes(block_x);
            }
            if !self.debug_logged && std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
                let grid_x_total = ((rows as u32) + block_x - 1) / block_x;
                eprintln!(
                    "[mod_god_mode] fast kernel: block_x={} grid_x={} shmem={} bytes",
                    block_x, grid_x_total, shmem_bytes
                );
                unsafe {
                    (*(self as *const _ as *mut CudaModGodMode)).debug_logged = true;
                }
            }
            let max_blocks = 65_535usize;
            let rows_per_launch = max_blocks.saturating_mul(block_x as usize);
            let mut launched = 0usize;
            while launched < rows {
                let chunk = std::cmp::min(rows - launched, rows_per_launch);
                let mut high_ptr = d_high
                    .as_ref()
                    .map(|b| b.as_device_ptr().as_raw())
                    .unwrap_or(0);
                let mut low_ptr = d_low
                    .as_ref()
                    .map(|b| b.as_device_ptr().as_raw())
                    .unwrap_or(0);
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut vol_ptr = d_volume
                    .as_ref()
                    .map(|b| b.as_device_ptr().as_raw())
                    .unwrap_or(0);
                let mut len_i = n as i32;
                let mut first_i = first_valid as i32;
                let mut rows_i = chunk as i32;
                let mut n1_ptr =
                    d_n1s.as_device_ptr().as_raw() + (launched * std::mem::size_of::<i32>()) as u64;
                let mut n2_ptr =
                    d_n2s.as_device_ptr().as_raw() + (launched * std::mem::size_of::<i32>()) as u64;
                let mut n3_ptr =
                    d_n3s.as_device_ptr().as_raw() + (launched * std::mem::size_of::<i32>()) as u64;
                let mut modes_ptr = d_modes.as_device_ptr().as_raw()
                    + (launched * std::mem::size_of::<i32>()) as u64;
                let mut use_vol_i = if use_vol { 1i32 } else { 0i32 };
                let mut wt_ptr = d_wt.as_device_ptr().as_raw()
                    + (launched * n * std::mem::size_of::<f32>()) as u64;
                let mut sig_ptr = d_sig.as_device_ptr().as_raw()
                    + (launched * n * std::mem::size_of::<f32>()) as u64;
                let mut hist_ptr = d_hist.as_device_ptr().as_raw()
                    + (launched * n * std::mem::size_of::<f32>()) as u64;
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut n1_ptr as *mut _ as *mut c_void,
                    &mut n2_ptr as *mut _ as *mut c_void,
                    &mut n3_ptr as *mut _ as *mut c_void,
                    &mut modes_ptr as *mut _ as *mut c_void,
                    &mut use_vol_i as *mut _ as *mut c_void,
                    &mut wt_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                    &mut hist_ptr as *mut _ as *mut c_void,
                ];
                let grid_x = ((chunk as u32) + block_x - 1) / block_x;
                let grid: GridSize = (grid_x.max(1), 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                self.validate_launch(grid, block)?;
                unsafe {
                    self.stream
                        .launch(&func_fast, grid, block, (shmem_bytes as u32), args)?;
                }
                launched += chunk;
            }
        }

        if !large_idxs.is_empty() {
            let func_fallback =
                self.module
                    .get_function("mod_god_mode_batch_f32")
                    .map_err(|_| CudaModGodModeError::MissingKernelSymbol {
                        name: "mod_god_mode_batch_f32",
                    })?;
            let block_x: u32 = match self.policy.batch {
                BatchKernelPolicy::Auto => 128,
                BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            };
            let block: BlockSize = (block_x, 1, 1).into();
            let max_blocks = 65_535usize;
            let rows_per_launch = max_blocks.saturating_mul(block_x as usize);
            let mut use_vol_i = if use_vol { 1i32 } else { 0i32 };

            let mut launch_range =
                |range_start: usize, range_len: usize| -> Result<(), CudaModGodModeError> {
                    let mut launched = 0usize;
                    while launched < range_len {
                        let chunk = std::cmp::min(range_len - launched, rows_per_launch);
                        let start_row = range_start + launched;

                        let mut high_ptr = d_high
                            .as_ref()
                            .map(|b| b.as_device_ptr().as_raw())
                            .unwrap_or(0);
                        let mut low_ptr = d_low
                            .as_ref()
                            .map(|b| b.as_device_ptr().as_raw())
                            .unwrap_or(0);
                        let mut close_ptr = d_close.as_device_ptr().as_raw();
                        let mut vol_ptr = d_volume
                            .as_ref()
                            .map(|b| b.as_device_ptr().as_raw())
                            .unwrap_or(0);
                        let mut len_i = n as i32;
                        let mut first_i = first_valid as i32;
                        let mut rows_i = chunk as i32;

                        let mut n1_ptr = d_n1s.as_device_ptr().as_raw()
                            + (start_row * std::mem::size_of::<i32>()) as u64;
                        let mut n2_ptr = d_n2s.as_device_ptr().as_raw()
                            + (start_row * std::mem::size_of::<i32>()) as u64;
                        let mut n3_ptr = d_n3s.as_device_ptr().as_raw()
                            + (start_row * std::mem::size_of::<i32>()) as u64;
                        let mut modes_ptr = d_modes.as_device_ptr().as_raw()
                            + (start_row * std::mem::size_of::<i32>()) as u64;

                        let out_off = start_row * n;
                        let mut wt_ptr = d_wt.as_device_ptr().as_raw()
                            + (out_off * std::mem::size_of::<f32>()) as u64;
                        let mut sig_ptr = d_sig.as_device_ptr().as_raw()
                            + (out_off * std::mem::size_of::<f32>()) as u64;
                        let mut hist_ptr = d_hist.as_device_ptr().as_raw()
                            + (out_off * std::mem::size_of::<f32>()) as u64;

                        let args: &mut [*mut c_void] = &mut [
                            &mut high_ptr as *mut _ as *mut c_void,
                            &mut low_ptr as *mut _ as *mut c_void,
                            &mut close_ptr as *mut _ as *mut c_void,
                            &mut vol_ptr as *mut _ as *mut c_void,
                            &mut len_i as *mut _ as *mut c_void,
                            &mut first_i as *mut _ as *mut c_void,
                            &mut rows_i as *mut _ as *mut c_void,
                            &mut n1_ptr as *mut _ as *mut c_void,
                            &mut n2_ptr as *mut _ as *mut c_void,
                            &mut n3_ptr as *mut _ as *mut c_void,
                            &mut modes_ptr as *mut _ as *mut c_void,
                            &mut use_vol_i as *mut _ as *mut c_void,
                            &mut wt_ptr as *mut _ as *mut c_void,
                            &mut sig_ptr as *mut _ as *mut c_void,
                            &mut hist_ptr as *mut _ as *mut c_void,
                        ];

                        let grid_x = ((chunk as u32) + block_x - 1) / block_x;
                        let grid: GridSize = (grid_x.max(1), 1, 1).into();
                        self.validate_launch(grid, block)?;
                        unsafe {
                            self.stream.launch(&func_fallback, grid, block, 0, args)?;
                        }
                        launched += chunk;
                    }
                    Ok(())
                };

            let mut range_start = large_idxs[0];
            let mut prev = range_start;
            for &idx in large_idxs.iter().skip(1) {
                if idx == prev + 1 {
                    prev = idx;
                    continue;
                }
                launch_range(range_start, (prev + 1) - range_start)?;
                range_start = idx;
                prev = idx;
            }
            launch_range(range_start, (prev + 1) - range_start)?;
        }
        self.stream.synchronize()?;

        let outputs = DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 {
                buf: d_wt,
                rows,
                cols: n,
            },
            wt2: DeviceArrayF32 {
                buf: d_sig,
                rows,
                cols: n,
            },
            hist: DeviceArrayF32 {
                buf: d_hist,
                rows,
                cols: n,
            },
        };
        Ok(CudaModGodModeBatchResult { outputs, combos })
    }

    pub fn mod_god_mode_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        volume_tm: Option<&[f32]>,
        cols: usize,
        rows: usize,
        params: &ModGodModeParams,
    ) -> Result<DeviceArrayF32Triplet, CudaModGodModeError> {
        if cols == 0 || rows == 0 {
            return Err(CudaModGodModeError::InvalidInput(
                "cols/rows must be > 0".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaModGodModeError::InvalidInput("cols*rows overflow".into()))?;
        if high_tm.len() != elems || low_tm.len() != elems || close_tm.len() != elems {
            return Err(CudaModGodModeError::InvalidInput(
                "time-major inputs must be cols*rows".into(),
            ));
        }
        if let Some(v) = volume_tm {
            if v.len() != elems {
                return Err(CudaModGodModeError::InvalidInput(
                    "volume_tm length mismatch".into(),
                ));
            }
        }
        let n1 = params.n1.unwrap_or(17);
        let n2 = params.n2.unwrap_or(6);
        let n3 = params.n3.unwrap_or(4);
        let mode_i = match params.mode.unwrap_or(ModGodModeMode::TraditionMg) {
            ModGodModeMode::Godmode => 0,
            ModGodModeMode::Tradition => 1,
            ModGodModeMode::GodmodeMg => 2,
            ModGodModeMode::TraditionMg => 3,
        };
        if cols == 0 || rows == 0 {
            return Err(CudaModGodModeError::InvalidInput(
                "cols/rows must be > 0".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaModGodModeError::InvalidInput("cols*rows overflow".into()))?;
        if high_tm.len() != elems || low_tm.len() != elems || close_tm.len() != elems {
            return Err(CudaModGodModeError::InvalidInput(
                "time-major inputs must be cols*rows".into(),
            ));
        }
        if let Some(v) = volume_tm {
            if v.len() != elems {
                return Err(CudaModGodModeError::InvalidInput(
                    "volume_tm length mismatch".into(),
                ));
            }
        }
        let n1 = params.n1.unwrap_or(17);
        let n2 = params.n2.unwrap_or(6);
        let n3 = params.n3.unwrap_or(4);
        let mode_i = match params.mode.unwrap_or(ModGodModeMode::TraditionMg) {
            ModGodModeMode::Godmode => 0,
            ModGodModeMode::Tradition => 1,
            ModGodModeMode::GodmodeMg => 2,
            ModGodModeMode::TraditionMg => 3,
        };
        let use_vol = params.use_volume.unwrap_or(false) && volume_tm.is_some();

        let d_close = DeviceBuffer::from_slice(close_tm)?;
        let d_high = if use_vol {
            Some(DeviceBuffer::from_slice(high_tm)?)
        } else {
            None
        };
        let d_low = if use_vol {
            Some(DeviceBuffer::from_slice(low_tm)?)
        } else {
            None
        };
        let d_vol = if let Some(v) = volume_tm {
            Some(DeviceBuffer::from_slice(v)?)
        } else {
            None
        };
        let mut d_wt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };
        let mut d_sig: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };
        let mut d_hist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };

        let func = self
            .module
            .get_function("mod_god_mode_many_series_one_param_time_major_f32")
            .map_err(|_| CudaModGodModeError::MissingKernelSymbol {
                name: "mod_god_mode_many_series_one_param_time_major_f32",
            })?;

        let bx: u32 = 1;
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (bx, 1, 1).into();
        self.validate_launch(grid, block)?;
        if !self.debug_logged && std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[mod_god_mode] many-series policy selected: block_x={} grid.x={}",
                bx, cols
            );
            unsafe {
                (*(self as *const _ as *mut CudaModGodMode)).debug_logged = true;
            }
        }
        unsafe {
            let mut high_ptr = d_high
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut low_ptr = d_low
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut n1_i = n1 as i32;
            let mut n2_i = n2 as i32;
            let mut n3_i = n3 as i32;
            let mut mode_i32 = mode_i as i32;
            let mut use_vol_i = if use_vol { 1i32 } else { 0i32 };
            let mut wt_ptr = d_wt.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut n1_i = n1 as i32;
            let mut n2_i = n2 as i32;
            let mut n3_i = n3 as i32;
            let mut mode_i32 = mode_i as i32;
            let mut use_vol_i = if use_vol { 1i32 } else { 0i32 };
            let mut wt_ptr = d_wt.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut n1_i as *mut _ as *mut c_void,
                &mut n2_i as *mut _ as *mut c_void,
                &mut n3_i as *mut _ as *mut c_void,
                &mut mode_i32 as *mut _ as *mut c_void,
                &mut use_vol_i as *mut _ as *mut c_void,
                &mut wt_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 {
                buf: d_wt,
                rows,
                cols,
            },
            wt2: DeviceArrayF32 {
                buf: d_sig,
                rows,
                cols,
            },
            hist: DeviceArrayF32 {
                buf: d_hist,
                rows,
                cols,
            },
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 500_000;
    const PARAM_SWEEP: usize = 100;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct MgBatchDeviceState {
        cuda: CudaModGodMode,
        len: usize,
        first_valid: usize,
        rows: usize,
        block_x: u32,
        shmem_bytes: u32,
        d_close: DeviceBuffer<f32>,
        d_n1s: DeviceBuffer<i32>,
        d_n2s: DeviceBuffer<i32>,
        d_n3s: DeviceBuffer<i32>,
        d_modes: DeviceBuffer<i32>,
        d_wt: DeviceBuffer<f32>,
        d_sig: DeviceBuffer<f32>,
        d_hist: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MgBatchDeviceState {
        fn launch(&mut self) {
            let func_fast = self
                .cuda
                .module
                .get_function("mod_god_mode_batch_f32_shared_fast")
                .expect("mod_god_mode_batch_f32_shared_fast");

            let grid_x = ((self.rows as u32) + self.block_x - 1) / self.block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();
            self.cuda
                .validate_launch(grid, block)
                .expect("mgm validate launch");

            unsafe {
                let mut high_ptr: u64 = 0;
                let mut low_ptr: u64 = 0;
                let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                let mut vol_ptr: u64 = 0;
                let mut len_i = self.len as i32;
                let mut first_i = self.first_valid as i32;
                let mut rows_i = self.rows as i32;
                let mut n1_ptr = self.d_n1s.as_device_ptr().as_raw();
                let mut n2_ptr = self.d_n2s.as_device_ptr().as_raw();
                let mut n3_ptr = self.d_n3s.as_device_ptr().as_raw();
                let mut modes_ptr = self.d_modes.as_device_ptr().as_raw();
                let mut use_vol_i = 0i32;
                let mut wt_ptr = self.d_wt.as_device_ptr().as_raw();
                let mut sig_ptr = self.d_sig.as_device_ptr().as_raw();
                let mut hist_ptr = self.d_hist.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut n1_ptr as *mut _ as *mut c_void,
                    &mut n2_ptr as *mut _ as *mut c_void,
                    &mut n3_ptr as *mut _ as *mut c_void,
                    &mut modes_ptr as *mut _ as *mut c_void,
                    &mut use_vol_i as *mut _ as *mut c_void,
                    &mut wt_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                    &mut hist_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func_fast, grid, block, self.shmem_bytes, args)
                    .expect("mgm fast launch");
            }
            self.cuda.stream.synchronize().expect("mgm sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaModGodMode::new(0).expect("cuda mgm");
        let close = gen_series(ONE_SERIES_LEN);
        let sweep = ModGodModeBatchRange {
            n1: (10, 10 + PARAM_SWEEP - 1, 1),
            n2: (6, 6, 0),
            n3: (4, 4, 0),
            mode: ModGodModeMode::TraditionMg,
        };

        let rows = CudaModGodMode::expand_range(&sweep)
            .expect("expand_range")
            .len();
        let first_valid = close.iter().position(|v| v.is_finite()).unwrap_or(0);
        let mut n1s: Vec<i32> = Vec::with_capacity(rows);
        let mut n2s: Vec<i32> = Vec::with_capacity(rows);
        let mut n3s: Vec<i32> = Vec::with_capacity(rows);
        let mut modes: Vec<i32> = Vec::with_capacity(rows);
        for i in 0..rows {
            n1s.push((10 + i) as i32);
            n2s.push(6);
            n3s.push(4);
            modes.push(3);
        }

        let d_close = DeviceBuffer::from_slice(&close).expect("d_close");
        let d_n1s = DeviceBuffer::from_slice(&n1s).expect("d_n1s");
        let d_n2s = DeviceBuffer::from_slice(&n2s).expect("d_n2s");
        let d_n3s = DeviceBuffer::from_slice(&n3s).expect("d_n3s");
        let d_modes = DeviceBuffer::from_slice(&modes).expect("d_modes");
        let out_elems = rows * ONE_SERIES_LEN;
        let d_wt: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("d_wt");
        let d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("d_sig");
        let d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("d_hist");

        let mut block_x = CudaModGodMode::fast_block_x();
        let mut shmem_bytes = CudaModGodMode::fast_shared_bytes(block_x);
        let max_dyn_default: usize = 48 * 1024;
        while shmem_bytes > max_dyn_default && block_x > 1 {
            block_x /= 2;
            shmem_bytes = CudaModGodMode::fast_shared_bytes(block_x);
        }
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(MgBatchDeviceState {
            cuda,
            len: ONE_SERIES_LEN,
            first_valid,
            rows,
            block_x,
            shmem_bytes: shmem_bytes as u32,
            d_close,
            d_n1s,
            d_n2s,
            d_n3s,
            d_modes,
            d_wt,
            d_sig,
            d_hist,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "mod_god_mode",
            "one_series_many_params",
            "mod_god_mode_cuda_batch_dev",
            "500k_x_100",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
