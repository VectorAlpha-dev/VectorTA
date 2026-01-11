#![cfg(feature = "cuda")]

//! CUDA wrapper for Negative Volume Index (NVI).
//!
//! Parity with ALMA wrapper policy:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/nvi_kernel.ptx"))
//!   using DetermineTargetFromContext + OptLevel O2, with simpler fallbacks.
//! - NON_BLOCKING stream, simple VRAM guard with typed errors and launch checks.
//! - Public device entry points for:
//!   - Batch (one series × many params): here a single 1×N row (NVI has no params).
//!   - Many-series × one-param (time-major layout).
//! - Decision: CUDA enabled for single-row and many-series paths; kernels keep strict
//!   sequential dependency to match CPU/streaming NVI, and VRAM handles integrate with
//!   Python via CAI v3 + DLPack v1.x.

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::env;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaNviError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaNvi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaNvi {
    pub fn new(device_id: usize) -> Result<Self, CudaNviError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let ctx = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/nvi_kernel.ptx"));
        // Prefer arch from current context + O2 with fallbacks for brittle drivers.
        let primary_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, primary_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) =
                    Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
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
            context: ctx,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaNviError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn first_valid_pair(close: &[f32], volume: &[f32]) -> Result<usize, CudaNviError> {
        if close.is_empty() || volume.is_empty() {
            return Err(CudaNviError::InvalidInput("empty inputs".into()));
        }
        if close.len() != volume.len() {
            return Err(CudaNviError::InvalidInput("length mismatch".into()));
        }
        let first = close
            .iter()
            .zip(volume.iter())
            .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
            .ok_or_else(|| {
                CudaNviError::InvalidInput("all values are NaN in one/both inputs".into())
            })?;
        if close.len() - first < 2 {
            return Err(CudaNviError::InvalidInput(
                "not enough valid data (need >= 2 after first valid)".into(),
            ));
        }
        Ok(first)
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
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
    fn validate_launch_dims(&self, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Result<(), CudaNviError> {
        let dev = Device::get_device(self.device_id)?;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaNviError::InvalidInput("zero-sized grid or block".into()));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz
        {
            return Err(CudaNviError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    pub fn nvi_batch_dev(
        &self,
        close: &[f32],
        volume: &[f32],
    ) -> Result<DeviceArrayF32, CudaNviError> {
        let first = Self::first_valid_pair(close, volume)?;
        let len = close.len();

        // VRAM estimate: 2 inputs + 1 output
        let elem = std::mem::size_of::<f32>();
        let elems = 3usize
            .checked_mul(len)
            .ok_or_else(|| CudaNviError::InvalidInput("len overflow in VRAM estimate".into()))?;
        let bytes = elems
            .checked_mul(elem)
            .ok_or_else(|| CudaNviError::InvalidInput("len*elem_size overflow".into()))?;
        let headroom = 64usize << 20;
        if !Self::will_fit(bytes, headroom) {
            let (free, _) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaNviError::OutOfMemory {
                required: bytes,
                free,
                headroom,
            });
        }

        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;

        // Launch one-warp kernel (warp-scan over 32-step tiles)
        let func = self
            .module
            .get_function("nvi_batch_f32")
            .map_err(|_| CudaNviError::MissingKernelSymbol { name: "nvi_batch_f32" })?;
        let grid_dims = (1u32, 1u32, 1u32);
        let block_dims = (16u32, 1u32, 1u32);
        self.validate_launch_dims(grid_dims, block_dims)?;
        let grid: GridSize = grid_dims.into();
        let block: BlockSize = block_dims.into();
        unsafe {
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    pub fn nvi_many_series_one_param_time_major_dev(
        &self,
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaNviError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNviError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaNviError::InvalidInput("rows*cols overflow".into()))?;
        if close_tm.len() != expected || volume_tm.len() != expected {
            return Err(CudaNviError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }

        // First-valid per series (host) — row-major scan for cache locality.
        let rows_i32 = rows as i32;
        let mut first_valids = vec![rows_i32; cols];
        let mut remaining = cols;

        'outer: for t in 0..rows {
            let row_off = t * cols;
            for s in 0..cols {
                if first_valids[s] == rows_i32 {
                    let c = close_tm[row_off + s];
                    let v = volume_tm[row_off + s];
                    if !c.is_nan() && !v.is_nan() {
                        first_valids[s] = t as i32;
                        remaining -= 1;
                        if remaining == 0 {
                            break 'outer;
                        }
                    }
                }
            }
        }

        // Require at least 2 valid samples per series for a non-trivial scan
        for s in 0..cols {
            if (rows_i32 - first_valids[s]) < 2 {
                return Err(CudaNviError::InvalidInput(format!(
                    "series {}: not enough valid data (need >= 2 after first valid)",
                    s
                )));
            }
        }

        // VRAM estimate: 2 inputs + 1 output + first_valids
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let elems_main = 3usize
            .checked_mul(expected)
            .ok_or_else(|| {
                CudaNviError::InvalidInput("expected overflow in VRAM estimate".into())
            })?;
        let bytes_main = elems_main
            .checked_mul(elem_f32)
            .ok_or_else(|| {
                CudaNviError::InvalidInput("expected*elem_f32 overflow in VRAM estimate".into())
            })?;
        let bytes_first = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| {
                CudaNviError::InvalidInput("cols*elem_i32 overflow in VRAM estimate".into())
            })?;
        let bytes = bytes_main
            .checked_add(bytes_first)
            .ok_or_else(|| {
                CudaNviError::InvalidInput("total VRAM bytes overflow".into())
            })?;
        let headroom = 64usize << 20;
        if !Self::will_fit(bytes, headroom) {
            let (free, _) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaNviError::OutOfMemory {
                required: bytes,
                free,
                headroom,
            });
        }

        let d_close = DeviceBuffer::from_slice(close_tm)?;
        let d_volume = DeviceBuffer::from_slice(volume_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }?;

        let func = self
            .module
            .get_function("nvi_many_series_one_param_f32")
            .map_err(|_| CudaNviError::MissingKernelSymbol {
                name: "nvi_many_series_one_param_f32",
            })?;

        // One thread per series
        let block_x: u32 = 256;
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid_dims = (grid_x.max(1), 1u32, 1u32);
        let block_dims = (block_x, 1u32, 1u32);
        self.validate_launch_dims(grid_dims, block_dims)?;
        let grid: GridSize = grid_dims.into();
        let block: BlockSize = block_dims.into();
        unsafe {
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    /// Launch NVI (one series) using device-resident buffers in-place.
    /// The kernel is enqueued on `self.stream` and this method returns immediately.
    /// Callers must ensure device buffers outlive the enqueued work and synchronize as needed.
    pub fn nvi_batch_dev_inplace(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_volume: &DeviceBuffer<f32>,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNviError> {
        let len = d_close.len();
        if len == 0 {
            return Err(CudaNviError::InvalidInput("empty inputs".into()));
        }
        if d_volume.len() != len || d_out.len() != len {
            return Err(CudaNviError::InvalidInput("length mismatch".into()));
        }

        let func = self
            .module
            .get_function("nvi_batch_f32")
            .map_err(|_| CudaNviError::MissingKernelSymbol { name: "nvi_batch_f32" })?;
        let grid_dims = (1u32, 1u32, 1u32);
        let block_dims = (16u32, 1u32, 1u32);
        self.validate_launch_dims(grid_dims, block_dims)?;
        let grid: GridSize = grid_dims.into();
        let block: BlockSize = block_dims.into();
        unsafe {
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = (first_valid as i32).max(0);
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 5] = [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        Ok(())
    }

    /// Launch NVI (many series, time-major) using device-resident buffers in-place.
    /// `d_first_valids` must have length == cols and live on device.
    /// The kernel is enqueued on `self.stream`; synchronize externally when needed.
    pub fn nvi_many_series_one_param_time_major_dev_inplace(
        &self,
        d_close_tm: &DeviceBuffer<f32>,
        d_volume_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNviError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNviError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaNviError::InvalidInput("rows*cols overflow".into()))?;
        if d_close_tm.len() != expected
            || d_volume_tm.len() != expected
            || d_out_tm.len() != expected
            || d_first_valids.len() != cols
        {
            return Err(CudaNviError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }

        let func = self
            .module
            .get_function("nvi_many_series_one_param_f32")
            .map_err(|_| CudaNviError::MissingKernelSymbol {
                name: "nvi_many_series_one_param_f32",
            })?;

        // One thread per series; kernel grid-strides over s.
        let block_x: u32 = 256;
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid_dims = (grid_x.max(1), 1u32, 1u32);
        let block_dims = (block_x, 1u32, 1u32);
        self.validate_launch_dims(grid_dims, block_dims)?;
        let grid: GridSize = grid_dims.into();
        let block: BlockSize = block_dims.into();

        unsafe {
            let mut close_ptr = d_close_tm.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        Ok(())
    }
}

// ---------------- Bench profiles ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 512;
    const MANY_SERIES_ROWS: usize = 8_192;

    fn bytes_one_series() -> usize {
        // 2 inputs + 1 output + ~64MB headroom
        (3 * ONE_SERIES_LEN * std::mem::size_of::<f32>()) + (64 << 20)
    }
    fn bytes_many_series() -> usize {
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        (3 * n * std::mem::size_of::<f32>())
            + (MANY_SERIES_COLS * std::mem::size_of::<i32>())
            + (64 << 20)
    }

    struct NviOneSeriesState {
        cuda: CudaNvi,
        d_close: DeviceBuffer<f32>,
        d_volume: DeviceBuffer<f32>,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for NviOneSeriesState {
        fn launch(&mut self) {
            self.cuda
                .nvi_batch_dev_inplace(
                    &self.d_close,
                    &self.d_volume,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("nvi one-series");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaNvi::new(0).expect("cuda nvi");
        let mut close = gen_series(ONE_SERIES_LEN);
        let mut volume = gen_series(ONE_SERIES_LEN);
        // Ensure a valid warmup starts at 0
        if close[0].is_nan() || volume[0].is_nan() {
            close[0] = 100.0;
            volume[0] = 1000.0;
        }
        let first_valid = CudaNvi::first_valid_pair(&close, &volume).expect("first_valid");
        let d_close = DeviceBuffer::from_slice(&close).expect("d_close");
        let d_volume = DeviceBuffer::from_slice(&volume).expect("d_volume");
        let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(ONE_SERIES_LEN) }
            .expect("d_out");
        Box::new(NviOneSeriesState {
            cuda,
            d_close,
            d_volume,
            first_valid,
            d_out,
        })
    }

    struct NviManySeriesState {
        cuda: CudaNvi,
        d_close_tm: DeviceBuffer<f32>,
        d_volume_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        d_first_valids: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for NviManySeriesState {
        fn launch(&mut self) {
            self.cuda
                .nvi_many_series_one_param_time_major_dev_inplace(
                    &self.d_close_tm,
                    &self.d_volume_tm,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("nvi many-series");
            self.cuda.synchronize().expect("nvi many-series sync");
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaNvi::new(0).expect("cuda nvi");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let n = cols * rows;
        let mut close_tm = vec![f32::NAN; n];
        let mut volume_tm = vec![f32::NAN; n];
        for s in 0..cols {
            // Stagger warmups per series
            for t in s.min(8)..rows {
                let x = (t as f32) + (s as f32) * 0.11;
                close_tm[t * cols + s] = (x * 0.0021).sin() + 0.0002 * x + 100.0;
                volume_tm[t * cols + s] = (x * 0.0017).cos().abs() * 500.0 + 100.0;
            }
        }

        // First-valid per series (host) - row-major scan for cache locality.
        let rows_i32 = rows as i32;
        let mut first_valids = vec![rows_i32; cols];
        let mut remaining = cols;
        'outer: for t in 0..rows {
            let row_off = t * cols;
            for s in 0..cols {
                if first_valids[s] == rows_i32 {
                    let c = close_tm[row_off + s];
                    let v = volume_tm[row_off + s];
                    if !c.is_nan() && !v.is_nan() {
                        first_valids[s] = t as i32;
                        remaining -= 1;
                        if remaining == 0 {
                            break 'outer;
                        }
                    }
                }
            }
        }

        let d_close_tm = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
        let d_volume_tm = DeviceBuffer::from_slice(&volume_tm).expect("d_volume_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }.expect("d_out_tm");
        cuda.synchronize().expect("nvi many prep sync");
        Box::new(NviManySeriesState {
            cuda,
            d_close_tm,
            d_volume_tm,
            cols,
            rows,
            d_first_valids,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("nvi", "nvi", "nvi_cuda_one_series", "1m", prep_one_series)
                .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "nvi",
                "nvi",
                "nvi_cuda_many_series_time_major",
                "512x8192",
                prep_many_series,
            )
            .with_mem_required(bytes_many_series()),
        ]
    }
}
