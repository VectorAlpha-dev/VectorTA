#![cfg(feature = "cuda")]

//! CUDA wrapper for Negative Volume Index (NVI).
//!
//! Parity with ALMA wrapper policy:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/nvi_kernel.ptx"))
//!   using DetermineTargetFromContext + OptLevel O4, with simpler fallbacks.
//! - NON_BLOCKING stream, simple VRAM guard.
//! - Public device entry points for:
//!   - Batch (one series × many params): here a single 1×N row (NVI has no params).
//!   - Many-series × one-param (time-major layout).

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::error::Error;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaNviError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaNviError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "CUDA error: {}", e),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl Error for CudaNviError {}

pub struct CudaNvi {
    module: Module,
    stream: Stream,
    _ctx: Context,
}

impl CudaNvi {
    pub fn new(device_id: usize) -> Result<Self, CudaNviError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/nvi_kernel.ptx"));
        // Prefer arch from current context + O4 (driver's most optimized level).
        let primary_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, primary_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _ctx: ctx,
        })
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
        if let Ok((free, _)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    pub fn nvi_batch_dev(
        &self,
        close: &[f32],
        volume: &[f32],
    ) -> Result<DeviceArrayF32, CudaNviError> {
        let first = Self::first_valid_pair(close, volume)?;
        let len = close.len();

        // VRAM estimate: 2 inputs + 1 output
        let bytes = (2 * len + len) * std::mem::size_of::<f32>();
        if !Self::will_fit(bytes, 64 << 20) {
            return Err(CudaNviError::Cuda("insufficient free VRAM".into()));
        }

        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let d_volume =
            DeviceBuffer::from_slice(volume).map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;

        // Launch single-thread kernel (sequential scan)
        let func = self
            .module
            .get_function("nvi_batch_f32")
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
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
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;

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
        let bytes = (3 * expected) * std::mem::size_of::<f32>() + cols * std::mem::size_of::<i32>();
        if !Self::will_fit(bytes, 64 << 20) {
            return Err(CudaNviError::Cuda("insufficient free VRAM".into()));
        }

        let d_close =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let d_volume =
            DeviceBuffer::from_slice(volume_tm).map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("nvi_many_series_one_param_f32")
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;

        // One thread per series
        let block_x: u32 = 256;
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;

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
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
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
            .map_err(|e| CudaNviError::Cuda(e.to_string()))?;

        // One thread per series; kernel grid-strides over s.
        let block_x: u32 = 256;
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaNviError::Cuda(e.to_string()))?;
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
        (3 * n * std::mem::size_of::<f32>()) + (64 << 20)
    }

    struct NviOneSeriesState {
        cuda: CudaNvi,
        close: Vec<f32>,
        volume: Vec<f32>,
    }
    impl CudaBenchState for NviOneSeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .nvi_batch_dev(&self.close, &self.volume)
                .expect("nvi one-series");
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
        Box::new(NviOneSeriesState {
            cuda,
            close,
            volume,
        })
    }

    struct NviManySeriesState {
        cuda: CudaNvi,
        close_tm: Vec<f32>,
        volume_tm: Vec<f32>,
    }
    impl CudaBenchState for NviManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .nvi_many_series_one_param_time_major_dev(
                    &self.close_tm,
                    &self.volume_tm,
                    MANY_SERIES_COLS,
                    MANY_SERIES_ROWS,
                )
                .expect("nvi many-series");
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaNvi::new(0).expect("cuda nvi");
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let mut close_tm = vec![f32::NAN; n];
        let mut volume_tm = vec![f32::NAN; n];
        for s in 0..MANY_SERIES_COLS {
            // Stagger warmups per series
            for t in s.min(8)..MANY_SERIES_ROWS {
                let x = (t as f32) + (s as f32) * 0.11;
                close_tm[t * MANY_SERIES_COLS + s] = (x * 0.0021).sin() + 0.0002 * x + 100.0;
                volume_tm[t * MANY_SERIES_COLS + s] = (x * 0.0017).cos().abs() * 500.0 + 100.0;
            }
        }
        Box::new(NviManySeriesState {
            cuda,
            close_tm,
            volume_tm,
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
