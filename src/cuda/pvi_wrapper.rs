#![cfg(feature = "cuda")]

//! CUDA wrapper for Positive Volume Index (PVI).
//!
//! Parity with ALMA wrapper policy:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/pvi_kernel.ptx"))
//!   using DetermineTargetFromContext + OptLevel O2, with simpler fallbacks.
//! - NON_BLOCKING stream, VRAM estimation + ~64MB headroom.
//! - Public device entry points:
//!   - Batch (one series × many params): many initial values for a single series.
//!   - Many-series × one-param (time-major): shared initial value, per-series warmup.

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
pub enum CudaPviError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaPviError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "CUDA error: {}", e),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl Error for CudaPviError {}

pub struct CudaPvi {
    module: Module,
    stream: Stream,
    _ctx: Context,
}

impl CudaPvi {
    pub fn new(device_id: usize) -> Result<Self, CudaPviError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pvi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _ctx: ctx,
        })
    }

    #[inline]
    fn first_valid_pair(close: &[f32], volume: &[f32]) -> Result<usize, CudaPviError> {
        if close.is_empty() || volume.is_empty() {
            return Err(CudaPviError::InvalidInput("empty inputs".into()));
        }
        if close.len() != volume.len() {
            return Err(CudaPviError::InvalidInput("length mismatch".into()));
        }
        let first = close
            .iter()
            .zip(volume.iter())
            .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
            .ok_or_else(|| {
                CudaPviError::InvalidInput("all values are NaN in one/both inputs".into())
            })?;
        if close.len() - first < 2 {
            return Err(CudaPviError::InvalidInput(
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

    /// One series × many params (initial values).
    pub fn pvi_batch_dev(
        &self,
        close: &[f32],
        volume: &[f32],
        initial_values: &[f32], // rows
    ) -> Result<DeviceArrayF32, CudaPviError> {
        let first = Self::first_valid_pair(close, volume)?;
        let len = close.len();
        let rows = initial_values.len();
        if rows == 0 {
            return Err(CudaPviError::InvalidInput(
                "no initial values provided".into(),
            ));
        }

        // VRAM estimate: close + volume + scale + initial_values + out
        let bytes = (2 * len + len + rows + rows * len) * std::mem::size_of::<f32>();
        if !Self::will_fit(bytes, 64 << 20) {
            return Err(CudaPviError::Cuda("insufficient free VRAM".into()));
        }

        // Upload inputs
        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let d_volume =
            DeviceBuffer::from_slice(volume).map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let d_inits = DeviceBuffer::from_slice(initial_values)
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let mut d_scale: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;

        // Build scale
        let build = self
            .module
            .get_function("pvi_build_scale_f32")
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        unsafe {
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first as i32;
            let mut scale_ptr = d_scale.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut scale_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (1, 1, 1).into();
            let block: BlockSize = (1, 1, 1).into();
            self.stream
                .launch(&build, grid, block, 0, args)
                .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        }

        // Apply scale across rows
        let apply = self
            .module
            .get_function("pvi_apply_scale_batch_f32")
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        unsafe {
            let mut scale_ptr = d_scale.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first as i32;
            let mut inits_ptr = d_inits.as_device_ptr().as_raw();
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut scale_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut inits_ptr as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let block_x: u32 = 256;
            let grid_x: u32 = ((len as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.stream
                .launch(&apply, grid, block, 0, args)
                .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: len,
        })
    }

    /// Many series × one initial value (time-major layout)
    pub fn pvi_many_series_one_param_time_major_dev(
        &self,
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
        initial_value: f32,
    ) -> Result<DeviceArrayF32, CudaPviError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPviError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaPviError::InvalidInput("rows*cols overflow".into()))?;
        if close_tm.len() != expected || volume_tm.len() != expected {
            return Err(CudaPviError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }

        // First-valid per series (host)
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let c = close_tm[t * cols + s];
                let v = volume_tm[t * cols + s];
                if !c.is_nan() && !v.is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            if (rows as i32 - first_valids[s]) < 2 {
                return Err(CudaPviError::InvalidInput(format!(
                    "series {}: not enough valid data (need >= 2 after first valid)",
                    s
                )));
            }
        }

        // VRAM estimate: 2 inputs + 1 output + first_valids
        let bytes = (3 * expected) * std::mem::size_of::<f32>() + cols * std::mem::size_of::<i32>();
        if !Self::will_fit(bytes, 64 << 20) {
            return Err(CudaPviError::Cuda("insufficient free VRAM".into()));
        }

        let d_close =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let d_volume =
            DeviceBuffer::from_slice(volume_tm).map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("pvi_many_series_one_param_f32")
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        unsafe {
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut init_f = initial_value;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut init_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let block_x: u32 = 256;
            let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaPviError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
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

    fn bytes_one_series(rows: usize) -> usize {
        // close + volume + scale + init_values + out + ~64MB
        (2 * ONE_SERIES_LEN + ONE_SERIES_LEN + rows + rows * ONE_SERIES_LEN)
            * std::mem::size_of::<f32>()
            + (64 << 20)
    }
    fn bytes_many_series() -> usize {
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        (3 * n * std::mem::size_of::<f32>()) + (64 << 20)
    }

    struct PviOneSeriesState {
        cuda: CudaPvi,
        close: Vec<f32>,
        volume: Vec<f32>,
        inits: Vec<f32>,
    }
    impl CudaBenchState for PviOneSeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .pvi_batch_dev(&self.close, &self.volume, &self.inits)
                .expect("pvi one-series");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaPvi::new(0).expect("cuda pvi");
        let mut close = gen_series(ONE_SERIES_LEN);
        let mut volume = gen_series(ONE_SERIES_LEN);
        // Ensure warm starts at 0
        if close[0].is_nan() || volume[0].is_nan() {
            close[0] = 100.0;
            volume[0] = 1000.0;
        }
        // Try 64 rows of different initial values
        let mut inits = vec![0f32; 64];
        for i in 0..inits.len() {
            inits[i] = 500.0 + (i as f32) * 25.0;
        }
        Box::new(PviOneSeriesState {
            cuda,
            close,
            volume,
            inits,
        })
    }

    struct PviManySeriesState {
        cuda: CudaPvi,
        close_tm: Vec<f32>,
        volume_tm: Vec<f32>,
    }
    impl CudaBenchState for PviManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .pvi_many_series_one_param_time_major_dev(
                    &self.close_tm,
                    &self.volume_tm,
                    MANY_SERIES_COLS,
                    MANY_SERIES_ROWS,
                    1000.0,
                )
                .expect("pvi many-series");
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaPvi::new(0).expect("cuda pvi");
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let mut close_tm = vec![f32::NAN; n];
        let mut volume_tm = vec![f32::NAN; n];
        for s in 0..MANY_SERIES_COLS {
            for t in s.min(8)..MANY_SERIES_ROWS {
                let x = (t as f32) + (s as f32) * 0.11;
                close_tm[t * MANY_SERIES_COLS + s] = (x * 0.0021).sin() + 0.0002 * x + 100.0;
                volume_tm[t * MANY_SERIES_COLS + s] = (x * 0.0017).cos().abs() * 500.0 + 100.0;
            }
        }
        Box::new(PviManySeriesState {
            cuda,
            close_tm,
            volume_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "pvi",
                "pvi",
                "pvi_cuda_one_series",
                "1m x 64",
                prep_one_series,
            )
            .with_mem_required(bytes_one_series(64)),
            CudaBenchScenario::new(
                "pvi",
                "pvi",
                "pvi_cuda_many_series_time_major",
                "512x8192",
                prep_many_series,
            )
            .with_mem_required(bytes_many_series()),
        ]
    }
}
