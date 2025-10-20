#![cfg(feature = "cuda")]

//! CUDA wrapper for Volume Price Trend (VPT).
//!
//! Parity with ALMA wrapper policy:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/vpt_kernel.ptx")) with
//!   DetermineTargetFromContext + OptLevel O2, falling back to simpler JIT options.
//! - NON_BLOCKING stream.
//! - VRAM estimation with ~64MB headroom; simple chunking not required as grids are small.
//! - Public device entry points:
//!   - `vpt_batch_dev(&[f32], &[f32]) -> DeviceArrayF32` (one series; single row)
//!   - `vpt_many_series_one_param_time_major_dev(&[f32], &[f32], cols, rows) -> DeviceArrayF32`

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
pub enum CudaVptError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaVptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "CUDA error: {}", e),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl Error for CudaVptError {}

pub struct CudaVpt {
    module: Module,
    stream: Stream,
    _ctx: Context,
}

impl CudaVpt {
    pub fn new(device_id: usize) -> Result<Self, CudaVptError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vpt_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _ctx: ctx,
        })
    }

    #[inline]
    fn first_valid_pair(price: &[f32], volume: &[f32]) -> Result<usize, CudaVptError> {
        if price.is_empty() || volume.is_empty() {
            return Err(CudaVptError::InvalidInput("empty inputs".into()));
        }
        if price.len() != volume.len() {
            return Err(CudaVptError::InvalidInput("length mismatch".into()));
        }
        for i in 1..price.len() {
            let p0 = price[i - 1];
            let p1 = price[i];
            let v1 = volume[i];
            if p0.is_finite() && p0 != 0.0 && p1.is_finite() && v1.is_finite() {
                return Ok(i);
            }
        }
        Err(CudaVptError::InvalidInput(
            "not enough valid data (need a valid pair i-1,i)".into(),
        ))
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    /// One series (single row). Writes warmup NaNs to match scalar semantics.
    pub fn vpt_batch_dev(
        &self,
        price: &[f32],
        volume: &[f32],
    ) -> Result<DeviceArrayF32, CudaVptError> {
        let len = price.len().min(volume.len());
        if len == 0 {
            return Err(CudaVptError::InvalidInput("empty input".into()));
        }
        if price.len() != volume.len() {
            return Err(CudaVptError::InvalidInput("length mismatch".into()));
        }

        let first = Self::first_valid_pair(price, volume)?;

        // VRAM: inputs + output
        let bytes = (2 * len + len) * std::mem::size_of::<f32>();
        if !Self::will_fit(bytes, 64 << 20) {
            return Err(CudaVptError::Cuda("insufficient free VRAM".into()));
        }

        let d_price =
            DeviceBuffer::from_slice(price).map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let d_volume =
            DeviceBuffer::from_slice(volume).map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("vpt_batch_f32")
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        unsafe {
            let mut price_ptr = d_price.as_device_ptr().as_raw();
            let mut volume_ptr = d_volume.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut price_ptr as *mut _ as *mut c_void,
                &mut volume_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (1, 1, 1).into();
            let block: BlockSize = (1, 1, 1).into();
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    /// Many-series × one-param (no real params for VPT). Time-major layout.
    pub fn vpt_many_series_one_param_time_major_dev(
        &self,
        price_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaVptError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVptError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaVptError::InvalidInput("rows*cols overflow".into()))?;
        if price_tm.len() != expected || volume_tm.len() != expected {
            return Err(CudaVptError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }

        // First-valid per series (host): earliest i>=1 satisfying pair constraints
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 1..rows {
                let p0 = price_tm[(t - 1) * cols + s];
                let p1 = price_tm[t * cols + s];
                let v1 = volume_tm[t * cols + s];
                if p0.is_finite() && p0 != 0.0 && p1.is_finite() && v1.is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            if (rows as i32 - first_valids[s]) < 2 {
                return Err(CudaVptError::InvalidInput(format!(
                    "series {}: not enough valid data (need >= 2 after first valid pair)",
                    s
                )));
            }
        }

        // VRAM: 2 inputs + first_valids + output
        let bytes = (3 * expected * std::mem::size_of::<f32>())
            .saturating_add(cols * std::mem::size_of::<i32>())
            + (64 << 20);
        if !Self::will_fit(bytes, 0) {
            return Err(CudaVptError::Cuda("insufficient free VRAM".into()));
        }

        let d_price =
            DeviceBuffer::from_slice(price_tm).map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let d_volume =
            DeviceBuffer::from_slice(volume_tm).map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("vpt_many_series_one_param_f32")
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        unsafe {
            let mut price_ptr = d_price.as_device_ptr().as_raw();
            let mut volume_ptr = d_volume.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut price_ptr as *mut _ as *mut c_void,
                &mut volume_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let block_x: u32 = 256;
            let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaVptError::Cuda(e.to_string()))?;
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

    fn bytes_one_series() -> usize {
        // price + volume + out + ~64MB
        (3 * ONE_SERIES_LEN * std::mem::size_of::<f32>()) + (64 << 20)
    }
    fn bytes_many_series() -> usize {
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        (3 * n * std::mem::size_of::<f32>())
            + (MANY_SERIES_COLS * std::mem::size_of::<i32>())
            + (64 << 20)
    }

    struct OneSeriesState {
        cuda: CudaVpt,
        price: Vec<f32>,
        volume: Vec<f32>,
    }
    impl CudaBenchState for OneSeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .vpt_batch_dev(&self.price, &self.volume)
                .expect("vpt one-series");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaVpt::new(0).expect("cuda vpt");
        let mut price = gen_series(ONE_SERIES_LEN);
        let mut volume = gen_series(ONE_SERIES_LEN);
        // Ensure there is a valid pair after index 0
        if !price[1].is_finite() || price[0] == 0.0 || !volume[1].is_finite() {
            price[0] = 100.0;
            price[1] = 100.1;
            volume[1] = 500.0;
        }
        Box::new(OneSeriesState {
            cuda,
            price,
            volume,
        })
    }

    struct ManySeriesState {
        cuda: CudaVpt,
        price_tm: Vec<f32>,
        volume_tm: Vec<f32>,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .vpt_many_series_one_param_time_major_dev(
                    &self.price_tm,
                    &self.volume_tm,
                    MANY_SERIES_COLS,
                    MANY_SERIES_ROWS,
                )
                .expect("vpt many-series");
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaVpt::new(0).expect("cuda vpt");
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let mut price_tm = vec![f32::NAN; n];
        let mut volume_tm = vec![f32::NAN; n];
        for s in 0..MANY_SERIES_COLS {
            for t in s.min(8)..MANY_SERIES_ROWS {
                let x = (t as f32) + (s as f32) * 0.13;
                price_tm[t * MANY_SERIES_COLS + s] = (x * 0.0021).sin() + 0.0002 * x + 100.0;
                volume_tm[t * MANY_SERIES_COLS + s] = (x * 0.0017).cos().abs() * 500.0 + 100.0;
            }
        }
        Box::new(ManySeriesState {
            cuda,
            price_tm,
            volume_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "vpt",
                "one_series",
                "vpt_cuda_one_series",
                "1m",
                prep_one_series,
            )
            .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "vpt",
                "many_series_one_param",
                "vpt_cuda_many_series_time_major",
                "512x8192",
                prep_many_series,
            )
            .with_mem_required(bytes_many_series()),
        ]
    }
}
