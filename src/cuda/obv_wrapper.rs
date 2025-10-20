#![cfg(feature = "cuda")]

//! CUDA scaffolding for the OBV (On-Balance Volume) indicator.
//!
//! Parity goals with ALMA wrapper:
//! - PTX load with DetermineTargetFromContext + OptLevel O2 (and graceful fallback)
//! - Non-blocking stream
//! - VRAM checks with headroom and grid.y chunking
//! - Public device entry points for batch and many-series (time-major)
//! - Bench profiles exposed via `benches::bench_profiles()`

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub enum ObvBatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ObvManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

impl Default for ObvBatchKernelPolicy {
    fn default() -> Self {
        ObvBatchKernelPolicy::Auto
    }
}
impl Default for ObvManySeriesKernelPolicy {
    fn default() -> Self {
        ObvManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaObvPolicy {
    pub batch: ObvBatchKernelPolicy,
    pub many_series: ObvManySeriesKernelPolicy,
}

#[derive(Debug)]
pub enum CudaObvError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaObvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaObvError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaObvError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaObvError {}

pub struct CudaObv {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaObvPolicy,
}

impl CudaObv {
    pub fn new(device_id: usize) -> Result<Self, CudaObvError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaObvError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaObvError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaObvError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/obv_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaObvError::Cuda(e.to_string()))?
                }
            }
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaObvError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaObvPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, policy: CudaObvPolicy) {
        self.policy = policy;
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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // ---------- Public API: one-series × many-params (OBV: single row) ----------

    pub fn obv_batch_dev(
        &self,
        close: &[f32],
        volume: &[f32],
    ) -> Result<DeviceArrayF32, CudaObvError> {
        if close.is_empty() || volume.is_empty() {
            return Err(CudaObvError::InvalidInput("empty input".into()));
        }
        if close.len() != volume.len() {
            return Err(CudaObvError::InvalidInput(
                "mismatched input lengths".into(),
            ));
        }
        let series_len = close.len();
        let first_valid = (0..series_len)
            .find(|&i| !close[i].is_nan() && !volume[i].is_nan())
            .ok_or_else(|| CudaObvError::InvalidInput("all values are NaN".into()))?;

        // VRAM estimate: 2 inputs + 1 output
        let bytes = (close.len() + volume.len() + series_len) * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaObvError::InvalidInput(
                "insufficient device memory for OBV batch".into(),
            ));
        }

        // H2D copies
        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaObvError::Cuda(e.to_string()))?;
        let d_volume =
            DeviceBuffer::from_slice(volume).map_err(|e| CudaObvError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaObvError::Cuda(e.to_string()))?;

        self.launch_obv_batch(&d_close, &d_volume, series_len, 1, first_valid, &mut d_out)?;

        self.stream
            .synchronize()
            .map_err(|e| CudaObvError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: series_len,
        })
    }

    fn launch_obv_batch(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_volume: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaObvError> {
        let func = self
            .module
            .get_function("obv_batch_f32")
            .map_err(|e| CudaObvError::Cuda(e.to_string()))?;

        let block_x = match self.policy.batch {
            ObvBatchKernelPolicy::Plain { block_x } => block_x,
            ObvBatchKernelPolicy::Auto => env::var("OBV_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| matches!(v, 128 | 256 | 512))
                .unwrap_or(256),
        };
        // grid.x sized to cover warmup prefix writes; serial tail is done by tid=0
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;

        // Chunk grid.y for safety (though n_combos is typically 1)
        let mut remaining = n_combos;
        let mut start = 0usize;
        while remaining > 0 {
            let chunk = remaining.min(65_535);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            // cust launch! macro requires an identifier for the stream
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_close.as_device_ptr(),
                        d_volume.as_device_ptr(),
                        series_len as i32,
                        chunk as i32, // n_combos in this slice
                        first_valid as i32,
                        d_out.as_device_ptr().add(start * series_len)
                    )
                )
                .map_err(|e| CudaObvError::Cuda(e.to_string()))?;
            }
            start += chunk;
            remaining -= chunk;
        }
        Ok(())
    }

    // ---------- Public API: many-series × one-param (time-major) ----------

    pub fn obv_many_series_one_param_time_major_dev(
        &self,
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaObvError> {
        if cols == 0 || rows == 0 {
            return Err(CudaObvError::InvalidInput("empty dims".into()));
        }
        if close_tm.len() != volume_tm.len() || close_tm.len() != cols * rows {
            return Err(CudaObvError::InvalidInput(
                "mismatched input sizes for time-major matrix".into(),
            ));
        }
        // Compute per-series first_valid
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let idx = t * cols + s;
                if !close_tm[idx].is_nan() && !volume_tm[idx].is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv == rows as i32 {
                return Err(CudaObvError::InvalidInput(format!(
                    "series {}: all values are NaN",
                    s
                )));
            }
            first_valids[s] = fv;
        }

        // VRAM estimate: 2 inputs + first_valids + 1 output
        let bytes =
            (close_tm.len() + volume_tm.len() + cols + close_tm.len()) * std::mem::size_of::<f32>();
        let bytes = bytes + cols * std::mem::size_of::<i32>();
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaObvError::InvalidInput(
                "insufficient device memory for OBV many-series".into(),
            ));
        }

        let d_close =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaObvError::Cuda(e.to_string()))?;
        let d_volume =
            DeviceBuffer::from_slice(volume_tm).map_err(|e| CudaObvError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaObvError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaObvError::Cuda(e.to_string()))?;

        self.launch_obv_many_series_tm(&d_close, &d_volume, &d_first, cols, rows, &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaObvError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_obv_many_series_tm(
        &self,
        d_close_tm: &DeviceBuffer<f32>,
        d_volume_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaObvError> {
        let func = self
            .module
            .get_function("obv_many_series_one_param_time_major_f32")
            .map_err(|e| CudaObvError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ObvManySeriesKernelPolicy::OneD { block_x } => block_x,
            ObvManySeriesKernelPolicy::Auto => env::var("OBV_MS_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| matches!(v, 128 | 256 | 512))
                .unwrap_or(256),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut c_ptr = d_close_tm.as_device_ptr().as_raw();
            let mut v_ptr = d_volume_tm.as_device_ptr().as_raw();
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut c_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaObvError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_ROWS: usize = 200_000;
    const MANY_COLS: usize = 128;

    fn bytes_one_series() -> usize {
        // 2 inputs + 1 output
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 32 * 1024 * 1024
    }

    fn bytes_many_series() -> usize {
        let elems = MANY_ROWS * MANY_COLS;
        // 2 inputs + 1 output + first_valids
        (2 * elems + elems) * std::mem::size_of::<f32>()
            + MANY_COLS * std::mem::size_of::<i32>()
            + 32 * 1024 * 1024
    }

    fn synth_volume_from_price(close: &[f32]) -> Vec<f32> {
        let mut v = vec![0f32; close.len()];
        for i in 0..close.len() {
            let x = i as f32 * 0.00077;
            v[i] = (x.cos().abs() + 0.5) * 1000.0;
        }
        v
    }

    struct ObvBatchState {
        cuda: CudaObv,
        close: Vec<f32>,
        volume: Vec<f32>,
    }
    impl CudaBenchState for ObvBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .obv_batch_dev(&self.close, &self.volume)
                .expect("obv batch");
        }
    }

    struct ObvManySeriesState {
        cuda: CudaObv,
        close_tm: Vec<f32>,
        volume_tm: Vec<f32>,
    }
    impl CudaBenchState for ObvManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .obv_many_series_one_param_time_major_dev(
                    &self.close_tm,
                    &self.volume_tm,
                    MANY_COLS,
                    MANY_ROWS,
                )
                .expect("obv many-series");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaObv::new(0).expect("cuda obv");
        let close = gen_series(ONE_SERIES_LEN);
        let volume = synth_volume_from_price(&close);
        Box::new(ObvBatchState {
            cuda,
            close,
            volume,
        })
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaObv::new(0).expect("cuda obv");
        let mut close_tm = vec![f32::NAN; MANY_COLS * MANY_ROWS];
        let mut volume_tm = vec![f32::NAN; MANY_COLS * MANY_ROWS];
        for s in 0..MANY_COLS {
            for t in 0..MANY_ROWS {
                let x = (t as f32) * 0.001 + (s as f32) * 0.01;
                close_tm[t * MANY_COLS + s] = (x * 0.79).sin() + 0.002 * x;
                volume_tm[t * MANY_COLS + s] = (x * 0.37).cos().abs() * 800.0 + 50.0;
            }
        }
        Box::new(ObvManySeriesState {
            cuda,
            close_tm,
            volume_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("obv", "one_series", "obv_cuda_batch", "1m", prep_one_series)
                .with_sample_size(10)
                .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "obv",
                "many_series",
                "obv_cuda_many_series_tm",
                "200k x 128",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
