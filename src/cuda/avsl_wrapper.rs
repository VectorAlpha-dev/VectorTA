//! CUDA scaffolding for the AVSL (Anti-Volume Stop Loss) indicator.
//!
//! Pattern: recurrence/time-scan per parameter (no big precompute to share).
//! We parallelize across rows (batch) or across series (many-series) and scan time
//! within each thread to preserve scalar semantics exactly.

#![cfg(feature = "cuda")]

use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;

use crate::indicators::avsl::{AvslBatchRange, AvslParams};

use super::moving_averages::alma_wrapper::DeviceArrayF32;

#[derive(Debug)]
pub enum CudaAvslError {
    Cuda(String),
    InvalidInput(String),
}

impl std::fmt::Display for CudaAvslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaAvslError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAvslError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaAvslError {}

pub struct CudaAvsl {
    module: Module,
    stream: Stream,
    _ctx: Context,
}

impl CudaAvsl {
    pub fn new(device_id: usize) -> Result<Self, CudaAvslError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/avsl_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaAvslError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _ctx: ctx,
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        cust::memory::mem_get_info().ok()
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

    fn expand_grid(range: &AvslBatchRange) -> Vec<AvslParams> {
        // Keep in sync with avsl.rs expand_grid_avsl
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                return vec![s];
            }
            (s..=e).step_by(st).collect()
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                return vec![s];
            }
            let mut v = Vec::new();
            let mut x = s;
            while x <= e + 1e-12 {
                v.push(x);
                x += st;
            }
            v
        }
        let fs = axis_usize(range.fast_period);
        let ss = axis_usize(range.slow_period);
        let ms = axis_f64(range.multiplier);
        let mut out = Vec::with_capacity(fs.len() * ss.len() * ms.len());
        for &f in &fs {
            for &s in &ss {
                for &m in &ms {
                    out.push(AvslParams {
                        fast_period: Some(f),
                        slow_period: Some(s),
                        multiplier: Some(m),
                    });
                }
            }
        }
        out
    }

    fn prepare_batch_inputs(
        close_f32: &[f32],
        low_f32: &[f32],
        volume_f32: &[f32],
        sweep: &AvslBatchRange,
    ) -> Result<(Vec<AvslParams>, usize, usize), CudaAvslError> {
        if close_f32.is_empty() {
            return Err(CudaAvslError::InvalidInput("empty input".into()));
        }
        if close_f32.len() != low_f32.len() || close_f32.len() != volume_f32.len() {
            return Err(CudaAvslError::InvalidInput("length mismatch".into()));
        }
        let len = close_f32.len();
        let fa = close_f32.iter().position(|v| !v.is_nan());
        let fb = low_f32.iter().position(|v| !v.is_nan());
        let fc = volume_f32.iter().position(|v| !v.is_nan());
        let first_valid = match (fa, fb, fc) {
            (Some(a), Some(b), Some(c)) => a.max(b).max(c),
            _ => return Err(CudaAvslError::InvalidInput("all values are NaN".into())),
        };
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaAvslError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for c in &combos {
            let f = c.fast_period.unwrap_or(12);
            let s = c.slow_period.unwrap_or(26);
            if f == 0 || s == 0 {
                return Err(CudaAvslError::InvalidInput("period must be >=1".into()));
            }
            if len - first_valid < s {
                return Err(CudaAvslError::InvalidInput(
                    "insufficient valid data for slow period".into(),
                ));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_vol: &DeviceBuffer<f32>,
        d_fast: &DeviceBuffer<i32>,
        d_slow: &DeviceBuffer<i32>,
        d_mult: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        rows: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAvslError> {
        let mut func: Function = self
            .module
            .get_function("avsl_batch_f32")
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        // Block size: suggest from CUDA, allow override via AVSL_BLOCK_X
        let block_x: u32 = match std::env::var("AVSL_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                let (_min, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_close = d_close.as_device_ptr().as_raw();
            let mut p_low = d_low.as_device_ptr().as_raw();
            let mut p_vol = d_vol.as_device_ptr().as_raw();
            let mut p_fast = d_fast.as_device_ptr().as_raw();
            let mut p_slow = d_slow.as_device_ptr().as_raw();
            let mut p_mult = d_mult.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut first_i = first_valid as i32;
            let mut rows_i = rows as i32;
            let mut p_out = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut p_close as *mut _ as *mut c_void,
                &mut p_low as *mut _ as *mut c_void,
                &mut p_vol as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut p_fast as *mut _ as *mut c_void,
                &mut p_slow as *mut _ as *mut c_void,
                &mut p_mult as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn avsl_batch_dev(
        &self,
        close_f32: &[f32],
        low_f32: &[f32],
        volume_f32: &[f32],
        sweep: &AvslBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<AvslParams>), CudaAvslError> {
        let (combos, first_valid, len) =
            Self::prepare_batch_inputs(close_f32, low_f32, volume_f32, sweep)?;

        // VRAM estimate
        let rows = combos.len();
        let bytes_required = len * std::mem::size_of::<f32>() * 3
            + rows * (std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>())
            + rows * len * std::mem::size_of::<f32>();
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes_required, headroom) {
            return Err(CudaAvslError::InvalidInput("insufficient VRAM".into()));
        }

        // Host-side combos → arrays
        let fast: Vec<i32> = combos
            .iter()
            .map(|c| c.fast_period.unwrap() as i32)
            .collect();
        let slow: Vec<i32> = combos
            .iter()
            .map(|c| c.slow_period.unwrap() as i32)
            .collect();
        let mult: Vec<f32> = combos
            .iter()
            .map(|c| c.multiplier.unwrap() as f32)
            .collect();

        // Async path with pinned host buffers
        let h_close =
            LockedBuffer::from_slice(close_f32).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let h_low =
            LockedBuffer::from_slice(low_f32).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let h_vol =
            LockedBuffer::from_slice(volume_f32).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let h_fast =
            LockedBuffer::from_slice(&fast).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let h_slow =
            LockedBuffer::from_slice(&slow).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let h_mult =
            LockedBuffer::from_slice(&mult).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        let mut d_close = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_low = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_vol = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_fast = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_slow = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_mult = unsafe { DeviceBuffer::<f32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let elems = rows * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        unsafe {
            d_close
                .async_copy_from(&h_close, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
            d_low
                .async_copy_from(&h_low, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
            d_vol
                .async_copy_from(&h_vol, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
            d_fast
                .async_copy_from(&h_fast, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
            d_slow
                .async_copy_from(&h_slow, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
            d_mult
                .async_copy_from(&h_mult, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        }

        self.launch_batch(
            &d_close,
            &d_low,
            &d_vol,
            &d_fast,
            &d_slow,
            &d_mult,
            len,
            first_valid,
            rows,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    // ----- Many-series, one param (time-major) -----
    fn prepare_many_series_inputs(
        close_tm_f32: &[f32],
        low_tm_f32: &[f32],
        vol_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AvslParams,
    ) -> Result<(Vec<i32>, usize, usize, usize), CudaAvslError> {
        if close_tm_f32.len() != cols * rows
            || low_tm_f32.len() != cols * rows
            || vol_tm_f32.len() != cols * rows
        {
            return Err(CudaAvslError::InvalidInput("matrix size mismatch".into()));
        }
        let fast = params.fast_period.unwrap_or(12);
        let slow = params.slow_period.unwrap_or(26);
        if fast == 0 || slow == 0 {
            return Err(CudaAvslError::InvalidInput("period must be >=1".into()));
        }
        // first_valid per series = max first-valid across close/low/vol for that column
        let mut firsts = vec![0i32; cols];
        for c in 0..cols {
            let mut fa: Option<usize> = None;
            let mut fb: Option<usize> = None;
            let mut fc: Option<usize> = None;
            for r in 0..rows {
                let idx = r * cols + c;
                if fa.is_none() && !close_tm_f32[idx].is_nan() {
                    fa = Some(r);
                }
                if fb.is_none() && !low_tm_f32[idx].is_nan() {
                    fb = Some(r);
                }
                if fc.is_none() && !vol_tm_f32[idx].is_nan() {
                    fc = Some(r);
                }
                if fa.is_some() && fb.is_some() && fc.is_some() {
                    break;
                }
            }
            let first = match (fa, fb, fc) {
                (Some(a), Some(b), Some(c3)) => a.max(b).max(c3),
                _ => return Err(CudaAvslError::InvalidInput("all-NaN series column".into())),
            };
            if rows - first < slow {
                return Err(CudaAvslError::InvalidInput(
                    "insufficient valid data for slow".into(),
                ));
            }
            firsts[c] = first as i32;
        }
        Ok((firsts, cols, rows, slow))
    }

    fn launch_many_series(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_vol: &DeviceBuffer<f32>,
        d_first: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        fast: usize,
        slow: usize,
        multiplier: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAvslError> {
        let mut func: Function = self
            .module
            .get_function("avsl_many_series_one_param_f32")
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        let block_x: u32 = match std::env::var("AVSL_MS_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                let (_min, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_close = d_close.as_device_ptr().as_raw();
            let mut p_low = d_low.as_device_ptr().as_raw();
            let mut p_vol = d_vol.as_device_ptr().as_raw();
            let mut p_first = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fast_i = fast as i32;
            let mut slow_i = slow as i32;
            let mut mult = multiplier as f32;
            let mut p_out = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut p_close as *mut _ as *mut c_void,
                &mut p_low as *mut _ as *mut c_void,
                &mut p_vol as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fast_i as *mut _ as *mut c_void,
                &mut slow_i as *mut _ as *mut c_void,
                &mut mult as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn avsl_many_series_one_param_time_major_dev(
        &self,
        close_tm_f32: &[f32],
        low_tm_f32: &[f32],
        vol_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AvslParams,
    ) -> Result<DeviceArrayF32, CudaAvslError> {
        let (firsts, cols, rows, slow) = Self::prepare_many_series_inputs(
            close_tm_f32,
            low_tm_f32,
            vol_tm_f32,
            cols,
            rows,
            params,
        )?;

        // Rough VRAM check
        let bytes =
            cols * rows * std::mem::size_of::<f32>() * 4 + cols * std::mem::size_of::<i32>();
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaAvslError::InvalidInput("insufficient VRAM".into()));
        }

        let h_close = LockedBuffer::from_slice(close_tm_f32)
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let h_low =
            LockedBuffer::from_slice(low_tm_f32).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let h_vol =
            LockedBuffer::from_slice(vol_tm_f32).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let h_first =
            LockedBuffer::from_slice(&firsts).map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        let mut d_close =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_low =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_vol =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_first = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        unsafe {
            d_close
                .async_copy_from(&h_close, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
            d_low
                .async_copy_from(&h_low, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
            d_vol
                .async_copy_from(&h_vol, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
            d_first
                .async_copy_from(&h_first, &self.stream)
                .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;
        }

        self.launch_many_series(
            &d_close,
            &d_low,
            &d_vol,
            &d_first,
            cols,
            rows,
            params.fast_period.unwrap_or(12),
            slow,
            params.multiplier.unwrap_or(2.0) as f32,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaAvslError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------- Bench Profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        // Batch: 100k samples, ~64 combos
        v.push(
            CudaBenchScenario::new(
                "avsl",
                "one_series_many_params",
                "avsl/batch",
                "100k x 64",
                || {
                    struct State {
                        cuda: CudaAvsl,
                        close: Vec<f32>,
                        low: Vec<f32>,
                        vol: Vec<f32>,
                        sweep: AvslBatchRange,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            let _ = self.cuda.avsl_batch_dev(
                                &self.close,
                                &self.low,
                                &self.vol,
                                &self.sweep,
                            );
                        }
                    }
                    let n = 100_000usize;
                    let mut close = vec![f32::NAN; n];
                    let mut low = vec![f32::NAN; n];
                    let mut vol = vec![f32::NAN; n];
                    for i in 200..n {
                        let x = i as f32;
                        close[i] = (x * 0.00123).sin() + 0.0002 * x;
                        low[i] = close[i] - 0.5 * (0.5 + (x * 0.01).cos().abs());
                        vol[i] = (x * 0.0007).cos().abs() + 0.7;
                    }
                    let sweep = AvslBatchRange {
                        fast_period: (4, 28, 4),
                        slow_period: (32, 128, 16),
                        multiplier: (2.0, 2.0, 0.0),
                    };
                    Box::new(State {
                        cuda: CudaAvsl::new(0).unwrap(),
                        close,
                        low,
                        vol,
                        sweep,
                    })
                },
            )
            .with_sample_size(20),
        );
        v
    }
}
