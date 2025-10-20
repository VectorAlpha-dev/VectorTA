//! CUDA wrapper for the CCI Cycle indicator.
//!
//! Category: Recurrence/IIR per row. We keep the kernel simple and
//! one-thread-per-parameter style (like Wavetrend/WILLR batches), with
//! FP32 math and NaN/warmup semantics matching the scalar path within
//! reasonable tolerance.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::cci_cycle::{CciCycleBatchRange, CciCycleParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaCciCycleError {
    Cuda(String),
    InvalidInput(String),
    NotImplemented,
}

impl fmt::Display for CudaCciCycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCciCycleError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCciCycleError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaCciCycleError::NotImplemented => write!(f, "Not implemented"),
        }
    }
}
impl std::error::Error for CudaCciCycleError {}

pub struct CudaCciCycle {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaCciCycle {
    pub fn new(device_id: usize) -> Result<Self, CudaCciCycleError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cci_cycle_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                Module::from_ptx(ptx, &[]).map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    pub fn cci_cycle_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &CciCycleBatchRange,
    ) -> Result<DeviceArrayF32, CudaCciCycleError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;

        let rows = combos.len();
        if rows == 0 {
            return Err(CudaCciCycleError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let params_bytes = rows * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = rows * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaCciCycleError::InvalidInput(
                "insufficient VRAM for batch".into(),
            ));
        }

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)
                .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?
        };
        let lengths: Vec<i32> = combos
            .iter()
            .map(|p| p.length.unwrap_or(0) as i32)
            .collect();
        let factors: Vec<f32> = combos
            .iter()
            .map(|p| p.factor.unwrap_or(0.5) as f32)
            .collect();
        let d_lengths = DeviceBuffer::from_slice(&lengths)
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        let d_factors = DeviceBuffer::from_slice(&factors)
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(rows * series_len)
                .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            series_len,
            first_valid,
            rows,
            &d_lengths,
            &d_factors,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_lengths: &DeviceBuffer<i32>,
        d_factors: &DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCciCycleError> {
        let func = self
            .module
            .get_function("cci_cycle_batch_f32")
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;

        // 1D grid over rows
        let block: BlockSize = (256, 1, 1).into();
        let grid_x = ((n_combos + 255) / 256).min(65_535) as u32;
        let grid: GridSize = (grid_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut first_i = first_valid as i32;
            let mut n_i = n_combos as i32;
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut factors_ptr = d_factors.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut lengths_ptr as *mut _ as *mut c_void,
                &mut factors_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn cci_cycle_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CciCycleParams,
    ) -> Result<DeviceArrayF32, CudaCciCycleError> {
        if data_tm_f32.len() != cols * rows {
            return Err(CudaCciCycleError::InvalidInput(
                "time-major matrix size mismatch".into(),
            ));
        }
        let length = params.length.unwrap_or(10);
        let factor = params.factor.unwrap_or(0.5) as f32;
        if length == 0 {
            return Err(CudaCciCycleError::InvalidInput("length must be > 0".into()));
        }

        // First-valid per series (host)
        let mut first_valids = vec![0i32; rows];
        for r in 0..rows {
            let mut fv = 0usize;
            while fv < cols {
                let v = data_tm_f32[r * cols + fv];
                if !v.is_nan() {
                    break;
                }
                fv += 1;
            }
            first_valids[r] = fv as i32;
        }

        // VRAM estimate
        let bytes = data_tm_f32.len() * std::mem::size_of::<f32>()
            + rows * std::mem::size_of::<i32>()
            + data_tm_f32.len() * std::mem::size_of::<f32>();
        if !Self::will_fit(bytes, 64 * 1024 * 1024) {
            return Err(CudaCciCycleError::InvalidInput(
                "insufficient VRAM for many-series".into(),
            ));
        }

        // Upload
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)
                .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?
        };
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(cols * rows)
                .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?
        };

        // Launch
        let func = self
            .module
            .get_function("cci_cycle_many_series_one_param_f32")
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        // One thread per series (sequential over time). Chunk rows across blocks of 256.
        let block: BlockSize = (256, 1, 1).into();
        let grid: GridSize = (((rows + 255) / 256) as u32, 1, 1).into();
        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut len_i = length as i32;
            let mut factor_f = factor;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut factor_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaCciCycleError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn prepare_batch_inputs(
        data: &[f32],
        sweep: &CciCycleBatchRange,
    ) -> Result<(Vec<CciCycleParams>, usize, usize), CudaCciCycleError> {
        let len = data.len();
        if len == 0 {
            return Err(CudaCciCycleError::InvalidInput("empty input".into()));
        }
        let first_valid = data
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaCciCycleError::InvalidInput("all values NaN".into()))?;
        let combos = expand_grid(sweep);
        // Validate max length
        let max_len = combos
            .iter()
            .map(|p| p.length.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_len == 0 || max_len > len {
            return Err(CudaCciCycleError::InvalidInput(
                "invalid length in sweep".into(),
            ));
        }
        if len - first_valid < max_len * 2 {
            return Err(CudaCciCycleError::InvalidInput(
                "not enough valid data for largest window".into(),
            ));
        }
        Ok((combos, first_valid, len))
    }
}

fn expand_grid(r: &CciCycleBatchRange) -> Vec<CciCycleParams> {
    let (ls, le, ld) = r.length;
    let (fs, fe, fd) = r.factor;
    let mut len_vals = Vec::new();
    if ld == 0 || ls == le {
        len_vals.push(ls);
    } else {
        let mut v = ls;
        while v <= le {
            len_vals.push(v);
            v = v.saturating_add(ld);
        }
    }
    let mut fac_vals = Vec::new();
    if fd == 0.0 || (fs == fe) {
        fac_vals.push(fs);
    } else {
        let mut v = fs;
        while v <= fe + 1e-12 {
            fac_vals.push(v);
            v += fd;
            if fd.abs() < 1e-12 {
                break;
            }
        }
    }

    let mut out = Vec::with_capacity(len_vals.len() * fac_vals.len());
    for &l in &len_vals {
        for &f in &fac_vals {
            out.push(CciCycleParams {
                length: Some(l),
                factor: Some(f),
            });
        }
    }
    out
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn mem_bytes() -> usize {
        let in_b = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_b = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_b + out_b + 64 * 1024 * 1024
    }

    struct CciCycleBatchState {
        cuda: CudaCciCycle,
        data: Vec<f32>,
        sweep: CciCycleBatchRange,
    }
    impl CudaBenchState for CciCycleBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .cci_cycle_batch_dev(&self.data, &self.sweep)
                .expect("cci_cycle batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCciCycle::new(0).expect("cuda cci_cycle");
        let mut data = vec![f32::NAN; ONE_SERIES_LEN];
        for i in 128..ONE_SERIES_LEN {
            let x = i as f32;
            data[i] = (x * 0.0013).sin() * 0.8 + (x * 0.00077).cos();
        }
        let sweep = CciCycleBatchRange {
            length: (10, 10 + PARAM_SWEEP as usize - 1, 1),
            factor: (0.3, 0.7, 0.0016),
        };
        Box::new(CciCycleBatchState { cuda, data, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "cci_cycle",
            "one_series_many_params",
            "cci_cycle_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(mem_bytes())]
    }
}
