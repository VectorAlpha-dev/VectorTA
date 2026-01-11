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
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaCciCycleError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
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

pub struct CudaCciCycle {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    sm_count: i32,
    max_grid_x: i32,
}

impl CudaCciCycle {
    pub fn new(device_id: usize) -> Result<Self, CudaCciCycleError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cci_cycle_kernel.ptx"));
        
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)?;
        let max_grid_x = device.get_attribute(DeviceAttribute::MaxGridDimX)?;

        Ok(Self { module, stream, context, device_id: device_id as u32, sm_count, max_grid_x })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    fn will_fit(required_bytes: usize, headroom: usize) -> Result<(), CudaCciCycleError> {
        if let Ok((free, _total)) = mem_get_info() {
            let dyn_headroom = (free as f64 * 0.05) as usize; 
            let keep = dyn_headroom.max(headroom);
            if required_bytes.saturating_add(keep) <= free {
                Ok(())
            } else {
                Err(CudaCciCycleError::OutOfMemory { required: required_bytes, free, headroom: keep })
            }
        } else {
            Ok(())
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
            return Err(CudaCciCycleError::InvalidInput("no parameter combinations".into()));
        }

        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaCciCycleError::InvalidInput("size overflow".into()))?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaCciCycleError::InvalidInput("size overflow".into()))?;
        let out_elems = rows
            .checked_mul(series_len)
            .ok_or_else(|| CudaCciCycleError::InvalidInput("rows*series_len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaCciCycleError::InvalidInput("size overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaCciCycleError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        
        let h_prices = LockedBuffer::from_slice(data_f32).map_err(CudaCciCycleError::Cuda)?;
        let lengths: Vec<i32> = combos.iter().map(|p| p.length.unwrap_or(0) as i32).collect();
        let factors: Vec<f32> = combos.iter().map(|p| p.factor.unwrap_or(0.5) as f32).collect();
        let h_lengths = LockedBuffer::from_slice(&lengths).map_err(CudaCciCycleError::Cuda)?;
        let h_factors = LockedBuffer::from_slice(&factors).map_err(CudaCciCycleError::Cuda)?;

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(&h_prices, &self.stream)
                .map_err(CudaCciCycleError::Cuda)?
        };
        let d_lengths = unsafe {
            DeviceBuffer::from_slice_async(&h_lengths, &self.stream)
                .map_err(CudaCciCycleError::Cuda)?
        };
        let d_factors = unsafe {
            DeviceBuffer::from_slice_async(&h_factors, &self.stream)
                .map_err(CudaCciCycleError::Cuda)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(rows * series_len, &self.stream)
                .map_err(CudaCciCycleError::Cuda)?
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

        self.stream.synchronize().map_err(CudaCciCycleError::Cuda)?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols: series_len })
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
            .map_err(|_| CudaCciCycleError::MissingKernelSymbol { name: "cci_cycle_batch_f32" })?;

        
        let block: BlockSize = (256, 1, 1).into();
        let needed_blocks = ((n_combos + 255) / 256) as u32;
        let min_blocks = (self.sm_count as u32).saturating_mul(16);
        let max_grid_x = self.max_grid_x as u32;
        let grid_x = needed_blocks.max(min_blocks).min(max_grid_x);
        if grid_x == 0 || block.x == 0 || block.x > 1024 {
            return Err(CudaCciCycleError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block.x, by: 1, bz: 1 });
        }
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
                .map_err(CudaCciCycleError::Cuda)?;
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
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCciCycleError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaCciCycleError::InvalidInput(
                "time-major matrix size mismatch".into(),
            ));
        }
        let length = params.length.unwrap_or(10);
        let factor = params.factor.unwrap_or(0.5) as f32;
        if length == 0 {
            return Err(CudaCciCycleError::InvalidInput("length must be > 0".into()));
        }

        
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

        
        let bytes = data_tm_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|v| v.checked_add(rows.checked_mul(std::mem::size_of::<i32>())?))
            .and_then(|v| v.checked_add(data_tm_f32.len().checked_mul(std::mem::size_of::<f32>())?))
            .ok_or_else(|| CudaCciCycleError::InvalidInput("size overflow".into()))?;
        Self::will_fit(bytes, 64 * 1024 * 1024)?;

        
        let h_prices = LockedBuffer::from_slice(data_tm_f32).map_err(CudaCciCycleError::Cuda)?;
        let h_firsts = LockedBuffer::from_slice(&first_valids).map_err(CudaCciCycleError::Cuda)?;
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(&h_prices, &self.stream)
                .map_err(CudaCciCycleError::Cuda)?
        };
        let d_first = unsafe {
            DeviceBuffer::from_slice_async(&h_firsts, &self.stream)
                .map_err(CudaCciCycleError::Cuda)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(expected, &self.stream)
                .map_err(CudaCciCycleError::Cuda)?
        };
        

        
        let func = self
            .module
            .get_function("cci_cycle_many_series_one_param_f32")
            .map_err(|_| CudaCciCycleError::MissingKernelSymbol { name: "cci_cycle_many_series_one_param_f32" })?;
        
        let block: BlockSize = (256, 1, 1).into();
        let grid_x = ((rows + 255) / 256) as u32;
        if grid_x == 0 || block.x == 0 || block.x > 1024 {
            return Err(CudaCciCycleError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block.x, by: 1, bz: 1 });
        }
        let grid: GridSize = (grid_x, 1, 1).into();
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
                .map_err(CudaCciCycleError::Cuda)?;
        }

        self.stream.synchronize().map_err(CudaCciCycleError::Cuda)?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
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
        let combos = expand_grid(sweep)?;
        
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
        let needed = max_len
            .checked_mul(2)
            .ok_or_else(|| CudaCciCycleError::InvalidInput("max_len*2 overflow".into()))?;
        if len - first_valid < needed {
            return Err(CudaCciCycleError::InvalidInput(
                "not enough valid data for largest window".into(),
            ));
        }
        Ok((combos, first_valid, len))
    }
}

fn expand_grid(r: &CciCycleBatchRange) -> Result<Vec<CciCycleParams>, CudaCciCycleError> {
    fn axis_usize((s, e, st): (usize, usize, usize)) -> Result<Vec<usize>, CudaCciCycleError> {
        if st == 0 || s == e {
            return Ok(vec![s]);
        }
        let mut vals = Vec::new();
        if s < e {
            let mut v = s;
            while v <= e {
                vals.push(v);
                v = match v.checked_add(st) {
                    Some(n) => n,
                    None => break,
                };
            }
        } else {
            let mut v = s;
            while v >= e {
                vals.push(v);
                if v < st {
                    break;
                }
                v -= st;
                if v == 0 && e > 0 {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(CudaCciCycleError::InvalidInput(
                "empty length range in cci_cycle CUDA sweep".into(),
            ));
        }
        Ok(vals)
    }
    fn axis_f64((s, e, st): (f64, f64, f64)) -> Result<Vec<f64>, CudaCciCycleError> {
        if !st.is_finite() {
            return Err(CudaCciCycleError::InvalidInput(
                "non-finite factor step in cci_cycle CUDA sweep".into(),
            ));
        }
        if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
            return Ok(vec![s]);
        }
        let mut vals = Vec::new();
        let step = st.abs();
        let eps = 1e-12;
        if s <= e {
            let mut x = s;
            while x <= e + eps {
                vals.push(x);
                x += step;
            }
        } else {
            let mut x = s;
            while x >= e - eps {
                vals.push(x);
                x -= step;
            }
        }
        if vals.is_empty() {
            return Err(CudaCciCycleError::InvalidInput(
                "empty factor range in cci_cycle CUDA sweep".into(),
            ));
        }
        Ok(vals)
    }

    let lens = axis_usize(r.length)?;
    let facs = axis_f64(r.factor)?;
    let cap = lens
        .len()
        .checked_mul(facs.len())
        .ok_or_else(|| CudaCciCycleError::InvalidInput("rows*cols overflow in cci_cycle CUDA sweep".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &l in &lens {
        for &f in &facs {
            out.push(CciCycleParams {
                length: Some(l),
                factor: Some(f),
            });
        }
    }
    Ok(out)
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    
    const ONE_SERIES_LEN: usize = 100_000;
    const PARAM_SWEEP: usize = 250;

    fn mem_bytes() -> usize {
        let in_b = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_b = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_b + out_b + 64 * 1024 * 1024
    }

    struct CciCycleBatchDeviceState {
        cuda: CudaCciCycle,
        d_prices: DeviceBuffer<f32>,
        d_lengths: DeviceBuffer<i32>,
        d_factors: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for CciCycleBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    self.series_len,
                    self.first_valid,
                    self.n_combos,
                    &self.d_lengths,
                    &self.d_factors,
                    &mut self.d_out,
                )
                .expect("cci_cycle launch_batch_kernel");
            let _ = self.cuda.stream.synchronize();
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
            factor: (0.5, 0.5, 0.0),
        };
        let (combos, first_valid, series_len) =
            CudaCciCycle::prepare_batch_inputs(&data, &sweep).expect("prepare_batch_inputs");
        let mut lengths: Vec<i32> = Vec::with_capacity(combos.len());
        let mut factors: Vec<f32> = Vec::with_capacity(combos.len());
        for c in &combos {
            lengths.push(c.length.unwrap() as i32);
            factors.push(c.factor.unwrap() as f32);
        }
        let d_prices = DeviceBuffer::from_slice(&data).expect("d_prices");
        let d_lengths = DeviceBuffer::from_slice(&lengths).expect("d_lengths");
        let d_factors = DeviceBuffer::from_slice(&factors).expect("d_factors");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }.expect("d_out");
        Box::new(CciCycleBatchDeviceState {
            cuda,
            d_prices,
            d_lengths,
            d_factors,
            d_out,
            series_len,
            first_valid,
            n_combos: combos.len(),
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "cci_cycle",
            "one_series_many_params",
            "cci_cycle_cuda_batch_dev",
            "100k_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(mem_bytes())]
    }
}
