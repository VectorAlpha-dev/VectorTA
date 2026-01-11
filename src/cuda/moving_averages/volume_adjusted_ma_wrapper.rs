//! CUDA wrapper for the Volume Adjusted Moving Average kernels.
//!
//! Mirrors the VRAM-first design used by the ALMA and WMA wrappers: the
//! methods below operate on FP32 buffers already staged on-device and expose
//! `DeviceArrayF32` handles for higher layers to decide when host copies are
//! required.
//!
//! Notes/parity with ALMA wrapper:
//! - PTX JIT options: DetermineTargetFromContext + OptLevel O4 with fallbacks.
//! - Non-blocking stream.
//! - VRAM checks with ~64MB headroom for batch, ~48MB for many-series.
//! - Batch grid.y chunking to <= 65_535 with pointer offsetting.
//! - Simple kernel policy for block size selection + one-time BENCH_DEBUG log.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::cuda::moving_averages::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::indicators::moving_averages::volume_adjusted_ma::{
    VolumeAdjustedMaBatchRange, VolumeAdjustedMaParams,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaVamaError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("not implemented")]
    NotImplemented,
    #[error("out of memory: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("output slice length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
}

pub struct CudaVama {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaVamaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
}

impl CudaVama {
    pub fn new(device_id: usize) -> Result<Self, CudaVamaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/volume_adjusted_ma_kernel.ptx"));
        
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

        

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaVamaPolicy::default(),
            last_batch: None,
            last_many: None,
        })
    }

    /// Create using an explicit policy.
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaVamaPolicy,
    ) -> Result<Self, CudaVamaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaVamaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaVamaPolicy {
        &self.policy
    }
    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self._context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaVamaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaVamaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn validate_launch_dims(&self, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Result<(), CudaVamaError> {
        use cust::device::DeviceAttribute;
        let dev = Device::get_device(self.device_id)?;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let (gx, gy, gz) = grid; let (bx, by, bz) = block;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaVamaError::InvalidInput("zero-sized grid or block".into()));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz {
            return Err(CudaVamaError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    fn expand_range(range: &VolumeAdjustedMaBatchRange) -> Vec<VolumeAdjustedMaParams> {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e { return vec![s]; }
            if s < e {
                let mut v = Vec::new();
                let mut x = s;
                while x <= e {
                    v.push(x);
                    match x.checked_add(st) {
                        Some(nx) if nx > x => x = nx,
                        _ => break,
                    }
                }
                v
            } else {
                let mut v = Vec::new();
                let mut x = s;
                while x >= e {
                    v.push(x);
                    match x.checked_sub(st) {
                        Some(nx) => x = nx,
                        None => break,
                    }
                }
                v
            }
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                vec![s]
            } else {
                let mut v = Vec::new();
                let mut x = s;
                if st > 0.0 {
                    while x <= e + 1e-12 {
                        v.push(x);
                        x += st;
                    }
                } else {
                    while x >= e - 1e-12 {
                        v.push(x);
                        x += st; 
                    }
                }
                v
            }
        }

        let lengths = axis_usize(range.length);
        let vi_factors = axis_f64(range.vi_factor);
        let sample_periods = axis_usize(range.sample_period);
        let stricts: Vec<bool> = match range.strict {
            Some(b) => vec![b],
            None => vec![true, false],
        };

        let mut combos = Vec::with_capacity(
            lengths.len() * vi_factors.len() * sample_periods.len() * stricts.len(),
        );
        for &len in &lengths {
            for &vf in &vi_factors {
                for &sp in &sample_periods {
                    for &st in &stricts {
                        combos.push(VolumeAdjustedMaParams {
                            length: Some(len),
                            vi_factor: Some(vf),
                            sample_period: Some(sp),
                            strict: Some(st),
                        });
                    }
                }
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        volumes: &[f32],
        sweep: &VolumeAdjustedMaBatchRange,
    ) -> Result<(Vec<VolumeAdjustedMaParams>, usize, usize, usize), CudaVamaError> {
        if prices.is_empty() {
            return Err(CudaVamaError::InvalidInput("empty price data".into()));
        }
        if volumes.is_empty() {
            return Err(CudaVamaError::InvalidInput("empty volume data".into()));
        }
        if prices.len() != volumes.len() {
            return Err(CudaVamaError::InvalidInput(format!(
                "price/volume length mismatch: {} vs {}",
                prices.len(),
                volumes.len()
            )));
        }

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaVamaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = prices.len();
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaVamaError::InvalidInput("all price values are NaN".into()))?;

        let mut max_length = 0usize;
        for prm in &combos {
            let length = prm.length.unwrap_or(0);
            let vi_factor = prm.vi_factor.unwrap_or(0.0);
            if length == 0 || length > series_len {
                return Err(CudaVamaError::InvalidInput(format!(
                    "invalid length {} (series len {})",
                    length, series_len
                )));
            }
            if !(vi_factor.is_finite()) || vi_factor <= 0.0 {
                return Err(CudaVamaError::InvalidInput(format!(
                    "invalid vi_factor {}",
                    vi_factor
                )));
            }
            let valid = series_len - first_valid;
            if valid < length {
                return Err(CudaVamaError::InvalidInput(format!(
                    "not enough valid data: need >= {}, valid = {}",
                    length, valid
                )));
            }
            max_length = max_length.max(length);
        }

        Ok((combos, first_valid, series_len, max_length))
    }

    fn build_prefix_sums(prices: &[f32], volumes: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut prefix_vol = Vec::with_capacity(volumes.len());
        let mut prefix_price_vol = Vec::with_capacity(volumes.len());
        let mut accum_vol = 0.0f32;
        let mut accum_price_vol = 0.0f32;
        for (&p, &v) in prices.iter().zip(volumes.iter()) {
            let vol_nz = if v.is_nan() { 0.0f32 } else { v };
            let price_nz = if p.is_nan() { 0.0f32 } else { p };
            accum_vol += vol_nz;
            accum_price_vol += price_nz * vol_nz;
            prefix_vol.push(accum_vol);
            prefix_price_vol.push(accum_price_vol);
        }
        (prefix_vol, prefix_price_vol)
    }

    fn build_prefix_sums_time_major(
        prices_tm: &[f32],
        volumes_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut prefix_vol = vec![0.0f32; cols * rows];
        let mut prefix_price_vol = vec![0.0f32; cols * rows];
        for series in 0..cols {
            let mut accum_vol = 0.0f32;
            let mut accum_price_vol = 0.0f32;
            for t in 0..rows {
                let idx = t * cols + series;
                let vol = volumes_tm[idx];
                let price = prices_tm[idx];
                let vol_nz = if vol.is_nan() { 0.0f32 } else { vol };
                let price_nz = if price.is_nan() { 0.0f32 } else { price };
                accum_vol += vol_nz;
                accum_price_vol += price_nz * vol_nz;
                prefix_vol[idx] = accum_vol;
                prefix_price_vol[idx] = accum_price_vol;
            }
        }
        (prefix_vol, prefix_price_vol)
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        d_prefix_volumes: &DeviceBuffer<f32>,
        d_prefix_price_volumes: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_vi_factors: &DeviceBuffer<f32>,
        d_sample_periods: &DeviceBuffer<i32>,
        d_strict_flags: &DeviceBuffer<u8>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        let func = self
            .module
            .get_function("volume_adjusted_ma_batch_f32")
            .map_err(|_| CudaVamaError::MissingKernelSymbol { name: "volume_adjusted_ma_batch_f32" })?;

        
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            _ => 256,
        };

        
        unsafe {
            let this = self as *const _ as *mut CudaVama;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        
        const MAX_GRID_Y: usize = 65_535;
        let mut launched = 0usize;
        while launched < n_combos {
            let len = (n_combos - launched).min(MAX_GRID_Y);

            let grid_x = ((series_len as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch_dims((grid_x.max(1), len as u32, 1), (block_x, 1, 1))?;

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut volumes_ptr = d_volumes.as_device_ptr().as_raw();
                let mut prefix_vol_ptr = d_prefix_volumes.as_device_ptr().as_raw();
                let mut prefix_price_vol_ptr = d_prefix_price_volumes.as_device_ptr().as_raw();
                let mut lengths_ptr = d_lengths.as_device_ptr().add(launched).as_raw();
                let mut vi_factors_ptr = d_vi_factors.as_device_ptr().add(launched).as_raw();
                let mut sample_periods_ptr =
                    d_sample_periods.as_device_ptr().add(launched).as_raw();
                let mut strict_ptr = d_strict_flags.as_device_ptr().add(launched).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(launched * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut volumes_ptr as *mut _ as *mut c_void,
                    &mut prefix_vol_ptr as *mut _ as *mut c_void,
                    &mut prefix_price_vol_ptr as *mut _ as *mut c_void,
                    &mut lengths_ptr as *mut _ as *mut c_void,
                    &mut vi_factors_ptr as *mut _ as *mut c_void,
                    &mut sample_periods_ptr as *mut _ as *mut c_void,
                    &mut strict_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaVamaError::Cuda)?;
            }
            launched += len;
        }
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        volumes: &[f32],
        combos: &[VolumeAdjustedMaParams],
        first_valid: usize,
        series_len: usize,
        _max_length: usize,
    ) -> Result<DeviceArrayF32, CudaVamaError> {
        let n_combos = combos.len();
        let (prefix_vol, prefix_price_vol) = Self::build_prefix_sums(prices, volumes);

        let lengths_i32: Vec<i32> = combos.iter().map(|p| p.length.unwrap() as i32).collect();
        let vi_factors_f32: Vec<f32> = combos.iter().map(|p| p.vi_factor.unwrap() as f32).collect();
        let sample_periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.sample_period.unwrap_or(0) as i32)
            .collect();
        let strict_flags: Vec<u8> = combos
            .iter()
            .map(|p| if p.strict.unwrap_or(true) { 1 } else { 0 })
            .collect();

        let item = std::mem::size_of::<f32>();
        let base_bytes = 2usize
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(item))
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let prefix_bytes = 2usize
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(item))
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let param_each = (std::mem::size_of::<i32>() * 2)
            + std::mem::size_of::<f32>()
            + std::mem::size_of::<u8>();
        let param_bytes = n_combos
            .checked_mul(param_each)
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(item))
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let required = base_bytes
            .checked_add(prefix_bytes)
            .and_then(|x| x.checked_add(param_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; 
        Self::will_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(prices).map_err(CudaVamaError::Cuda)?;
        let d_volumes = DeviceBuffer::from_slice(volumes).map_err(CudaVamaError::Cuda)?;
        let d_prefix_volumes = DeviceBuffer::from_slice(&prefix_vol).map_err(CudaVamaError::Cuda)?;
        let d_prefix_price_volumes = DeviceBuffer::from_slice(&prefix_price_vol).map_err(CudaVamaError::Cuda)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths_i32).map_err(CudaVamaError::Cuda)?;
        let d_vi_factors = DeviceBuffer::from_slice(&vi_factors_f32).map_err(CudaVamaError::Cuda)?;
        let d_sample_periods = DeviceBuffer::from_slice(&sample_periods_i32).map_err(CudaVamaError::Cuda)?;
        let d_strict_flags = DeviceBuffer::from_slice(&strict_flags).map_err(CudaVamaError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
            .map_err(CudaVamaError::Cuda)?;

        self.launch_batch_kernel(
            &d_prices,
            &d_volumes,
            &d_prefix_volumes,
            &d_prefix_price_volumes,
            &d_lengths,
            &d_vi_factors,
            &d_sample_periods,
            &d_strict_flags,
            series_len,
            n_combos,
            first_valid,
            &mut d_out,
        )?;
        
        
        self.stream
            .synchronize()
            .map_err(CudaVamaError::Cuda)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn vama_batch_dev(
        &self,
        prices: &[f32],
        volumes: &[f32],
        sweep: &VolumeAdjustedMaBatchRange,
    ) -> Result<DeviceArrayF32, CudaVamaError> {
        let (combos, first_valid, series_len, max_length) =
            Self::prepare_batch_inputs(prices, volumes, sweep)?;
        self.run_batch_kernel(
            prices,
            volumes,
            &combos,
            first_valid,
            series_len,
            max_length,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VolumeAdjustedMaParams,
    ) -> Result<(Vec<i32>, usize, f32, usize, bool), CudaVamaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVamaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows || volume_tm_f32.len() != cols * rows {
            return Err(CudaVamaError::InvalidInput(
                "price/volume length mismatch with cols*rows".into(),
            ));
        }

        let length = params.length.unwrap_or(0);
        let vi_factor = params.vi_factor.unwrap_or(0.0);
        let sample_period = params.sample_period.unwrap_or(0);
        let strict = params.strict.unwrap_or(true);

        if length == 0 || length > rows {
            return Err(CudaVamaError::InvalidInput(format!(
                "invalid length {} (series_len {})",
                length, rows
            )));
        }
        if !vi_factor.is_finite() || vi_factor <= 0.0 {
            return Err(CudaVamaError::InvalidInput(format!(
                "invalid vi_factor {}",
                vi_factor
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let first = fv
                .ok_or_else(|| CudaVamaError::InvalidInput(format!("series {} all NaN", series)))?;
            if rows - first < length {
                return Err(CudaVamaError::InvalidInput(format!(
                    "series {} lacks data: need >= {}, valid = {}",
                    series,
                    length,
                    rows - first
                )));
            }
            first_valids[series] = first as i32;
        }

        Ok((
            first_valids,
            length,
            vi_factor as f32,
            sample_period,
            strict,
        ))
    }

    fn next_pow2_le(x: usize) -> u32 {
        if x == 0 {
            return 1;
        }
        let mut p: u32 = 1;
        while (p as usize) << 1 <= x {
            p <<= 1;
        }
        p
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        d_prefix_volumes: &DeviceBuffer<f32>,
        d_prefix_price_volumes: &DeviceBuffer<f32>,
        period: usize,
        vi_factor: f32,
        sample_period: usize,
        strict: bool,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        
        let func = self
            .module
            .get_function("volume_adjusted_ma_multi_series_one_param_time_major_f32")
            .map_err(|_| CudaVamaError::MissingKernelSymbol { name: "volume_adjusted_ma_multi_series_one_param_time_major_f32" })?;

        
        let (suggested_block_x, min_grid) = func
            .suggested_launch_configuration(0, (0u32, 0u32, 0u32).into())
            .map_err(CudaVamaError::Cuda)?;

        let mut threads = Self::next_pow2_le(cols)
            .min(suggested_block_x.max(128).min(1024))
            .min(256);
        if threads < 32 {
            threads = 32;
        }

        
        let grid_x = (rows as u32).min(min_grid.max(1));
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (threads, 1, 1).into();
        self.validate_launch_dims((grid_x, 1, 1), (threads, 1, 1))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut volumes_ptr = d_volumes.as_device_ptr().as_raw();
            let mut prefix_vol_ptr = d_prefix_volumes.as_device_ptr().as_raw();
            let mut prefix_price_vol_ptr = d_prefix_price_volumes.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut vi_factor_f = vi_factor;
            let mut sample_period_i = sample_period as i32;
            let mut strict_flag: u8 = if strict { 1 } else { 0 };
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut volumes_ptr as *mut _ as *mut c_void,
                &mut prefix_vol_ptr as *mut _ as *mut c_void,
                &mut prefix_price_vol_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut vi_factor_f as *mut _ as *mut c_void,
                &mut sample_period_i as *mut _ as *mut c_void,
                &mut strict_flag as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(CudaVamaError::Cuda)?;
        }
        
        unsafe {
            let this = self as *const _ as *mut CudaVama;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x: threads });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        length: usize,
        vi_factor: f32,
        sample_period: usize,
        strict: bool,
    ) -> Result<DeviceArrayF32, CudaVamaError> {
        let (prefix_vol, prefix_price_vol) =
            Self::build_prefix_sums_time_major(data_tm_f32, volume_tm_f32, cols, rows);

        let total = cols * rows;
        let f32_sz = std::mem::size_of::<f32>();
        let base_bytes = 2usize
            .checked_mul(total)
            .and_then(|x| x.checked_mul(f32_sz))
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let prefix_bytes = 2usize
            .checked_mul(total)
            .and_then(|x| x.checked_mul(f32_sz))
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = total
            .checked_mul(f32_sz)
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let required = base_bytes
            .checked_add(prefix_bytes)
            .and_then(|x| x.checked_add(first_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaVamaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 48 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaVamaError::Cuda)?;
        let d_volumes = DeviceBuffer::from_slice(volume_tm_f32).map_err(CudaVamaError::Cuda)?;
        let d_prefix_volumes = DeviceBuffer::from_slice(&prefix_vol).map_err(CudaVamaError::Cuda)?;
        let d_prefix_price_volumes = DeviceBuffer::from_slice(&prefix_price_vol).map_err(CudaVamaError::Cuda)?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids).map_err(CudaVamaError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
            .map_err(CudaVamaError::Cuda)?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_volumes,
            &d_prefix_volumes,
            &d_prefix_price_volumes,
            length,
            vi_factor,
            sample_period,
            strict,
            cols,
            rows,
            &d_first_valids,
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn vama_batch_into_host_f32(
        &self,
        prices: &[f32],
        volumes: &[f32],
        sweep: &VolumeAdjustedMaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<VolumeAdjustedMaParams>), CudaVamaError> {
        let (combos, first_valid, series_len, max_length) =
            Self::prepare_batch_inputs(prices, volumes, sweep)?;
        let expected = combos
            .len()
            .checked_mul(series_len)
            .ok_or_else(|| CudaVamaError::InvalidInput("size overflow".into()))?;
        if out.len() != expected {
            return Err(CudaVamaError::OutputLengthMismatch { expected, got: out.len() });
        }
        let arr = self.run_batch_kernel(
            prices,
            volumes,
            &combos,
            first_valid,
            series_len,
            max_length,
        )?;
        
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(expected).map_err(CudaVamaError::Cuda)?
        };
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaVamaError::Cuda)?;
        }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn vama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        d_prefix_volumes: &DeviceBuffer<f32>,
        d_prefix_price_volumes: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_vi_factors: &DeviceBuffer<f32>,
        d_sample_periods: &DeviceBuffer<i32>,
        d_strict_flags: &DeviceBuffer<u8>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaVamaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_volumes,
            d_prefix_volumes,
            d_prefix_price_volumes,
            d_lengths,
            d_vi_factors,
            d_sample_periods,
            d_strict_flags,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            d_out,
        )
    }

    
    #[inline]
    pub fn volume_adjusted_ma_batch_dev(
        &self,
        prices: &[f32],
        volumes: &[f32],
        sweep: &VolumeAdjustedMaBatchRange,
    ) -> Result<DeviceArrayF32, CudaVamaError> {
        self.vama_batch_dev(prices, volumes, sweep)
    }

    #[inline]
    pub fn volume_adjusted_ma_batch_into_host_f32(
        &self,
        prices: &[f32],
        volumes: &[f32],
        sweep: &VolumeAdjustedMaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<VolumeAdjustedMaParams>), CudaVamaError> {
        self.vama_batch_into_host_f32(prices, volumes, sweep, out)
    }

    #[inline]
    pub fn volume_adjusted_ma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        d_prefix_volumes: &DeviceBuffer<f32>,
        d_prefix_price_volumes: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_vi_factors: &DeviceBuffer<f32>,
        d_sample_periods: &DeviceBuffer<i32>,
        d_strict_flags: &DeviceBuffer<u8>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        self.vama_batch_device(
            d_prices,
            d_volumes,
            d_prefix_volumes,
            d_prefix_price_volumes,
            d_lengths,
            d_vi_factors,
            d_sample_periods,
            d_strict_flags,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn vama_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VolumeAdjustedMaParams,
    ) -> Result<DeviceArrayF32, CudaVamaError> {
        let (first_valids, length, vi_factor, sample_period, strict) =
            Self::prepare_many_series_inputs(data_tm_f32, volume_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            volume_tm_f32,
            cols,
            rows,
            &first_valids,
            length,
            vi_factor,
            sample_period,
            strict,
        )
    }

    #[inline]
    pub fn volume_adjusted_ma_many_series_one_param_time_major_dev(
        &self,
        price_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VolumeAdjustedMaParams,
    ) -> Result<DeviceArrayF32, CudaVamaError> {
        self.vama_multi_series_one_param_time_major_dev(
            price_tm_f32,
            volume_tm_f32,
            cols,
            rows,
            params,
        )
    }

    pub fn vama_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VolumeAdjustedMaParams,
        out: &mut [f32],
    ) -> Result<(), CudaVamaError> {
        if out.len() != cols * rows {
            return Err(CudaVamaError::OutputLengthMismatch { expected: cols * rows, got: out.len() });
        }
        let arr = self.vama_multi_series_one_param_time_major_dev(
            data_tm_f32,
            volume_tm_f32,
            cols,
            rows,
            params,
        )?;
        
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(cols * rows)
                .map_err(CudaVamaError::Cuda)?
        };
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaVamaError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaVamaError::Cuda)?;
        out.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    #[inline]
    pub fn volume_adjusted_ma_many_series_one_param_time_major_into_host_f32(
        &self,
        price_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VolumeAdjustedMaParams,
        out: &mut [f32],
    ) -> Result<(), CudaVamaError> {
        self.vama_multi_series_one_param_time_major_into_host_f32(
            price_tm_f32,
            volume_tm_f32,
            cols,
            rows,
            params,
            out,
        )
    }

    pub fn vama_multi_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        d_prefix_volumes: &DeviceBuffer<f32>,
        d_prefix_price_volumes: &DeviceBuffer<f32>,
        period: i32,
        vi_factor: f32,
        sample_period: i32,
        strict: bool,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaVamaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            d_volumes,
            d_prefix_volumes,
            d_prefix_price_volumes,
            period as usize,
            vi_factor,
            sample_period.max(0) as usize,
            strict,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out,
        )
    }

    #[inline]
    pub fn volume_adjusted_ma_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        d_prefix_volumes: &DeviceBuffer<f32>,
        d_prefix_price_volumes: &DeviceBuffer<f32>,
        period: i32,
        vi_factor: f32,
        sample_period: i32,
        strict: bool,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        self.vama_multi_series_one_param_time_major_device(
            d_prices,
            d_volumes,
            d_prefix_volumes,
            d_prefix_price_volumes,
            period,
            vi_factor,
            sample_period,
            strict,
            num_series,
            series_len,
            d_first_valids,
            d_out,
        )
    }
}


#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::Bound;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32")]
pub struct DeviceArrayF32Py {
    pub inner: Option<DeviceArrayF32>,
    _ctx_guard: Arc<Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("buffer already exported via __dlpack__"))?;
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (inner.cols * itemsize, itemsize))?;
        let size = inner.rows.saturating_mul(inner.cols);
        let ptr_val: usize = if size == 0 { 0 } else { inner.buf.as_device_ptr().as_raw() as usize };
        d.set_item("data", (ptr_val, false))?;
        // Stream omitted: producing stream is synchronized before return (see wrapper)
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self._device_id as i32) }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        // Producer stream is synchronized before returning the handle; ignore consumer stream.
        let _ = stream;

        // Compute allocation device and validate optional dl_device hint.
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "dl_device mismatch for __dlpack__",
                        ));
                    }
                }
            }
        }

        // Move out the device buffer once; subsequent calls error.
        let inner = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl DeviceArrayF32Py {
    pub fn new_from_rust(inner: DeviceArrayF32, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self { inner: Some(inner), _ctx_guard: ctx_guard, _device_id: device_id }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices, gen_time_major_volumes};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        // prices + volumes + prefix_volumes + prefix_price_volumes
        let in_bytes = 4 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = PARAM_SWEEP
            * (std::mem::size_of::<i32>()  // length
                + std::mem::size_of::<f32>() // vi_factor
                + std::mem::size_of::<i32>() // sample_period
                + std::mem::size_of::<u8>()); // strict flag
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        // prices + volumes + prefix_volumes + prefix_price_volumes
        let in_bytes = 4 * elems * std::mem::size_of::<f32>();
        let first_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + first_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct VamaBatchDevState {
        cuda: CudaVama,
        d_prices: DeviceBuffer<f32>,
        d_volumes: DeviceBuffer<f32>,
        d_prefix_volumes: DeviceBuffer<f32>,
        d_prefix_price_volumes: DeviceBuffer<f32>,
        d_lengths: DeviceBuffer<i32>,
        d_vi_factors: DeviceBuffer<f32>,
        d_sample_periods: DeviceBuffer<i32>,
        d_strict_flags: DeviceBuffer<u8>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for VamaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_volumes,
                    &self.d_prefix_volumes,
                    &self.d_prefix_price_volumes,
                    &self.d_lengths,
                    &self.d_vi_factors,
                    &self.d_sample_periods,
                    &self.d_strict_flags,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("volume_adjusted_ma batch kernel");
            self.cuda.stream.synchronize().expect("vama sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaVama::new(0).expect("cuda vama");
        let price = gen_series(ONE_SERIES_LEN);
        let volume = gen_series(ONE_SERIES_LEN)
            .into_iter()
            .map(|v| {
                if v.is_nan() {
                    v
                } else {
                    (v.abs() + 1.0) * 700.0
                }
            })
            .collect::<Vec<f32>>();
        let sweep = VolumeAdjustedMaBatchRange {
            length: (16, 16 + PARAM_SWEEP - 1, 1),
            vi_factor: (1.0, 1.0, 0.0),
            sample_period: (1, 1, 0),
            strict: Some(true),
        };
        let (combos, first_valid, series_len, _max_length) =
            CudaVama::prepare_batch_inputs(&price, &volume, &sweep).expect("vama prepare batch");
        let n_combos = combos.len();
        let mut lengths = Vec::with_capacity(n_combos);
        let mut vi_factors = Vec::with_capacity(n_combos);
        let mut sample_periods = Vec::with_capacity(n_combos);
        let mut strict_flags = Vec::with_capacity(n_combos);
        for prm in &combos {
            lengths.push(prm.length.unwrap_or(0) as i32);
            vi_factors.push(prm.vi_factor.unwrap_or(0.0) as f32);
            sample_periods.push(prm.sample_period.unwrap_or(0) as i32);
            strict_flags.push(if prm.strict.unwrap_or(true) { 1u8 } else { 0u8 });
        }

        let (prefix_volumes, prefix_price_volumes) = CudaVama::build_prefix_sums(&price, &volume);

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_volumes = DeviceBuffer::from_slice(&volume).expect("d_volumes");
        let d_prefix_volumes = DeviceBuffer::from_slice(&prefix_volumes).expect("d_prefix_volumes");
        let d_prefix_price_volumes =
            DeviceBuffer::from_slice(&prefix_price_volumes).expect("d_prefix_price_volumes");
        let d_lengths = DeviceBuffer::from_slice(&lengths).expect("d_lengths");
        let d_vi_factors = DeviceBuffer::from_slice(&vi_factors).expect("d_vi_factors");
        let d_sample_periods = DeviceBuffer::from_slice(&sample_periods).expect("d_sample_periods");
        let d_strict_flags = DeviceBuffer::from_slice(&strict_flags).expect("d_strict_flags");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len.checked_mul(n_combos).expect("out size"))
        }
        .expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(VamaBatchDevState {
            cuda,
            d_prices,
            d_volumes,
            d_prefix_volumes,
            d_prefix_price_volumes,
            d_lengths,
            d_vi_factors,
            d_sample_periods,
            d_strict_flags,
            series_len,
            n_combos,
            first_valid,
            d_out,
        })
    }

    struct VamaManyDevState {
        cuda: CudaVama,
        d_prices_tm: DeviceBuffer<f32>,
        d_volumes_tm: DeviceBuffer<f32>,
        d_prefix_volumes_tm: DeviceBuffer<f32>,
        d_prefix_price_volumes_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        length: usize,
        vi_factor: f32,
        sample_period: usize,
        strict: bool,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for VamaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_volumes_tm,
                    &self.d_prefix_volumes_tm,
                    &self.d_prefix_price_volumes_tm,
                    self.length,
                    self.vi_factor,
                    self.sample_period,
                    self.strict,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("volume_adjusted_ma many-series kernel");
            self.cuda.stream.synchronize().expect("vama many-series sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaVama::new(0).expect("cuda vama");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let price_tm = gen_time_major_prices(cols, rows);
        let vol_tm = gen_time_major_volumes(cols, rows);
        let params = VolumeAdjustedMaParams {
            length: Some(64),
            vi_factor: Some(1.0),
            sample_period: Some(1),
            strict: Some(true),
        };
        let (first_valids, length, vi_factor, sample_period, strict) =
            CudaVama::prepare_many_series_inputs(&price_tm, &vol_tm, cols, rows, &params)
                .expect("vama prepare many-series");
        let (prefix_volumes_tm, prefix_price_volumes_tm) =
            CudaVama::build_prefix_sums_time_major(&price_tm, &vol_tm, cols, rows);

        let d_prices_tm = DeviceBuffer::from_slice(&price_tm).expect("d_prices_tm");
        let d_volumes_tm = DeviceBuffer::from_slice(&vol_tm).expect("d_volumes_tm");
        let d_prefix_volumes_tm =
            DeviceBuffer::from_slice(&prefix_volumes_tm).expect("d_prefix_volumes_tm");
        let d_prefix_price_volumes_tm =
            DeviceBuffer::from_slice(&prefix_price_volumes_tm).expect("d_prefix_price_volumes_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols.checked_mul(rows).expect("out size")) }
                .expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(VamaManyDevState {
            cuda,
            d_prices_tm,
            d_volumes_tm,
            d_prefix_volumes_tm,
            d_prefix_price_volumes_tm,
            d_first_valids,
            cols,
            rows,
            length,
            vi_factor,
            sample_period,
            strict,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "volume_adjusted_ma",
                "one_series_many_params",
                "vama_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "volume_adjusted_ma",
                "many_series_one_param",
                "vama_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

// -------- Kernel selection policy and introspection (parity with ALMA style) --------

#[derive(Clone, Copy, Debug)]
pub struct CudaVamaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaVamaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

impl CudaVama {
    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] VAMA batch selected kernel: {:?}", sel);
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] VAMA many-series selected kernel: {:?}", sel);
                }
            }
        }
    }
}
