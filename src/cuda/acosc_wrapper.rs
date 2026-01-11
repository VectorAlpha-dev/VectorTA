//! CUDA support for the Accelerator Oscillator (ACOSC).
//!
//! Mirrors ALMA-style integration where applicable:
//! - PTX loaded via include_str!(.../acosc_kernel.ptx) with DetermineTargetFromContext + OptLevel O2
//! - Stream NON_BLOCKING
//! - Conservative VRAM checks with ~64MB headroom
//! - Public device entry points for one-series×batch (degenerate single row) and
//!   many-series×one-param (time-major)

#![cfg(feature = "cuda")]

use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

// --- Kernel constants (must match CUDA kernels) ---
const P5: usize = 5;
const P34: usize = 34;
const WARP: usize = 32;
// Dynamic shared memory required by the warp kernel (bytes)
const SHMEM_WARP_BYTES: u32 = ((P34 + P5 + P5) * WARP * std::mem::size_of::<f32>()) as u32;

#[derive(Debug, Error)]
pub enum CudaAcoscError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

pub struct DeviceArrayF32Acosc {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Acosc {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

pub struct DeviceAcoscPair {
    pub osc: DeviceArrayF32Acosc,
    pub change: DeviceArrayF32Acosc,
}
impl DeviceAcoscPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.osc.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.osc.cols
    }
}

pub struct CudaAcosc {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaAcosc {
    pub fn new(device_id: usize) -> Result<Self, CudaAcoscError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/acosc_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[])?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self { module, stream, _context: context, device_id: device_id as u32 })
    }

    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaAcoscError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free { Ok(()) } else { Err(CudaAcoscError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes }) }
        } else { Ok(()) }
    }

    // -------- Batch: one series (degenerate single row) --------
    pub fn acosc_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
    ) -> Result<DeviceAcoscPair, CudaAcoscError> {
        let len = high_f32.len();
        if len == 0 || low_f32.len() != len { return Err(CudaAcoscError::InvalidInput("input slices are empty or mismatched".into())); }
        let first_valid = (0..len)
            .find(|&i| high_f32[i].is_finite() && low_f32[i].is_finite())
            .unwrap_or(len);

        // VRAM estimate: 2 inputs + 2 outputs
        let in_bytes = 2usize.saturating_mul(len).saturating_mul(std::mem::size_of::<f32>());
        let out_bytes = 2usize.saturating_mul(len).saturating_mul(std::mem::size_of::<f32>());
        let required = in_bytes.saturating_add(out_bytes);
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_high = DeviceBuffer::from_slice(high_f32).map_err(CudaAcoscError::Cuda)?;
        let d_low = DeviceBuffer::from_slice(low_f32).map_err(CudaAcoscError::Cuda)?;
        let mut d_osc: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(CudaAcoscError::Cuda)?;
        let mut d_change: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(CudaAcoscError::Cuda)?;

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            len as i32,
            first_valid as i32,
            &mut d_osc,
            &mut d_change,
        )?;

        Ok(DeviceAcoscPair {
            osc: DeviceArrayF32Acosc { buf: d_osc, rows: 1, cols: len, ctx: self._context.clone(), device_id: self.device_id },
            change: DeviceArrayF32Acosc { buf: d_change, rows: 1, cols: len, ctx: self._context.clone(), device_id: self.device_id },
        })
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        series_len: i32,
        first_valid: i32,
        d_osc: &mut DeviceBuffer<f32>,
        d_change: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAcoscError> {
        if series_len <= 0 {
            return Ok(());
        }
        if series_len <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("acosc_batch_f32")
            .map_err(|_| CudaAcoscError::MissingKernelSymbol { name: "acosc_batch_f32" })?;

        let block_x: u32 = 256;
        let mut grid_x = (series_len as u32 + block_x - 1) / block_x;
        if grid_x == 0 {
            grid_x = 1;
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut len_i = series_len;
            let mut fv_i = first_valid;
            let mut osc_ptr = d_osc.as_device_ptr().as_raw();
            let mut chg_ptr = d_change.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut osc_ptr as *mut _ as *mut c_void,
                &mut chg_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        // Ensure temporaries on device live until the kernel completes.
        self.stream.synchronize()?;
        Ok(())
    }

    // -------- Many-series × one-param (time-major) --------
    pub fn acosc_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
    ) -> Result<DeviceAcoscPair, CudaAcoscError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaAcoscError::InvalidInput("empty dimensions".into()));
        }
        if high_tm_f32.len() != low_tm_f32.len() || high_tm_f32.len() != num_series * series_len {
            return Err(CudaAcoscError::InvalidInput(
                "time-major inputs must be same length and match rows*cols".into(),
            ));
        }
        // first_valid per series
        let mut first_valids = vec![series_len as i32; num_series];
        for s in 0..num_series {
            for t in 0..series_len {
                let idx = t * num_series + s;
                if high_tm_f32[idx].is_finite() && low_tm_f32[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let sz_f32 = std::mem::size_of::<f32>();
        let in_bytes = num_series
            .checked_mul(series_len)
            .and_then(|e| e.checked_mul(2))
            .and_then(|e| e.checked_mul(sz_f32))
            .ok_or_else(|| CudaAcoscError::InvalidInput("size overflow (inputs)".into()))?;
        let out_bytes = num_series
            .checked_mul(series_len)
            .and_then(|e| e.checked_mul(2))
            .and_then(|e| e.checked_mul(sz_f32))
            .ok_or_else(|| CudaAcoscError::InvalidInput("size overflow (outputs)".into()))?;
        let aux_bytes = num_series
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaAcoscError::InvalidInput("size overflow (aux)".into()))?;
        let required = in_bytes
            .checked_add(out_bytes)
            .and_then(|e| e.checked_add(aux_bytes))
            .ok_or_else(|| CudaAcoscError::InvalidInput("size overflow (required)".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_high = DeviceBuffer::from_slice(high_tm_f32).map_err(CudaAcoscError::Cuda)?;
        let d_low = DeviceBuffer::from_slice(low_tm_f32).map_err(CudaAcoscError::Cuda)?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaAcoscError::Cuda)?;
        let mut d_osc: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(num_series * series_len) }
                .map_err(CudaAcoscError::Cuda)?;
        let mut d_change: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(num_series * series_len) }
                .map_err(CudaAcoscError::Cuda)?;

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_first,
            num_series as i32,
            series_len as i32,
            &mut d_osc,
            &mut d_change,
        )?;

        Ok(DeviceAcoscPair {
            osc: DeviceArrayF32Acosc { buf: d_osc, rows: num_series, cols: series_len, ctx: self._context.clone(), device_id: self.device_id },
            change: DeviceArrayF32Acosc { buf: d_change, rows: num_series, cols: series_len, ctx: self._context.clone(), device_id: self.device_id },
        })
    }

    /// Many-series × one-param (time-major) using device-resident inputs to avoid H2D copies.
    /// `d_first_valids` must have length `num_series`.
    pub fn acosc_many_series_one_param_time_major_dev_device_inputs(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
    ) -> Result<DeviceAcoscPair, CudaAcoscError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaAcoscError::InvalidInput("empty dimensions".into()));
        }
        let elems = num_series
            .checked_mul(series_len)
            .ok_or_else(|| CudaAcoscError::InvalidInput("size overflow (rows*cols)".into()))?;
        let mut d_osc: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(CudaAcoscError::Cuda)?;
        let mut d_change: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(CudaAcoscError::Cuda)?;

        self.launch_many_series_kernel(
            d_high_tm,
            d_low_tm,
            d_first_valids,
            num_series as i32,
            series_len as i32,
            &mut d_osc,
            &mut d_change,
        )?;

        Ok(DeviceAcoscPair {
            osc: DeviceArrayF32Acosc { buf: d_osc, rows: num_series, cols: series_len, ctx: self._context.clone(), device_id: self.device_id },
            change: DeviceArrayF32Acosc { buf: d_change, rows: num_series, cols: series_len, ctx: self._context.clone(), device_id: self.device_id },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: i32,
        series_len: i32,
        d_osc_tm: &mut DeviceBuffer<f32>,
        d_change_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAcoscError> {
        if num_series <= 0 || series_len <= 0 {
            return Ok(());
        }
        // Heuristic: prefer warp-striped kernel for large series counts with enough time steps
        let use_warp = (num_series as usize) >= 64 && (series_len as usize) >= 128;
        unsafe {
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut ns = num_series;
            let mut sl = series_len;
            let mut osc_ptr = d_osc_tm.as_device_ptr().as_raw();
            let mut chg_ptr = d_change_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut ns as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut osc_ptr as *mut _ as *mut c_void,
                &mut chg_ptr as *mut _ as *mut c_void,
            ];

            if use_warp {
                if let Ok(func) = self
                    .module
                    .get_function("acosc_many_series_one_param_f32_warp")
                {
                    let grid: GridSize = (((num_series as u32) + 31) / 32, 1, 1).into();
                    let block: BlockSize = (32, 1, 1).into();
                    if 32 > 1024 || grid.x == 0 { return Err(CudaAcoscError::LaunchConfigTooLarge { gx: grid.x, gy: grid.y, gz: grid.z, bx: 32, by: 1, bz: 1 }); }
                    self.stream.launch(&func, grid, block, SHMEM_WARP_BYTES, &mut args)?;
                } else {
                    let func = self
                        .module
                        .get_function("acosc_many_series_one_param_f32")
                        .map_err(|_| CudaAcoscError::MissingKernelSymbol { name: "acosc_many_series_one_param_f32" })?;
                    let grid: GridSize = (num_series as u32, 1, 1).into();
                    let block: BlockSize = (256, 1, 1).into();
                    if 256 > 1024 || grid.x == 0 { return Err(CudaAcoscError::LaunchConfigTooLarge { gx: grid.x, gy: grid.y, gz: grid.z, bx: 256, by: 1, bz: 1 }); }
                    self.stream.launch(&func, grid, block, 0, &mut args)?;
                }
            } else {
                let func = self
                    .module
                    .get_function("acosc_many_series_one_param_f32")
                    .map_err(|_| CudaAcoscError::MissingKernelSymbol { name: "acosc_many_series_one_param_f32" })?;
                let grid: GridSize = (num_series as u32, 1, 1).into();
                let block: BlockSize = (256, 1, 1).into();
                if 256 > 1024 || grid.x == 0 { return Err(CudaAcoscError::LaunchConfigTooLarge { gx: grid.x, gy: grid.y, gz: grid.z, bx: 256, by: 1, bz: 1 }); }
                self.stream.launch(&func, grid, block, 0, &mut args)?;
            }
        }
        // Ensure temporary inputs aren’t dropped until this work finishes.
        self.stream.synchronize()?;
        Ok(())
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const NUM_SERIES: usize = 512;
    const SERIES_LEN: usize = 4096;

    fn bytes_one_series() -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let elems = NUM_SERIES * SERIES_LEN;
        let in_bytes = 2 * elems * std::mem::size_of::<f32>();
        let out_bytes = 2 * elems * std::mem::size_of::<f32>();
        let aux = NUM_SERIES * std::mem::size_of::<i32>();
        in_bytes + out_bytes + aux + 64 * 1024 * 1024
    }

    fn synth_hl_from_base(base: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = base.to_vec();
        let mut low = base.to_vec();
        for i in 0..base.len() {
            let v = base[i];
            if !v.is_finite() {
                continue;
            }
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0031;
            let off = (0.0049 * x.sin()).abs() + 0.13;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct OneSeriesDeviceState {
        cuda: CudaAcosc,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_osc: DeviceBuffer<f32>,
        d_change: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
    }
    impl CudaBenchState for OneSeriesDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_high,
                    &self.d_low,
                    self.len as i32,
                    self.first_valid as i32,
                    &mut self.d_osc,
                    &mut self.d_change,
                )
                .expect("acosc launch");
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaAcosc::new(0).unwrap();
        let base = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_base(&base);
        let len = high.len();
        let first_valid = (0..len)
            .find(|&i| high[i].is_finite() && low[i].is_finite())
            .unwrap_or(len);

        let d_high = DeviceBuffer::from_slice(&high).expect("acosc d_high H2D");
        let d_low = DeviceBuffer::from_slice(&low).expect("acosc d_low H2D");
        let d_osc: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.expect("acosc d_osc alloc");
        let d_change: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.expect("acosc d_change alloc");

        Box::new(OneSeriesDeviceState {
            cuda,
            d_high,
            d_low,
            d_osc,
            d_change,
            len,
            first_valid,
        })
    }

    struct ManySeriesDeviceState {
        cuda: CudaAcosc,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_osc_tm: DeviceBuffer<f32>,
        d_change_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManySeriesDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_high_tm,
                    &self.d_low_tm,
                    &self.d_first_valids,
                    NUM_SERIES as i32,
                    SERIES_LEN as i32,
                    &mut self.d_osc_tm,
                    &mut self.d_change_tm,
                )
                .expect("acosc many-series launch");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaAcosc::new(0).unwrap();
        // Build time-major arrays
        let mut high_tm = vec![f32::NAN; NUM_SERIES * SERIES_LEN];
        let mut low_tm = vec![f32::NAN; NUM_SERIES * SERIES_LEN];
        for s in 0..NUM_SERIES {
            let base = gen_series(SERIES_LEN);
            let (h, l) = synth_hl_from_base(&base);
            for t in 0..SERIES_LEN {
                let idx = t * NUM_SERIES + s;
                high_tm[idx] = h[t];
                low_tm[idx] = l[t];
            }
        }

        let mut first_valids = vec![SERIES_LEN as i32; NUM_SERIES];
        for s in 0..NUM_SERIES {
            for t in 0..SERIES_LEN {
                let idx = t * NUM_SERIES + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("acosc d_high_tm H2D");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("acosc d_low_tm H2D");
        let d_first_valids =
            DeviceBuffer::from_slice(&first_valids).expect("acosc d_first_valids H2D");
        let elems = NUM_SERIES * SERIES_LEN;
        let d_osc_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("acosc d_osc_tm alloc");
        let d_change_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("acosc d_change_tm alloc");

        Box::new(ManySeriesDeviceState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_first_valids,
            d_osc_tm,
            d_change_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "acosc",
                "one_series",
                "acosc_cuda_batch_dev",
                "1m",
                prep_one_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "acosc",
                "many_series_one_param",
                "acosc_cuda_many_series_one_param_dev",
                "512x4096",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
