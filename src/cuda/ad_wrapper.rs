#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::error::CudaError;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub enum CudaAdError {
    Cuda(CudaError),
    InvalidInput(String),
    MissingKernelSymbol { name: &'static str },
    OutOfMemory { required: usize, free: usize, headroom: usize },
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    InvalidPolicy(&'static str),
    NotImplemented,
}

impl fmt::Display for CudaAdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAdError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAdError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CudaAdError::MissingKernelSymbol { name } => write!(f, "Missing kernel symbol: {}", name),
            CudaAdError::OutOfMemory { required, free, headroom } => write!(
                f,
                "Out of memory on device: required={}B, free={}B, headroom={}B",
                required, free, headroom
            ),
            CudaAdError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz } => write!(
                f,
                "Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))"
            ),
            CudaAdError::InvalidPolicy(p) => write!(f, "Invalid policy: {}", p),
            CudaAdError::NotImplemented => write!(f, "Not implemented"),
        }
    }
}

impl std::error::Error for CudaAdError {}

pub struct CudaAd {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAd {
    pub fn new(device_id: usize) -> Result<Self, CudaAdError> {
        cust::init(CudaFlags::empty()).map_err(CudaAdError::Cuda)?;
        let device =
            Device::get_device(device_id as u32).map_err(CudaAdError::Cuda)?;
        let context = Arc::new(Context::new(device).map_err(CudaAdError::Cuda)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ad_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(CudaAdError::Cuda)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(CudaAdError::Cuda)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaAdError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaAdError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn validate_launch(&self, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Result<(), CudaAdError> {
        let dev = Device::get_device(self.device_id).map_err(CudaAdError::Cuda)?;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .map_err(CudaAdError::Cuda)? as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaAdError::Cuda)? as u32;
        if block.0 == 0 || block.0 > max_bx || grid.0 == 0 || grid.0 > max_gx {
            return Err(CudaAdError::LaunchConfigTooLarge { gx: grid.0, gy: grid.1, gz: grid.2, bx: block.0, by: block.1, bz: block.2 });
        }
        Ok(())
    }

    fn validate_hlcv(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
    ) -> Result<usize, CudaAdError> {
        if high.is_empty() {
            return Err(CudaAdError::InvalidInput("empty inputs".into()));
        }
        let n = high.len();
        if low.len() != n || close.len() != n || volume.len() != n {
            return Err(CudaAdError::InvalidInput(
                "input slice length mismatch".into(),
            ));
        }
        Ok(n)
    }

    // --- Single series (row-major kernel supports multiple, we pass n_series=1) ---
    fn launch_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_volume: &DeviceBuffer<f32>,
        len: usize,
        n_series: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdError> {
        let func = self
            .module
            .get_function("ad_series_f32")
            .map_err(|_| CudaAdError::MissingKernelSymbol { name: "ad_series_f32" })?;

        let block_x: u32 = 256;
        let grid_x = ((n_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch((grid_x.max(1), 1, 1), (block_x, 1, 1))?;

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut len_i: i32 = len
                .try_into()
                .map_err(|_| CudaAdError::InvalidInput("length exceeds i32".into()))?;
            let mut n_i: i32 = n_series
                .try_into()
                .map_err(|_| CudaAdError::InvalidInput("n_series exceeds i32".into()))?;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaAdError::Cuda)?;
        }
        Ok(())
    }

    pub fn ad_series_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
    ) -> Result<DeviceArrayF32, CudaAdError> {
        let len = Self::validate_hlcv(high, low, close, volume)?;

        // Rough VRAM estimate: 4 inputs + 1 output, plus headroom
        let bytes_inputs = 4usize
            .checked_mul(len)
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        let bytes_output = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_output)
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        // Pinned host buffers for async HtoD copies
        let h_high = LockedBuffer::from_slice(high).map_err(CudaAdError::Cuda)?;
        let h_low = LockedBuffer::from_slice(low).map_err(CudaAdError::Cuda)?;
        let h_close = LockedBuffer::from_slice(close).map_err(CudaAdError::Cuda)?;
        let h_vol = LockedBuffer::from_slice(volume).map_err(CudaAdError::Cuda)?;

        let mut d_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(CudaAdError::Cuda)?;
        let mut d_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(CudaAdError::Cuda)?;
        let mut d_close: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(CudaAdError::Cuda)?;
        let mut d_vol: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(CudaAdError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(CudaAdError::Cuda)?;

        // Async copies on the stream
        unsafe {
            d_high.async_copy_from(h_high.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_low.async_copy_from(h_low.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_close.async_copy_from(h_close.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_vol.async_copy_from(h_vol.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
        }

        self.launch_series_kernel(&d_high, &d_low, &d_close, &d_vol, len, 1, &mut d_out)?;
        self.stream.synchronize().map_err(CudaAdError::Cuda)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    // --- Many-series, time-major ---
    fn launch_many_series_kernel(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        d_volume_tm: &DeviceBuffer<f32>,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdError> {
        let func = self
            .module
            .get_function("ad_many_series_one_param_time_major_f32")
            .map_err(|_| CudaAdError::MissingKernelSymbol { name: "ad_many_series_one_param_time_major_f32" })?;

        // Time-major fast path: one thread per series with coalesced loads across a warp.
        // Use a larger block so many threads process many series concurrently.
        let block_x: u32 = 256; // should match/default to AD_BLOCK_SIZE_TM in the kernel
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch((grid_x.max(1), 1, 1), (block_x, 1, 1))?;

        unsafe {
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut close_ptr = d_close_tm.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume_tm.as_device_ptr().as_raw();
            let mut n_i: i32 = num_series
                .try_into()
                .map_err(|_| CudaAdError::InvalidInput("num_series exceeds i32 range".into()))?;
            let mut len_i: i32 = series_len
                .try_into()
                .map_err(|_| CudaAdError::InvalidInput("series_len exceeds i32 range".into()))?;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaAdError::Cuda)?;
        }
        Ok(())
    }

    pub fn ad_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaAdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAdError::InvalidInput(
                "cols and rows must be > 0".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAdError::InvalidInput("cols*rows overflow".into()))?;
        if high_tm.len() != elems
            || low_tm.len() != elems
            || close_tm.len() != elems
            || volume_tm.len() != elems
        {
            return Err(CudaAdError::InvalidInput(
                "time-major buffers must be cols*rows in length".into(),
            ));
        }

        // VRAM estimate: 4 inputs + 1 output
        let bytes_inputs = 4usize
            .checked_mul(elems)
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        let bytes_output = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_output)
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        // Pinned host + async copies
        let h_high = LockedBuffer::from_slice(high_tm).map_err(CudaAdError::Cuda)?;
        let h_low = LockedBuffer::from_slice(low_tm).map_err(CudaAdError::Cuda)?;
        let h_close = LockedBuffer::from_slice(close_tm).map_err(CudaAdError::Cuda)?;
        let h_vol = LockedBuffer::from_slice(volume_tm).map_err(CudaAdError::Cuda)?;
        let mut d_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(CudaAdError::Cuda)?;
        let mut d_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(CudaAdError::Cuda)?;
        let mut d_close: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(CudaAdError::Cuda)?;
        let mut d_vol: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(CudaAdError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(CudaAdError::Cuda)?;

        unsafe {
            d_high.async_copy_from(h_high.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_low.async_copy_from(h_low.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_close.async_copy_from(h_close.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_vol.async_copy_from(h_vol.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
        }

        self.launch_many_series_kernel(&d_high, &d_low, &d_close, &d_vol, cols, rows, &mut d_out)?;
        // Synchronize the producing stream so Python __cuda_array_interface__ can omit 'stream'.
        self.stream.synchronize().map_err(CudaAdError::Cuda)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows, // time-major rows
            cols, // number of series
        })
    }

    /// Expose the internal CUDA stream for async pipelines.
    #[inline]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Async variant: single-series; no forced synchronize.
    pub fn ad_series_dev_async(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
    ) -> Result<DeviceArrayF32, CudaAdError> {
        let len = Self::validate_hlcv(high, low, close, volume)?;

        let bytes_inputs = 4usize
            .checked_mul(len)
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        let bytes_output = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_output)
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let h_high = LockedBuffer::from_slice(high).map_err(CudaAdError::Cuda)?;
        let h_low = LockedBuffer::from_slice(low).map_err(CudaAdError::Cuda)?;
        let h_close = LockedBuffer::from_slice(close).map_err(CudaAdError::Cuda)?;
        let h_vol = LockedBuffer::from_slice(volume).map_err(CudaAdError::Cuda)?;

        let mut d_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaAdError::Cuda)?;
        let mut d_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaAdError::Cuda)?;
        let mut d_close: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaAdError::Cuda)?;
        let mut d_vol: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaAdError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaAdError::Cuda)?;

        unsafe {
            d_high.async_copy_from(h_high.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_low.async_copy_from(h_low.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_close.async_copy_from(h_close.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_vol.async_copy_from(h_vol.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
        }
        self.launch_series_kernel(&d_high, &d_low, &d_close, &d_vol, len, 1, &mut d_out)?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    /// Async variant: many-series time-major; no forced synchronize.
    pub fn ad_many_series_one_param_time_major_dev_async(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaAdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAdError::InvalidInput(
                "cols and rows must be > 0".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAdError::InvalidInput("cols*rows overflow".into()))?;
        if high_tm.len() != elems
            || low_tm.len() != elems
            || close_tm.len() != elems
            || volume_tm.len() != elems
        {
            return Err(CudaAdError::InvalidInput(
                "time-major buffers must be cols*rows in length".into(),
            ));
        }

        let bytes_inputs = 4usize
            .checked_mul(elems)
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        let bytes_output = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_output)
            .ok_or_else(|| CudaAdError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let h_high = LockedBuffer::from_slice(high_tm).map_err(CudaAdError::Cuda)?;
        let h_low = LockedBuffer::from_slice(low_tm).map_err(CudaAdError::Cuda)?;
        let h_close = LockedBuffer::from_slice(close_tm).map_err(CudaAdError::Cuda)?;
        let h_vol = LockedBuffer::from_slice(volume_tm).map_err(CudaAdError::Cuda)?;
        let mut d_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaAdError::Cuda)?;
        let mut d_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaAdError::Cuda)?;
        let mut d_close: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaAdError::Cuda)?;
        let mut d_vol: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaAdError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaAdError::Cuda)?;

        unsafe {
            d_high.async_copy_from(h_high.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_low.async_copy_from(h_low.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_close.async_copy_from(h_close.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
            d_vol.async_copy_from(h_vol.as_slice(), &self.stream).map_err(CudaAdError::Cuda)?;
        }
        self.launch_many_series_kernel(&d_high, &d_low, &d_close, &d_vol, cols, rows, &mut d_out)?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    /// Device-to-device overload: row-major.
    pub fn ad_series_device_inplace(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_volume: &DeviceBuffer<f32>,
        len: usize,
        n_series: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdError> {
        self.launch_series_kernel(d_high, d_low, d_close, d_volume, len, n_series, d_out)
    }

    /// Device-to-device overload: time-major many-series.
    pub fn ad_many_series_one_param_time_major_device_inplace(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        d_volume_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdError> {
        self.launch_many_series_kernel(
            d_high_tm,
            d_low_tm,
            d_close_tm,
            d_volume_tm,
            cols,
            rows,
            d_out_tm,
        )
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 200;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series() -> usize {
        // 4 inputs + 1 output
        let in_bytes = 4 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 32 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = 4 * elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlcv_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        let mut vol = vec![0.0f32; close.len()];
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0021;
            let off = (0.0033 * x.cos()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
            vol[i] = ((x * 0.71).sin().abs() + 0.9) * 1500.0;
        }
        (high, low, vol)
    }

    struct OneSeriesState {
        cuda: CudaAd,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        vol: Vec<f32>,
    }
    impl CudaBenchState for OneSeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .ad_series_dev(&self.high, &self.low, &self.close, &self.vol)
                .expect("ad series");
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaAd::new(0).expect("cuda ad");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low, vol) = synth_hlcv_from_close(&close);
        Box::new(OneSeriesState {
            cuda,
            high,
            low,
            close,
            vol,
        })
    }

    struct ManyState {
        cuda: CudaAd,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        vol_tm: Vec<f32>,
        cols: usize,
        rows: usize,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .ad_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    &self.vol_tm,
                    self.cols,
                    self.rows,
                )
                .expect("ad many");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaAd::new(0).expect("cuda ad");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let prices_tm = gen_time_major_prices(cols, rows);
        let mut high_tm = prices_tm.clone();
        let mut low_tm = prices_tm.clone();
        let mut vol_tm = vec![0f32; prices_tm.len()];
        for t in 0..rows {
            for s in 0..cols {
                let idx = t * cols + s;
                let v = prices_tm[idx];
                let x = (t as f32) * 0.0019 + (s as f32) * 0.03;
                let off = (0.0027 * x.sin()).abs() + 0.11;
                high_tm[idx] = v + off;
                low_tm[idx] = v - off;
                vol_tm[idx] = ((x * 1.13).cos().abs() + 0.85) * 1200.0;
            }
        }
        Box::new(ManyState {
            cuda,
            high_tm,
            low_tm,
            close_tm: prices_tm,
            vol_tm,
            cols,
            rows,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("ad", "one_series", "ad_cuda_series", "1m", prep_one_series)
                .with_sample_size(10)
                .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "ad",
                "many_series_one_param",
                "ad_cuda_many_series_time_major",
                "200x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
