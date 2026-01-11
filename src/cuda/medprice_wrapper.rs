//! CUDA scaffolding for the MEDPRICE (Median Price) indicator.
//!
//! Mirrors the scalar CPU path: given `high` and `low` slices it writes
//! `(high + low) * 0.5` into the output buffer. Warmup/NaN semantics match
//! scalar: indices before the first valid element are NaN; any NaN input
//! yields NaN at that position.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu; 
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaMedpriceError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

pub struct CudaMedprice {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    batch_policy: BatchKernelPolicy,
    many_policy: ManySeriesKernelPolicy,
    sm_count: u32,
}

impl CudaMedprice {
    pub fn new(device_id: usize) -> Result<Self, CudaMedpriceError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/medprice_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let sm_count = sm_count_from_current_ctx()?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            batch_policy: BatchKernelPolicy::Auto,
            many_policy: ManySeriesKernelPolicy::Auto,
            sm_count,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaMedpriceError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMedpriceError> {
        if let Ok((free, _total)) = mem_get_info() {
            let total = required_bytes.saturating_add(headroom_bytes);
            if total > free {
                return Err(CudaMedpriceError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    pub fn medprice_dev(
        &self,
        high: &[f32],
        low: &[f32],
    ) -> Result<DeviceArrayF32, CudaMedpriceError> {
        let len = high.len().min(low.len());
        if len == 0 {
            return Err(CudaMedpriceError::InvalidInput("empty input".into()));
        }

        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan())
            .ok_or_else(|| CudaMedpriceError::InvalidInput("all values are NaN".into()))?;

        
        let elem = std::mem::size_of::<f32>();
        let in_bytes = len
            .checked_mul(2)
            .and_then(|n| n.checked_mul(elem))
            .ok_or_else(|| CudaMedpriceError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = len
            .checked_mul(elem)
            .ok_or_else(|| CudaMedpriceError::InvalidInput("byte size overflow".into()))?;
        let required = in_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaMedpriceError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_high = DeviceBuffer::from_slice(&high[..len])?;
        let d_low = DeviceBuffer::from_slice(&low[..len])?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;

        self.medprice_device(&d_high, &d_low, len, first_valid, &mut d_out)?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    pub fn medprice_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMedpriceError> {
        if len == 0 {
            return Err(CudaMedpriceError::InvalidInput(
                "len must be positive".into(),
            ));
        }

        let func = self
            .module
            .get_function("medprice_kernel_f32")
            .map_err(|_| CudaMedpriceError::MissingKernelSymbol { name: "medprice_kernel_f32" })?;

        
        let block_x: u32 = 256;
        let (grid, block) = grid_1d_for(len, block_x, self.sm_count);

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = (first_valid.min(len)) as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    
    pub fn medprice_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
    ) -> Result<DeviceArrayF32, CudaMedpriceError> {
        let len = high.len().min(low.len());
        if len == 0 {
            return Err(CudaMedpriceError::InvalidInput("empty input".into()));
        }
        let _first = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan())
            .ok_or_else(|| CudaMedpriceError::InvalidInput("all values are NaN".into()))?;

        
        let elem = std::mem::size_of::<f32>();
        let in_bytes = len
            .checked_mul(2)
            .and_then(|n| n.checked_mul(elem))
            .ok_or_else(|| CudaMedpriceError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = len
            .checked_mul(elem)
            .ok_or_else(|| CudaMedpriceError::InvalidInput("byte size overflow".into()))?;
        let meta_bytes = std::mem::size_of::<i32>();
        let required = in_bytes
            .checked_add(out_bytes)
            .and_then(|b| b.checked_add(meta_bytes))
            .ok_or_else(|| CudaMedpriceError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_high = DeviceBuffer::from_slice(&high[..len])?;
        let d_low = DeviceBuffer::from_slice(&low[..len])?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;

        let func = self
            .module
            .get_function("medprice_batch_f32")
            .map_err(|_| CudaMedpriceError::MissingKernelSymbol { name: "medprice_batch_f32" })?;
        
        let block_x = match self.batch_policy { BatchKernelPolicy::Auto => 256, BatchKernelPolicy::Plain{block_x} => block_x.max(32) };
        let (grid, block) = grid_1d_for(len, block_x, self.sm_count);

        
        let mut fv_ptr: u64 = 0;

        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut rows_i = 1i32;
            let mut fv = fv_ptr;
            let mut out_p = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut out_p as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows: 1, cols: len })
    }

    
    pub fn medprice_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaMedpriceError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMedpriceError::InvalidInput(
                "cols/rows must be > 0".into(),
            ));
        }
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMedpriceError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() < n || low_tm.len() < n {
            return Err(CudaMedpriceError::InvalidInput(
                "input size mismatch".into(),
            ));
        }

        
        let elem = std::mem::size_of::<f32>();
        let bytes_inputs_outputs = 3usize
            .checked_mul(n)
            .and_then(|m| m.checked_mul(elem))
            .ok_or_else(|| CudaMedpriceError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(bytes_inputs_outputs, 64 * 1024 * 1024)?;

        let d_high = DeviceBuffer::from_slice(&high_tm[..n])?;
        let d_low = DeviceBuffer::from_slice(&low_tm[..n])?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;

        let func = self
            .module
            .get_function("medprice_many_series_one_param_f32")
            .map_err(|_| CudaMedpriceError::MissingKernelSymbol { name: "medprice_many_series_one_param_f32" })?;
        let block_x = match self.many_policy { ManySeriesKernelPolicy::Auto => 256, ManySeriesKernelPolicy::OneD{block_x} => block_x.max(32) };
        let (grid, block) = grid_1d_for(cols, block_x, self.sm_count);

        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            
            let mut fv: u64 = 0;
            let mut out_p = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut out_p as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}



pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series() -> usize {
        
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0021;
            let off = (0.002 * x.sin()).abs() + 0.10;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct MedState {
        cuda: CudaMedprice,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MedState {
        fn launch(&mut self) {
            self.cuda
                .medprice_device(
                    &self.d_high,
                    &self.d_low,
                    self.len,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("medprice kernel");
            self.cuda.synchronize().expect("medprice sync");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaMedprice::new(0).expect("cuda medprice");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_close(&close);
        let len = high.len().min(low.len());
        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan())
            .unwrap_or(0);
        let d_high = DeviceBuffer::from_slice(&high[..len]).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&low[..len]).expect("d_low");
        let d_out = unsafe { DeviceBuffer::<f32>::uninitialized(len) }.expect("d_out");
        Box::new(MedState {
            cuda,
            d_high,
            d_low,
            len,
            first_valid,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "medprice",
            "one_series",
            "medprice_cuda_series",
            "1m",
            prep_one_series,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series())]
    }
}



fn sm_count_from_current_ctx() -> Result<u32, CudaMedpriceError> {
    unsafe {
        let mut dev: cu::CUdevice = 0;
        let r1 = cu::cuCtxGetDevice(&mut dev as *mut _);
        if r1 != cu::CUresult::CUDA_SUCCESS {
            return Err(CudaMedpriceError::InvalidInput(format!(
                "cuCtxGetDevice failed: {:?}", r1
            )));
        }
        let mut sms: std::os::raw::c_int = 0;
        let r2 = cu::cuDeviceGetAttribute(
            &mut sms as *mut _,
            cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            dev,
        );
        if r2 != cu::CUresult::CUDA_SUCCESS {
            return Err(CudaMedpriceError::InvalidInput(format!(
                "cuDeviceGetAttribute(MP_COUNT) failed: {:?}", r2
            )));
        }
        Ok(sms as u32)
    }
}

#[inline]
fn grid_1d_for(n: usize, block_x: u32, sm_count: u32) -> (GridSize, BlockSize) {
    
    let full = ((n as u32).saturating_add(block_x - 1)) / block_x;
    let cap = sm_count.saturating_mul(4).max(1);
    let gx = full.min(cap).max(1);
    ((gx, 1, 1).into(), (block_x, 1, 1).into())
}
