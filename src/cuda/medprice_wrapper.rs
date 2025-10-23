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
use std::ffi::c_void;
use std::fmt;
use cust::sys as cu; // raw driver API for SM count and low-level opts

#[derive(Debug)]
pub enum CudaMedpriceError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaMedpriceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMedpriceError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMedpriceError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaMedpriceError {}

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
    _context: Context,
    batch_policy: BatchKernelPolicy,
    many_policy: ManySeriesKernelPolicy,
    sm_count: u32,
}

impl CudaMedprice {
    pub fn new(device_id: usize) -> Result<Self, CudaMedpriceError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/medprice_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            // Keep the JIT at maximum optimization for these tiny kernels.
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(e) => return Err(CudaMedpriceError::Cuda(e.to_string())),
            },
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;

        let sm_count = sm_count_from_current_ctx()?;

        Ok(Self {
            module,
            stream,
            _context: context,
            batch_policy: BatchKernelPolicy::Auto,
            many_policy: ManySeriesKernelPolicy::Auto,
            sm_count,
        })
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

        let d_high = DeviceBuffer::from_slice(&high[..len])
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(&low[..len])
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;

        self.medprice_device(&d_high, &d_low, len, first_valid, &mut d_out)?;

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
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;

        // SM-aware grid sizing for grid-stride loop
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

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    // -------- Batch: one series × many params (rows=1 for medprice) --------
    pub fn medprice_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
    ) -> Result<DeviceArrayF32, CudaMedpriceError> {
        let len = high.len().min(low.len());
        if len == 0 { return Err(CudaMedpriceError::InvalidInput("empty input".into())); }
        let _first = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan())
            .ok_or_else(|| CudaMedpriceError::InvalidInput("all values are NaN".into()))?;

        // VRAM check (2 inputs + 1 output + 1 i32 first_valid + 64MB headroom)
        if let Ok((free, _)) = mem_get_info() {
            let need: usize = 2 * len * std::mem::size_of::<f32>()
                + len * std::mem::size_of::<f32>()
                + std::mem::size_of::<i32>()
                + 64 * 1024 * 1024;
            if need > free {
                return Err(CudaMedpriceError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_high = DeviceBuffer::from_slice(&high[..len])
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(&low[..len])
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("medprice_batch_f32")
            .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        // rows=1; grid.y <= 65_535 satisfied
        let block_x = match self.batch_policy { BatchKernelPolicy::Auto => 256, BatchKernelPolicy::Plain{block_x} => block_x.max(32) };
        let (grid, block) = grid_1d_for(len, block_x, self.sm_count);

        // Skip building first_valids; pass null pointer to kernel (fv=0 fallback is correct for MEDPRICE)
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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        }

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    // -------- Many-series × one-param (time-major) --------
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
        let n = cols * rows;
        if high_tm.len() < n || low_tm.len() < n {
            return Err(CudaMedpriceError::InvalidInput(
                "input size mismatch".into(),
            ));
        }

        // VRAM check: 2*n inputs + n out + 64MB headroom (no first_valids allocation)
        if let Ok((free, _)) = mem_get_info() {
            let need: usize = 3 * n * std::mem::size_of::<f32>()
                + 64 * 1024 * 1024;
            if need > free {
                return Err(CudaMedpriceError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_high = DeviceBuffer::from_slice(&high_tm[..n]).map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(&low_tm[..n]).map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }.map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;

        let func = self.module.get_function("medprice_many_series_one_param_f32").map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        let block_x = match self.many_policy { ManySeriesKernelPolicy::Auto => 256, ManySeriesKernelPolicy::OneD{block_x} => block_x.max(32) };
        let (grid, block) = grid_1d_for(cols, block_x, self.sm_count);

        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            // Pass null for first_valids; kernel falls back to fv=0 (correct for MEDPRICE)
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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaMedpriceError::Cuda(e.to_string()))?;
        }

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series() -> usize {
        // 2 inputs + 1 output + 64MB headroom
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
        high: Vec<f32>,
        low: Vec<f32>,
    }
    impl CudaBenchState for MedState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .medprice_dev(&self.high, &self.low)
                .expect("medprice kernel");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaMedprice::new(0).expect("cuda medprice");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_close(&close);
        Box::new(MedState { cuda, high, low })
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

// ---- Private helpers ----

fn sm_count_from_current_ctx() -> Result<u32, CudaMedpriceError> {
    unsafe {
        let mut dev: cu::CUdevice = 0;
        let r1 = cu::cuCtxGetDevice(&mut dev as *mut _);
        if r1 != cu::CUresult::CUDA_SUCCESS {
            return Err(CudaMedpriceError::Cuda(format!(
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
            return Err(CudaMedpriceError::Cuda(format!(
                "cuDeviceGetAttribute(MP_COUNT) failed: {:?}", r2
            )));
        }
        Ok(sms as u32)
    }
}

#[inline]
fn grid_1d_for(n: usize, block_x: u32, sm_count: u32) -> (GridSize, BlockSize) {
    // Heuristic: a few blocks per SM is typically sufficient for grid‑stride loops.
    let full = ((n as u32).saturating_add(block_x - 1)) / block_x;
    let cap = sm_count.saturating_mul(4).max(1);
    let gx = full.min(cap).max(1);
    ((gx, 1, 1).into(), (block_x, 1, 1).into())
}
