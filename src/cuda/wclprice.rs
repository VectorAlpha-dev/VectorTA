//! CUDA scaffolding for the WCLPRICE (Weighted Close Price) indicator.
//!
//! The GPU implementation mirrors the scalar CPU path: given high/low/close
//! price slices it writes the weighted close `(high + low + 2 * close) / 4`
//! into the output buffer, preserving NaN semantics and respecting the
//! calculated warm-up prefix.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaWclpriceError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWclpriceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWclpriceError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWclpriceError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaWclpriceError {}

pub struct CudaWclprice {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaWclprice {
    pub fn new(device_id: usize) -> Result<Self, CudaWclpriceError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wclprice_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn wclprice_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DeviceArrayF32, CudaWclpriceError> {
        let len = high.len().min(low.len()).min(close.len());
        if len == 0 {
            return Err(CudaWclpriceError::InvalidInput("empty input".into()));
        }

        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
            .ok_or_else(|| CudaWclpriceError::InvalidInput("all values are NaN".into()))?;

        let high_buf = DeviceBuffer::from_slice(&high[..len])
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let low_buf = DeviceBuffer::from_slice(&low[..len])
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let close_buf = DeviceBuffer::from_slice(&close[..len])
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let mut out_buf: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        self.wclprice_device(
            &high_buf,
            &low_buf,
            &close_buf,
            len,
            first_valid,
            &mut out_buf,
        )?;

        Ok(DeviceArrayF32 {
            buf: out_buf,
            rows: 1,
            cols: len,
        })
    }

    pub fn wclprice_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWclpriceError> {
        if len == 0 {
            return Err(CudaWclpriceError::InvalidInput(
                "len must be positive".into(),
            ));
        }

        let func = self
            .module
            .get_function("wclprice_kernel_f32")
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        let block_dim: u32 = 256;
        let mut grid_x = ((len as u32) + block_dim - 1) / block_dim;
        if grid_x == 0 {
            grid_x = 1;
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_dim, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_valid_i = (first_valid.min(len)) as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
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

    fn bytes_one_series() -> usize {
        // 3 inputs (H/L/C) + 1 output
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 32 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct WclState {
        cuda: CudaWclprice,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
    }
    impl CudaBenchState for WclState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .wclprice_dev(&self.high, &self.low, &self.close)
                .expect("wclprice kernel");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaWclprice::new(0).expect("cuda wclprice");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        Box::new(WclState {
            cuda,
            high,
            low,
            close,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "wclprice",
            "one_series",
            "wclprice_cuda_series",
            "1m",
            prep_one_series,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series())]
    }
}
