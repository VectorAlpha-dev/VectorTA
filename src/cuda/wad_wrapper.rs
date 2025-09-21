#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaWadError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWadError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWadError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaWadError {}

pub struct CudaWad {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaWad {
    pub fn new(device_id: usize) -> Result<Self, CudaWadError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wad_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn validate_inputs(high: &[f32], low: &[f32], close: &[f32]) -> Result<usize, CudaWadError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaWadError::InvalidInput("empty input slices".into()));
        }
        let len = high.len();
        if low.len() != len || close.len() != len {
            return Err(CudaWadError::InvalidInput(
                "input slice length mismatch".into(),
            ));
        }
        let all_nan_high = high.iter().all(|v| v.is_nan());
        let all_nan_low = low.iter().all(|v| v.is_nan());
        let all_nan_close = close.iter().all(|v| v.is_nan());
        if all_nan_high || all_nan_low || all_nan_close {
            return Err(CudaWadError::InvalidInput(
                "all values are NaN in one of the inputs".into(),
            ));
        }
        Ok(len)
    }

    fn launch_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        n_series: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWadError> {
        let func = self
            .module
            .get_function("wad_series_f32")
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        let block_x: u32 = 256;
        let grid_x = ((n_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut series_i = n_series as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut series_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_single_series(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<(DeviceBuffer<f32>, usize), CudaWadError> {
        let len = Self::validate_inputs(high, low, close)?;

        let d_high =
            DeviceBuffer::from_slice(high).map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low).map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        self.launch_series_kernel(&d_high, &d_low, &d_close, len, 1, &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;

        Ok((d_out, len))
    }

    pub fn wad_series_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DeviceArrayF32, CudaWadError> {
        let (buf, len) = self.run_single_series(high, low, close)?;
        Ok(DeviceArrayF32 {
            buf,
            rows: 1,
            cols: len,
        })
    }

    pub fn wad_into_host_f32(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        out: &mut [f32],
    ) -> Result<usize, CudaWadError> {
        let (buf, len) = self.run_single_series(high, low, close)?;
        if out.len() != len {
            return Err(CudaWadError::InvalidInput(format!(
                "output slice length mismatch (expected {}, got {})",
                len,
                out.len()
            )));
        }
        buf.copy_to(out)
            .map_err(|e| CudaWadError::Cuda(e.to_string()))?;
        Ok(len)
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::gen_series;

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
            let x = i as f32 * 0.0027;
            let off = (0.0031 * x.cos()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct WadState {
        cuda: CudaWad,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
    }
    impl CudaBenchState for WadState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .wad_series_dev(&self.high, &self.low, &self.close)
                .expect("wad kernel");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaWad::new(0).expect("cuda wad");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        Box::new(WadState { cuda, high, low, close })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "wad",
                "one_series",
                "wad_cuda_series",
                "1m",
                prep_one_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series()),
        ]
    }
}
