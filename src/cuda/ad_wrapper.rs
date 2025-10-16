#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaAdError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAdError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAdError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaAdError {}

pub struct CudaAd {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaAd {
    pub fn new(device_id: usize) -> Result<Self, CudaAdError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAdError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ad_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAdError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
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
            .map_err(|e| CudaAdError::Cuda(e.to_string()))?;

        let block_x: u32 = 256;
        let grid_x = ((n_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n_series as i32;
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
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
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

        // Pinned host buffers for async HtoD copies
        let h_high = LockedBuffer::from_slice(high).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let h_low = LockedBuffer::from_slice(low).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let h_close =
            LockedBuffer::from_slice(close).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let h_vol = LockedBuffer::from_slice(volume).map_err(|e| CudaAdError::Cuda(e.to_string()))?;

        let mut d_high: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_low: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_close: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_vol: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;

        // Async copies on the stream
        unsafe {
            d_high
                .async_copy_from(h_high.as_slice(), &self.stream)
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
            d_low
                .async_copy_from(h_low.as_slice(), &self.stream)
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
            d_close
                .async_copy_from(h_close.as_slice(), &self.stream)
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
            d_vol
                .async_copy_from(h_vol.as_slice(), &self.stream)
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        }

        self.launch_series_kernel(&d_high, &d_low, &d_close, &d_vol, len, 1, &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaAdError::Cuda(e.to_string()))?;

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
            .map_err(|e| CudaAdError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (32, 1, 1).into(); // thread 0 performs the scan

        unsafe {
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut close_ptr = d_close_tm.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume_tm.as_device_ptr().as_raw();
            let mut n_i = num_series as i32;
            let mut len_i = series_len as i32;
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
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
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
            return Err(CudaAdError::InvalidInput("cols and rows must be > 0".into()));
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

        // Pinned host + async copies
        let h_high = LockedBuffer::from_slice(high_tm).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let h_low = LockedBuffer::from_slice(low_tm).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let h_close =
            LockedBuffer::from_slice(close_tm).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let h_vol = LockedBuffer::from_slice(volume_tm).map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_high: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_low: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_close: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_vol: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaAdError::Cuda(e.to_string()))?;

        unsafe {
            d_high
                .async_copy_from(h_high.as_slice(), &self.stream)
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
            d_low
                .async_copy_from(h_low.as_slice(), &self.stream)
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
            d_close
                .async_copy_from(h_close.as_slice(), &self.stream)
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
            d_vol
                .async_copy_from(h_vol.as_slice(), &self.stream)
                .map_err(|e| CudaAdError::Cuda(e.to_string()))?;
        }

        self.launch_many_series_kernel(&d_high, &d_low, &d_close, &d_vol, cols, rows, &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaAdError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,       // time-major rows
            cols,       // number of series
        })
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
            if v.is_nan() { continue; }
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
        Box::new(OneSeriesState { cuda, high, low, close, vol })
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
        Box::new(ManyState { cuda, high_tm, low_tm, close_tm: prices_tm, vol_tm, cols, rows })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "ad",
                "one_series",
                "ad_cuda_series",
                "1m",
                prep_one_series,
            )
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
