#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaVptError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaVpt {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    sm_count: u32,
    block_x: u32,
}

impl CudaVpt {
    pub fn new(device_id: usize) -> Result<Self, CudaVptError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let ctx = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vpt_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let block_x = 256u32;

        Ok(Self {
            module,
            stream,
            ctx,
            device_id: device_id as u32,
            sm_count,
            block_x,
        })
    }

    #[inline]
    pub fn context(&self) -> Arc<Context> {
        self.ctx.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn first_valid_pair(price: &[f32], volume: &[f32]) -> Result<usize, CudaVptError> {
        if price.is_empty() || volume.is_empty() {
            return Err(CudaVptError::InvalidInput("empty inputs".into()));
        }
        if price.len() != volume.len() {
            return Err(CudaVptError::InvalidInput("length mismatch".into()));
        }
        for i in 1..price.len() {
            let p0 = price[i - 1];
            let p1 = price[i];
            let v1 = volume[i];
            if p0.is_finite() && p0 != 0.0 && p1.is_finite() && v1.is_finite() {
                return Ok(i);
            }
        }
        Err(CudaVptError::InvalidInput(
            "not enough valid data (need a valid pair i-1,i)".into(),
        ))
    }

    #[inline]
    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVptError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVptError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    pub fn vpt_batch_dev(
        &self,
        price: &[f32],
        volume: &[f32],
    ) -> Result<DeviceArrayF32, CudaVptError> {
        let len = price.len().min(volume.len());
        if len == 0 {
            return Err(CudaVptError::InvalidInput("empty input".into()));
        }
        if price.len() != volume.len() {
            return Err(CudaVptError::InvalidInput("length mismatch".into()));
        }

        let first = Self::first_valid_pair(price, volume)?;

        let el = std::mem::size_of::<f32>();
        let bytes = len
            .checked_mul(3)
            .and_then(|x| x.checked_mul(el))
            .ok_or_else(|| CudaVptError::InvalidInput("size overflow".into()))?;
        Self::will_fit(bytes, 64 << 20)?;

        let d_price = DeviceBuffer::from_slice(price)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;

        let func = self.module.get_function("vpt_batch_f32").map_err(|_| {
            CudaVptError::MissingKernelSymbol {
                name: "vpt_batch_f32",
            }
        })?;
        let stream = &self.stream;
        unsafe {
            launch!(func<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                d_price.as_device_ptr(),
                d_volume.as_device_ptr(),
                len as i32,
                first as i32,
                d_out.as_device_ptr()
            ))?
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    pub fn vpt_many_series_one_param_time_major_dev(
        &self,
        price_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaVptError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVptError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaVptError::InvalidInput("rows*cols overflow".into()))?;
        if price_tm.len() != expected || volume_tm.len() != expected {
            return Err(CudaVptError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }

        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 1..rows {
                let p0 = price_tm[(t - 1) * cols + s];
                let p1 = price_tm[t * cols + s];
                let v1 = volume_tm[t * cols + s];
                if p0.is_finite() && p0 != 0.0 && p1.is_finite() && v1.is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let el_f32 = std::mem::size_of::<f32>();
        let el_i32 = std::mem::size_of::<i32>();
        let bytes_inputs_outputs = 3usize
            .checked_mul(expected)
            .and_then(|x| x.checked_mul(el_f32))
            .ok_or_else(|| CudaVptError::InvalidInput("size overflow".into()))?;
        let bytes_first = cols
            .checked_mul(el_i32)
            .ok_or_else(|| CudaVptError::InvalidInput("size overflow".into()))?;
        let required = bytes_inputs_outputs
            .checked_add(bytes_first)
            .ok_or_else(|| CudaVptError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 << 20)?;

        let d_price = DeviceBuffer::from_slice(price_tm)?;
        let d_volume = DeviceBuffer::from_slice(volume_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }?;

        let func = self
            .module
            .get_function("vpt_many_series_one_param_f32")
            .map_err(|_| CudaVptError::MissingKernelSymbol {
                name: "vpt_many_series_one_param_f32",
            })?;

        let block_x = self.block_x;
        let mut grid_x = ((cols as u32) + block_x - 1) / block_x;
        let max_blocks = self.sm_count.saturating_mul(16);
        if grid_x > max_blocks {
            grid_x = max_blocks.max(1);
        }
        let stream = &self.stream;
        unsafe {
            launch!(func<<<(grid_x, 1, 1), (block_x, 1, 1), 0, stream>>>(
                d_price.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                rows as i32,
                d_first.as_device_ptr(),
                d_out.as_device_ptr()
            ))?
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use cust::function::{BlockSize, GridSize};
    use std::ffi::c_void;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 512;
    const MANY_SERIES_ROWS: usize = 8_192;

    fn bytes_one_series() -> usize {
        (3 * ONE_SERIES_LEN * std::mem::size_of::<f32>()) + (64 << 20)
    }
    fn bytes_many_series() -> usize {
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        (3 * n * std::mem::size_of::<f32>())
            + (MANY_SERIES_COLS * std::mem::size_of::<i32>())
            + (64 << 20)
    }

    struct OneSeriesState {
        cuda: CudaVpt,
        d_price: DeviceBuffer<f32>,
        d_volume: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first: usize,
    }
    impl CudaBenchState for OneSeriesState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("vpt_batch_f32")
                .expect("vpt_batch_f32");
            let stream = &self.cuda.stream;
            let grid: GridSize = (1, 1, 1).into();
            let block: BlockSize = (1, 1, 1).into();
            unsafe {
                let mut price_ptr = self.d_price.as_device_ptr().as_raw();
                let mut vol_ptr = self.d_volume.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut first_i = self.first as i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut price_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                stream.launch(&func, grid, block, 0, args).expect("launch");
            }
            self.cuda.stream.synchronize().expect("vpt sync");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaVpt::new(0).expect("cuda vpt");
        let mut price = gen_series(ONE_SERIES_LEN);
        let mut volume = gen_series(ONE_SERIES_LEN);

        if !price[1].is_finite() || price[0] == 0.0 || !volume[1].is_finite() {
            price[0] = 100.0;
            price[1] = 100.1;
            volume[1] = 500.0;
        }
        let first = CudaVpt::first_valid_pair(&price, &volume).expect("first_valid_pair");
        let d_price = DeviceBuffer::from_slice(&price).expect("d_price");
        let d_volume = DeviceBuffer::from_slice(&volume).expect("d_volume");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(ONE_SERIES_LEN) }.expect("d_out");
        cuda.stream.synchronize().expect("vpt prep sync");
        Box::new(OneSeriesState {
            cuda,
            d_price,
            d_volume,
            d_out,
            len: ONE_SERIES_LEN,
            first,
        })
    }

    struct ManySeriesState {
        cuda: CudaVpt,
        d_price: DeviceBuffer<f32>,
        d_volume: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        grid: GridSize,
        block: BlockSize,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("vpt_many_series_one_param_f32")
                .expect("vpt_many_series_one_param_f32");
            let stream = &self.cuda.stream;
            unsafe {
                let mut price_ptr = self.d_price.as_device_ptr().as_raw();
                let mut vol_ptr = self.d_volume.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut first_ptr = self.d_first.as_device_ptr().as_raw();
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut price_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                stream
                    .launch(&func, self.grid, self.block, 0, args)
                    .expect("launch");
            }
            self.cuda.stream.synchronize().expect("vpt sync");
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaVpt::new(0).expect("cuda vpt");
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let mut price_tm = vec![f32::NAN; n];
        let mut volume_tm = vec![f32::NAN; n];
        for s in 0..MANY_SERIES_COLS {
            for t in s.min(8)..MANY_SERIES_ROWS {
                let x = (t as f32) + (s as f32) * 0.13;
                price_tm[t * MANY_SERIES_COLS + s] = (x * 0.0021).sin() + 0.0002 * x + 100.0;
                volume_tm[t * MANY_SERIES_COLS + s] = (x * 0.0017).cos().abs() * 500.0 + 100.0;
            }
        }
        let (cols, rows) = (MANY_SERIES_COLS, MANY_SERIES_ROWS);
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 1..rows {
                let p0 = price_tm[(t - 1) * cols + s];
                let p1 = price_tm[t * cols + s];
                let v1 = volume_tm[t * cols + s];
                if p0.is_finite() && p0 != 0.0 && p1.is_finite() && v1.is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let d_price = DeviceBuffer::from_slice(&price_tm).expect("d_price_tm");
        let d_volume = DeviceBuffer::from_slice(&volume_tm).expect("d_volume_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        let block_x = cuda.block_x;
        let mut grid_x = ((cols as u32) + block_x - 1) / block_x;
        let max_blocks = cuda.sm_count.saturating_mul(16);
        if grid_x > max_blocks {
            grid_x = max_blocks.max(1);
        }
        cuda.stream.synchronize().expect("vpt prep sync");
        Box::new(ManySeriesState {
            cuda,
            d_price,
            d_volume,
            d_first,
            d_out,
            cols,
            rows,
            grid: (grid_x, 1, 1).into(),
            block: (block_x, 1, 1).into(),
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "vpt",
                "one_series",
                "vpt_cuda_one_series",
                "1m",
                prep_one_series,
            )
            .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "vpt",
                "many_series_one_param",
                "vpt_cuda_many_series_time_major",
                "512x8192",
                prep_many_series,
            )
            .with_mem_required(bytes_many_series()),
        ]
    }
}
