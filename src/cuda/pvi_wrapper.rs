#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::CacheConfig;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaPviError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: named symbol not found: {name}")]
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
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("NotImplemented: {0}")]
    NotImplemented(&'static str),
}

pub struct CudaPvi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaPvi {
    pub fn new(device_id: usize) -> Result<Self, CudaPviError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let ctx = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pvi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        if let Ok(mut func) = module.get_function("pvi_apply_scale_batch_f32") {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }
        Ok(Self {
            module,
            stream,
            context: ctx,
            device_id: device_id as u32,
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

    #[inline]
    fn first_valid_pair(close: &[f32], volume: &[f32]) -> Result<usize, CudaPviError> {
        if close.is_empty() || volume.is_empty() {
            return Err(CudaPviError::InvalidInput("empty inputs".into()));
        }
        if close.len() != volume.len() {
            return Err(CudaPviError::InvalidInput("length mismatch".into()));
        }
        let first = close
            .iter()
            .zip(volume.iter())
            .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
            .ok_or_else(|| {
                CudaPviError::InvalidInput("all values are NaN in one/both inputs".into())
            })?;
        if close.len() - first < 2 {
            return Err(CudaPviError::InvalidInput(
                "not enough valid data (need >= 2 after first valid)".into(),
            ));
        }
        Ok(first)
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaPviError> {
        if let Some((free, _total)) = Self::device_mem_info() {
            let required = required_bytes.saturating_add(headroom_bytes);
            if required > free {
                return Err(CudaPviError::OutOfMemory {
                    required,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    #[inline]
    fn validate_launch_dims(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaPviError> {
        let dev = Device::get_device(self.device_id)?;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaPviError::InvalidInput(
                "zero-sized grid or block".into(),
            ));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz {
            return Err(CudaPviError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        Ok(())
    }

    pub fn pvi_batch_dev(
        &self,
        close: &[f32],
        volume: &[f32],
        initial_values: &[f32],
    ) -> Result<DeviceArrayF32, CudaPviError> {
        let first = Self::first_valid_pair(close, volume)?;
        let len = close.len();
        let rows = initial_values.len();
        let has_nan_after_first = close[first..].iter().any(|&v| v.is_nan())
            || volume[first..].iter().any(|&v| v.is_nan());
        if rows == 0 {
            return Err(CudaPviError::InvalidInput(
                "no initial values provided".into(),
            ));
        }

        let elem_size = std::mem::size_of::<f32>();
        let two_len = len
            .checked_mul(2)
            .ok_or_else(|| CudaPviError::InvalidInput("len*2 overflow".into()))?;
        let stage1_elems = two_len
            .checked_add(len)
            .ok_or_else(|| CudaPviError::InvalidInput("len accumulation overflow".into()))?;
        let bytes_stage1 = stage1_elems
            .checked_mul(elem_size)
            .ok_or_else(|| CudaPviError::InvalidInput("bytes_stage1 overflow".into()))?;
        let rows_len = rows
            .checked_mul(len)
            .ok_or_else(|| CudaPviError::InvalidInput("rows*len overflow".into()))?;
        let stage2_elems = len
            .checked_add(rows)
            .and_then(|v| v.checked_add(rows_len))
            .ok_or_else(|| CudaPviError::InvalidInput("stage2 element count overflow".into()))?;
        let bytes_stage2 = stage2_elems
            .checked_mul(elem_size)
            .ok_or_else(|| CudaPviError::InvalidInput("bytes_stage2 overflow".into()))?;
        let bytes_peak = bytes_stage1.max(bytes_stage2);
        Self::will_fit(bytes_peak, 64 << 20)?;

        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_inits = DeviceBuffer::from_slice(initial_values)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows_len) }?;

        {
            let mut d_scale: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;
            let (build, block_x): (Function, u32) = if has_nan_after_first {
                (
                    self.module
                        .get_function("pvi_build_scale_f32")
                        .map_err(|_| CudaPviError::MissingKernelSymbol {
                            name: "pvi_build_scale_f32",
                        })?,
                    1,
                )
            } else {
                (
                    self.module
                        .get_function("pvi_build_scale_warp16_f32")
                        .map_err(|_| CudaPviError::MissingKernelSymbol {
                            name: "pvi_build_scale_warp16_f32",
                        })?,
                    16,
                )
            };
            unsafe {
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut vol_ptr = d_volume.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first as i32;
                let mut scale_ptr = d_scale.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut scale_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = (1, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                self.validate_launch_dims((1, 1, 1), (block_x, 1, 1))?;
                self.stream.launch(&build, grid, block, 0, args)?;
            }

            drop(d_close);
            drop(d_volume);

            let apply = self
                .module
                .get_function("pvi_apply_scale_batch_f32")
                .map_err(|_| CudaPviError::MissingKernelSymbol {
                    name: "pvi_apply_scale_batch_f32",
                })?;
            unsafe {
                let mut scale_ptr = d_scale.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first as i32;
                let block_x: u32 = 256;
                let block_y: u32 = 4;
                let grid_x: u32 = ((len as u32) + block_x - 1) / block_x;
                let block: BlockSize = (block_x, block_y, 1).into();

                const MAX_GRID_Y: usize = 65_535;
                let mut start_row = 0usize;
                while start_row < rows {
                    let chunk = (rows - start_row).min(MAX_GRID_Y);
                    let grid_y: u32 = ((chunk as u32) + block_y - 1) / block_y;
                    let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();

                    let mut inits_ptr = d_inits
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start_row * std::mem::size_of::<f32>()) as u64);
                    let mut rows_i = chunk as i32;
                    let mut out_ptr = d_out
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start_row * len * std::mem::size_of::<f32>()) as u64);
                    let args: &mut [*mut c_void] = &mut [
                        &mut scale_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut inits_ptr as *mut _ as *mut c_void,
                        &mut rows_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];

                    self.validate_launch_dims(
                        (grid_x.max(1), grid_y.max(1), 1),
                        (block_x, block_y, 1),
                    )?;
                    self.stream.launch(&apply, grid, block, 0, args)?;
                    start_row += chunk;
                }
            }
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: len,
        })
    }

    pub fn pvi_many_series_one_param_time_major_dev(
        &self,
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
        initial_value: f32,
    ) -> Result<DeviceArrayF32, CudaPviError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPviError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaPviError::InvalidInput("rows*cols overflow".into()))?;
        if close_tm.len() != expected || volume_tm.len() != expected {
            return Err(CudaPviError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }

        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let c = close_tm[t * cols + s];
                let v = volume_tm[t * cols + s];
                if !c.is_nan() && !v.is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            if (rows as i32 - first_valids[s]) < 2 {
                return Err(CudaPviError::InvalidInput(format!(
                    "series {}: not enough valid data (need >= 2 after first valid)",
                    s
                )));
            }
        }

        let elem = std::mem::size_of::<f32>();
        let bytes_f32 = expected
            .checked_mul(3)
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(|| CudaPviError::InvalidInput("VRAM bytes overflow".into()))?;
        let bytes_first = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaPviError::InvalidInput("first_valids bytes overflow".into()))?;
        let bytes = bytes_f32
            .checked_add(bytes_first)
            .ok_or_else(|| CudaPviError::InvalidInput("total VRAM bytes overflow".into()))?;
        Self::will_fit(bytes, 64 << 20)?;

        let d_close = DeviceBuffer::from_slice(close_tm)?;
        let d_volume = DeviceBuffer::from_slice(volume_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }?;

        let func = self
            .module
            .get_function("pvi_many_series_one_param_f32")
            .map_err(|_| CudaPviError::MissingKernelSymbol {
                name: "pvi_many_series_one_param_f32",
            })?;
        unsafe {
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut init_f = initial_value;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut init_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let block_x: u32 = 256;
            let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch_dims((grid_x.max(1), 1, 1), (block_x, 1, 1))?;
            self.stream.launch(&func, grid, block, 0, args)?;
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

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 512;
    const MANY_SERIES_ROWS: usize = 8_192;

    fn bytes_one_series(rows: usize) -> usize {
        (2 * ONE_SERIES_LEN + ONE_SERIES_LEN + rows + rows * ONE_SERIES_LEN)
            * std::mem::size_of::<f32>()
            + (64 << 20)
    }
    fn bytes_many_series() -> usize {
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        (3 * n * std::mem::size_of::<f32>()) + (64 << 20)
    }

    struct PviOneSeriesState {
        cuda: CudaPvi,
        d_close: DeviceBuffer<f32>,
        d_volume: DeviceBuffer<f32>,
        d_inits: DeviceBuffer<f32>,
        d_scale: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first: usize,
        rows: usize,
        build_block_x: u32,
        apply_grid: GridSize,
        apply_block: BlockSize,
    }
    impl CudaBenchState for PviOneSeriesState {
        fn launch(&mut self) {
            let build_name = if self.build_block_x == 1 {
                "pvi_build_scale_f32"
            } else {
                "pvi_build_scale_warp16_f32"
            };
            let build = self
                .cuda
                .module
                .get_function(build_name)
                .expect("pvi build_scale");
            unsafe {
                let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                let mut vol_ptr = self.d_volume.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut first_i = self.first as i32;
                let mut scale_ptr = self.d_scale.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut scale_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = (1, 1, 1).into();
                let block: BlockSize = (self.build_block_x, 1, 1).into();
                self.cuda
                    .stream
                    .launch(&build, grid, block, 0, args)
                    .expect("pvi build launch");
            }

            let apply = self
                .cuda
                .module
                .get_function("pvi_apply_scale_batch_f32")
                .expect("pvi_apply_scale_batch_f32");
            unsafe {
                let mut scale_ptr = self.d_scale.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut first_i = self.first as i32;
                let mut inits_ptr = self.d_inits.as_device_ptr().as_raw();
                let mut rows_i = self.rows as i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut scale_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut inits_ptr as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&apply, self.apply_grid, self.apply_block, 0, args)
                    .expect("pvi apply launch");
            }

            self.cuda.stream.synchronize().expect("pvi one-series sync");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaPvi::new(0).expect("cuda pvi");
        let mut close = gen_series(ONE_SERIES_LEN);
        let mut volume = gen_series(ONE_SERIES_LEN);

        if close[0].is_nan() || volume[0].is_nan() {
            close[0] = 100.0;
            volume[0] = 1000.0;
        }

        let mut inits = vec![0f32; 64];
        for i in 0..inits.len() {
            inits[i] = 500.0 + (i as f32) * 25.0;
        }
        let first = CudaPvi::first_valid_pair(&close, &volume).expect("first_valid_pair");
        let has_nan_after_first = close[first..].iter().any(|&v| v.is_nan())
            || volume[first..].iter().any(|&v| v.is_nan());

        let d_close = DeviceBuffer::from_slice(&close).expect("d_close");
        let d_volume = DeviceBuffer::from_slice(&volume).expect("d_volume");
        let d_inits = DeviceBuffer::from_slice(&inits).expect("d_inits");
        let d_scale: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(ONE_SERIES_LEN) }.expect("d_scale");
        let out_elems = inits
            .len()
            .checked_mul(ONE_SERIES_LEN)
            .expect("rows*len overflow");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("d_out");

        let build_block_x: u32 = if has_nan_after_first { 1 } else { 16 };
        let block_x: u32 = 256;
        let block_y: u32 = 4;
        let grid_x: u32 = ((ONE_SERIES_LEN as u32) + block_x - 1) / block_x;
        let grid_y: u32 = ((inits.len() as u32) + block_y - 1) / block_y;
        let apply_grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let apply_block: BlockSize = (block_x, block_y, 1).into();
        cuda.stream.synchronize().expect("pvi prep sync");
        Box::new(PviOneSeriesState {
            cuda,
            d_close,
            d_volume,
            d_inits,
            d_scale,
            d_out,
            len: ONE_SERIES_LEN,
            first,
            rows: inits.len(),
            build_block_x,
            apply_grid,
            apply_block,
        })
    }

    struct PviManySeriesState {
        cuda: CudaPvi,
        d_close: DeviceBuffer<f32>,
        d_volume: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        init: f32,
        grid: GridSize,
        block: BlockSize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for PviManySeriesState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("pvi_many_series_one_param_f32")
                .expect("pvi_many_series_one_param_f32");
            unsafe {
                let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                let mut vol_ptr = self.d_volume.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut first_ptr = self.d_first.as_device_ptr().as_raw();
                let mut init_f = self.init;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut init_f as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, args)
                    .expect("pvi many-series launch");
            }
            self.cuda
                .stream
                .synchronize()
                .expect("pvi many-series sync");
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaPvi::new(0).expect("cuda pvi");
        let n = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let mut close_tm = vec![f32::NAN; n];
        let mut volume_tm = vec![f32::NAN; n];
        for s in 0..MANY_SERIES_COLS {
            for t in s.min(8)..MANY_SERIES_ROWS {
                let x = (t as f32) + (s as f32) * 0.11;
                close_tm[t * MANY_SERIES_COLS + s] = (x * 0.0021).sin() + 0.0002 * x + 100.0;
                volume_tm[t * MANY_SERIES_COLS + s] = (x * 0.0017).cos().abs() * 500.0 + 100.0;
            }
        }
        let mut first_valids = vec![MANY_SERIES_ROWS as i32; MANY_SERIES_COLS];
        for s in 0..MANY_SERIES_COLS {
            for t in 0..MANY_SERIES_ROWS {
                let c = close_tm[t * MANY_SERIES_COLS + s];
                let v = volume_tm[t * MANY_SERIES_COLS + s];
                if !c.is_nan() && !v.is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let d_close = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
        let d_volume = DeviceBuffer::from_slice(&volume_tm).expect("d_volume_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let init = 1000.0f32;
        let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }.expect("d_out_tm");
        let block_x: u32 = 256;
        let grid_x: u32 = ((MANY_SERIES_COLS as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        cuda.stream.synchronize().expect("pvi prep sync");
        Box::new(PviManySeriesState {
            cuda,
            d_close,
            d_volume,
            d_first,
            cols: MANY_SERIES_COLS,
            rows: MANY_SERIES_ROWS,
            init,
            grid,
            block,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "pvi",
                "pvi",
                "pvi_cuda_one_series",
                "1m x 64",
                prep_one_series,
            )
            .with_mem_required(bytes_one_series(64)),
            CudaBenchScenario::new(
                "pvi",
                "pvi",
                "pvi_cuda_many_series_time_major",
                "512x8192",
                prep_many_series,
            )
            .with_mem_required(bytes_many_series()),
        ]
    }
}
