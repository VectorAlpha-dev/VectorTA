#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::rsx::{RsxBatchRange, RsxParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaRsxError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
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

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaRsxPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

pub struct CudaRsx {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaRsxPolicy,
}

impl CudaRsx {
    pub fn new(device_id: usize) -> Result<Self, CudaRsxError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/rsx_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaRsxPolicy::default(),
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaRsxPolicy) {
        self.policy = p;
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaRsxError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaRsxError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _)) => {
                if required_bytes.saturating_add(headroom_bytes) <= free {
                    Ok(())
                } else {
                    Err(CudaRsxError::OutOfMemory {
                        required: required_bytes,
                        free,
                        headroom: headroom_bytes,
                    })
                }
            }
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn validate_launch_dims(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaRsxError> {
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
            return Err(CudaRsxError::InvalidInput(
                "zero-sized grid or block".into(),
            ));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz {
            return Err(CudaRsxError::LaunchConfigTooLarge {
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

    pub fn rsx_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &RsxBatchRange,
    ) -> Result<DeviceArrayF32, CudaRsxError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(prices_f32, sweep)?;
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let in_bytes = prices_f32
            .len()
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaRsxError::InvalidInput("size overflow in input bytes".into()))?;
        let params_bytes = periods_i32
            .len()
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaRsxError::InvalidInput("size overflow in params bytes".into()))?;
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaRsxError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaRsxError::InvalidInput("size overflow in output bytes".into()))?;
        let logical_plain = in_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaRsxError::InvalidInput("total VRAM size overflow".into()))?;
        let logical_tm = logical_plain
            .checked_add(out_bytes)
            .ok_or_else(|| CudaRsxError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;

        let prefer_tm = n_combos >= 32 && len >= 4_096;
        let env_tm = env::var("RSX_USE_TM").ok().as_deref() == Some("1");
        let use_tm = prefer_tm && env_tm && Self::will_fit(logical_tm, headroom).is_ok();
        if !use_tm {
            Self::will_fit(logical_plain, headroom)?;
        }

        let d_prices = unsafe { to_device_buffer_async(&self.stream, prices_f32)? };
        let d_periods = unsafe { to_device_buffer_async(&self.stream, &periods_i32)? };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            let total = len
                .checked_mul(n_combos)
                .ok_or_else(|| CudaRsxError::InvalidInput("rows*cols overflow".into()))?;
            DeviceBuffer::uninitialized(total)?
        };

        if use_tm {
            let mut d_out_tm: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(out_elems)? };
            self.launch_batch_tm(
                &d_prices,
                &d_periods,
                len,
                first_valid,
                n_combos,
                &mut d_out_tm,
            )?;
            self.launch_transpose_tm_to_rm(&d_out_tm, len, n_combos, &mut d_out)?;
        } else {
            self.launch_batch(
                &d_prices,
                &d_periods,
                len,
                first_valid,
                n_combos,
                &mut d_out,
            )?;
        }
        self.stream.synchronize().map_err(CudaRsxError::from)?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    fn launch_batch(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRsxError> {
        if n_combos == 0 {
            return Ok(());
        }
        let func = self.module.get_function("rsx_batch_f32").map_err(|_| {
            CudaRsxError::MissingKernelSymbol {
                name: "rsx_batch_f32",
            }
        })?;

        let block_x = self.policy.batch_block_x.unwrap_or(256);
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch_dims((grid_x.max(1), 1, 1), (block_x, 1, 1))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaRsxError::from)?;
        }
        Ok(())
    }

    fn launch_batch_tm(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRsxError> {
        if n_combos == 0 {
            return Ok(());
        }
        let func = self.module.get_function("rsx_batch_tm_f32").map_err(|_| {
            CudaRsxError::MissingKernelSymbol {
                name: "rsx_batch_tm_f32",
            }
        })?;

        let block_x = self.policy.batch_block_x.unwrap_or(256);
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch_dims((grid_x.max(1), 1, 1), (block_x, 1, 1))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaRsxError::from)?;
        }
        Ok(())
    }

    fn launch_transpose_tm_to_rm(
        &self,
        d_in_tm: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        d_out_rm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRsxError> {
        let func = self
            .module
            .get_function("transpose_tm_to_rm_f32")
            .map_err(|_| CudaRsxError::MissingKernelSymbol {
                name: "transpose_tm_to_rm_f32",
            })?;
        let grid_x = ((cols as u32) + 31) / 32;
        let grid_y = ((rows as u32) + 31) / 32;
        let block: BlockSize = (32u32, 8u32, 1u32).into();
        self.validate_launch_dims((grid_x.max(1), grid_y.max(1), 1), (32, 8, 1))?;
        unsafe {
            let mut in_ptr = d_in_tm.as_device_ptr().as_raw();
            let mut r_i = rows as i32;
            let mut c_i = cols as i32;
            let mut out_ptr = d_out_rm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut c_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1u32).into();
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    pub fn rsx_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaRsxError> {
        if cols == 0 || rows == 0 {
            return Err(CudaRsxError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaRsxError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaRsxError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaRsxError::InvalidInput("period must be > 0".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = prices_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaRsxError::InvalidInput(format!("series {} all NaN", s)));
            }
            first_valids[s] = fv;
        }

        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let n = expected;
        let in_bytes = n
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaRsxError::InvalidInput("size overflow in input bytes".into()))?;
        let out_bytes = n
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaRsxError::InvalidInput("size overflow in output bytes".into()))?;
        let first_bytes = cols.checked_mul(elem_i32).ok_or_else(|| {
            CudaRsxError::InvalidInput("size overflow in first_valid bytes".into())
        })?;
        let logical = in_bytes
            .checked_add(out_bytes)
            .and_then(|x| x.checked_add(first_bytes))
            .ok_or_else(|| CudaRsxError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        Self::will_fit(logical, headroom)?;

        let d_prices = unsafe { to_device_buffer_async(&self.stream, prices_tm_f32)? };
        let d_first = unsafe { to_device_buffer_async(&self.stream, &first_valids)? };
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected)? };

        self.launch_many_series(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many_series(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRsxError> {
        let func = self
            .module
            .get_function("rsx_many_series_one_param_f32")
            .map_err(|_| CudaRsxError::MissingKernelSymbol {
                name: "rsx_many_series_one_param_f32",
            })?;
        let block_x = self.policy.many_block_x.unwrap_or(128);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch_dims((grid_x.max(1), 1, 1), (block_x, 1, 1))?;
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaRsxError::from)?;
        }
        self.stream.synchronize().map_err(CudaRsxError::from)?;
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &RsxBatchRange,
    ) -> Result<(Vec<RsxParams>, usize, usize), CudaRsxError> {
        let len = prices.len();
        if len == 0 {
            return Err(CudaRsxError::InvalidInput("empty prices".into()));
        }

        let (start, end, step) = sweep.period;
        let combos = {
            let axis = |triple: (usize, usize, usize)| -> Result<Vec<usize>, CudaRsxError> {
                let (s, e, st) = triple;
                if st == 0 || s == e {
                    return Ok(vec![s]);
                }
                if s < e {
                    let step = st.max(1);
                    let v: Vec<usize> = (s..=e).step_by(step).collect();
                    if v.is_empty() {
                        return Err(CudaRsxError::InvalidInput("empty period expansion".into()));
                    }
                    return Ok(v);
                }
                let mut v = Vec::new();
                let mut x = s as isize;
                let end_i = e as isize;
                let step = (st as isize).max(1);
                while x >= end_i {
                    v.push(x as usize);
                    x -= step;
                }
                if v.is_empty() {
                    return Err(CudaRsxError::InvalidInput(
                        "empty reversed period expansion".into(),
                    ));
                }
                Ok(v)
            };
            let periods = axis((start, end, step))?;
            if periods.is_empty() {
                return Err(CudaRsxError::InvalidInput("no period combos".into()));
            }
            periods
                .into_iter()
                .map(|p| RsxParams { period: Some(p) })
                .collect::<Vec<_>>()
        };

        let first_valid = (0..len)
            .find(|&i| !prices[i].is_nan())
            .ok_or_else(|| CudaRsxError::InvalidInput("all values NaN".into()))?;
        let max_p = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_p == 0 {
            return Err(CudaRsxError::InvalidInput("period must be > 0".into()));
        }
        let valid = len - first_valid;
        if valid < max_p {
            return Err(CudaRsxError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_p, valid
            )));
        }
        Ok((combos, first_valid, len))
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_COLS: usize = 1024;
    const MANY_ROWS: usize = 8192;
    const PARAM_SWEEP: usize = 200;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let n = MANY_COLS * MANY_ROWS;
        let in_bytes = n * std::mem::size_of::<f32>();
        let out_bytes = n * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct RsxBatchDeviceState {
        cuda: CudaRsx,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for RsxBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_prices,
                    &self.d_periods,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("rsx launch_batch");
            self.cuda.synchronize().expect("rsx sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaRsx::new(0).expect("cuda rsx");
        let mut prices = gen_series(ONE_SERIES_LEN);

        for i in 0..8 {
            prices[i] = f32::NAN;
        }
        for i in 8..ONE_SERIES_LEN {
            let x = i as f32 * 0.0019;
            prices[i] += 0.0005 * x.sin();
        }
        let sweep = RsxBatchRange {
            period: (2, 1 + PARAM_SWEEP, 1),
        };

        let (combos, first_valid, len) =
            CudaRsx::prepare_batch_inputs(&prices, &sweep).expect("rsx prepare_batch_inputs");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        let d_prices = unsafe { to_device_buffer_async(&cuda.stream, &prices) }.expect("d_prices");
        let d_periods =
            unsafe { to_device_buffer_async(&cuda.stream, &periods_i32) }.expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len * n_combos) }.expect("d_out");
        cuda.synchronize().expect("sync after prep");

        Box::new(RsxBatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            d_out,
            len,
            first_valid,
            n_combos,
        })
    }

    struct RsxManySeriesDeviceState {
        cuda: CudaRsx,
        d_prices_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for RsxManySeriesDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series(
                    &self.d_prices_tm,
                    &self.d_first,
                    self.cols,
                    self.rows,
                    self.period,
                    &mut self.d_out_tm,
                )
                .expect("rsx launch_many_series");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaRsx::new(0).expect("cuda rsx");
        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let n = cols * rows;
        let mut base = gen_series(n);
        let mut prices = vec![f32::NAN; n];
        for s in 0..cols {
            for t in s..rows {
                let idx = t * cols + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.01;
                prices[idx] = base[idx] + 0.05 * x.sin();
            }
        }
        let first_valids: Vec<i32> = (0..cols).map(|i| i as i32).collect();
        let d_prices_tm =
            unsafe { to_device_buffer_async(&cuda.stream, &prices) }.expect("d_prices_tm");
        let d_first =
            unsafe { to_device_buffer_async(&cuda.stream, &first_valids) }.expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n) }.expect("d_out_tm");
        cuda.synchronize().expect("sync after prep");
        Box::new(RsxManySeriesDeviceState {
            cuda,
            d_prices_tm,
            d_first,
            d_out_tm,
            cols,
            rows,
            period: 14,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "rsx",
                "one_series_many_params",
                "rsx_cuda_batch_dev",
                "1m_x_200",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "rsx",
                "many_series_one_param",
                "rsx_cuda_many_series_one_param_dev",
                "1024x8192",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}

#[inline]
unsafe fn to_device_buffer_async<T: cust::memory::DeviceCopy>(
    stream: &Stream,
    host: &[T],
) -> Result<DeviceBuffer<T>, CudaRsxError> {
    const PIN_BYTES: usize = 4 * 1024 * 1024;
    let bytes = host
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| CudaRsxError::InvalidInput("host size overflow".into()))?;
    if bytes >= PIN_BYTES {
        let pinned = LockedBuffer::from_slice(host)?;
        DeviceBuffer::from_slice_async(pinned.as_slice(), stream).map_err(CudaRsxError::from)
    } else {
        DeviceBuffer::from_slice(host).map_err(CudaRsxError::from)
    }
}
