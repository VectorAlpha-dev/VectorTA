#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::rsi::{RsiBatchRange, RsiParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaRsiError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
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
pub struct CudaRsiPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

pub struct CudaRsi {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaRsiPolicy,
    max_grid_x: u32,
}

impl CudaRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaRsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/rsi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O3),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(65_535) as u32;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaRsiPolicy::default(),
            max_grid_x,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaRsiPolicy) {
        self.policy = p;
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaRsiError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaRsiError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn validate_launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaRsiError> {
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if bx > max_bx || by > max_by || bz > max_bz || gx > max_gx || gy > max_gy || gz > max_gz {
            return Err(CudaRsiError::LaunchConfigTooLarge {
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

    pub fn rsi_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &RsiBatchRange,
    ) -> Result<DeviceArrayF32, CudaRsiError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(prices_f32, sweep)?;
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        let out_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaRsiError::InvalidInput("rows*cols overflow".into()))?;
        let elem_bytes = std::mem::size_of::<f32>();
        let param_bytes = std::mem::size_of::<i32>();
        let in_bytes = prices_f32
            .len()
            .checked_mul(elem_bytes)
            .ok_or_else(|| CudaRsiError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = periods_i32
            .len()
            .checked_mul(param_bytes)
            .ok_or_else(|| CudaRsiError::InvalidInput("params bytes overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_bytes)
            .ok_or_else(|| CudaRsiError::InvalidInput("output bytes overflow".into()))?;
        let required = in_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaRsiError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64usize * 1024 * 1024)?;

        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len)? };
        let mut d_periods: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(n_combos)? };
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems)? };

        let h_prices = LockedBuffer::from_slice(prices_f32)?;
        let h_periods = LockedBuffer::from_slice(&periods_i32)?;
        unsafe {
            d_prices.async_copy_from(&h_prices, &self.stream)?;
            d_periods.async_copy_from(&h_periods, &self.stream)?;
        }

        self.launch_batch(
            &d_prices,
            &d_periods,
            len,
            first_valid,
            n_combos,
            &mut d_out,
        )?;

        self.stream.synchronize()?;
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
    ) -> Result<(), CudaRsiError> {
        let func = self.module.get_function("rsi_batch_f32").map_err(|_| {
            CudaRsiError::MissingKernelSymbol {
                name: "rsi_batch_f32",
            }
        })?;
        let mut block_x: u32 = self.policy.batch_block_x.unwrap_or(256);
        block_x = block_x.max(32);
        block_x -= block_x % 32;
        let warps_per_block = (block_x / 32).max(1);
        let max_grid_x: u32 = self.max_grid_x.max(1);
        let combos_per_launch: usize = (warps_per_block as usize) * (max_grid_x as usize);
        let mut launched = 0usize;
        while launched < n_combos {
            let this_chunk = (n_combos - launched).min(combos_per_launch);
            let grid_x = ((this_chunk as u32) + warps_per_block - 1) / warps_per_block;
            let grid = (grid_x.max(1), 1, 1);
            let block = (block_x, 1, 1);
            self.validate_launch(grid, block)?;
            let grid: GridSize = grid.into();
            let block: BlockSize = block.into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut series_len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut combos_i = this_chunk as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(((launched * len) * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += this_chunk;
        }
        Ok(())
    }

    pub fn rsi_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaRsiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaRsiError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaRsiError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaRsiError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaRsiError::InvalidInput("period must be > 0".into()));
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
            }
            if fv < 0 {
                return Err(CudaRsiError::InvalidInput(format!("series {} all NaN", s)));
            }
            first_valids[s] = fv;
        }

        let elem_bytes = std::mem::size_of::<f32>();
        let first_bytes = std::mem::size_of::<i32>();
        let two_n = expected
            .checked_mul(2)
            .ok_or_else(|| CudaRsiError::InvalidInput("2*n overflow".into()))?;
        let data_bytes = two_n
            .checked_mul(elem_bytes)
            .ok_or_else(|| CudaRsiError::InvalidInput("data bytes overflow".into()))?;
        let first_valid_bytes = cols
            .checked_mul(first_bytes)
            .ok_or_else(|| CudaRsiError::InvalidInput("first_valid bytes overflow".into()))?;
        let required = data_bytes
            .checked_add(first_valid_bytes)
            .ok_or_else(|| CudaRsiError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_prices = DeviceBuffer::from_slice(prices_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected)? };

        self.launch_many_series(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
        self.stream.synchronize()?;
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
    ) -> Result<(), CudaRsiError> {
        let func = self
            .module
            .get_function("rsi_many_series_one_param_f32")
            .map_err(|_| CudaRsiError::MissingKernelSymbol {
                name: "rsi_many_series_one_param_f32",
            })?;
        let block_x = self.policy.many_block_x.unwrap_or(256);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid = (grid_x.max(1), 1, 1);
        let block = (block_x, 1, 1);
        self.validate_launch(grid, block)?;
        let grid: GridSize = grid.into();
        let block: BlockSize = block.into();
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
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &RsiBatchRange,
    ) -> Result<(Vec<RsiParams>, usize, usize), CudaRsiError> {
        let len = prices.len();
        if len == 0 {
            return Err(CudaRsiError::InvalidInput("empty prices".into()));
        }

        let (start, end, step) = sweep.period;
        let mut combos = Vec::new();
        if step == 0 || start == end {
            combos.push(RsiParams {
                period: Some(start),
            });
        } else {
            let (lo, hi) = if start <= end {
                (start, end)
            } else {
                (end, start)
            };
            let mut v = lo;
            loop {
                combos.push(RsiParams { period: Some(v) });
                if v == hi {
                    break;
                }
                v = v
                    .checked_add(step)
                    .ok_or_else(|| CudaRsiError::InvalidInput("period range overflow".into()))?;
                if v > hi {
                    break;
                }
            }
        }
        if combos.is_empty() {
            return Err(CudaRsiError::InvalidInput(format!(
                "invalid period range: start={} end={} step={}",
                start, end, step
            )));
        }

        let first_valid = (0..len)
            .find(|&i| !prices[i].is_nan())
            .ok_or_else(|| CudaRsiError::InvalidInput("all values NaN".into()))?;
        let max_p = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_p == 0 {
            return Err(CudaRsiError::InvalidInput("period must be > 0".into()));
        }
        let valid = len - first_valid;
        if valid < max_p {
            return Err(CudaRsiError::InvalidInput(format!(
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

    struct RsiBatchDeviceState {
        cuda: CudaRsi,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for RsiBatchDeviceState {
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
                .expect("rsi launch");
            self.cuda.stream.synchronize().expect("rsi sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaRsi::new(0).expect("cuda rsi");
        let mut prices = gen_series(ONE_SERIES_LEN);

        for i in 0..8 {
            prices[i] = f32::NAN;
        }
        for i in 8..ONE_SERIES_LEN {
            let x = i as f32 * 0.0019;
            prices[i] += 0.0005 * x.sin();
        }
        let sweep = RsiBatchRange {
            period: (2, 1 + PARAM_SWEEP, 1),
        };

        let (combos, first_valid, len) =
            CudaRsi::prepare_batch_inputs(&prices, &sweep).expect("prepare_batch_inputs");
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();
        let n_combos = periods_i32.len();

        let d_prices =
            unsafe { DeviceBuffer::from_slice_async(&prices, &cuda.stream) }.expect("d_prices H2D");
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &cuda.stream) }
            .expect("d_periods H2D");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len * n_combos, &cuda.stream) }
                .expect("d_out alloc");
        cuda.stream.synchronize().expect("rsi prep sync");

        Box::new(RsiBatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            len,
            first_valid,
            n_combos,
            d_out,
        })
    }

    struct RsiManyDeviceState {
        cuda: CudaRsi,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for RsiManyDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    MANY_COLS,
                    MANY_ROWS,
                    14,
                    &mut self.d_out_tm,
                )
                .expect("rsi many launch");
            self.cuda.stream.synchronize().expect("rsi many sync");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaRsi::new(0).expect("cuda rsi");
        let n = MANY_COLS * MANY_ROWS;
        let mut base = gen_series(n);
        let mut prices = vec![f32::NAN; n];
        for s in 0..MANY_COLS {
            for t in s..MANY_ROWS {
                let idx = t * MANY_COLS + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.01;
                prices[idx] = base[idx] + 0.05 * x.sin();
            }
        }

        let mut first_valids = vec![0i32; MANY_COLS];
        for s in 0..MANY_COLS {
            let mut fv = -1i32;
            for t in 0..MANY_ROWS {
                let v = prices[t * MANY_COLS + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv.max(0);
        }

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(&prices, &cuda.stream) }
            .expect("d_prices_tm H2D");
        let d_first_valids = unsafe { DeviceBuffer::from_slice_async(&first_valids, &cuda.stream) }
            .expect("d_first_valids H2D");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &cuda.stream) }.expect("d_out_tm alloc");
        cuda.stream.synchronize().expect("rsi many prep sync");

        Box::new(RsiManyDeviceState {
            cuda,
            d_prices_tm,
            d_first_valids,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "rsi",
                "one_series_many_params",
                "rsi_cuda_batch_dev",
                "1m_x_200",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "rsi",
                "many_series_one_param",
                "rsi_cuda_many_series_one_param_dev",
                "1024x8192",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
