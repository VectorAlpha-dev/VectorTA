#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::nadaraya_watson_envelope::{NweBatchRange, NweParams};
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
pub enum CudaNweError {
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

pub struct DeviceNwePair {
    pub upper: DeviceArrayF32,
    pub lower: DeviceArrayF32,
}

impl DeviceNwePair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.upper.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.upper.cols
    }
}

pub struct CudaNwe {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
}

impl CudaNwe {
    pub fn new(device_id: usize) -> Result<Self, CudaNweError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);
        let ptx = include_str!(concat!(
            env!("OUT_DIR"),
            "/nadaraya_watson_envelope_kernel.ptx"
        ));

        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        )
        .or_else(|_| {
            Module::from_ptx(
                ptx,
                &[
                    ModuleJitOption::DetermineTargetFromContext,
                    ModuleJitOption::OptLevel(OptLevel::O2),
                ],
            )
        })
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    fn will_fit(required: usize, headroom: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            required.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaNweError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaNweError::LaunchConfigTooLarge {
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

    pub fn context_arc(&self) -> std::sync::Arc<Context> {
        self.context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    pub fn synchronize(&self) -> Result<(), CudaNweError> {
        Ok(self.stream.synchronize()?)
    }

    fn expand_grid(r: &NweBatchRange) -> Result<Vec<NweParams>, CudaNweError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaNweError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            if start < end {
                return Ok((start..=end).step_by(step.max(1)).collect());
            }
            let mut v = Vec::new();
            let mut x = start as isize;
            let end_i = end as isize;
            let st = (step as isize).max(1);
            while x >= end_i {
                v.push(x as usize);
                x -= st;
            }
            if v.is_empty() {
                return Err(CudaNweError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaNweError> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return Ok(vec![start]);
            }
            let st = step.abs();
            if start < end {
                let mut v = Vec::new();
                let mut x = start;
                while x <= end + 1e-12 {
                    v.push(x);
                    x += st;
                }
                if v.is_empty() {
                    return Err(CudaNweError::InvalidInput(format!(
                        "Invalid range: start={}, end={}, step={}",
                        start, end, step
                    )));
                }
                return Ok(v);
            }
            let mut v = Vec::new();
            let mut x = start;
            while x + 1e-12 >= end {
                v.push(x);
                x -= st;
            }
            if v.is_empty() {
                return Err(CudaNweError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }

        let bw = axis_f64(r.bandwidth)?;
        let m = axis_f64(r.multiplier)?;
        let lb = axis_usize(r.lookback)?;

        let cap = bw
            .len()
            .checked_mul(m.len())
            .and_then(|x| x.checked_mul(lb.len()))
            .ok_or_else(|| CudaNweError::InvalidInput("range size overflow".into()))?;

        let mut out = Vec::with_capacity(cap);
        for &b in &bw {
            for &mm in &m {
                for &l in &lb {
                    out.push(NweParams {
                        bandwidth: Some(b),
                        multiplier: Some(mm),
                        lookback: Some(l),
                    });
                }
            }
        }
        Ok(out)
    }

    fn compute_weights_row(bandwidth: f64, lookback: usize) -> (Vec<f32>, usize) {
        let mut w = Vec::with_capacity(lookback);
        let mut den = 0.0f64;
        for k in 0..lookback {
            let wk = (-(k as f64) * (k as f64) / (2.0 * bandwidth * bandwidth)).exp();
            w.push(wk as f32);
            den += wk;
        }

        let inv_den = if den != 0.0 {
            1.0f32 / (den as f32)
        } else {
            0.0f32
        };
        for x in &mut w {
            *x *= inv_den;
        }
        (w, lookback)
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &NweBatchRange,
    ) -> Result<
        (
            Vec<NweParams>,
            usize,
            usize,
            Vec<i32>,
            Vec<f32>,
            Vec<f32>,
            usize,
        ),
        CudaNweError,
    > {
        if prices.is_empty() {
            return Err(CudaNweError::InvalidInput("empty series".into()));
        }
        let len = prices.len();
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaNweError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep)?;

        let mut lookbacks = Vec::with_capacity(combos.len());
        let mut multipliers = Vec::with_capacity(combos.len());
        let mut max_lb = 1usize;
        for prm in &combos {
            let lb = prm.lookback.unwrap_or(500);
            if lb == 0 {
                return Err(CudaNweError::InvalidInput("lookback must be > 0".into()));
            }
            if len - first_valid < lb {
                return Err(CudaNweError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    lb,
                    len - first_valid
                )));
            }
            lookbacks.push(lb as i32);
            multipliers.push(prm.multiplier.unwrap_or(3.0) as f32);
            max_lb = max_lb.max(lb);
        }

        let cap = combos
            .len()
            .checked_mul(max_lb)
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        let mut weights_flat = vec![0f32; cap];
        for (row, prm) in combos.iter().enumerate() {
            let lb = prm.lookback.unwrap_or(500);
            let (row_w, _l) = Self::compute_weights_row(prm.bandwidth.unwrap_or(8.0), lb);
            let base = row * max_lb;
            weights_flat[base..base + lb].copy_from_slice(&row_w);
        }

        Ok((
            combos,
            first_valid,
            len,
            lookbacks,
            multipliers,
            weights_flat,
            max_lb,
        ))
    }

    pub fn nwe_batch_dev(
        &self,
        prices: &[f32],
        sweep: &NweBatchRange,
    ) -> Result<(DeviceNwePair, Vec<NweParams>), CudaNweError> {
        let (combos, first_valid, len, lookbacks, multipliers, weights_flat, max_lb) =
            Self::prepare_batch_inputs(prices, sweep)?;
        let n = combos.len();

        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let mut required = 0usize;
        required = required
            .checked_add(
                len.checked_mul(sz_f32)
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        required = required
            .checked_add(
                n.checked_mul(max_lb)
                    .and_then(|x| x.checked_mul(sz_f32))
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        required = required
            .checked_add(
                n.checked_mul(sz_i32)
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        required = required
            .checked_add(
                n.checked_mul(sz_f32)
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        required = required
            .checked_add(
                n.checked_mul(len)
                    .and_then(|x| x.checked_mul(2))
                    .and_then(|x| x.checked_mul(sz_f32))
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;

        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Ok((free, _)) = mem_get_info() {
                return Err(CudaNweError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaNweError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let h_prices = LockedBuffer::from_slice(prices).map_err(CudaNweError::from)?;
        let h_weights = LockedBuffer::from_slice(&weights_flat).map_err(CudaNweError::from)?;

        let mut d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaNweError::from)?;
        let weights_len = n
            .checked_mul(max_lb)
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        let mut d_weights: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(weights_len) }.map_err(CudaNweError::from)?;
        let d_looks = DeviceBuffer::from_slice(&lookbacks).map_err(CudaNweError::from)?;
        let d_mults = DeviceBuffer::from_slice(&multipliers).map_err(CudaNweError::from)?;
        let out_len = n
            .checked_mul(len)
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        let mut d_upper: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_len) }.map_err(CudaNweError::from)?;
        let mut d_lower: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_len) }.map_err(CudaNweError::from)?;

        unsafe {
            d_prices
                .async_copy_from(&h_prices, &self.stream)
                .map_err(CudaNweError::from)?;
            d_weights
                .async_copy_from(&h_weights, &self.stream)
                .map_err(CudaNweError::from)?;
        }

        self.nwe_batch_device(
            &d_prices,
            &d_weights,
            &d_looks,
            &d_mults,
            len,
            n,
            first_valid,
            max_lb,
            &mut d_upper,
            &mut d_lower,
        )?;

        self.synchronize()?;

        let pair = DeviceNwePair {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows: n,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows: n,
                cols: len,
            },
        };
        Ok((pair, combos))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn nwe_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_looks: &DeviceBuffer<i32>,
        d_mults: &DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        max_lb: usize,
        d_upper: &mut DeviceBuffer<f32>,
        d_lower: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNweError> {
        if d_prices.len() != len {
            return Err(CudaNweError::InvalidInput("prices length mismatch".into()));
        }
        if d_looks.len() != n_combos {
            return Err(CudaNweError::InvalidInput(
                "lookbacks length must match n_combos".into(),
            ));
        }
        if d_mults.len() != n_combos {
            return Err(CudaNweError::InvalidInput(
                "multipliers length must match n_combos".into(),
            ));
        }
        let expected_w = n_combos
            .checked_mul(max_lb)
            .ok_or_else(|| CudaNweError::InvalidInput("weights length overflow".into()))?;
        if d_weights.len() != expected_w {
            return Err(CudaNweError::InvalidInput("weights length mismatch".into()));
        }
        let expected_out = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaNweError::InvalidInput("rows*cols overflow".into()))?;
        if d_upper.len() != expected_out || d_lower.len() != expected_out {
            return Err(CudaNweError::InvalidInput("output length mismatch".into()));
        }

        let func = self
            .module
            .get_function("nadaraya_watson_envelope_batch_f32")
            .map_err(|_| CudaNweError::MissingKernelSymbol {
                name: "nadaraya_watson_envelope_batch_f32",
            })?;

        const NWE_THREADS: u32 = 128;
        const NWE_TILE_T: usize = 64;
        let grid = GridSize::xy(1, n_combos as u32);
        let block = BlockSize::xyz(NWE_THREADS, 1, 1);
        self.validate_launch(grid.x, grid.y, grid.z, block.x, block.y, block.z)?;

        let smem_elems = max_lb + 2 * (max_lb + NWE_TILE_T - 1);
        let smem_bytes = (smem_elems * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_p = d_prices.as_device_ptr().as_raw();
            let mut weights_p = d_weights.as_device_ptr().as_raw();
            let mut looks_p = d_looks.as_device_ptr().as_raw();
            let mut mults_p = d_mults.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n_combos as i32;
            let mut fv_i = first_valid as i32;
            let mut max_lb_i = max_lb as i32;
            let mut upper_p = d_upper.as_device_ptr().as_raw();
            let mut lower_p = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_p as *mut _ as *mut c_void,
                &mut weights_p as *mut _ as *mut c_void,
                &mut looks_p as *mut _ as *mut c_void,
                &mut mults_p as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut max_lb_i as *mut _ as *mut c_void,
                &mut upper_p as *mut _ as *mut c_void,
                &mut lower_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, smem_bytes, args)
                .map_err(CudaNweError::from)?;
        }
        Ok(())
    }

    pub fn nwe_many_series_one_param_time_major_dev(
        &self,
        data_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &NweParams,
    ) -> Result<DeviceNwePair, CudaNweError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNweError::InvalidInput("empty matrix".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaNweError::InvalidInput("empty matrix".into()));
        }
        if data_tm.len() != cols * rows {
            return Err(CudaNweError::InvalidInput(
                "matrix shape mismatch (time-major)".into(),
            ));
            return Err(CudaNweError::InvalidInput(
                "matrix shape mismatch (time-major)".into(),
            ));
        }
        let bandwidth = params.bandwidth.unwrap_or(8.0);
        let lookback = params.lookback.unwrap_or(500);
        let multiplier = params.multiplier.unwrap_or(3.0) as f32;
        if lookback == 0 {
            return Err(CudaNweError::InvalidInput("lookback must be > 0".into()));
        }
        if lookback == 0 {
            return Err(CudaNweError::InvalidInput("lookback must be > 0".into()));
        }

        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let v = data_tm[t * cols + s];
                if !v.is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        for s in 0..cols {
            for t in 0..rows {
                let v = data_tm[t * cols + s];
                if !v.is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let (w_row, _l) = Self::compute_weights_row(bandwidth, lookback);

        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let mut required = 0usize;
        required = required
            .checked_add(
                data_tm
                    .len()
                    .checked_mul(sz_f32)
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        required = required
            .checked_add(
                w_row
                    .len()
                    .checked_mul(sz_f32)
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        required = required
            .checked_add(
                first_valids
                    .len()
                    .checked_mul(sz_i32)
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        required = required
            .checked_add(
                data_tm
                    .len()
                    .checked_mul(2)
                    .and_then(|x| x.checked_mul(sz_f32))
                    .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?,
            )
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;

        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Ok((free, _)) = mem_get_info() {
                return Err(CudaNweError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaNweError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let h_tm = LockedBuffer::from_slice(data_tm).map_err(CudaNweError::from)?;
        let len_tm = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaNweError::InvalidInput("size overflow".into()))?;
        let mut d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len_tm) }.map_err(CudaNweError::from)?;
        unsafe {
            d_prices
                .async_copy_from(&h_tm, &self.stream)
                .map_err(CudaNweError::from)?;
        }
        let d_weights = DeviceBuffer::from_slice(&w_row).map_err(CudaNweError::from)?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaNweError::from)?;
        let mut d_upper: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len_tm) }.map_err(CudaNweError::from)?;
        let mut d_lower: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len_tm) }.map_err(CudaNweError::from)?;

        let func = self
            .module
            .get_function("nadaraya_watson_envelope_many_series_one_param_f32")
            .map_err(|_| CudaNweError::MissingKernelSymbol {
                name: "nadaraya_watson_envelope_many_series_one_param_f32",
            })?;
        let grid = GridSize::xy(1, cols as u32);
        let block = BlockSize::xyz(128, 1, 1);
        self.validate_launch(grid.x, grid.y, grid.z, block.x, block.y, block.z)?;

        unsafe {
            let mut prices_p = d_prices.as_device_ptr().as_raw();
            let mut weights_p = d_weights.as_device_ptr().as_raw();
            let mut lookback_i = lookback as i32;
            let mut mult_f = multiplier;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_p = d_first.as_device_ptr().as_raw();
            let mut out_u_p = d_upper.as_device_ptr().as_raw();
            let mut out_l_p = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_p as *mut _ as *mut c_void,
                &mut weights_p as *mut _ as *mut c_void,
                &mut lookback_i as *mut _ as *mut c_void,
                &mut mult_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_p as *mut _ as *mut c_void,
                &mut out_u_p as *mut _ as *mut c_void,
                &mut out_l_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaNweError::from)?;
        }

        self.stream.synchronize().map_err(CudaNweError::from)?;

        Ok(DeviceNwePair {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols,
            },
        })
    }
}

#[cfg(feature = "cuda")]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use cust::memory::LockedBuffer;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000;

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        struct BatchDevState {
            cuda: CudaNwe,
            d_prices: DeviceBuffer<f32>,
            d_weights: DeviceBuffer<f32>,
            d_looks: DeviceBuffer<i32>,
            d_mults: DeviceBuffer<f32>,
            len: usize,
            n_combos: usize,
            first_valid: usize,
            max_lb: usize,
            d_upper: DeviceBuffer<f32>,
            d_lower: DeviceBuffer<f32>,
        }
        impl CudaBenchState for BatchDevState {
            fn launch(&mut self) {
                self.cuda
                    .nwe_batch_device(
                        &self.d_prices,
                        &self.d_weights,
                        &self.d_looks,
                        &self.d_mults,
                        self.len,
                        self.n_combos,
                        self.first_valid,
                        self.max_lb,
                        &mut self.d_upper,
                        &mut self.d_lower,
                    )
                    .expect("nwe batch device");
                self.cuda.synchronize().expect("sync");
            }
        }
        let prep_batch = || {
            let cuda = CudaNwe::new(0).expect("cuda");
            let price = gen_series(ONE_SERIES_LEN);
            let sweep = NweBatchRange {
                bandwidth: (6.0, 12.0, 2.0),
                multiplier: (2.0, 3.0, 0.5),
                lookback: (128, 512, 64),
            };
            let (_combos, first_valid, len, lookbacks, multipliers, weights_flat, max_lb) =
                CudaNwe::prepare_batch_inputs(&price, &sweep).expect("prep");
            let n_combos = lookbacks.len();
            let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
            let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
            let d_looks = DeviceBuffer::from_slice(&lookbacks).expect("d_looks");
            let d_mults = DeviceBuffer::from_slice(&multipliers).expect("d_mults");
            let out_len = n_combos * len;
            let d_upper: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(out_len) }.expect("d_upper");
            let d_lower: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(out_len) }.expect("d_lower");
            Box::new(BatchDevState {
                cuda,
                d_prices,
                d_weights,
                d_looks,
                d_mults,
                len,
                n_combos,
                first_valid,
                max_lb,
                d_upper,
                d_lower,
            }) as Box<dyn CudaBenchState>
        };

        struct ManyState {
            cuda: CudaNwe,
            d_prices_tm: DeviceBuffer<f32>,
            d_weights: DeviceBuffer<f32>,
            d_first_valids: DeviceBuffer<i32>,
            cols: usize,
            rows: usize,
            lookback: i32,
            multiplier: f32,
            d_upper: DeviceBuffer<f32>,
            d_lower: DeviceBuffer<f32>,
        }
        impl CudaBenchState for ManyState {
            fn launch(&mut self) {
                let func = self
                    .cuda
                    .module
                    .get_function("nadaraya_watson_envelope_many_series_one_param_f32")
                    .expect("nadaraya_watson_envelope_many_series_one_param_f32");
                let grid = GridSize::xy(1, self.cols as u32);
                let block = BlockSize::xyz(128, 1, 1);
                self.cuda
                    .validate_launch(grid.x, grid.y, grid.z, block.x, block.y, block.z)
                    .expect("nwe many validate");

                unsafe {
                    let mut prices_p = self.d_prices_tm.as_device_ptr().as_raw();
                    let mut weights_p = self.d_weights.as_device_ptr().as_raw();
                    let mut lookback_i = self.lookback as i32;
                    let mut mult_f = self.multiplier;
                    let mut num_series_i = self.cols as i32;
                    let mut series_len_i = self.rows as i32;
                    let mut first_p = self.d_first_valids.as_device_ptr().as_raw();
                    let mut out_u_p = self.d_upper.as_device_ptr().as_raw();
                    let mut out_l_p = self.d_lower.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_p as *mut _ as *mut c_void,
                        &mut weights_p as *mut _ as *mut c_void,
                        &mut lookback_i as *mut _ as *mut c_void,
                        &mut mult_f as *mut _ as *mut c_void,
                        &mut num_series_i as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut first_p as *mut _ as *mut c_void,
                        &mut out_u_p as *mut _ as *mut c_void,
                        &mut out_l_p as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&func, grid, block, 0, args)
                        .expect("nwe many launch");
                }
                self.cuda.synchronize().expect("nwe many sync");
            }
        }
        let prep_many = || {
            let cuda = CudaNwe::new(0).expect("cuda");
            let cols = MANY_SERIES_COLS;
            let rows = MANY_SERIES_LEN;
            let data_tm = gen_time_major_prices(cols, rows);

            let bandwidth = 8.0;
            let lookback = 256usize;
            let multiplier = 3.0f32;

            let mut first_valids = vec![rows as i32; cols];
            for s in 0..cols {
                for t in 0..rows {
                    let v = data_tm[t * cols + s];
                    if !v.is_nan() {
                        first_valids[s] = t as i32;
                        break;
                    }
                }
            }
            let (w_row, _l) = CudaNwe::compute_weights_row(bandwidth, lookback);

            let h_tm = LockedBuffer::from_slice(&data_tm).expect("h_tm locked");
            let len_tm = cols * rows;
            let mut d_prices_tm: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(len_tm) }.expect("d_prices_tm");
            unsafe {
                d_prices_tm
                    .async_copy_from(&h_tm, &cuda.stream)
                    .expect("d_prices_tm H2D");
            }
            let d_weights = DeviceBuffer::from_slice(&w_row).expect("d_weights");
            let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
            let d_upper: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(len_tm) }.expect("d_upper");
            let d_lower: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(len_tm) }.expect("d_lower");
            cuda.stream.synchronize().expect("nwe prep sync");
            Box::new(ManyState {
                cuda,
                d_prices_tm,
                d_weights,
                d_first_valids,
                cols,
                rows,
                lookback: lookback as i32,
                multiplier,
                d_upper,
                d_lower,
            }) as Box<dyn CudaBenchState>
        };
        let bytes_batch = ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + (ONE_SERIES_LEN * 256) * std::mem::size_of::<f32>()
            + 64 * 1024 * 1024;
        let bytes_many = MANY_SERIES_COLS * MANY_SERIES_LEN * 3 * std::mem::size_of::<f32>()
            + 256usize * std::mem::size_of::<f32>()
            + MANY_SERIES_COLS * std::mem::size_of::<i32>()
            + 64 * 1024 * 1024;
        vec![
            CudaBenchScenario::new(
                "nwe",
                "one_series_many_params",
                "nwe_cuda_batch_dev",
                "1m_x_grid",
                prep_batch,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_batch),
            CudaBenchScenario::new(
                "nwe",
                "many_series_one_param",
                "nwe_cuda_many_series_one_param",
                "256x1m",
                prep_many,
            )
            .with_mem_required(bytes_many),
        ]
    }
}
