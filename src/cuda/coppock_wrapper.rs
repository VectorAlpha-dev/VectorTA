#![cfg(feature = "cuda")]

use crate::indicators::coppock::CoppockBatchRange;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaCoppockError {
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

#[derive(Clone, Copy, Debug)]
pub struct CudaCoppockPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaCoppockPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaCoppock {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaCoppockPolicy,

    debug_batch_logged: bool,
    debug_many_logged: bool,
}

pub struct DeviceArrayF32Coppock {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}

impl DeviceArrayF32Coppock {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }
}

impl CudaCoppock {
    pub fn new(device_id: usize) -> Result<Self, CudaCoppockError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/coppock_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaCoppockPolicy {
                batch: BatchKernelPolicy::Auto,
                many_series: ManySeriesKernelPolicy::Auto,
            },
            debug_batch_logged: false,
            debug_many_logged: false,
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
    fn will_fit(&self, required: usize, headroom: usize) -> Result<(), CudaCoppockError> {
        match mem_get_info() {
            Ok((free, _total)) => {
                if required.saturating_add(headroom) > free {
                    return Err(CudaCoppockError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    });
                }
                Ok(())
            }
            Err(e) => Err(CudaCoppockError::Cuda(e)),
        }
    }

    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaCoppockError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)?
            .max(1) as u32;
        let max_grid_x = device.get_attribute(DeviceAttribute::MaxGridDimX)?.max(1) as u32;
        let max_grid_y = device.get_attribute(DeviceAttribute::MaxGridDimY)?.max(1) as u32;
        let max_grid_z = device.get_attribute(DeviceAttribute::MaxGridDimZ)?.max(1) as u32;

        let threads_per_block = bx.saturating_mul(by).saturating_mul(bz);
        if threads_per_block > max_threads || gx > max_grid_x || gy > max_grid_y || gz > max_grid_z
        {
            return Err(CudaCoppockError::LaunchConfigTooLarge {
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

    #[inline]
    pub fn set_policy(&mut self, p: CudaCoppockPolicy) {
        self.policy = p;
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaCoppockError> {
        self.stream.synchronize().map_err(Into::into)
    }

    pub fn coppock_batch_dev(
        &self,
        price: &[f32],
        sweep: &CoppockBatchRange,
    ) -> Result<DeviceArrayF32Coppock, CudaCoppockError> {
        let len = price.len();
        if len == 0 {
            return Err(CudaCoppockError::InvalidInput("empty series".into()));
        }
        let first_valid = price
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaCoppockError::InvalidInput("all values are NaN".into()))?;

        let (shorts, longs, ma_periods) = expand_grid(sweep)?;
        let rows = ma_periods.len();
        if rows == 0 {
            return Err(CudaCoppockError::InvalidInput("no parameter combos".into()));
        }

        for ((&s, &l), &m) in shorts.iter().zip(longs.iter()).zip(ma_periods.iter()) {
            let (s_u, l_u, m_u) = (s as usize, l as usize, m as usize);
            if s_u == 0 || l_u == 0 || m_u == 0 || s_u > len || l_u > len || m_u > len {
                return Err(CudaCoppockError::InvalidInput(format!(
                    "invalid params s={} l={} m={} for len {}",
                    s_u, l_u, m_u, len
                )));
            }
            let largest = s_u.max(l_u);
            if len - first_valid < largest {
                return Err(CudaCoppockError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
        }

        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let prices_bytes = len
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaCoppockError::InvalidInput("price bytes overflow".into()))?;
        let inv_bytes = len
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaCoppockError::InvalidInput("inv bytes overflow".into()))?;
        let params_bytes = rows
            .checked_mul(3usize)
            .and_then(|v| v.checked_mul(elem_i32))
            .ok_or_else(|| CudaCoppockError::InvalidInput("params bytes overflow".into()))?;
        let out_elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaCoppockError::InvalidInput("rows*len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaCoppockError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(inv_bytes)
            .and_then(|v| v.checked_add(params_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaCoppockError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        self.will_fit(required, headroom)?;

        let d_price = DeviceBuffer::from_slice(price)?;
        let mut inv = vec![0f32; len];
        for i in 0..len {
            inv[i] = 1.0f32 / price[i];
        }
        let d_inv = DeviceBuffer::from_slice(&inv)?;

        let bytes_params = rows
            .checked_mul(3usize)
            .and_then(|v| v.checked_mul(elem_i32))
            .unwrap_or(0);
        let bytes_out_total = out_elems.checked_mul(elem_f32).unwrap_or(0);
        let headroom = 64usize * 1024 * 1024;
        let fits_single = match mem_get_info() {
            Ok((free, _)) => {
                bytes_params
                    .saturating_add(bytes_out_total)
                    .saturating_add(headroom)
                    <= free
            }
            Err(_) => true,
        };

        if fits_single {
            let d_s = DeviceBuffer::from_slice(&shorts)?;
            let d_l = DeviceBuffer::from_slice(&longs)?;
            let d_m = DeviceBuffer::from_slice(&ma_periods)?;
            let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }?;
            self.launch_batch(
                &d_price,
                &d_inv,
                len,
                first_valid,
                &d_s,
                &d_l,
                &d_m,
                rows,
                &mut d_out,
                0,
            )?;
            self.maybe_log_batch_debug();
            return Ok(DeviceArrayF32Coppock {
                buf: d_out,
                rows,
                cols: len,
                ctx: Arc::clone(&self.context),
                device_id: self.device_id,
            });
        }

        let try_out: Result<DeviceBuffer<f32>, _> =
            unsafe { DeviceBuffer::uninitialized(rows * len) };
        if let Ok(mut d_out_full) = try_out {
            let mut start = 0usize;
            let max_chunk = 65_535usize;
            while start < rows {
                let remain = rows - start;
                let chunk = remain.min(max_chunk);
                let d_s = DeviceBuffer::from_slice(&shorts[start..start + chunk])
                    .map_err(CudaCoppockError::Cuda)?;
                let d_l = DeviceBuffer::from_slice(&longs[start..start + chunk])
                    .map_err(CudaCoppockError::Cuda)?;
                let d_m = DeviceBuffer::from_slice(&ma_periods[start..start + chunk])
                    .map_err(CudaCoppockError::Cuda)?;
                self.launch_batch(
                    &d_price,
                    &d_inv,
                    len,
                    first_valid,
                    &d_s,
                    &d_l,
                    &d_m,
                    chunk,
                    &mut d_out_full,
                    start * len,
                )?;
                start += chunk;
            }
            self.maybe_log_batch_debug();
            return Ok(DeviceArrayF32Coppock {
                buf: d_out_full,
                rows,
                cols: len,
                ctx: Arc::clone(&self.context),
                device_id: self.device_id,
            });
        }

        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }?;
        let mut start = 0usize;
        let max_chunk = 65_535usize;
        while start < rows {
            let remain = rows - start;
            let chunk = remain.min(max_chunk);
            let d_s = DeviceBuffer::from_slice(&shorts[start..start + chunk])?;
            let d_l = DeviceBuffer::from_slice(&longs[start..start + chunk])?;
            let d_m = DeviceBuffer::from_slice(&ma_periods[start..start + chunk])?;
            self.launch_batch(
                &d_price,
                &d_inv,
                len,
                first_valid,
                &d_s,
                &d_l,
                &d_m,
                chunk,
                &mut d_out,
                start * len,
            )?;

            start += chunk;
        }
        Ok(DeviceArrayF32Coppock {
            buf: d_out,
            rows,
            cols: len,
            ctx: Arc::clone(&self.context),
            device_id: self.device_id,
        })
    }

    fn launch_batch(
        &self,
        d_price: &DeviceBuffer<f32>,
        d_inv: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_short: &DeviceBuffer<i32>,
        d_long: &DeviceBuffer<i32>,
        d_ma: &DeviceBuffer<i32>,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
        out_offset_elems: usize,
    ) -> Result<(), CudaCoppockError> {
        let func = self
            .module
            .get_function("coppock_batch_time_parallel_f32")
            .map_err(|_| CudaCoppockError::MissingKernelSymbol {
                name: "coppock_batch_time_parallel_f32",
            })?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };

        if block_x == 0 {
            return Err(CudaCoppockError::InvalidPolicy("block_x must be > 0"));
        }
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let gx = grid_x.max(1);
        self.validate_launch(gx, n_combos as u32, 1, block_x, 1, 1)?;
        let grid: GridSize = (gx, n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut price_ptr = d_price.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut s_ptr = d_short.as_device_ptr().as_raw();
            let mut l_ptr = d_long.as_device_ptr().as_raw();
            let mut m_ptr = d_ma.as_device_ptr().as_raw();
            let mut n_i = n_combos as i32;

            let base = d_out.as_device_ptr();
            let off = base.add(out_offset_elems);
            let mut out_ptr = off.as_raw();
            let mut args: [*mut c_void; 9] = [
                &mut price_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut m_ptr as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        self.stream.synchronize().map_err(Into::into)
    }

    pub fn coppock_many_series_one_param_time_major_dev(
        &self,
        price_tm: &[f32],
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        ma_period: usize,
    ) -> Result<DeviceArrayF32Coppock, CudaCoppockError> {
        if cols == 0 || rows == 0 {
            return Err(CudaCoppockError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCoppockError::InvalidInput("rows*cols overflow".into()))?;
        if price_tm.len() != expected {
            return Err(CudaCoppockError::InvalidInput(
                "time-major input mismatch".into(),
            ));
        }
        if short == 0 || long == 0 || ma_period == 0 {
            return Err(CudaCoppockError::InvalidInput("invalid periods".into()));
        }

        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = price_tm[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaCoppockError::InvalidInput(format!("series {} all NaN", s)))?;
            let largest = short.max(long);
            if rows - (fv as usize) < largest {
                return Err(CudaCoppockError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
            firsts[s] = fv;
        }

        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let first_bytes = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaCoppockError::InvalidInput("first_valid bytes overflow".into()))?;
        let price_bytes = expected
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaCoppockError::InvalidInput("price_tm bytes overflow".into()))?;
        let inv_bytes = expected
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaCoppockError::InvalidInput("inv_tm bytes overflow".into()))?;
        let out_bytes = expected
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaCoppockError::InvalidInput("out_tm bytes overflow".into()))?;
        let required = price_bytes
            .checked_add(inv_bytes)
            .and_then(|v| v.checked_add(first_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaCoppockError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        self.will_fit(required, headroom)?;

        let d_price = DeviceBuffer::from_slice(price_tm)?;

        let mut inv_tm = vec![0f32; cols * rows];
        for idx in 0..inv_tm.len() {
            inv_tm[idx] = 1.0f32 / price_tm[idx];
        }
        let d_inv = DeviceBuffer::from_slice(&inv_tm)?;
        let d_first = DeviceBuffer::from_slice(&firsts)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;
        self.launch_many(
            &d_price, &d_inv, &d_first, cols, rows, short, long, ma_period, &mut d_out,
        )?;
        self.maybe_log_many_debug();
        Ok(DeviceArrayF32Coppock {
            buf: d_out,
            rows,
            cols,
            ctx: Arc::clone(&self.context),
            device_id: self.device_id,
        })
    }

    fn launch_many(
        &self,
        d_price_tm: &DeviceBuffer<f32>,
        d_inv_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        ma_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCoppockError> {
        let func = self
            .module
            .get_function("coppock_many_series_one_param_f32")
            .map_err(|_| CudaCoppockError::MissingKernelSymbol {
                name: "coppock_many_series_one_param_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 256,
        };
        if block_x == 0 {
            return Err(CudaCoppockError::InvalidPolicy("block_x must be > 0"));
        }
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let gx = grid_x.max(1);
        self.validate_launch(gx, 1, 1, block_x, 1, 1)?;
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_ptr = d_price_tm.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_tm.as_device_ptr().as_raw();
            let mut f_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut s_i = short as i32;
            let mut l_i = long as i32;
            let mut m_i = ma_period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 9] = [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut l_i as *mut _ as *mut c_void,
                &mut m_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        self.stream.synchronize().map_err(Into::into)
    }

    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[DEBUG] Coppock batch selected kernel: 1D combos {{ block_x: {} }}",
                match self.policy.batch {
                    BatchKernelPolicy::Plain { block_x } => block_x,
                    BatchKernelPolicy::Auto => 256,
                }
            );
            unsafe {
                (*(self as *const _ as *mut CudaCoppock)).debug_batch_logged = true;
            }
        }
    }
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[DEBUG] Coppock many-series selected kernel: OneD {{ block_x: {} }}",
                match self.policy.many_series {
                    ManySeriesKernelPolicy::OneD { block_x } => block_x,
                    ManySeriesKernelPolicy::Auto => 256,
                }
            );
            unsafe {
                (*(self as *const _ as *mut CudaCoppock)).debug_many_logged = true;
            }
        }
    }
}

fn expand_grid(r: &CoppockBatchRange) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), CudaCoppockError> {
    fn axis((s, e, st): (usize, usize, usize)) -> Result<Vec<i32>, CudaCoppockError> {
        if st == 0 || s == e {
            return Ok(vec![s as i32]);
        }
        let mut out = Vec::new();
        if s < e {
            let mut cur = s;
            loop {
                out.push(cur as i32);
                if cur == e {
                    break;
                }
                cur = cur.checked_add(st).ok_or_else(|| {
                    CudaCoppockError::InvalidInput("short/long/ma range overflow".into())
                })?;
                if cur > e {
                    break;
                }
            }
        } else {
            let mut cur = s;
            loop {
                out.push(cur as i32);
                if cur == e {
                    break;
                }
                cur = cur.checked_sub(st).ok_or_else(|| {
                    CudaCoppockError::InvalidInput("short/long/ma range overflow".into())
                })?;
                if cur < e {
                    break;
                }
            }
        }
        if out.is_empty() {
            return Err(CudaCoppockError::InvalidInput(
                "empty parameter range".into(),
            ));
        }
        Ok(out)
    }
    let shorts_u = axis(r.short)?;
    let longs_u = axis(r.long)?;
    let mas_u = axis(r.ma)?;
    if shorts_u.is_empty() || longs_u.is_empty() || mas_u.is_empty() {
        return Err(CudaCoppockError::InvalidInput(
            "empty parameter grid".into(),
        ));
    }
    let cap = shorts_u
        .len()
        .checked_mul(longs_u.len())
        .and_then(|v| v.checked_mul(mas_u.len()))
        .ok_or_else(|| CudaCoppockError::InvalidInput("parameter grid too large".into()))?;
    let mut shorts = Vec::new();
    let mut longs = Vec::new();
    let mut mas = Vec::new();
    for &s in &shorts_u {
        for &l in &longs_u {
            for &m in &mas_u {
                shorts.push(s);
                longs.push(l);
                mas.push(m);
            }
        }
    }
    Ok((shorts, longs, mas))
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = PARAM_SWEEP * 3 * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct CoppockBatchDeviceState {
        cuda: CudaCoppock,
        d_price: DeviceBuffer<f32>,
        d_inv: DeviceBuffer<f32>,
        d_short: DeviceBuffer<i32>,
        d_long: DeviceBuffer<i32>,
        d_ma: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_rows: usize,
    }
    impl CudaBenchState for CoppockBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_price,
                    &self.d_inv,
                    self.len,
                    self.first_valid,
                    &self.d_short,
                    &self.d_long,
                    &self.d_ma,
                    self.n_rows,
                    &mut self.d_out,
                    0,
                )
                .expect("coppock launch_batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCoppock::new(0).expect("cuda coppock");
        let price = gen_series(ONE_SERIES_LEN);

        let sweep = CoppockBatchRange {
            short: (8, 18, 2),
            long: (20, 30, 2),
            ma: (8, 16, 2),
        };
        let first_valid = price.iter().position(|v| v.is_finite()).unwrap_or(0);
        let (shorts, longs, ma_periods) = expand_grid(&sweep).expect("expand_grid");
        let n_rows = ma_periods.len();

        let mut inv = vec![0f32; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN {
            inv[i] = 1.0f32 / price[i];
        }

        let d_price = DeviceBuffer::from_slice(&price).expect("d_price");
        let d_inv = DeviceBuffer::from_slice(&inv).expect("d_inv");
        let d_short = DeviceBuffer::from_slice(&shorts).expect("d_short");
        let d_long = DeviceBuffer::from_slice(&longs).expect("d_long");
        let d_ma = DeviceBuffer::from_slice(&ma_periods).expect("d_ma");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_rows * ONE_SERIES_LEN) }.expect("d_out");
        Box::new(CoppockBatchDeviceState {
            cuda,
            d_price,
            d_inv,
            d_short,
            d_long,
            d_ma,
            d_out,
            len: ONE_SERIES_LEN,
            first_valid,
            n_rows,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "coppock",
            "one_series_many_params",
            "coppock_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
