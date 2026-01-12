#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::devstop::DevStopBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaDevStopError {
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

#[derive(Clone, Debug)]
pub struct DevStopCombo {
    pub period: usize,
    pub mult: f32,
}

pub struct CudaDevStop {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Float2 {
    x: f32,
    y: f32,
}
unsafe impl cust::memory::DeviceCopy for Float2 {}

impl CudaDevStop {
    pub fn new(device_id: usize) -> Result<Self, CudaDevStopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/devstop_kernel.ptx"));
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        ) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[])?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaDevStopError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            let need = required_bytes.saturating_add(headroom_bytes);
            if need > free {
                return Err(CudaDevStopError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    #[inline]
    fn validate_launch(grid: GridSize, block: BlockSize) -> Result<(), CudaDevStopError> {
        let (gx, gy, gz) = (grid.x, grid.y, grid.z);
        let (bx, by, bz) = (block.x, block.y, block.z);
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaDevStopError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        if bx.saturating_mul(by).saturating_mul(bz) > 1024 {
            return Err(CudaDevStopError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        if gx > 65_535 || gy > 65_535 || gz > 65_535 {
            return Err(CudaDevStopError::LaunchConfigTooLarge {
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

    fn expand_grid(
        range: &DevStopBatchRange,
    ) -> Result<Vec<(usize, f32, usize)>, CudaDevStopError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaDevStopError> {
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
                return Err(CudaDevStopError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaDevStopError> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return Ok(vec![start]);
            }
            if start < end {
                let mut v = Vec::new();
                let mut x = start;
                let st = step.abs();
                while x <= end + 1e-12 {
                    v.push(x);
                    x += st;
                }
                if v.is_empty() {
                    return Err(CudaDevStopError::InvalidInput(format!(
                        "Invalid range: start={}, end={}, step={}",
                        start, end, step
                    )));
                }
                return Ok(v);
            }
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x + 1e-12 >= end {
                v.push(x);
                x -= st;
            }
            if v.is_empty() {
                return Err(CudaDevStopError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }
        let periods = axis_usize(range.period)?;
        let mults = axis_f64(range.mult)?;
        let devtypes = axis_usize(range.devtype)?;
        let cap = periods
            .len()
            .checked_mul(mults.len())
            .and_then(|x| x.checked_mul(devtypes.len()))
            .ok_or_else(|| CudaDevStopError::InvalidInput("range size overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &p in &periods {
            for &m in &mults {
                for &d in &devtypes {
                    out.push((p, m as f32, d));
                }
            }
        }
        if out.is_empty() {
            return Err(CudaDevStopError::InvalidInput(
                "empty batch expansion".into(),
            ));
        }
        Ok(out)
    }

    fn first_valid_hl(high: &[f32], low: &[f32]) -> Option<usize> {
        let fh = high.iter().position(|v| !v.is_nan());
        let fl = low.iter().position(|v| !v.is_nan());
        match (fh, fl) {
            (Some(h), Some(l)) => Some(h.min(l)),
            _ => None,
        }
    }

    fn build_range_prefixes(
        high: &[f32],
        low: &[f32],
    ) -> (Vec<Float2>, Vec<Float2>, Vec<i32>, usize) {
        let len = high.len().min(low.len());
        let first = Self::first_valid_hl(high, low).unwrap_or(0);

        let mut p1 = vec![Float2 { x: 0.0, y: 0.0 }; len + 1];
        let mut p2 = vec![Float2 { x: 0.0, y: 0.0 }; len + 1];
        let mut pc = vec![0i32; len + 1];

        let mut s1_hi = 0.0f32;
        let mut s1_lo = 0.0f32;
        let mut s2_hi = 0.0f32;
        let mut s2_lo = 0.0f32;
        let mut accc = 0i32;
        let mut prev_h = if first < len { high[first] } else { f32::NAN };
        let mut prev_l = if first < len { low[first] } else { f32::NAN };

        #[inline(always)]
        fn two_sum(a: f32, b: f32) -> (f32, f32) {
            let s = a + b;
            let bb = s - a;
            let e = (a - (s - bb)) + (b - bb);
            (s, e)
        }
        #[inline(always)]
        fn quick_two_sum(a: f32, b: f32) -> (f32, f32) {
            let s = a + b;
            let e = b - (s - a);
            (s, e)
        }
        #[inline(always)]
        fn ds_add_host(hi: &mut f32, lo: &mut f32, x: f32) {
            let (s, e) = two_sum(*hi, x);
            let (hh, ll) = quick_two_sum(s, *lo + e);
            *hi = hh;
            *lo = ll;
        }

        for i in 0..len {
            if i >= first + 1 {
                let h = high[i];
                let l = low[i];
                if !h.is_nan() && !l.is_nan() && !prev_h.is_nan() && !prev_l.is_nan() {
                    let hi2 = if h > prev_h { h } else { prev_h };
                    let lo2 = if l < prev_l { l } else { prev_l };
                    let r = hi2 - lo2;
                    let r2 = r * r;
                    ds_add_host(&mut s1_hi, &mut s1_lo, r);
                    ds_add_host(&mut s2_hi, &mut s2_lo, r2);
                    accc += 1;
                }
                prev_h = h;
                prev_l = l;
            }
            p1[i + 1] = Float2 { x: s1_hi, y: s1_lo };
            p2[i + 1] = Float2 { x: s2_hi, y: s2_lo };
            pc[i + 1] = accc;
        }
        (p1, p2, pc, first)
    }

    pub fn devstop_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &DevStopBatchRange,
        is_long: bool,
    ) -> Result<(DeviceArrayF32, Vec<(usize, f32)>), CudaDevStopError> {
        let len = high.len().min(low.len());
        if len == 0 {
            return Err(CudaDevStopError::InvalidInput("empty inputs".into()));
        }
        let first = Self::first_valid_hl(high, low)
            .ok_or_else(|| CudaDevStopError::InvalidInput("all values are NaN".into()))?;

        let combos_raw = Self::expand_grid(sweep)?;

        for &(_, _, dt) in &combos_raw {
            if dt != 0 {
                return Err(CudaDevStopError::InvalidInput(
                    "unsupported devtype (only 0=stddev supported in CUDA batch)".into(),
                ));
            }
        }

        let mut groups: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        for (p, m, _dt) in combos_raw {
            groups.entry(p).or_default().push(m);
        }

        let (p1, p2, pc, first_valid) = Self::build_range_prefixes(high, low);

        let mut d_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;
        let mut d_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;
        let mut d_p1: DeviceBuffer<Float2> = unsafe { DeviceBuffer::uninitialized(p1.len()) }?;
        let mut d_p2: DeviceBuffer<Float2> = unsafe { DeviceBuffer::uninitialized(p2.len()) }?;
        let mut d_pc: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(pc.len()) }?;
        unsafe {
            d_high.async_copy_from(&high[..len], &self.stream)?;
            d_low.async_copy_from(&low[..len], &self.stream)?;
            d_p1.async_copy_from(&p1, &self.stream)?;
            d_p2.async_copy_from(&p2, &self.stream)?;
            d_pc.async_copy_from(&pc, &self.stream)?;
        }

        let mut total_rows: usize = 0;
        for v in groups.values() {
            total_rows = total_rows
                .checked_add(v.len())
                .ok_or_else(|| CudaDevStopError::InvalidInput("rows overflow".into()))?;
        }
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_f2 = std::mem::size_of::<Float2>();
        let sz_i32 = std::mem::size_of::<i32>();
        let bytes_high = high
            .len()
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        let bytes_low = low
            .len()
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        let bytes_p1 = p1
            .len()
            .checked_mul(sz_f2)
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        let bytes_p2 = p2
            .len()
            .checked_mul(sz_f2)
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        let bytes_pc = pc
            .len()
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        let bytes_inputs = bytes_high
            .checked_add(bytes_low)
            .and_then(|b| b.checked_add(bytes_p1))
            .and_then(|b| b.checked_add(bytes_p2))
            .and_then(|b| b.checked_add(bytes_pc))
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        let elems_out = total_rows
            .checked_mul(len)
            .ok_or_else(|| CudaDevStopError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_out = elems_out
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_out)
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems_out) }?;

        let func = self
            .module
            .get_function("devstop_batch_grouped_f32")
            .map_err(|_| CudaDevStopError::MissingKernelSymbol {
                name: "devstop_batch_grouped_f32",
            })?;

        let mut out_row_base = 0usize;
        let mut meta_launch_order: Vec<(usize, f32)> = Vec::with_capacity(total_rows);
        for (period, mults_host) in groups.into_iter() {
            if period == 0 || period > len {
                return Err(CudaDevStopError::InvalidInput(format!(
                    "invalid period {}",
                    period
                )));
            }
            if period == 0 || period > len {
                return Err(CudaDevStopError::InvalidInput(format!(
                    "invalid period {}",
                    period
                )));
            }
            let n = mults_host.len();
            let d_mults = DeviceBuffer::from_slice(&mults_host)?;

            for &m in &mults_host {
                meta_launch_order.push((period, m));
            }

            let block_x: u32 = 64;
            let grid_x: u32 = (n as u32).max(1);
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            Self::validate_launch(grid, block)?;

            let per_block = std::mem::size_of::<f32>() + std::mem::size_of::<i32>();
            let shmem_bytes_usize = period
                .checked_mul(per_block)
                .ok_or_else(|| CudaDevStopError::InvalidInput("shared mem size overflow".into()))?;
            let shmem_bytes = if shmem_bytes_usize > u32::MAX as usize {
                return Err(CudaDevStopError::InvalidInput(
                    "shared mem size exceeds u32".into(),
                ));
            } else {
                shmem_bytes_usize as u32
            };

            unsafe {
                let mut high_ptr = d_high.as_device_ptr().as_raw();
                let mut low_ptr = d_low.as_device_ptr().as_raw();
                let mut p1_ptr = d_p1.as_device_ptr().as_raw();
                let mut p2_ptr = d_p2.as_device_ptr().as_raw();
                let mut pc_ptr = d_pc.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = (first_valid as i32).min(len_i);
                let mut period_i = period as i32;
                let mut mults_ptr = d_mults.as_device_ptr().as_raw();
                let mut n_i = n as i32;
                let mut long_i = if is_long { 1i32 } else { 0i32 };
                let mut base_i = out_row_base as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut p1_ptr as *mut _ as *mut c_void,
                    &mut p2_ptr as *mut _ as *mut c_void,
                    &mut pc_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut mults_ptr as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut long_i as *mut _ as *mut c_void,
                    &mut base_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, shmem_bytes, args)?;
            }
            out_row_base += n;
        }

        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: total_rows,
                cols: len,
            },
            meta_launch_order,
        ))
    }

    pub fn devstop_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        is_long: bool,
    ) -> Result<DeviceArrayF32, CudaDevStopError> {
        if cols == 0 || rows == 0 {
            return Err(CudaDevStopError::InvalidInput(
                "cols/rows must be > 0".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDevStopError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != low_tm.len() || high_tm.len() != expected {
            return Err(CudaDevStopError::InvalidInput(
                "time-major arrays must match cols*rows".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaDevStopError::InvalidInput("invalid period".into()));
        }

        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let h = high_tm[t * cols + s];
                let l = low_tm[t * cols + s];
                if !h.is_nan() && !l.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
                let l = low_tm[t * cols + s];
                if !h.is_nan() && !l.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            firsts[s] = fv.unwrap_or(0);
        }

        let d_high = DeviceBuffer::from_slice(high_tm)?;
        let d_low = DeviceBuffer::from_slice(low_tm)?;
        let d_firsts = DeviceBuffer::from_slice(&firsts)?;
        let elems_out = expected;
        let bytes_out = elems_out
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDevStopError::InvalidInput("size overflow".into()))?;
        Self::will_fit(bytes_out, 32 * 1024 * 1024)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems_out) }?;

        let func = self
            .module
            .get_function("devstop_many_series_one_param_f32")
            .map_err(|_| CudaDevStopError::MissingKernelSymbol {
                name: "devstop_many_series_one_param_f32",
            })?;

        let grid: GridSize = ((cols as u32).max(1), 1, 1).into();
        let block: BlockSize = (64, 1, 1).into();
        Self::validate_launch(grid, block)?;

        let per_block = 2 * std::mem::size_of::<f32>() + std::mem::size_of::<i32>();
        let shmem_bytes_usize = period
            .checked_mul(per_block)
            .ok_or_else(|| CudaDevStopError::InvalidInput("shared mem size overflow".into()))?;
        let shmem_bytes = if shmem_bytes_usize > u32::MAX as usize {
            return Err(CudaDevStopError::InvalidInput(
                "shared mem size exceeds u32".into(),
            ));
        } else {
            shmem_bytes_usize as u32
        };

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut mult_f = mult as f32;
            let mut is_long_i = if is_long { 1i32 } else { 0i32 };
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut mult_f as *mut _ as *mut c_void,
                &mut is_long_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shmem_bytes, args)?;
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
    const BATCH_PERIOD: usize = 20;
    const MULT_SWEEP: usize = 200;

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0021;
            let off = 0.20 + 0.01 * (x.sin().abs());
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct DevStopBatchDevInplaceState {
        cuda: CudaDevStop,
        len: usize,
        first_valid: usize,
        period: usize,
        is_long: bool,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_p1: DeviceBuffer<Float2>,
        d_p2: DeviceBuffer<Float2>,
        d_pc: DeviceBuffer<i32>,
        d_mults: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for DevStopBatchDevInplaceState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("devstop_batch_grouped_f32")
                .expect("devstop_batch_grouped_f32");
            let n_combos = self.d_mults.len() as i32;
            let grid: GridSize = ((n_combos as u32).max(1), 1, 1).into();
            let block: BlockSize = (64u32, 1, 1).into();
            CudaDevStop::validate_launch(grid, block).expect("devstop validate launch");
            let per_block = std::mem::size_of::<f32>() + std::mem::size_of::<i32>();
            let shmem_bytes = (self.period * per_block) as u32;
            unsafe {
                let mut high_ptr = self.d_high.as_device_ptr().as_raw();
                let mut low_ptr = self.d_low.as_device_ptr().as_raw();
                let mut p1_ptr = self.d_p1.as_device_ptr().as_raw();
                let mut p2_ptr = self.d_p2.as_device_ptr().as_raw();
                let mut pc_ptr = self.d_pc.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut first_i = self.first_valid as i32;
                let mut period_i = self.period as i32;
                let mut mults_ptr = self.d_mults.as_device_ptr().as_raw();
                let mut n_i = n_combos;
                let mut is_long_i = if self.is_long { 1i32 } else { 0i32 };
                let mut out_row_base_i = 0i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut p1_ptr as *mut _ as *mut c_void,
                    &mut p2_ptr as *mut _ as *mut c_void,
                    &mut pc_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut mults_ptr as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut is_long_i as *mut _ as *mut c_void,
                    &mut out_row_base_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, shmem_bytes, args)
                    .expect("devstop launch");
            }
            self.cuda.stream.synchronize().expect("devstop sync");
        }
    }

    fn prep_batch_dev_inplace() -> Box<dyn CudaBenchState> {
        let cuda = CudaDevStop::new(0).expect("cuda devstop");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_close(&close);
        let len = high.len().min(low.len());
        let (p1, p2, pc, first_valid) = CudaDevStop::build_range_prefixes(&high, &low);

        let mult_end = 0.01f64 * ((MULT_SWEEP - 1) as f64);
        let mut mults = Vec::with_capacity(MULT_SWEEP);
        let mut x = 0.0f64;
        while x <= mult_end + 1e-12 {
            mults.push(x as f32);
            x += 0.01;
        }
        if mults.is_empty() {
            mults.push(0.0);
        }

        let rows = mults.len();
        let elems_out = rows.checked_mul(len).expect("devstop bench size overflow");

        let d_high =
            unsafe { DeviceBuffer::from_slice_async(&high[..len], &cuda.stream) }.expect("d_high");
        let d_low =
            unsafe { DeviceBuffer::from_slice_async(&low[..len], &cuda.stream) }.expect("d_low");
        let d_p1 = unsafe { DeviceBuffer::from_slice_async(&p1, &cuda.stream) }.expect("d_p1");
        let d_p2 = unsafe { DeviceBuffer::from_slice_async(&p2, &cuda.stream) }.expect("d_p2");
        let d_pc = unsafe { DeviceBuffer::from_slice_async(&pc, &cuda.stream) }.expect("d_pc");
        let d_mults =
            unsafe { DeviceBuffer::from_slice_async(&mults, &cuda.stream) }.expect("d_mults");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, &cuda.stream) }.expect("d_out");
        cuda.stream.synchronize().expect("devstop sync");

        Box::new(DevStopBatchDevInplaceState {
            cuda,
            len,
            first_valid,
            period: BATCH_PERIOD,
            is_long: true,
            d_high,
            d_low,
            d_p1,
            d_p2,
            d_pc,
            d_mults,
            d_out,
        })
    }

    struct DevStopManySeriesState {
        cuda: CudaDevStop,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        is_long: bool,
        shmem_bytes: u32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for DevStopManySeriesState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("devstop_many_series_one_param_f32")
                .expect("devstop_many_series_one_param_f32");
            let grid: GridSize = ((self.cols as u32).max(1), 1, 1).into();
            let block: BlockSize = (64, 1, 1).into();
            unsafe {
                let mut high_ptr = self.d_high_tm.as_device_ptr().as_raw();
                let mut low_ptr = self.d_low_tm.as_device_ptr().as_raw();
                let mut firsts_ptr = self.d_first_valids.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut period_i = self.period as i32;
                let mut mult_f = self.mult as f32;
                let mut is_long_i = if self.is_long { 1i32 } else { 0i32 };
                let mut out_ptr = self.d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut firsts_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut mult_f as *mut _ as *mut c_void,
                    &mut is_long_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, self.shmem_bytes, args)
                    .expect("devstop many-series launch");
            }
            self.cuda.stream.synchronize().expect("devstop sync");
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaDevStop::new(0).expect("cuda devstop");
        let cols = 128usize;
        let rows = 1_000_000usize / cols;
        let close = gen_series(cols * rows);

        let mut high_tm = close.clone();
        let mut low_tm = close.clone();
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let v = close[idx];
                if v.is_nan() {
                    continue;
                }
                let x = (t as f32) * 0.002 + s as f32 * 0.01;
                let off = 0.18 + 0.01 * (x.cos().abs());
                high_tm[idx] = v + off;
                low_tm[idx] = v - off;
            }
        }

        let mut first_valids: Vec<i32> = vec![0; cols];
        for s in 0..cols {
            let mut fv = 0i32;
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }
        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        let period = 20usize;
        let per_block = 2 * std::mem::size_of::<f32>() + std::mem::size_of::<i32>();
        let shmem_bytes = (period * per_block) as u32;
        let is_long = true;
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(DevStopManySeriesState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_first_valids,
            cols,
            rows,
            period,
            mult: 1.5,
            is_long,
            shmem_bytes,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "devstop",
                "batch_dev",
                "devstop_cuda_batch_dev_inplace",
                "1m_p20_x_200",
                prep_batch_dev_inplace,
            )
            .with_sample_size(10),
            CudaBenchScenario::new(
                "devstop",
                "many_series_one_param",
                "devstop_cuda_many_series_one_param_dev",
                "128x8k",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_inner_iters(3),
        ]
    }
}
