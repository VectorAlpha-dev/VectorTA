//! CUDA scaffolding for Bollinger Bands (SMA + standard deviation path).
//!
//! Matches ALMA-style conventions for PTX loading, stream usage, simple
//! policies, VRAM checks, and warmup/NaN semantics. For now, this CUDA path
//! supports only `matype = "sma"` and `devtype = 0` (stddev) to mirror the
//! optimized scalar path; other variants should use the CPU implementation.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::bollinger_bands::BollingerBandsBatchRange;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashSet;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaBollingerError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Debug)]
struct BbCombo {
    period: usize,
    devup: f32,
    devdn: f32,
}

/// VRAM-backed array handle for Bollinger Bands with context guard and device id.
pub struct DeviceArrayF32Bb {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Bb {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows.saturating_mul(self.cols) }
}

pub struct CudaBollingerBands {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    sm_count: u32,
    device_id: u32,
}

impl CudaBollingerBands {
    pub fn new(device_id: usize) -> Result<Self, CudaBollingerError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map(|v| v as u32)
            .unwrap_or(64);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/bollinger_bands_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O3),
        ];
        let module = match Module::from_ptx(ptx, jit) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self { module, stream, _context: context, sm_count, device_id: device_id as u32 })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self._context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaBollingerError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free { Ok(()) }
            else { Err(CudaBollingerError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes }) }
        } else { Ok(()) }
    }

    fn validate_launch(grid: GridSize, block: BlockSize) -> Result<(), CudaBollingerError> {
        let (gx, gy, gz) = (grid.x, grid.y, grid.z);
        let (bx, by, bz) = (block.x, block.y, block.z);
        let max_threads_per_block = 1024u32;
        if bx.saturating_mul(by).saturating_mul(bz) > max_threads_per_block {
            return Err(CudaBollingerError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    #[inline(always)]
    fn grid_x_for_len(&self, len: usize, block_x: u32) -> u32 {
        let need = ((len as u32) + block_x - 1) / block_x;
        let cap  = (self.sm_count.saturating_mul(4)).max(1);
        need.min(cap)
    }

    fn expand_combos(
        range: &BollingerBandsBatchRange,
    ) -> Result<Vec<(usize, f64, f64, String, usize)>, CudaBollingerError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaBollingerError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            let mut v = Vec::new();
            if start < end {
                let mut cur = start;
                while cur <= end {
                    v.push(cur);
                    cur = cur.saturating_add(step);
                    if cur == *v.last().unwrap() {
                        break;
                    }
                }
            } else {
                let mut cur = start;
                while cur >= end {
                    v.push(cur);
                    let next = cur.saturating_sub(step);
                    if next == cur {
                        break;
                    }
                    cur = next;
                    if cur == 0 && end > 0 {
                        break;
                    }
                }
            }
            if v.is_empty() {
                return Err(CudaBollingerError::InvalidInput(format!(
                    "invalid usize range: start={start} end={end} step={step}"
                )));
            }
            Ok(v)
        }
        fn axis_f64(
            (start, end, step): (f64, f64, f64),
        ) -> Result<Vec<f64>, CudaBollingerError> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return Ok(vec![start]);
            }
            let mut out = Vec::new();
            if start < end {
                let st = if step > 0.0 { step } else { -step };
                let mut x = start;
                while x <= end + 1e-12 {
                    out.push(x);
                    x += st;
                }
            } else {
                let st = if step > 0.0 { -step } else { step };
                if st.abs() < 1e-12 {
                    return Ok(vec![start]);
                }
                let mut x = start;
                while x >= end - 1e-12 {
                    out.push(x);
                    x += st;
                }
            }
            if out.is_empty() {
                return Err(CudaBollingerError::InvalidInput(format!(
                    "invalid f64 range: start={start} end={end} step={step}"
                )));
            }
            Ok(out)
        }
        fn axis_str((s, e, _): (String, String, usize)) -> Vec<String> {
            if s == e { vec![s] } else { vec![s, e] }
        }
        let periods = axis_usize(range.period)?;
        let devups = axis_f64(range.devup)?;
        let devdns = axis_f64(range.devdn)?;
        let matypes = axis_str(range.matype.clone());
        let devtypes = axis_usize(range.devtype)?;
        let mut out = Vec::with_capacity(
            periods
                .len()
                .saturating_mul(devups.len())
                .saturating_mul(devdns.len())
                .saturating_mul(matypes.len())
                .saturating_mul(devtypes.len()),
        );
        for &p in &periods {
            for &u in &devups {
                for &d in &devdns {
                    for m in &matypes {
                        for &t in &devtypes {
                            out.push((p, u, d, m.clone(), t));
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &BollingerBandsBatchRange,
    ) -> Result<(Vec<BbCombo>, usize, usize), CudaBollingerError> {
        if data_f32.is_empty() {
            return Err(CudaBollingerError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaBollingerError::InvalidInput("all values are NaN".into()))?;

        let raw = Self::expand_combos(sweep)?;
        if raw.is_empty() {
            return Err(CudaBollingerError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut unsupported_ma: HashSet<String> = HashSet::new();
        let mut combos = Vec::with_capacity(raw.len());
        for (p, u, d, ma, devt) in raw {
            if p == 0 || p > len {
                return Err(CudaBollingerError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaBollingerError::InvalidInput(format!(
                    "not enough valid data for period {} (valid after first {}: {})",
                    p,
                    first_valid,
                    len - first_valid
                )));
            }
            if devt != 0 {
                return Err(CudaBollingerError::InvalidInput(format!(
                    "unsupported devtype {} (only 0=stddev)",
                    devt
                )));
            }
            if ma.to_ascii_lowercase() != "sma" { unsupported_ma.insert(ma); continue; }
            combos.push(BbCombo { period: p, devup: u as f32, devdn: d as f32 });
        }
        if combos.is_empty() {
            if unsupported_ma.is_empty() {
                return Err(CudaBollingerError::InvalidInput(
                    "no supported combos (require ma_type='sma' and devtype=0)".into(),
                ));
                return Err(CudaBollingerError::InvalidInput(
                    "no supported combos (require ma_type='sma' and devtype=0)".into(),
                ));
            } else {
                return Err(CudaBollingerError::InvalidInput(format!(
                    "unsupported ma_type(s): {} (only 'sma' supported for CUDA)",
                    unsupported_ma.into_iter().collect::<Vec<_>>().join(", ")
                )));
            }
        }
        Ok((combos, first_valid, len))
    }

    
    #[inline(always)]
    fn two_sum(a: f32, b: f32) -> (f32, f32) {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        (s, e)
    }

    #[inline(always)]
    fn ds_add_inplace(hi: &mut f32, lo: &mut f32, bhi: f32, blo: f32) {
        let (s, e1) = Self::two_sum(*hi, bhi);
        let e = e1 + *lo + blo;
        let (t, lo_new) = Self::two_sum(s, e);
        *hi = t;
        *lo = lo_new;
    }

    fn build_prefixes(data: &[f32]) -> (Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<i32>) {
        let n = data.len();
        let mut ps  = vec![[0.0f32; 2]; n + 1];
        let mut ps2 = vec![[0.0f32; 2]; n + 1];
        let mut pn  = vec![0i32;        n + 1];
        let (mut s_hi,  mut s_lo)  = (0.0f32, 0.0f32);
        let (mut s2_hi, mut s2_lo) = (0.0f32, 0.0f32);
        let mut an = 0i32;
        for i in 0..n {
            let v = data[i];
            if v.is_nan() {
                an += 1;
            } else {
                Self::ds_add_inplace(&mut s_hi, &mut s_lo, v, 0.0);
                let p = v * v; let err = v.mul_add(v, -p);
                Self::ds_add_inplace(&mut s2_hi, &mut s2_lo, p, err);
            }
            pn[i + 1]  = an;
            ps[i + 1]  = [s_hi,  s_lo];
            ps2[i + 1] = [s2_hi, s2_lo];
        }
        (ps, ps2, pn)
    }

    fn launch_batch_kernel(
        &self,
        d_ps: &DeviceBuffer<[f32; 2]>,
        d_ps2: &DeviceBuffer<[f32; 2]>,
        d_pn: &DeviceBuffer<i32>,
        d_periods: &DeviceBuffer<i32>,
        d_devups: &DeviceBuffer<f32>,
        d_devdns: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_up: &mut DeviceBuffer<f32>,
        d_mid: &mut DeviceBuffer<f32>,
        d_lo: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBollingerError> {
        let func = self
            .module
            .get_function("bollinger_bands_sma_prefix_f32")
            .map_err(|_| CudaBollingerError::MissingKernelSymbol { name: "bollinger_bands_sma_prefix_f32" })?;

        let block_x: u32 = 256;
        let grid_x = self.grid_x_for_len(len, block_x);
        let block: BlockSize = (block_x, 1, 1).into();

        const MAX_GRID_Y: usize = 65_535;
        let mut start = 0usize;
        while start < n_combos {
            let chunk = (n_combos - start).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            unsafe {
                
                let mut p_data: u64 = 0;
                let mut p_ps = d_ps.as_device_ptr().as_raw();
                let mut p_ps2 = d_ps2.as_device_ptr().as_raw();
                let mut p_pn = d_pn.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut p_per = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                let mut p_up = d_devups
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * std::mem::size_of::<f32>()) as u64);
                let mut p_dn = d_devdns
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * std::mem::size_of::<f32>()) as u64);
                let mut n_i = chunk as i32;
                let mut p_o_up = d_up
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                let mut p_o_mid = d_mid
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                let mut p_o_lo = d_lo
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut p_data as *mut _ as *mut c_void,
                    &mut p_ps as *mut _ as *mut c_void,
                    &mut p_ps2 as *mut _ as *mut c_void,
                    &mut p_pn as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut p_per as *mut _ as *mut c_void,
                    &mut p_up as *mut _ as *mut c_void,
                    &mut p_dn as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut p_o_up as *mut _ as *mut c_void,
                    &mut p_o_mid as *mut _ as *mut c_void,
                    &mut p_o_lo as *mut _ as *mut c_void,
                ];
                Self::validate_launch(grid, block)?;
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            start += chunk;
        }
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[BbCombo],
        first_valid: usize,
    ) -> Result<(DeviceArrayF32Bb, DeviceArrayF32Bb, DeviceArrayF32Bb), CudaBollingerError> {
        let len = data_f32.len();
        let (ps, ps2, pn) = Self::build_prefixes(data_f32);
        let d_ps:  DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps)?;
        let d_ps2: DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps2)?;
        let d_pn:  DeviceBuffer<i32>      = DeviceBuffer::from_slice(&pn)?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period as i32).collect();
        let devups: Vec<f32> = combos.iter().map(|c| c.devup).collect();
        let devdns: Vec<f32> = combos.iter().map(|c| c.devdn).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_devups = DeviceBuffer::from_slice(&devups)?;
        let d_devdns = DeviceBuffer::from_slice(&devdns)?;
        let devups: Vec<f32> = combos.iter().map(|c| c.devup).collect();
        let devdns: Vec<f32> = combos.iter().map(|c| c.devdn).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_devups = DeviceBuffer::from_slice(&devups)?;
        let d_devdns = DeviceBuffer::from_slice(&devdns)?;

        let elems = combos.len().checked_mul(len).ok_or(CudaBollingerError::InvalidInput("output elems overflow".into()))?;
        let _ = Self::will_fit(elems.checked_mul(std::mem::size_of::<f32>()).ok_or(CudaBollingerError::InvalidInput("bytes overflow".into()))?.checked_mul(3).ok_or(CudaBollingerError::InvalidInput("bytes overflow".into()))?, 32 * 1024 * 1024)?;
        let mut d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_mid: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_lo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        self.launch_batch_kernel(
            &d_ps, &d_ps2, &d_pn, &d_periods, &d_devups, &d_devdns,
            len, first_valid, combos.len(), &mut d_up, &mut d_mid, &mut d_lo,
        )?;
        self.stream.synchronize()?;

        let ctx = self.context_arc();
        let dev = self.device_id();
        Ok((
            DeviceArrayF32Bb { buf: d_up,  rows: combos.len(), cols: len, ctx: ctx.clone(), device_id: dev },
            DeviceArrayF32Bb { buf: d_mid, rows: combos.len(), cols: len, ctx: ctx.clone(), device_id: dev },
            DeviceArrayF32Bb { buf: d_lo,  rows: combos.len(), cols: len, ctx, device_id: dev },
        ))
    }

    pub fn bollinger_bands_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &BollingerBandsBatchRange,
    ) -> Result<(DeviceArrayF32Bb, DeviceArrayF32Bb, DeviceArrayF32Bb), CudaBollingerError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        
        let item_f32 = std::mem::size_of::<f32>();
        let item_i32 = std::mem::size_of::<i32>();
        let prefix_each = 2 * std::mem::size_of::<[f32; 2]>() + item_i32;
        let prefix_bytes = (len + 1).checked_mul(prefix_each).ok_or(CudaBollingerError::InvalidInput("prefix bytes overflow".into()))?;
        let out_elems = combos.len().checked_mul(len).ok_or(CudaBollingerError::InvalidInput("output elems overflow".into()))?;
        let out_bytes = out_elems.checked_mul(3 * item_f32).ok_or(CudaBollingerError::InvalidInput("output bytes overflow".into()))?;
        let required = prefix_bytes.checked_add(out_bytes).ok_or(CudaBollingerError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;
        self.run_batch_kernel(data_f32, &combos, first_valid)
    }

    
    pub fn bollinger_bands_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        devup: f32,
        devdn: f32,
    ) -> Result<(DeviceArrayF32Bb, DeviceArrayF32Bb, DeviceArrayF32Bb), CudaBollingerError> {
        if cols == 0 || rows == 0 {
            return Err(CudaBollingerError::InvalidInput(
                "cols or rows is zero".into(),
            ));
        }
        let elems_tm = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaBollingerError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != elems_tm {
            return Err(CudaBollingerError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                elems_tm
            )));
        }
        if period == 0 || period > rows {
            return Err(CudaBollingerError::InvalidInput("invalid period".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaBollingerError::InvalidInput("invalid period".into()));
        }

        
        let rows_p1 = rows
            .checked_add(1)
            .ok_or_else(|| CudaBollingerError::InvalidInput("rows+1 overflow".into()))?;
        let prefix_elems = rows_p1
            .checked_mul(cols)
            .ok_or_else(|| CudaBollingerError::InvalidInput("(rows+1)*cols overflow".into()))?;
        let mut ps  = vec![[0.0f32; 2]; prefix_elems];
        let mut ps2 = vec![[0.0f32; 2]; prefix_elems];
        let mut pn  = vec![0i32;        prefix_elems];
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let (mut s_hi,  mut s_lo)  = (0.0f32, 0.0f32);
            let (mut s2_hi, mut s2_lo) = (0.0f32, 0.0f32);
            let mut an=0i32; let mut fv: Option<usize>=None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if v.is_nan() {
                    an += 1;
                } else {
                    Self::ds_add_inplace(&mut s_hi, &mut s_lo, v, 0.0);
                    let p = v * v; let err = v.mul_add(v, -p);
                    Self::ds_add_inplace(&mut s2_hi, &mut s2_lo, p, err);
                    fv.get_or_insert(t);
                }
                let idx = (t + 1) * cols + s;
                ps[idx]=[s_hi, s_lo]; ps2[idx]=[s2_hi, s2_lo]; pn[idx]=an;
            }
            let fv = fv
                .ok_or_else(|| CudaBollingerError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < period {
                return Err(CudaBollingerError::InvalidInput(format!(
                    "series {} not enough valid data (needed {}, valid {})",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }

        let d_ps:  DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps)?;
        let d_ps2: DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps2)?;
        let d_pn  = DeviceBuffer::from_slice(&pn)?;
        let d_fv  = DeviceBuffer::from_slice(&first_valids)?;

        let item_f32 = std::mem::size_of::<f32>();
        let item_i32 = std::mem::size_of::<i32>();
        let prefix_each = 2 * std::mem::size_of::<[f32; 2]>() + item_i32;
        let prefix_bytes = rows_p1
            .checked_mul(cols)
            .and_then(|e| e.checked_mul(prefix_each))
            .ok_or_else(|| CudaBollingerError::InvalidInput("prefix bytes overflow".into()))?;
        let out_bytes = elems_tm
            .checked_mul(3 * item_f32)
            .ok_or_else(|| CudaBollingerError::InvalidInput("output bytes overflow".into()))?;
        let required = prefix_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaBollingerError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let mut d_up_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems_tm) }?;
        let mut d_md_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems_tm) }?;
        let mut d_lo_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems_tm) }?;

        
        let func = self
            .module
            .get_function("bollinger_bands_many_series_one_param_f32")
            .map_err(|_| CudaBollingerError::MissingKernelSymbol { name: "bollinger_bands_many_series_one_param_f32" })?;
        let block_x: u32 = 256;
        let grid_x = self.grid_x_for_len(rows, block_x);
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_ps = d_ps.as_device_ptr().as_raw();
            let mut p_ps = d_ps.as_device_ptr().as_raw();
            let mut p_ps2 = d_ps2.as_device_ptr().as_raw();
            let mut p_pn = d_pn.as_device_ptr().as_raw();
            let mut p_pn = d_pn.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut devup_f = devup as f32;
            let mut devdn_f = devdn as f32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_up = d_up_tm.as_device_ptr().as_raw();
            let mut p_md = d_md_tm.as_device_ptr().as_raw();
            let mut p_lo = d_lo_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ps as *mut _ as *mut c_void,
                &mut p_ps2 as *mut _ as *mut c_void,
                &mut p_pn as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut devup_f as *mut _ as *mut c_void,
                &mut devdn_f as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_up as *mut _ as *mut c_void,
                &mut p_md as *mut _ as *mut c_void,
                &mut p_lo as *mut _ as *mut c_void,
            ];
            Self::validate_launch(grid, block)?;
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        self.stream.synchronize()?;

        let ctx = self.context_arc();
        let dev = self.device_id();
        Ok((
            DeviceArrayF32Bb { buf: d_up_tm, rows, cols, ctx: ctx.clone(), device_id: dev },
            DeviceArrayF32Bb { buf: d_md_tm, rows, cols, ctx: ctx.clone(), device_id: dev },
            DeviceArrayF32Bb { buf: d_lo_tm, rows, cols, ctx, device_id: dev },
        ))
    }

    pub fn synchronize(&self) -> Result<(), CudaBollingerError> { self.stream.synchronize().map_err(Into::into) }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 128;
    const MANY_SERIES_ROWS: usize = 8_192;

    fn bytes_one_series_many_params() -> usize {
        let prefix = (ONE_SERIES_LEN + 1)
            * (2 * std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<i32>());
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        prefix + out_bytes + 64 * 1024 * 1024
    }

    fn bytes_many_series_one_param() -> usize {
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let rows_p1 = rows + 1;
        let prefix_each =
            2 * std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<i32>();
        let prefix_bytes = rows_p1 * cols * prefix_each;
        let fv_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = 3 * cols * rows * std::mem::size_of::<f32>();
        prefix_bytes + fv_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BbBatchState {
        cuda: CudaBollingerBands,
        d_ps: DeviceBuffer<[f32; 2]>,
        d_ps2: DeviceBuffer<[f32; 2]>,
        d_pn: DeviceBuffer<i32>,
        d_periods: DeviceBuffer<i32>,
        d_devups: DeviceBuffer<f32>,
        d_devdns: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_up: DeviceBuffer<f32>,
        d_mid: DeviceBuffer<f32>,
        d_lo: DeviceBuffer<f32>,
    }

    impl CudaBenchState for BbBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_ps,
                    &self.d_ps2,
                    &self.d_pn,
                    &self.d_periods,
                    &self.d_devups,
                    &self.d_devdns,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_up,
                    &mut self.d_mid,
                    &mut self.d_lo,
                )
                .unwrap();
            self.cuda.synchronize().unwrap();
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaBollingerBands::new(0).expect("cuda bb");
        let data = gen_series(ONE_SERIES_LEN);
        
        let sweep = BollingerBandsBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            devup: (2.0, 2.0, 0.0),
            devdn: (2.0, 2.0, 0.0),
            matype: ("sma".to_string(), "sma".to_string(), 0),
            devtype: (0, 0, 0),
        };

        let (combos, first_valid, len) =
            CudaBollingerBands::prepare_batch_inputs(&data, &sweep).expect("prepare batch");
        let (ps, ps2, pn) = CudaBollingerBands::build_prefixes(&data);

        let periods: Vec<i32> = combos.iter().map(|c| c.period as i32).collect();
        let devups: Vec<f32> = combos.iter().map(|c| c.devup).collect();
        let devdns: Vec<f32> = combos.iter().map(|c| c.devdn).collect();

        let d_ps = DeviceBuffer::from_slice(&ps).expect("ps H2D");
        let d_ps2 = DeviceBuffer::from_slice(&ps2).expect("ps2 H2D");
        let d_pn = DeviceBuffer::from_slice(&pn).expect("pn H2D");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("periods H2D");
        let d_devups = DeviceBuffer::from_slice(&devups).expect("devups H2D");
        let d_devdns = DeviceBuffer::from_slice(&devdns).expect("devdns H2D");

        let elems = combos.len() * len;
        let d_up = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("out up");
        let d_mid = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("out mid");
        let d_lo = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("out lo");

        Box::new(BbBatchState {
            cuda,
            d_ps,
            d_ps2,
            d_pn,
            d_periods,
            d_devups,
            d_devdns,
            len,
            first_valid,
            n_combos: combos.len(),
            d_up,
            d_mid,
            d_lo,
        })
    }

    struct BbManySeriesState {
        cuda: CudaBollingerBands,
        d_ps: DeviceBuffer<[f32; 2]>,
        d_ps2: DeviceBuffer<[f32; 2]>,
        d_pn: DeviceBuffer<i32>,
        d_fv: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        devup: f32,
        devdn: f32,
        d_up_tm: DeviceBuffer<f32>,
        d_mid_tm: DeviceBuffer<f32>,
        d_lo_tm: DeviceBuffer<f32>,
    }
    
    impl CudaBenchState for BbManySeriesState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("bollinger_bands_many_series_one_param_f32")
                .expect("bollinger_bands_many_series_one_param_f32");

            let block_x: u32 = 256;
            let grid_x = self.cuda.grid_x_for_len(self.rows, block_x);
            let grid: GridSize = (grid_x.max(1), self.cols as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut p_ps = self.d_ps.as_device_ptr().as_raw();
                let mut p_ps2 = self.d_ps2.as_device_ptr().as_raw();
                let mut p_pn = self.d_pn.as_device_ptr().as_raw();
                let mut period_i = self.period as i32;
                let mut devup_f = self.devup as f32;
                let mut devdn_f = self.devdn as f32;
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut p_fv = self.d_fv.as_device_ptr().as_raw();
                let mut p_up = self.d_up_tm.as_device_ptr().as_raw();
                let mut p_md = self.d_mid_tm.as_device_ptr().as_raw();
                let mut p_lo = self.d_lo_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_ps as *mut _ as *mut c_void,
                    &mut p_ps2 as *mut _ as *mut c_void,
                    &mut p_pn as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut devup_f as *mut _ as *mut c_void,
                    &mut devdn_f as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut p_fv as *mut _ as *mut c_void,
                    &mut p_up as *mut _ as *mut c_void,
                    &mut p_md as *mut _ as *mut c_void,
                    &mut p_lo as *mut _ as *mut c_void,
                ];
                CudaBollingerBands::validate_launch(grid, block).expect("bb validate launch");
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, args)
                    .expect("bb many launch");
            }
            self.cuda.synchronize().unwrap();
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaBollingerBands::new(0).expect("cuda bb");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let period = 20usize;
        let devup = 2.0f32;
        let devdn = 2.0f32;
        let tm = gen_time_major_prices(cols, rows);

        let rows_p1 = rows + 1;
        let prefix_elems = rows_p1 * cols;
        let mut ps = vec![[0.0f32; 2]; prefix_elems];
        let mut ps2 = vec![[0.0f32; 2]; prefix_elems];
        let mut pn = vec![0i32; prefix_elems];
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let (mut s_hi, mut s_lo) = (0.0f32, 0.0f32);
            let (mut s2_hi, mut s2_lo) = (0.0f32, 0.0f32);
            let mut an = 0i32;
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let v = tm[t * cols + s];
                if v.is_nan() {
                    an += 1;
                } else {
                    CudaBollingerBands::ds_add_inplace(&mut s_hi, &mut s_lo, v, 0.0);
                    let p = v * v;
                    let err = v.mul_add(v, -p);
                    CudaBollingerBands::ds_add_inplace(&mut s2_hi, &mut s2_lo, p, err);
                    fv.get_or_insert(t);
                }
                let idx = (t + 1) * cols + s;
                ps[idx] = [s_hi, s_lo];
                ps2[idx] = [s2_hi, s2_lo];
                pn[idx] = an;
            }
            let fv = fv.unwrap_or(0);
            if rows - fv < period {
                panic!("bb many-series: series {s} has insufficient valid data");
            }
            first_valids[s] = fv as i32;
        }

        let d_ps: DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps).expect("bb d_ps");
        let d_ps2: DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps2).expect("bb d_ps2");
        let d_pn = DeviceBuffer::from_slice(&pn).expect("bb d_pn");
        let d_fv = DeviceBuffer::from_slice(&first_valids).expect("bb d_fv");

        let elems_tm = cols * rows;
        let d_up_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_tm) }.expect("bb d_up_tm");
        let d_mid_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_tm) }.expect("bb d_mid_tm");
        let d_lo_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_tm) }.expect("bb d_lo_tm");
        cuda.synchronize().expect("bb sync after prep");
        Box::new(BbManySeriesState {
            cuda,
            d_ps,
            d_ps2,
            d_pn,
            d_fv,
            cols,
            rows,
            period,
            devup,
            devdn,
            d_up_tm,
            d_mid_tm,
            d_lo_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "bollinger_bands",
                "one_series_many_params",
                "bollinger_bands_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "bollinger_bands",
                "many_series_one_param",
                "bollinger_bands_cuda_many_series_one_param",
                "128x8k",
                prep_many_series,
            )
            .with_inner_iters(3)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
