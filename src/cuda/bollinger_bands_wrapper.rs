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
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashSet;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaBollingerError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaBollingerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaBollingerError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaBollingerError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}
impl std::error::Error for CudaBollingerError {}

#[derive(Clone, Debug)]
struct BbCombo {
    period: usize,
    devup: f32,
    devdn: f32,
}

pub struct CudaBollingerBands {
    module: Module,
    stream: Stream,
    _context: Context,
    sm_count: u32,
}

impl CudaBollingerBands {
    pub fn new(device_id: usize) -> Result<Self, CudaBollingerError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        // Query SM count for launch sizing heuristics
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map(|v| v as u32)
            .unwrap_or(64);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/bollinger_bands_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O3),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        Ok(Self { module, stream, _context: context, sm_count })
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { required_bytes.saturating_add(headroom_bytes) <= free } else { true }
    }

    #[inline(always)]
    fn grid_x_for_len(&self, len: usize, block_x: u32) -> u32 {
        let need = ((len as u32) + block_x - 1) / block_x;
        let cap  = (self.sm_count.saturating_mul(4)).max(1);
        need.min(cap)
    }

    fn expand_combos(range: &BollingerBandsBatchRange) -> Vec<(usize, f64, f64, String, usize)> {
        
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                vec![s]
            } else {
                (s..=e).step_by(st).collect()
            }
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                vec![s]
            } else {
                let mut v = Vec::new();
                let mut x = s;
                while x <= e + 1e-12 {
                    v.push(x);
                    x += st;
                }
                v
            }
        }
        fn axis_str((s, e, _): (String, String, usize)) -> Vec<String> {
            if s == e {
                vec![s]
            } else {
                vec![s, e]
            }
        }
        let periods = axis_usize(range.period);
        let devups = axis_f64(range.devup);
        let devdns = axis_f64(range.devdn);
        let devups = axis_f64(range.devup);
        let devdns = axis_f64(range.devdn);
        let matypes = axis_str(range.matype.clone());
        let devtypes = axis_usize(range.devtype);
        let mut out = Vec::with_capacity(
            periods.len() * devups.len() * devdns.len() * matypes.len() * devtypes.len(),
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
        let devtypes = axis_usize(range.devtype);
        let mut out = Vec::with_capacity(
            periods.len() * devups.len() * devdns.len() * matypes.len() * devtypes.len(),
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
        out
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

        let raw = Self::expand_combos(sweep);
        if raw.is_empty() {
            return Err(CudaBollingerError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
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
            
            if ma.to_ascii_lowercase() != "sma" {
                unsupported_ma.insert(ma);
                continue;
            }
            combos.push(BbCombo {
                period: p,
                devup: u as f32,
                devdn: d as f32,
            });
            if ma.to_ascii_lowercase() != "sma" {
                unsupported_ma.insert(ma);
                continue;
            }
            combos.push(BbCombo {
                period: p,
                devup: u as f32,
                devdn: d as f32,
            });
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

    // -------- Host-side double-single prefix builders --------
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
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        let block_x: u32 = 256;
        let grid_x = self.grid_x_for_len(len, block_x);
        let block: BlockSize = (block_x, 1, 1).into();

        const MAX_GRID_Y: usize = 65_535;
        let mut start = 0usize;
        while start < n_combos {
            let chunk = (n_combos - start).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            unsafe {
                // Kernel ignores `data` pointer; pass null (0)
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
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
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
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaBollingerError> {
        let len = data_f32.len();
        let (ps, ps2, pn) = Self::build_prefixes(data_f32);
        let d_ps:  DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_ps2: DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps2).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_pn:  DeviceBuffer<i32>      = DeviceBuffer::from_slice(&pn).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period as i32).collect();
        let devups: Vec<f32> = combos.iter().map(|c| c.devup).collect();
        let devdns: Vec<f32> = combos.iter().map(|c| c.devdn).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_devups = DeviceBuffer::from_slice(&devups)
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_devdns = DeviceBuffer::from_slice(&devdns)
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let devups: Vec<f32> = combos.iter().map(|c| c.devup).collect();
        let devdns: Vec<f32> = combos.iter().map(|c| c.devdn).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_devups = DeviceBuffer::from_slice(&devups)
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_devdns = DeviceBuffer::from_slice(&devdns)
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let mut d_mid: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let mut d_lo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_ps, &d_ps2, &d_pn, &d_periods, &d_devups, &d_devdns,
            len, first_valid, combos.len(), &mut d_up, &mut d_mid, &mut d_lo,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 { buf: d_up,  rows: combos.len(), cols: len },
            DeviceArrayF32 { buf: d_mid, rows: combos.len(), cols: len },
            DeviceArrayF32 { buf: d_lo,  rows: combos.len(), cols: len },
        ))
    }

    pub fn bollinger_bands_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &BollingerBandsBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaBollingerError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;

        let prefix_bytes = (len + 1) * (2 * std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<i32>());
        let out_bytes = 3 * combos.len() * len * std::mem::size_of::<f32>();
        let required = prefix_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaBollingerError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }
        self.run_batch_kernel(data_f32, &combos, first_valid)
    }

    // ------------------- Many-series, one param (time-major) -------------------
    pub fn bollinger_bands_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        devup: f32,
        devdn: f32,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaBollingerError> {
        if cols == 0 || rows == 0 {
            return Err(CudaBollingerError::InvalidInput(
                "cols or rows is zero".into(),
            ));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaBollingerError::InvalidInput(
                "cols or rows is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaBollingerError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        if period == 0 || period > rows {
            return Err(CudaBollingerError::InvalidInput("invalid period".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaBollingerError::InvalidInput("invalid period".into()));
        }

        // Build time-major double-single prefixes (rows+1)×cols; track first_valid per series
        let mut ps  = vec![[0.0f32; 2]; (rows + 1) * cols];
        let mut ps2 = vec![[0.0f32; 2]; (rows + 1) * cols];
        let mut pn  = vec![0i32;        (rows + 1) * cols];
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

        let d_ps:  DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_ps2: DeviceBuffer<[f32; 2]> = DeviceBuffer::from_slice(&ps2).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_pn  = DeviceBuffer::from_slice(&pn).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let d_fv  = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        let mut d_up_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let mut d_md_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let mut d_lo_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let mut d_up_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let mut d_md_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        let mut d_lo_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        // Launch many-series kernel
        let func = self
            .module
            .get_function("bollinger_bands_many_series_one_param_f32")
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_up_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_md_tm,
                rows,
                cols,
            },
            DeviceArrayF32 { buf: d_lo_tm, rows, cols },
        ))
    }

    pub fn synchronize(&self) -> Result<(), CudaBollingerError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaBollingerError::Cuda(e.to_string()))?;
        Ok(())
    }
}

// ------------- Benches -------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let prefix = (ONE_SERIES_LEN + 1)
            * (2 * std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<i32>());
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        prefix + out_bytes + 64 * 1024 * 1024
    }

    struct BbBatchState {
        cuda: CudaBollingerBands,
        data: Vec<f32>,
        sweep: BollingerBandsBatchRange,
    }
    
    impl CudaBenchState for BbBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .bollinger_bands_batch_dev(&self.data, &self.sweep)
                .unwrap();
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaBollingerBands::new(0).expect("cuda bb");
        let data = gen_series(ONE_SERIES_LEN);
        // Sweep periods; devup=devdn=2.0; SMA; devtype=0
        let sweep = BollingerBandsBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            devup: (2.0, 2.0, 0.0),
            devdn: (2.0, 2.0, 0.0),
            matype: ("sma".to_string(), "sma".to_string(), 0),
            devtype: (0, 0, 0),
        };
        Box::new(BbBatchState { cuda, data, sweep })
    }

    struct BbManySeriesState {
        cuda: CudaBollingerBands,
        tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        devup: f32,
        devdn: f32,
    }
    
    impl CudaBenchState for BbManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .bollinger_bands_many_series_one_param_time_major_dev(
                    &self.tm,
                    self.cols,
                    self.rows,
                    self.period,
                    self.devup,
                    self.devdn,
                )
                .unwrap();
        }
    }

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaBollingerBands::new(0).expect("cuda bb");
        let cols = 250usize;
        let rows = 1_000_000usize;
        let period = 20usize;
        let devup = 2.0f32;
        let devdn = 2.0f32;
        let tm = gen_time_major_prices(cols, rows);
        Box::new(BbManySeriesState {
            cuda,
            tm,
            cols,
            rows,
            period,
            devup,
            devdn,
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
                "250x1m",
                prep_many_series,
            )
            .with_inner_iters(3),
        ]
    }
}
