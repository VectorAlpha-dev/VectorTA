//! CUDA wrapper for Polynomial Regression Bands (PRB)
//!
//! Parity goals with ALMA-style wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/prb_kernel.ptx"))
//! - NON_BLOCKING stream
//! - Policy enums for batch and many-series kernels; simple defaults
//! - VRAM estimate + ~64MB headroom; chunk grid.y for batch <= 65_535
//! - Host precompute when beneficial:
//!     - Expand parameter grid (smooth/period/order/offset)
//!     - Optional SSF smoothing per unique smooth_period
//!     - Precompute A^{-1} of the fixed normal matrix (per row)
//!     - Precompute contiguous-valid counts to mirror warmup/NaN resets

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::prb::{PrbBatchRange, PrbParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaPrbError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
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
pub struct CudaPrbPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaPrbPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaPrb {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaPrbPolicy,
}

impl CudaPrb {
    pub fn new(device_id: usize) -> Result<Self, CudaPrbError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/prb_kernel.ptx"));
        
        
        let jit_opts = &[ModuleJitOption::DetermineTargetFromContext];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaPrbPolicy::default(),
        })
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaPrbPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaPrbPolicy {
        &self.policy
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
    pub fn synchronize(&self) -> Result<(), CudaPrbError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaPrbError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _total)) => {
                let need = required.saturating_add(headroom);
                if need > free {
                    Err(CudaPrbError::OutOfMemory {
                        required: need,
                        free,
                        headroom,
                    })
                } else {
                    Ok(())
                }
            }
            Err(_) => Ok(()),
        }
    }

    
    fn axis_usize(a: (usize, usize, usize)) -> Result<Vec<usize>, CudaPrbError> {
        let (s, e, st) = a;
        if st == 0 || s == e {
            return Ok(vec![s]);
        }
        let mut out = Vec::new();
        if s < e {
            let mut v = s;
            let step = st.max(1);
            while v <= e {
                out.push(v);
                let next = match v.checked_add(step) {
                    Some(n) if n != v => n,
                    _ => break,
                };
                v = next;
            }
        } else {
            let mut v = s as isize;
            let end_i = e as isize;
            let step = (st as isize).max(1);
            while v >= end_i {
                out.push(v as usize);
                v -= step;
            }
        }
        if out.is_empty() {
            return Err(CudaPrbError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                s, e, st
            )));
        }
        Ok(out)
    }
    fn axis_i32(a: (i32, i32, i32)) -> Result<Vec<i32>, CudaPrbError> {
        let (s, e, st) = a;
        if st == 0 || s == e {
            return Ok(vec![s]);
        }
        let mut out = Vec::new();
        if s < e {
            let mut x = s;
            let step = st.max(1);
            while x <= e {
                out.push(x);
                let next = match x.checked_add(step) {
                    Some(n) if n != x => n,
                    _ => break,
                };
                x = next;
            }
        } else {
            let mut x = s;
            let step = st.abs().max(1);
            while x >= e {
                out.push(x);
                let next = match x.checked_sub(step) {
                    Some(n) if n != x => n,
                    _ => break,
                };
                x = next;
            }
        }
        if out.is_empty() {
            return Err(CudaPrbError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                s, e, st
            )));
        }
        Ok(out)
    }
    fn expand_grid(range: &PrbBatchRange, smooth_flag: bool) -> Result<Vec<PrbParams>, CudaPrbError> {
        let sps = Self::axis_usize(range.smooth_period)?;
        let rps = Self::axis_usize(range.regression_period)?;
        let pos = Self::axis_usize(range.polynomial_order)?;
        let ros = Self::axis_i32(range.regression_offset)?;
        let cap = sps
            .len()
            .checked_mul(rps.len())
            .and_then(|x| x.checked_mul(pos.len()))
            .and_then(|x| x.checked_mul(ros.len()))
            .ok_or_else(|| CudaPrbError::InvalidInput("rows*cols overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &sp in &sps {
            for &rp in &rps {
                for &po in &pos {
                    for &ro in &ros {
                        out.push(PrbParams {
                            smooth_data: Some(smooth_flag),
                            smooth_period: Some(sp),
                            regression_period: Some(rp),
                            polynomial_order: Some(po),
                            regression_offset: Some(ro),
                            ndev: Some(2.0),
                            equ_from: Some(0),
                        });
                    }
                }
            }
        }
        if out.is_empty() {
            return Err(CudaPrbError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        Ok(out)
    }

    fn ssf_filter_f32(data: &[f32], period: usize, first: usize) -> Vec<f32> {
        let len = data.len();
        let mut out = vec![f32::NAN; len];
        if len == 0 {
            return out;
        }
        let pi = core::f32::consts::PI;
        let omega = 2.0f32 * pi / (period as f32);
        let a = (-core::f32::consts::SQRT_2 * pi / (period as f32)).exp();
        let b = 2.0f32 * a * ((core::f32::consts::SQRT_2 / 2.0f32) * omega).cos();
        let c3 = -a * a;
        let c2 = b;
        let c1 = 1.0f32 - c2 - c3;
        let mut y1 = f32::NAN;
        let mut y2 = f32::NAN;
        for i in first..len {
            let x = data[i];
            
            let prev1 = if y1.is_nan() { x } else { y1 };
            let prev2 = if y2.is_nan() { prev1 } else { y2 };
            let y = c1 * x + c2 * prev1 + c3 * prev2;
            out[i] = y;
            y2 = y1;
            y1 = y;
        }
        out
    }

    fn contig_valid(series: &[f32]) -> Vec<i32> {
        let mut v = vec![0i32; series.len()];
        let mut c: i32 = 0;
        for (i, &x) in series.iter().enumerate() {
            if x.is_nan() {
                c = 0;
            } else {
                c += 1;
            }
            v[i] = c;
        }
        v
    }

    fn build_a_inv(n: usize, k: usize) -> Vec<f32> {
        
        let m = k + 1;
        let max_m = 8usize;
        let mut a = vec![0.0f64; m * m];
        
        let mut sx = vec![0.0f64; 2 * k + 1];
        for j in 1..=n {
            let jf = j as f64;
            let mut p = 1.0f64;
            sx[0] += 1.0;
            for t in 1..=2 * k {
                p *= jf;
                sx[t] += p;
            }
        }
        for i in 0..m {
            for j in 0..m {
                a[i * m + j] = sx[i + j];
            }
        }
        
        let mut aug = vec![0.0f64; m * 2 * m];
        for r in 0..m {
            for c in 0..m {
                aug[r * (2 * m) + c] = a[r * m + c];
            }
            aug[r * (2 * m) + (m + r)] = 1.0;
        }
        for i in 0..m {
            
            let mut piv = i;
            let mut best = aug[i * (2 * m) + i].abs();
            for r in (i + 1)..m {
                let val = aug[r * (2 * m) + i].abs();
                if val > best {
                    best = val;
                    piv = r;
                }
            }
            if piv != i {
                for c in 0..(2 * m) {
                    aug.swap(i * (2 * m) + c, piv * (2 * m) + c);
                }
            }
            let diag = aug[i * (2 * m) + i];
            let invd = 1.0f64 / diag;
            for c in 0..(2 * m) {
                aug[i * (2 * m) + c] *= invd;
            }
            for r in 0..m {
                if r == i {
                    continue;
                }
                let f = aug[r * (2 * m) + i];
                if f == 0.0 {
                    continue;
                }
                for c in 0..(2 * m) {
                    aug[r * (2 * m) + c] -= f * aug[i * (2 * m) + c];
                }
            }
        }
        let mut inv = vec![0.0f32; max_m * max_m]; 
        for r in 0..m {
            for c in 0..m {
                inv[r * max_m + c] = aug[r * (2 * m) + (m + c)] as f32;
            }
        }
        inv
    }

    pub fn prb_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PrbBatchRange,
        smooth_data: bool,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaPrbError> {
        if data_f32.is_empty() {
            return Err(CudaPrbError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaPrbError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep, smooth_data)?;

        
        let mut uniq_nk: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut uniq_sp: BTreeSet<usize> = BTreeSet::new();
        for c in &combos {
            let n = c.regression_period.unwrap();
            let k = c.polynomial_order.unwrap();
            if n == 0 || n > len {
                return Err(CudaPrbError::InvalidInput(
                    "invalid regression_period".into(),
                ));
            }
            uniq_nk.insert((n, k));
            if smooth_data {
                uniq_sp.insert(c.smooth_period.unwrap_or(10));
            }
        }
        
        let mut a_inv_map: BTreeMap<(usize, usize), Vec<f32>> = BTreeMap::new();
        for &(n, k) in &uniq_nk {
            a_inv_map.insert((n, k), Self::build_a_inv(n, k));
        }
        
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        if smooth_data {
            for (idx, c) in combos.iter().enumerate() {
                let sp = c.smooth_period.unwrap_or(10);
                groups.entry(sp).or_default().push(idx);
            }
        } else {
            groups.insert(0, (0..combos.len()).collect());
        }

        
        let total_elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaPrbError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_inputs = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaPrbError::InvalidInput("input bytes overflow".into()))?;
        let bytes_out = 3usize
            .checked_mul(total_elems)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaPrbError::InvalidInput("output bytes overflow".into()))?;
        let bytes_params = combos
            .len()
            .checked_mul(3 * std::mem::size_of::<i32>() + 64)
            .ok_or_else(|| CudaPrbError::InvalidInput("params bytes overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_out)
            .and_then(|x| x.checked_add(bytes_params))
            .and_then(|x| x.checked_add(64 * 1024))
            .ok_or_else(|| CudaPrbError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let mut d_main: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_lo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;

        
        let mut d_src: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;
        let mut d_contig: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(len) }?;

        
        
        let mut h_src_keepalive: Vec<LockedBuffer<f32>> = Vec::new();
        let mut h_contig_keepalive: Vec<LockedBuffer<i32>> = Vec::new();

        
        
        let mut keep_periods: Vec<DeviceBuffer<i32>> = Vec::new();
        let mut keep_orders: Vec<DeviceBuffer<i32>> = Vec::new();
        let mut keep_offsets: Vec<DeviceBuffer<i32>> = Vec::new();
        let mut keep_ainv: Vec<DeviceBuffer<f32>> = Vec::new();
        let mut keep_rowmap: Vec<DeviceBuffer<i32>> = Vec::new();
        for (sp, rows_idx) in groups.iter() {
            
            let source: Vec<f32> = if smooth_data {
                Self::ssf_filter_f32(data_f32, *sp, first_valid)
            } else {
                data_f32.to_vec()
            };
            let has_nan_after_first = source[first_valid..].iter().any(|v| v.is_nan());
            let contig = if smooth_data {
                Self::contig_valid(&source)
            } else {
                
                Self::contig_valid(data_f32)
            };

            
            let h_src = LockedBuffer::from_slice(&source)?;
            let h_contig = LockedBuffer::from_slice(&contig)?;
            unsafe {
                d_src
                    .async_copy_from(h_src.as_slice(), &self.stream)
                    ?;
                d_contig
                    .async_copy_from(h_contig.as_slice(), &self.stream)
                    ?;
            }
            
            h_src_keepalive.push(h_src);
            h_contig_keepalive.push(h_contig);

            
            let mut periods: Vec<i32> = Vec::with_capacity(rows_idx.len());
            let mut orders: Vec<i32> = Vec::with_capacity(rows_idx.len());
            let mut offsets: Vec<i32> = Vec::with_capacity(rows_idx.len());
            let mut a_invs: Vec<f32> = Vec::with_capacity(rows_idx.len() * 64);
            let mut row_map: Vec<i32> = Vec::with_capacity(rows_idx.len());
            let max_m: i32 = 8;
            let ainv_stride_elems = (max_m as usize) * (max_m as usize);
            for &row in rows_idx {
                let c = &combos[row];
                let n = c.regression_period.unwrap();
                let k = c.polynomial_order.unwrap();
                let off = c.regression_offset.unwrap_or(0);
                periods.push(n as i32);
                orders.push(k as i32);
                offsets.push(off as i32);
                let ainv = a_inv_map.get(&(n, k)).expect("missing ainv");
                a_invs.extend_from_slice(ainv);
                row_map.push(row as i32);
            }
            let d_periods = DeviceBuffer::from_slice(&periods)?;
            let d_orders = DeviceBuffer::from_slice(&orders)?;
            let d_offsets = DeviceBuffer::from_slice(&offsets)?;
            let d_ainv = DeviceBuffer::from_slice(&a_invs)?;
            let d_rowmap = DeviceBuffer::from_slice(&row_map)?;

            
            const PRB_BATCH_CHUNK_LEN: usize = 4096;
            let use_chunked = matches!(self.policy.batch, BatchKernelPolicy::Auto) && !has_nan_after_first;
            let (func, block_x, grid_x): (Function, u32, u32) = if use_chunked {
                let f = self.module.get_function("prb_batch_chunked_f32").map_err(|_| {
                    CudaPrbError::MissingKernelSymbol {
                        name: "prb_batch_chunked_f32",
                    }
                })?;
                let bx = 32u32;
                let chunks = len.div_ceil(PRB_BATCH_CHUNK_LEN);
                let gx = chunks.div_ceil(bx as usize) as u32;
                (f, bx, gx.max(1))
            } else {
                if let BatchKernelPolicy::Plain { block_x } = self.policy.batch {
                    if block_x != 1 {
                        return Err(CudaPrbError::InvalidPolicy(
                            "prb_batch_f32 requires block_x=1 (serial row kernel)",
                        ));
                    }
                }
                let f = self.module.get_function("prb_batch_f32").map_err(|_| {
                    CudaPrbError::MissingKernelSymbol {
                        name: "prb_batch_f32",
                    }
                })?;
                (f, 1u32, 1u32)
            };
            let block: BlockSize = (block_x, 1, 1).into();
            let dev = Device::get_device(self.device_id)?;
            let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
            if block_x > max_bx {
                return Err(CudaPrbError::LaunchConfigTooLarge {
                    gx: grid_x,
                    gy: 1,
                    gz: 1,
                    bx: block_x,
                    by: 1,
                    bz: 1,
                });
            }

            const MAX_GRID_Y: usize = 65_535; 
            let mut start = 0usize;
            while start < rows_idx.len() {
                let chunk = (rows_idx.len() - start).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x, chunk as u32, 1).into();
                unsafe {
                    let mut p_src = d_src.as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut first_i = first_valid as i32;
                    let mut p_per = d_periods
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut p_ord = d_orders
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut p_off = d_offsets
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut combos_i = chunk as i32;
                    let mut max_m_i = max_m;
                    let mut p_ainv = d_ainv
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * ainv_stride_elems * std::mem::size_of::<f32>()) as u64);
                    let mut stride_i = ainv_stride_elems as i32;
                    let mut p_contig = d_contig.as_device_ptr().as_raw();
                    let mut ndev_f = 2.0f32; 
                    let mut p_rowmap = d_rowmap
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut p_out_m = d_main.as_device_ptr().as_raw();
                    let mut p_out_u = d_up.as_device_ptr().as_raw();
                    let mut p_out_l = d_lo.as_device_ptr().as_raw();
                    let mut args: [*mut c_void; 16] = [
                        &mut p_src as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut p_per as *mut _ as *mut c_void,
                        &mut p_ord as *mut _ as *mut c_void,
                        &mut p_off as *mut _ as *mut c_void,
                        &mut combos_i as *mut _ as *mut c_void,
                        &mut max_m_i as *mut _ as *mut c_void,
                        &mut p_ainv as *mut _ as *mut c_void,
                        &mut stride_i as *mut _ as *mut c_void,
                        &mut p_contig as *mut _ as *mut c_void,
                        &mut ndev_f as *mut _ as *mut c_void,
                        &mut p_rowmap as *mut _ as *mut c_void,
                        &mut p_out_m as *mut _ as *mut c_void,
                        &mut p_out_u as *mut _ as *mut c_void,
                        &mut p_out_l as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, &mut args)?;
                }
                start += chunk;
            }
            
            keep_periods.push(d_periods);
            keep_orders.push(d_orders);
            keep_offsets.push(d_offsets);
            keep_ainv.push(d_ainv);
            keep_rowmap.push(d_rowmap);
            
        }

        
        self.stream.synchronize()?;
        drop(h_src_keepalive);
        drop(h_contig_keepalive);

        Ok((
            DeviceArrayF32 {
                buf: d_main,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_up,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_lo,
                rows: combos.len(),
                cols: len,
            },
        ))
    }

    pub fn prb_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PrbParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaPrbError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPrbError::InvalidInput("empty grid".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaPrbError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != elems {
            return Err(CudaPrbError::InvalidInput("data length mismatch".into()));
        }
        let n = params.regression_period.unwrap_or(100);
        let k = params.polynomial_order.unwrap_or(2);
        let off = params.regression_offset.unwrap_or(0);
        if n == 0 || n > rows {
            return Err(CudaPrbError::InvalidInput(
                "invalid regression_period".into(),
            ));
        }
        
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            firsts[s] = fv;
            if fv >= 0 {
                if (rows - fv as usize) < n {
                    return Err(CudaPrbError::InvalidInput("not enough valid data".into()));
                }
            }
        }
        
        let smooth = params.smooth_data.unwrap_or(true);
        let sp = params.smooth_period.unwrap_or(10);
        let mut sm_tm = vec![f32::NAN; elems];
        if smooth {
            for s in 0..cols {
                let fv = firsts[s];
                if fv < 0 {
                    continue;
                }
                
                let mut col = vec![f32::NAN; rows];
                for t in 0..rows {
                    col[t] = data_tm_f32[t * cols + s];
                }
                let sm = Self::ssf_filter_f32(&col, sp, fv as usize);
                for t in 0..rows {
                    sm_tm[t * cols + s] = sm[t];
                }
            }
        } else {
            sm_tm.copy_from_slice(data_tm_f32);
        }
        let contig_tm = {
            let mut v = vec![0i32; elems];
            for s in 0..cols {
                let mut c = 0i32;
                for t in 0..rows {
                    let y = sm_tm[t * cols + s];
                    if y.is_nan() {
                        c = 0;
                    } else {
                        c += 1;
                    }
                    v[t * cols + s] = c;
                }
            }
            v
        };

        let ainv = Self::build_a_inv(n, k);
        
        let mut d_prices_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_contig_tm: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;
        let h_prices_tm = LockedBuffer::from_slice(&sm_tm)?;
        let h_contig_tm = LockedBuffer::from_slice(&contig_tm)?;
        unsafe {
            d_prices_tm
                .async_copy_from(h_prices_tm.as_slice(), &self.stream)
                ?;
            d_contig_tm
                .async_copy_from(h_contig_tm.as_slice(), &self.stream)
                ?;
        }
        let d_ainv = DeviceBuffer::from_slice(&ainv)?;
        let mut d_m: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_u: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_l: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;

        let func = self
            .module
            .get_function("prb_many_series_one_param_f32")
            .map_err(|_| CudaPrbError::MissingKernelSymbol {
                name: "prb_many_series_one_param_f32",
            })?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        if block_x > max_bx || grid_x > max_gx {
            return Err(CudaPrbError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let d_firsts = DeviceBuffer::from_slice(&firsts)?;
        unsafe {
            let mut p_tm = d_prices_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = n as i32;
            let mut order_i = k as i32;
            let mut off_i = off as i32;
            let mut max_m_i = 8i32;
            let mut stride_i = (8 * 8) as i32;
            let mut p_ainv = d_ainv.as_device_ptr().as_raw();
            let mut p_contig_tm = d_contig_tm.as_device_ptr().as_raw();
            let mut p_firsts = d_firsts.as_device_ptr().as_raw();
            let mut ndev_f = params.ndev.unwrap_or(2.0) as f32;
            let mut p_m = d_m.as_device_ptr().as_raw();
            let mut p_u = d_u.as_device_ptr().as_raw();
            let mut p_l = d_l.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 15] = [
                &mut p_tm as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut order_i as *mut _ as *mut c_void,
                &mut off_i as *mut _ as *mut c_void,
                &mut max_m_i as *mut _ as *mut c_void,
                &mut p_ainv as *mut _ as *mut c_void,
                &mut stride_i as *mut _ as *mut c_void,
                &mut p_contig_tm as *mut _ as *mut c_void,
                &mut p_firsts as *mut _ as *mut c_void,
                &mut ndev_f as *mut _ as *mut c_void,
                &mut p_m as *mut _ as *mut c_void,
                &mut p_u as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        self.stream.synchronize()?;
        drop(h_prices_tm);
        drop(h_contig_tm);
        Ok((
            DeviceArrayF32 {
                buf: d_m,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_u,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_l,
                rows,
                cols,
            },
        ))
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = 3 * elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaPrb,
        d_src: DeviceBuffer<f32>,
        d_contig: DeviceBuffer<i32>,
        d_periods: DeviceBuffer<i32>,
        d_orders: DeviceBuffer<i32>,
        d_offsets: DeviceBuffer<i32>,
        d_ainv: DeviceBuffer<f32>,
        d_rowmap: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        grid: GridSize,
        block: BlockSize,
        use_chunked: bool,
        d_main: DeviceBuffer<f32>,
        d_up: DeviceBuffer<f32>,
        d_lo: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let func_name = if self.use_chunked {
                "prb_batch_chunked_f32"
            } else {
                "prb_batch_f32"
            };
            let func = self
                .cuda
                .module
                .get_function(func_name)
                .expect("prb batch kernel");

            unsafe {
                let mut p_src = self.d_src.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut first_i = self.first_valid as i32;
                let mut p_per = self.d_periods.as_device_ptr().as_raw();
                let mut p_ord = self.d_orders.as_device_ptr().as_raw();
                let mut p_off = self.d_offsets.as_device_ptr().as_raw();
                let mut combos_i = self.rows as i32;
                let mut max_m_i = 8i32;
                let mut p_ainv = self.d_ainv.as_device_ptr().as_raw();
                let mut stride_i = (8 * 8) as i32;
                let mut p_contig = self.d_contig.as_device_ptr().as_raw();
                let mut ndev_f = 2.0f32;
                let mut p_rowmap = self.d_rowmap.as_device_ptr().as_raw();
                let mut p_out_m = self.d_main.as_device_ptr().as_raw();
                let mut p_out_u = self.d_up.as_device_ptr().as_raw();
                let mut p_out_l = self.d_lo.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 16] = [
                    &mut p_src as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut p_per as *mut _ as *mut c_void,
                    &mut p_ord as *mut _ as *mut c_void,
                    &mut p_off as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut max_m_i as *mut _ as *mut c_void,
                    &mut p_ainv as *mut _ as *mut c_void,
                    &mut stride_i as *mut _ as *mut c_void,
                    &mut p_contig as *mut _ as *mut c_void,
                    &mut ndev_f as *mut _ as *mut c_void,
                    &mut p_rowmap as *mut _ as *mut c_void,
                    &mut p_out_m as *mut _ as *mut c_void,
                    &mut p_out_u as *mut _ as *mut c_void,
                    &mut p_out_l as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, &mut args)
                    .expect("prb batch launch");
            }
            self.cuda.stream.synchronize().expect("prb batch sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaPrb::new(0).expect("cuda");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = PrbBatchRange {
            smooth_period: (10, 10, 0),
            regression_period: (100, 100 + PARAM_SWEEP - 1, 1),
            polynomial_order: (2, 2, 0),
            regression_offset: (0, 0, 0),
        };

        let len = price.len();
        let first_valid = price.iter().position(|v| !v.is_nan()).unwrap_or(0);
        let combos = CudaPrb::expand_grid(&sweep, false).expect("prb expand_grid");
        let rows = combos.len();
        let contig = CudaPrb::contig_valid(&price);

        let mut periods: Vec<i32> = Vec::with_capacity(rows);
        let mut orders: Vec<i32> = Vec::with_capacity(rows);
        let mut offsets: Vec<i32> = Vec::with_capacity(rows);
        let mut a_invs: Vec<f32> = Vec::with_capacity(rows * 64);
        let mut row_map: Vec<i32> = Vec::with_capacity(rows);
        for (row, c) in combos.iter().enumerate() {
            let n = c.regression_period.unwrap();
            let k = c.polynomial_order.unwrap();
            let off = c.regression_offset.unwrap_or(0);
            periods.push(n as i32);
            orders.push(k as i32);
            offsets.push(off as i32);
            a_invs.extend_from_slice(&CudaPrb::build_a_inv(n, k));
            row_map.push(row as i32);
        }

        let use_chunked = price[first_valid..].iter().all(|v| !v.is_nan());
        let (block_x, grid_x) = if use_chunked {
            const PRB_BATCH_CHUNK_LEN: usize = 4096;
            let bx = 32u32;
            let chunks = len.div_ceil(PRB_BATCH_CHUNK_LEN);
            let gx = chunks.div_ceil(bx as usize) as u32;
            (bx, gx.max(1))
        } else {
            (1u32, 1u32)
        };

        let d_src = DeviceBuffer::from_slice(&price).expect("d_src");
        let d_contig = DeviceBuffer::from_slice(&contig).expect("d_contig");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_orders = DeviceBuffer::from_slice(&orders).expect("d_orders");
        let d_offsets = DeviceBuffer::from_slice(&offsets).expect("d_offsets");
        let d_ainv = DeviceBuffer::from_slice(&a_invs).expect("d_ainv");
        let d_rowmap = DeviceBuffer::from_slice(&row_map).expect("d_rowmap");

        let total_elems = rows.checked_mul(len).expect("rows*len overflow");
        let d_main: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }.expect("d_main");
        let d_up: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }.expect("d_up");
        let d_lo: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }.expect("d_lo");

        let grid: GridSize = (grid_x, (rows as u32).max(1), 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        cuda.stream.synchronize().expect("prb prep sync");

        Box::new(BatchState {
            cuda,
            d_src,
            d_contig,
            d_periods,
            d_orders,
            d_offsets,
            d_ainv,
            d_rowmap,
            len,
            first_valid,
            rows,
            grid,
            block,
            use_chunked,
            d_main,
            d_up,
            d_lo,
        })
    }

    struct ManyState {
        cuda: CudaPrb,
        d_prices_tm: DeviceBuffer<f32>,
        d_contig_tm: DeviceBuffer<i32>,
        d_firsts: DeviceBuffer<i32>,
        d_ainv: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: i32,
        order: i32,
        offset: i32,
        grid: GridSize,
        block: BlockSize,
        ndev: f32,
        d_m: DeviceBuffer<f32>,
        d_u: DeviceBuffer<f32>,
        d_l: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("prb_many_series_one_param_f32")
                .expect("prb_many_series_one_param_f32");
            unsafe {
                let mut p_tm = self.d_prices_tm.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut period_i = self.period;
                let mut order_i = self.order;
                let mut off_i = self.offset;
                let mut max_m_i = 8i32;
                let mut stride_i = (8 * 8) as i32;
                let mut p_ainv = self.d_ainv.as_device_ptr().as_raw();
                let mut p_contig_tm = self.d_contig_tm.as_device_ptr().as_raw();
                let mut p_firsts = self.d_firsts.as_device_ptr().as_raw();
                let mut ndev_f = self.ndev;
                let mut p_m = self.d_m.as_device_ptr().as_raw();
                let mut p_u = self.d_u.as_device_ptr().as_raw();
                let mut p_l = self.d_l.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 15] = [
                    &mut p_tm as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut order_i as *mut _ as *mut c_void,
                    &mut off_i as *mut _ as *mut c_void,
                    &mut max_m_i as *mut _ as *mut c_void,
                    &mut p_ainv as *mut _ as *mut c_void,
                    &mut stride_i as *mut _ as *mut c_void,
                    &mut p_contig_tm as *mut _ as *mut c_void,
                    &mut p_firsts as *mut _ as *mut c_void,
                    &mut ndev_f as *mut _ as *mut c_void,
                    &mut p_m as *mut _ as *mut c_void,
                    &mut p_u as *mut _ as *mut c_void,
                    &mut p_l as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, &mut args)
                    .expect("prb many-series launch");
            }
            self.cuda.stream.synchronize().expect("prb many-series sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaPrb::new(0).expect("cuda");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = PrbParams {
            smooth_data: Some(false),
            smooth_period: Some(10),
            regression_period: Some(100),
            polynomial_order: Some(2),
            regression_offset: Some(0),
            ndev: Some(2.0),
            equ_from: Some(0),
        };
        let n = params.regression_period.unwrap_or(100);
        let k = params.polynomial_order.unwrap_or(2);
        let off = params.regression_offset.unwrap_or(0);
        let ndev = params.ndev.unwrap_or(2.0) as f32;

        
        let elems = cols.checked_mul(rows).expect("cols*rows overflow");
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            firsts[s] = fv;
        }
        let contig_tm = {
            let mut v = vec![0i32; elems];
            for s in 0..cols {
                let mut c = 0i32;
                for t in 0..rows {
                    let y = data_tm[t * cols + s];
                    if y.is_nan() {
                        c = 0;
                    } else {
                        c += 1;
                    }
                    v[t * cols + s] = c;
                }
            }
            v
        };
        let ainv = CudaPrb::build_a_inv(n, k);

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_contig_tm = DeviceBuffer::from_slice(&contig_tm).expect("d_contig_tm");
        let d_firsts = DeviceBuffer::from_slice(&firsts).expect("d_firsts");
        let d_ainv = DeviceBuffer::from_slice(&ainv).expect("d_ainv");

        let d_m: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_m");
        let d_u: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_u");
        let d_l: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_l");

        let block_x: u32 = match cuda.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        cuda.stream.synchronize().expect("prb prep sync");

        Box::new(ManyState {
            cuda,
            d_prices_tm,
            d_contig_tm,
            d_firsts,
            d_ainv,
            cols,
            rows,
            period: n as i32,
            order: k as i32,
            offset: off as i32,
            grid,
            block,
            ndev,
            d_m,
            d_u,
            d_l,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "prb",
                "one_series_many_params",
                "prb_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "prb",
                "many_series_one_param",
                "prb_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
