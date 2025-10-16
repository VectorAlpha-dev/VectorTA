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
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaPrbError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaPrbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaPrbError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaPrbError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaPrbError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy { Auto, Plain { block_x: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy { Auto, OneD { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub struct CudaPrbPolicy { pub batch: BatchKernelPolicy, pub many_series: ManySeriesKernelPolicy }
impl Default for CudaPrbPolicy {
    fn default() -> Self { Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto } }
}

pub struct CudaPrb {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaPrbPolicy,
}

impl CudaPrb {
    pub fn new(device_id: usize) -> Result<Self, CudaPrbError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/prb_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        Ok(Self { module, stream, _context: context, policy: CudaPrbPolicy::default() })
    }

    #[inline] pub fn set_policy(&mut self, policy: CudaPrbPolicy) { self.policy = policy; }
    #[inline] pub fn policy(&self) -> &CudaPrbPolicy { &self.policy }
    #[inline] pub fn synchronize(&self) -> Result<(), CudaPrbError> { self.stream.synchronize().map_err(|e| CudaPrbError::Cuda(e.to_string())) }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"), Err(_) => true }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { required.saturating_add(headroom) <= free } else { true }
    }

    // -------- Helpers --------
    fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
        if st == 0 || s == e { vec![s] } else { (s..=e).step_by(st).collect() }
    }
    fn axis_i32((s, e, st): (i32, i32, i32)) -> Vec<i32> {
        if st == 0 || s == e { vec![s] } else { let mut v=Vec::new(); let mut x=s; while x<=e { v.push(x); x+=st; } v }
    }
    fn expand_grid(range: &PrbBatchRange, smooth_flag: bool) -> Vec<PrbParams> {
        let sps = Self::axis_usize(range.smooth_period);
        let rps = Self::axis_usize(range.regression_period);
        let pos = Self::axis_usize(range.polynomial_order);
        let ros = Self::axis_i32(range.regression_offset);
        let mut out = Vec::with_capacity(sps.len()*rps.len()*pos.len()*ros.len());
        for &sp in &sps { for &rp in &rps { for &po in &pos { for &ro in &ros {
            out.push(PrbParams{ smooth_data:Some(smooth_flag), smooth_period:Some(sp), regression_period:Some(rp), polynomial_order:Some(po), regression_offset:Some(ro), ndev:Some(2.0), equ_from:Some(0) });
        }}}}
        out
    }

    fn ssf_filter_f32(data: &[f32], period: usize, first: usize) -> Vec<f32> {
        let len = data.len();
        let mut out = vec![f32::NAN; len];
        if len == 0 { return out; }
        let pi = core::f32::consts::PI;
        let omega = 2.0f32 * pi / (period as f32);
        let a = (-core::f32::consts::SQRT_2 * pi / (period as f32)).exp();
        let b = 2.0f32 * a * ((core::f32::consts::SQRT_2 / 2.0f32) * omega).cos();
        let c3 = -a * a; let c2 = b; let c1 = 1.0f32 - c2 - c3;
        let mut y1 = f32::NAN; let mut y2 = f32::NAN;
        for i in first..len {
            let x = data[i];
            // Pine fallback: prev1 = nz(y1, x); prev2 = nz(y2, nz(y1, x))
            let prev1 = if y1.is_nan() { x } else { y1 };
            let prev2 = if y2.is_nan() { prev1 } else { y2 };
            let y = c1 * x + c2 * prev1 + c3 * prev2;
            out[i] = y; y2 = y1; y1 = y;
        }
        out
    }

    fn contig_valid(series: &[f32]) -> Vec<i32> {
        let mut v = vec![0i32; series.len()];
        let mut c: i32 = 0;
        for (i, &x) in series.iter().enumerate() {
            if x.is_nan() { c = 0; } else { c += 1; }
            v[i] = c;
        }
        v
    }

    fn build_a_inv(n: usize, k: usize) -> Vec<f32> {
        // Build normal matrix A (m x m), then invert with Gauss-Jordan (double), return row-major f32.
        let m = k + 1; let max_m = 8usize; let mut a = vec![0.0f64; m * m];
        // Power sums Sx[p] for p in 0..=2k
        let mut sx = vec![0.0f64; 2 * k + 1];
        for j in 1..=n { let jf = j as f64; let mut p = 1.0f64; sx[0] += 1.0; for t in 1..=2 * k { p *= jf; sx[t] += p; } }
        for i in 0..m { for j in 0..m { a[i * m + j] = sx[i + j]; } }
        // Augment with identity
        let mut aug = vec![0.0f64; m * 2 * m];
        for r in 0..m { for c in 0..m { aug[r * (2 * m) + c] = a[r * m + c]; } aug[r * (2 * m) + (m + r)] = 1.0; }
        for i in 0..m {
            // pivot
            let mut piv = i; let mut best = aug[i * (2 * m) + i].abs();
            for r in (i + 1)..m { let val = aug[r * (2 * m) + i].abs(); if val > best { best = val; piv = r; } }
            if piv != i { for c in 0..(2 * m) { aug.swap(i * (2 * m) + c, piv * (2 * m) + c); } }
            let diag = aug[i * (2 * m) + i];
            let invd = 1.0f64 / diag;
            for c in 0..(2 * m) { aug[i * (2 * m) + c] *= invd; }
            for r in 0..m { if r == i { continue; } let f = aug[r * (2 * m) + i]; if f == 0.0 { continue; } for c in 0..(2 * m) { aug[r * (2 * m) + c] -= f * aug[i * (2 * m) + c]; } }
        }
        let mut inv = vec![0.0f32; max_m * max_m]; // pad to max stride expected by kernels
        for r in 0..m { for c in 0..m { inv[r * max_m + c] = aug[r * (2 * m) + (m + c)] as f32; } }
        inv
    }

    pub fn prb_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PrbBatchRange,
        smooth_data: bool,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaPrbError> {
        if data_f32.is_empty() { return Err(CudaPrbError::InvalidInput("empty data".into())); }
        let len = data_f32.len();
        let first_valid = data_f32.iter().position(|v| !v.is_nan()).ok_or_else(|| CudaPrbError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep, smooth_data);
        if combos.is_empty() { return Err(CudaPrbError::InvalidInput("no parameter combinations".into())); }

        // Validate params; collect unique (n,k) and unique smooth_period
        let mut uniq_nk: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut uniq_sp: BTreeSet<usize> = BTreeSet::new();
        for c in &combos { let n = c.regression_period.unwrap(); let k = c.polynomial_order.unwrap(); if n == 0 || n > len { return Err(CudaPrbError::InvalidInput("invalid regression_period".into())); } uniq_nk.insert((n,k)); if smooth_data { uniq_sp.insert(c.smooth_period.unwrap_or(10)); } }
        // Precompute A^{-1} per (n,k)
        let mut a_inv_map: BTreeMap<(usize, usize), Vec<f32>> = BTreeMap::new();
        for &(n,k) in &uniq_nk { a_inv_map.insert((n,k), Self::build_a_inv(n, k)); }
        // Prepare groups by smooth_period
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        if smooth_data {
            for (idx, c) in combos.iter().enumerate() { let sp = c.smooth_period.unwrap_or(10); groups.entry(sp).or_default().push(idx); }
        } else {
            groups.insert(0, (0..combos.len()).collect());
        }

        // Allocate output buffers on device once
        let total_elems = combos.len() * len;
        let bytes_inputs = len * std::mem::size_of::<f32>();
        let bytes_out = 3 * total_elems * std::mem::size_of::<f32>();
        let bytes_params = combos.len() * (3 * std::mem::size_of::<i32>() + 64); // rough
        let required = bytes_inputs + bytes_out + bytes_params + 64 * 1024; // plus contig/a_inv
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaPrbError::InvalidInput("insufficient VRAM for PRB batch".into()));
        }

        let mut d_main: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }.map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let mut d_up: DeviceBuffer<f32>   = unsafe { DeviceBuffer::uninitialized(total_elems) }.map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let mut d_lo: DeviceBuffer<f32>   = unsafe { DeviceBuffer::uninitialized(total_elems) }.map_err(|e| CudaPrbError::Cuda(e.to_string()))?;

        // For each smoothing group: build source, contig, and rows slice; then launch
        for (sp, rows_idx) in groups.iter() {
            // Build source series (possibly smoothed)
            let source: Vec<f32> = if smooth_data { Self::ssf_filter_f32(data_f32, *sp, first_valid) } else { data_f32.to_vec() };
            let contig = Self::contig_valid(&source);

            // H2D for this group
            let h_src = LockedBuffer::from_slice(&source).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            let h_contig = LockedBuffer::from_slice(&contig).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            let mut d_src: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            let mut d_contig: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(len) }.map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            unsafe {
                d_src.async_copy_from(h_src.as_slice(), &self.stream).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
                d_contig.async_copy_from(h_contig.as_slice(), &self.stream).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            }

            // Row-wise parameters for this group
            let mut periods: Vec<i32> = Vec::with_capacity(rows_idx.len());
            let mut orders: Vec<i32> = Vec::with_capacity(rows_idx.len());
            let mut offsets: Vec<i32> = Vec::with_capacity(rows_idx.len());
            let mut a_invs: Vec<f32> = Vec::with_capacity(rows_idx.len() * 64);
            let mut row_map: Vec<i32> = Vec::with_capacity(rows_idx.len());
            let max_m: i32 = 8;
            for &row in rows_idx {
                let c = &combos[row];
                let n = c.regression_period.unwrap();
                let k = c.polynomial_order.unwrap();
                let off = c.regression_offset.unwrap_or(0);
                periods.push(n as i32); orders.push(k as i32); offsets.push(off as i32);
                let ainv = a_inv_map.get(&(n, k)).expect("missing ainv");
                a_invs.extend_from_slice(ainv);
                row_map.push(row as i32);
            }
            let d_periods = DeviceBuffer::from_slice(&periods).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            let d_orders  = DeviceBuffer::from_slice(&orders).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            let d_offsets = DeviceBuffer::from_slice(&offsets).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            let d_ainv    = DeviceBuffer::from_slice(&a_invs).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            let d_rowmap  = DeviceBuffer::from_slice(&row_map).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;

            // Launch kernel: grid.y chunking to <= 65535
            let func = self.module.get_function("prb_batch_f32").map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
            let block_x: u32 = match self.policy.batch { BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x, _ => 1 };
            let grid_x: u32 = 1; let block: BlockSize = (block_x, 1, 1).into();

            const MAX_GRID_Y: usize = 65_535;
            let mut start = 0usize;
            while start < rows_idx.len() {
                let chunk = (rows_idx.len() - start).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x, chunk as u32, 1).into();
                unsafe {
                    let mut p_src = d_src.as_device_ptr().as_raw();
                    let mut len_i = len as i32; let mut first_i = first_valid as i32;
                    let mut p_per = d_periods.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut p_ord = d_orders.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut p_off = d_offsets.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut combos_i = chunk as i32;
                    let mut max_m_i = max_m; let mut stride_i = (8 * 8) as i32; // using 8x8 padded storage
                    let mut p_ainv = d_ainv.as_device_ptr().as_raw().wrapping_add((start * (8 * 8) * std::mem::size_of::<f32>()) as u64);
                    let mut p_contig = d_contig.as_device_ptr().as_raw();
                    let mut ndev_f = 2.0f32; // fixed for batch (as in CPU batch grid)
                    let mut p_rowmap = d_rowmap.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
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
                    self.stream.launch(&func, grid, block, 0, &mut args)
                        .map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
                }
                start += chunk;
            }
            // sync group copies before next group
            self.stream.synchronize().map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        }

        Ok((
            DeviceArrayF32 { buf: d_main, rows: combos.len(), cols: len },
            DeviceArrayF32 { buf: d_up,   rows: combos.len(), cols: len },
            DeviceArrayF32 { buf: d_lo,   rows: combos.len(), cols: len },
        ))
    }

    pub fn prb_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PrbParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaPrbError> {
        if cols == 0 || rows == 0 { return Err(CudaPrbError::InvalidInput("empty grid".into())); }
        if data_tm_f32.len() != cols * rows { return Err(CudaPrbError::InvalidInput("data length mismatch".into())); }
        let n = params.regression_period.unwrap_or(100);
        let k = params.polynomial_order.unwrap_or(2);
        let off = params.regression_offset.unwrap_or(0);
        if n == 0 || n > rows { return Err(CudaPrbError::InvalidInput("invalid regression_period".into())); }
        // Per-column first_valid
        let mut firsts = vec![0i32; cols];
        for s in 0..cols { let mut fv = -1i32; for t in 0..rows { let v = data_tm_f32[t * cols + s]; if !v.is_nan() { fv = t as i32; break; } } firsts[s] = fv; if fv >= 0 { if (rows - fv as usize) < n { return Err(CudaPrbError::InvalidInput("not enough valid data".into())); } } }
        // Host smoothing (single smooth_period per call)
        let smooth = params.smooth_data.unwrap_or(true);
        let sp = params.smooth_period.unwrap_or(10);
        let mut sm_tm = vec![f32::NAN; cols * rows];
        if smooth {
            for s in 0..cols {
                let fv = firsts[s];
                if fv < 0 { continue; }
                // gather column
                let mut col = vec![f32::NAN; rows]; for t in 0..rows { col[t] = data_tm_f32[t * cols + s]; }
                let sm = Self::ssf_filter_f32(&col, sp, fv as usize);
                for t in 0..rows { sm_tm[t * cols + s] = sm[t]; }
            }
        } else {
            sm_tm.copy_from_slice(data_tm_f32);
        }
        let contig_tm = {
            let mut v = vec![0i32; cols * rows];
            for s in 0..cols { let mut c=0i32; for t in 0..rows { let y = sm_tm[t * cols + s]; if y.is_nan() { c=0; } else { c+=1; } v[t * cols + s] = c; } }
            v
        };

        let ainv = Self::build_a_inv(n, k);
        let d_prices_tm = DeviceBuffer::from_slice(&sm_tm).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let d_contig_tm = DeviceBuffer::from_slice(&contig_tm).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let d_ainv = DeviceBuffer::from_slice(&ainv).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let mut d_m  : DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let mut d_u  : DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let mut d_l  : DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaPrbError::Cuda(e.to_string()))?;

        let func = self.module.get_function("prb_many_series_one_param_f32").map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x, _ => 256 };
        let grid: GridSize = (((cols as u32) + block_x - 1) / block_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let d_firsts = DeviceBuffer::from_slice(&firsts).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        unsafe {
            let mut p_tm = d_prices_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32; let mut rows_i = rows as i32;
            let mut period_i = n as i32; let mut order_i = k as i32; let mut off_i = off as i32;
            let mut max_m_i = 8i32; let mut stride_i = (8 * 8) as i32;
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
            self.stream.launch(&func, grid, block, 0, &mut args).map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        }
        self.stream.synchronize().map_err(|e| CudaPrbError::Cuda(e.to_string()))?;
        Ok((
            DeviceArrayF32 { buf: d_m, rows, cols },
            DeviceArrayF32 { buf: d_u, rows, cols },
            DeviceArrayF32 { buf: d_l, rows, cols },
        ))
    }
}

// ---------------- Benches ----------------
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

    struct BatchState { cuda: CudaPrb, price: Vec<f32>, sweep: PrbBatchRange }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.prb_batch_dev(&self.price, &self.sweep, false).expect("prb batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaPrb::new(0).expect("cuda");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = PrbBatchRange { smooth_period: (10, 10, 0), regression_period: (100, 100 + PARAM_SWEEP - 1, 1), polynomial_order: (2, 2, 0), regression_offset: (0, 0, 0) };
        Box::new(BatchState { cuda, price, sweep })
    }

    struct ManyState { cuda: CudaPrb, data_tm: Vec<f32>, cols: usize, rows: usize, params: PrbParams }
    impl CudaBenchState for ManyState { fn launch(&mut self) { let _ = self.cuda.prb_many_series_one_param_time_major_dev(&self.data_tm, self.cols, self.rows, &self.params).expect("prb many"); } }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaPrb::new(0).expect("cuda");
        let cols = MANY_SERIES_COLS; let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = PrbParams { smooth_data: Some(false), smooth_period: Some(10), regression_period: Some(100), polynomial_order: Some(2), regression_offset: Some(0), ndev: Some(2.0), equ_from: Some(0) };
        Box::new(ManyState { cuda, data_tm, cols, rows, params })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("prb", "one_series_many_params", "prb_cuda_batch_dev", "1m_x_250", prep_one_series_many_params)
                .with_sample_size(10).with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new("prb", "many_series_one_param", "prb_cuda_many_series_one_param", "250x1m", prep_many_series_one_param)
                .with_sample_size(5).with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
