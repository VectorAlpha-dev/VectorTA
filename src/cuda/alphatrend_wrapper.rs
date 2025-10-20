#![cfg(feature = "cuda")]

//! CUDA wrapper for AlphaTrend (AT).
//! - Mirrors ALMA-style conventions: JIT options, NON_BLOCKING stream, optional VRAM checks,
//!   simple policies, and chunked launches.
//! - Category: Recurrence/IIR. Host precomputes shared TR and momentum arrays.

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::alphatrend::{AlphaTrendBatchRange, AlphaTrendParams};
use crate::indicators::mfi::{mfi_with_kernel, MfiInput, MfiParams};
use crate::indicators::rsi::{rsi_with_kernel, RsiInput, RsiParams};
use crate::utilities::enums::Kernel;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashMap;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaAlphaTrendError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAlphaTrendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAlphaTrendError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAlphaTrendError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}
impl std::error::Error for CudaAlphaTrendError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaAlphaTrendPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaAlphaTrendBatch {
    pub k1: DeviceArrayF32,
    pub k2: DeviceArrayF32,
    pub combos: Vec<AlphaTrendParams>,
}

pub struct CudaAlphaTrend {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaAlphaTrendPolicy,
}

impl CudaAlphaTrend {
    pub fn new(device_id: usize) -> Result<Self, CudaAlphaTrendError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/alphatrend_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                Module::from_ptx(ptx, &[]).map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaAlphaTrendPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, p: CudaAlphaTrendPolicy) {
        self.policy = p;
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if env::var("CUDA_MEM_CHECK").ok().as_deref() == Some("0") {
            return true;
        }
        match mem_get_info() {
            Ok((free, _)) => required_bytes.saturating_add(headroom_bytes) <= free,
            Err(_) => true,
        }
    }

    // ---- helpers: expand grid ----
    fn expand_grid(r: &AlphaTrendBatchRange) -> Vec<AlphaTrendParams> {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                return vec![s];
            }
            (s..=e).step_by(st).collect()
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                return vec![s];
            }
            let mut out = Vec::new();
            let mut v = s;
            while v <= e + 1e-12 {
                out.push(v);
                v += st;
            }
            out
        }
        let coeffs = axis_f64(r.coeff);
        let periods = axis_usize(r.period);
        let mut out = Vec::with_capacity(coeffs.len() * periods.len());
        for &c in &coeffs {
            for &p in &periods {
                out.push(AlphaTrendParams {
                    coeff: Some(c),
                    period: Some(p),
                    no_volume: Some(r.no_volume),
                });
            }
        }
        out
    }

    // ---- host precompute: TR (shared) ----
    fn build_tr_f32(
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<(Vec<f32>, usize), CudaAlphaTrendError> {
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaAlphaTrendError::InvalidInput(
                "inconsistent data lengths".into(),
            ));
        }
        if high.is_empty() {
            return Err(CudaAlphaTrendError::InvalidInput("empty input".into()));
        }
        let len = close.len();
        let first = close
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaAlphaTrendError::InvalidInput("all values are NaN".into()))?;
        let mut tr = vec![f32::NAN; len];
        if first < len {
            tr[first] = high[first] - low[first];
        }
        for i in (first + 1)..len {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let m = hl.max(hc.max(lc));
            tr[i] = m;
        }
        Ok((tr, first))
    }

    // ---- host precompute: momentum per unique period (RSI or MFI) ----
    fn build_momentum_table_f32(
        no_volume: bool,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
        unique_periods: &[usize],
    ) -> Result<HashMap<usize, Vec<f32>>, CudaAlphaTrendError> {
        let len = close.len();
        let mut out: HashMap<usize, Vec<f32>> = HashMap::with_capacity(unique_periods.len());
        if no_volume {
            // RSI(close)
            let close64: Vec<f64> = close.iter().map(|&v| v as f64).collect();
            for &p in unique_periods {
                let rsi = rsi_with_kernel(
                    &RsiInput::from_slice(&close64, RsiParams { period: Some(p) }),
                    Kernel::Scalar,
                )
                .map_err(|e| CudaAlphaTrendError::InvalidInput(format!("rsi: {}", e)))?;
                out.insert(p, rsi.values.into_iter().map(|v| v as f32).collect());
            }
        } else {
            // MFI(HLC3, volume)
            let mut hlc3_64 = vec![0f64; len];
            for i in 0..len {
                let h = high[i] as f64;
                let l = low[i] as f64;
                let c = close[i] as f64;
                hlc3_64[i] = (h + l + c) / 3.0f64;
            }
            let volume64: Vec<f64> = volume.iter().map(|&v| v as f64).collect();
            for &p in unique_periods {
                let mfi = mfi_with_kernel(
                    &MfiInput::from_slices(&hlc3_64, &volume64, MfiParams { period: Some(p) }),
                    Kernel::Scalar,
                )
                .map_err(|e| CudaAlphaTrendError::InvalidInput(format!("mfi: {}", e)))?;
                out.insert(p, mfi.values.into_iter().map(|v| v as f32).collect());
            }
        }
        Ok(out)
    }

    fn launch_batch(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_tr: &DeviceBuffer<f32>,
        d_momentum_flat: &DeviceBuffer<f32>,
        d_mrow_for_combo: &DeviceBuffer<i32>,
        d_coeffs: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        n_mrows: usize,
        d_k1: &mut DeviceBuffer<f32>,
        d_k2: &mut DeviceBuffer<f32>,
        policy: BatchKernelPolicy,
        combo_offset: usize,
    ) -> Result<(), CudaAlphaTrendError> {
        let func = self
            .module
            .get_function("alphatrend_batch_f32")
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        let block_x = match policy {
            BatchKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let max_grid_x = 65_535u32;
        let needed_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid_x = needed_x.min(max_grid_x).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            // Slice outputs for this combo chunk
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut tr_ptr = d_tr.as_device_ptr().as_raw();
            let mut mom_ptr = d_momentum_flat.as_device_ptr().as_raw();
            // Advance scalar arrays by combo_offset elements
            let mut map_ptr = d_mrow_for_combo
                .as_device_ptr()
                .as_raw()
                .wrapping_add((combo_offset * std::mem::size_of::<i32>()) as u64);
            let mut coeff_ptr = d_coeffs
                .as_device_ptr()
                .as_raw()
                .wrapping_add((combo_offset * std::mem::size_of::<f32>()) as u64);
            let mut period_ptr = d_periods
                .as_device_ptr()
                .as_raw()
                .wrapping_add((combo_offset * std::mem::size_of::<i32>()) as u64);
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut ncomb_i = n_combos as i32;
            let mut nmrows_i = n_mrows as i32;
            // offset outputs by combo_offset * len
            let out_off_bytes = (combo_offset * len * std::mem::size_of::<f32>()) as u64;
            let mut k1_ptr = d_k1.as_device_ptr().as_raw().wrapping_add(out_off_bytes);
            let mut k2_ptr = d_k2.as_device_ptr().as_raw().wrapping_add(out_off_bytes);
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut mom_ptr as *mut _ as *mut c_void,
                &mut map_ptr as *mut _ as *mut c_void,
                &mut coeff_ptr as *mut _ as *mut c_void,
                &mut period_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut ncomb_i as *mut _ as *mut c_void,
                &mut nmrows_i as *mut _ as *mut c_void,
                &mut k1_ptr as *mut _ as *mut c_void,
                &mut k2_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn alphatrend_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        volume_f32: &[f32],
        sweep: &AlphaTrendBatchRange,
    ) -> Result<CudaAlphaTrendBatch, CudaAlphaTrendError> {
        let len = close_f32.len();
        if high_f32.len() != len || low_f32.len() != len || volume_f32.len() != len {
            return Err(CudaAlphaTrendError::InvalidInput(
                "inconsistent data lengths".into(),
            ));
        }
        let (tr, first) = Self::build_tr_f32(high_f32, low_f32, close_f32)?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaAlphaTrendError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // Validate combos and collect unique periods
        let mut unique: Vec<usize> = Vec::new();
        for p in &combos {
            let period = p.period.unwrap_or(14);
            if period == 0 || period > len {
                return Err(CudaAlphaTrendError::InvalidInput(format!(
                    "invalid period {}",
                    period
                )));
            }
            if len - first < period {
                return Err(CudaAlphaTrendError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
            unique.push(period);
        }
        unique.sort_unstable();
        unique.dedup();
        let mom_map = Self::build_momentum_table_f32(
            sweep.no_volume,
            high_f32,
            low_f32,
            close_f32,
            volume_f32,
            &unique,
        )?;
        let n_mrows = mom_map.len();

        // Build flat momentum table (rows = unique periods)
        let mut period_to_row: HashMap<usize, i32> = HashMap::with_capacity(n_mrows);
        let mut momentum_flat = Vec::<f32>::with_capacity(n_mrows * len);
        for (row_idx, &p) in unique.iter().enumerate() {
            period_to_row.insert(p, row_idx as i32);
            let row = mom_map.get(&p).expect("momentum row");
            momentum_flat.extend_from_slice(row);
        }
        let coeffs: Vec<f32> = combos
            .iter()
            .map(|c| c.coeff.unwrap_or(1.0) as f32)
            .collect();
        let periods: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(14) as i32)
            .collect();
        let map_rows: Vec<i32> = combos
            .iter()
            .map(|c| {
                period_to_row
                    .get(&c.period.unwrap_or(14))
                    .copied()
                    .unwrap_or(-1)
            })
            .collect();

        // VRAM estimation
        let bytes = len * 4 * 4  // high, low, close, tr
            + momentum_flat.len() * 4
            + (coeffs.len() + periods.len() + map_rows.len()) * 4
            + combos.len() * len * 4 * 2; // k1+k2
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaAlphaTrendError::InvalidInput(
                "estimated device memory exceeds free VRAM".into(),
            ));
        }

        // Upload inputs
        let d_high = DeviceBuffer::from_slice(high_f32)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_f32)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close_f32)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_tr =
            DeviceBuffer::from_slice(&tr).map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_mom = DeviceBuffer::from_slice(&momentum_flat)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_map = DeviceBuffer::from_slice(&map_rows)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_coeffs = DeviceBuffer::from_slice(&coeffs)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        // Allocate outputs
        let rows = combos.len();
        let elems = rows * len;
        let mut d_k1: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let mut d_k2: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        // Chunk combo dimension if needed for grid.x limit
        let block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        } as usize;
        let max_grid_x = 65_535usize;
        let max_combos_per_launch = block_x * max_grid_x;
        let mut launched = 0usize;
        while launched < rows {
            let chunk = (rows - launched).min(max_combos_per_launch);
            self.launch_batch(
                &d_high,
                &d_low,
                &d_close,
                &d_tr,
                &d_mom,
                &d_map,
                &d_coeffs,
                &d_periods,
                len,
                first,
                chunk,
                n_mrows,
                &mut d_k1,
                &mut d_k2,
                self.policy.batch,
                launched,
            )?;
            launched += chunk;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        Ok(CudaAlphaTrendBatch {
            k1: DeviceArrayF32 {
                buf: d_k1,
                rows,
                cols: len,
            },
            k2: DeviceArrayF32 {
                buf: d_k2,
                rows,
                cols: len,
            },
            combos,
        })
    }

    pub fn alphatrend_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        coeff: f64,
        period: usize,
        no_volume: bool,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaAlphaTrendError> {
        if high_tm_f32.len() != cols * rows
            || low_tm_f32.len() != cols * rows
            || close_tm_f32.len() != cols * rows
            || volume_tm_f32.len() != cols * rows
        {
            return Err(CudaAlphaTrendError::InvalidInput(
                "inconsistent time-major shapes".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaAlphaTrendError::InvalidInput("invalid period".into()));
        }

        // Build first_valid per series and TR_tm (time-major)
        let mut first_valids = vec![0i32; cols];
        let mut tr_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if fv.is_none() && !close_tm_f32[idx].is_nan() {
                    fv = Some(t);
                }
                if t == 0 {
                    continue;
                }
                let hl = high_tm_f32[idx] - low_tm_f32[idx];
                let pc = close_tm_f32[(t - 1) * cols + s];
                let hc = (high_tm_f32[idx] - pc).abs();
                let lc = (low_tm_f32[idx] - pc).abs();
                tr_tm[idx] = hl.max(hc.max(lc));
            }
            first_valids[s] = fv.unwrap_or(rows as usize) as i32;
            if let Some(f) = fv {
                tr_tm[f * cols + s] = high_tm_f32[f * cols + s] - low_tm_f32[f * cols + s];
            }
        }

        // Momentum_tm (RSI or MFI) per series.
        // Note: host-side computation for correctness; may be optimized in future.
        let mut momentum_tm = vec![f32::NAN; cols * rows];
        if no_volume {
            // RSI on close per series
            for s in 0..cols {
                let mut col = vec![0f64; rows];
                for t in 0..rows {
                    col[t] = close_tm_f32[t * cols + s] as f64;
                }
                let mv = rsi_with_kernel(
                    &RsiInput::from_slice(
                        &col,
                        RsiParams {
                            period: Some(period),
                        },
                    ),
                    Kernel::Scalar,
                )
                .map_err(|e| CudaAlphaTrendError::InvalidInput(format!("rsi: {}", e)))?
                .values;
                for t in 0..rows {
                    momentum_tm[t * cols + s] = mv[t] as f32;
                }
            }
        } else {
            // MFI on HLC3 + volume per series
            for s in 0..cols {
                let mut hlc3 = vec![0f64; rows];
                let mut vol = vec![0f64; rows];
                for t in 0..rows {
                    let idx = t * cols + s;
                    hlc3[t] =
                        ((high_tm_f32[idx] + low_tm_f32[idx] + close_tm_f32[idx]) as f64) / 3.0;
                    vol[t] = volume_tm_f32[idx] as f64;
                }
                let mv = mfi_with_kernel(
                    &MfiInput::from_slices(
                        &hlc3,
                        &vol,
                        MfiParams {
                            period: Some(period),
                        },
                    ),
                    Kernel::Scalar,
                )
                .map_err(|e| CudaAlphaTrendError::InvalidInput(format!("mfi: {}", e)))?
                .values;
                for t in 0..rows {
                    momentum_tm[t * cols + s] = mv[t] as f32;
                }
            }
        }

        // VRAM estimate
        let bytes = 4 * cols * rows * 2 // tr_tm + momentum_tm
            + 4 * cols                  // first_valids
            + 4 * cols * rows * 2; // k1 + k2 outputs
        if !Self::will_fit(bytes, 64 * 1024 * 1024) {
            return Err(CudaAlphaTrendError::InvalidInput(
                "estimated device memory exceeds free VRAM".into(),
            ));
        }

        // Upload
        let d_high_tm = DeviceBuffer::from_slice(high_tm_f32)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_low_tm = DeviceBuffer::from_slice(low_tm_f32)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_tr_tm = DeviceBuffer::from_slice(&tr_tm)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_mom_tm = DeviceBuffer::from_slice(&momentum_tm)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        let mut d_k1_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let mut d_k2_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        // Launch
        let func = self
            .module
            .get_function("alphatrend_many_series_one_param_f32")
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut tr_ptr = d_tr_tm.as_device_ptr().as_raw();
            let mut mom_ptr = d_mom_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut coeff_f = coeff as f32;
            let mut period_i = period as i32;
            let mut k1_ptr = d_k1_tm.as_device_ptr().as_raw();
            let mut k2_ptr = d_k2_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut mom_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut coeff_f as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut k1_ptr as *mut _ as *mut c_void,
                &mut k2_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaAlphaTrendError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_k1_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_k2_tm,
                rows,
                cols,
            },
        ))
    }
}

// -------- Benches (batch only for now) --------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 200_000;
    const PARAM_SWEEP: usize = 64;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * 4 * 4; // high, low, close, tr
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_SWEEP * 4; // k1+k2
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct AtBatchState {
        cuda: CudaAlphaTrend,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        volume: Vec<f32>,
        sweep: AlphaTrendBatchRange,
    }
    impl CudaBenchState for AtBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .alphatrend_batch_dev(
                    &self.high,
                    &self.low,
                    &self.close,
                    &self.volume,
                    &self.sweep,
                )
                .expect("alphatrend batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaAlphaTrend::new(0).expect("cuda alphatrend");
        // Generate synthetic HLCV with NaNs prefix
        let (h, l, c) = {
            let mut h = vec![f32::NAN; ONE_SERIES_LEN];
            let mut l = vec![f32::NAN; ONE_SERIES_LEN];
            let mut c = vec![f32::NAN; ONE_SERIES_LEN];
            for t in 3..ONE_SERIES_LEN {
                let x = t as f32;
                h[t] = (x * 0.0012).sin() + 0.03;
                l[t] = h[t] - 0.02 - 0.006 * (x * 0.0009).cos().abs();
                c[t] = 0.5 * (h[t] + l[t]) + 0.0007 * (x * 0.0011).cos();
            }
            (h, l, c)
        };
        let mut v = vec![f32::NAN; ONE_SERIES_LEN];
        for i in 3..ONE_SERIES_LEN {
            v[i] = (i as f32 * 0.0009).cos().abs() + 0.5;
        }
        let sweep = AlphaTrendBatchRange {
            coeff: (0.8, 1.6, 0.0125),
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            no_volume: true,
        };
        Box::new(AtBatchState {
            cuda,
            high: h,
            low: l,
            close: c,
            volume: v,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "alphatrend",
            "one_series_many_params",
            "alphatrend_cuda_batch_dev",
            "200k_x_64",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
