//! CUDA support for the VAR indicator (rolling variance with nbdev scaling).
//!
//! Parity targets (mirrors ALMA/Deviation wrappers):
//! - PTX load via DetermineTargetFromContext + OptLevel O2 with simple fallback
//! - NON_BLOCKING stream
//! - VRAM checks with optional headroom and grid.y chunking
//! - Public entry points:
//!     - one-series × many-params (batch)
//!     - many-series × one-param (time‑major)
//! - Numerics: warmup/NaN identical to scalar; compute windows via host-built
//!   prefix sums in f64 to reduce drift; outputs in f32 scaled by nbdev^2.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::var::{var_expand_grid, VarBatchRange, VarParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, DeviceCopy, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaVarError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaVarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaVarError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaVarError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaVarError {}

// ---------- Float-float host POD to mirror CUDA float2 ----------
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Float2 {
    hi: f32,
    lo: f32,
}

// SAFETY: Plain old data with no pointers/reference fields, C layout
unsafe impl DeviceCopy for Float2 {}

#[inline]
fn split_f64_to_float2_vec(src: &[f64]) -> Vec<Float2> {
    let mut v = Vec::with_capacity(src.len());
    for &d in src {
        let hi = d as f32;
        let lo = (d - (hi as f64)) as f32;
        v.push(Float2 { hi, lo });
    }
    v
}

impl CudaVar {
    /// Best-effort: request an L2 persisting cache window for read-mostly spans.
    fn try_enable_persisting_l2(&self, base_dev_ptr: u64, bytes: usize) {
        unsafe {
            use cust::device::Device as CuDevice;
            use cust::sys::{
                cuCtxSetLimit, cuDeviceGetAttribute, cuStreamSetAttribute,
                CUaccessPolicyWindow_v1 as CUaccessPolicyWindow,
                CUaccessProperty_enum as AccessProp, CUdevice_attribute_enum as DevAttr,
                CUlimit_enum as CULimit, CUstreamAttrID_enum as StreamAttrId,
                CUstreamAttrValue_v1 as CUstreamAttrValue,
            };

            // Max window size supported by device
            let mut max_window_bytes_i32: i32 = 0;
            if let Ok(dev) = CuDevice::get_device(self.device_id) {
                let _ = cuDeviceGetAttribute(
                    &mut max_window_bytes_i32 as *mut _,
                    DevAttr::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                    dev.as_raw(),
                );
            }
            let max_window_bytes = (max_window_bytes_i32.max(0) as usize).min(bytes);
            if max_window_bytes == 0 {
                return;
            }

            // Best-effort set-aside for L2 persistence
            let _ = cuCtxSetLimit(CULimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, max_window_bytes);

            // Configure the stream access policy window
            let mut val: CUstreamAttrValue = std::mem::zeroed();
            val.accessPolicyWindow = CUaccessPolicyWindow {
                base_ptr: base_dev_ptr as *mut std::ffi::c_void,
                num_bytes: max_window_bytes,
                hitRatio: 0.6f32,
                hitProp: AccessProp::CU_ACCESS_PROPERTY_PERSISTING,
                missProp: AccessProp::CU_ACCESS_PROPERTY_STREAMING,
            };
            let _ = cuStreamSetAttribute(
                self.stream.as_inner(),
                StreamAttrId::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &mut val as *mut _,
            );
        }
    }
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
pub struct CudaVarPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaVarPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaVar {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaVarPolicy,
    debug_logged: std::sync::atomic::AtomicBool,
}

impl CudaVar {
    pub fn new(device_id: usize) -> Result<Self, CudaVarError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/var_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaVarError::Cuda(e.to_string()))?
                }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaVarPolicy::default(),
            debug_logged: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaVarPolicy) -> Result<Self, CudaVarError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    fn headroom_bytes() -> usize {
        env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    fn build_prefixes_1d(data_f32: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize, usize) {
        let len = data_f32.len();
        let first_valid = data_f32.iter().position(|v| !v.is_nan()).unwrap_or(len);
        let mut ps = vec![0f64; len + 1];
        let mut ps2 = vec![0f64; len + 1];
        let mut pn = vec![0i32; len + 1];
        let mut a = 0.0f64;
        let mut b = 0.0f64;
        let mut c = 0i32;
        for i in 0..len {
            if i >= first_valid {
                let v = data_f32[i];
                if v.is_nan() {
                    c += 1;
                } else {
                    let dv = v as f64;
                    a += dv;
                    b += dv * dv;
                }
            }
            ps[i + 1] = a;
            ps2[i + 1] = b;
            pn[i + 1] = c;
        }
        (ps, ps2, pn, first_valid, len)
    }

    fn build_prefixes_time_major(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
    ) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        // prefix at (t,s) stored at index (t*cols + s) + 1
        let total = data_tm_f32.len();
        let mut ps = vec![0.0f64; total + 1];
        let mut ps2 = vec![0.0f64; total + 1];
        let mut pn = vec![0i32; total + 1];
        for s in 0..cols {
            let fv = first_valids[s].max(0) as usize;
            let mut a = 0.0f64;
            let mut b = 0.0f64;
            let mut c = 0i32;
            for t in 0..rows {
                let idx = t * cols + s;
                if t >= fv {
                    let v = data_tm_f32[idx];
                    if v.is_nan() {
                        c += 1;
                    } else {
                        let dv = v as f64;
                        a += dv;
                        b += dv * dv;
                    }
                }
                let w = idx + 1;
                ps[w] = a;
                ps2[w] = b;
                pn[w] = c;
            }
        }
        (ps, ps2, pn)
    }

    // -------------------------- Batch entry point --------------------------
    pub fn var_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VarBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<VarParams>), CudaVarError> {
        if data_f32.is_empty() {
            return Err(CudaVarError::InvalidInput("empty data".into()));
        }
        let (ps, ps2, pn, first_valid, len) = Self::build_prefixes_1d(data_f32);

        let mut combos = var_expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaVarError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            if p == 0 || p > len {
                return Err(CudaVarError::InvalidInput("invalid period".into()));
            }
            if len - first_valid < p {
                return Err(CudaVarError::InvalidInput(
                    "not enough valid data after first valid".into(),
                ));
            }
            let nb = prm.nbdev.unwrap_or(1.0);
            if !nb.is_finite() {
                return Err(CudaVarError::InvalidInput("nbdev not finite".into()));
            }
        }

        // VRAM estimate + chunking (grid.y and memory)
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let nb2: Vec<f32> = combos
            .iter()
            .map(|c| {
                let x = c.nbdev.unwrap_or(1.0) as f32;
                x * x
            })
            .collect();
        let rows = combos.len();
        let out_elems = rows * len;
        let out_bytes = out_elems * std::mem::size_of::<f32>();

        // Convert f64 prefixes to float2 (double-single) to match device
        let ps_ff: Vec<Float2> = split_f64_to_float2_vec(&ps);
        let ps2_ff: Vec<Float2> = split_f64_to_float2_vec(&ps2);

        let in_bytes = (ps_ff.len() + ps2_ff.len()) * std::mem::size_of::<Float2>()
            + pn.len() * std::mem::size_of::<i32>()
            + periods.len() * std::mem::size_of::<i32>()
            + nb2.len() * std::mem::size_of::<f32>();
        let headroom = Self::headroom_bytes();
        let total_est = in_bytes + out_bytes + headroom;
        let mut y_chunks = 1usize;
        if let Ok((free, _)) = mem_get_info() {
            if total_est > free {
                let bytes_per_row = len * std::mem::size_of::<f32>();
                let max_rows = ((free.saturating_sub(in_bytes + headroom)) / bytes_per_row).max(1);
                y_chunks = (rows + max_rows - 1) / max_rows;
            }
        }
        let grid_y_limit = 65_535usize;
        if rows / y_chunks > grid_y_limit {
            y_chunks = (rows + grid_y_limit - 1) / grid_y_limit;
        }

        if !self.debug_logged.load(std::sync::atomic::Ordering::Relaxed)
            && env::var("BENCH_DEBUG").ok().as_deref() == Some("1")
        {
            eprintln!(
                "[var] policy={:?}/{:?} len={} rows={} chunks={}",
                self.policy.batch, self.policy.many_series, len, rows, y_chunks
            );
            self.debug_logged
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        // Upload static inputs (pinned host + async copies for large arrays)
        let mut d_ps: DeviceBuffer<Float2> = unsafe { DeviceBuffer::uninitialized_async(ps_ff.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let mut d_ps2: DeviceBuffer<Float2> = unsafe { DeviceBuffer::uninitialized_async(ps2_ff.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let mut d_pn: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized_async(pn.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let mut d_periods: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized_async(periods.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let mut d_nb2: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(nb2.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        let h_ps  = LockedBuffer::from_slice(&ps_ff).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let h_ps2 = LockedBuffer::from_slice(&ps2_ff).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let h_pn  = LockedBuffer::from_slice(&pn).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let h_periods = LockedBuffer::from_slice(&periods).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let h_nb2 = LockedBuffer::from_slice(&nb2).map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        unsafe {
            d_ps.async_copy_from(&h_ps, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
            d_ps2.async_copy_from(&h_ps2, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
            d_pn.async_copy_from(&h_pn, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
            d_periods.async_copy_from(&h_periods, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
            d_nb2.async_copy_from(&h_nb2, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        }

        // Best-effort persisting L2 for prefix buffer
        self.try_enable_persisting_l2(d_ps.as_device_ptr().as_raw() as u64,
            ps_ff.len() * std::mem::size_of::<Float2>());
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        // Launch in chunks across parameter rows
        let chunk_rows = (rows + y_chunks - 1) / y_chunks;
        for c in 0..y_chunks {
            let start_row = c * chunk_rows;
            if start_row >= rows {
                break;
            }
            let end_row = ((c + 1) * chunk_rows).min(rows);
            let n_rows = end_row - start_row;

            // Sub-pointer views
            let periods_ptr = unsafe {
                d_periods
                    .as_device_ptr()
                    .offset((start_row as isize).try_into().unwrap())
            };
            let nb2_ptr = unsafe {
                d_nb2
                    .as_device_ptr()
                    .offset((start_row as isize).try_into().unwrap())
            };
            let out_ptr = unsafe {
                d_out
                    .as_device_ptr()
                    .offset(((start_row * len) as isize).try_into().unwrap())
            };
            self.launch_batch_kernel_ptrs(
                &d_ps,
                &d_ps2,
                &d_pn,
                periods_ptr,
                nb2_ptr,
                len,
                first_valid,
                n_rows,
                out_ptr,
            )?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    fn launch_batch_kernel_ptrs(
        &self,
        d_ps: &DeviceBuffer<Float2>,
        d_ps2: &DeviceBuffer<Float2>,
        d_pn: &DeviceBuffer<i32>,
        periods_ptr: cust::memory::DevicePointer<i32>,
        nb2_ptr: cust::memory::DevicePointer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        out_ptr: cust::memory::DevicePointer<f32>,
    ) -> Result<(), CudaVarError> {
        let func = self
            .module
            .get_function("var_batch_f32")
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        if len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaVarError::InvalidInput(
                "inputs exceed kernel argument width".into(),
            ));
        }

        // The kernel tiles a small group of combos per block in grid.y (4 per group).
        const TILE: u32 = 4;
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let grid_y_groups = (((n_combos as u32) + TILE - 1) / TILE).max(1);
        let grid: GridSize = (grid_x.max(1), grid_y_groups, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut ps_ptr = d_ps.as_device_ptr().as_raw();
            let mut ps2_ptr = d_ps2.as_device_ptr().as_raw();
            let mut pn_ptr = d_pn.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut periods_ptr = periods_ptr.as_raw();
            let mut nb2_ptr = nb2_ptr.as_raw();
            let mut combos_i = n_combos as i32;
            let mut out_ptr = out_ptr.as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut ps_ptr as *mut _ as *mut c_void,
                &mut ps2_ptr as *mut _ as *mut c_void,
                &mut pn_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut nb2_ptr as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // ----------------------- Many-series: one param -----------------------
    pub fn var_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VarParams,
    ) -> Result<DeviceArrayF32, CudaVarError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVarError::InvalidInput(
                "matrix dims must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaVarError::InvalidInput("matrix shape mismatch".into()));
        }
        let period = params.period.unwrap_or(14);
        if period == 0 || period > rows {
            return Err(CudaVarError::InvalidInput("period out of range".into()));
        }
        let nbdev = params.nbdev.unwrap_or(1.0);
        if !nbdev.is_finite() {
            return Err(CudaVarError::InvalidInput("nbdev not finite".into()));
        }
        let nb2 = (nbdev as f32) * (nbdev as f32);

        // Compute first-valid per series
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaVarError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < period {
                return Err(CudaVarError::InvalidInput(format!(
                    "series {} insufficient tail for period {}",
                    s, period
                )));
            }
            first_valids[s] = fv as i32;
        }

        let (ps_tm, ps2_tm, pn_tm) =
            Self::build_prefixes_time_major(data_tm_f32, cols, rows, &first_valids);
        let ps_tm_ff = split_f64_to_float2_vec(&ps_tm);
        let ps2_tm_ff = split_f64_to_float2_vec(&ps2_tm);

        let mut d_ps_tm: DeviceBuffer<Float2> = unsafe { DeviceBuffer::uninitialized_async(ps_tm_ff.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let mut d_ps2_tm: DeviceBuffer<Float2> = unsafe { DeviceBuffer::uninitialized_async(ps2_tm_ff.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let mut d_pn_tm: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized_async(pn_tm.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let mut d_first: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized_async(first_valids.len(), &self.stream) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        let h_ps_tm  = LockedBuffer::from_slice(&ps_tm_ff).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let h_ps2_tm = LockedBuffer::from_slice(&ps2_tm_ff).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let h_pn_tm  = LockedBuffer::from_slice(&pn_tm).map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        let h_first  = LockedBuffer::from_slice(&first_valids).map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        unsafe {
            d_ps_tm.async_copy_from(&h_ps_tm, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
            d_ps2_tm.async_copy_from(&h_ps2_tm, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
            d_pn_tm.async_copy_from(&h_pn_tm, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
            d_first.async_copy_from(&h_first, &self.stream)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        }

        self.try_enable_persisting_l2(d_ps_tm.as_device_ptr().as_raw() as u64,
            ps_tm_ff.len() * std::mem::size_of::<Float2>());
        let mut d_out_tm = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_ps_tm,
            &d_ps2_tm,
            &d_pn_tm,
            &d_first,
            cols,
            rows,
            period,
            nb2,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    fn launch_many_series_kernel(
        &self,
        d_ps_tm: &DeviceBuffer<Float2>,
        d_ps2_tm: &DeviceBuffer<Float2>,
        d_pn_tm: &DeviceBuffer<i32>,
        d_first: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        nb2: f32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVarError> {
        let func = self
            .module
            .get_function("var_many_series_one_param_f32")
            .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        if cols > i32::MAX as usize || rows > i32::MAX as usize || period > i32::MAX as usize {
            return Err(CudaVarError::InvalidInput(
                "inputs exceed kernel limits".into(),
            ));
        }
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x; // iterate over time
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut ps_ptr = d_ps_tm.as_device_ptr().as_raw();
            let mut ps2_ptr = d_ps2_tm.as_device_ptr().as_raw();
            let mut pn_ptr = d_pn_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut nb2_f = nb2;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut ps_ptr as *mut _ as *mut c_void,
                &mut ps2_ptr as *mut _ as *mut c_void,
                &mut pn_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut nb2_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVarError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct VarBatchState {
        cuda: CudaVar,
        price: Vec<f32>,
        sweep: VarBatchRange,
    }
    impl CudaBenchState for VarBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .var_batch_dev(&self.price, &self.sweep)
                .expect("var batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaVar::new(0).expect("cuda var");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = VarBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            nbdev: (1.0, 1.0, 0.0),
        };
        Box::new(VarBatchState { cuda, price, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "var",
            "one_series_many_params",
            "var_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
