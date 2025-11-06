//! CUDA wrapper for StochF (Fast Stochastic Oscillator: %K and %D).
//!
//! Mirrors ALMA/KDJ integration patterns:
//! - PTX loaded via include_str!(concat!(env!("OUT_DIR"), "/stochf_kernel.ptx"))
//! - Stream NON_BLOCKING; JIT options: DetermineTargetFromContext + O2 with fallbacks
//! - VRAM checks with ~64MB headroom; chunk rows to <= 65_535
//! - Batch (one series × many-params) and many-series (time-major, one-param)
//! - Batch uses WILLR sparse tables for HH/LL queries (shared precompute across rows)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::stochf::{StochfBatchRange, StochfParams};
use crate::indicators::willr::build_willr_gpu_tables;
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaStochfError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaStochfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaStochfError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaStochfError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaStochfError {}

pub struct DeviceArrayF32Pair {
    pub a: DeviceArrayF32,
    pub b: DeviceArrayF32,
}
impl DeviceArrayF32Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.a.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.a.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self { Self::Auto }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self { Self::Auto }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaStochfPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaStochf {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaStochfPolicy,
}

impl CudaStochf {
    pub fn new(device_id: usize) -> Result<Self, CudaStochfError> {
        Self::new_with_policy(device_id, CudaStochfPolicy::default())
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaStochfPolicy,
    ) -> Result<Self, CudaStochfError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/stochf_kernel.ptx"));
        let jit = [
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, &jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        Ok(Self {
            module,
            stream,
            _context: context,
            policy,
        })
    }

    #[inline]
    fn mem_ok(bytes: usize, headroom: usize) -> bool {
        if env::var("CUDA_MEM_CHECK")
            .ok()
            .filter(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .is_some()
        {
            return true;
        }
        mem_get_info()
            .map(|(free, _)| bytes.saturating_add(headroom) <= free)
            .unwrap_or(true)
    }

    // ---- Batch: one series × many params ----
    pub fn stochf_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &StochfBatchRange,
    ) -> Result<(DeviceArrayF32Pair, Vec<StochfParams>), CudaStochfError> {
        if high_f32.len() != low_f32.len() || high_f32.len() != close_f32.len() {
            return Err(CudaStochfError::InvalidInput("length mismatch".into()));
        }
        let len = high_f32.len();
        if len == 0 {
            return Err(CudaStochfError::InvalidInput("empty input".into()));
        }

        // Expand parameter grid
        fn axis(a: (usize, usize, usize)) -> Vec<usize> {
            let (s, e, st) = a;
            if st == 0 || s == e {
                vec![s]
            } else {
                (s..=e).step_by(st).collect()
            }
        }
        let fastks = axis(sweep.fastk_period);
        let fastds = axis(sweep.fastd_period);
        let mut combos = Vec::<StochfParams>::with_capacity(fastks.len() * fastds.len());
        for &k in &fastks {
            for &d in &fastds {
                combos.push(StochfParams {
                    fastk_period: Some(k),
                    fastd_period: Some(d),
                    fastd_matype: Some(0),
                });
            }
        }
        if combos.is_empty() {
            return Err(CudaStochfError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // first_valid
        let first_valid = (0..len)
            .find(|&i| {
                high_f32[i].is_finite() && low_f32[i].is_finite() && close_f32[i].is_finite()
            })
            .ok_or_else(|| CudaStochfError::InvalidInput("all values NaN".into()))?;
        // Check sufficient data for max fastk
        let max_fk = combos
            .iter()
            .map(|p| p.fastk_period.unwrap())
            .max()
            .unwrap();
        if len - first_valid < max_fk {
            return Err(CudaStochfError::InvalidInput(
                "insufficient data after first_valid".into(),
            ));
        }
        let max_fk = combos
            .iter()
            .map(|p| p.fastk_period.unwrap())
            .max()
            .unwrap();
        if len - first_valid < max_fk {
            return Err(CudaStochfError::InvalidInput(
                "insufficient data after first_valid".into(),
            ));
        }

        // ── VRAM estimate (device): close + WILLR tables + param arrays + outputs
        // We do NOT upload high/low for the batch path.
        let rows = combos.len();
        let in_bytes_close = len * std::mem::size_of::<f32>();
        let params_bytes   = 3 * rows * std::mem::size_of::<i32>();
        let out_bytes      = 2 * rows * len * std::mem::size_of::<f32>();
        // Rough WILLR table upper bound (host->device): keep as ballpark
        let tables_overhead = 8 * len * std::mem::size_of::<f32>();
        let required = in_bytes_close + tables_overhead + params_bytes + out_bytes;
        if !Self::mem_ok(required, 64 * 1024 * 1024) {
            return Err(CudaStochfError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0*1024.0)
            )));
        }

        // Build WILLR tables once on host (reuse across rows)
        let tables = build_willr_gpu_tables(high_f32, low_f32);

        // Precise VRAM check using actual table sizes
        let tables_bytes =
            tables.log2.len() * std::mem::size_of::<i32>() +
            tables.level_offsets.len() * std::mem::size_of::<i32>() +
            tables.st_max.len() * std::mem::size_of::<f32>() +
            tables.st_min.len() * std::mem::size_of::<f32>() +
            tables.nan_psum.len() * std::mem::size_of::<i32>();
        let required_exact = (len * std::mem::size_of::<f32>()) + tables_bytes + params_bytes + out_bytes;
        if !Self::mem_ok(required_exact, 64 * 1024 * 1024) {
            return Err(CudaStochfError::InvalidInput(format!(
                "device memory required {:.2} MB exceeds free VRAM",
                required_exact as f64 / (1024.0*1024.0)
            )));
        }

        // Upload ONLY close (kernel does not deref high/low in batch path)
        let use_pinned = len >= 131_072;
        let mut pinned_close: Option<LockedBuffer<f32>> = None;
        let d_close = if use_pinned {
            let host = LockedBuffer::from_slice(close_f32).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            let mut dc = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
                .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            unsafe { dc.async_copy_from(&host, &self.stream) }.map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            pinned_close = Some(host);
            dc
        } else {
            DeviceBuffer::from_slice(close_f32).map_err(|e| CudaStochfError::Cuda(e.to_string()))?
        };

        // Upload WILLR tables
        let d_log2 = DeviceBuffer::from_slice(&tables.log2)
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let d_offs = DeviceBuffer::from_slice(&tables.level_offsets)
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let d_st_max = DeviceBuffer::from_slice(&tables.st_max)
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let d_st_min = DeviceBuffer::from_slice(&tables.st_min)
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let d_nan_ps = DeviceBuffer::from_slice(&tables.nan_psum)
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;

        // Allocate outputs
        let mut d_k = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let mut d_d = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;

        // Flatten param arrays once, upload once, then offset per launch
        let fk_host: Vec<i32> = combos.iter().map(|p| p.fastk_period.unwrap() as i32).collect();
        let fd_host: Vec<i32> = combos.iter().map(|p| p.fastd_period.unwrap() as i32).collect();
        let mt_host: Vec<i32> = combos.iter().map(|p| p.fastd_matype.unwrap_or(0) as i32).collect();

        let d_fk_all = DeviceBuffer::from_slice(&fk_host).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let d_fd_all = DeviceBuffer::from_slice(&fd_host).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let d_mt_all = DeviceBuffer::from_slice(&mt_host).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;

        // Prepare kernel
        let mut func: Function = self.module
            .get_function("stochf_batch_f32")
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        let combos_per_launch = 65_535usize; // grid.x limit
        let mut row0 = 0usize;
        while row0 < rows {
            let n = (rows - row0).min(combos_per_launch);
            let grid: GridSize = (n as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                // Kernel ignores high/low; pass null device pointers
                let mut high_ptr: u64 = 0;
                let mut low_ptr:  u64 = 0;
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut log2_ptr = d_log2.as_device_ptr().as_raw();
                let mut offs_ptr = d_offs.as_device_ptr().as_raw();
                let mut stmax_ptr = d_st_max.as_device_ptr().as_raw();
                let mut stmin_ptr = d_st_min.as_device_ptr().as_raw();
                let mut npsum_ptr = d_nan_ps.as_device_ptr().as_raw();
                let mut fk_ptr    = d_fk_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut fd_ptr    = d_fd_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut mt_ptr    = d_mt_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut len_i     = len as i32;
                let mut first_i   = first_valid as i32;
                let mut levels_i  = (tables.level_offsets.len() as i32);
                let mut n_i       = n as i32;
                let mut k_out_ptr = d_k.as_device_ptr().offset((row0 * len) as isize).as_raw();
                let mut d_out_ptr = d_d.as_device_ptr().offset((row0 * len) as isize).as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut log2_ptr as *mut _ as *mut c_void,
                    &mut offs_ptr as *mut _ as *mut c_void,
                    &mut stmax_ptr as *mut _ as *mut c_void,
                    &mut stmin_ptr as *mut _ as *mut c_void,
                    &mut npsum_ptr as *mut _ as *mut c_void,
                    &mut fk_ptr as *mut _ as *mut c_void,
                    &mut fd_ptr as *mut _ as *mut c_void,
                    &mut mt_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut levels_i as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut k_out_ptr as *mut _ as *mut c_void,
                    &mut d_out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            }
            row0 += n;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32Pair {
                a: DeviceArrayF32 { buf: d_k, rows, cols: len },
                b: DeviceArrayF32 { buf: d_d, rows, cols: len },
            },
            combos,
        ))
    }

    // ---- Many-series: time-major, one param ----
    pub fn stochf_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &StochfParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaStochfError> {
        if cols == 0 || rows == 0 {
            return Err(CudaStochfError::InvalidInput(
                "series dims must be positive".into(),
            ));
        }
        if high_tm_f32.len() != cols * rows
            || low_tm_f32.len() != cols * rows
            || close_tm_f32.len() != cols * rows
        {
            return Err(CudaStochfError::InvalidInput(
                "time-major slices mismatch dims".into(),
            ));
        }
        let fk = params.fastk_period.unwrap_or(5);
        let fd = params.fastd_period.unwrap_or(3);
        let mt = params.fastd_matype.unwrap_or(0);

        // first_valid per series
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm_f32[idx].is_finite()
                    && low_tm_f32[idx].is_finite()
                    && close_tm_f32[idx].is_finite()
                {
                    fv = Some(t as i32);
                    break;
                }
            }
            let f =
                fv.ok_or_else(|| CudaStochfError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (f as usize) < fk {
                return Err(CudaStochfError::InvalidInput(format!(
                    "series {} insufficient data for fk {}",
                    s, fk
                )));
            }
            first_valids[s] = f;
        }

        // Upload inputs (use pinned host and async copies when large)
        let tm_len = cols * rows;
        let mut pinned_h: Option<LockedBuffer<f32>> = None;
        let mut pinned_l: Option<LockedBuffer<f32>> = None;
        let mut pinned_c: Option<LockedBuffer<f32>> = None;
        let (d_h, d_l, d_c) = if tm_len >= 131_072 {
            let h_h = LockedBuffer::from_slice(high_tm_f32).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            let h_l = LockedBuffer::from_slice(low_tm_f32 ).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            let h_c = LockedBuffer::from_slice(close_tm_f32 ).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            let mut d_h = unsafe { DeviceBuffer::<f32>::uninitialized_async(tm_len, &self.stream) }
                .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            let mut d_l = unsafe { DeviceBuffer::<f32>::uninitialized_async(tm_len, &self.stream) }
                .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            let mut d_c = unsafe { DeviceBuffer::<f32>::uninitialized_async(tm_len, &self.stream) }
                .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            unsafe { d_h.async_copy_from(&h_h, &self.stream) }.map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            unsafe { d_l.async_copy_from(&h_l, &self.stream) }.map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            unsafe { d_c.async_copy_from(&h_c, &self.stream) }.map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
            pinned_h = Some(h_h);
            pinned_l = Some(h_l);
            pinned_c = Some(h_c);
            (d_h, d_l, d_c)
        } else {
            (
                DeviceBuffer::from_slice(high_tm_f32).map_err(|e| CudaStochfError::Cuda(e.to_string()))?,
                DeviceBuffer::from_slice(low_tm_f32 ).map_err(|e| CudaStochfError::Cuda(e.to_string()))?,
                DeviceBuffer::from_slice(close_tm_f32 ).map_err(|e| CudaStochfError::Cuda(e.to_string()))?,
            )
        };
        let d_fv= DeviceBuffer::from_slice(&first_valids ).map_err(|e| CudaStochfError::Cuda(e.to_string()))?;

        let mut d_k = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let mut d_d = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;

        // Prepare kernel
        let mut func: Function = self.module
            .get_function("stochf_many_series_one_param_f32")
            .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut h_ptr = d_h.as_device_ptr().as_raw();
            let mut l_ptr = d_l.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut fk_i = fk as i32;
            let mut fd_i = fd as i32;
            let mut mt_i = mt as i32;
            let mut ko_ptr = d_k.as_device_ptr().as_raw();
            let mut do_ptr = d_d.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut fk_i as *mut _ as *mut c_void,
                &mut fd_i as *mut _ as *mut c_void,
                &mut mt_i as *mut _ as *mut c_void,
                &mut ko_ptr as *mut _ as *mut c_void,
                &mut do_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)
                .map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        }

        // Ensure results are ready (match batch path semantics) and keep pinned buffers alive
        self.stream.synchronize().map_err(|e| CudaStochfError::Cuda(e.to_string()))?;
        drop(pinned_h); drop(pinned_l); drop(pinned_c);

        Ok((
            DeviceArrayF32 { buf: d_k, rows, cols },
            DeviceArrayF32 { buf: d_d, rows, cols },
        ))
    }
}

// ------------------- Benches -------------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::stochf::StochfBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 256; // sweep fastk only

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() { continue; }
            let x = i as f32 * 0.0019;
            let off = (0.0031 * x.sin()).abs() + 0.08;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    fn bytes_one_series_many_params() -> usize {
        // Batch kernel reads only close on device (HH/LL via WILLR tables)
        let in_bytes = 1 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct StochfBatchState {
        cuda: CudaStochf,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: StochfBatchRange,
    }
    impl CudaBenchState for StochfBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .stochf_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .expect("stochf batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaStochf::new(0).expect("cuda stochf");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        // Sweep fast_k only; keep fast_d fixed to 3
        let sweep = StochfBatchRange {
            fastk_period: (5, 5 + PARAM_SWEEP - 1, 1),
            fastd_period: (3, 3, 0),
        };
        Box::new(StochfBatchState {
            cuda,
            high,
            low,
            close,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "stochf",
            "one_series_many_params",
            "stochf_cuda_batch_dev",
            "1m_x_256",
            prep_one_series_many_params,
        )
        .with_mem_required(bytes_one_series_many_params())
        .with_sample_size(10)]
    }
}
