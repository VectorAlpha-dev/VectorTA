//! CUDA support for the Chande Kroll Stop (CKSP) indicator.
//!
//! Goals and parity:
//! - API mirrors ALMA wrapper patterns: PTX load with `DetermineTargetFromContext`,
//!   NON_BLOCKING stream, VRAM checks with headroom, and Y-dimension chunking.
//! - Two entry points:
//!     - one-series × many-params (batch)
//!     - many-series × one-param (time‑major)
//! - Numerics match scalar: warmup index = first_valid + p + q − 1; write NaN
//!   before warmup; ATR as RMA with α=1/p; use f32 with FMA where applicable.
//!
//! Kernels: see `kernels/cuda/cksp_kernel.cu` for symbol signatures.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::cksp::{CkspBatchRange, CkspParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaCkspError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaCkspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCkspError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCkspError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaCkspError {}

// Simple pair of device arrays for (long, short)
pub struct DeviceArrayF32Pair {
    pub long: DeviceArrayF32,
    pub short: DeviceArrayF32,
}

// Selection policy skeleton to match ALMA’s explicit/auto pattern.
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
pub struct CudaCkspPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaCkspPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

pub struct CudaCksp {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaCkspPolicy,
}

impl CudaCksp {
    pub fn new(device_id: usize) -> Result<Self, CudaCkspError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cksp_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaCkspError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        Ok(Self { module, stream, _context: context, policy: CudaCkspPolicy::default() })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaCkspPolicy) -> Result<Self, CudaCkspError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
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

    // -------- Batch: one series × many params --------
    pub fn cksp_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &CkspBatchRange,
    ) -> Result<(DeviceArrayF32Pair, Vec<CkspParams>), CudaCkspError> {
        if high.len() == 0 || low.len() != high.len() || close.len() != high.len() {
            return Err(CudaCkspError::InvalidInput(
                "inputs must be non-empty and same length".into(),
            ));
        }
        let len = close.len();
        let first_valid = close
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaCkspError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_cksp_combos(sweep);
        if combos.is_empty() {
            return Err(CudaCkspError::InvalidInput("no parameter combinations".into()));
        }
        // Param sanity and gather host vectors
        let mut p_i32 = Vec::with_capacity(combos.len());
        let mut x_f32 = Vec::with_capacity(combos.len());
        let mut q_i32 = Vec::with_capacity(combos.len());
        let mut max_q: usize = 0;
        for prm in &combos {
            let p = prm.p.unwrap_or(10);
            let q = prm.q.unwrap_or(9);
            let x = prm.x.unwrap_or(1.0) as f32;
            if p == 0 || q == 0 {
                return Err(CudaCkspError::InvalidInput("p and q must be > 0".into()));
            }
            if p > len || q > len {
                return Err(CudaCkspError::InvalidInput("p/q exceed data length".into()));
            }
            if len - first_valid < p {
                return Err(CudaCkspError::InvalidInput(
                    "not enough valid data for ATR warmup".into(),
                ));
            }
            p_i32.push(p as i32);
            q_i32.push(q as i32);
            x_f32.push(x);
            if q > max_q { max_q = q; }
        }

        // VRAM check (inputs + params + outputs)
        let in_bytes = 3 * len * std::mem::size_of::<f32>();
        let params_bytes = combos.len() * (2 * std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = 2 * combos.len() * len * std::mem::size_of::<f32>();
        let required = in_bytes + params_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaCkspError::InvalidInput(
                "insufficient device memory for cksp batch".into(),
            ));
        }

        // H2D
        let d_high = DeviceBuffer::from_slice(high).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let d_p = DeviceBuffer::from_slice(&p_i32).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let d_x = DeviceBuffer::from_slice(&x_f32).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let d_q = DeviceBuffer::from_slice(&q_i32).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_long: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let mut d_short: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        // grid.y chunking to respect 65_535
        let rows = combos.len();
        let y_limit = 65_535usize;
        let mut start = 0usize;
        while start < rows {
            let count = (rows - start).min(y_limit);
            self.launch_batch_kernel_subrange(
                &d_high,
                &d_low,
                &d_close,
                len as i32,
                first_valid as i32,
                &d_p,
                &d_x,
                &d_q,
                start,
                count,
                (max_q + 1) as i32,
                &mut d_long,
                &mut d_short,
            )?;
            start += count;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32Pair {
                long: DeviceArrayF32 { buf: d_long, rows: combos.len(), cols: len },
                short: DeviceArrayF32 { buf: d_short, rows: combos.len(), cols: len },
            },
            combos,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel_subrange(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        series_len: i32,
        first_valid: i32,
        d_p: &DeviceBuffer<i32>,
        d_x: &DeviceBuffer<f32>,
        d_q: &DeviceBuffer<i32>,
        start_row: usize,
        n_rows: usize,
        cap_max: i32,
        d_long: &mut DeviceBuffer<f32>,
        d_short: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCkspError> {
        if series_len <= 0 || n_rows == 0 || cap_max <= 1 {
            return Err(CudaCkspError::InvalidInput("invalid launch dims".into()));
        }
        let func = self
            .module
            .get_function("cksp_batch_f32")
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256u32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64).min(1024),
        };
        let grid: GridSize = (1u32, n_rows as u32, 1u32).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut hp = d_high.as_device_ptr().as_raw();
            let mut lp = d_low.as_device_ptr().as_raw();
            let mut cp = d_close.as_device_ptr().as_raw();
            let mut sl = series_len;
            let mut fv = first_valid;
            let mut pp = d_p.as_device_ptr().add(start_row).as_raw();
            let mut xp = d_x.as_device_ptr().add(start_row).as_raw();
            let mut qp = d_q.as_device_ptr().add(start_row).as_raw();
            let mut nc = n_rows as i32;
            let mut cm = cap_max;
            let mut ol = d_long.as_device_ptr().add(start_row * (series_len as usize)).as_raw();
            let mut os = d_short.as_device_ptr().add(start_row * (series_len as usize)).as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp as *mut _ as *mut c_void,
                &mut lp as *mut _ as *mut c_void,
                &mut cp as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut xp as *mut _ as *mut c_void,
                &mut qp as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut cm as *mut _ as *mut c_void,
                &mut ol as *mut _ as *mut c_void,
                &mut os as *mut _ as *mut c_void,
            ];
            // Dynamic shared memory needed for deques (cap_max per array)
            let shmem: u32 = ((4 * cap_max as usize * std::mem::size_of::<i32>())
                + (2 * cap_max as usize * std::mem::size_of::<f32>()))
                .try_into()
                .unwrap();
            self.stream
                .launch(&func, grid, block, shmem, args)
                .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // -------- Many-series × one param (time‑major) --------
    pub fn cksp_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &CkspParams,
    ) -> Result<DeviceArrayF32Pair, CudaCkspError> {
        if rows == 0 || cols == 0 {
            return Err(CudaCkspError::InvalidInput("empty dims".into()));
        }
        if high_tm.len() != rows * cols || low_tm.len() != rows * cols || close_tm.len() != rows * cols {
            return Err(CudaCkspError::InvalidInput(
                "time-major inputs must be rows*cols in length".into(),
            ));
        }
        let p = params.p.unwrap_or(10);
        let x = params.x.unwrap_or(1.0) as f32;
        let q = params.q.unwrap_or(9);
        if p == 0 || q == 0 { return Err(CudaCkspError::InvalidInput("p/q must be > 0".into())); }

        // first_valid per series from inputs (high/low/close must be finite)
        let mut first_valids = vec![cols as i32; rows];
        for s in 0..rows {
            for t in 0..cols {
                let idx = t * rows + s; // time-major stride = rows
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() && close_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let in_bytes = 3 * rows * cols * std::mem::size_of::<f32>();
        let out_bytes = 2 * rows * cols * std::mem::size_of::<f32>();
        let aux_bytes = rows * std::mem::size_of::<i32>();
        let required = in_bytes + out_bytes + aux_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaCkspError::InvalidInput("insufficient device memory".into()));
        }

        let d_high = DeviceBuffer::from_slice(high_tm).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_tm).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close_tm).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let mut d_long: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * cols) }
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let mut d_short: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * cols) }
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_close,
            &d_first,
            rows as i32,
            cols as i32,
            p as i32,
            x,
            q as i32,
            (q + 1) as i32,
            &mut d_long,
            &mut d_short,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32Pair {
            long: DeviceArrayF32 { buf: d_long, rows, cols },
            short: DeviceArrayF32 { buf: d_short, rows, cols },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: i32,
        series_len: i32,
        p: i32,
        x: f32,
        q: i32,
        cap_max: i32,
        d_long_tm: &mut DeviceBuffer<f32>,
        d_short_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCkspError> {
        let func = self
            .module
            .get_function("cksp_many_series_one_param_f32")
            .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64).min(1024),
        };
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut hp = d_high_tm.as_device_ptr().as_raw();
            let mut lp = d_low_tm.as_device_ptr().as_raw();
            let mut cp = d_close_tm.as_device_ptr().as_raw();
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut ns = num_series;
            let mut sl = series_len;
            let mut pp = p;
            let mut xx = x;
            let mut qq = q;
            let mut cm = cap_max;
            let mut ol = d_long_tm.as_device_ptr().as_raw();
            let mut os = d_short_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp as *mut _ as *mut c_void,
                &mut lp as *mut _ as *mut c_void,
                &mut cp as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut ns as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut xx as *mut _ as *mut c_void,
                &mut qq as *mut _ as *mut c_void,
                &mut cm as *mut _ as *mut c_void,
                &mut ol as *mut _ as *mut c_void,
                &mut os as *mut _ as *mut c_void,
            ];
            let shmem_usize = (4 * cap_max as usize * std::mem::size_of::<i32>())
                + (2 * cap_max as usize * std::mem::size_of::<f32>());
            let shmem: u32 = shmem_usize.try_into().unwrap_or(u32::MAX);
            self.stream
                .launch(&func, grid, block, shmem, args)
                .map_err(|e| CudaCkspError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

fn expand_cksp_combos(r: &CkspBatchRange) -> Vec<CkspParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 { v.push(x); x += step; }
        v
    }
    let ps = axis_usize(r.p);
    let xs = axis_f64(r.x);
    let qs = axis_usize(r.q);
    let mut out = Vec::with_capacity(ps.len() * xs.len() * qs.len());
    for &p in &ps { for &x in &xs { for &q in &qs {
        out.push(CkspParams { p: Some(p), x: Some(x), q: Some(q) });
    } } }
    out
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::gen_series;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_ROWS: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_ROWS * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct CkspBatchState {
        cuda: CudaCksp,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: CkspBatchRange,
    }
    impl CudaBenchState for CkspBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .cksp_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .expect("cksp batch");
        }
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() { continue; }
            let x = i as f32 * 0.0029;
            let off = (0.0015 * x.sin()).abs() + 0.35;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low, close.to_vec())
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCksp::new(0).expect("cuda cksp");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low, close) = synth_hlc_from_close(&close);
        let sweep = CkspBatchRange { p: (10, 10 + PARAM_ROWS as usize - 1, 1), x: (1.0, 1.0, 0.0), q: (9, 9, 0) };
        Box::new(CkspBatchState { cuda, high, low, close, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "cksp",
            "one_series_many_params",
            "cksp_cuda_batch_dev",
            "1m_x_128",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
