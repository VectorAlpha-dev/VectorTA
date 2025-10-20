//! CUDA scaffolding for KST (Know Sure Thing)
//!
//! Parity goals (aligned with ALMA wrapper):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/kst_kernel.ptx"))
//!   using DetermineTargetFromContext + OptLevel O2, with conservative fallbacks
//! - Stream NON_BLOCKING
//! - Warmup/NaN semantics identical to scalar
//! - VRAM checks and simple chunking where applicable

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::kst::{KstBatchRange, KstParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

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
pub struct CudaKstPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaKstPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Debug)]
pub enum CudaKstError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaKstError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaKstError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaKstError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaKstError {}

/// Pair of VRAM-backed arrays produced by the KST kernels (line + signal).
pub struct DeviceKstPair {
    pub line: DeviceArrayF32,
    pub signal: DeviceArrayF32,
}
impl DeviceKstPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.line.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.line.cols
    }
}

pub struct CudaKst {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaKstPolicy,
}

impl CudaKst {
    pub fn new(device_id: usize) -> Result<Self, CudaKstError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaKstError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kst_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                // fallbacks
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaKstError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaKstPolicy::default(),
        })
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaKstPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaKstPolicy {
        &self.policy
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaKstError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaKstError::Cuda(e.to_string()))
    }

    #[inline]
    fn device_mem_ok(bytes: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => bytes.saturating_add(64 * 1024 * 1024) <= free,
            Err(_) => true,
        }
    }

    // ------------- Batch (one series × many params) -----------------

    fn expand_grid(range: &KstBatchRange) -> Vec<KstParams> {
        fn axis(t: (usize, usize, usize)) -> Vec<usize> {
            if t.2 == 0 || t.0 == t.1 {
                vec![t.0]
            } else {
                (t.0..=t.1).step_by(t.2).collect()
            }
        }
        let s1 = axis(range.sma_period1);
        let s2 = axis(range.sma_period2);
        let s3 = axis(range.sma_period3);
        let s4 = axis(range.sma_period4);
        let r1 = axis(range.roc_period1);
        let r2 = axis(range.roc_period2);
        let r3 = axis(range.roc_period3);
        let r4 = axis(range.roc_period4);
        let sg = axis(range.signal_period);
        let mut out = Vec::with_capacity(
            s1.len()
                * s2.len()
                * s3.len()
                * s4.len()
                * r1.len()
                * r2.len()
                * r3.len()
                * r4.len()
                * sg.len(),
        );
        for &a in &s1 {
            for &b in &s2 {
                for &c in &s3 {
                    for &d in &s4 {
                        for &e in &r1 {
                            for &f in &r2 {
                                for &g in &r3 {
                                    for &h in &r4 {
                                        for &q in &sg {
                                            out.push(KstParams {
                                                sma_period1: Some(a),
                                                sma_period2: Some(b),
                                                sma_period3: Some(c),
                                                sma_period4: Some(d),
                                                roc_period1: Some(e),
                                                roc_period2: Some(f),
                                                roc_period3: Some(g),
                                                roc_period4: Some(h),
                                                signal_period: Some(q),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    pub fn kst_batch_dev(
        &self,
        prices: &[f32],
        sweep: &KstBatchRange,
    ) -> Result<(DeviceKstPair, Vec<KstParams>), CudaKstError> {
        if prices.is_empty() {
            return Err(CudaKstError::InvalidInput("empty price input".into()));
        }
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaKstError::InvalidInput("empty parameter sweep".into()));
        }
        let len = prices.len();
        let first_valid = (0..len)
            .find(|&i| !prices[i].is_nan())
            .ok_or_else(|| CudaKstError::InvalidInput("all values are NaN".into()))?;

        // Validate: need at least warm_sig samples available
        let mut max_warm_line = 0usize;
        let mut max_sig = 0usize;
        for c in &combos {
            let wl = (c.roc_period1.unwrap() + c.sma_period1.unwrap() - 1)
                .max(c.roc_period2.unwrap() + c.sma_period2.unwrap() - 1)
                .max(c.roc_period3.unwrap() + c.sma_period3.unwrap() - 1)
                .max(c.roc_period4.unwrap() + c.sma_period4.unwrap() - 1);
            let ws = wl + c.signal_period.unwrap() - 1;
            if wl > max_warm_line {
                max_warm_line = wl;
            }
            if ws > max_sig {
                max_sig = ws;
            }
        }
        if len - first_valid <= max_warm_line {
            return Err(CudaKstError::InvalidInput(
                "not enough valid data for KST warmup".into(),
            ));
        }

        let rows = combos.len();
        let req_bytes = len * 4 + 9 * rows * 4 + 2 * rows * len * 4 + 64 * 1024 * 1024; // prices + periods + outputs + headroom
        if !Self::device_mem_ok(req_bytes) {
            return Err(CudaKstError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(prices, &self.stream) }
                .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        // Compact period arrays
        let to_i32 = |v: usize| -> i32 { v as i32 };
        let s1: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.sma_period1.unwrap()))
            .collect();
        let s2: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.sma_period2.unwrap()))
            .collect();
        let s3: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.sma_period3.unwrap()))
            .collect();
        let s4: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.sma_period4.unwrap()))
            .collect();
        let r1: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.roc_period1.unwrap()))
            .collect();
        let r2: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.roc_period2.unwrap()))
            .collect();
        let r3: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.roc_period3.unwrap()))
            .collect();
        let r4: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.roc_period4.unwrap()))
            .collect();
        let sg: Vec<i32> = combos
            .iter()
            .map(|c| to_i32(c.signal_period.unwrap()))
            .collect();
        let d_s1 = DeviceBuffer::from_slice(&s1).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_s2 = DeviceBuffer::from_slice(&s2).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_s3 = DeviceBuffer::from_slice(&s3).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_s4 = DeviceBuffer::from_slice(&s4).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_r1 = DeviceBuffer::from_slice(&r1).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_r2 = DeviceBuffer::from_slice(&r2).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_r3 = DeviceBuffer::from_slice(&r3).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_r4 = DeviceBuffer::from_slice(&r4).map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_sg = DeviceBuffer::from_slice(&sg).map_err(|e| CudaKstError::Cuda(e.to_string()))?;

        let mut d_line: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let mut d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaKstError::Cuda(e.to_string()))?;

        self.launch_batch(
            &d_prices,
            &d_s1,
            &d_s2,
            &d_s3,
            &d_s4,
            &d_r1,
            &d_r2,
            &d_r3,
            &d_r4,
            &d_sg,
            len,
            rows,
            first_valid,
            &mut d_line,
            &mut d_signal,
        )?;
        self.synchronize()?;

        let pair = DeviceKstPair {
            line: DeviceArrayF32 {
                buf: d_line,
                rows,
                cols: len,
            },
            signal: DeviceArrayF32 {
                buf: d_signal,
                rows,
                cols: len,
            },
        };
        Ok((pair, combos))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_s1: &DeviceBuffer<i32>,
        d_s2: &DeviceBuffer<i32>,
        d_s3: &DeviceBuffer<i32>,
        d_s4: &DeviceBuffer<i32>,
        d_r1: &DeviceBuffer<i32>,
        d_r2: &DeviceBuffer<i32>,
        d_r3: &DeviceBuffer<i32>,
        d_r4: &DeviceBuffer<i32>,
        d_sig: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_line: &mut DeviceBuffer<f32>,
        d_signal: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKstError> {
        let func = self
            .module
            .get_function("kst_batch_f32")
            .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p0 = d_prices.as_device_ptr().as_raw();
            let mut s1 = d_s1.as_device_ptr().as_raw();
            let mut s2 = d_s2.as_device_ptr().as_raw();
            let mut s3 = d_s3.as_device_ptr().as_raw();
            let mut s4 = d_s4.as_device_ptr().as_raw();
            let mut r1 = d_r1.as_device_ptr().as_raw();
            let mut r2 = d_r2.as_device_ptr().as_raw();
            let mut r3 = d_r3.as_device_ptr().as_raw();
            let mut r4 = d_r4.as_device_ptr().as_raw();
            let mut sg = d_sig.as_device_ptr().as_raw();
            let mut sl = series_len as i32;
            let mut nc = n_combos as i32;
            let mut fv = first_valid as i32;
            let mut out_l = d_line.as_device_ptr().as_raw();
            let mut out_s = d_signal.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut p0 as *mut _ as *mut c_void,
                &mut s1 as *mut _ as *mut c_void,
                &mut s2 as *mut _ as *mut c_void,
                &mut s3 as *mut _ as *mut c_void,
                &mut s4 as *mut _ as *mut c_void,
                &mut r1 as *mut _ as *mut c_void,
                &mut r2 as *mut _ as *mut c_void,
                &mut r3 as *mut _ as *mut c_void,
                &mut r4 as *mut _ as *mut c_void,
                &mut sg as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut out_l as *mut _ as *mut c_void,
                &mut out_s as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // ------------- Many series × one param (time-major) -----------------
    pub fn kst_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &KstParams,
    ) -> Result<DeviceKstPair, CudaKstError> {
        if cols == 0 || rows == 0 {
            return Err(CudaKstError::InvalidInput(
                "cols/rows must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaKstError::InvalidInput(
                "time-major buffer size mismatch".into(),
            ));
        }
        let s1 = params.sma_period1.unwrap_or(10);
        let s2 = params.sma_period2.unwrap_or(10);
        let s3 = params.sma_period3.unwrap_or(10);
        let s4 = params.sma_period4.unwrap_or(15);
        let r1 = params.roc_period1.unwrap_or(10);
        let r2 = params.roc_period2.unwrap_or(15);
        let r3 = params.roc_period3.unwrap_or(20);
        let r4 = params.roc_period4.unwrap_or(30);
        let sig = params.signal_period.unwrap_or(9);

        // First-valid per series
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                fv = rows as i32;
            }
            first_valids[s] = fv;
        }

        let req = data_tm_f32.len() * 4 + cols * 4 + 2 * data_tm_f32.len() * 4 + 64 * 1024 * 1024;
        if !Self::device_mem_ok(req) {
            return Err(CudaKstError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_prices_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
                .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let mut d_line_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let mut d_sig_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaKstError::Cuda(e.to_string()))?;

        self.launch_many_series(
            &d_prices_tm,
            cols,
            rows,
            s1,
            s2,
            s3,
            s4,
            r1,
            r2,
            r3,
            r4,
            sig,
            &d_first,
            &mut d_line_tm,
            &mut d_sig_tm,
        )?;
        self.synchronize()?;

        Ok(DeviceKstPair {
            line: DeviceArrayF32 {
                buf: d_line_tm,
                rows,
                cols,
            },
            signal: DeviceArrayF32 {
                buf: d_sig_tm,
                rows,
                cols,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        s1: usize,
        s2: usize,
        s3: usize,
        s4: usize,
        r1: usize,
        r2: usize,
        r3: usize,
        r4: usize,
        sig: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_line_tm: &mut DeviceBuffer<f32>,
        d_sig_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKstError> {
        let func = self
            .module
            .get_function("kst_many_series_one_param_f32")
            .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p = d_prices_tm.as_device_ptr().as_raw();
            let mut ns = cols as i32;
            let mut sl = rows as i32;
            let mut s1_ = s1 as i32;
            let mut s2_ = s2 as i32;
            let mut s3_ = s3 as i32;
            let mut s4_ = s4 as i32;
            let mut r1_ = r1 as i32;
            let mut r2_ = r2 as i32;
            let mut r3_ = r3 as i32;
            let mut r4_ = r4 as i32;
            let mut sig_ = sig as i32;
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut out_l = d_line_tm.as_device_ptr().as_raw();
            let mut out_s = d_sig_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p as *mut _ as *mut c_void,
                &mut ns as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut s1_ as *mut _ as *mut c_void,
                &mut s2_ as *mut _ as *mut c_void,
                &mut s3_ as *mut _ as *mut c_void,
                &mut s4_ as *mut _ as *mut c_void,
                &mut r1_ as *mut _ as *mut c_void,
                &mut r2_ as *mut _ as *mut c_void,
                &mut r3_ as *mut _ as *mut c_void,
                &mut r4_ as *mut _ as *mut c_void,
                &mut sig_ as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut out_l as *mut _ as *mut c_void,
                &mut out_s as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaKstError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// ---------------- Benches ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 125; // keep memory reasonable (2 outputs)
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * 4;
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * 4 * 2; // line + signal
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * 4;
        let out_bytes = elems * 4 * 2; // line + signal
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaKst,
        price: Vec<f32>,
        sweep: KstBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .kst_batch_dev(&self.price, &self.sweep)
                .expect("kst batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKst::new(0).expect("cuda kst");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = KstBatchRange {
            sma_period1: (10, 10 + PARAM_SWEEP - 1, 1),
            sma_period2: (10, 10, 0),
            sma_period3: (10, 10, 0),
            sma_period4: (15, 15, 0),
            roc_period1: (10, 10, 0),
            roc_period2: (15, 15, 0),
            roc_period3: (20, 20, 0),
            roc_period4: (30, 30, 0),
            signal_period: (9, 9, 0),
        };
        Box::new(BatchState { cuda, price, sweep })
    }

    struct ManyState {
        cuda: CudaKst,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: KstParams,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .kst_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("kst many");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaKst::new(0).expect("cuda kst");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = KstParams {
            sma_period1: Some(10),
            sma_period2: Some(10),
            sma_period3: Some(10),
            sma_period4: Some(15),
            roc_period1: Some(10),
            roc_period2: Some(15),
            roc_period3: Some(20),
            roc_period4: Some(30),
            signal_period: Some(9),
        };
        Box::new(ManyState {
            cuda,
            data_tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "kst",
                "one_series_many_params",
                "kst_cuda_batch_dev",
                "1m_x_125",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "kst",
                "many_series_one_param",
                "kst_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(6)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
