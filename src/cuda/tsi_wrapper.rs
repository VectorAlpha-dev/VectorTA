//! CUDA scaffolding for True Strength Index (TSI)
//!
//! Parity with ALMA/CWMA wrappers:
//! - NON_BLOCKING stream, PTX loaded with DetermineTargetFromContext + OptLevel(O2) fallbacks
//! - Policy enums for batch and many-series kernel selection
//! - VRAM checks with ~64MB headroom; chunking not needed for 1D grids here
//! - Warmup/NaN semantics identical to scalar `src/indicators/tsi.rs`

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::{
    BatchKernelPolicy, BatchKernelSelected, ManySeriesKernelPolicy, ManySeriesKernelSelected,
};
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::tsi::{TsiBatchRange, TsiParams};
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
pub enum CudaTsiError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaTsiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTsiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTsiError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaTsiError {}

#[derive(Clone, Copy, Debug)]
pub struct CudaTsiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaTsiPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaTsi {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaTsiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // Device scratch reused across invocations (fast path)
    scratch: Option<TsiScratch>,
}

impl CudaTsi {
    pub fn new(device_id: usize) -> Result<Self, CudaTsiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/tsi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaTsiError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaTsiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            scratch: None,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaTsiPolicy) {
        self.policy = p;
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaTsiError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))
    }

    // ---------------- One-series × many-params (batch) ----------------
    pub fn tsi_batch_dev(
        &mut self,
        prices_f32: &[f32],
        sweep: &TsiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<TsiParams>), CudaTsiError> {
        if prices_f32.is_empty() {
            return Err(CudaTsiError::InvalidInput("empty input".into()));
        }
        let len = prices_f32.len();
        let first_valid = prices_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaTsiError::InvalidInput("all values are NaN/inf".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaTsiError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        // Validate and marshal params
        let mut longs_i32 = Vec::<i32>::with_capacity(combos.len());
        let mut shorts_i32 = Vec::<i32>::with_capacity(combos.len());
        for p in &combos {
            let l = p.long_period.unwrap_or(25);
            let s = p.short_period.unwrap_or(13);
            if l == 0 || s == 0 || l > len || s > len {
                return Err(CudaTsiError::InvalidInput("invalid period in combo".into()));
            }
            let needed = 1 + l + s;
            if len - first_valid < needed {
                return Err(CudaTsiError::InvalidInput(format!(
                    "not enough valid data: need {}, have {}",
                    needed,
                    len - first_valid
                )));
            }
            longs_i32.push(l as i32);
            shorts_i32.push(s as i32);
        }

        // VRAM estimates: choose between plain row-major path and fast path (TM + transpose)
        let in_bytes = len * std::mem::size_of::<f32>();
        let params_bytes = 2 * combos.len() * std::mem::size_of::<i32>();
        let out_bytes = combos.len() * len * std::mem::size_of::<f32>();
        let plain_required = in_bytes + params_bytes + out_bytes;
        let fast_extra = (2 * len + (len * combos.len())) * std::mem::size_of::<f32>(); // mom, amom, out_tm
        let fast_required = plain_required + fast_extra;
        let head = 64usize * 1024 * 1024;
        let (mut free_ok_plain, mut free_ok_fast) = (true, true);
        if let Ok((free, _)) = mem_get_info() {
            free_ok_plain = plain_required.saturating_add(head) <= free;
            free_ok_fast = fast_required.saturating_add(head) <= free;
            if !free_ok_plain {
                return Err(CudaTsiError::InvalidInput("insufficient VRAM".into()));
            }
        }

        let d_prices =
            DeviceBuffer::from_slice(prices_f32).map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let d_longs =
            DeviceBuffer::from_slice(&longs_i32).map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let d_shorts =
            DeviceBuffer::from_slice(&shorts_i32).map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(combos.len() * len)
                .map_err(|e| CudaTsiError::Cuda(e.to_string()))?
        };

        // Heuristic: prefer fast param-parallel path on larger sweeps if VRAM allows
        let prefer_fast = combos.len() >= 32 && len >= 4_096 && free_ok_fast;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };

        if prefer_fast {
            self.ensure_scratch(len, combos.len())?;
            if self.scratch.is_some() {
                // Temporarily take scratch to avoid simultaneous &mut borrows of self
                let mut s = self.scratch.take().unwrap();
                // 1) momentum precompute
                self.launch_prepare_momentum(
                    &d_prices,
                    len,
                    first_valid,
                    &mut s.mom,
                    &mut s.amom,
                )?;
                // 2) param-parallel time-major compute
                self.launch_param_parallel_tm(
                    &s.mom,
                    &s.amom,
                    &d_longs,
                    &d_shorts,
                    len,
                    first_valid,
                    combos.len(),
                    &mut s.out_tm,
                    block_x,
                )?;
                // 3) transpose into row-major layout expected by callers
                self.launch_transpose_tm_to_rm(&s.out_tm, len, combos.len(), &mut d_out)?;
                // Put scratch back
                self.scratch = Some(s);
            }
            self.last_batch = Some(BatchKernelSelected::Plain { block_x });
            self.maybe_log_batch_debug();
            self.synchronize()?;
        } else {
            // Fallback: legacy row-major kernel
            self.launch_batch_kernel(
                &d_prices,
                &d_longs,
                &d_shorts,
                len,
                first_valid,
                combos.len(),
                &mut d_out,
            )?;
            self.synchronize()?;
        }
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    fn launch_batch_kernel(
        &mut self,
        d_prices: &DeviceBuffer<f32>,
        d_longs: &DeviceBuffer<i32>,
        d_shorts: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsiError> {
        if len == 0 || n_combos == 0 {
            return Ok(());
        }
        if len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || first_valid > i32::MAX as usize
        {
            return Err(CudaTsiError::InvalidInput(
                "inputs exceed kernel limits".into(),
            ));
        }
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.last_batch = Some(BatchKernelSelected::Plain { block_x });
        self.maybe_log_batch_debug();

        let func = self
            .module
            .get_function("tsi_batch_f32")
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;

        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut l_ptr = d_longs.as_device_ptr().as_raw();
            let mut s_ptr = d_shorts.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn maybe_log_batch_debug(&mut self) {
        if !self.debug_batch_logged && env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[TSI CUDA] batch kernel selected: {:?}", sel);
                self.debug_batch_logged = true;
            }
        }
    }

    fn maybe_log_many_debug(&mut self) {
        if !self.debug_many_logged && env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[TSI CUDA] many-series kernel selected: {:?}", sel);
                self.debug_many_logged = true;
            }
        }
    }

    // ------------- Many-series × one-param (time-major) -------------
    pub fn tsi_many_series_one_param_time_major_dev(
        &mut self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        long_period: usize,
        short_period: usize,
    ) -> Result<DeviceArrayF32, CudaTsiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTsiError::InvalidInput("cols/rows zero".into()));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaTsiError::InvalidInput("matrix size mismatch".into()));
        }
        if long_period == 0 || short_period == 0 {
            return Err(CudaTsiError::InvalidInput("periods must be > 0".into()));
        }

        // Per-series first_valids
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let v = prices_tm_f32[t * cols + s];
                if v.is_finite() {
                    fv = t as i32;
                    break;
                }
            }
            if (rows as i32) - fv < (1 + long_period + short_period) as i32 {
                return Err(CudaTsiError::InvalidInput(format!(
                    "series {} insufficient data for long+short={}, have {}",
                    s,
                    long_period + short_period + 1,
                    (rows as i32) - fv
                )));
            }
            first_valids[s] = fv;
        }

        // VRAM: inputs + first_valids + output
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTsiError::InvalidInput("overflow".into()))?;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let fv_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        if let Ok((free, _)) = mem_get_info() {
            let head = 64usize * 1024 * 1024;
            let required = in_bytes + fv_bytes + out_bytes;
            if required.saturating_add(head) > free {
                return Err(CudaTsiError::InvalidInput("insufficient VRAM".into()));
            }
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(elems).map_err(|e| CudaTsiError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices_tm,
            cols,
            rows,
            long_period,
            short_period,
            &d_first,
            &mut d_out_tm,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    fn launch_many_series_kernel(
        &mut self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        long_period: usize,
        short_period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsiError> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }
        if cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaTsiError::InvalidInput(
                "inputs exceed kernel limits".into(),
            ));
        }
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 128,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        self.maybe_log_many_debug();

        let func = self
            .module
            .get_function("tsi_many_series_one_param_f32")
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;

        unsafe {
            let mut p_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut l_i = long_period as i32;
            let mut s_i = short_period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut l_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// -------- Scratch arena and new kernel launchers --------

struct TsiScratch {
    mom: DeviceBuffer<f32>,
    amom: DeviceBuffer<f32>,
    out_tm: DeviceBuffer<f32>,
    len_cap: usize,
    combos_cap: usize,
}

impl CudaTsi {
    fn ensure_scratch(&mut self, len: usize, combos: usize) -> Result<(), CudaTsiError> {
        let need_new = match &self.scratch {
            None => true,
            Some(s) => s.len_cap < len || s.combos_cap < combos,
        };
        if !need_new {
            return Ok(());
        }
        let mom = unsafe { DeviceBuffer::<f32>::uninitialized(len) }
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let amom = unsafe { DeviceBuffer::<f32>::uninitialized(len) }
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let out_tm = unsafe { DeviceBuffer::<f32>::uninitialized(len * combos) }
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        self.scratch = Some(TsiScratch {
            mom,
            amom,
            out_tm,
            len_cap: len,
            combos_cap: combos,
        });
        Ok(())
    }

    fn launch_prepare_momentum(
        &mut self,
        d_prices: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_mom: &mut DeviceBuffer<f32>,
        d_amom: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsiError> {
        let func = self
            .module
            .get_function("tsi_prepare_momentum_f32")
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut mom_ptr = d_mom.as_device_ptr().as_raw();
            let mut amom_ptr = d_amom.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut mom_ptr as *mut _ as *mut c_void,
                &mut amom_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (1u32, 1u32, 1u32).into();
            let block: BlockSize = (1u32, 1u32, 1u32).into();
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn launch_param_parallel_tm(
        &mut self,
        d_mom: &DeviceBuffer<f32>,
        d_amom: &DeviceBuffer<f32>,
        d_longs: &DeviceBuffer<i32>,
        d_shorts: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
        block_x: u32,
    ) -> Result<(), CudaTsiError> {
        let func = self
            .module
            .get_function("tsi_one_series_many_params_tm_f32")
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        unsafe {
            let mut mom_ptr = d_mom.as_device_ptr().as_raw();
            let mut amom_ptr = d_amom.as_device_ptr().as_raw();
            let mut l_ptr = d_longs.as_device_ptr().as_raw();
            let mut s_ptr = d_shorts.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut mom_ptr as *mut _ as *mut c_void,
                &mut amom_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x.max(1), 1u32, 1u32).into();
            let block: BlockSize = (block_x, 1u32, 1u32).into();
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn launch_transpose_tm_to_rm(
        &mut self,
        d_in_tm: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        d_out_rm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsiError> {
        let func = self
            .module
            .get_function("transpose_tm_to_rm_f32")
            .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        let grid_x = ((cols as u32) + 31) / 32;
        let grid_y = ((rows as u32) + 31) / 32;
        let block: BlockSize = (32u32, 8u32, 1u32).into();
        unsafe {
            let mut in_ptr = d_in_tm.as_device_ptr().as_raw();
            let mut r_i = rows as i32;
            let mut c_i = cols as i32;
            let mut out_ptr = d_out_rm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut c_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1u32).into();
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTsiError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

fn expand_grid(r: &TsiBatchRange) -> Vec<TsiParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let longs = axis_usize(r.long_period);
    let shorts = axis_usize(r.short_period);
    let mut out = Vec::with_capacity(longs.len() * shorts.len());
    for &l in &longs {
        for &s in &shorts {
            out.push(TsiParams {
                long_period: Some(l),
                short_period: Some(s),
            });
        }
    }
    out
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const LEN: usize = 1_000_000; // 1M samples
    const ROWS: usize = 128; // number of parameter pairs

    struct TsiBatchState {
        cuda: CudaTsi,
        price: Vec<f32>,
        sweep: TsiBatchRange,
    }
    impl CudaBenchState for TsiBatchState {
        fn launch(&mut self) {
            let _ = self.cuda.tsi_batch_dev(&self.price, &self.sweep).unwrap();
        }
    }
    fn prep_batch() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaTsi::new(0).expect("cuda tsi");
        cuda.set_policy(CudaTsiPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });
        let price = gen_series(LEN);
        // Build a reasonable sweep of (long, short)
        let sweep = TsiBatchRange {
            long_period: (10, 100, (100 - 10).max(1) / ROWS.max(1)),
            short_period: (5, 30, (30 - 5).max(1) / (ROWS / 2).max(1)),
        };
        Box::new(TsiBatchState { cuda, price, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "tsi",
            "batch_dev",
            "tsi_cuda_batch_dev",
            "1m_x_128",
            prep_batch,
        )
        .with_inner_iters(4)]
    }
}
