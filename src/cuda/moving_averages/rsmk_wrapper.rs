//! CUDA wrapper for RSMK (Relative Strength Mark).
//!
//! Mirrors ALMA/CWMA conventions:
//! - PTX loaded via include_str!(concat!(env!("OUT_DIR"), "/rsmk_kernel.ptx"))
//! - Stream NON_BLOCKING; JIT options: DetermineTargetFromContext + O2 (fallbacks applied)
//! - VRAM checks with ~64MB headroom
//! - Public device entry points:
//!     - `rsmk_batch_dev(&[f32], &[f32], &RsmkBatchRange)` -> (indicator, signal, combos)
//!     - `rsmk_many_series_one_param_time_major_dev(&[f32], &[f32], cols, rows, &RsmkParams)` -> (indicator, signal)
//!
//! Batch implementation reuses shared precompute across rows:
//! - For each unique lookback, compute momentum once on device (rsmk_momentum_f32)
//! - Apply EMA/EMA path per-row via a light single-row kernel

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::rsmk::{RsmkBatchRange, RsmkParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaRsmkError {
    Cuda(String),
    InvalidInput(String),
    Unsupported(String),
}

impl fmt::Display for CudaRsmkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaRsmkError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaRsmkError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            CudaRsmkError::Unsupported(s) => write!(f, "Unsupported: {}", s),
        }
    }
}
impl std::error::Error for CudaRsmkError {}

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

pub struct CudaRsmk {
    module: Module,
    pub(crate) stream: Stream,
    _context: Context,
}

impl CudaRsmk {
    pub fn new(device_id: usize) -> Result<Self, CudaRsmkError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/rsmk_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaRsmkError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // Utility: find first index where both series finite and compare != 0
    fn first_valid(main: &[f32], compare: &[f32]) -> Option<usize> {
        main.iter()
            .zip(compare.iter())
            .position(|(&m, &c)| m.is_finite() && c.is_finite() && c != 0.0)
    }

    /// Batch: one series × many params (EMA/EMA path as in scalar batch expand_grid)
    pub fn rsmk_batch_dev(
        &self,
        main_f32: &[f32],
        compare_f32: &[f32],
        sweep: &RsmkBatchRange,
    ) -> Result<(DeviceArrayF32Pair, Vec<RsmkParams>), CudaRsmkError> {
        if main_f32.len() != compare_f32.len() {
            return Err(CudaRsmkError::InvalidInput("length mismatch".into()));
        }
        let len = main_f32.len();
        if len == 0 {
            return Err(CudaRsmkError::InvalidInput("empty input".into()));
        }
        let first_valid = Self::first_valid(main_f32, compare_f32)
            .ok_or_else(|| CudaRsmkError::InvalidInput("all values NaN or compare==0".into()))?;

        // Expand parameter grid (mirrors scalar expand_grid: EMA/EMA only)
        fn axis(a: (usize, usize, usize)) -> Vec<usize> {
            if a.2 == 0 || a.0 == a.1 {
                return vec![a.0];
            }
            (a.0..=a.1).step_by(a.2).collect()
        }
        let looks = axis(sweep.lookback);
        let periods = axis(sweep.period);
        let sigs = axis(sweep.signal_period);
        let mut combos = Vec::with_capacity(looks.len() * periods.len() * sigs.len());
        for &l in &looks {
            for &p in &periods {
                for &s in &sigs {
                    combos.push(RsmkParams {
                        lookback: Some(l),
                        period: Some(p),
                        signal_period: Some(s),
                        matype: Some("ema".into()),
                        signal_matype: Some("ema".into()),
                    });
                }
            }
        }
        if combos.is_empty() {
            return Err(CudaRsmkError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // VRAM estimate: indicator + signal + momentum buffers per unique lookback
        let rows = combos.len();
        let uniq_looks: BTreeSet<usize> = combos.iter().map(|p| p.lookback.unwrap()).collect();
        let out_bytes = 2usize * rows * len * std::mem::size_of::<f32>();
        let mom_bytes = uniq_looks.len() * len * std::mem::size_of::<f32>();
        let in_bytes = 2usize * len * std::mem::size_of::<f32>();
        let required = out_bytes + mom_bytes + in_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaRsmkError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        // Upload inputs
        let d_main =
            DeviceBuffer::from_slice(main_f32).map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let d_comp = DeviceBuffer::from_slice(compare_f32)
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;

        // Prepare output buffers (row-major: rows x len)
        let mut d_indicator: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let mut d_signal: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;

        // Kernel handles
        let mut k_mom: Function = self
            .module
            .get_function("rsmk_momentum_f32")
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let mut k_apply_row: Function = self
            .module
            .get_function("rsmk_apply_mom_single_row_ema_ema_f32")
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;

        // Build momentum per unique lookback and cache device buffers
        let mut mom_dev: HashMap<usize, DeviceBuffer<f32>> = HashMap::new();
        for &lb in &uniq_looks {
            let mut d_m = unsafe { DeviceBuffer::<f32>::uninitialized(len) }
                .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
            unsafe {
                let mut main_ptr = d_main.as_device_ptr().as_raw();
                let mut comp_ptr = d_comp.as_device_ptr().as_raw();
                let mut lb_i = lb as i32;
                let mut fv_i = first_valid as i32;
                let mut len_i = len as i32;
                let mut mom_ptr = d_m.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut main_ptr as *mut _ as *mut c_void,
                    &mut comp_ptr as *mut _ as *mut c_void,
                    &mut lb_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut mom_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(
                        &mut k_mom,
                        GridSize::xyz(1, 1, 1),
                        BlockSize::xyz(1, 1, 1),
                        0,
                        args,
                    )
                    .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
            }
            mom_dev.insert(lb, d_m);
        }

        // Apply per-row (EMA/EMA) using the cached momentum
        for (row, prm) in combos.iter().enumerate() {
            let lb = prm.lookback.unwrap();
            let period = prm.period.unwrap();
            let sig = prm.signal_period.unwrap();
            let first_mom = first_valid + lb;
            let d_m = mom_dev.get(&lb).expect("mom dev by lookback");
            unsafe {
                let mut mom_ptr = d_m.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut fv_m_i = first_mom as i32;
                let mut p_i = period as i32;
                let mut s_i = sig as i32;
                // row section pointers
                let mut ind_ptr = unsafe {
                    d_indicator
                        .as_device_ptr()
                        .offset((row * len) as isize)
                        .as_raw()
                };
                let mut sig_ptr = unsafe {
                    d_signal
                        .as_device_ptr()
                        .offset((row * len) as isize)
                        .as_raw()
                };
                let args: &mut [*mut c_void] = &mut [
                    &mut mom_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_m_i as *mut _ as *mut c_void,
                    &mut p_i as *mut _ as *mut c_void,
                    &mut s_i as *mut _ as *mut c_void,
                    &mut ind_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(
                        &mut k_apply_row,
                        GridSize::xyz(1, 1, 1),
                        BlockSize::xyz(1, 1, 1),
                        0,
                        args,
                    )
                    .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32Pair {
                a: DeviceArrayF32 {
                    buf: d_indicator,
                    rows,
                    cols: len,
                },
                b: DeviceArrayF32 {
                    buf: d_signal,
                    rows,
                    cols: len,
                },
            },
            combos,
        ))
    }

    /// Many-series × one-param (time-major), EMA/EMA path.
    pub fn rsmk_many_series_one_param_time_major_dev(
        &self,
        main_tm_f32: &[f32],
        compare_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &RsmkParams,
    ) -> Result<DeviceArrayF32Pair, CudaRsmkError> {
        if main_tm_f32.len() != compare_tm_f32.len() || main_tm_f32.len() != rows * cols {
            return Err(CudaRsmkError::InvalidInput(
                "time-major dims mismatch".into(),
            ));
        }
        let lb = params.lookback.unwrap_or(90);
        let p = params.period.unwrap_or(3);
        let s = params.signal_period.unwrap_or(20);
        // Only EMA/EMA documented for batch; keep many-series aligned
        if !params
            .matype
            .as_deref()
            .unwrap_or("ema")
            .eq_ignore_ascii_case("ema")
            || !params
                .signal_matype
                .as_deref()
                .unwrap_or("ema")
                .eq_ignore_ascii_case("ema")
        {
            return Err(CudaRsmkError::Unsupported(
                "only EMA/EMA path is implemented on CUDA for RSMK".into(),
            ));
        }

        // Build first_valids per series on host
        let mut firsts = vec![0i32; cols];
        for sidx in 0..cols {
            let mut fv = -1i32;
            for r in 0..rows {
                let m = main_tm_f32[r * cols + sidx];
                let c = compare_tm_f32[r * cols + sidx];
                if m.is_finite() && c.is_finite() && c != 0.0 {
                    fv = r as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaRsmkError::InvalidInput(
                    "all values NaN or compare==0 in a series".into(),
                ));
            }
            firsts[sidx] = fv;
        }

        // VRAM estimate: indicator + signal + inputs + firsts
        let out_bytes = 2usize * rows * cols * std::mem::size_of::<f32>();
        let in_bytes =
            2usize * rows * cols * std::mem::size_of::<f32>() + cols * std::mem::size_of::<i32>();
        if !Self::will_fit(out_bytes + in_bytes, 64 * 1024 * 1024) {
            return Err(CudaRsmkError::InvalidInput(
                "insufficient VRAM for RSMK many-series".into(),
            ));
        }

        // Upload inputs
        let d_main = DeviceBuffer::from_slice(main_tm_f32)
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let d_comp = DeviceBuffer::from_slice(compare_tm_f32)
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let h_firsts =
            LockedBuffer::from_slice(&firsts).map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let mut d_firsts: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(cols, &self.stream) }
                .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        unsafe {
            d_firsts
                .async_copy_from(&h_firsts, &self.stream)
                .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        }

        // Outputs
        let mut d_indicator: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * cols) }
                .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let mut d_signal: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * cols) }
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;

        // Launch kernel (one thread per series)
        let mut func: Function = self
            .module
            .get_function("rsmk_many_series_one_param_time_major_ema_ema_f32")
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        let block = BlockSize::xyz(1, 1, 1);
        let grid = GridSize::xyz(1, cols as u32, 1);
        unsafe {
            let mut main_ptr = d_main.as_device_ptr().as_raw();
            let mut comp_ptr = d_comp.as_device_ptr().as_raw();
            let mut first_ptr = d_firsts.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut lb_i = lb as i32;
            let mut p_i = p as i32;
            let mut s_i = s as i32;
            let mut ind_ptr = d_indicator.as_device_ptr().as_raw();
            let mut sig_ptr = d_signal.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut main_ptr as *mut _ as *mut c_void,
                &mut comp_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut lb_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut ind_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&mut func, grid, block, 0, args)
                .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaRsmkError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32Pair {
            a: DeviceArrayF32 {
                buf: d_indicator,
                rows,
                cols,
            },
            b: DeviceArrayF32 {
                buf: d_signal,
                rows,
                cols,
            },
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "rsmk",
                "batch_dev",
                "rsmk_cuda_batch_dev",
                "45k_x_27combos",
                prep_rsmk_batch_box,
            )
            .with_inner_iters(4),
            CudaBenchScenario::new(
                "rsmk",
                "many_series_one_param",
                "rsmk_cuda_many_series_one_param",
                "64x500k",
                prep_rsmk_many_series_box,
            )
            .with_inner_iters(2),
        ]
    }

    struct RsmkBatchState {
        cuda: CudaRsmk,
        main: Vec<f32>,
        comp: Vec<f32>,
        sweep: RsmkBatchRange,
    }
    impl CudaBenchState for RsmkBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .rsmk_batch_dev(&self.main, &self.comp, &self.sweep)
                .unwrap();
            self.cuda.stream.synchronize().unwrap();
        }
    }
    fn prep_rsmk_batch() -> RsmkBatchState {
        let cuda = CudaRsmk::new(0).expect("cuda rsmk");
        let len = 45_000usize;
        let mut main = vec![f32::NAN; len];
        let mut comp = vec![f32::NAN; len];
        for i in 5..len {
            let x = i as f32;
            main[i] = (x * 0.0011).sin() + 0.0002 * x;
            comp[i] = (x * 0.0009).cos().abs() + 0.5;
        }
        let sweep = RsmkBatchRange {
            lookback: (30, 42, 6),
            period: (3, 9, 3),
            signal_period: (10, 22, 6),
        };
        RsmkBatchState {
            cuda,
            main,
            comp,
            sweep,
        }
    }
    fn prep_rsmk_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_rsmk_batch())
    }

    struct RsmkManyState {
        cuda: CudaRsmk,
        tm_main: Vec<f32>,
        tm_comp: Vec<f32>,
        cols: usize,
        rows: usize,
        p: RsmkParams,
    }
    impl CudaBenchState for RsmkManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .rsmk_many_series_one_param_time_major_dev(
                    &self.tm_main,
                    &self.tm_comp,
                    self.cols,
                    self.rows,
                    &self.p,
                )
                .unwrap();
            self.cuda.stream.synchronize().unwrap();
        }
    }
    fn prep_rsmk_many_series() -> RsmkManyState {
        let cuda = CudaRsmk::new(0).expect("cuda rsmk");
        let cols = 64usize;
        let rows = 500_000usize;
        let mut tm_main = vec![f32::NAN; cols * rows];
        let mut tm_comp = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for r in s..rows {
                let x = (r as f32) + 0.3 * (s as f32);
                tm_main[r * cols + s] = (x * 0.0017).sin() + 0.0003 * x;
                tm_comp[r * cols + s] = (x * 0.0012).cos().abs() + 0.4;
            }
        }
        let p = RsmkParams {
            lookback: Some(90),
            period: Some(3),
            signal_period: Some(20),
            matype: Some("ema".into()),
            signal_matype: Some("ema".into()),
        };
        RsmkManyState {
            cuda,
            tm_main,
            tm_comp,
            cols,
            rows,
            p,
        }
    }
    fn prep_rsmk_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_rsmk_many_series())
    }
}
