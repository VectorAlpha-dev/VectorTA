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
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaRsmkError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("rsmk cuda: invalid input: {0}")]
    InvalidInput(String),
    #[error("rsmk cuda: unsupported: {0}")]
    Unsupported(String),
    #[error("rsmk cuda: out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("rsmk cuda: missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("rsmk cuda: invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error(
        "rsmk cuda: launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("rsmk cuda: device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("rsmk cuda: not implemented")]
    NotImplemented,
}

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
    context: Arc<Context>,
    device_id: u32,
}

impl CudaRsmk {
    pub fn new(device_id: usize) -> Result<Self, CudaRsmkError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

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
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
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
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaRsmkError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _total)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaRsmkError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
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
            let (start, end, step) = a;
            if step == 0 || start == end {
                return vec![start];
            }
            let mut vals = Vec::new();
            if start <= end {
                let st = step.max(1);
                for v in (start..=end).step_by(st) {
                    vals.push(v);
                }
            } else {
                let mut cur = start;
                let s = step.max(1);
                loop {
                    vals.push(cur);
                    if cur <= end {
                        break;
                    }
                    if cur < s {
                        break;
                    }
                    let next = cur - s;
                    if next == cur {
                        break;
                    }
                    cur = next;
                }
            }
            vals
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
        let el = std::mem::size_of::<f32>();
        let rows_len = rows
            .checked_mul(len)
            .ok_or_else(|| CudaRsmkError::InvalidInput("rows*len overflow".into()))?;
        let out_bytes = rows_len
            .checked_mul(2 * el)
            .ok_or_else(|| CudaRsmkError::InvalidInput("output size overflow".into()))?;
        let mom_bytes = uniq_looks
            .len()
            .checked_mul(len)
            .and_then(|x| x.checked_mul(el))
            .ok_or_else(|| CudaRsmkError::InvalidInput("momentum size overflow".into()))?;
        let in_bytes = len
            .checked_mul(2 * el)
            .ok_or_else(|| CudaRsmkError::InvalidInput("input size overflow".into()))?;
        let required = out_bytes
            .checked_add(mom_bytes)
            .and_then(|x| x.checked_add(in_bytes))
            .ok_or_else(|| CudaRsmkError::InvalidInput("VRAM size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        // Upload inputs
        let d_main = DeviceBuffer::from_slice(main_f32)?;
        let d_comp = DeviceBuffer::from_slice(compare_f32)?;

        // Prepare output buffers (row-major: rows x len)
        let mut d_indicator: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows_len) }?;
        let mut d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows_len) }?;

        // Kernel handles
        let mut k_mom: Function = self
            .module
            .get_function("rsmk_momentum_f32")
            .map_err(|_| CudaRsmkError::MissingKernelSymbol {
                name: "rsmk_momentum_f32",
            })?;
        let mut k_apply_row: Function = self
            .module
            .get_function("rsmk_apply_mom_single_row_ema_ema_f32")
            .map_err(|_| CudaRsmkError::MissingKernelSymbol {
                name: "rsmk_apply_mom_single_row_ema_ema_f32",
            })?;

        // Build momentum per unique lookback and cache device buffers
        let mut mom_dev: HashMap<usize, DeviceBuffer<f32>> = HashMap::new();
        for &lb in &uniq_looks {
            let mut d_m: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
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
                self.stream.launch(
                    &mut k_mom,
                    GridSize::xyz(1, 1, 1),
                    BlockSize::xyz(1, 1, 1),
                    0,
                    args,
                )?;
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
                self.stream.launch(
                    &mut k_apply_row,
                    GridSize::xyz(1, 1, 1),
                    BlockSize::xyz(1, 1, 1),
                    0,
                    args,
                )?;
            }
        }

        self.stream.synchronize()?;

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
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaRsmkError::InvalidInput("rows*cols overflow".into()))?;
        if main_tm_f32.len() != compare_tm_f32.len() || main_tm_f32.len() != expected {
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
        let el_f32 = std::mem::size_of::<f32>();
        let el_i32 = std::mem::size_of::<i32>();
        let series_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaRsmkError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = series_elems
            .checked_mul(2 * el_f32)
            .ok_or_else(|| CudaRsmkError::InvalidInput("output size overflow".into()))?;
        let in_bytes_main = series_elems
            .checked_mul(2 * el_f32)
            .ok_or_else(|| CudaRsmkError::InvalidInput("input size overflow".into()))?;
        let in_bytes_firsts = cols
            .checked_mul(el_i32)
            .ok_or_else(|| CudaRsmkError::InvalidInput("firsts size overflow".into()))?;
        let required = out_bytes
            .checked_add(in_bytes_main)
            .and_then(|x| x.checked_add(in_bytes_firsts))
            .ok_or_else(|| CudaRsmkError::InvalidInput("VRAM size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        // Upload inputs
        let d_main = DeviceBuffer::from_slice(main_tm_f32)?;
        let d_comp = DeviceBuffer::from_slice(compare_tm_f32)?;
        let h_firsts = LockedBuffer::from_slice(&firsts)?;
        let mut d_firsts: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(cols, &self.stream) }?;
        unsafe {
            d_firsts.async_copy_from(&h_firsts, &self.stream)?;
        }

        // Outputs
        let mut d_indicator: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_elems) }?;
        let mut d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_elems) }?;

        // Launch kernel (one thread per series)
        let mut func: Function = self
            .module
            .get_function("rsmk_many_series_one_param_time_major_ema_ema_f32")
            .map_err(|_| CudaRsmkError::MissingKernelSymbol {
                name: "rsmk_many_series_one_param_time_major_ema_ema_f32",
            })?;
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
                .launch(&mut func, grid, block, 0, args)?;
        }

        self.stream.synchronize()?;
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
    use std::collections::{BTreeSet, HashMap};
    use std::ffi::c_void;

    const BATCH_LEN: usize = 45_000;
    const MANY_COLS: usize = 64;
    const MANY_ROWS: usize = 500_000;

    fn bytes_batch() -> usize {
        // combos: 3 lookbacks * 3 periods * 3 signal_periods = 27
        let rows = 27usize;
        let in_bytes = 2 * BATCH_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * rows * BATCH_LEN * std::mem::size_of::<f32>();
        let mom_bytes = 3 * BATCH_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + mom_bytes + 32 * 1024 * 1024
    }
    fn bytes_many() -> usize {
        let elems = MANY_COLS * MANY_ROWS;
        let in_bytes = 2 * elems * std::mem::size_of::<f32>();
        let out_bytes = 2 * elems * std::mem::size_of::<f32>();
        let first_bytes = MANY_COLS * std::mem::size_of::<i32>();
        in_bytes + out_bytes + first_bytes + 64 * 1024 * 1024
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "rsmk",
                "batch_dev",
                "rsmk_cuda_batch_dev",
                "45k_x_27combos",
                prep_rsmk_batch_box,
            )
            .with_inner_iters(4)
            .with_mem_required(bytes_batch()),
            CudaBenchScenario::new(
                "rsmk",
                "many_series_one_param",
                "rsmk_cuda_many_series_one_param",
                "64x500k",
                prep_rsmk_many_series_box,
            )
            .with_inner_iters(2)
            .with_mem_required(bytes_many()),
        ]
    }

    struct RsmkBatchDevState {
        cuda: CudaRsmk,
        d_main: DeviceBuffer<f32>,
        d_comp: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        lookbacks: Vec<usize>,
        d_moms: Vec<DeviceBuffer<f32>>,
        row_mom_idx: Vec<usize>,
        periods: Vec<usize>,
        signals: Vec<usize>,
        first_moms: Vec<usize>,
        d_indicator: DeviceBuffer<f32>,
        d_signal: DeviceBuffer<f32>,
    }

    impl CudaBenchState for RsmkBatchDevState {
        fn launch(&mut self) {
            let mut k_mom = self
                .cuda
                .module
                .get_function("rsmk_momentum_f32")
                .expect("rsmk_momentum_f32");
            let mut k_apply = self
                .cuda
                .module
                .get_function("rsmk_apply_mom_single_row_ema_ema_f32")
                .expect("rsmk_apply_mom_single_row_ema_ema_f32");

            for (idx, &lb) in self.lookbacks.iter().enumerate() {
                unsafe {
                    let mut main_ptr = self.d_main.as_device_ptr().as_raw();
                    let mut comp_ptr = self.d_comp.as_device_ptr().as_raw();
                    let mut fv_i = self.first_valid as i32;
                    let mut lb_i = lb as i32;
                    let mut len_i = self.len as i32;
                    let mut mom_ptr = self.d_moms[idx].as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut main_ptr as *mut _ as *mut c_void,
                        &mut comp_ptr as *mut _ as *mut c_void,
                        &mut fv_i as *mut _ as *mut c_void,
                        &mut lb_i as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut mom_ptr as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(
                            &mut k_mom,
                            GridSize::xyz(1, 1, 1),
                            BlockSize::xyz(1, 1, 1),
                            0,
                            args,
                        )
                        .expect("rsmk_momentum_f32 launch");
                }
            }

            for row in 0..self.periods.len() {
                let mom_idx = self.row_mom_idx[row];
                let d_m = &self.d_moms[mom_idx];
                unsafe {
                    let mut mom_ptr = d_m.as_device_ptr().as_raw();
                    let mut len_i = self.len as i32;
                    let mut fv_m_i = self.first_moms[row] as i32;
                    let mut p_i = self.periods[row] as i32;
                    let mut s_i = self.signals[row] as i32;
                    let mut ind_ptr = self
                        .d_indicator
                        .as_device_ptr()
                        .offset((row * self.len) as isize)
                        .as_raw();
                    let mut sig_ptr = self
                        .d_signal
                        .as_device_ptr()
                        .offset((row * self.len) as isize)
                        .as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut mom_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut fv_m_i as *mut _ as *mut c_void,
                        &mut p_i as *mut _ as *mut c_void,
                        &mut s_i as *mut _ as *mut c_void,
                        &mut ind_ptr as *mut _ as *mut c_void,
                        &mut sig_ptr as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(
                            &mut k_apply,
                            GridSize::xyz(1, 1, 1),
                            BlockSize::xyz(1, 1, 1),
                            0,
                            args,
                        )
                        .expect("rsmk_apply launch");
                }
            }

            self.cuda.stream.synchronize().unwrap();
        }
    }

    fn prep_rsmk_batch_box() -> Box<dyn CudaBenchState> {
        let cuda = CudaRsmk::new(0).expect("cuda rsmk");
        let len = BATCH_LEN;
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
        let first_valid = CudaRsmk::first_valid(&main, &comp).expect("first_valid");

        fn axis(a: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = a;
            if step == 0 || start == end {
                return vec![start];
            }
            let mut vals = Vec::new();
            if start <= end {
                let st = step.max(1);
                for v in (start..=end).step_by(st) {
                    vals.push(v);
                }
            } else {
                let mut cur = start;
                let s = step.max(1);
                loop {
                    vals.push(cur);
                    if cur <= end {
                        break;
                    }
                    if cur < s {
                        break;
                    }
                    let next = cur - s;
                    if next == cur {
                        break;
                    }
                    cur = next;
                }
            }
            vals
        }

        let looks = axis(sweep.lookback);
        let periods = axis(sweep.period);
        let sigs = axis(sweep.signal_period);
        let mut row_lookbacks = Vec::with_capacity(looks.len() * periods.len() * sigs.len());
        let mut row_periods = Vec::with_capacity(row_lookbacks.capacity());
        let mut row_sigs = Vec::with_capacity(row_lookbacks.capacity());
        for &l in &looks {
            for &p in &periods {
                for &s in &sigs {
                    row_lookbacks.push(l);
                    row_periods.push(p);
                    row_sigs.push(s);
                }
            }
        }
        let uniq: BTreeSet<usize> = row_lookbacks.iter().copied().collect();
        let lookbacks: Vec<usize> = uniq.into_iter().collect();
        let mut map = HashMap::new();
        for (i, &lb) in lookbacks.iter().enumerate() {
            map.insert(lb, i);
        }
        let row_mom_idx: Vec<usize> = row_lookbacks
            .iter()
            .map(|lb| *map.get(lb).expect("lb idx"))
            .collect();
        let first_moms: Vec<usize> = row_lookbacks.iter().map(|&lb| first_valid + lb).collect();

        let d_main = DeviceBuffer::from_slice(&main).expect("d_main");
        let d_comp = DeviceBuffer::from_slice(&comp).expect("d_comp");
        let mut d_moms = Vec::with_capacity(lookbacks.len());
        for _ in 0..lookbacks.len() {
            d_moms.push(unsafe { DeviceBuffer::uninitialized(len) }.expect("d_mom"));
        }
        let n_rows = row_lookbacks.len();
        let rows_len = n_rows.checked_mul(len).expect("rows*len");
        let d_indicator: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows_len) }.expect("d_indicator");
        let d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows_len) }.expect("d_signal");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(RsmkBatchDevState {
            cuda,
            d_main,
            d_comp,
            len,
            first_valid,
            lookbacks,
            d_moms,
            row_mom_idx,
            periods: row_periods,
            signals: row_sigs,
            first_moms,
            d_indicator,
            d_signal,
        })
    }

    struct RsmkManyDevState {
        cuda: CudaRsmk,
        d_main: DeviceBuffer<f32>,
        d_comp: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        lb: usize,
        p: usize,
        s: usize,
        d_indicator: DeviceBuffer<f32>,
        d_signal: DeviceBuffer<f32>,
    }

    impl CudaBenchState for RsmkManyDevState {
        fn launch(&mut self) {
            let mut func = self
                .cuda
                .module
                .get_function("rsmk_many_series_one_param_time_major_ema_ema_f32")
                .expect("rsmk_many_series_one_param_time_major_ema_ema_f32");
            let block = BlockSize::xyz(1, 1, 1);
            let grid = GridSize::xyz(1, self.cols as u32, 1);
            unsafe {
                let mut main_ptr = self.d_main.as_device_ptr().as_raw();
                let mut comp_ptr = self.d_comp.as_device_ptr().as_raw();
                let mut first_ptr = self.d_first.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut lb_i = self.lb as i32;
                let mut p_i = self.p as i32;
                let mut s_i = self.s as i32;
                let mut ind_ptr = self.d_indicator.as_device_ptr().as_raw();
                let mut sig_ptr = self.d_signal.as_device_ptr().as_raw();
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
                self.cuda
                    .stream
                    .launch(&mut func, grid, block, 0, args)
                    .expect("rsmk many-series launch");
            }
            self.cuda.stream.synchronize().unwrap();
        }
    }

    fn prep_rsmk_many_series_box() -> Box<dyn CudaBenchState> {
        let cuda = CudaRsmk::new(0).expect("cuda rsmk");
        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let mut tm_main = vec![f32::NAN; cols * rows];
        let mut tm_comp = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for r in s..rows {
                let x = (r as f32) + 0.3 * (s as f32);
                tm_main[r * cols + s] = (x * 0.0017).sin() + 0.0003 * x;
                tm_comp[r * cols + s] = (x * 0.0012).cos().abs() + 0.4;
            }
        }
        let params = RsmkParams {
            lookback: Some(90),
            period: Some(3),
            signal_period: Some(20),
            matype: Some("ema".into()),
            signal_matype: Some("ema".into()),
        };
        let lb = params.lookback.unwrap_or(90);
        let p = params.period.unwrap_or(3);
        let s = params.signal_period.unwrap_or(20);

        let mut firsts = vec![0i32; cols];
        for sidx in 0..cols {
            let mut fv = -1i32;
            for r in 0..rows {
                let idx = r * cols + sidx;
                let m = tm_main[idx];
                let c = tm_comp[idx];
                if m.is_finite() && c.is_finite() && c != 0.0 {
                    fv = r as i32;
                    break;
                }
            }
            assert!(fv >= 0, "series has no valid sample");
            firsts[sidx] = fv;
        }

        let d_main = DeviceBuffer::from_slice(&tm_main).expect("d_main_tm");
        let d_comp = DeviceBuffer::from_slice(&tm_comp).expect("d_comp_tm");
        let d_first = DeviceBuffer::from_slice(&firsts).expect("d_first");
        let elems = cols.checked_mul(rows).expect("elems");
        let d_indicator: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_indicator");
        let d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_signal");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(RsmkManyDevState {
            cuda,
            d_main,
            d_comp,
            d_first,
            cols,
            rows,
            lb,
            p,
            s,
            d_indicator,
            d_signal,
        })
    }
}
