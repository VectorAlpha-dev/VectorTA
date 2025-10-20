//! CUDA wrapper for Squeeze Momentum Indicator (SMI).
//!
//! Parity with ALMA-style wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/squeeze_momentum_kernel.ptx"))
//! - Stream NON_BLOCKING
//! - Minimal policies + introspection
//! - VRAM checks and grid chunking (grid.y <= 65_535)
//!
//! Exposes two device APIs:
//! - Batch (one series × many params): returns three DeviceArrayF32 handles
//! - Many-series × one param (time-major): returns three DeviceArrayF32 handles

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::squeeze_momentum::SqueezeMomentumBatchRange;
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaSmiError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaSmiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSmiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSmiError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaSmiError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaSmiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaSmiPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

// Parameter combo expanded on host
#[derive(Clone, Debug)]
struct SmCombo {
    lbb: usize,
    mbb: f32,
    lkc: usize,
    mkc: f32,
}

pub struct CudaSqueezeMomentum {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaSmiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
}

impl CudaSqueezeMomentum {
    pub fn new(device_id: usize) -> Result<Self, CudaSmiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/squeeze_momentum_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaSmiPolicy::default(),
            last_batch: None,
            last_many: None,
        })
    }

    pub fn set_policy(&mut self, policy: CudaSmiPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaSmiPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaSmiError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // ---------- Batch helpers ----------

    fn expand_grid(sweep: &SqueezeMomentumBatchRange) -> Vec<SmCombo> {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                vec![s]
            } else {
                (s..=e).step_by(st).collect()
            }
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                vec![s]
            } else {
                let mut v = Vec::new();
                let mut x = s;
                while x <= e + 1e-12 {
                    v.push(x);
                    x += st;
                }
                v
            }
        }
        let lbb = axis_usize(sweep.length_bb);
        let mbb = axis_f64(sweep.mult_bb);
        let lkc = axis_usize(sweep.length_kc);
        let mkc = axis_f64(sweep.mult_kc);
        let mut out = Vec::with_capacity(lbb.len() * mbb.len() * lkc.len() * mkc.len());
        for &a in &lbb {
            for &b in &mbb {
                for &c in &lkc {
                    for &d in &mkc {
                        out.push(SmCombo {
                            lbb: a,
                            mbb: b as f32,
                            lkc: c,
                            mkc: d as f32,
                        });
                    }
                }
            }
        }
        out
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &SqueezeMomentumBatchRange,
    ) -> Result<(Vec<SmCombo>, usize, usize), CudaSmiError> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaSmiError::InvalidInput(
                "inconsistent array lengths".into(),
            ));
        }
        if close.is_empty() {
            return Err(CudaSmiError::InvalidInput("empty data".into()));
        }
        let len = close.len();
        let first_valid = (0..len)
            .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
            .ok_or_else(|| CudaSmiError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaSmiError::InvalidInput("no parameter combos".into()));
        }
        // Feasibility: require tail >= max(lbb, lkc)
        let mut need = 0usize;
        for c in &combos {
            need = need.max(c.lbb.max(c.lkc));
        }
        let tail = len - first_valid;
        if tail < need {
            return Err(CudaSmiError::InvalidInput(format!(
                "not enough valid data: needed {}, valid {}",
                need, tail
            )));
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_lbb: &DeviceBuffer<i32>,
        d_mbb: &DeviceBuffer<f32>,
        d_lkc: &DeviceBuffer<i32>,
        d_mkc: &DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        d_sq: &mut DeviceBuffer<f32>,
        d_mo: &mut DeviceBuffer<f32>,
        d_si: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmiError> {
        let mut func: Function = self
            .module
            .get_function("squeeze_momentum_batch_f32")
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_h = d_high.as_device_ptr().as_raw();
            let mut p_l = d_low.as_device_ptr().as_raw();
            let mut p_c = d_close.as_device_ptr().as_raw();
            let mut p_lbb = d_lbb.as_device_ptr().as_raw();
            let mut p_mbb = d_mbb.as_device_ptr().as_raw();
            let mut p_lkc = d_lkc.as_device_ptr().as_raw();
            let mut p_mkc = d_mkc.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n_combos as i32;
            let mut fv_i = first_valid as i32;
            let mut p_sq = d_sq.as_device_ptr().as_raw();
            let mut p_mo = d_mo.as_device_ptr().as_raw();
            let mut p_si = d_si.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 13] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_lbb as *mut _ as *mut c_void,
                &mut p_mbb as *mut _ as *mut c_void,
                &mut p_lkc as *mut _ as *mut c_void,
                &mut p_mkc as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_si as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaSqueezeMomentum)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        Ok(())
    }

    pub fn squeeze_momentum_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &SqueezeMomentumBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaSmiError> {
        let (combos, first_valid, len) =
            Self::prepare_batch_inputs(high_f32, low_f32, close_f32, sweep)?;

        // Rough VRAM budget
        let in_bytes = 3 * len * std::mem::size_of::<f32>();
        let params_bytes =
            combos.len() * (2 * std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>());
        let out_bytes = 3 * combos.len() * len * std::mem::size_of::<f32>();
        let required = in_bytes + params_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaSmiError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Upload inputs
        let d_h =
            DeviceBuffer::from_slice(high_f32).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let d_l =
            DeviceBuffer::from_slice(low_f32).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let d_c =
            DeviceBuffer::from_slice(close_f32).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        let v_lbb: Vec<i32> = combos.iter().map(|c| c.lbb as i32).collect();
        let v_mbb: Vec<f32> = combos.iter().map(|c| c.mbb).collect();
        let v_lkc: Vec<i32> = combos.iter().map(|c| c.lkc as i32).collect();
        let v_mkc: Vec<f32> = combos.iter().map(|c| c.mkc).collect();
        let d_lbb =
            DeviceBuffer::from_slice(&v_lbb).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let d_mbb =
            DeviceBuffer::from_slice(&v_mbb).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let d_lkc =
            DeviceBuffer::from_slice(&v_lkc).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let d_mkc =
            DeviceBuffer::from_slice(&v_mkc).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        // Allocate outputs
        let elems = combos.len() * len;
        let mut d_sq: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let mut d_mo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let mut d_si: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_h,
            &d_l,
            &d_c,
            &d_lbb,
            &d_mbb,
            &d_lkc,
            &d_mkc,
            len,
            combos.len(),
            first_valid,
            &mut d_sq,
            &mut d_mo,
            &mut d_si,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_sq,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_mo,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_si,
                rows: combos.len(),
                cols: len,
            },
        ))
    }

    // ---------- Many-series, one param (time-major) ----------
    pub fn squeeze_momentum_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        lbb: usize,
        mbb: f32,
        lkc: usize,
        mkc: f32,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaSmiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSmiError::InvalidInput("cols or rows is zero".into()));
        }
        let expected = cols * rows;
        if high_tm_f32.len() != expected
            || low_tm_f32.len() != expected
            || close_tm_f32.len() != expected
        {
            return Err(CudaSmiError::InvalidInput(
                "time-major arrays length mismatch".into(),
            ));
        }
        if lbb == 0 || lkc == 0 || lbb > rows || lkc > rows {
            return Err(CudaSmiError::InvalidInput("invalid window lengths".into()));
        }

        // Build first_valid per series
        let mut fv = vec![0i32; cols];
        for s in 0..cols {
            let mut found = None;
            for r in 0..rows {
                let idx = r * cols + s;
                let h = high_tm_f32[idx];
                let l = low_tm_f32[idx];
                let c = close_tm_f32[idx];
                if !(h.is_nan() || l.is_nan() || c.is_nan()) {
                    found = Some(r);
                    break;
                }
            }
            let fv_s =
                found.ok_or_else(|| CudaSmiError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv_s < lbb.max(lkc) {
                return Err(CudaSmiError::InvalidInput(format!(
                    "series {} not enough valid data (needed {}, valid {})",
                    s,
                    lbb.max(lkc),
                    rows - fv_s
                )));
            }
            fv[s] = fv_s as i32;
        }

        let d_h =
            DeviceBuffer::from_slice(high_tm_f32).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let d_l =
            DeviceBuffer::from_slice(low_tm_f32).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let d_c = DeviceBuffer::from_slice(close_tm_f32)
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&fv).map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let mut d_sq_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let mut d_mo_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        let mut d_si_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("squeeze_momentum_many_series_one_param_f32")
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 1,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };
        let grid: GridSize = (1, cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_h = d_h.as_device_ptr().as_raw();
            let mut p_l = d_l.as_device_ptr().as_raw();
            let mut p_c = d_c.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut cols_i = cols as i32; // num_series
            let mut rows_i = rows as i32; // series_len
            let mut lbb_i = lbb as i32;
            let mut mbb_f = mbb as f32;
            let mut lkc_i = lkc as i32;
            let mut mkc_f = mkc as f32;
            let mut p_sq = d_sq_tm.as_device_ptr().as_raw();
            let mut p_mo = d_mo_tm.as_device_ptr().as_raw();
            let mut p_si = d_si_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 13] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut lbb_i as *mut _ as *mut c_void,
                &mut mbb_f as *mut _ as *mut c_void,
                &mut lkc_i as *mut _ as *mut c_void,
                &mut mkc_f as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_si as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaSmiError::Cuda(e.to_string()))?;
        unsafe {
            (*(self as *const _ as *mut CudaSqueezeMomentum)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }

        Ok((
            DeviceArrayF32 {
                buf: d_sq_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_mo_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_si_tm,
                rows,
                cols,
            },
        ))
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_00; // 100k default to keep VRAM reasonable
    const PARAM_SWEEP: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct SmiBatchState {
        cuda: CudaSqueezeMomentum,
        h: Vec<f32>,
        l: Vec<f32>,
        c: Vec<f32>,
        sweep: SqueezeMomentumBatchRange,
    }
    impl CudaBenchState for SmiBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .squeeze_momentum_batch_dev(&self.h, &self.l, &self.c, &self.sweep)
                .unwrap();
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSqueezeMomentum::new(0).expect("cuda smi");
        let h = gen_series(ONE_SERIES_LEN);
        let mut l = h.clone();
        for v in &mut l {
            *v -= 0.5;
        }
        let mut c = h.clone();
        for v in &mut c {
            *v -= 0.25;
        }
        let sweep = SqueezeMomentumBatchRange {
            length_bb: (10, 10 + PARAM_SWEEP - 1, 1),
            mult_bb: (2.0, 2.0, 0.0),
            length_kc: (10, 10, 0),
            mult_kc: (1.5, 1.5, 0.0),
        };
        Box::new(SmiBatchState {
            cuda,
            h,
            l,
            c,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "squeeze_momentum",
            "one_series_many_params",
            "squeeze_momentum_cuda_batch_dev",
            "100k_x_128",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
