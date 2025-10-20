//! CUDA wrapper for TTM Squeeze (momentum + squeeze state).
//!
//! Parity with ALMA-style wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/ttm_squeeze_kernel.ptx"))
//! - NON_BLOCKING stream
//! - Simple policy enums (Auto/OneD) and last-selected introspection
//! - VRAM check with ~64MB headroom and sensible grid sizing
//! - Batch (one-series × many-params) and Many-series × one-param (time-major)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::ttm_squeeze::TtmSqueezeBatchRange;
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
pub enum CudaTtmSqueezeError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaTtmSqueezeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTtmSqueezeError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTtmSqueezeError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaTtmSqueezeError {}

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
pub struct CudaTtmSqueezePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaTtmSqueezePolicy {
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

#[derive(Clone, Debug)]
struct Combo {
    length: i32,
    bb_mult: f32,
    kc_high: f32,
    kc_mid: f32,
    kc_low: f32,
}

pub struct CudaTtmSqueeze {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaTtmSqueezePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
}

impl CudaTtmSqueeze {
    pub fn new(device_id: usize) -> Result<Self, CudaTtmSqueezeError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ttm_squeeze_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;

        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaTtmSqueezePolicy::default(),
            last_batch: None,
            last_many: None,
        })
    }

    pub fn set_policy(&mut self, policy: CudaTtmSqueezePolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTtmSqueezePolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaTtmSqueezeError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))
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

    fn expand_grid(range: &TtmSqueezeBatchRange) -> Vec<Combo> {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                vec![s]
            } else {
                (s..=e).step_by(st).collect()
            }
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
            if (st.abs() < 1e-12) || ((s - e).abs() < 1e-12) {
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
        let lengths = axis_usize(range.length);
        let bb = axis_f64(range.bb_mult);
        let kh = axis_f64(range.kc_high);
        let km = axis_f64(range.kc_mid);
        let kl = axis_f64(range.kc_low);
        let mut out = Vec::with_capacity(lengths.len() * bb.len() * kh.len() * km.len() * kl.len());
        for &l in &lengths {
            for &b in &bb {
                for &h in &kh {
                    for &m in &km {
                        for &lo in &kl {
                            out.push(Combo {
                                length: l as i32,
                                bb_mult: b as f32,
                                kc_high: h as f32,
                                kc_mid: m as f32,
                                kc_low: lo as f32,
                            });
                        }
                    }
                }
            }
        }
        out
    }

    fn prepare_batch_inputs(
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &TtmSqueezeBatchRange,
    ) -> Result<(Vec<Combo>, usize, usize), CudaTtmSqueezeError> {
        if high_f32.len() != low_f32.len() || low_f32.len() != close_f32.len() {
            return Err(CudaTtmSqueezeError::InvalidInput(format!(
                "inconsistent lengths: high={}, low={}, close={}",
                high_f32.len(),
                low_f32.len(),
                close_f32.len()
            )));
        }
        if close_f32.is_empty() {
            return Err(CudaTtmSqueezeError::InvalidInput("empty series".into()));
        }
        let len = close_f32.len();
        let first_valid = close_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTtmSqueezeError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaTtmSqueezeError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for c in &combos {
            if c.length <= 0 || (len - first_valid) < (c.length as usize) {
                return Err(CudaTtmSqueezeError::InvalidInput(
                    "invalid length or insufficient data".into(),
                ));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_len: &DeviceBuffer<i32>,
        d_bb: &DeviceBuffer<f32>,
        d_kh: &DeviceBuffer<f32>,
        d_km: &DeviceBuffer<f32>,
        d_kl: &DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        d_mo: &mut DeviceBuffer<f32>,
        d_sq: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTtmSqueezeError> {
        let mut func: Function = self
            .module
            .get_function("ttm_squeeze_batch_f32")
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_h = d_high.as_device_ptr().as_raw();
            let mut p_l = d_low.as_device_ptr().as_raw();
            let mut p_c = d_close.as_device_ptr().as_raw();
            let mut p_len = d_len.as_device_ptr().as_raw();
            let mut p_bb = d_bb.as_device_ptr().as_raw();
            let mut p_kh = d_kh.as_device_ptr().as_raw();
            let mut p_km = d_km.as_device_ptr().as_raw();
            let mut p_kl = d_kl.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n_combos as i32;
            let mut fv_i = first_valid as i32;
            let mut p_mo = d_mo.as_device_ptr().as_raw();
            let mut p_sq = d_sq.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 14] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_len as *mut _ as *mut c_void,
                &mut p_bb as *mut _ as *mut c_void,
                &mut p_kh as *mut _ as *mut c_void,
                &mut p_km as *mut _ as *mut c_void,
                &mut p_kl as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaTtmSqueeze)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        Ok(())
    }

    pub fn ttm_squeeze_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &TtmSqueezeBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaTtmSqueezeError> {
        let (combos, first_valid, len) =
            Self::prepare_batch_inputs(high_f32, low_f32, close_f32, sweep)?;

        // VRAM estimate: inputs + params + outputs + headroom
        let in_bytes = 3 * len * std::mem::size_of::<f32>();
        let params_bytes =
            combos.len() * (std::mem::size_of::<i32>() + 4 * std::mem::size_of::<f32>());
        let out_bytes = 2 * combos.len() * len * std::mem::size_of::<f32>();
        let required = in_bytes + params_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaTtmSqueezeError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Upload inputs
        let d_h = DeviceBuffer::from_slice(high_f32)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_l = DeviceBuffer::from_slice(low_f32)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_c = DeviceBuffer::from_slice(close_f32)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;

        // Upload params
        let v_len: Vec<i32> = combos.iter().map(|c| c.length).collect();
        let v_bb: Vec<f32> = combos.iter().map(|c| c.bb_mult).collect();
        let v_kh: Vec<f32> = combos.iter().map(|c| c.kc_high).collect();
        let v_km: Vec<f32> = combos.iter().map(|c| c.kc_mid).collect();
        let v_kl: Vec<f32> = combos.iter().map(|c| c.kc_low).collect();
        let d_len = DeviceBuffer::from_slice(&v_len)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_bb = DeviceBuffer::from_slice(&v_bb)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_kh = DeviceBuffer::from_slice(&v_kh)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_km = DeviceBuffer::from_slice(&v_km)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_kl = DeviceBuffer::from_slice(&v_kl)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;

        // Allocate outputs
        let elems = combos.len() * len;
        let mut d_mo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let mut d_sq: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_h,
            &d_l,
            &d_c,
            &d_len,
            &d_bb,
            &d_kh,
            &d_km,
            &d_kl,
            len,
            combos.len(),
            first_valid,
            &mut d_mo,
            &mut d_sq,
        )?;
        Ok((
            DeviceArrayF32 {
                buf: d_mo,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_sq,
                rows: combos.len(),
                cols: len,
            },
        ))
    }

    fn launch_many_series_kernel(
        &self,
        d_h_tm: &DeviceBuffer<f32>,
        d_l_tm: &DeviceBuffer<f32>,
        d_c_tm: &DeviceBuffer<f32>,
        d_first: &DeviceBuffer<i32>,
        rows: usize,
        cols: usize,
        length: usize,
        bb_mult: f32,
        kh: f32,
        km: f32,
        kl: f32,
        d_mo_tm: &mut DeviceBuffer<f32>,
        d_sq_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTtmSqueezeError> {
        let mut func: Function = self
            .module
            .get_function("ttm_squeeze_many_series_one_param_f32")
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 64,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (1u32, rows as u32, 1u32).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_h = d_h_tm.as_device_ptr().as_raw();
            let mut p_l = d_l_tm.as_device_ptr().as_raw();
            let mut p_c = d_c_tm.as_device_ptr().as_raw();
            let mut p_fv = d_first.as_device_ptr().as_raw();
            let mut nser = rows as i32;
            let mut slen = cols as i32;
            let mut l_i = length as i32;
            let mut bb = bb_mult as f32;
            let mut khf = kh as f32;
            let mut kmf = km as f32;
            let mut klf = kl as f32;
            let mut p_mo = d_mo_tm.as_device_ptr().as_raw();
            let mut p_sq = d_sq_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 14] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut nser as *mut _ as *mut c_void,
                &mut slen as *mut _ as *mut c_void,
                &mut l_i as *mut _ as *mut c_void,
                &mut bb as *mut _ as *mut c_void,
                &mut khf as *mut _ as *mut c_void,
                &mut kmf as *mut _ as *mut c_void,
                &mut klf as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaTtmSqueeze)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        Ok(())
    }

    pub fn ttm_squeeze_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        length: usize,
        bb_mult: f32,
        kc_high: f32,
        kc_mid: f32,
        kc_low: f32,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaTtmSqueezeError> {
        if high_tm_f32.len() != low_tm_f32.len() || low_tm_f32.len() != close_tm_f32.len() {
            return Err(CudaTtmSqueezeError::InvalidInput(
                "inconsistent time-major inputs".into(),
            ));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaTtmSqueezeError::InvalidInput("zero dims".into()));
        }
        if high_tm_f32.len() != cols * rows {
            return Err(CudaTtmSqueezeError::InvalidInput(
                "dims mismatch with buffer length".into(),
            ));
        }

        // Compute per-series first_valid indices on host (time-major layout)
        let mut first_valids: Vec<i32> = vec![0; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = close_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaTtmSqueezeError::InvalidInput(format!(
                    "series {} all NaN",
                    s
                )));
            }
            if (rows as i32 - fv) < (length as i32) {
                return Err(CudaTtmSqueezeError::InvalidInput(
                    "insufficient valid data for length".into(),
                ));
            }
            first_valids[s] = fv;
        }

        // VRAM estimate
        let elems = cols * rows;
        let in_bytes = 3 * elems * std::mem::size_of::<f32>();
        let out_bytes = 2 * elems * std::mem::size_of::<f32>();
        let params = cols * std::mem::size_of::<i32>();
        let required = in_bytes + out_bytes + params;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaTtmSqueezeError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Upload inputs
        let d_h = DeviceBuffer::from_slice(high_tm_f32)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_l = DeviceBuffer::from_slice(low_tm_f32)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_c = DeviceBuffer::from_slice(close_tm_f32)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;

        let mut d_mo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;
        let mut d_sq: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaTtmSqueezeError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_h, &d_l, &d_c, &d_fv, cols, rows, length, bb_mult, kc_high, kc_mid, kc_low,
            &mut d_mo, &mut d_sq,
        )?;
        Ok((
            DeviceArrayF32 {
                buf: d_mo,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_sq,
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

    const ONE_SERIES_LEN: usize = 100_000;
    const PARAM_SWEEP: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct TtmBatchState {
        cuda: CudaTtmSqueeze,
        h: Vec<f32>,
        l: Vec<f32>,
        c: Vec<f32>,
        sweep: TtmSqueezeBatchRange,
    }
    impl CudaBenchState for TtmBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .ttm_squeeze_batch_dev(&self.h, &self.l, &self.c, &self.sweep)
                .unwrap();
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaTtmSqueeze::new(0).expect("cuda ttm squeeze");
        let h = gen_series(ONE_SERIES_LEN);
        let mut l = h.clone();
        for v in &mut l {
            *v -= 0.5;
        }
        let mut c = h.clone();
        for v in &mut c {
            *v -= 0.25;
        }
        let sweep = TtmSqueezeBatchRange {
            length: (10, 10 + PARAM_SWEEP - 1, 1),
            bb_mult: (2.0, 2.0, 0.0),
            kc_high: (1.0, 1.0, 0.0),
            kc_mid: (1.5, 1.5, 0.0),
            kc_low: (2.0, 2.0, 0.0),
        };
        Box::new(TtmBatchState {
            cuda,
            h,
            l,
            c,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "ttm_squeeze",
            "one_series_many_params",
            "ttm_squeeze_cuda_batch_dev",
            "100k_x_128",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
