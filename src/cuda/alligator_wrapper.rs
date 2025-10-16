//! CUDA wrapper for the Bill Williams Alligator indicator.
//!
//! Mirrors ALMA/CWMA wrapper shape: policy enums, VRAM checks, PTX load via
//! include_str!(concat!(env!("OUT_DIR"), "/alligator_kernel.ptx")), NON_BLOCKING
//! stream, and public device entry points for:
//!   - one-series × many-params (batch)
//!   - many-series × one-param (time-major)
//!
//! Math category: Recurrence/IIR — three SMMA lines (jaw/teeth/lips) computed
//! in one pass. Warmup/NaN semantics match the scalar path.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::alligator::{AlligatorBatchRange, AlligatorParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaAlligatorError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAlligatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAlligatorError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAlligatorError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaAlligatorError {}

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
pub struct CudaAlligatorPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaAlligatorPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

pub struct DeviceArrayF32Trio {
    pub jaw: DeviceArrayF32,
    pub teeth: DeviceArrayF32,
    pub lips: DeviceArrayF32,
}
impl DeviceArrayF32Trio {
    #[inline] pub fn rows(&self) -> usize { self.jaw.rows }
    #[inline] pub fn cols(&self) -> usize { self.jaw.cols }
}

pub struct CudaAlligatorBatchResult {
    pub outputs: DeviceArrayF32Trio,
    pub combos: Vec<AlligatorParams>,
}

pub struct CudaAlligator {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaAlligatorPolicy,
}

impl CudaAlligator {
    pub fn new(device_id: usize) -> Result<Self, CudaAlligatorError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/alligator_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O3),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        Ok(Self { module, stream, _context: context, policy: CudaAlligatorPolicy::default() })
    }

    pub fn set_policy(&mut self, policy: CudaAlligatorPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaAlligatorPolicy { &self.policy }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &AlligatorBatchRange,
    ) -> Result<(Vec<AlligatorParams>, usize, usize), CudaAlligatorError> {
        if data_f32.is_empty() {
            return Err(CudaAlligatorError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaAlligatorError::InvalidInput("all values are NaN".into()))?;

        // Local grid expansion (usize axes)
        fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end { return vec![start]; }
            (start..=end).step_by(step).collect()
        }
        let jp_v = axis(sweep.jaw_period);
        let jo_v = axis(sweep.jaw_offset);
        let tp_v = axis(sweep.teeth_period);
        let to_v = axis(sweep.teeth_offset);
        let lp_v = axis(sweep.lips_period);
        let lo_v = axis(sweep.lips_offset);
        let mut combos = Vec::with_capacity(
            jp_v.len() * jo_v.len() * tp_v.len() * to_v.len() * lp_v.len() * lo_v.len(),
        );
        for &jp in &jp_v { for &jo in &jo_v { for &tp in &tp_v { for &to in &to_v { for &lp in &lp_v { for &lo in &lo_v {
            combos.push(AlligatorParams {
                jaw_period: Some(jp), jaw_offset: Some(jo),
                teeth_period: Some(tp), teeth_offset: Some(to),
                lips_period: Some(lp), lips_offset: Some(lo),
            });
        }}}}}}
        if combos.is_empty() {
            return Err(CudaAlligatorError::InvalidInput("no parameter combinations".into()));
        }
        let len = data_f32.len();
        for c in &combos {
            let pj = c.jaw_period.unwrap();
            let pt = c.teeth_period.unwrap();
            let pl = c.lips_period.unwrap();
            if pj == 0 || pt == 0 || pl == 0 { return Err(CudaAlligatorError::InvalidInput("period must be > 0".into())); }
            if pj > len || pt > len || pl > len {
                return Err(CudaAlligatorError::InvalidInput("period exceeds data length".into()));
            }
            let need = pj.max(pt).max(pl);
            if len - first_valid < need {
                return Err(CudaAlligatorError::InvalidInput("not enough valid data".into()));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_jp: &DeviceBuffer<i32>,
        d_jo: &DeviceBuffer<i32>,
        d_tp: &DeviceBuffer<i32>,
        d_to: &DeviceBuffer<i32>,
        d_lp: &DeviceBuffer<i32>,
        d_lo: &DeviceBuffer<i32>,
        first_valid: usize,
        len: usize,
        n: usize,
        d_jaw: &mut DeviceBuffer<f32>,
        d_teeth: &mut DeviceBuffer<f32>,
        d_lips: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlligatorError> {
        if len == 0 || n == 0 { return Err(CudaAlligatorError::InvalidInput("empty geometry".into())); }
        if first_valid > i32::MAX as usize || len > i32::MAX as usize || n > i32::MAX as usize {
            return Err(CudaAlligatorError::InvalidInput("geometry exceeds i32::MAX".into()));
        }
        let mut func = self
            .module
            .get_function("alligator_batch_f32")
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch { BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x, _ => 128 };
        let grid_x = ((n as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_prices = d_prices.as_device_ptr().as_raw();
            let mut p_jp = d_jp.as_device_ptr().as_raw();
            let mut p_jo = d_jo.as_device_ptr().as_raw();
            let mut p_tp = d_tp.as_device_ptr().as_raw();
            let mut p_to = d_to.as_device_ptr().as_raw();
            let mut p_lp = d_lp.as_device_ptr().as_raw();
            let mut p_lo = d_lo.as_device_ptr().as_raw();
            let mut fv = first_valid as i32;
            let mut series_len = len as i32;
            let mut combos = n as i32;
            let mut p_out_j = d_jaw.as_device_ptr().as_raw();
            let mut p_out_t = d_teeth.as_device_ptr().as_raw();
            let mut p_out_l = d_lips.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_prices as *mut _ as *mut c_void,
                &mut p_jp as *mut _ as *mut c_void,
                &mut p_jo as *mut _ as *mut c_void,
                &mut p_tp as *mut _ as *mut c_void,
                &mut p_to as *mut _ as *mut c_void,
                &mut p_lp as *mut _ as *mut c_void,
                &mut p_lo as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut series_len as *mut _ as *mut c_void,
                &mut combos as *mut _ as *mut c_void,
                &mut p_out_j as *mut _ as *mut c_void,
                &mut p_out_t as *mut _ as *mut c_void,
                &mut p_out_l as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn alligator_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &AlligatorBatchRange,
    ) -> Result<CudaAlligatorBatchResult, CudaAlligatorError> {
        let (combos, first, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n = combos.len();

        // VRAM estimate (prices + 6 params + 3 outputs)
        let prices_bytes = len * std::mem::size_of::<f32>();
        let params_bytes = 6 * n * std::mem::size_of::<i32>();
        let out_bytes = 3 * n * len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64usize * 1024 * 1024;
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAlligatorError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (required as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let jaw_p: Vec<i32> = combos.iter().map(|c| c.jaw_period.unwrap() as i32).collect();
        let jaw_o: Vec<i32> = combos.iter().map(|c| c.jaw_offset.unwrap() as i32).collect();
        let tee_p: Vec<i32> = combos.iter().map(|c| c.teeth_period.unwrap() as i32).collect();
        let tee_o: Vec<i32> = combos.iter().map(|c| c.teeth_offset.unwrap() as i32).collect();
        let lip_p: Vec<i32> = combos.iter().map(|c| c.lips_period.unwrap() as i32).collect();
        let lip_o: Vec<i32> = combos.iter().map(|c| c.lips_offset.unwrap() as i32).collect();

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let d_jp = DeviceBuffer::from_slice(&jaw_p)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let d_jo = DeviceBuffer::from_slice(&jaw_o)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let d_tp = DeviceBuffer::from_slice(&tee_p)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let d_to = DeviceBuffer::from_slice(&tee_o)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let d_lp = DeviceBuffer::from_slice(&lip_p)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let d_lo = DeviceBuffer::from_slice(&lip_o)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;

        let mut d_jaw: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n * len) }
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let mut d_teeth: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n * len) }
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let mut d_lips: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n * len) }
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices, &d_jp, &d_jo, &d_tp, &d_to, &d_lp, &d_lo, first, len, n, &mut d_jaw,
            &mut d_teeth, &mut d_lips,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;

        let outputs = DeviceArrayF32Trio {
            jaw: DeviceArrayF32 { buf: d_jaw, rows: n, cols: len },
            teeth: DeviceArrayF32 { buf: d_teeth, rows: n, cols: len },
            lips: DeviceArrayF32 { buf: d_lips, rows: n, cols: len },
        };
        Ok(CudaAlligatorBatchResult { outputs, combos })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlligatorParams,
    ) -> Result<(Vec<i32>, usize, usize, usize, usize, usize, usize), CudaAlligatorError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAlligatorError::InvalidInput("invalid cols/rows".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaAlligatorError::InvalidInput("data length != cols*rows".into()));
        }
        let mut first_valids = vec![0i32; cols];
        for j in 0..cols {
            let mut first = rows as i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + j];
                if !v.is_nan() { first = t as i32; break; }
            }
            first_valids[j] = first.min(rows as i32 - 1).max(0);
        }
        let jp = params.jaw_period.unwrap_or(13);
        let jo = params.jaw_offset.unwrap_or(8);
        let tp = params.teeth_period.unwrap_or(8);
        let to = params.teeth_offset.unwrap_or(5);
        let lp = params.lips_period.unwrap_or(5);
        let lo = params.lips_offset.unwrap_or(3);
        if jp == 0 || tp == 0 || lp == 0 {
            return Err(CudaAlligatorError::InvalidInput("period must be > 0".into()));
        }
        Ok((first_valids, jp, jo, tp, to, lp, lo))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        jp: usize,
        jo: usize,
        tp: usize,
        to: usize,
        lp: usize,
        lo: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_jaw_tm: &mut DeviceBuffer<f32>,
        d_teeth_tm: &mut DeviceBuffer<f32>,
        d_lips_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlligatorError> {
        let mut func = self
            .module
            .get_function("alligator_many_series_one_param_f32")
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x, _ => 128 };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_prices = d_prices_tm.as_device_ptr().as_raw();
            let mut jp_i = jp as i32; let mut jo_i = jo as i32;
            let mut tp_i = tp as i32; let mut to_i = to as i32;
            let mut lp_i = lp as i32; let mut lo_i = lo as i32;
            let mut cols_i = cols as i32; let mut rows_i = rows as i32;
            let mut p_first = d_first_valids.as_device_ptr().as_raw();
            let mut p_out_j = d_jaw_tm.as_device_ptr().as_raw();
            let mut p_out_t = d_teeth_tm.as_device_ptr().as_raw();
            let mut p_out_l = d_lips_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_prices as *mut _ as *mut c_void,
                &mut jp_i as *mut _ as *mut c_void,
                &mut jo_i as *mut _ as *mut c_void,
                &mut tp_i as *mut _ as *mut c_void,
                &mut to_i as *mut _ as *mut c_void,
                &mut lp_i as *mut _ as *mut c_void,
                &mut lo_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_out_j as *mut _ as *mut c_void,
                &mut p_out_t as *mut _ as *mut c_void,
                &mut p_out_l as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn alligator_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlligatorParams,
    ) -> Result<DeviceArrayF32Trio, CudaAlligatorError> {
        let (first_valids, jp, jo, tp, to, lp, lo) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let mut d_jaw_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let mut d_teeth_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;
        let mut d_lips_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm, jp, jo, tp, to, lp, lo, cols, rows, &d_first_valids, &mut d_jaw_tm,
            &mut d_teeth_tm, &mut d_lips_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaAlligatorError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32Trio {
            jaw: DeviceArrayF32 { buf: d_jaw_tm, rows, cols },
            teeth: DeviceArrayF32 { buf: d_teeth_tm, rows, cols },
            lips: DeviceArrayF32 { buf: d_lips_tm, rows, cols },
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    struct AlligatorBatchState {
        cuda: CudaAlligator,
        data: Vec<f32>,
        sweep: AlligatorBatchRange,
    }
    impl CudaBenchState for AlligatorBatchState {
        fn launch(&mut self) {
            let _ = self.cuda.alligator_batch_dev(&self.data, &self.sweep).unwrap();
        }
    }

    fn prep_alligator_batch() -> AlligatorBatchState {
        let mut cuda = CudaAlligator::new(0).expect("cuda alligator");
        cuda.set_policy(CudaAlligatorPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto });
        let len = 60_000usize;
        let mut data = vec![f32::NAN; len];
        for i in 12..len { let x = i as f32; data[i] = (x * 0.0013).sin() + 0.0002 * x; }
        let sweep = AlligatorBatchRange {
            jaw_period: (10, 34, 8), jaw_offset: (3, 8, 1),
            teeth_period: (6, 21, 5), teeth_offset: (2, 6, 1),
            lips_period: (3, 13, 5), lips_offset: (1, 4, 1),
        };
        AlligatorBatchState { cuda, data, sweep }
    }

    struct AlligatorManySeriesState {
        cuda: CudaAlligator,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: AlligatorParams,
    }
    impl CudaBenchState for AlligatorManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .alligator_many_series_one_param_time_major_dev(
                    &self.data_tm, self.cols, self.rows, &self.params,
                )
                .unwrap();
        }
    }

    fn prep_alligator_many_series() -> AlligatorManySeriesState {
        let mut cuda = CudaAlligator::new(0).expect("cuda alligator");
        cuda.set_policy(CudaAlligatorPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::OneD { block_x: 256 } });
        let cols = 256usize;
        let rows = 1_000_000usize / cols * cols; // ensure multiple of cols
        let mut data_tm = vec![f32::NAN; cols * rows];
        for t in 8..rows { for j in 0..cols { let idx = t * cols + j; let x = (t as f32) + (j as f32) * 0.07; data_tm[idx] = (x * 0.0021).cos() + 0.0006 * x; } }
        let params = AlligatorParams::default();
        AlligatorManySeriesState { cuda, data_tm, cols, rows, params }
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        use crate::cuda::bench::CudaBenchScenario;
        vec![
            CudaBenchScenario::new(
                "alligator",
                "batch_dev",
                "alligator_cuda_batch_dev",
                "60k_x_many",
                || Box::new(prep_alligator_batch()),
            )
            .with_inner_iters(6),
            CudaBenchScenario::new(
                "alligator",
                "many_series_one_param",
                "alligator_cuda_many_series_one_param",
                "256x1m",
                || Box::new(prep_alligator_many_series()),
            )
            .with_inner_iters(3),
        ]
    }
}
