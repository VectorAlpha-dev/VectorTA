//! CUDA scaffolding for Laguerre RSI (LRSI)
//!
//! Mirrors ALMA wrapper conventions: policy enums, NON_BLOCKING stream, PTX
//! load with DetermineTargetFromContext + OptLevel(O2) fallbacks, VRAM checks
//! with ~64MB headroom, and optional bench profiles. Numeric semantics match
//! the scalar Rust implementation in src/indicators/lrsi.rs (warmup/NaN rules).

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::cuda::moving_averages::alma_wrapper::{BatchKernelSelected, ManySeriesKernelSelected};
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::lrsi::{LrsiBatchRange, LrsiParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaLrsiError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaLrsiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaLrsiPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

pub struct CudaLrsi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaLrsiPolicy,
    
    sm_count: u32,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaLrsi {
    pub fn new(device_id: usize) -> Result<Self, CudaLrsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/lrsi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
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
            policy: CudaLrsiPolicy::default(),
            sm_count,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn pick_block_grid(
        &self,
        work_items: usize,
        policy_block_x: Option<u32>,
        default_block: u32,
    ) -> (u32, u32) {
        const WARP: u32 = 32;
        const MAX_BLOCK: u32 = 256;

        let sm = self.sm_count.max(1);
        let target_blocks = sm.saturating_mul(4);

        let mut block_x = policy_block_x
            .unwrap_or(default_block)
            .max(WARP)
            .min(MAX_BLOCK);

        let mut grid_x = if work_items == 0 {
            1
        } else {
            ((work_items as u32) + block_x - 1) / block_x
        };

        if grid_x < target_blocks && work_items > 0 {
            let mut b = ((work_items as u32) + target_blocks - 1) / target_blocks;
            if b < WARP {
                b = WARP;
            }
            b = ((b + WARP - 1) / WARP) * WARP;
            b = b.min(MAX_BLOCK);
            block_x = b;

            grid_x = ((work_items as u32) + block_x - 1) / block_x;
            if grid_x == 0 {
                grid_x = 1;
            }
        }

        (block_x, grid_x.max(1))
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaLrsiPolicy) {
        self.policy = p;
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
    pub fn synchronize(&self) -> Result<(), CudaLrsiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    
    pub fn lrsi_batch_dev(
        &mut self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &LrsiBatchRange,
    ) -> Result<DeviceArrayF32, CudaLrsiError> {
        if high_f32.is_empty() || low_f32.len() != high_f32.len() {
            return Err(CudaLrsiError::InvalidInput(
                "high/low empty or length mismatch".into(),
            ));
        }
        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaLrsiError::InvalidInput("no alpha values".into()));
        }

        
        let len = high_f32.len();
        let mut prices = vec![f32::NAN; len];
        let mut first = None;
        for i in 0..len {
            let p = 0.5f32 * (high_f32[i] + low_f32[i]);
            prices[i] = p;
            if first.is_none() && p.is_finite() {
                first = Some(i);
            }
        }
        let first = first.ok_or_else(|| CudaLrsiError::InvalidInput("all prices NaN".into()))?;
        if len - first < 4 {
            return Err(CudaLrsiError::InvalidInput(format!(
                "not enough valid data: needed 4, have {}",
                len - first
            )));
        }

        
        let mut alphas = Vec::with_capacity(combos.len());
        for p in &combos {
            let a = p.alpha.unwrap_or(0.2);
            if !(a > 0.0 && a < 1.0) {
                return Err(CudaLrsiError::InvalidInput("alpha out of range".into()));
            }
            alphas.push(a as f32);
        }

        
        let in_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaLrsiError::InvalidInput("size overflow".into()))?;
        let param_bytes = combos
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaLrsiError::InvalidInput("size overflow".into()))?;
        let out_bytes = combos
            .len()
            .checked_mul(len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaLrsiError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(param_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaLrsiError::InvalidInput("size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaLrsiError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaLrsiError::InvalidInput("insufficient VRAM".into()));
            }
        }

        let d_prices = DeviceBuffer::from_slice(&prices)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(
                combos
                    .len()
                    .checked_mul(len)
                    .ok_or_else(|| CudaLrsiError::InvalidInput("output size overflow".into()))?,
            )?
        };

        self.launch_batch_kernel(&d_prices, &d_alphas, len, first, combos.len(), &mut d_out)?;
        self.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len })
    }

    fn launch_batch_kernel(
        &mut self,
        d_prices: &DeviceBuffer<f32>,
        d_alphas: &DeviceBuffer<f32>,
        len: usize,
        first: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLrsiError> {
        if len == 0 || n_combos == 0 {
            return Ok(());
        }
        if len > i32::MAX as usize || n_combos > i32::MAX as usize || first > i32::MAX as usize {
            return Err(CudaLrsiError::InvalidInput(
                "inputs exceed kernel limits".into(),
            ));
        }

        
        let policy_block = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => Some(block_x),
            _ => None,
        };
        
        let (block_x, grid_x) = self.pick_block_grid(n_combos, policy_block, 256);

        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if grid_x == 0 || block_x == 0 {
            return Err(CudaLrsiError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        self.last_batch = Some(BatchKernelSelected::Plain { block_x });
        self.maybe_log_batch_debug();

        let func = self
            .module
            .get_function("lrsi_batch_f32")
            .map_err(|_| CudaLrsiError::MissingKernelSymbol { name: "lrsi_batch_f32" })?;

        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut a_ptr = d_alphas.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut a_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn maybe_log_batch_debug(&mut self) {
        if !self.debug_batch_logged && std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[LRSI CUDA] batch kernel selected: {:?}", sel);
                self.debug_batch_logged = true;
            }
        }
    }

    fn maybe_log_many_debug(&mut self) {
        if !self.debug_many_logged && std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[LRSI CUDA] many-series kernel selected: {:?}", sel);
                self.debug_many_logged = true;
            }
        }
    }

    
    pub fn lrsi_many_series_one_param_time_major_dev(
        &mut self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        alpha: f64,
    ) -> Result<DeviceArrayF32, CudaLrsiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaLrsiError::InvalidInput(
                "cols/rows must be positive".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaLrsiError::InvalidInput("cols*rows overflow".into()))?;
        if high_tm_f32.len() != elems || low_tm_f32.len() != elems {
            return Err(CudaLrsiError::InvalidInput("matrix shape mismatch".into()));
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(CudaLrsiError::InvalidInput("alpha out of range".into()));
        }

        
        let mut prices_tm = vec![f32::NAN; elems];
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let idx = t * cols + s;
                let p = 0.5f32 * (high_tm_f32[idx] + low_tm_f32[idx]);
                prices_tm[idx] = p;
                if fv.is_none() && p.is_finite() {
                    fv = Some(t);
                }
            }
            let fv =
                fv.ok_or_else(|| CudaLrsiError::InvalidInput(format!("series {s} all NaN")))?;
            if rows - fv < 4 {
                return Err(CudaLrsiError::InvalidInput(format!(
                    "series {s} insufficient data: need 4, have {}",
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }

        
        let in_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaLrsiError::InvalidInput("prices byte size overflow".into()))?;
        let fv_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaLrsiError::InvalidInput("first_valid byte size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaLrsiError::InvalidInput("output byte size overflow".into()))?;
        let required = in_bytes
            .checked_add(fv_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaLrsiError::InvalidInput("required VRAM size overflow".into()))?;
        let head = 64usize * 1024 * 1024;
        if !Self::will_fit(required, head) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaLrsiError::OutOfMemory {
                    required,
                    free,
                    headroom: head,
                });
            } else {
                return Err(CudaLrsiError::InvalidInput("insufficient VRAM".into()));
            }
        }

        let d_prices_tm = DeviceBuffer::from_slice(&prices_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(elems)
                .map_err(CudaLrsiError::Cuda)?
        };

        self.launch_many_series_kernel(
            &d_prices_tm,
            alpha as f32,
            cols,
            rows,
            &d_first,
            &mut d_out_tm,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out_tm, rows, cols })
    }

    fn launch_many_series_kernel(
        &mut self,
        d_prices_tm: &DeviceBuffer<f32>,
        alpha: f32,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLrsiError> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }
        if cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaLrsiError::InvalidInput(
                "inputs exceed kernel limits".into(),
            ));
        }
        let policy_block = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => Some(block_x),
            _ => None,
        };
        let (block_x, grid_x) = self.pick_block_grid(cols, policy_block, 256);
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if grid_x == 0 || block_x == 0 {
            return Err(CudaLrsiError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        self.last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        self.maybe_log_many_debug();

        let func = self
            .module
            .get_function("lrsi_many_series_one_param_f32")
            .map_err(|_| CudaLrsiError::MissingKernelSymbol {
                name: "lrsi_many_series_one_param_f32",
            })?;

        unsafe {
            let mut p_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut alpha_v = alpha;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut alpha_v as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }
}

fn expand_grid(r: &LrsiBatchRange) -> Result<Vec<LrsiParams>, CudaLrsiError> {
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaLrsiError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(CudaLrsiError::InvalidInput(format!(
                    "invalid alpha range: start={start}, end={end}, step={step}"
                )));
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end {
            v.push(x);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaLrsiError::InvalidInput(format!(
                "invalid alpha range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(v)
    }

    let alphas = axis_f64(r.alpha)?;
    let mut out = Vec::with_capacity(alphas.len());
    for &a in &alphas {
        out.push(LrsiParams { alpha: Some(a) });
    }
    Ok(out)
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const LEN: usize = 1_000_000;
    const ROWS: usize = 256; 

    struct LrsiBatchState {
        cuda: CudaLrsi,
        d_prices: DeviceBuffer<f32>,
        d_alphas: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for LrsiBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_alphas,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("lrsi launch_batch_kernel");
            let _ = self.cuda.synchronize();
        }
    }
    fn prep_batch() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaLrsi::new(0).expect("cuda lrsi");
        cuda.set_policy(CudaLrsiPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });
        let base = gen_series(LEN);
        let mut high = base.clone();
        let mut low = base.clone();
        for i in 0..LEN {
            if base[i].is_nan() {
                continue;
            }
            let off = (0.003f32 * (i as f32)).sin().abs() + 0.1;
            high[i] = base[i] + off;
            low[i] = base[i] - off;
        }
        let sweep = LrsiBatchRange {
            alpha: (0.05, 0.80, (0.80 - 0.05) / (ROWS as f64 - 1.0)),
        };
        let combos = expand_grid(&sweep).expect("expand_grid");
        let alphas: Vec<f32> = combos.iter().map(|c| c.alpha.unwrap() as f32).collect();
        let mut prices = vec![f32::NAN; LEN];
        for i in 0..LEN {
            prices[i] = 0.5f32 * (high[i] + low[i]);
        }
        let first_valid = prices.iter().position(|v| v.is_finite()).unwrap_or(0);
        let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
        let d_alphas = DeviceBuffer::from_slice(&alphas).expect("d_alphas");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(combos.len() * LEN) }.expect("d_out");
        Box::new(LrsiBatchState {
            cuda,
            d_prices,
            d_alphas,
            d_out,
            len: LEN,
            first_valid,
            n_combos: combos.len(),
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "lrsi",
            "batch_dev",
            "lrsi_cuda_batch_dev",
            "1m_x_256",
            prep_batch,
        )
        .with_inner_iters(4)]
    }
}
