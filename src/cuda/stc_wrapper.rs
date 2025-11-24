#![cfg(feature = "cuda")]

//! CUDA wrapper for STC (Schaff Trend Cycle)
//!
//! Parity targets with ALMA/CWMA-style wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/stc_kernel.ptx")) with
//!   DetermineTargetFromContext + OptLevel O2 fallbacks.
//! - NON_BLOCKING stream.
//! - Warmup/NaN semantics match scalar stc.rs exactly: warm = first_valid + max(fast,slow,k,d) - 1.
//! - Batch (one-series × many-params) and many-series × one-param (time-major) entry points.
//! - VRAM checks + grid chunking (<= 65_535 rows per launch).
//! - Math pattern: Recurrence/IIR. We implement the classic EMA/EMA path on device. SMA/SMA is
//!   a future extension; exotic MA types can be composed by precomputing MAs via the CUDA MA
//!   selector and applying a small, dedicated kernel (not included here for brevity).

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::stc::{StcBatchRange, StcParams};
use cust::context::Context;
use cust::context::{CacheConfig, SharedMemoryConfig};
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys::{cuFuncSetAttribute, CUfunction_attribute_enum as CUfuncAttr};
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

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
pub struct CudaStcPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaStcPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Error, Debug)]
pub enum CudaStcError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("out of memory: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaStc {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaStcPolicy,
}

impl CudaStc {
    #[inline(always)]
    fn stc_batch_smem_bytes(max_k: usize) -> usize {
        // 2 float rings + 4 int index-deques
        max_k * (2 * size_of::<f32>() + 4 * size_of::<i32>())
    }
    pub fn new(device_id: usize) -> Result<Self, CudaStcError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/stc_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[])?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaStcPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, policy: CudaStcPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaStcPolicy {
        &self.policy
    }
    pub fn synchronize(&self) -> Result<(), CudaStcError> {
        self.stream.synchronize().map_err(Into::into)
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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn grid_x_chunks(n_rows: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..n_rows).step_by(MAX).map(move |start| {
            let len = (n_rows - start).min(MAX);
            (start, len)
        })
    }

    fn expand_axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaStcError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            return Ok((start..=end).step_by(step.max(1)).collect());
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaStcError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(v)
    }

    fn expand_grid(sweep: &StcBatchRange) -> Result<Vec<StcParams>, CudaStcError> {
        let fs = Self::expand_axis(sweep.fast_period)?;
        let ss = Self::expand_axis(sweep.slow_period)?;
        let ks = Self::expand_axis(sweep.k_period)?;
        let ds = Self::expand_axis(sweep.d_period)?;

        let cap = fs
            .len()
            .checked_mul(ss.len())
            .and_then(|v| v.checked_mul(ks.len()))
            .and_then(|v| v.checked_mul(ds.len()))
            .ok_or_else(|| {
                CudaStcError::InvalidInput("parameter grid size overflow".into())
            })?;

        let mut out = Vec::with_capacity(cap);
        for &f in &fs {
            for &s in &ss {
                for &k in &ks {
                    for &d in &ds {
                        out.push(StcParams {
                            fast_period: Some(f),
                            slow_period: Some(s),
                            k_period: Some(k),
                            d_period: Some(d),
                            fast_ma_type: None,
                            slow_ma_type: None,
                        });
                    }
                }
            }
        }
        out
    }

    fn validate_first_valid(data: &[f32], max_needed: usize) -> Result<usize, CudaStcError> {
        if data.is_empty() {
            return Err(CudaStcError::InvalidInput("empty data".into()));
        }
        let first = data
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaStcError::InvalidInput("all values are NaN".into()))?;
        if data.len() - first < max_needed {
            return Err(CudaStcError::InvalidInput("not enough valid data".into()));
        }
        Ok(first)
    }

    // ---------- Batch: one series × many params (EMA/EMA path) ----------
    pub fn stc_batch_dev(
        &self,
        data: &[f32],
        sweep: &StcBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<StcParams>), CudaStcError> {
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaStcError::InvalidInput("empty sweep".into()));
        }

        let len = data.len();

        let max_needed = combos
            .iter()
            .map(|c| {
                c.fast_period
                    .unwrap()
                    .max(c.slow_period.unwrap())
                    .max(c.k_period.unwrap())
                    .max(c.d_period.unwrap())
            })
            .max()
            .unwrap();
        let first_valid = Self::validate_first_valid(data, max_needed)?;

        // VRAM estimate: input + param arrays + output
        let rows = combos.len();
        let rows_len = rows
            .checked_mul(len)
            .ok_or_else(|| CudaStcError::InvalidInput("rows*len overflow".into()))?;
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let in_and_out_bytes = len
            .checked_add(rows_len)
            .and_then(|v| v.checked_mul(elem_f32))
            .ok_or_else(|| CudaStcError::InvalidInput("byte size overflow".into()))?;
        let param_bytes = rows
            .checked_mul(4)
            .and_then(|v| v.checked_mul(elem_i32))
            .ok_or_else(|| CudaStcError::InvalidInput("byte size overflow".into()))?;
        let req = in_and_out_bytes
            .checked_add(param_bytes)
            .ok_or_else(|| CudaStcError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(req, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaStcError::OutOfMemory {
                    required: req,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaStcError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // Upload input
        let h = LockedBuffer::from_slice(data)?;
        let mut d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        unsafe {
            d_prices
                .async_copy_from(&h, &self.stream)?;
        }

        // Prepare param arrays (host pinned)
        let fasts: Vec<i32> = combos
            .iter()
            .map(|c| c.fast_period.unwrap() as i32)
            .collect();
        let slows: Vec<i32> = combos
            .iter()
            .map(|c| c.slow_period.unwrap() as i32)
            .collect();
        let ks: Vec<i32> = combos.iter().map(|c| c.k_period.unwrap() as i32).collect();
        let ds: Vec<i32> = combos.iter().map(|c| c.d_period.unwrap() as i32).collect();

        let h_f = LockedBuffer::from_slice(&fasts)?;
        let h_s = LockedBuffer::from_slice(&slows)?;
        let h_k = LockedBuffer::from_slice(&ks)?;
        let h_d = LockedBuffer::from_slice(&ds)?;

        // Upload parameter arrays once (async)
        let mut d_f: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(rows, &self.stream) }?;
        let mut d_s: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(rows, &self.stream) }?;
        let mut d_k: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(rows, &self.stream) }?;
        let mut d_d: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(rows, &self.stream) }?;
        unsafe {
            d_f.async_copy_from(&h_f, &self.stream)
                ?;
            d_s.async_copy_from(&h_s, &self.stream)
                ?;
            d_k.async_copy_from(&h_k, &self.stream)
                ?;
            d_d.async_copy_from(&h_d, &self.stream)
                ?;
        }

        // Output buffer
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows_len, &self.stream) }?;

        // Launch in chunks to respect grid limit
        let mut func = self
            .module
            .get_function("stc_batch_f32")
            .map_err(|_| CudaStcError::MissingKernelSymbol { name: "stc_batch_f32" })?;
        // Kernel is sequential per-row; use 1 thread per block by default for best occupancy.
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 1,
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
        };
        // Prefer shared carveout and 4-byte banks for FP32 traffic
        func.set_cache_config(CacheConfig::PreferShared)
            ?;
        func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize)
            ?;

        for (start, count) in Self::grid_x_chunks(rows) {
            // Per-chunk dynamic shared mem sizing based on local max(k)
            let local_max_k = ks[start..start + count]
                .iter()
                .copied()
                .map(|v| v as usize)
                .max()
                .unwrap_or(1);
            let shmem_bytes = Self::stc_batch_smem_bytes(local_max_k);
            if shmem_bytes > 48 * 1024 {
                unsafe {
                    let raw = func.to_raw();
                    let _ = cuFuncSetAttribute(
                        raw,
                        CUfuncAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shmem_bytes as i32,
                    );
                }
            }

            unsafe {
                let grid: GridSize = (count as u32, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let mut p_ptr = d_prices.as_device_ptr().as_raw();
                // param pointers offset to chunk start
                let mut f_ptr = d_f.as_device_ptr().add(start).as_raw();
                let mut s_ptr = d_s.as_device_ptr().add(start).as_raw();
                let mut k_ptr = d_k.as_device_ptr().add(start).as_raw();
                let mut d_ptr = d_d.as_device_ptr().add(start).as_raw();
                let mut n_i = len as i32;
                let mut fv_i = first_valid as i32;
                let mut r_i = count as i32;
                let mut mk_i = local_max_k as i32;
                // out pointer offset to the start of this chunk
                let mut o_ptr = d_out
                    .as_device_ptr()
                    .add(start * len)
                    .as_raw();
                let mut args: [*mut c_void; 10] = [
                    &mut p_ptr as *mut _ as *mut c_void,
                    &mut f_ptr as *mut _ as *mut c_void,
                    &mut s_ptr as *mut _ as *mut c_void,
                    &mut k_ptr as *mut _ as *mut c_void,
                    &mut d_ptr as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut r_i as *mut _ as *mut c_void,
                    &mut mk_i as *mut _ as *mut c_void,
                    &mut o_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(
                    &func,
                    grid,
                    block,
                    shmem_bytes.try_into().unwrap_or(u32::MAX),
                    &mut args,
                )?;
            }
        }

        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            },
            combos,
        ))
        
    }

    // ---------- Many-series × one param (time-major; EMA/EMA) ----------
    pub fn stc_many_series_one_param_time_major_dev(
        &self,
        data_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &StcParams,
    ) -> Result<DeviceArrayF32, CudaStcError> {
        if cols == 0 || rows == 0 {
            return Err(CudaStcError::InvalidInput("empty matrix".into()));
        }
        let cells = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaStcError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm.len() != cells {
            return Err(CudaStcError::InvalidInput("matrix shape mismatch".into()));
        }

        let fast = params.fast_period.unwrap_or(23);
        let slow = params.slow_period.unwrap_or(50);
        
        let k = params.k_period.unwrap_or(10);
        let d = params.d_period.unwrap_or(3);

        // Per-series first_valids
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                
                if data_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            let fv = first_valids[s] as usize;
            let warm = fv + fast.max(slow).max(k).max(d) - 1;
            
            if warm >= rows {
                return Err(CudaStcError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        // VRAM estimate: inputs + first_valids + outputs
        let elem = std::mem::size_of::<f32>();
        let data_bytes = cells
            .checked_mul(elem)
            .ok_or_else(|| CudaStcError::InvalidInput("byte size overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaStcError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = cells
            .checked_mul(elem)
            .ok_or_else(|| CudaStcError::InvalidInput("byte size overflow".into()))?;
        let req = data_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaStcError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(req, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaStcError::OutOfMemory {
                    required: req,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaStcError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // Upload inputs (pinned + async)
        let h_tm = LockedBuffer::from_slice(data_tm)?;
        let h_first = LockedBuffer::from_slice(&first_valids)?;
        let mut d_data: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cells, &self.stream) }?;
        let mut d_first: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(cols, &self.stream) }?;
        unsafe {
            d_data
                .async_copy_from(&h_tm, &self.stream)?;
            d_first
                .async_copy_from(&h_first, &self.stream)?;
        }
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cells, &self.stream) }?;

        // Launch kernel
        
        let func = self
            .module
            .get_function("stc_many_series_one_param_f32")
            .map_err(|_| CudaStcError::MissingKernelSymbol {
                name: "stc_many_series_one_param_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        unsafe {
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut d_ptr = d_data.as_device_ptr().as_raw();
            let mut f_ptr = d_first.as_device_ptr().as_raw();
            
            let mut c_i = cols as i32;
            let mut r_i = rows as i32;
            let mut fast_i = fast as i32;
            let mut slow_i = slow as i32;
            let mut k_i = k as i32;
            let mut d_i = d as i32;
            let mut o_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut d_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut c_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut fast_i as *mut _ as *mut c_void,
                &mut slow_i as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
                &mut d_i as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// -------- Bench Profiles --------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        // Batch: 100k samples, modest parameter grid
        v.push(
            CudaBenchScenario::new(
                "stc",
                "one_series_many_params",
                "stc/batch",
                "100k x 64",
                || {
                    struct State {
                        cuda: CudaStc,
                        data: Vec<f32>,
                        sweep: StcBatchRange,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            let _ = self.cuda.stc_batch_dev(&self.data, &self.sweep);
                        }
                    }
                    let n = 100_000usize;
                    let mut data = vec![f32::NAN; n];
                    for i in 200..n {
                        let x = i as f32;
                        data[i] = (x * 0.0013).sin() + 0.0002 * x;
                    }
                    let sweep = StcBatchRange {
                        fast_period: (10, 25, 5),
                        slow_period: (30, 60, 10),
                        k_period: (10, 10, 0),
                        d_period: (3, 3, 0),
                    };
                    Box::new(State {
                        cuda: CudaStc::new(0).unwrap(),
                        data,
                        sweep,
                    })
                },
            )
            .with_sample_size(20),
        );
        v.push(
            CudaBenchScenario::new(
                "stc",
                "one_series_many_params",
                "stc/batch",
                "100k x 64",
                || {
                    struct State {
                        cuda: CudaStc,
                        data: Vec<f32>,
                        sweep: StcBatchRange,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            let _ = self.cuda.stc_batch_dev(&self.data, &self.sweep);
                        }
                    }
                    let n = 100_000usize;
                    let mut data = vec![f32::NAN; n];
                    for i in 200..n {
                        let x = i as f32;
                        data[i] = (x * 0.0013).sin() + 0.0002 * x;
                    }
                    let sweep = StcBatchRange {
                        fast_period: (10, 25, 5),
                        slow_period: (30, 60, 10),
                        k_period: (10, 10, 0),
                        d_period: (3, 3, 0),
                    };
                    Box::new(State {
                        cuda: CudaStc::new(0).unwrap(),
                        data,
                        sweep,
                    })
                },
            )
            .with_sample_size(20),
        );

        // Many-series: 512 series x 2048 rows
        v.push(
            CudaBenchScenario::new(
                "stc",
                "many_series_one_param",
                "stc/many_series",
                "2048r x 512c",
                || {
                    struct State {
                        cuda: CudaStc,
                        data_tm: Vec<f32>,
                        cols: usize,
                        rows: usize,
                        params: StcParams,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            let _ = self.cuda.stc_many_series_one_param_time_major_dev(
                                &self.data_tm,
                                self.cols,
                                self.rows,
                                &self.params,
                            );
                        }
                    }
                    let cols = 512usize;
                    let rows = 2048usize;
                    let mut tm = vec![f32::NAN; cols * rows];
                    for s in 0..cols {
                        for t in s..rows {
                            let x = t as f32 + (s as f32) * 0.1;
                            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                        }
                    }
                    let params = StcParams {
                        fast_period: Some(23),
                        slow_period: Some(50),
                        k_period: Some(10),
                        d_period: Some(3),
                        fast_ma_type: None,
                        slow_ma_type: None,
                    };
                    Box::new(State {
                        cuda: CudaStc::new(0).unwrap(),
                        data_tm: tm,
                        cols,
                        rows,
                        params,
                    })
                },
            )
            .with_sample_size(20),
        );
        v.push(
            CudaBenchScenario::new(
                "stc",
                "many_series_one_param",
                "stc/many_series",
                "2048r x 512c",
                || {
                    struct State {
                        cuda: CudaStc,
                        data_tm: Vec<f32>,
                        cols: usize,
                        rows: usize,
                        params: StcParams,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            let _ = self.cuda.stc_many_series_one_param_time_major_dev(
                                &self.data_tm,
                                self.cols,
                                self.rows,
                                &self.params,
                            );
                        }
                    }
                    let cols = 512usize;
                    let rows = 2048usize;
                    let mut tm = vec![f32::NAN; cols * rows];
                    for s in 0..cols {
                        for t in s..rows {
                            let x = t as f32 + (s as f32) * 0.1;
                            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                        }
                    }
                    let params = StcParams {
                        fast_period: Some(23),
                        slow_period: Some(50),
                        k_period: Some(10),
                        d_period: Some(3),
                        fast_ma_type: None,
                        slow_ma_type: None,
                    };
                    Box::new(State {
                        cuda: CudaStc::new(0).unwrap(),
                        data_tm: tm,
                        cols,
                        rows,
                        params,
                    })
                },
            )
            .with_sample_size(20),
        );

        v
    }
}
