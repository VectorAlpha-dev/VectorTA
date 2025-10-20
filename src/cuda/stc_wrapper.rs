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

#[derive(Debug)]
pub enum CudaStcError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaStcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaStcError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaStcError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaStcError {}

pub struct CudaStc {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaStcPolicy,
}

impl CudaStc {
    pub fn new(device_id: usize) -> Result<Self, CudaStcError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaStcError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/stc_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaStcError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaStcError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
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
        self.stream
            .synchronize()
            .map_err(|e| CudaStcError::Cuda(e.to_string()))
    }

    #[inline]
    fn device_mem_ok(bytes: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => bytes.saturating_add(64 * 1024 * 1024) <= free,
            Err(_) => true,
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

    fn expand_axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        }
    }
    fn expand_grid(sweep: &StcBatchRange) -> Vec<StcParams> {
        let fs = Self::expand_axis(sweep.fast_period);
        let ss = Self::expand_axis(sweep.slow_period);
        let ks = Self::expand_axis(sweep.k_period);
        let ds = Self::expand_axis(sweep.d_period);
        let mut out = Vec::with_capacity(fs.len() * ss.len() * ks.len() * ds.len());
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
        let combos = Self::expand_grid(sweep);
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
        let req =
            (len + rows * len) * std::mem::size_of::<f32>() + rows * 4 * std::mem::size_of::<i32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaStcError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload input
        let h = LockedBuffer::from_slice(data).map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        let mut d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        unsafe {
            d_prices
                .async_copy_from(&h, &self.stream)
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        }

        // Prepare param arrays (host)
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
        let max_k = combos.iter().map(|c| c.k_period.unwrap()).max().unwrap();

        // Output buffer
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;

        // Launch in chunks to respect grid limit
        let func = self
            .module
            .get_function("stc_batch_f32")
            .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        let shmem_bytes =
            (2 * max_k * std::mem::size_of::<f32>()) + (2 * max_k * std::mem::size_of::<u8>());

        for (start, count) in Self::grid_x_chunks(rows) {
            let mut d_f = DeviceBuffer::from_slice(&fasts[start..start + count])
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
            let mut d_s = DeviceBuffer::from_slice(&slows[start..start + count])
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
            let mut d_k = DeviceBuffer::from_slice(&ks[start..start + count])
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
            let mut d_d = DeviceBuffer::from_slice(&ds[start..start + count])
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;

            unsafe {
                let grid: GridSize = (count as u32, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let mut p_ptr = d_prices.as_device_ptr().as_raw();
                let mut f_ptr = d_f.as_device_ptr().as_raw();
                let mut s_ptr = d_s.as_device_ptr().as_raw();
                let mut k_ptr = d_k.as_device_ptr().as_raw();
                let mut d_ptr = d_d.as_device_ptr().as_raw();
                let mut n_i = len as i32;
                let mut fv_i = first_valid as i32;
                let mut r_i = count as i32;
                let mut mk_i = max_k as i32;
                // out pointer offset to the start of this chunk
                let mut o_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
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
                self.stream
                    .launch(
                        &func,
                        grid,
                        block,
                        shmem_bytes.try_into().unwrap_or(u32::MAX),
                        &mut args,
                    )
                    .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
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
        if data_tm.len() != cols * rows {
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
        let req = (cols * rows + cols + cols * rows) * std::mem::size_of::<f32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaStcError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload inputs
        let mut d_data =
            DeviceBuffer::from_slice(data_tm).map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        let mut d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;

        // Launch kernel
        let func = self
            .module
            .get_function("stc_many_series_one_param_f32")
            .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaStcError::Cuda(e.to_string()))?;
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

        v
    }
}
