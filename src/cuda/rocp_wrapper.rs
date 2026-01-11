//! CUDA wrapper for ROCP (Rate of Change Percentage without 100x).
//!
//! Parity goals (aligned with ALMA):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/rocp_kernel.ptx"))
//!   with DetermineTargetFromContext and OptLevel O2 fallback.
//! - NON_BLOCKING stream.
//! - Warmup/NaN semantics match scalar rocp.rs exactly.
//! - Batch path optionally reuses host-precomputed reciprocals across rows.
//! - Many-series × one-param uses time-major layout with per-series first_valid indices.
//! - Typed CUDA errors, VRAM checks via `will_fit`, and launch config validation
//!   to avoid oversubscribing the device.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::rocp::{RocpBatchRange, RocpParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

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
pub struct CudaRocpPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaRocpPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Debug)]
pub enum CudaRocpError {
    Cuda(CudaError),
    InvalidInput(String),
    MissingKernelSymbol { name: &'static str },
    OutOfMemory { required: usize, free: usize, headroom: usize },
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    InvalidPolicy(&'static str),
    DeviceMismatch { buf: u32, current: u32 },
    NotImplemented,
}
impl fmt::Display for CudaRocpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaRocpError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaRocpError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaRocpError::MissingKernelSymbol { name } => {
                write!(f, "Missing kernel symbol: {}", name)
            }
            CudaRocpError::OutOfMemory {
                required,
                free,
                headroom,
            } => write!(
                f,
                "Out of memory on device: required={}B, free={}B, headroom={}B",
                required, free, headroom
            ),
            CudaRocpError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz } => write!(
                f,
                "Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))"
            ),
            CudaRocpError::InvalidPolicy(p) => write!(f, "Invalid policy: {}", p),
            CudaRocpError::DeviceMismatch { buf, current } => write!(
                f,
                "Device mismatch for buffer (buf device={} current={})",
                buf, current
            ),
            CudaRocpError::NotImplemented => write!(f, "Not implemented"),
        }
    }
}
impl std::error::Error for CudaRocpError {}

pub struct CudaRocp {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaRocpPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaRocp {
    pub fn new(device_id: usize) -> Result<Self, CudaRocpError> {
        cust::init(CudaFlags::empty()).map_err(CudaRocpError::Cuda)?;
        let device = Device::get_device(device_id as u32).map_err(CudaRocpError::Cuda)?;
        let context = Arc::new(Context::new(device).map_err(CudaRocpError::Cuda)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/rocp_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(CudaRocpError::Cuda)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).map_err(CudaRocpError::Cuda)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaRocpPolicy::default(),
            debug_batch_logged: false,
            debug_many_logged: false,
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

    pub fn set_policy(&mut self, policy: CudaRocpPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaRocpPolicy {
        &self.policy
    }
    pub fn synchronize(&self) -> Result<(), CudaRocpError> {
        self.stream.synchronize().map_err(CudaRocpError::Cuda)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaRocpError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaRocpError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaRocpError> {
        let dev = Device::get_device(self.device_id).map_err(CudaRocpError::Cuda)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .map_err(CudaRocpError::Cuda)? as u32;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .map_err(CudaRocpError::Cuda)? as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .map_err(CudaRocpError::Cuda)? as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .map_err(CudaRocpError::Cuda)? as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaRocpError::Cuda)? as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .map_err(CudaRocpError::Cuda)? as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
            .map_err(CudaRocpError::Cuda)? as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaRocpError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        Ok(())
    }

    fn expand_periods(sweep: &RocpBatchRange) -> Result<Vec<usize>, CudaRocpError> {
        let (start, end, step) = sweep.period;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            let vals: Vec<usize> = (start..=end).step_by(st).collect();
            if vals.is_empty() {
                return Err(CudaRocpError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            return Ok(vals);
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
            return Err(CudaRocpError::InvalidInput(format!(
                "Invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(v)
    }

    
    fn prepare_batch(
        data: &[f32],
        sweep: &RocpBatchRange,
    ) -> Result<(Vec<RocpParams>, usize, usize), CudaRocpError> {
        if data.is_empty() {
            return Err(CudaRocpError::InvalidInput("empty data".into()));
        }
        let len = data.len();
        let first_valid = data
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaRocpError::InvalidInput("all values are NaN".into()))?;
        let periods = Self::expand_periods(sweep)?;
        if periods.is_empty() {
            return Err(CudaRocpError::InvalidInput("empty period sweep".into()));
        }
        let max_p = *periods.iter().max().unwrap();
        if len - first_valid < max_p {
            return Err(CudaRocpError::InvalidInput("not enough valid data".into()));
        }
        let combos: Vec<RocpParams> = periods
            .iter()
            .map(|&p| RocpParams { period: Some(p) })
            .collect();
        Ok((combos, first_valid, len))
    }

    fn build_reciprocals(data: &[f32]) -> Vec<f32> {
        let mut inv = Vec::with_capacity(data.len());
        for &v in data {
            inv.push(1.0f32 / v);
        }
        inv
    }

    pub fn rocp_batch_dev(
        &self,
        data: &[f32],
        sweep: &RocpBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<RocpParams>), CudaRocpError> {
        let (combos, first_valid, len) = Self::prepare_batch(data, sweep)?;
        // Rough VRAM estimate: inputs + periods + out
        let rows = combos.len();
        let in_bytes = 2usize
            .checked_mul(len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        let out_bytes = rows
            .checked_mul(len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(out_bytes)
            .and_then(|x| x.checked_add(param_bytes))
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if let Err(e) = Self::will_fit(required, headroom) {
            if let Ok((free, _)) = mem_get_info() {
                return Err(CudaRocpError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(e);
            }
        }

        // Host precompute reciprocals (shared across rows)
        let inv_host = Self::build_reciprocals(data);

        // Upload inputs without page-locked staging (synchronous Vec->Device)
        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods_host).map_err(CudaRocpError::Cuda)?;
        let d_data = DeviceBuffer::from_slice(data).map_err(CudaRocpError::Cuda)?;
        let d_inv = DeviceBuffer::from_slice(&inv_host).map_err(CudaRocpError::Cuda)?;
        let mut d_out = unsafe {
            DeviceBuffer::<f32>::uninitialized_async(
                rows.checked_mul(len)
                    .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?,
                &self.stream,
            )
        }
        .map_err(CudaRocpError::Cuda)?;

        self.launch_batch(&d_data, &d_inv, &d_periods, len, rows, first_valid, &mut d_out)?;

        // Ensure completion before handing off VRAM handle (CAI stream omitted).
        self.stream.synchronize().map_err(CudaRocpError::Cuda)?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    
    pub fn rocp_batch_into_host_f32(
        &self,
        data: &[f32],
        sweep: &RocpBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<RocpParams>), CudaRocpError> {
        let (arr, combos) = self.rocp_batch_dev(data, sweep)?;
        let need = arr
            .rows
            .checked_mul(arr.cols)
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        if out.len() != need {
            return Err(CudaRocpError::InvalidInput(format!(
                "output slice wrong length: got {}, need {}",
                out.len(),
                need
            )));
        }
        // Ensure all work queued on our NON_BLOCKING stream has completed before D->H copy
        self.stream.synchronize().map_err(CudaRocpError::Cuda)?;
        arr.buf.copy_to(out).map_err(CudaRocpError::Cuda)?;
        Ok((arr.rows, arr.cols, combos))
    }

    fn launch_batch(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_inv: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        rows: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRocpError> {
        let func = self
            .module
            .get_function("rocp_batch_f32")
            .map_err(|_| CudaRocpError::MissingKernelSymbol { name: "rocp_batch_f32" })?;
        let suggested = func
            .suggested_launch_configuration(0, (256u32, 1u32, 1u32).into())
            .map(|(_min_grid, bs)| bs)
            .unwrap_or(256);
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => suggested.clamp(32, 1024),
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged {
            
            eprintln!(
                "[rocp] batch kernel: block_x={} rows={} len={}",
                block_x, rows, len
            );
            unsafe {
                (*(self as *const _ as *mut CudaRocp)).debug_batch_logged = true;
            }
        }
        unsafe {
            // Grid: one block per row (combo)
            let gx = rows as u32;
            let grid: GridSize = (gx, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(gx, 1, 1, block_x, 1, 1)?;
            let mut d_ptr = d_data.as_device_ptr().as_raw();
            let mut i_ptr = d_inv.as_device_ptr().as_raw();
            let mut p_ptr = d_periods.as_device_ptr().as_raw();
            let mut n_i = len as i32;
            let mut f_i = first_valid as i32;
            let mut r_i = rows as i32;
            let mut o_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut d_ptr as *mut _ as *mut c_void,
                &mut i_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut f_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(CudaRocpError::Cuda)?;
            
        }
        Ok(())
    }

    
    pub fn rocp_many_series_one_param_time_major_dev(
        &self,
        data_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaRocpError> {
        if cols == 0 || rows == 0 {
            return Err(CudaRocpError::InvalidInput("empty matrix".into()));
        }
        if data_tm.len()
            != cols
                .checked_mul(rows)
                .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?
        {
            return Err(CudaRocpError::InvalidInput("matrix shape mismatch".into()));
        }
        if period == 0 {
            return Err(CudaRocpError::InvalidInput("period must be > 0".into()));
        }

        // Per-series first_valid detection
        let mut firsts = vec![rows as i32; cols];
        
        for s in 0..cols {
            for t in 0..rows {
                let v = data_tm[t * cols + s];
                if !v.is_nan() {
                    firsts[s] = t as i32;
                    break;
                }
            }
        }
        let max_first = *firsts.iter().max().unwrap_or(&0);

        if (rows as i32) - max_first < period as i32 {
            return Err(CudaRocpError::InvalidInput("not enough valid data".into()));
        }

        // VRAM estimate (data + out + firsts)
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        let data_bytes = elems
            .checked_mul(2)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        let firsts_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        let req = data_bytes
            .checked_add(firsts_bytes)
            .ok_or_else(|| CudaRocpError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(req, headroom)?;

        // Upload buffers (synchronous copy; avoids extra host->host copy)
        let d_data = DeviceBuffer::from_slice(data_tm).map_err(CudaRocpError::Cuda)?;
        let d_firsts = DeviceBuffer::from_slice(&firsts).map_err(CudaRocpError::Cuda)?;
        let mut d_out = unsafe {
            DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream)
        }
        .map_err(CudaRocpError::Cuda)?;

        self.launch_many_series(&d_data, &d_firsts, cols, rows, period, &mut d_out)?;
        // Ensure completion before handing off VRAM handle.
        self.stream.synchronize().map_err(CudaRocpError::Cuda)?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    fn launch_many_series(&self, d_data: &DeviceBuffer<f32>, d_firsts: &DeviceBuffer<i32>, cols: usize, rows: usize, period: usize, d_out: &mut DeviceBuffer<f32>) -> Result<(), CudaRocpError> {
        let func = self
            .module
            .get_function("rocp_many_series_one_param_f32")
            .map_err(|_| CudaRocpError::MissingKernelSymbol {
                name: "rocp_many_series_one_param_f32",
            })?;
        let suggested = func
            .suggested_launch_configuration(0, (256u32, 1u32, 1u32).into())
            .map(|(_min_grid, bs)| bs)
            .unwrap_or(256);
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => suggested.clamp(32, 1024),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_many_logged {
            
            eprintln!(
                "[rocp] many-series kernel: block_x={} cols={} rows={} period={}",
                block_x, cols, rows, period
            );
            unsafe {
                (*(self as *const _ as *mut CudaRocp)).debug_many_logged = true;
            }
        }
        unsafe {
            let grid_x = ((cols as u32).saturating_add(block_x - 1)) / block_x;
            let gx = grid_x.max(1);
            let grid: GridSize = (gx, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(gx, 1, 1, block_x, 1, 1)?;
            let mut d_ptr = d_data.as_device_ptr().as_raw();
            let mut f_ptr = d_firsts.as_device_ptr().as_raw();
            let mut c_i = cols as i32;
            let mut r_i = rows as i32;
            let mut p_i = period as i32;
            let mut o_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut d_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut c_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(CudaRocpError::Cuda)?;
            
        }
        Ok(())
    }
}

// ---------- Bench Profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            // Batch: 100k samples, 256 combos (device-resident)
            CudaBenchScenario::new(
                "rocp",
                "one_series_many_params",
                "rocp/batch",
                "100k x 256",
                || {
                    struct State {
                        cuda: CudaRocp,
                        d_data: DeviceBuffer<f32>,
                        d_inv: DeviceBuffer<f32>,
                        d_periods: DeviceBuffer<i32>,
                        d_out: DeviceBuffer<f32>,
                        len: usize,
                        rows: usize,
                        first_valid: usize,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            self.cuda
                                .launch_batch(
                                    &self.d_data,
                                    &self.d_inv,
                                    &self.d_periods,
                                    self.len,
                                    self.rows,
                                    self.first_valid,
                                    &mut self.d_out,
                                )
                                .expect("rocp launch_batch");
                            let _ = self.cuda.stream.synchronize();
                        }
                    }
                    let n = 100_000usize;
                    let mut data = vec![f32::NAN; n];
                    for i in 500..n {
                        let x = i as f32;
                        data[i] = (x * 0.00123).sin() + 0.0002 * x;
                    }
                    let sweep = RocpBatchRange {
                        period: (4, 4 + 255, 1),
                    };
                    let (combos, first_valid, len) =
                        CudaRocp::prepare_batch(&data, &sweep).expect("prepare_batch");
                    let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
                    let inv_host = CudaRocp::build_reciprocals(&data);
                    let rows = combos.len();
                    let cuda = CudaRocp::new(0).unwrap();
                    let d_data = DeviceBuffer::from_slice(&data).expect("d_data");
                    let d_inv = DeviceBuffer::from_slice(&inv_host).expect("d_inv");
                    let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
                    let d_out: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(rows * len) }.expect("d_out");
                    Box::new(State {
                        cuda,
                        d_data,
                        d_inv,
                        d_periods,
                        d_out,
                        len,
                        rows,
                        first_valid,
                    })
                },
            )
            .with_sample_size(20),
            // Many-series: 1024 rows, 512 columns (device-resident)
            CudaBenchScenario::new(
                "rocp",
                "many_series_one_param",
                "rocp/many_series",
                "1024r x 512c",
                || {
                    struct State {
                        cuda: CudaRocp,
                        d_data: DeviceBuffer<f32>,
                        d_firsts: DeviceBuffer<i32>,
                        d_out: DeviceBuffer<f32>,
                        cols: usize,
                        rows: usize,
                        period: usize,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            self.cuda
                                .launch_many_series(
                                    &self.d_data,
                                    &self.d_firsts,
                                    self.cols,
                                    self.rows,
                                    self.period,
                                    &mut self.d_out,
                                )
                                .expect("rocp launch_many_series");
                            let _ = self.cuda.stream.synchronize();
                        }
                    }
                    let cols = 512usize;
                    let rows = 1024usize;
                    let mut tm = vec![f32::NAN; cols * rows];
                    for s in 0..cols {
                        for t in s..rows {
                            let x = t as f32 + (s as f32) * 0.1;
                            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                        }
                    }
                    let mut firsts = vec![rows as i32; cols];
                    for s in 0..cols {
                        for t in 0..rows {
                            let v = tm[t * cols + s];
                            if !v.is_nan() {
                                firsts[s] = t as i32;
                                break;
                            }
                        }
                    }
                    let period = 14usize;
                    let cuda = CudaRocp::new(0).unwrap();
                    let d_data = DeviceBuffer::from_slice(&tm).expect("d_data");
                    let d_firsts = DeviceBuffer::from_slice(&firsts).expect("d_firsts");
                    let d_out: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out");
                    Box::new(State {
                        cuda,
                        d_data,
                        d_firsts,
                        d_out,
                        cols,
                        rows,
                        period,
                    })
                },
            )
            .with_sample_size(20),
        ]
    }
}
