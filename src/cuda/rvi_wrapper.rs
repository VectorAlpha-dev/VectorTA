//! CUDA wrapper for RVI (Relative Volatility Index).
//!
//! Parity targets (aligned with ALMA/CWMA wrappers):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/rvi_kernel.ptx")) with
//!   DetermineTargetFromContext + OptLevel O2 fallback to default.
//! - NON_BLOCKING stream.
//! - Warmup/NaN semantics match scalar rvi.rs exactly: warm = first_valid + (period-1) + (ma_len-1).
//! - Batch (one-series × many-params) and many-series × one-param (time-major) entry points.
//! - Simple VRAM checks and grid-y chunking (<= 65_535 rows per launch).
//! - Devtype support: 0=StdDev, 1=MeanAbsDev. Devtype=2 (median-abs-dev) is not implemented by the
//!   GPU kernel yet; the wrapper rejects such combos for now.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::rvi::{RviBatchRange, RviParams};
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
pub struct CudaRviPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaRviPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Debug)]
pub enum CudaRviError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaRviError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaRviError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaRviError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaRviError {}

pub struct CudaRvi {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaRviPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaRvi {
    pub fn new(device_id: usize) -> Result<Self, CudaRviError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/rvi_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaRviError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaRviPolicy::default(),
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaRviPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaRviPolicy {
        &self.policy
    }
    pub fn synchronize(&self) -> Result<(), CudaRviError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaRviError::Cuda(e.to_string()))
    }

    #[inline]
    fn device_mem_ok(bytes: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => bytes.saturating_add(64 * 1024 * 1024) <= free,
            Err(_) => true,
        }
    }

    #[inline]
    fn grid_y_chunks(n_rows: usize) -> impl Iterator<Item = (usize, usize)> {
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
    fn expand_grid(sweep: &RviBatchRange) -> Vec<RviParams> {
        let periods = Self::expand_axis(sweep.period);
        let ma_lens = Self::expand_axis(sweep.ma_len);
        let matypes = Self::expand_axis(sweep.matype);
        let devtypes = Self::expand_axis(sweep.devtype);
        let mut out =
            Vec::with_capacity(periods.len() * ma_lens.len() * matypes.len() * devtypes.len());
        for &p in &periods {
            for &m in &ma_lens {
                for &t in &matypes {
                    for &d in &devtypes {
                        out.push(RviParams {
                            period: Some(p),
                            ma_len: Some(m),
                            matype: Some(t),
                            devtype: Some(d),
                        });
                    }
                }
            }
        }
        out
    }

    fn prepare_batch(
        data: &[f32],
        sweep: &RviBatchRange,
    ) -> Result<(Vec<RviParams>, usize, usize, usize, usize), CudaRviError> {
        if data.is_empty() {
            return Err(CudaRviError::InvalidInput("empty data".into()));
        }
        let len = data.len();
        let first_valid = data
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaRviError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaRviError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // Reject unsupported devtype=2 for now (median abs dev)
        if combos.iter().any(|c| c.devtype.unwrap_or(0) == 2) {
            return Err(CudaRviError::InvalidInput(
                "devtype=2 (median abs dev) not supported by CUDA kernel yet".into(),
            ));
        }

        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        let max_ma_len = combos
            .iter()
            .map(|c| c.ma_len.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 || max_ma_len == 0 {
            return Err(CudaRviError::InvalidInput("invalid period/ma_len".into()));
        }
        if len - first_valid <= (max_period - 1) + (max_ma_len - 1) {
            return Err(CudaRviError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
        }
        Ok((combos, first_valid, len, max_period, max_ma_len))
    }

    pub fn rvi_batch_dev(
        &self,
        data: &[f32],
        sweep: &RviBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<RviParams>), CudaRviError> {
        let (combos, first_valid, len, max_period, max_ma_len) = Self::prepare_batch(data, sweep)?;
        let rows = combos.len();

        // VRAM estimate: inputs + param arrays + outputs
        let req =
            (len + rows * len) * std::mem::size_of::<f32>() + rows * 4 * std::mem::size_of::<i32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaRviError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload inputs
        let h_data =
            LockedBuffer::from_slice(data).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let mut d_data = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        unsafe {
            d_data
                .async_copy_from(&h_data, &self.stream)
                .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        }

        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaRviError::Cuda(e.to_string()))?;

        // Prepare param arrays (host)
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let ma_lens: Vec<i32> = combos.iter().map(|c| c.ma_len.unwrap() as i32).collect();
        let matypes: Vec<i32> = combos.iter().map(|c| c.matype.unwrap() as i32).collect();
        let devtypes: Vec<i32> = combos.iter().map(|c| c.devtype.unwrap() as i32).collect();

        // Shared memory size per block (uniform across grid). Layout must match kernel.
        let shmem_bytes = (2 * max_ma_len * std::mem::size_of::<f32>())
            + (max_period * std::mem::size_of::<f32>())
            + (max_period * std::mem::size_of::<u8>());

        // Chunk over rows if needed (grid.y limit)
        for (start, count) in Self::grid_y_chunks(rows) {
            let p = &periods[start..start + count];
            let m = &ma_lens[start..start + count];
            let t = &matypes[start..start + count];
            let d = &devtypes[start..start + count];
            let mut d_p =
                DeviceBuffer::from_slice(p).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
            let mut d_m =
                DeviceBuffer::from_slice(m).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
            let mut d_t =
                DeviceBuffer::from_slice(t).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
            let mut d_d =
                DeviceBuffer::from_slice(d).map_err(|e| CudaRviError::Cuda(e.to_string()))?;

            self.launch_batch(
                &d_data,
                &mut d_out,
                &mut d_p,
                &mut d_m,
                &mut d_t,
                &mut d_d,
                len,
                first_valid,
                count,
                max_period,
                max_ma_len,
                start,
                shmem_bytes,
            )?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
        d_periods: &mut DeviceBuffer<i32>,
        d_ma_lens: &mut DeviceBuffer<i32>,
        d_matypes: &mut DeviceBuffer<i32>,
        d_devtypes: &mut DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_rows: usize,
        max_period: usize,
        max_ma_len: usize,
        row_offset: usize,
        shmem_bytes: usize,
    ) -> Result<(), CudaRviError> {
        let func = self
            .module
            .get_function("rvi_batch_f32")
            .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged {
            eprintln!(
                "[rvi] batch kernel: block_x={} rows_chunk={} len={} max_p={} max_m={}",
                block_x, n_rows, len, max_period, max_ma_len
            );
            unsafe {
                (*(self as *const _ as *mut CudaRvi)).debug_batch_logged = true;
            }
        }
        unsafe {
            let grid: GridSize = (n_rows as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut d_ptr = d_data.as_device_ptr().as_raw();
            let mut p_ptr = d_periods.as_device_ptr().as_raw();
            let mut m_ptr = d_ma_lens.as_device_ptr().as_raw();
            let mut t_ptr = d_matypes.as_device_ptr().as_raw();
            let mut dv_ptr = d_devtypes.as_device_ptr().as_raw();
            let mut n_i = len as i32;
            let mut f_i = first_valid as i32;
            let mut r_i = n_rows as i32;
            let mut maxp_i = max_period as i32;
            let mut maxm_i = max_ma_len as i32;
            // out pointer offset to the start of this chunk
            let mut o_ptr = d_out
                .as_device_ptr()
                .as_raw()
                .wrapping_add((row_offset * len * std::mem::size_of::<f32>()) as u64);
            let mut args: [*mut c_void; 11] = [
                &mut d_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut m_ptr as *mut _ as *mut c_void,
                &mut t_ptr as *mut _ as *mut c_void,
                &mut dv_ptr as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut f_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut maxp_i as *mut _ as *mut c_void,
                &mut maxm_i as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, (shmem_bytes as u32), &mut args)
                .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn rvi_many_series_one_param_time_major_dev(
        &self,
        data_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &RviParams,
    ) -> Result<DeviceArrayF32, CudaRviError> {
        if cols == 0 || rows == 0 {
            return Err(CudaRviError::InvalidInput("empty matrix".into()));
        }
        if data_tm.len() != cols * rows {
            return Err(CudaRviError::InvalidInput("matrix shape mismatch".into()));
        }
        let period = params.period.unwrap_or(10);
        let ma_len = params.ma_len.unwrap_or(14);
        let matype = params.matype.unwrap_or(1);
        let devtype = params.devtype.unwrap_or(0);
        if devtype == 2 {
            return Err(CudaRviError::InvalidInput(
                "devtype=2 (median abs dev) not supported by CUDA kernel yet".into(),
            ));
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
        if (rows as i32) - max_first <= (period as i32 - 1 + ma_len as i32 - 1) {
            return Err(CudaRviError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
        }

        // VRAM estimate
        let req = ((cols * rows) * 2 + cols) * std::mem::size_of::<f32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaRviError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload buffers
        let h_data =
            LockedBuffer::from_slice(data_tm).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let h_firsts =
            LockedBuffer::from_slice(&firsts).map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let mut d_data =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let mut d_firsts = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }
            .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        unsafe {
            d_data
                .async_copy_from(&h_data, &self.stream)
                .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
            d_firsts
                .async_copy_from(&h_firsts, &self.stream)
                .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        }

        self.launch_many_series(
            &d_data, &d_firsts, cols, rows, period, ma_len, matype, devtype, &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many_series(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_firsts: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        ma_len: usize,
        matype: usize,
        devtype: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRviError> {
        let func = self
            .module
            .get_function("rvi_many_series_one_param_f32")
            .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_many_logged {
            eprintln!("[rvi] many-series kernel: block_x={} cols={} rows={} period={} ma_len={} matype={} devtype= {}", block_x, cols, rows, period, ma_len, matype, devtype);
            unsafe {
                (*(self as *const _ as *mut CudaRvi)).debug_many_logged = true;
            }
        }
        unsafe {
            let grid_x = ((cols as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut d_ptr = d_data.as_device_ptr().as_raw();
            let mut f_ptr = d_firsts.as_device_ptr().as_raw();
            let mut c_i = cols as i32;
            let mut r_i = rows as i32;
            let mut p_i = period as i32;
            let mut m_i = ma_len as i32;
            let mut t_i = matype as i32;
            let mut d_i = devtype as i32;
            let mut o_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut d_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut c_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut m_i as *mut _ as *mut c_void,
                &mut t_i as *mut _ as *mut c_void,
                &mut d_i as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                std::ptr::null_mut(), // placeholder to keep array length stable if modified
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaRviError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// -------- Bench Profiles --------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        // Batch: 100k samples, modest combo grid
        v.push(
            CudaBenchScenario::new(
                "rvi",
                "one_series_many_params",
                "rvi/batch",
                "100k x 128",
                || {
                    struct State {
                        cuda: CudaRvi,
                        data: Vec<f32>,
                        sweep: RviBatchRange,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            let _ = self.cuda.rvi_batch_dev(&self.data, &self.sweep);
                        }
                    }
                    let n = 100_000usize;
                    let mut data = vec![f32::NAN; n];
                    for i in 500..n {
                        let x = i as f32;
                        data[i] = (x * 0.00123).sin() + 0.0002 * x;
                    }
                    let sweep = RviBatchRange {
                        period: (10, 25, 1),
                        ma_len: (14, 14, 0),
                        matype: (1, 1, 0),
                        devtype: (0, 0, 0),
                    };
                    Box::new(State {
                        cuda: CudaRvi::new(0).unwrap(),
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
                "rvi",
                "many_series_one_param",
                "rvi/many_series",
                "2048r x 512c",
                || {
                    struct State {
                        cuda: CudaRvi,
                        data_tm: Vec<f32>,
                        cols: usize,
                        rows: usize,
                        params: RviParams,
                    }
                    impl CudaBenchState for State {
                        fn launch(&mut self) {
                            let _ = self.cuda.rvi_many_series_one_param_time_major_dev(
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
                    let params = RviParams {
                        period: Some(10),
                        ma_len: Some(14),
                        matype: Some(1),
                        devtype: Some(0),
                    };
                    Box::new(State {
                        cuda: CudaRvi::new(0).unwrap(),
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
