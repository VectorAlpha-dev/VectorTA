//! CUDA wrapper for Coppock Curve (sum of two ROCs smoothed by WMA)
//!
//! Pattern: simple per-index computation with windowed WMA smoothing.
//! - Batch (one series × many params): grid.y = combos, grid.x over time.
//! - Many-series × one-param (time-major): 1D grid over series; each thread scans time.
//!
//! Semantics are identical to the scalar path in `indicators::coppock`:
//! - Warmup = first_valid + max(short,long) + (ma_period - 1)
//! - Leading values are NaN; we don’t mask mid-stream NaNs (match CPU path).

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::coppock::CoppockBatchRange;
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
pub enum CudaCoppockError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaCoppockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCoppockError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCoppockError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaCoppockError {}

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
pub struct CudaCoppockPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaCoppockPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaCoppock {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaCoppockPolicy,
    // debug logging once per instance when BENCH_DEBUG=1
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaCoppock {
    pub fn new(device_id: usize) -> Result<Self, CudaCoppockError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/coppock_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaCoppockPolicy {
                batch: BatchKernelPolicy::Auto,
                many_series: ManySeriesKernelPolicy::Auto,
            },
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaCoppockPolicy) {
        self.policy = p;
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaCoppockError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))
    }

    // ---------- Batch (one series × many params) ----------
    pub fn coppock_batch_dev(
        &self,
        price: &[f32],
        sweep: &CoppockBatchRange,
    ) -> Result<DeviceArrayF32, CudaCoppockError> {
        let len = price.len();
        if len == 0 {
            return Err(CudaCoppockError::InvalidInput("empty series".into()));
        }
        let first_valid = price
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaCoppockError::InvalidInput("all values are NaN".into()))?;

        // Expand grid
        let (shorts, longs, ma_periods) = expand_grid(sweep);
        let rows = ma_periods.len();
        if rows == 0 {
            return Err(CudaCoppockError::InvalidInput("no parameter combos".into()));
        }

        // Validate warmups
        for ((&s, &l), &m) in shorts.iter().zip(longs.iter()).zip(ma_periods.iter()) {
            let (s_u, l_u, m_u) = (s as usize, l as usize, m as usize);
            if s_u == 0 || l_u == 0 || m_u == 0 || s_u > len || l_u > len || m_u > len {
                return Err(CudaCoppockError::InvalidInput(format!(
                    "invalid params s={} l={} m={} for len {}",
                    s_u, l_u, m_u, len
                )));
            }
            let largest = s_u.max(l_u);
            if len - first_valid < largest {
                return Err(CudaCoppockError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
        }

        // Device buffers
        let d_price =
            DeviceBuffer::from_slice(price).map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        // Shared precompute: reciprocals of price
        let mut inv = vec![0f32; len];
        for i in 0..len {
            inv[i] = 1.0f32 / price[i];
        }
        let d_inv =
            DeviceBuffer::from_slice(&inv).map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;

        // Chunking for grid.y & VRAM
        let bytes_params = rows * (std::mem::size_of::<i32>() * 3);
        let bytes_out_total = rows * len * std::mem::size_of::<f32>();
        let headroom = 64usize * 1024 * 1024;
        let fits = match mem_get_info() {
            Ok((free, _)) => bytes_params + bytes_out_total + headroom <= free,
            Err(_) => true,
        };
        let max_y = 65_535usize;

        if fits && rows <= max_y {
            // Single launch path
            let d_s = DeviceBuffer::from_slice(&shorts)
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            let d_l = DeviceBuffer::from_slice(&longs)
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            let d_m = DeviceBuffer::from_slice(&ma_periods)
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            let mut d_out: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(rows * len) }
                    .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            self.launch_batch(
                &d_price,
                &d_inv,
                len,
                first_valid,
                &d_s,
                &d_l,
                &d_m,
                rows,
                &mut d_out,
            )?;
            self.maybe_log_batch_debug();
            return Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            });
        }

        // Chunked path across combos (y)
        let mut host_out = vec![0f32; rows * len];
        let mut start = 0usize;
        while start < rows {
            let remain = rows - start;
            let chunk = remain.min(max_y);
            let d_s = DeviceBuffer::from_slice(&shorts[start..start + chunk])
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            let d_l = DeviceBuffer::from_slice(&longs[start..start + chunk])
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            let d_m = DeviceBuffer::from_slice(&ma_periods[start..start + chunk])
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(chunk * len) }
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            self.launch_batch(
                &d_price,
                &d_inv,
                len,
                first_valid,
                &d_s,
                &d_l,
                &d_m,
                chunk,
                &mut d_out,
            )?;
            d_out
                .copy_to(&mut host_out[start * len..start * len + chunk * len])
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
            start += chunk;
        }
        let d_out = DeviceBuffer::from_slice(&host_out)
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: len,
        })
    }

    fn launch_batch(
        &self,
        d_price: &DeviceBuffer<f32>,
        d_inv: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_short: &DeviceBuffer<i32>,
        d_long: &DeviceBuffer<i32>,
        d_ma: &DeviceBuffer<i32>,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCoppockError> {
        let func = self
            .module
            .get_function("coppock_batch_f32")
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut price_ptr = d_price.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut s_ptr = d_short.as_device_ptr().as_raw();
            let mut l_ptr = d_long.as_device_ptr().as_raw();
            let mut m_ptr = d_ma.as_device_ptr().as_raw();
            let mut n_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 9] = [
                &mut price_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut m_ptr as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn coppock_many_series_one_param_time_major_dev(
        &self,
        price_tm: &[f32],
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        ma_period: usize,
    ) -> Result<DeviceArrayF32, CudaCoppockError> {
        if cols == 0 || rows == 0 {
            return Err(CudaCoppockError::InvalidInput("invalid dims".into()));
        }
        if price_tm.len() != cols * rows {
            return Err(CudaCoppockError::InvalidInput(
                "time-major input mismatch".into(),
            ));
        }
        if short == 0 || long == 0 || ma_period == 0 {
            return Err(CudaCoppockError::InvalidInput("invalid periods".into()));
        }

        // Per-series first_valid
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = price_tm[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaCoppockError::InvalidInput(format!("series {} all NaN", s)))?;
            let largest = short.max(long);
            if rows - (fv as usize) < largest {
                return Err(CudaCoppockError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
            firsts[s] = fv;
        }

        let d_price = DeviceBuffer::from_slice(price_tm)
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        // Precompute reciprocals for time-major buffer
        let mut inv_tm = vec![0f32; cols * rows];
        for idx in 0..inv_tm.len() {
            inv_tm[idx] = 1.0f32 / price_tm[idx];
        }
        let d_inv =
            DeviceBuffer::from_slice(&inv_tm).map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        let d_first =
            DeviceBuffer::from_slice(&firsts).map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        self.launch_many(
            &d_price, &d_inv, &d_first, cols, rows, short, long, ma_period, &mut d_out,
        )?;
        self.maybe_log_many_debug();
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many(
        &self,
        d_price_tm: &DeviceBuffer<f32>,
        d_inv_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        ma_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCoppockError> {
        let func = self
            .module
            .get_function("coppock_many_series_one_param_f32")
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 128,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_ptr = d_price_tm.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_tm.as_device_ptr().as_raw();
            let mut f_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut s_i = short as i32;
            let mut l_i = long as i32;
            let mut m_i = ma_period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 9] = [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut l_i as *mut _ as *mut c_void,
                &mut m_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaCoppockError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaCoppockError::Cuda(e.to_string()))
    }

    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[DEBUG] Coppock batch selected kernel: Plain {{ block_x: {} }}",
                match self.policy.batch {
                    BatchKernelPolicy::Plain { block_x } => block_x,
                    BatchKernelPolicy::Auto => 256,
                }
            );
            unsafe {
                (*(self as *const _ as *mut CudaCoppock)).debug_batch_logged = true;
            }
        }
    }
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[DEBUG] Coppock many-series selected kernel: OneD {{ block_x: {} }}",
                match self.policy.many_series {
                    ManySeriesKernelPolicy::OneD { block_x } => block_x,
                    ManySeriesKernelPolicy::Auto => 128,
                }
            );
            unsafe {
                (*(self as *const _ as *mut CudaCoppock)).debug_many_logged = true;
            }
        }
    }
}

fn expand_grid(r: &CoppockBatchRange) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    fn axis((s, e, st): (usize, usize, usize)) -> Vec<i32> {
        if st == 0 || s == e {
            vec![s as i32]
        } else {
            (s..=e).step_by(st).map(|x| x as i32).collect()
        }
    }
    let shorts_u = axis(r.short);
    let longs_u = axis(r.long);
    let mas_u = axis(r.ma);
    let mut shorts = Vec::new();
    let mut longs = Vec::new();
    let mut mas = Vec::new();
    for &s in &shorts_u {
        for &l in &longs_u {
            for &m in &mas_u {
                shorts.push(s);
                longs.push(l);
                mas.push(m);
            }
        }
    }
    (shorts, longs, mas)
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // number of (short,long,ma) combos

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = PARAM_SWEEP * 3 * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct CoppockBatchState {
        cuda: CudaCoppock,
        price: Vec<f32>,
        sweep: CoppockBatchRange,
    }
    impl CudaBenchState for CoppockBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .coppock_batch_dev(&self.price, &self.sweep)
                .expect("coppock batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCoppock::new(0).expect("cuda coppock");
        let price = gen_series(ONE_SERIES_LEN);
        // Rough sweep around defaults
        let sweep = CoppockBatchRange {
            short: (8, 18, 2),
            long: (20, 30, 2),
            ma: (8, 16, 2),
        };
        Box::new(CoppockBatchState { cuda, price, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "coppock",
            "one_series_many_params",
            "coppock_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
