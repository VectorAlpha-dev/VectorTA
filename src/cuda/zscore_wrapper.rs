//! CUDA scaffolding for the Zscore indicator (SMA + stddev path).
//!
//! Follows the same VRAM-first design as the ALMA and Buff Averages wrappers.
//! Inputs are converted to `f32`, prefix sums are precomputed on the host using
//! `f64` accumulators for numerical stability, and the CUDA kernel evaluates all
//! parameter combinations in parallel.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::zscore::ZscoreBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashSet;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaZscoreError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaZscoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaZscoreError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaZscoreError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaZscoreError {}

#[derive(Clone, Debug)]
struct ZscoreCombo {
    period: usize,
    nbdev: f32,
}

pub struct CudaZscore {
    module: Module,
    stream: Stream,
    _context: Context,
    // Selection policy + introspection (ALMA parity)
    policy: CudaZscorePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaZscore {
    pub fn new(device_id: usize) -> Result<Self, CudaZscoreError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/zscore_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaZscorePolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaZscorePolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaZscorePolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    fn expand_combos(range: &ZscoreBatchRange) -> Vec<(usize, f64, String, usize)> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return vec![start];
            }
            let mut out = Vec::new();
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += step;
            }
            out
        }
        fn axis_str((start, end, step): (String, String, String)) -> Vec<String> {
            if start == end {
                return vec![start];
            }
            vec![start]
        }

        let periods = axis_usize(range.period);
        let ma_types = axis_str(range.ma_type.clone());
        let nbdevs = axis_f64(range.nbdev);
        let devtypes = axis_usize(range.devtype);

        let mut combos =
            Vec::with_capacity(periods.len() * ma_types.len() * nbdevs.len() * devtypes.len());
        for &p in &periods {
            for mt in &ma_types {
                for &nb in &nbdevs {
                    for &dt in &devtypes {
                        combos.push((p, nb, mt.clone(), dt));
                    }
                }
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &ZscoreBatchRange,
    ) -> Result<(Vec<ZscoreCombo>, usize, usize), CudaZscoreError> {
        if data_f32.is_empty() {
            return Err(CudaZscoreError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaZscoreError::InvalidInput("all values are NaN".into()))?;

        let combos_raw = Self::expand_combos(sweep);
        if combos_raw.is_empty() {
            return Err(CudaZscoreError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut seen_ma = HashSet::new();
        let mut out = Vec::with_capacity(combos_raw.len());
        for (period, nbdev, ma_type, devtype) in combos_raw {
            if period == 0 {
                return Err(CudaZscoreError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaZscoreError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaZscoreError::InvalidInput(format!(
                    "not enough valid data for period {} (valid after first {}: {})",
                    period,
                    first_valid,
                    len - first_valid
                )));
            }
            if devtype != 0 {
                return Err(CudaZscoreError::InvalidInput(format!(
                    "unsupported devtype {} (only devtype=0 supported)",
                    devtype
                )));
            }
            if ma_type != "sma" {
                seen_ma.insert(ma_type);
                continue;
            }
            out.push(ZscoreCombo {
                period,
                nbdev: nbdev as f32,
            });
        }

        if out.is_empty() {
            if seen_ma.is_empty() {
                return Err(CudaZscoreError::InvalidInput(
                    "no supported parameter combinations (require ma_type='sma' and devtype=0)"
                        .into(),
                ));
            } else {
                return Err(CudaZscoreError::InvalidInput(format!(
                    "unsupported ma_type(s): {} (only 'sma' supported for CUDA)",
                    seen_ma.into_iter().collect::<Vec<_>>().join(", ")
                )));
            }
        }

        Ok((out, first_valid, len))
    }

    fn build_prefixes(data: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        let len = data.len();
        let mut prefix_sum = vec![0.0f64; len + 1];
        let mut prefix_sum_sq = vec![0.0f64; len + 1];
        let mut prefix_nan = vec![0i32; len + 1];

        let mut acc_sum = 0.0f64;
        let mut acc_sq = 0.0f64;
        let mut acc_nan = 0i32;

        for i in 0..len {
            let v = data[i];
            if v.is_nan() {
                acc_nan += 1;
            } else {
                let dv = v as f64;
                acc_sum += dv;
                acc_sq += dv * dv;
            }
            prefix_sum[i + 1] = acc_sum;
            prefix_sum_sq[i + 1] = acc_sq;
            prefix_nan[i + 1] = acc_nan;
        }

        (prefix_sum, prefix_sum_sq, prefix_nan)
    }

    fn launch_batch_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_prefix_sum: &DeviceBuffer<f64>,
        d_prefix_sum_sq: &DeviceBuffer<f64>,
        d_prefix_nan: &DeviceBuffer<i32>,
        d_periods: &DeviceBuffer<i32>,
        d_nbdevs: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZscoreError> {
        let func = self
            .module
            .get_function("zscore_sma_prefix_f32")
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        // record selection once
        unsafe {
            (*(self as *const _ as *mut CudaZscore)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let grid_x = ((len as u32) + block_x - 1) / block_x;
        // Chunk grid.y to ≤ 65_535
        for (start, chunk_len) in Self::grid_y_chunks(n_combos) {
            let grid: GridSize = (grid_x.max(1), chunk_len as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let mut prefix_sum_ptr = d_prefix_sum.as_device_ptr().as_raw();
                let mut prefix_sum_sq_ptr = d_prefix_sum_sq.as_device_ptr().as_raw();
                let mut prefix_nan_ptr = d_prefix_nan.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut nbdevs_ptr = d_nbdevs.as_device_ptr().add(start).as_raw();
                let mut combos_i = chunk_len as i32;
                // Offset output by start * len
                let out_off = start * len;
                let mut out_ptr = d_out.as_device_ptr().add(out_off).as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut prefix_sum_ptr as *mut _ as *mut c_void,
                    &mut prefix_sum_sq_ptr as *mut _ as *mut c_void,
                    &mut prefix_nan_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut nbdevs_ptr as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
            }
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[ZscoreCombo],
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaZscoreError> {
        let len = data_f32.len();
        let (prefix_sum, prefix_sum_sq, prefix_nan) = Self::build_prefixes(data_f32);

        let d_data =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let d_prefix_sum = DeviceBuffer::from_slice(&prefix_sum)
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let d_prefix_sum_sq = DeviceBuffer::from_slice(&prefix_sum_sq)
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let d_prefix_nan = DeviceBuffer::from_slice(&prefix_nan)
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period as i32).collect();
        let nbdevs: Vec<f32> = combos.iter().map(|c| c.nbdev).collect();
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let d_nbdevs =
            DeviceBuffer::from_slice(&nbdevs).map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_data,
            &d_prefix_sum,
            &d_prefix_sum_sq,
            &d_prefix_nan,
            &d_periods,
            &d_nbdevs,
            len,
            first_valid,
            combos.len(),
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn zscore_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &ZscoreBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<(usize, f32)>), CudaZscoreError> {
        let (combos, first_valid, _len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        // VRAM estimate
        let len = data_f32.len();
        let prefixes =
            2 * (len + 1) * std::mem::size_of::<f64>() + (len + 1) * std::mem::size_of::<i32>();
        let params = combos.len() * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let input_bytes = len * std::mem::size_of::<f32>();
        let out_bytes = combos.len() * len * std::mem::size_of::<f32>();
        let required = prefixes + params + input_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaZscoreError::Cuda("insufficient VRAM".into()));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid)?;
        let meta = combos.iter().map(|c| (c.period, c.nbdev)).collect();
        Ok((dev, meta))
    }

    pub fn zscore_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &ZscoreBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<(usize, f32)>), CudaZscoreError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaZscoreError::InvalidInput(format!(
                "output slice length mismatch (expected {}, got {})",
                expected,
                out.len()
            )));
        }

        let dev = self.run_batch_kernel(data_f32, &combos, first_valid)?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let meta = combos.iter().map(|c| (c.period, c.nbdev)).collect();
        Ok((combos.len(), len, meta))
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn zscore_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        nbdev: f32,
    ) -> Result<DeviceArrayF32, CudaZscoreError> {
        if cols == 0 || rows == 0 {
            return Err(CudaZscoreError::InvalidInput("empty matrix".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaZscoreError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaZscoreError::InvalidInput("period must be > 0".into()));
        }
        // Compute per-series first_valid
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut fv = -1;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        // VRAM estimate (inputs + first_valids + output + ~64MB headroom)
        let bytes_in = cols * rows * std::mem::size_of::<f32>();
        let bytes_fv = cols * std::mem::size_of::<i32>();
        let bytes_out = cols * rows * std::mem::size_of::<f32>();
        let required = bytes_in + bytes_fv + bytes_out;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaZscoreError::Cuda("insufficient VRAM".into()));
        }

        let d_in = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("zscore_many_series_one_param_f32")
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        unsafe {
            (*(self as *const _ as *mut CudaZscore)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut in_ptr = d_in.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut nbdev_f = nbdev as f32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut nbdev_f as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn zscore_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        nbdev: f32,
        out_tm: &mut [f32],
    ) -> Result<(), CudaZscoreError> {
        let expected = cols * rows;
        if out_tm.len() != expected {
            return Err(CudaZscoreError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                expected
            )));
        }
        let arr = self.zscore_many_series_one_param_time_major_dev(
            data_tm_f32,
            cols,
            rows,
            period,
            nbdev,
        )?;
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(arr.len())
                .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?
        };
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaZscoreError::Cuda(e.to_string()))?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::zscore::ZscoreBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct ZscoreBatchState {
        cuda: CudaZscore,
        price: Vec<f32>,
        sweep: ZscoreBatchRange,
    }
    impl CudaBenchState for ZscoreBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .zscore_batch_dev(&self.price, &self.sweep)
                .expect("zscore batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaZscore::new(0).expect("cuda zscore");
        let price = gen_series(ONE_SERIES_LEN);
        // vary period; nbdev=2.0; ma_type="sma"; devtype=0 (supported)
        let sweep = ZscoreBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
            nbdev: (2.0, 2.0, 0.0),
            devtype: (0, 0, 0),
        };
        Box::new(ZscoreBatchState { cuda, price, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "zscore",
            "one_series_many_params",
            "zscore_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}

// ---------- Policies, introspection, and helpers ----------

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}
#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaZscorePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

impl Default for CudaZscorePolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

impl CudaZscore {
    #[inline]
    fn mem_check_enabled() -> bool {
        env::var("CUDA_MEM_CHECK")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true)
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
    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const LIM: usize = 65_535;
        let mut start = 0;
        std::iter::from_fn(move || {
            if start >= n {
                return None;
            }
            let len = (n - start).min(LIM);
            let cur = (start, len);
            start += len;
            Some(cur)
        })
    }
    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] zscore batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaZscore)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] zscore many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaZscore)).debug_many_logged = true;
                }
            }
        }
    }
}
