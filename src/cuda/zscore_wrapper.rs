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
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaZscoreError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error(
        "out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes"
    )]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Float2 {
    hi: f32,
    lo: f32,
}
unsafe impl cust::memory::DeviceCopy for Float2 {}

#[inline(always)]
fn pack_ds(v: f64) -> Float2 {
    let hi = v as f32;
    let lo = (v - hi as f64) as f32;
    Float2 { hi, lo }
}

#[derive(Clone, Debug)]
struct ZscoreCombo {
    period: usize,
    nbdev: f32,
}

pub struct CudaZscore {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,

    policy: CudaZscorePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaZscore {
    pub fn new(device_id: usize) -> Result<Self, CudaZscoreError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/zscore_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
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
    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaZscoreError> {
        self.stream.synchronize().map_err(Into::into)
    }

    fn expand_combos(
        range: &ZscoreBatchRange,
    ) -> Result<Vec<(usize, f64, String, usize)>, CudaZscoreError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaZscoreError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            let mut vals = Vec::new();
            if start < end {
                let mut v = start;
                while v <= end {
                    vals.push(v);
                    match v.checked_add(step) {
                        Some(next) => {
                            if next == v {
                                break;
                            }
                            v = next;
                        }
                        None => break,
                    }
                }
            } else {
                let mut v = start;
                loop {
                    vals.push(v);
                    if v == end {
                        break;
                    }
                    let next = v.saturating_sub(step);
                    if next == v {
                        break;
                    }
                    v = next;
                    if v < end {
                        break;
                    }
                }
            }
            if vals.is_empty() {
                return Err(CudaZscoreError::InvalidInput(format!(
                    "invalid usize range: start={} end={} step={}",
                    start, end, step
                )));
            }
            Ok(vals)
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaZscoreError> {
            if !step.is_finite() {
                return Err(CudaZscoreError::InvalidInput(format!(
                    "non-finite step in range: start={} end={} step={}",
                    start, end, step
                )));
            }
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return Ok(vec![start]);
            }
            let mut out = Vec::new();
            let tol = 1e-12;
            if start <= end {
                let step_pos = step.abs();
                let mut x = start;
                while x <= end + tol {
                    out.push(x);
                    x += step_pos;
                    if !x.is_finite() {
                        break;
                    }
                }
            } else {
                let step_neg = -step.abs();
                let mut x = start;
                while x >= end - tol {
                    out.push(x);
                    x += step_neg;
                    if !x.is_finite() {
                        break;
                    }
                }
            }
            if out.is_empty() {
                return Err(CudaZscoreError::InvalidInput(format!(
                    "invalid f64 range: start={} end={} step={}",
                    start, end, step
                )));
            }
            Ok(out)
        }
        fn axis_str((start, end, _step): (String, String, String)) -> Vec<String> {
            if start == end {
                vec![start]
            } else {
                vec![start]
            }
        }

        let periods = axis_usize(range.period)?;
        let ma_types = axis_str(range.ma_type.clone());
        let nbdevs = axis_f64(range.nbdev)?;
        let devtypes = axis_usize(range.devtype)?;

        let total = periods
            .len()
            .checked_mul(ma_types.len())
            .and_then(|v| v.checked_mul(nbdevs.len()))
            .and_then(|v| v.checked_mul(devtypes.len()))
            .ok_or_else(|| CudaZscoreError::InvalidInput("parameter grid too large".into()))?;

        let mut combos = Vec::with_capacity(total);
        for &p in &periods {
            for mt in &ma_types {
                for &nb in &nbdevs {
                    for &dt in &devtypes {
                        combos.push((p, nb, mt.clone(), dt));
                    }
                }
            }
        }
        Ok(combos)
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

        let combos_raw = Self::expand_combos(sweep)?;
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

    fn build_prefixes_ds_pinned(
        data: &[f32],
    ) -> Result<
        (
            LockedBuffer<Float2>,
            LockedBuffer<Float2>,
            LockedBuffer<i32>,
        ),
        CudaZscoreError,
    > {
        let n = data.len();
        let mut ps = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaZscoreError::InvalidInput(e.to_string()))?;
        let mut ps2 = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaZscoreError::InvalidInput(e.to_string()))?;
        let mut pnan = unsafe { LockedBuffer::<i32>::uninitialized(n + 1) }
            .map_err(|e| CudaZscoreError::InvalidInput(e.to_string()))?;

        ps.as_mut_slice()[0] = Float2::default();
        ps2.as_mut_slice()[0] = Float2::default();
        pnan.as_mut_slice()[0] = 0;

        let (mut s, mut s2) = (0.0f64, 0.0f64);
        let mut nan = 0i32;
        for i in 0..n {
            let v = data[i];
            if v.is_nan() {
                nan += 1;
                ps.as_mut_slice()[i + 1] = ps.as_slice()[i];
                ps2.as_mut_slice()[i + 1] = ps2.as_slice()[i];
            } else {
                let d = v as f64;
                s += d;
                s2 += d * d;
                ps.as_mut_slice()[i + 1] = pack_ds(s);
                ps2.as_mut_slice()[i + 1] = pack_ds(s2);
            }
            pnan.as_mut_slice()[i + 1] = nan;
        }

        Ok((ps, ps2, pnan))
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
            .map_err(|_| CudaZscoreError::MissingKernelSymbol {
                name: "zscore_sma_prefix_f32",
            })?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };

        unsafe {
            (*(self as *const _ as *mut CudaZscore)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let grid_x = ((len as u32) + block_x - 1) / block_x;

        const COMBO_TILE: usize = 4;

        for (start, chunk_len) in Self::grid_y_chunks(n_combos) {
            let grid_y = ((chunk_len + (COMBO_TILE - 1)) / COMBO_TILE) as u32;
            let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
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

                self.stream.launch(&func, grid, block, 0, args)?;
            }
        }

        Ok(())
    }

    fn launch_batch_kernel_ds(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_prefix_sum: &DeviceBuffer<Float2>,
        d_prefix_sum_sq: &DeviceBuffer<Float2>,
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
            .get_function("zscore_sma_prefix_f32ds")
            .map_err(|_| CudaZscoreError::MissingKernelSymbol {
                name: "zscore_sma_prefix_f32ds",
            })?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        unsafe {
            (*(self as *const _ as *mut CudaZscore)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        let grid_x = ((len as u32) + block_x - 1) / block_x;

        const COMBO_TILE: usize = 4;

        for (start, chunk_len) in Self::grid_y_chunks(n_combos) {
            let grid_y = ((chunk_len + (COMBO_TILE - 1)) / COMBO_TILE) as u32;
            let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let mut ps_ptr = d_prefix_sum.as_device_ptr().as_raw();
                let mut ps2_ptr = d_prefix_sum_sq.as_device_ptr().as_raw();
                let mut pnan_ptr = d_prefix_nan.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut nbdevs_ptr = d_nbdevs.as_device_ptr().add(start).as_raw();
                let mut combos_i = chunk_len as i32;
                let out_off = start * len;
                let mut out_ptr = d_out.as_device_ptr().add(out_off).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut ps_ptr as *mut _ as *mut c_void,
                    &mut ps2_ptr as *mut _ as *mut c_void,
                    &mut pnan_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut nbdevs_ptr as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn select_batch_impl(len: usize, combos: usize) -> bool {
        let work = len.saturating_mul(combos);
        work >= 5_000_000
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[ZscoreCombo],
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaZscoreError> {
        let len = data_f32.len();
        let use_ds = Self::select_batch_impl(len, combos.len());

        let d_data = DeviceBuffer::from_slice(data_f32)?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period as i32).collect();
        let nbdevs: Vec<f32> = combos.iter().map(|c| c.nbdev).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_nbdevs = DeviceBuffer::from_slice(&nbdevs)?;

        let elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaZscoreError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        if use_ds {
            let (ps, ps2, pnan) = Self::build_prefixes_ds_pinned(data_f32)?;
            let d_ps: DeviceBuffer<Float2> = DeviceBuffer::from_slice(ps.as_slice())?;
            let d_ps2: DeviceBuffer<Float2> = DeviceBuffer::from_slice(ps2.as_slice())?;
            let d_pnan: DeviceBuffer<i32> = DeviceBuffer::from_slice(pnan.as_slice())?;

            self.launch_batch_kernel_ds(
                &d_data,
                &d_ps,
                &d_ps2,
                &d_pnan,
                &d_periods,
                &d_nbdevs,
                len,
                first_valid,
                combos.len(),
                &mut d_out,
            )?;
        } else {
            let (prefix_sum, prefix_sum_sq, prefix_nan) = Self::build_prefixes(data_f32);
            let d_prefix_sum = DeviceBuffer::from_slice(&prefix_sum)?;
            let d_prefix_sum_sq = DeviceBuffer::from_slice(&prefix_sum_sq)?;
            let d_prefix_nan = DeviceBuffer::from_slice(&prefix_nan)?;

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
        }

        self.stream.synchronize()?;

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

        let len = data_f32.len();
        let len1 = len
            .checked_add(1)
            .ok_or_else(|| CudaZscoreError::InvalidInput("len+1 overflow".into()))?;
        let prefixes_f2 = len1
            .checked_mul(2 * std::mem::size_of::<Float2>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("prefix Float2 bytes overflow".into()))?;
        let prefixes_i32 = len1
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("prefix i32 bytes overflow".into()))?;
        let prefixes = prefixes_f2
            .checked_add(prefixes_i32)
            .ok_or_else(|| CudaZscoreError::InvalidInput("prefix bytes overflow".into()))?;
        let params = combos
            .len()
            .checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("params bytes overflow".into()))?;
        let input_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("input bytes overflow".into()))?;
        let out_bytes = combos
            .len()
            .checked_mul(len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaZscoreError::InvalidInput("output bytes overflow".into()))?;
        let required = prefixes
            .checked_add(params)
            .and_then(|v| v.checked_add(input_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaZscoreError::InvalidInput("VRAM requirement overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;
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

        let len1 = len
            .checked_add(1)
            .ok_or_else(|| CudaZscoreError::InvalidInput("len+1 overflow".into()))?;
        let prefixes_f2 = len1
            .checked_mul(2 * std::mem::size_of::<Float2>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("prefix Float2 bytes overflow".into()))?;
        let prefixes_i32 = len1
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("prefix i32 bytes overflow".into()))?;
        let prefixes = prefixes_f2
            .checked_add(prefixes_i32)
            .ok_or_else(|| CudaZscoreError::InvalidInput("prefix bytes overflow".into()))?;
        let params = combos
            .len()
            .checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("params bytes overflow".into()))?;
        let input_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("input bytes overflow".into()))?;
        let out_bytes = combos
            .len()
            .checked_mul(len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaZscoreError::InvalidInput("output bytes overflow".into()))?;
        let required = prefixes
            .checked_add(params)
            .and_then(|v| v.checked_add(input_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaZscoreError::InvalidInput("VRAM requirement overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let expected = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaZscoreError::InvalidInput("rows*cols overflow".into()))?;
        if out.len() != expected {
            return Err(CudaZscoreError::InvalidInput(format!(
                "output slice length mismatch (expected {}, got {})",
                expected,
                out.len()
            )));
        }

        let dev = self.run_batch_kernel(data_f32, &combos, first_valid)?;
        dev.buf.copy_to(out)?;
        let meta = combos.iter().map(|c| (c.period, c.nbdev)).collect();
        Ok((combos.len(), len, meta))
    }

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
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaZscoreError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaZscoreError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaZscoreError::InvalidInput("period must be > 0".into()));
        }

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

        let bytes_in = expected
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("input bytes overflow".into()))?;
        let bytes_fv = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("first_valid bytes overflow".into()))?;
        let bytes_out = expected
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaZscoreError::InvalidInput("output bytes overflow".into()))?;
        let required = bytes_in
            .checked_add(bytes_fv)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaZscoreError::InvalidInput("VRAM requirement overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_in = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(expected) }?;

        let func = self
            .module
            .get_function("zscore_many_series_one_param_f32")
            .map_err(|_| CudaZscoreError::MissingKernelSymbol {
                name: "zscore_many_series_one_param_f32",
            })?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        unsafe {
            (*(self as *const _ as *mut CudaZscore)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
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
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;

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
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaZscoreError::InvalidInput("rows*cols overflow".into()))?;
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
                .map_err(|e| CudaZscoreError::InvalidInput(e.to_string()))?
        };
        unsafe {
            arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::zscore::ZscoreBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let prefixes_bytes =
            (ONE_SERIES_LEN + 1) * (2 * std::mem::size_of::<Float2>() + std::mem::size_of::<i32>());
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + prefixes_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct ZscoreBatchState {
        cuda: CudaZscore,
        d_data: DeviceBuffer<f32>,
        d_ps: DeviceBuffer<Float2>,
        d_ps2: DeviceBuffer<Float2>,
        d_pnan: DeviceBuffer<i32>,
        d_periods: DeviceBuffer<i32>,
        d_nbdevs: DeviceBuffer<f32>,
        first_valid: usize,
        len: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ZscoreBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel_ds(
                    &self.d_data,
                    &self.d_ps,
                    &self.d_ps2,
                    &self.d_pnan,
                    &self.d_periods,
                    &self.d_nbdevs,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("zscore batch");
            self.cuda.synchronize().expect("zscore sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaZscore::new(0).expect("cuda zscore");
        let price = gen_series(ONE_SERIES_LEN);

        let sweep = ZscoreBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
            nbdev: (2.0, 2.0, 0.0),
            devtype: (0, 0, 0),
        };
        let (combos, first_valid, len) =
            CudaZscore::prepare_batch_inputs(&price, &sweep).expect("zscore prep");

        let d_data = DeviceBuffer::from_slice(&price).expect("d_data");
        let periods: Vec<i32> = combos.iter().map(|c| c.period as i32).collect();
        let nbdevs: Vec<f32> = combos.iter().map(|c| c.nbdev).collect();
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_nbdevs = DeviceBuffer::from_slice(&nbdevs).expect("d_nbdevs");
        let (ps, ps2, pnan) = CudaZscore::build_prefixes_ds_pinned(&price).expect("prefixes");
        let d_ps: DeviceBuffer<Float2> = DeviceBuffer::from_slice(ps.as_slice()).expect("d_ps");
        let d_ps2: DeviceBuffer<Float2> = DeviceBuffer::from_slice(ps2.as_slice()).expect("d_ps2");
        let d_pnan: DeviceBuffer<i32> = DeviceBuffer::from_slice(pnan.as_slice()).expect("d_pnan");
        let elems = len * combos.len();
        let d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("d_out");

        Box::new(ZscoreBatchState {
            cuda,
            d_data,
            d_ps,
            d_ps2,
            d_pnan,
            d_periods,
            d_nbdevs,
            first_valid,
            len,
            n_combos: combos.len(),
            d_out,
        })
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
    fn device_mem_info() -> Result<(usize, usize), CudaZscoreError> {
        mem_get_info().map_err(Into::into)
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaZscoreError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        let (free, _total) = Self::device_mem_info()?;
        if required_bytes.saturating_add(headroom_bytes) > free {
            return Err(CudaZscoreError::OutOfMemory {
                required: required_bytes,
                free,
                headroom: headroom_bytes,
            });
        }
        Ok(())
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
