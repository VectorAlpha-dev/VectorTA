#![cfg(feature = "cuda")]

use crate::indicators::apo::{ApoBatchRange, ApoParams};
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaApoError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Out of memory on device: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Launch configuration too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
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

pub struct DeviceArrayF32 {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    ctx_guard: Arc<Context>,
    device_id: u32,
}
impl DeviceArrayF32 {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
    #[inline]
    pub(crate) fn ctx(&self) -> Arc<Context> {
        self.ctx_guard.clone()
    }
    #[inline]
    pub(crate) fn device_id(&self) -> u32 {
        self.device_id
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

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaApoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaApo {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaApoPolicy,
}

impl CudaApo {
    pub fn new(device_id: usize) -> Result<Self, CudaApoError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/apo_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaApoPolicy::default(),
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaApoPolicy) -> Result<Self, CudaApoError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaApoError> {
        self.stream.synchronize().map_err(Into::into)
    }

    pub fn apo_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &ApoBatchRange,
    ) -> Result<DeviceArrayF32, CudaApoError> {
        let prep = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n = prep.combos.len();

        let item_f32 = std::mem::size_of::<f32>();
        let item_i32 = std::mem::size_of::<i32>();
        let prices_bytes = prep
            .series_len
            .checked_mul(item_f32)
            .ok_or_else(|| CudaApoError::InvalidInput("series_len bytes overflow".into()))?;
        let params_a = prep
            .short_periods
            .len()
            .checked_mul(item_i32)
            .ok_or_else(|| CudaApoError::InvalidInput("short_periods bytes overflow".into()))?;
        let params_b = prep
            .long_periods
            .len()
            .checked_mul(item_i32)
            .ok_or_else(|| CudaApoError::InvalidInput("long_periods bytes overflow".into()))?;
        let params_c = prep
            .short_alphas
            .len()
            .checked_mul(item_f32)
            .ok_or_else(|| CudaApoError::InvalidInput("short_alphas bytes overflow".into()))?;
        let params_d = prep
            .long_alphas
            .len()
            .checked_mul(item_f32)
            .ok_or_else(|| CudaApoError::InvalidInput("long_alphas bytes overflow".into()))?;
        let params_bytes = params_a
            .checked_add(params_b)
            .and_then(|x| x.checked_add(params_c))
            .and_then(|x| x.checked_add(params_d))
            .ok_or_else(|| CudaApoError::InvalidInput("param bytes overflow".into()))?;
        let out_elems = prep
            .series_len
            .checked_mul(n)
            .ok_or_else(|| CudaApoError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(item_f32)
            .ok_or_else(|| CudaApoError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaApoError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream)? };
        let d_sp = unsafe { DeviceBuffer::from_slice_async(&prep.short_periods, &self.stream)? };
        let d_lp = unsafe { DeviceBuffer::from_slice_async(&prep.long_periods, &self.stream)? };
        let d_sa = unsafe { DeviceBuffer::from_slice_async(&prep.short_alphas, &self.stream)? };
        let d_la = unsafe { DeviceBuffer::from_slice_async(&prep.long_alphas, &self.stream)? };
        let out_len = prep
            .series_len
            .checked_mul(n)
            .ok_or_else(|| CudaApoError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &self.stream)? };

        self.launch_batch_kernel(
            &d_prices,
            &d_sp,
            &d_sa,
            &d_lp,
            &d_la,
            prep.series_len,
            prep.first_valid,
            n,
            &mut d_out,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n,
            cols: prep.series_len,
            ctx_guard: self._context.clone(),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apo_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_short_periods: &DeviceBuffer<i32>,
        d_short_alphas: &DeviceBuffer<f32>,
        d_long_periods: &DeviceBuffer<i32>,
        d_long_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaApoError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaApoError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        let expected = series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaApoError::InvalidInput("rows*cols overflow".into()))?;
        if d_out.len() != expected {
            return Err(CudaApoError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_short_periods,
            d_short_alphas,
            d_long_periods,
            d_long_alphas,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )?;
        self.synchronize()
    }

    pub fn apo_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &ApoParams,
    ) -> Result<DeviceArrayF32, CudaApoError> {
        let (first_valids, sp, lp, a_s, a_l) =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let elems = num_series
            .checked_mul(series_len)
            .ok_or_else(|| CudaApoError::InvalidInput("num_series*series_len overflow".into()))?;
        let in_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaApoError::InvalidInput("in bytes overflow".into()))?;
        let first_bytes = first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaApoError::InvalidInput("first_valids bytes overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaApoError::InvalidInput("out bytes overflow".into()))?;
        let required = in_bytes
            .checked_add(first_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaApoError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)? };
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream)? };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream)? };

        self.launch_many_series_kernel(
            &d_prices, &d_first, sp, a_s, lp, a_l, num_series, series_len, &mut d_out,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
            ctx_guard: self._context.clone(),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_sp: &DeviceBuffer<i32>,
        d_sa: &DeviceBuffer<f32>,
        d_lp: &DeviceBuffer<i32>,
        d_la: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaApoError> {
        let func = self.module.get_function("apo_batch_f32").map_err(|_| {
            CudaApoError::MissingKernelSymbol {
                name: "apo_batch_f32",
            }
        })?;

        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => env::var("APO_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(32),
        };
        if block_x == 0 {
            block_x = 32;
        }
        block_x = block_x.max(32);
        let gx = u32::try_from(n_combos).map_err(|_| CudaApoError::LaunchConfigTooLarge {
            gx: u32::MAX,
            gy: 1,
            gz: 1,
            bx: block_x,
            by: 1,
            bz: 1,
        })?;
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut sp_ptr = d_sp.as_device_ptr().as_raw();
            let mut sa_ptr = d_sa.as_device_ptr().as_raw();
            let mut lp_ptr = d_lp.as_device_ptr().as_raw();
            let mut la_ptr = d_la.as_device_ptr().as_raw();
            let mut len_i = i32::try_from(series_len)
                .map_err(|_| CudaApoError::InvalidInput("series_len too large".into()))?;
            let mut first_i = i32::try_from(first_valid)
                .map_err(|_| CudaApoError::InvalidInput("first_valid too large".into()))?;
            let mut n_i = i32::try_from(n_combos)
                .map_err(|_| CudaApoError::InvalidInput("n_combos too large".into()))?;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut sp_ptr as *mut _ as *mut c_void,
                &mut sa_ptr as *mut _ as *mut c_void,
                &mut lp_ptr as *mut _ as *mut c_void,
                &mut la_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        short_period: i32,
        short_alpha: f32,
        long_period: i32,
        long_alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaApoError> {
        let func = self
            .module
            .get_function("apo_many_series_one_param_f32")
            .map_err(|_| CudaApoError::MissingKernelSymbol {
                name: "apo_many_series_one_param_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let gx = u32::try_from(num_series).map_err(|_| CudaApoError::LaunchConfigTooLarge {
            gx: u32::MAX,
            gy: 1,
            gz: 1,
            bx: block_x,
            by: 1,
            bz: 1,
        })?;
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut sp_i = short_period;
            let mut sa_f = short_alpha;
            let mut lp_i = long_period;
            let mut la_f = long_alpha;
            let mut ns_i = i32::try_from(num_series)
                .map_err(|_| CudaApoError::InvalidInput("num_series too large".into()))?;
            let mut sl_i = i32::try_from(series_len)
                .map_err(|_| CudaApoError::InvalidInput("series_len too large".into()))?;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut sp_i as *mut _ as *mut c_void,
                &mut sa_f as *mut _ as *mut c_void,
                &mut lp_i as *mut _ as *mut c_void,
                &mut la_f as *mut _ as *mut c_void,
                &mut ns_i as *mut _ as *mut c_void,
                &mut sl_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &ApoBatchRange,
    ) -> Result<PreparedApoBatch, CudaApoError> {
        if data_f32.is_empty() {
            return Err(CudaApoError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaApoError::InvalidInput(
                "no valid parameter combinations".into(),
            ));
        }
        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaApoError::InvalidInput("all values are NaN".into()))?;

        let max_long = combos
            .iter()
            .map(|c| c.long_period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_long == 0 || series_len - first_valid < max_long {
            return Err(CudaApoError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_long,
                series_len - first_valid
            )));
        }

        let mut short_periods = Vec::with_capacity(combos.len());
        let mut long_periods = Vec::with_capacity(combos.len());
        let mut short_alphas = Vec::with_capacity(combos.len());
        let mut long_alphas = Vec::with_capacity(combos.len());
        for p in &combos {
            let s = p.short_period.unwrap_or(0);
            let l = p.long_period.unwrap_or(0);
            if s == 0 || l == 0 || s >= l {
                return Err(CudaApoError::InvalidInput(
                    "invalid short/long periods".into(),
                ));
            }
            short_periods.push(s as i32);
            long_periods.push(l as i32);
            short_alphas.push(2.0f32 / (s as f32 + 1.0f32));
            long_alphas.push(2.0f32 / (l as f32 + 1.0f32));
        }

        Ok(PreparedApoBatch {
            combos,
            first_valid,
            series_len,
            short_periods,
            short_alphas,
            long_periods,
            long_alphas,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &ApoParams,
    ) -> Result<(Vec<i32>, i32, i32, f32, f32), CudaApoError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaApoError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaApoError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        let sp = params.short_period.unwrap_or(0) as i32;
        let lp = params.long_period.unwrap_or(0) as i32;
        if sp <= 0 || lp <= 0 || sp >= lp {
            return Err(CudaApoError::InvalidInput(
                "invalid short/long period".into(),
            ));
        }
        let a_s = 2.0f32 / (sp as f32 + 1.0f32);
        let a_l = 2.0f32 / (lp as f32 + 1.0f32);

        let mut first_valids = Vec::with_capacity(num_series);
        for s in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + s];
                if v.is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaApoError::InvalidInput(format!("series {} all NaN", s)))?;
            let remaining = series_len - fv as usize;
            if remaining < lp as usize {
                return Err(CudaApoError::InvalidInput(format!(
                    "series {} not enough valid data (need >= {}, have {})",
                    s, lp, remaining
                )));
            }
            first_valids.push(fv);
        }

        Ok((first_valids, sp, lp, a_s, a_l))
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaApoError> {
        match mem_get_info() {
            Ok((free, _)) => {
                if required_bytes.saturating_add(headroom_bytes) <= free {
                    Ok(())
                } else {
                    Err(CudaApoError::OutOfMemory {
                        required: required_bytes,
                        free,
                        headroom: headroom_bytes,
                    })
                }
            }
            Err(_) => Ok(()),
        }
    }
}

struct PreparedApoBatch {
    combos: Vec<ApoParams>,
    first_valid: usize,
    series_len: usize,
    short_periods: Vec<i32>,
    short_alphas: Vec<f32>,
    long_periods: Vec<i32>,
    long_alphas: Vec<f32>,
}

fn expand_grid(r: &ApoBatchRange) -> Result<Vec<ApoParams>, CudaApoError> {
    fn axis_u((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaApoError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                if let Some(n) = cur.checked_add(step) {
                    cur = n;
                } else {
                    break;
                }
            }
        } else {
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if let Some(n) = cur.checked_sub(step) {
                    cur = n;
                } else {
                    break;
                }
                if cur == usize::MAX {
                    break;
                }
            }
        }
        if v.is_empty() {
            return Err(CudaApoError::InvalidInput("empty parameter range".into()));
        }
        Ok(v)
    }
    let shorts = axis_u(r.short)?;
    let longs = axis_u(r.long)?;
    let mut out = Vec::with_capacity(shorts.len().saturating_mul(longs.len()));
    for &s in &shorts {
        for &l in &longs {
            if s > 0 && l > 0 && s < l {
                out.push(ApoParams {
                    short_period: Some(s),
                    long_period: Some(l),
                });
            }
        }
    }
    Ok(out)
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::apo::ApoParams;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct ApoBatchDevState {
        cuda: CudaApo,
        d_prices: DeviceBuffer<f32>,
        d_sp: DeviceBuffer<i32>,
        d_sa: DeviceBuffer<f32>,
        d_lp: DeviceBuffer<i32>,
        d_la: DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ApoBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_sp,
                    &self.d_sa,
                    &self.d_lp,
                    &self.d_la,
                    self.series_len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("apo batch kernel");
            self.cuda.synchronize().expect("apo sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaApo::new(0).expect("cuda apo");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = ApoBatchRange {
            short: (10, 10, 1),
            long: (20, 20 + PARAM_SWEEP - 1, 1),
        };
        let prep = CudaApo::prepare_batch_inputs(&price, &sweep).expect("apo prepare batch inputs");
        let n_combos = prep.combos.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_sp = DeviceBuffer::from_slice(&prep.short_periods).expect("d_sp");
        let d_sa = DeviceBuffer::from_slice(&prep.short_alphas).expect("d_sa");
        let d_lp = DeviceBuffer::from_slice(&prep.long_periods).expect("d_lp");
        let d_la = DeviceBuffer::from_slice(&prep.long_alphas).expect("d_la");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prep.series_len * n_combos) }.expect("d_out");
        cuda.synchronize().expect("sync after prep");

        Box::new(ApoBatchDevState {
            cuda,
            d_prices,
            d_sp,
            d_sa,
            d_lp,
            d_la,
            series_len: prep.series_len,
            first_valid: prep.first_valid,
            n_combos,
            d_out,
        })
    }

    struct ApoManyDevState {
        cuda: CudaApo,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        short_period: i32,
        short_alpha: f32,
        long_period: i32,
        long_alpha: f32,
        cols: usize,
        rows: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ApoManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.short_period,
                    self.short_alpha,
                    self.long_period,
                    self.long_alpha,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("apo many-series kernel");
            self.cuda.synchronize().expect("apo sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaApo::new(0).expect("cuda apo");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = ApoParams {
            short_period: Some(10),
            long_period: Some(20),
        };
        let (first_valids, sp, lp, a_s, a_l) =
            CudaApo::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("apo prepare many-series inputs");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.synchronize().expect("sync after prep");

        Box::new(ApoManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            short_period: sp,
            short_alpha: a_s,
            long_period: lp,
            long_alpha: a_l,
            cols,
            rows,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "apo",
                "one_series_many_params",
                "apo_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "apo",
                "many_series_one_param",
                "apo_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
