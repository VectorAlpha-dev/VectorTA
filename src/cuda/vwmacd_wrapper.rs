//! CUDA scaffolding for VWMACD (Volume-Weighted MACD).
//!
//! Goals (parity with ALMA wrapper style):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/vwmacd_kernel.ptx"))
//!   using DetermineTargetFromContext + OptLevel O2 with conservative fallbacks.
//! - Stream NON_BLOCKING.
//! - Warmup/NaN semantics identical to scalar classic path (SMA fast/slow; EMA signal).
//! - VRAM checks and simple chunking where necessary (not required for grid.x 1D here).

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::vwmacd::{VwmacdBatchRange, VwmacdParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;

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
pub struct CudaVwmacdPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaVwmacdPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum CudaVwmacdError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error(
        "launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

/// Triplet of VRAM-backed arrays produced by the VWMACD kernels (macd, signal, hist).
pub struct DeviceVwmacdTriplet {
    pub macd: DeviceArrayF32,
    pub signal: DeviceArrayF32,
    pub hist: DeviceArrayF32,
}
impl DeviceVwmacdTriplet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.macd.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.macd.cols
    }
}

pub struct CudaVwmacd {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaVwmacdPolicy,
}

impl CudaVwmacd {
    pub fn new(device_id: usize) -> Result<Self, CudaVwmacdError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vwmacd_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
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
            _context: context,
            policy: CudaVwmacdPolicy::default(),
        })
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaVwmacdPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaVwmacdPolicy {
        &self.policy
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaVwmacdError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaVwmacdError> {
        if let Ok((free, _)) = mem_get_info() {
            let required = required_bytes
                .checked_add(headroom_bytes)
                .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
            if required > free {
                return Err(CudaVwmacdError::OutOfMemory {
                    required,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    // ------------- Batch (one series × many params) -------------
    pub fn vwmacd_batch_dev(
        &self,
        prices_f32: &[f32],
        volumes_f32: &[f32],
        sweep: &VwmacdBatchRange,
    ) -> Result<(DeviceVwmacdTriplet, Vec<VwmacdParams>), CudaVwmacdError> {
        let len = prices_f32.len();
        if len == 0 || volumes_f32.len() != len {
            return Err(CudaVwmacdError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }
        // Support only default MA types for device kernels
        if !sweep.fast_ma_type.eq_ignore_ascii_case("sma")
            || !sweep.slow_ma_type.eq_ignore_ascii_case("sma")
            || !sweep.signal_ma_type.eq_ignore_ascii_case("ema")
        {
            return Err(CudaVwmacdError::InvalidPolicy(
                "CUDA VWMACD supports fast=\"sma\", slow=\"sma\", signal=\"ema\" only",
            ));
        }

        let combos = expand_grid(sweep)?;

        let first_valid = first_valid_pair_f32(prices_f32, volumes_f32)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("all values are NaN".into()))?;

        let mut max_macd_warm = 0usize;
        for c in &combos {
            let f = c.fast_period.unwrap();
            let s = c.slow_period.unwrap();
            let macd_warm = first_valid + f.max(s) - 1;
            if macd_warm > max_macd_warm {
                max_macd_warm = macd_warm;
            }
        }
        if len <= max_macd_warm {
            return Err(CudaVwmacdError::InvalidInput(
                "not enough valid data".into(),
            ));
        }

        // Prefix sums (f64) matching VWMA wrapper semantics
        let (pv_prefix, vol_prefix) =
            compute_prefix_sums(prices_f32, volumes_f32, first_valid, len);

        // Compact params (i32)
        let rows = combos.len();
        let fasts: Vec<i32> = combos
            .iter()
            .map(|c| c.fast_period.unwrap() as i32)
            .collect();
        let slows: Vec<i32> = combos
            .iter()
            .map(|c| c.slow_period.unwrap() as i32)
            .collect();
        let sigs: Vec<i32> = combos
            .iter()
            .map(|c| c.signal_period.unwrap() as i32)
            .collect();

        // VRAM check (prefixes + params + outputs)
        let f64_sz = std::mem::size_of::<f64>();
        let f32_sz = std::mem::size_of::<f32>();
        let i32_sz = std::mem::size_of::<i32>();
        let prefix_bytes = pv_prefix
            .len()
            .checked_mul(f64_sz)
            .and_then(|b| b.checked_add(vol_prefix.len().checked_mul(f64_sz)?))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let param_len = fasts
            .len()
            .checked_add(slows.len())
            .and_then(|n| n.checked_add(sigs.len()))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let param_bytes = param_len
            .checked_mul(i32_sz)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(3)
            .and_then(|n| n.checked_mul(f32_sz))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let bytes = prefix_bytes
            .checked_add(param_bytes)
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        CudaVwmacd::will_fit(bytes, 64 * 1024 * 1024)?;

        // Stage to device (pinned + async for large spans)
        let h_pv = LockedBuffer::from_slice(&pv_prefix)?;
        let h_vol = LockedBuffer::from_slice(&vol_prefix)?;
        let d_pv: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::from_slice_async(&*h_pv, &self.stream) }?;
        let d_vol: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::from_slice_async(&*h_vol, &self.stream) }?;
        let d_fasts = DeviceBuffer::from_slice(&fasts)?;
        let d_slows = DeviceBuffer::from_slice(&slows)?;
        let d_sigs = DeviceBuffer::from_slice(&sigs)?;

        let mut d_macd: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        self.launch_batch(
            &d_pv,
            &d_vol,
            &d_fasts,
            &d_slows,
            &d_sigs,
            len,
            first_valid,
            rows,
            &mut d_macd,
            &mut d_signal,
            &mut d_hist,
        )?;
        self.synchronize()?;

        let triplet = DeviceVwmacdTriplet {
            macd: DeviceArrayF32 {
                buf: d_macd,
                rows,
                cols: len,
            },
            signal: DeviceArrayF32 {
                buf: d_signal,
                rows,
                cols: len,
            },
            hist: DeviceArrayF32 {
                buf: d_hist,
                rows,
                cols: len,
            },
        };
        Ok((triplet, combos))
    }

    // ------------- Many-series × one-param (time-major) -------------
    pub fn vwmacd_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VwmacdParams,
    ) -> Result<DeviceVwmacdTriplet, CudaVwmacdError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        if cols == 0 || rows == 0 || prices_tm_f32.len() != expected || volumes_tm_f32.len() != expected {
            return Err(CudaVwmacdError::InvalidInput(
                "invalid time-major inputs".into(),
            ));
        }
        let f = params.fast_period.unwrap_or(12);
        let s = params.slow_period.unwrap_or(26);
        let g = params.signal_period.unwrap_or(9);
        if f == 0 || s == 0 || g == 0 {
            return Err(CudaVwmacdError::InvalidInput("zero period".into()));
        }
        if !params
            .fast_ma_type
            .as_deref()
            .unwrap_or("sma")
            .eq_ignore_ascii_case("sma")
            || !params
                .slow_ma_type
                .as_deref()
                .unwrap_or("sma")
                .eq_ignore_ascii_case("sma")
            || !params
                .signal_ma_type
                .as_deref()
                .unwrap_or("ema")
                .eq_ignore_ascii_case("ema")
        {
            return Err(CudaVwmacdError::InvalidPolicy(
                "CUDA VWMACD supports fast=\"sma\", slow=\"sma\", signal=\"ema\" only",
            ));
        }

        let first_valids = first_valids_time_major_f32(prices_tm_f32, volumes_tm_f32, cols, rows);
        // Quick check: at least one column has enough valid data
        let mut ok = false;
        for &fv in &first_valids {
            if (rows as i32 - fv) as usize > f.max(s) {
                ok = true;
                break;
            }
        }
        if !ok {
            return Err(CudaVwmacdError::InvalidInput(
                "not enough valid data".into(),
            ));
        }

        let (pv_prefix_tm, vol_prefix_tm) = compute_prefix_sums_time_major(
            prices_tm_f32,
            volumes_tm_f32,
            cols,
            rows,
            &first_valids,
        );

        // VRAM check
        let f64_sz = std::mem::size_of::<f64>();
        let f32_sz = std::mem::size_of::<f32>();
        let i32_sz = std::mem::size_of::<i32>();
        let prefix_bytes = pv_prefix_tm
            .len()
            .checked_mul(f64_sz)
            .and_then(|b| b.checked_add(vol_prefix_tm.len().checked_mul(f64_sz)?))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let param_bytes = first_valids
            .len()
            .checked_mul(i32_sz)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(3)
            .and_then(|n| n.checked_mul(f32_sz))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let bytes = prefix_bytes
            .checked_add(param_bytes)
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        CudaVwmacd::will_fit(bytes, 64 * 1024 * 1024)?;

        let h_pv = LockedBuffer::from_slice(&pv_prefix_tm)?;
        let h_vol = LockedBuffer::from_slice(&vol_prefix_tm)?;
        let d_pv: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::from_slice_async(&*h_pv, &self.stream) }?;
        let d_vol: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::from_slice_async(&*h_vol, &self.stream) }?;
        let d_first: DeviceBuffer<i32> = DeviceBuffer::from_slice(&first_valids)?;

        let mut d_macd: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        self.launch_many_series(
            &d_pv,
            &d_vol,
            &d_first,
            f,
            s,
            g,
            cols,
            rows,
            &mut d_macd,
            &mut d_signal,
            &mut d_hist,
        )?;
        self.synchronize()?;

        Ok(DeviceVwmacdTriplet {
            macd: DeviceArrayF32 {
                buf: d_macd,
                rows,
                cols,
            },
            signal: DeviceArrayF32 {
                buf: d_signal,
                rows,
                cols,
            },
            hist: DeviceArrayF32 {
                buf: d_hist,
                rows,
                cols,
            },
        })
    }

    // ---------- kernel launches ----------
    #[allow(clippy::too_many_arguments)]
    fn launch_batch(
        &self,
        d_pv: &DeviceBuffer<f64>,
        d_vol: &DeviceBuffer<f64>,
        d_fasts: &DeviceBuffer<i32>,
        d_slows: &DeviceBuffer<i32>,
        d_sigs: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        d_macd: &mut DeviceBuffer<f32>,
        d_signal: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwmacdError> {
        let func = self
            .module
            .get_function("vwmacd_batch_f32")
            .map_err(|_| CudaVwmacdError::MissingKernelSymbol {
                name: "vwmacd_batch_f32",
            })?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut pv_ptr = d_pv.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol.as_device_ptr().as_raw();
            let mut f_ptr = d_fasts.as_device_ptr().as_raw();
            let mut s_ptr = d_slows.as_device_ptr().as_raw();
            let mut g_ptr = d_sigs.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut fv_i = first_valid as i32;
            let mut rows_i = rows as i32;
            let mut macd_ptr = d_macd.as_device_ptr().as_raw();
            let mut sig_ptr = d_signal.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut g_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut macd_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series(
        &self,
        d_pv_tm: &DeviceBuffer<f64>,
        d_vol_tm: &DeviceBuffer<f64>,
        d_first: &DeviceBuffer<i32>,
        fast: usize,
        slow: usize,
        signal: usize,
        cols: usize,
        rows: usize,
        d_macd_tm: &mut DeviceBuffer<f32>,
        d_signal_tm: &mut DeviceBuffer<f32>,
        d_hist_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwmacdError> {
        let func = self
            .module
            .get_function("vwmacd_many_series_one_param_time_major_f32")
            .map_err(|_| CudaVwmacdError::MissingKernelSymbol {
                name: "vwmacd_many_series_one_param_time_major_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut pv_ptr = d_pv_tm.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut f_i = fast as i32;
            let mut s_i = slow as i32;
            let mut g_i = signal as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut macd_ptr = d_macd_tm.as_device_ptr().as_raw();
            let mut signal_ptr = d_signal_tm.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut f_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut g_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut macd_ptr as *mut _ as *mut c_void,
                &mut signal_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }
}

// ---------- Helpers (host) ----------

fn first_valid_pair_f32(close: &[f32], volume: &[f32]) -> Option<usize> {
    close
        .iter()
        .zip(volume)
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
}

fn compute_prefix_sums(
    prices: &[f32],
    volumes: &[f32],
    first_valid: usize,
    len: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut pv_prefix = vec![0f64; len];
    let mut vol_prefix = vec![0f64; len];
    let mut acc_pv = 0f64;
    let mut acc_vol = 0f64;
    for i in first_valid..len {
        let p = prices[i] as f64;
        let v = volumes[i] as f64;
        if p.is_nan() || v.is_nan() || acc_pv.is_nan() || acc_vol.is_nan() {
            acc_pv = f64::NAN;
            acc_vol = f64::NAN;
        } else {
            acc_pv += p * v;
            acc_vol += v;
        }
        pv_prefix[i] = acc_pv;
        vol_prefix[i] = acc_vol;
    }
    (pv_prefix, vol_prefix)
}

fn compute_prefix_sums_time_major(
    prices_tm: &[f32],
    volumes_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> (Vec<f64>, Vec<f64>) {
    let mut pv_prefix = vec![0f64; prices_tm.len()];
    let mut vol_prefix = vec![0f64; volumes_tm.len()];
    for series in 0..cols {
        let fv = first_valids[series].max(0) as usize;
        let mut acc_pv = 0f64;
        let mut acc_vol = 0f64;
        for r in 0..rows {
            let idx = r * cols + series;
            if r >= fv {
                let p = prices_tm[idx] as f64;
                let v = volumes_tm[idx] as f64;
                if p.is_nan() || v.is_nan() || acc_pv.is_nan() || acc_vol.is_nan() {
                    acc_pv = f64::NAN;
                    acc_vol = f64::NAN;
                } else {
                    acc_pv += p * v;
                    acc_vol += v;
                }
            }
            pv_prefix[idx] = acc_pv;
            vol_prefix[idx] = acc_vol;
        }
    }
    (pv_prefix, vol_prefix)
}

fn first_valids_time_major_f32(
    prices_tm: &[f32],
    volumes_tm: &[f32],
    cols: usize,
    rows: usize,
) -> Vec<i32> {
    let mut out = vec![0i32; cols];
    for series in 0..cols {
        let mut fv: i32 = -1;
        for r in 0..rows {
            let idx = r * cols + series;
            let c = prices_tm[idx];
            let v = volumes_tm[idx];
            if !c.is_nan() && !v.is_nan() {
                fv = r as i32;
                break;
            }
        }
        out[series] = if fv < 0 { rows as i32 } else { fv };
    }
    out
}

fn expand_grid(r: &VwmacdBatchRange) -> Result<Vec<VwmacdParams>, CudaVwmacdError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaVwmacdError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            let mut v = Vec::new();
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                let next = cur.saturating_add(st);
                if next == cur {
                    break;
                }
                cur = next;
            }
            if v.is_empty() {
                return Err(CudaVwmacdError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            return Ok(v);
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
            return Err(CudaVwmacdError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(v)
    }

    let fasts = axis_usize(r.fast)?;
    let slows = axis_usize(r.slow)?;
    let signals = axis_usize(r.signal)?;

    let cap = fasts
        .len()
        .checked_mul(slows.len())
        .and_then(|x| x.checked_mul(signals.len()))
        .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
    if cap == 0 {
        return Err(CudaVwmacdError::InvalidInput(
            "empty parameter sweep".into(),
        ));
    }

    let mut out = Vec::with_capacity(cap);
    for &f in &fasts {
        for &s in &slows {
            for &g in &signals {
                out.push(VwmacdParams {
                    fast_period: Some(f),
                    slow_period: Some(s),
                    signal_period: Some(g),
                    fast_ma_type: Some(r.fast_ma_type.clone()),
                    slow_ma_type: Some(r.slow_ma_type.clone()),
                    signal_ma_type: Some(r.signal_ma_type.clone()),
                });
            }
        }
    }
    Ok(out)
}

// ---------- Minimal benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices, gen_time_major_volumes};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_00; // 100k
    const SWEEP: (usize, usize, usize) = (8, 8 + 128 - 1, 1);
    const MANY_SERIES_COLS: usize = 128;
    const MANY_SERIES_LEN: usize = 100_000;

    fn bytes_one_series_many_params() -> usize {
        let rows = (SWEEP.1 - SWEEP.0 + 1) as usize * (SWEEP.1 - SWEEP.0 + 1) as usize; // worst approx
        let in_b = 2 * ONE_SERIES_LEN * 4;
        let out_b = 3 * rows * ONE_SERIES_LEN * 4;
        in_b + out_b + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_LEN * MANY_SERIES_COLS;
        3 * elems * 4 + 2 * elems * 4 + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaVwmacd,
        price: Vec<f32>,
        vol: Vec<f32>,
        sweep: VwmacdBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .vwmacd_batch_dev(&self.price, &self.vol, &self.sweep)
                .unwrap();
        }
    }
    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        v.push(
            CudaBenchScenario::new(
                "vwmacd",
                "one_series_many_params",
                "vwmacd_cuda_batch_dev",
                "100k_sweep",
                || {
                    let cuda = CudaVwmacd::new(0).unwrap();
                    let price = gen_series(ONE_SERIES_LEN);
                    let mut vol = gen_series(ONE_SERIES_LEN);
                    for x in &mut vol {
                        if x.is_finite() {
                            *x = x.abs() * 100.0 + 10.0;
                        }
                    }
                    let sweep = VwmacdBatchRange {
                        fast: (8, 64, 8),
                        slow: (16, 128, 8),
                        signal: (9, 9, 0),
                        fast_ma_type: "sma".into(),
                        slow_ma_type: "sma".into(),
                        signal_ma_type: "ema".into(),
                    };
                    Box::new(BatchState {
                        cuda,
                        price,
                        vol,
                        sweep,
                    })
                },
            )
            .with_mem_required(bytes_one_series_many_params()),
        );

        v.push(
            CudaBenchScenario::new(
                "vwmacd",
                "many_series_one_param",
                "vwmacd_cuda_many_series_one_param",
                "128x100k",
                || {
                    let cuda = CudaVwmacd::new(0).unwrap();
                    let price = gen_time_major_prices(MANY_SERIES_COLS, MANY_SERIES_LEN);
                    let mut vol = gen_time_major_volumes(MANY_SERIES_COLS, MANY_SERIES_LEN);
                    for x in &mut vol {
                        if x.is_finite() {
                            *x = x.abs() * 50.0 + 5.0;
                        }
                    }
                    let params = VwmacdParams {
                        fast_period: Some(12),
                        slow_period: Some(26),
                        signal_period: Some(9),
                        fast_ma_type: Some("sma".into()),
                        slow_ma_type: Some("sma".into()),
                        signal_ma_type: Some("ema".into()),
                    };
                    struct S {
                        cuda: CudaVwmacd,
                        p: Vec<f32>,
                        v: Vec<f32>,
                        cols: usize,
                        rows: usize,
                        prm: VwmacdParams,
                    }
                    impl CudaBenchState for S {
                        fn launch(&mut self) {
                            let _ = self
                                .cuda
                                .vwmacd_many_series_one_param_time_major_dev(
                                    &self.p, &self.v, self.cols, self.rows, &self.prm,
                                )
                                .unwrap();
                        }
                    }
                    Box::new(S {
                        cuda,
                        p: price,
                        v: vol,
                        cols: MANY_SERIES_COLS,
                        rows: MANY_SERIES_LEN,
                        prm: params,
                    })
                },
            )
            .with_mem_required(bytes_many_series_one_param()),
        );
        v
    }
}
