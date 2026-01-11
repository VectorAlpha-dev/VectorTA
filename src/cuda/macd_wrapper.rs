//! CUDA wrapper for MACD (EMA fast path)
//!
//! Category: Recurrence/IIR – per-parameter time scan with SMA seeding then EMA.
//! We currently support `ma_type = "ema"` to match the scalar classic path.
//! Other MA types can be layered via the Cuda MA selector in the future.
//!
//! Parity items mirrored from ALMA/CWMA:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/macd_kernel.ptx"))
//! - Stream NON_BLOCKING
//! - JIT opts DetermineTargetFromContext + OptLevel O2 (with graceful fallbacks)
//! - VRAM check using mem_get_info with ~64MB headroom and grid.x<=65_535
//! - Bench profile hooks

#![cfg(feature = "cuda")]

use crate::indicators::macd::{expand_grid as expand_grid_host, MacdBatchRange, MacdError, MacdParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaMacdError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Out of memory on device: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch configuration too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

/// VRAM-backed array handle for MACD CUDA outputs.
pub struct DeviceArrayF32Macd {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Macd {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaMacdPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaMacd {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaMacdPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: u32,
    max_grid_x: u32,
}

impl CudaMacd {
    pub fn new(device_id: usize) -> Result<Self, CudaMacdError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/macd_kernel.ptx"));
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
            policy: CudaMacdPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
            max_grid_x,
        })
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMacdError> {
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaMacdError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }


    #[inline]
    pub fn set_policy(&mut self, p: CudaMacdPolicy) { self.policy = p; }

    #[inline]
    fn launch_1d(&self, total_items: usize, user_block_x: Option<u32>) -> (GridSize, BlockSize, u32) {
        let block_x = user_block_x.unwrap_or(256);
        let blocks_needed = ((total_items as u32 + block_x - 1) / block_x).max(1);
        let max_blocks = self.sm_count.max(1).saturating_mul(6);
        let grid_x = blocks_needed.min(max_blocks);
        (((grid_x, 1, 1)).into(), ((block_x, 1, 1)).into(), block_x)
    }

    
    pub fn macd_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MacdBatchRange,
    ) -> Result<(DeviceMacdTriplet, Vec<MacdParams>), CudaMacdError> {
        let len = data_f32.len();
        if len == 0 {
            return Err(CudaMacdError::InvalidInput("input data is empty".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaMacdError::InvalidInput("all values are NaN".into()))?;

        
        let ma0 = &sweep.ma_type.0;
        if !ma0.eq_ignore_ascii_case("ema") {
            return Err(CudaMacdError::InvalidInput(format!(
                "CUDA MACD currently supports ma_type=\"ema\" only (got \"{}\")",
                ma0
            )));
        }

        let combos = expand_grid_host(sweep)
            .map_err(|e: MacdError| CudaMacdError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaMacdError::InvalidInput("no parameter combos".into()));
        }

        
        let rows = combos.len();
        let mut fasts: Vec<i32> = Vec::with_capacity(rows);
        let mut slows: Vec<i32> = Vec::with_capacity(rows);
        let mut signals: Vec<i32> = Vec::with_capacity(rows);
        for prm in &combos {
            let f = prm.fast_period.unwrap_or(12) as i32;
            let s = prm.slow_period.unwrap_or(26) as i32;
            let g = prm.signal_period.unwrap_or(9) as i32;
            if f <= 0 || s <= 0 || g <= 0 {
                return Err(CudaMacdError::InvalidInput("non-positive periods".into()));
            }
            if len - first_valid < s as usize {
                return Err(CudaMacdError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    s,
                    len - first_valid
                )));
            }
            fasts.push(f);
            slows.push(s);
            signals.push(g);
        }

        
        let item_f32 = std::mem::size_of::<f32>();
        let item_i32 = std::mem::size_of::<i32>();
        let bytes_prices = len
            .checked_mul(item_f32)
            .ok_or_else(|| CudaMacdError::InvalidInput("series_len bytes overflow".into()))?;
        let bytes_params = rows
            .checked_mul(3)
            .and_then(|v| v.checked_mul(item_i32))
            .ok_or_else(|| CudaMacdError::InvalidInput("params bytes overflow".into()))?;
        let elems_out = rows
            .checked_mul(len)
            .ok_or_else(|| CudaMacdError::InvalidInput("rows*len overflow".into()))?;
        let bytes_out = elems_out
            .checked_mul(item_f32)
            .ok_or_else(|| CudaMacdError::InvalidInput("output bytes overflow".into()))?;
        let required = bytes_prices
            .checked_add(bytes_params)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaMacdError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        
        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_f = DeviceBuffer::from_slice(&fasts)?;
        let d_s = DeviceBuffer::from_slice(&slows)?;
        let d_g = DeviceBuffer::from_slice(&signals)?;

        
        let mut d_macd: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_out) }?;
        let mut d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_out) }?;
        let mut d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_out) }?;

        self.launch_batch_kernel(
            &d_prices,
            &d_f,
            &d_s,
            &d_g,
            len,
            first_valid,
            rows,
            &mut d_macd,
            &mut d_sig,
            &mut d_hist,
        )?;
        self.stream.synchronize()?;
        self.maybe_log_batch_debug();

        let outputs = DeviceMacdTriplet {
            macd: DeviceArrayF32Macd {
                buf: d_macd,
                rows,
                cols: len,
                ctx: Arc::clone(&self._context),
                device_id: self.device_id,
            },
            signal: DeviceArrayF32Macd {
                buf: d_sig,
                rows,
                cols: len,
                ctx: Arc::clone(&self._context),
                device_id: self.device_id,
            },
            hist: DeviceArrayF32Macd {
                buf: d_hist,
                rows,
                cols: len,
                ctx: Arc::clone(&self._context),
                device_id: self.device_id,
            },
        };
        Ok((outputs, combos))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_f: &DeviceBuffer<i32>,
        d_s: &DeviceBuffer<i32>,
        d_g: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        d_macd: &mut DeviceBuffer<f32>,
        d_sig: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMacdError> {
        if len == 0 || rows == 0 {
            return Ok(());
        }

        
        let func = self
            .module
            .get_function("macd_batch_f32")
            .map_err(|_| CudaMacdError::MissingKernelSymbol { name: "macd_batch_f32" })?;
        let mut block_x: u32 = self.policy.batch_block_x.unwrap_or(256);
        block_x = block_x.max(32);
        block_x -= block_x % 32;
        let warps_per_block = (block_x / 32).max(1);
        let max_grid_x: u32 = self.max_grid_x.max(1);
        let combos_per_launch: usize = (warps_per_block as usize) * (max_grid_x as usize);

        unsafe {
            (*(self as *const _ as *mut CudaMacd)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }

        let mut launched = 0usize;
        while launched < rows {
            let this_chunk = (rows - launched).min(combos_per_launch);
            let grid_x = ((this_chunk as u32) + warps_per_block - 1) / warps_per_block;
            let grid: GridSize = ((grid_x.max(1), 1, 1)).into();
            let block: BlockSize = ((block_x, 1, 1)).into();

            unsafe {
                let mut p_ptr = d_prices.as_device_ptr().as_raw();
                let mut f_ptr = d_f
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut s_ptr = d_s
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut g_ptr = d_g
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut rows_i = this_chunk as i32;
                let mut macd_ptr = d_macd
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(((launched * len) * std::mem::size_of::<f32>()) as u64);
                let mut sig_ptr = d_sig
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(((launched * len) * std::mem::size_of::<f32>()) as u64);
                let mut hist_ptr = d_hist
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(((launched * len) * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut p_ptr as *mut _ as *mut c_void,
                    &mut f_ptr as *mut _ as *mut c_void,
                    &mut s_ptr as *mut _ as *mut c_void,
                    &mut g_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut macd_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                    &mut hist_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += this_chunk;
        }
        Ok(())
    }

    
    pub fn macd_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MacdParams,
    ) -> Result<DeviceMacdTriplet, CudaMacdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMacdError::InvalidInput("cols or rows is zero".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMacdError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaMacdError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                expected
            )));
        }
        let ma = params.ma_type.as_deref().unwrap_or("ema");
        if !ma.eq_ignore_ascii_case("ema") {
            return Err(CudaMacdError::InvalidInput(
                "many-series MACD supports ma_type=\"ema\" only".into(),
            ));
        }
        let fast = params.fast_period.unwrap_or(12);
        let slow = params.slow_period.unwrap_or(26);
        let signal = params.signal_period.unwrap_or(9);
        if fast == 0 || slow == 0 || signal == 0 {
            return Err(CudaMacdError::InvalidInput("non-positive periods".into()));
        }

        
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv
                .ok_or_else(|| CudaMacdError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < slow {
                return Err(CudaMacdError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    s,
                    slow,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }

        
        let item_f32 = std::mem::size_of::<f32>();
        let item_i32 = std::mem::size_of::<i32>();
        let bytes_data = expected
            .checked_mul(item_f32)
            .ok_or_else(|| CudaMacdError::InvalidInput("data bytes overflow".into()))?;
        let bytes_first = cols
            .checked_mul(item_i32)
            .ok_or_else(|| CudaMacdError::InvalidInput("first_valid bytes overflow".into()))?;
        let elems_out = expected
            .checked_mul(3)
            .ok_or_else(|| CudaMacdError::InvalidInput("output elements overflow".into()))?;
        let bytes_out = elems_out
            .checked_mul(item_f32)
            .ok_or_else(|| CudaMacdError::InvalidInput("output bytes overflow".into()))?;
        let required = bytes_data
            .checked_add(bytes_first)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaMacdError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_macd: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }?;
        let mut d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }?;
        let mut d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }?;

        let func = self
            .module
            .get_function("macd_many_series_one_param_f32")
            .map_err(|_| {
                CudaMacdError::MissingKernelSymbol {
                    name: "macd_many_series_one_param_f32",
                }
            })?;
        let (grid, block, block_x_used) = self.launch_1d(cols, self.policy.many_block_x);
        unsafe {
            (*(self as *const _ as *mut CudaMacd)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: block_x_used });
        }
        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fast_i = fast as i32;
            let mut slow_i = slow as i32;
            let mut sig_i = signal as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut macd_ptr = d_macd.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fast_i as *mut _ as *mut c_void,
                &mut slow_i as *mut _ as *mut c_void,
                &mut sig_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut macd_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        self.maybe_log_many_debug();
        Ok(DeviceMacdTriplet {
            macd: DeviceArrayF32Macd {
                buf: d_macd,
                rows,
                cols,
                ctx: Arc::clone(&self._context),
                device_id: self.device_id,
            },
            signal: DeviceArrayF32Macd {
                buf: d_sig,
                rows,
                cols,
                ctx: Arc::clone(&self._context),
                device_id: self.device_id,
            },
            hist: DeviceArrayF32Macd {
                buf: d_hist,
                rows,
                cols,
                ctx: Arc::clone(&self._context),
                device_id: self.device_id,
            },
        })
    }

    fn macd_many_series_one_param_time_major_device_inplace(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        fast: usize,
        slow: usize,
        signal: usize,
        d_macd: &mut DeviceBuffer<f32>,
        d_sig: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMacdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMacdError::InvalidInput("cols or rows is zero".into()));
        }
        if fast == 0 || slow == 0 || signal == 0 {
            return Err(CudaMacdError::InvalidInput("non-positive periods".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMacdError::InvalidInput("rows*cols overflow".into()))?;
        if d_prices_tm.len() != expected {
            return Err(CudaMacdError::InvalidInput(
                "device prices buffer wrong length".into(),
            ));
        }
        if d_first_valids.len() != cols {
            return Err(CudaMacdError::InvalidInput(
                "device first_valids buffer wrong length".into(),
            ));
        }
        if d_macd.len() != expected || d_sig.len() != expected || d_hist.len() != expected {
            return Err(CudaMacdError::InvalidInput(
                "device output buffer wrong length".into(),
            ));
        }

        let func = self
            .module
            .get_function("macd_many_series_one_param_f32")
            .map_err(|_| CudaMacdError::MissingKernelSymbol {
                name: "macd_many_series_one_param_f32",
            })?;
        let (grid, block, block_x_used) = self.launch_1d(cols, self.policy.many_block_x);
        unsafe {
            (*(self as *const _ as *mut CudaMacd)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: block_x_used });
        }

        unsafe {
            let mut p_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fast_i = fast as i32;
            let mut slow_i = slow as i32;
            let mut sig_i = signal as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut macd_ptr = d_macd.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fast_i as *mut _ as *mut c_void,
                &mut slow_i as *mut _ as *mut c_void,
                &mut sig_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut macd_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }
}

/// MACD triple outputs (macd, signal, hist) retained on device.
pub struct DeviceMacdTriplet {
    pub macd: DeviceArrayF32Macd,
    pub signal: DeviceArrayF32Macd,
    pub hist: DeviceArrayF32Macd,
}

impl CudaMacd {
    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[DEBUG] MACD batch selected kernel: {:?}", sel);
                unsafe {
                    (*(self as *const _ as *mut CudaMacd)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMacd)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[DEBUG] MACD many-series selected kernel: {:?}", sel);
                unsafe {
                    (*(self as *const _ as *mut CudaMacd)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMacd)).debug_many_logged = true;
                }
            }
        }
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_COLS: usize = 256;
    const MANY_ROWS: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_b = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_b = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_b + out_b + (64 * 1024 * 1024)
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_COLS * MANY_ROWS;
        let in_b = elems * std::mem::size_of::<f32>();
        let out_b = 3 * elems * std::mem::size_of::<f32>();
        in_b + out_b + (64 * 1024 * 1024)
    }

    struct MacdBatchDeviceState {
        cuda: CudaMacd,
        d_prices: DeviceBuffer<f32>,
        d_f: DeviceBuffer<i32>,
        d_s: DeviceBuffer<i32>,
        d_g: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        d_macd: DeviceBuffer<f32>,
        d_sig: DeviceBuffer<f32>,
        d_hist: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MacdBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_f,
                    &self.d_s,
                    &self.d_g,
                    self.len,
                    self.first_valid,
                    self.rows,
                    &mut self.d_macd,
                    &mut self.d_sig,
                    &mut self.d_hist,
                )
                .expect("macd launch");
            self.cuda.stream.synchronize().expect("macd sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMacd::new(0).expect("cuda macd");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = MacdBatchRange {
            fast_period: (12, 12 + PARAM_SWEEP - 1, 1),
            slow_period: (26, 26, 0),
            signal_period: (9, 9, 0),
            ma_type: ("ema".to_string(), "ema".to_string(), String::new()),
        };

        let len = price.len();
        let first_valid = price.iter().position(|v| !v.is_nan()).unwrap_or(len);
        let combos = expand_grid_host(&sweep).expect("expand_grid_host");
        let rows = combos.len();
        let mut fasts: Vec<i32> = Vec::with_capacity(rows);
        let mut slows: Vec<i32> = Vec::with_capacity(rows);
        let mut signals: Vec<i32> = Vec::with_capacity(rows);
        for prm in &combos {
            fasts.push(prm.fast_period.unwrap_or(12) as i32);
            slows.push(prm.slow_period.unwrap_or(26) as i32);
            signals.push(prm.signal_period.unwrap_or(9) as i32);
        }

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices H2D");
        let d_f = DeviceBuffer::from_slice(&fasts).expect("d_f H2D");
        let d_s = DeviceBuffer::from_slice(&slows).expect("d_s H2D");
        let d_g = DeviceBuffer::from_slice(&signals).expect("d_g H2D");

        let elems_out = rows * len;
        let d_macd: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_out) }.expect("d_macd alloc");
        let d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_out) }.expect("d_sig alloc");
        let d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_out) }.expect("d_hist alloc");
        cuda.stream.synchronize().expect("macd prep sync");

        Box::new(MacdBatchDeviceState {
            cuda,
            d_prices,
            d_f,
            d_s,
            d_g,
            len,
            first_valid,
            rows,
            d_macd,
            d_sig,
            d_hist,
        })
    }

    struct MacdManyState {
        cuda: CudaMacd,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        fast: usize,
        slow: usize,
        signal: usize,
        d_macd: DeviceBuffer<f32>,
        d_sig: DeviceBuffer<f32>,
        d_hist: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MacdManyState {
        fn launch(&mut self) {
            self.cuda
                .macd_many_series_one_param_time_major_device_inplace(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.fast,
                    self.slow,
                    self.signal,
                    &mut self.d_macd,
                    &mut self.d_sig,
                    &mut self.d_hist,
                )
                .expect("macd many launch");
            self.cuda.stream.synchronize().expect("macd many sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaMacd::new(0).expect("cuda macd");
        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let data_tm = gen_time_major_prices(cols, rows);
        let (fast, slow, signal) = (12usize, 26usize, 9usize);
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.unwrap_or(0);
            if rows - fv < slow {
                panic!("macd many-series: series {s} has insufficient valid data");
            }
            first_valids[s] = fv as i32;
        }

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm H2D");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids H2D");
        let expected = cols * rows;
        let d_macd: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.expect("d_macd alloc");
        let d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.expect("d_sig alloc");
        let d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.expect("d_hist alloc");
        cuda.stream.synchronize().expect("macd many prep sync");

        Box::new(MacdManyState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            fast,
            slow,
            signal,
            d_macd,
            d_sig,
            d_hist,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "macd",
                "one_series_many_params",
                "macd_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "macd",
                "many_series_one_param",
                "macd_cuda_many_series_one_param",
                "256x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
