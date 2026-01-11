//! CUDA scaffolding for KST (Know Sure Thing)
//!
//! Parity goals (aligned with ALMA wrapper):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/kst_kernel.ptx"))
//!   using DetermineTargetFromContext + OptLevel O2, with conservative fallbacks
//! - Stream NON_BLOCKING
//! - Warmup/NaN semantics identical to scalar
//! - VRAM checks and simple chunking where applicable

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::kst::{KstBatchRange, KstParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
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
pub struct CudaKstPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaKstPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Error, Debug)]
pub enum CudaKstError {
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
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

/// Pair of VRAM-backed arrays produced by the KST kernels (line + signal).
pub struct DeviceKstPair {
    pub line: DeviceArrayF32,
    pub signal: DeviceArrayF32,
}
impl DeviceKstPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.line.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.line.cols
    }
}



struct PackedParamPtrs {
    _buf: DeviceBuffer<i32>,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    r1: u64,
    r2: u64,
    r3: u64,
    r4: u64,
    sg: u64,
}

pub struct CudaKst {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaKstPolicy,
    
    sm_count: i32,
}

impl CudaKst {
    pub fn new(device_id: usize) -> Result<Self, CudaKstError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kst_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };

        
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map_err(CudaKstError::Cuda)? as i32;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaKstPolicy::default(),
            sm_count,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaKstPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaKstPolicy {
        &self.policy
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaKstError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaKstError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            let need = required_bytes
                .checked_add(headroom_bytes)
                .ok_or(CudaKstError::OutOfMemory {
                    required: usize::MAX,
                    free,
                    headroom: headroom_bytes,
                })?;
            if need <= free {
                Ok(())
            } else {
                Err(CudaKstError::OutOfMemory {
                    required: need,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    

    #[inline]
    fn launch_shape_1d(&self, n_items: usize, block_policy: Option<u32>) -> (GridSize, BlockSize) {
        let block_x = match block_policy {
            Some(bx) => bx.max(64),
            None => 256,
        };
        let grid_for_data = ((n_items as u32) + block_x - 1) / block_x;
        let target = (self.sm_count.max(1) as u32) * 32; 
        let grid_x = std::cmp::min(grid_for_data.max(1), target.max(1));
        ((grid_x, 1, 1).into(), (block_x, 1, 1).into())
    }

    const PIN_THRESHOLD_BYTES: usize = 1 << 20; 

    #[inline]
    fn copy_f32_to_device_async(&self, src: &[f32]) -> Result<DeviceBuffer<f32>, CudaKstError> {
        if src
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in copy_f32_to_device_async".into(),
            ))?
            >= Self::PIN_THRESHOLD_BYTES
        {
            let h_pinned = LockedBuffer::from_slice(src)?;
        let mut d = unsafe { DeviceBuffer::uninitialized_async(src.len(), &self.stream)? };
            unsafe {
                d.async_copy_from(&h_pinned, &self.stream)?;
            }
            Ok(d)
        } else {
            unsafe { DeviceBuffer::from_slice_async(src, &self.stream) }.map_err(Into::into)
        }
    }

    
    

    fn pack_params_async(&self, combos: &[KstParams]) -> Result<PackedParamPtrs, CudaKstError> {
        let rows = combos.len();
        let total = rows
            .checked_mul(9)
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in packed parameter buffer".into(),
            ))?;
        let mut host = Vec::<i32>::with_capacity(total);
        host.resize(total, 0);
        let (s1s, rem) = host.split_at_mut(rows);
        let (s2s, rem) = rem.split_at_mut(rows);
        let (s3s, rem) = rem.split_at_mut(rows);
        let (s4s, rem) = rem.split_at_mut(rows);
        let (r1s, rem) = rem.split_at_mut(rows);
        let (r2s, rem) = rem.split_at_mut(rows);
        let (r3s, rem) = rem.split_at_mut(rows);
        let (r4s, sgs) = rem.split_at_mut(rows);

        for (i, c) in combos.iter().enumerate() {
            s1s[i] = c.sma_period1.unwrap() as i32;
            s2s[i] = c.sma_period2.unwrap() as i32;
            s3s[i] = c.sma_period3.unwrap() as i32;
            s4s[i] = c.sma_period4.unwrap() as i32;
            r1s[i] = c.roc_period1.unwrap() as i32;
            r2s[i] = c.roc_period2.unwrap() as i32;
            r3s[i] = c.roc_period3.unwrap() as i32;
            r4s[i] = c.roc_period4.unwrap() as i32;
            sgs[i] = c.signal_period.unwrap() as i32;
        }

        let h_pinned = LockedBuffer::from_slice(&host)?;
        let mut d = unsafe {
            DeviceBuffer::uninitialized_async(total, &self.stream)?
        };
        unsafe {
            d.async_copy_from(&h_pinned, &self.stream)?;
        }

        let base = d.as_device_ptr();
        let off = |seg: usize| unsafe { base.offset((seg * rows) as isize).as_raw() };

        Ok(PackedParamPtrs {
            _buf: d,
            s1: off(0),
            s2: off(1),
            s3: off(2),
            s4: off(3),
            r1: off(4),
            r2: off(5),
            r3: off(6),
            r4: off(7),
            sg: off(8),
        })
    }

    

    fn expand_grid(range: &KstBatchRange) -> Vec<KstParams> {
        fn axis(t: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = t;
            if step == 0 {
                return vec![start];
            }
            if start == end {
                return vec![start];
            }
            let mut out = Vec::new();
            if start < end {
                let mut v = start;
                while v <= end {
                    out.push(v);
                    let next = match v.checked_add(step) {
                        Some(n) if n > v => n,
                        _ => break,
                    };
                    v = next;
                }
            } else {
                let mut v = start;
                while v >= end {
                    out.push(v);
                    if v - end < step {
                        break;
                    }
                    v -= step;
                }
            }
            out
        }
        let s1 = axis(range.sma_period1);
        let s2 = axis(range.sma_period2);
        let s3 = axis(range.sma_period3);
        let s4 = axis(range.sma_period4);
        let r1 = axis(range.roc_period1);
        let r2 = axis(range.roc_period2);
        let r3 = axis(range.roc_period3);
        let r4 = axis(range.roc_period4);
        let sg = axis(range.signal_period);
        let mut out = Vec::with_capacity(
            s1.len()
                * s2.len()
                * s3.len()
                * s4.len()
                * r1.len()
                * r2.len()
                * r3.len()
                * r4.len()
                * sg.len(),
        );
        for &a in &s1 {
            for &b in &s2 {
                for &c in &s3 {
                    for &d in &s4 {
                        for &e in &r1 {
                            for &f in &r2 {
                                for &g in &r3 {
                                    for &h in &r4 {
                                        for &q in &sg {
                                            out.push(KstParams {
                                                sma_period1: Some(a),
                                                sma_period2: Some(b),
                                                sma_period3: Some(c),
                                                sma_period4: Some(d),
                                                roc_period1: Some(e),
                                                roc_period2: Some(f),
                                                roc_period3: Some(g),
                                                roc_period4: Some(h),
                                                signal_period: Some(q),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    pub fn kst_batch_dev(
        &self,
        prices: &[f32],
        sweep: &KstBatchRange,
    ) -> Result<(DeviceKstPair, Vec<KstParams>), CudaKstError> {
        if prices.is_empty() {
            return Err(CudaKstError::InvalidInput("empty price input".into()));
        }
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaKstError::InvalidInput("empty parameter sweep".into()));
        }
        let len = prices.len();
        let first_valid = (0..len)
            .find(|&i| !prices[i].is_nan())
            .ok_or_else(|| CudaKstError::InvalidInput("all values are NaN".into()))?;

        
        let mut max_warm_line = 0usize;
        let mut max_sig = 0usize;
        for c in &combos {
            let wl = (c.roc_period1.unwrap() + c.sma_period1.unwrap() - 1)
                .max(c.roc_period2.unwrap() + c.sma_period2.unwrap() - 1)
                .max(c.roc_period3.unwrap() + c.sma_period3.unwrap() - 1)
                .max(c.roc_period4.unwrap() + c.sma_period4.unwrap() - 1);
            let ws = wl + c.signal_period.unwrap() - 1;
            if wl > max_warm_line {
                max_warm_line = wl;
            }
            if ws > max_sig {
                max_sig = ws;
            }
        }
        if len - first_valid <= max_warm_line {
            return Err(CudaKstError::InvalidInput(
                "not enough valid data for KST warmup".into(),
            ));
        }

        let rows = combos.len();
        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in prices_bytes".into(),
            ))?;
        let params_bytes = rows
            .checked_mul(9)
            .and_then(|x| x.checked_mul(std::mem::size_of::<i32>()))
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in params_bytes".into(),
            ))?;
        let out_bytes = rows
            .checked_mul(len)
            .and_then(|x| x.checked_mul(2))
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in output bytes".into(),
            ))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in total VRAM estimate".into(),
            ))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        
        let d_prices: DeviceBuffer<f32> = self.copy_f32_to_device_async(prices)?;
        
        let packed = self.pack_params_async(&combos)?;

        let out_len = rows
            .checked_mul(len)
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in device output buffers".into(),
            ))?;
        let mut d_line: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &self.stream) }?;
        let mut d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &self.stream) }?;

        self.launch_batch_packed(
            &d_prices,
            packed.s1,
            packed.s2,
            packed.s3,
            packed.s4,
            packed.r1,
            packed.r2,
            packed.r3,
            packed.r4,
            packed.sg,
            len,
            rows,
            first_valid,
            &mut d_line,
            &mut d_signal,
        )?;
        self.synchronize()?;

        let pair = DeviceKstPair {
            line: DeviceArrayF32 {
                buf: d_line,
                rows,
                cols: len,
            },
            signal: DeviceArrayF32 {
                buf: d_signal,
                rows,
                cols: len,
            },
        };
        Ok((pair, combos))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_packed(
        &self,
        d_prices: &DeviceBuffer<f32>,
        s1_ptr: u64,
        s2_ptr: u64,
        s3_ptr: u64,
        s4_ptr: u64,
        r1_ptr: u64,
        r2_ptr: u64,
        r3_ptr: u64,
        r4_ptr: u64,
        sig_ptr: u64,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_line: &mut DeviceBuffer<f32>,
        d_signal: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKstError> {
        let (grid, block) = {
            let bx = match self.policy.batch {
                BatchKernelPolicy::Auto => None,
                BatchKernelPolicy::Plain { block_x } => Some(block_x),
            };
            self.launch_shape_1d(n_combos, bx)
        };

        unsafe {
            let mut p0 = d_prices.as_device_ptr().as_raw();
            let mut s1 = s1_ptr;
            let mut s2 = s2_ptr;
            let mut s3 = s3_ptr;
            let mut s4 = s4_ptr;
            let mut r1 = r1_ptr;
            let mut r2 = r2_ptr;
            let mut r3 = r3_ptr;
            let mut r4 = r4_ptr;
            let mut sg = sig_ptr;
            let mut sl = series_len as i32;
            let mut nc = n_combos as i32;
            let mut fv = first_valid as i32;
            let mut out_l = d_line.as_device_ptr().as_raw();
            let mut out_s = d_signal.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut p0 as *mut _ as *mut c_void,
                &mut s1 as *mut _ as *mut c_void,
                &mut s2 as *mut _ as *mut c_void,
                &mut s3 as *mut _ as *mut c_void,
                &mut s4 as *mut _ as *mut c_void,
                &mut r1 as *mut _ as *mut c_void,
                &mut r2 as *mut _ as *mut c_void,
                &mut r3 as *mut _ as *mut c_void,
                &mut r4 as *mut _ as *mut c_void,
                &mut sg as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut out_l as *mut _ as *mut c_void,
                &mut out_s as *mut _ as *mut c_void,
            ];
            let func = self
                .module
                .get_function("kst_batch_f32")
                .map_err(|_| CudaKstError::MissingKernelSymbol { name: "kst_batch_f32" })?;
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaKstError::Cuda)?;
        }
        Ok(())
    }

    
    pub fn kst_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &KstParams,
    ) -> Result<DeviceKstPair, CudaKstError> {
        if cols == 0 || rows == 0 {
            return Err(CudaKstError::InvalidInput(
                "cols/rows must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaKstError::InvalidInput(
                "time-major buffer size mismatch".into(),
            ));
        }
        let s1 = params.sma_period1.unwrap_or(10);
        let s2 = params.sma_period2.unwrap_or(10);
        let s3 = params.sma_period3.unwrap_or(10);
        let s4 = params.sma_period4.unwrap_or(15);
        let r1 = params.roc_period1.unwrap_or(10);
        let r2 = params.roc_period2.unwrap_or(15);
        let r3 = params.roc_period3.unwrap_or(20);
        let r4 = params.roc_period4.unwrap_or(30);
        let sig = params.signal_period.unwrap_or(9);

        
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                fv = rows as i32;
            }
            first_valids[s] = fv;
        }

        let elems = data_tm_f32.len();
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in time-major prices_bytes".into(),
            ))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in first_valids bytes".into(),
            ))?;
        let out_bytes = elems
            .checked_mul(2)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in time-major outputs".into(),
            ))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in total VRAM estimate".into(),
            ))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_prices_tm: DeviceBuffer<f32> = self.copy_f32_to_device_async(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaKstError::Cuda)?;
        let out_len = cols
            .checked_mul(rows)
            .ok_or(CudaKstError::InvalidInput(
                "size overflow in many-series device outputs".into(),
            ))?;
        let mut d_line_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &self.stream) }?;
        let mut d_sig_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &self.stream) }?;

        self.launch_many_series(
            &d_prices_tm,
            cols,
            rows,
            s1,
            s2,
            s3,
            s4,
            r1,
            r2,
            r3,
            r4,
            sig,
            &d_first,
            &mut d_line_tm,
            &mut d_sig_tm,
        )?;
        self.synchronize()?;

        Ok(DeviceKstPair {
            line: DeviceArrayF32 {
                buf: d_line_tm,
                rows,
                cols,
            },
            signal: DeviceArrayF32 {
                buf: d_sig_tm,
                rows,
                cols,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        s1: usize,
        s2: usize,
        s3: usize,
        s4: usize,
        r1: usize,
        r2: usize,
        r3: usize,
        r4: usize,
        sig: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_line_tm: &mut DeviceBuffer<f32>,
        d_sig_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKstError> {
        let (grid, block) = {
            let bx = match self.policy.many_series {
                ManySeriesKernelPolicy::Auto => None,
                ManySeriesKernelPolicy::OneD { block_x } => Some(block_x),
            };
            self.launch_shape_1d(cols, bx)
        };

        unsafe {
            let mut p = d_prices_tm.as_device_ptr().as_raw();
            let mut ns = cols as i32;
            let mut sl = rows as i32;
            let mut s1_ = s1 as i32;
            let mut s2_ = s2 as i32;
            let mut s3_ = s3 as i32;
            let mut s4_ = s4 as i32;
            let mut r1_ = r1 as i32;
            let mut r2_ = r2 as i32;
            let mut r3_ = r3 as i32;
            let mut r4_ = r4 as i32;
            let mut sig_ = sig as i32;
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut out_l = d_line_tm.as_device_ptr().as_raw();
            let mut out_s = d_sig_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p as *mut _ as *mut c_void,
                &mut ns as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut s1_ as *mut _ as *mut c_void,
                &mut s2_ as *mut _ as *mut c_void,
                &mut s3_ as *mut _ as *mut c_void,
                &mut s4_ as *mut _ as *mut c_void,
                &mut r1_ as *mut _ as *mut c_void,
                &mut r2_ as *mut _ as *mut c_void,
                &mut r3_ as *mut _ as *mut c_void,
                &mut r4_ as *mut _ as *mut c_void,
                &mut sig_ as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut out_l as *mut _ as *mut c_void,
                &mut out_s as *mut _ as *mut c_void,
            ];
            let func = self
                .module
                .get_function("kst_many_series_one_param_f32")
                .map_err(|_| CudaKstError::MissingKernelSymbol {
                    name: "kst_many_series_one_param_f32",
                })?;
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaKstError::Cuda)?;
        }
        Ok(())
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 125; 
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * 4;
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * 4 * 2; 
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * 4;
        let out_bytes = elems * 4 * 2; 
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaKst,
        d_prices: DeviceBuffer<f32>,
        packed: PackedParamPtrs,
        d_line: DeviceBuffer<f32>,
        d_signal: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_packed(
                    &self.d_prices,
                    self.packed.s1,
                    self.packed.s2,
                    self.packed.s3,
                    self.packed.s4,
                    self.packed.r1,
                    self.packed.r2,
                    self.packed.r3,
                    self.packed.r4,
                    self.packed.sg,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_line,
                    &mut self.d_signal,
                )
                .expect("kst launch_batch_packed");
            let _ = self.cuda.synchronize();
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKst::new(0).expect("cuda kst");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = KstBatchRange {
            sma_period1: (10, 10 + PARAM_SWEEP - 1, 1),
            sma_period2: (10, 10, 0),
            sma_period3: (10, 10, 0),
            sma_period4: (15, 15, 0),
            roc_period1: (10, 10, 0),
            roc_period2: (15, 15, 0),
            roc_period3: (20, 20, 0),
            roc_period4: (30, 30, 0),
            signal_period: (9, 9, 0),
        };
        let combos = CudaKst::expand_grid(&sweep);
        let first_valid = price.iter().position(|v| !v.is_nan()).unwrap_or(0);
        let d_prices = cuda.copy_f32_to_device_async(&price).expect("d_prices");
        let packed = cuda.pack_params_async(&combos).expect("packed params");
        let out_len = combos.len() * ONE_SERIES_LEN;
        let d_line: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &cuda.stream) }.expect("d_line");
        let d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &cuda.stream) }.expect("d_signal");
        cuda.synchronize().expect("sync after prep");
        Box::new(BatchState {
            cuda,
            d_prices,
            packed,
            d_line,
            d_signal,
            series_len: ONE_SERIES_LEN,
            n_combos: combos.len(),
            first_valid,
        })
    }

    struct ManyState {
        cuda: CudaKst,
        d_prices_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_line_tm: DeviceBuffer<f32>,
        d_sig_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        params: KstParams,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series(
                    &self.d_prices_tm,
                    self.cols,
                    self.rows,
                    self.params.sma_period1.unwrap(),
                    self.params.sma_period2.unwrap(),
                    self.params.sma_period3.unwrap(),
                    self.params.sma_period4.unwrap(),
                    self.params.roc_period1.unwrap(),
                    self.params.roc_period2.unwrap(),
                    self.params.roc_period3.unwrap(),
                    self.params.roc_period4.unwrap(),
                    self.params.signal_period.unwrap(),
                    &self.d_first,
                    &mut self.d_line_tm,
                    &mut self.d_sig_tm,
                )
                .expect("kst launch_many_series");
            let _ = self.cuda.synchronize();
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaKst::new(0).expect("cuda kst");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = KstParams {
            sma_period1: Some(10),
            sma_period2: Some(10),
            sma_period3: Some(10),
            sma_period4: Some(15),
            roc_period1: Some(10),
            roc_period2: Some(15),
            roc_period3: Some(20),
            roc_period4: Some(30),
            signal_period: Some(9),
        };
        let first_valids: Vec<i32> = (0..cols).map(|s| s as i32).collect();
        let d_prices_tm = cuda.copy_f32_to_device_async(&data_tm).expect("d_prices_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let elems = cols * rows;
        let d_line_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_line_tm");
        let d_sig_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_sig_tm");
        cuda.synchronize().expect("sync after prep");
        Box::new(ManyState {
            cuda,
            d_prices_tm,
            d_first,
            d_line_tm,
            d_sig_tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "kst",
                "one_series_many_params",
                "kst_cuda_batch_dev",
                "1m_x_125",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "kst",
                "many_series_one_param",
                "kst_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(6)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
