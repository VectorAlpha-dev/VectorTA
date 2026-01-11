//! CUDA wrapper for Gator Oscillator (GATOR): upper/lower histograms and their 1-bar changes.
//!
//! Pattern classification: Recurrence/IIR per-parameter (sequential over time).
//! We implement:
//! - gatorosc_batch_f32: one series × many params (one block per row; shared rings)
//! - gatorosc_many_series_one_param_f32: many series × one param (one thread per series)
//!
//! Parity items (mirrors ALMA/CWMA wrappers):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/gatorosc_kernel.ptx"))
//! - Stream NON_BLOCKING
//! - JIT options: DetermineTargetFromContext + OptLevel O2 with fallbacks
//! - VRAM check with ~64MB headroom and grid.y (or chunk count) <= 65_535
//! - Logging of selected policy when BENCH_DEBUG=1

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::gatorosc::{GatorOscBatchRange, GatorOscParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaGatorOscError {
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

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaGatorOscPolicy {
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

pub struct DeviceGatorOscQuad {
    pub upper: DeviceArrayF32,
    pub lower: DeviceArrayF32,
    pub upper_change: DeviceArrayF32,
    pub lower_change: DeviceArrayF32,
}

pub struct CudaGatorOsc {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    max_grid_x: usize,
    max_smem_per_block: usize,
    policy: CudaGatorOscPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaGatorOsc {
    pub fn new(device_id: usize) -> Result<Self, CudaGatorOscError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let ctx = Arc::new(Context::new(device)?);
        
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(65_535) as usize;
        let max_smem_per_block = device
            .get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
            .unwrap_or(48 * 1024) as usize;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/gatorosc_kernel.ptx"));
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
            context: ctx,
            device_id: device_id as u32,
            policy: CudaGatorOscPolicy::default(),
            max_grid_x,
            max_smem_per_block,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }
    #[inline]
    pub fn ctx(&self) -> Arc<Context> { self.context.clone() }

    #[inline]
    pub fn set_policy(&mut self, p: CudaGatorOscPolicy) { self.policy = p; }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] GATOR batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaGatorOsc)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] GATOR many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaGatorOsc)).debug_many_logged = true; }
            }
        }
    }

    
    pub fn gatorosc_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &GatorOscBatchRange,
    ) -> Result<DeviceGatorOscQuad, CudaGatorOscError> {
        let len = data_f32.len();
        if len == 0 {
            return Err(CudaGatorOscError::InvalidInput("empty series".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaGatorOscError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep)?;
        
        
        let warp_scan_enabled =
            std::env::var("GATOROSC_BATCH_WARP_SCAN").ok().as_deref() == Some("1");

        
        let mut jl: Vec<i32> = Vec::with_capacity(combos.len());
        let mut js: Vec<i32> = Vec::with_capacity(combos.len());
        let mut tl: Vec<i32> = Vec::with_capacity(combos.len());
        let mut ts_: Vec<i32> = Vec::with_capacity(combos.len());
        let mut ll: Vec<i32> = Vec::with_capacity(combos.len());
        let mut ls: Vec<i32> = Vec::with_capacity(combos.len());
        let mut needed_max: usize = 0;
        for p in &combos {
            let jlen = p.jaws_length.unwrap_or(13) as i32;
            let jsh = p.jaws_shift.unwrap_or(8) as i32;
            let tlen = p.teeth_length.unwrap_or(8) as i32;
            let tsh = p.teeth_shift.unwrap_or(5) as i32;
            let llen = p.lips_length.unwrap_or(5) as i32;
            let lsh = p.lips_shift.unwrap_or(3) as i32;
            if jlen <= 0 || tlen <= 0 || llen <= 0 {
                return Err(CudaGatorOscError::InvalidInput(
                    "non-positive length".into(),
                ));
            }
            let upper_needed =
                (jlen as usize).max(tlen as usize) + (jsh as usize).max(tsh as usize);
            let lower_needed =
                (tlen as usize).max(llen as usize) + (tsh as usize).max(lsh as usize);
            needed_max = needed_max.max(upper_needed.max(lower_needed));
            jl.push(jlen); js.push(jsh); tl.push(tlen); ts_.push(tsh); ll.push(llen); ls.push(lsh);
        }
        let valid_tail = len - first_valid;
        if valid_tail < needed_max {
            return Err(CudaGatorOscError::InvalidInput(format!(
                "not enough valid data: needed >= {}, valid = {}",
                needed_max, valid_tail
            )));
        }

        
        let rows = combos.len();
        let elt = std::mem::size_of::<f32>();
        let total_elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("rows*len overflow".into()))?;
        let bytes_prices = len
            .checked_mul(elt)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("bytes overflow".into()))?;
        let bytes_out = total_elems
            .checked_mul(elt)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("bytes overflow".into()))?
            .checked_mul(4)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("bytes overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        let required = bytes_out
            .checked_add(bytes_prices)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("bytes overflow".into()))?;
        CudaGatorOsc::will_fit(required, headroom)?;

        
        let d_prices = DeviceBuffer::from_slice(data_f32)?;

        
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_uchn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_lchn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;

        
        let launch_chunk = |this: &CudaGatorOsc,
                            start: usize,
                            chunk: usize,
                            ring_len_i: i32|
         -> Result<(), CudaGatorOscError> {
            let d_jl = DeviceBuffer::from_slice(&jl[start..start + chunk])?;
            let d_js = DeviceBuffer::from_slice(&js[start..start + chunk])?;
            let d_tl = DeviceBuffer::from_slice(&tl[start..start + chunk])?;
            let d_ts = DeviceBuffer::from_slice(&ts_[start..start + chunk])?;
            let d_ll = DeviceBuffer::from_slice(&ll[start..start + chunk])?;
            let d_ls = DeviceBuffer::from_slice(&ls[start..start + chunk])?;

            let func = this
                .module
                .get_function("gatorosc_batch_f32")
                .map_err(|_| CudaGatorOscError::MissingKernelSymbol { name: "gatorosc_batch_f32" })?;
            let block_x = if warp_scan_enabled {
                
                
                let mut bx = this.policy.batch_block_x.unwrap_or(32).max(32);
                bx -= bx % 32;
                if bx == 0 { bx = 32; }
                bx
            } else {
                
                1
            };
            let grid: GridSize = (chunk as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            if (chunk as u32) as usize > this.max_grid_x {
                return Err(CudaGatorOscError::LaunchConfigTooLarge { gx: chunk as u32, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
            }
            unsafe {
                (*(this as *const _ as *mut CudaGatorOsc)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
                let mut p_ptr = d_prices.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut jl_ptr = d_jl.as_device_ptr().as_raw();
                let mut js_ptr = d_js.as_device_ptr().as_raw();
                let mut tl_ptr = d_tl.as_device_ptr().as_raw();
                let mut ts_ptr = d_ts.as_device_ptr().as_raw();
                let mut ll_ptr = d_ll.as_device_ptr().as_raw();
                let mut ls_ptr = d_ls.as_device_ptr().as_raw();
                let mut ncomb_i = chunk as i32;
                let mut ring_len_param = ring_len_i;

                let row_off_elems = start
                    .checked_mul(len)
                    .ok_or_else(|| CudaGatorOscError::InvalidInput("row offset overflow".into()))?;
                let row_off_bytes = row_off_elems
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| CudaGatorOscError::InvalidInput("row offset bytes overflow".into()))?
                    as u64;
                let mut u_ptr = d_upper.as_device_ptr().as_raw().wrapping_add(row_off_bytes);
                let mut l_ptr = d_lower.as_device_ptr().as_raw().wrapping_add(row_off_bytes);
                let mut uc_ptr = d_uchn.as_device_ptr().as_raw().wrapping_add(row_off_bytes);
                let mut lc_ptr = d_lchn.as_device_ptr().as_raw().wrapping_add(row_off_bytes);

                let mut args: [*mut c_void; 15] = [
                    &mut p_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut jl_ptr as *mut _ as *mut c_void,
                    &mut js_ptr as *mut _ as *mut c_void,
                    &mut tl_ptr as *mut _ as *mut c_void,
                    &mut ts_ptr as *mut _ as *mut c_void,
                    &mut ll_ptr as *mut _ as *mut c_void,
                    &mut ls_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut ring_len_param as *mut _ as *mut c_void,
                    &mut u_ptr as *mut _ as *mut c_void,
                    &mut l_ptr as *mut _ as *mut c_void,
                    &mut uc_ptr as *mut _ as *mut c_void,
                    &mut lc_ptr as *mut _ as *mut c_void,
                ];
                let dyn_shmem = (ring_len_i as usize) * 3 * std::mem::size_of::<f32>();
                this.stream
                    .launch(&func, grid, block, dyn_shmem.try_into().unwrap(), &mut args)?;
            }
            this.stream.synchronize()?;
            Ok(())
        };

        
        let mut launched = 0usize;
        while launched < rows {
            let chunk = (rows - launched).min(self.max_grid_x);
            let chunk_max_shift = max_shift_in_range(&js, &ts_, &ls, launched, chunk);
            let ring_len_i = if warp_scan_enabled {
                
                
                let min_ring = (chunk_max_shift + 1).max(64);
                ((min_ring + 31) / 32 * 32) as i32
            } else {
                
                (chunk_max_shift + 1) as i32
            };
            launch_chunk(self, launched, chunk, ring_len_i)?;
            launched += chunk;
        }
        self.maybe_log_batch_debug();

        Ok(DeviceGatorOscQuad {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols: len,
            },
            upper_change: DeviceArrayF32 {
                buf: d_uchn,
                rows,
                cols: len,
            },
            lower_change: DeviceArrayF32 {
                buf: d_lchn,
                rows,
                cols: len,
            },
        })
    }

    
    pub fn gatorosc_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        jaws_length: usize,
        jaws_shift: usize,
        teeth_length: usize,
        teeth_shift: usize,
        lips_length: usize,
        lips_shift: usize,
    ) -> Result<DeviceGatorOscQuad, CudaGatorOscError> {
        if cols == 0 || rows == 0 {
            return Err(CudaGatorOscError::InvalidInput("invalid dims".into()));
        }
        if jaws_length == 0 || teeth_length == 0 || lips_length == 0 {
            return Err(CudaGatorOscError::InvalidInput(
                "non-positive length".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm.len() != expected {
            return Err(CudaGatorOscError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                if !prices_tm[t * cols + s].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
                if !prices_tm[t * cols + s].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            first_valids[s] =
                fv.ok_or_else(|| CudaGatorOscError::InvalidInput(format!("series {} all NaN", s)))?;
        }
        let needed_upper = jaws_length.max(teeth_length) + jaws_shift.max(teeth_shift);
        let needed_lower = teeth_length.max(lips_length) + teeth_shift.max(lips_shift);
        let needed = needed_upper.max(needed_lower);
        for s in 0..cols {
            let tail = rows - (first_valids[s] as usize);
            if tail < needed {
                return Err(CudaGatorOscError::InvalidInput(format!(
                    "series {} not enough valid data: needed >= {}, valid = {}",
                    s, needed, tail
                )));
            }
        }

        
        let elt_f32 = std::mem::size_of::<f32>();
        let elt_i32 = std::mem::size_of::<i32>();
        let bytes_prices = prices_tm
            .len()
            .checked_mul(elt_f32)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("prices bytes overflow".into()))?;
        let bytes_first = first_valids
            .len()
            .checked_mul(elt_i32)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("first_valids bytes overflow".into()))?;
        let total_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_out = total_elems
            .checked_mul(elt_f32)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("output bytes overflow".into()))?
            .checked_mul(4)
            .ok_or_else(|| CudaGatorOscError::InvalidInput("output bytes overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        let required = bytes_prices
            .checked_add(bytes_first)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaGatorOscError::InvalidInput("bytes overflow".into()))?;
        CudaGatorOsc::will_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(prices_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_uchn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_lchn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;

        let func = self
            .module
            .get_function("gatorosc_many_series_one_param_f32")
            .map_err(|_| CudaGatorOscError::MissingKernelSymbol { name: "gatorosc_many_series_one_param_f32" })?;
        
        let ring_len = (jaws_shift.max(teeth_shift).max(lips_shift) + 1) as i32;
        let per_thread_smem = (ring_len as usize) * 3 * std::mem::size_of::<f32>();
        let smem_budget = self.max_smem_per_block.saturating_sub(1024);
        let requested_block_x = self.policy.many_block_x.unwrap_or(128);
        let max_by_smem = if per_thread_smem == 0 {
            requested_block_x as usize
        } else {
            smem_budget / per_thread_smem
        };
        let mut block_x = requested_block_x.min((max_by_smem as u32).max(32));
        block_x -= block_x % 32; if block_x == 0 { block_x = 32; }
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        if (grid_x as usize) > self.max_grid_x {
            return Err(CudaGatorOscError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaGatorOsc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
            (*(self as *const _ as *mut CudaGatorOsc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut fv_ptr = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut jl_i = jaws_length as i32;
            let mut js_i = jaws_shift as i32;
            let mut tl_i = teeth_length as i32;
            let mut ts_i = teeth_shift as i32;
            let mut ll_i = lips_length as i32;
            let mut ls_i = lips_shift as i32;
            let mut ring_i = ring_len;
            let mut u_ptr = d_upper.as_device_ptr().as_raw();
            let mut l_ptr = d_lower.as_device_ptr().as_raw();
            let mut uc_ptr = d_uchn.as_device_ptr().as_raw();
            let mut lc_ptr = d_lchn.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 15] = [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut jl_i as *mut _ as *mut c_void,
                &mut js_i as *mut _ as *mut c_void,
                &mut tl_i as *mut _ as *mut c_void,
                &mut ts_i as *mut _ as *mut c_void,
                &mut ll_i as *mut _ as *mut c_void,
                &mut ls_i as *mut _ as *mut c_void,
                &mut ring_i as *mut _ as *mut c_void,
                &mut u_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut uc_ptr as *mut _ as *mut c_void,
                &mut lc_ptr as *mut _ as *mut c_void,
            ];
            let dyn_shmem = per_thread_smem * (block_x as usize);
            self.stream
                .launch(&func, grid, block, dyn_shmem.try_into().unwrap(), &mut args)?;
        }
        self.stream.synchronize()?;

        self.maybe_log_many_debug();
        Ok(DeviceGatorOscQuad {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols,
            },
            upper_change: DeviceArrayF32 {
                buf: d_uchn,
                rows,
                cols,
            },
            lower_change: DeviceArrayF32 {
                buf: d_lchn,
                rows,
                cols,
            },
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaGatorOscError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _total)) = Self::device_mem_info() {
            let need = required_bytes.saturating_add(headroom_bytes);
            if need <= free { Ok(()) } else { Err(CudaGatorOscError::OutOfMemory { required: need, free, headroom: headroom_bytes }) }
        } else {
            Ok(())
        }
    }
}


fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    if start < end {
        
        return (start..=end).step_by(step.max(1)).collect();
    }
    
    let mut v = Vec::new();
    let mut cur = start;
    let s = step.max(1);
    while cur >= end {
        v.push(cur);
        if cur < end + s {
            break;
        }
        cur = cur.saturating_sub(s);
        if cur == usize::MAX {
            break;
        }
    }
    v
}
fn expand_grid(r: &GatorOscBatchRange) -> Result<Vec<GatorOscParams>, CudaGatorOscError> {
    let jl = axis(r.jaws_length);
    let js = axis(r.jaws_shift);
    let tl = axis(r.teeth_length);
    let ts = axis(r.teeth_shift);
    let ll = axis(r.lips_length);
    let ls = axis(r.lips_shift);
    if jl.is_empty() || js.is_empty() || tl.is_empty() || ts.is_empty() || ll.is_empty() || ls.is_empty() {
        return Err(CudaGatorOscError::InvalidInput("empty sweep expansion".into()));
    }
    let cap = jl.len()
        .checked_mul(js.len()).ok_or_else(|| CudaGatorOscError::InvalidInput("sweep overflow".into()))?
        .checked_mul(tl.len()).ok_or_else(|| CudaGatorOscError::InvalidInput("sweep overflow".into()))?
        .checked_mul(ts.len()).ok_or_else(|| CudaGatorOscError::InvalidInput("sweep overflow".into()))?
        .checked_mul(ll.len()).ok_or_else(|| CudaGatorOscError::InvalidInput("sweep overflow".into()))?
        .checked_mul(ls.len()).ok_or_else(|| CudaGatorOscError::InvalidInput("sweep overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &a in &jl {
        for &b in &js {
            for &c in &tl {
                for &d in &ts {
                    for &e in &ll {
                        for &f in &ls {
                            out.push(GatorOscParams {
                                jaws_length: Some(a),
                                jaws_shift: Some(b),
                                teeth_length: Some(c),
                                teeth_shift: Some(d),
                                lips_length: Some(e),
                                lips_shift: Some(f),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline]
fn max_shift_in_range(js: &[i32], ts: &[i32], ls: &[i32], start: usize, count: usize) -> usize {
    let mut m = 0usize;
    for i in start..start + count {
        m = m.max(js[i] as usize).max(ts[i] as usize).max(ls[i] as usize);
    }
    m
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 96; 

    fn mem_required() -> usize {
        let out_bytes = 4 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + (64 << 20)
    }

    struct GatorBatchState {
        cuda: CudaGatorOsc,
        data: Vec<f32>,
        sweep: GatorOscBatchRange,
    }
    impl CudaBenchState for GatorBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .gatorosc_batch_dev(&self.data, &self.sweep)
                .expect("gator batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaGatorOsc::new(0).expect("cuda gator");
        let data = gen_series(ONE_SERIES_LEN);
        
        let sweep = GatorOscBatchRange {
            jaws_length: (8, 14, 2),
            jaws_shift: (2, 6, 2),
            teeth_length: (6, 10, 2),
            teeth_shift: (1, 5, 2),
            lips_length: (4, 8, 2),
            lips_shift: (0, 4, 2),
        };
        Box::new(GatorBatchState { cuda, data, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        
        
        if std::env::var("CUDA_BENCH_ENABLE_GATOROSC").ok().as_deref() != Some("1") {
            return Vec::new();
        }
        vec![CudaBenchScenario::new(
            "gatorosc",
            "one_series_many_params",
            "gatorosc_cuda_batch_dev",
            "1m_x_96",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(mem_required())]
    }
}
