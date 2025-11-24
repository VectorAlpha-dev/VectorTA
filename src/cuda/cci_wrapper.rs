//! CUDA wrapper for CCI (Commodity Channel Index)
//!
//! Parity goals:
//! - API mirrors ALMA-style wrappers: PTX load via OUT_DIR, NON_BLOCKING stream
//! - VRAM estimation and basic headroom guard
//! - Batch (one series × many params) and many-series × one-param (time-major)
//! - Warmup/NaN semantics identical to scalar CCI

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::cci::{CciBatchRange, CciParams};
use cust::context::{CacheConfig, Context, SharedMemoryConfig};
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::sys as cu;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaCciError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid range: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaCci {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    debug_batch_logged: bool,
    smem_optin_limit: usize, // max per-block dyn shared memory (bytes), opt-in, minus reservation
}

impl CudaCci {
    pub fn new(device_id: usize) -> Result<Self, CudaCciError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cci_kernel.ptx"));
        let opt = match env::var("CCI_JIT_OPT").ok().as_deref() {
            Some("O0") => OptLevel::O0,
            Some("O1") => OptLevel::O1,
            Some("O2") => OptLevel::O2,
            Some("O3") => OptLevel::O3,
            Some("O4") => OptLevel::O4,
            _ => OptLevel::O2,
        };
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(opt),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let smem_optin_limit = Self::query_optin_smem_limit_bytes();

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            debug_batch_logged: false,
            smem_optin_limit,
        })
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn stream_handle_usize(&self) -> usize {
        self.stream.as_inner() as usize
    }

    fn query_optin_smem_limit_bytes() -> usize {
        unsafe {
            let mut dev: cu::CUdevice = std::mem::zeroed();
            if cu::cuCtxGetDevice(&mut dev) != cu::CUresult::CUDA_SUCCESS {
                return 48 * 1024; // safe baseline
            }

            let mut optin: i32 = 0;
            let _ = cu::cuDeviceGetAttribute(
                &mut optin as *mut i32,
                cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                dev,
            );

            let mut def: i32 = 0;
            let _ = cu::cuDeviceGetAttribute(
                &mut def as *mut i32,
                cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                dev,
            );

            let cap = if optin > 0 { optin as usize } else { def as usize };
            cap.saturating_sub(1024) // leave 1 KiB per-block reservation
        }
    }

    #[inline]
    fn use_async() -> bool {
        match env::var("CCI_ASYNC") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> Result<(), CudaCciError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Ok((free, _total)) = mem_get_info() {
            if required_bytes.saturating_add(headroom) <= free {
                Ok(())
            } else {
                Err(CudaCciError::OutOfMemory { required: required_bytes, free, headroom })
            }
        } else {
            // If we cannot query, allow and let allocation fail with a CUDA error.
            Ok(())
        }
    }

    fn expand_periods(range: &CciBatchRange) -> Result<Vec<CciParams>, CudaCciError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaCciError> {
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
                vals.sort_unstable();
                vals.dedup();
            }
            if vals.is_empty() {
                return Err(CudaCciError::InvalidRange { start, end, step });
            }
            Ok(vals)
        }

        let periods = axis_usize(range.period)?;
        let mut out = Vec::with_capacity(periods.len());
        for p in periods {
            out.push(CciParams { period: Some(p) });
        }
        Ok(out)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &CciBatchRange,
    ) -> Result<(Vec<CciParams>, usize, usize), CudaCciError> {
        if data_f32.is_empty() {
            return Err(CudaCciError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaCciError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_periods(sweep)?;
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaCciError::InvalidInput("period must be >=1".into()));
            }
            if p > len {
                return Err(CudaCciError::InvalidInput(format!(
                    "period {} > len {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaCciError::InvalidInput(format!(
                    "not enough valid data for period {} (tail={})",
                    p,
                    len - first_valid
                )));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
        periods_offset: usize,
        out_offset_elems: usize,
        dyn_smem_bytes: usize,
    ) -> Result<(), CudaCciError> {
        let mut func: Function = self
            .module
            .get_function("cci_batch_f32")
            .map_err(|_| CudaCciError::MissingKernelSymbol { name: "cci_batch_f32" })?;

        // Favor shared memory over L1 when beneficial on unified L1/SMEM arch; 4-byte banks for f32
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);

        // Clamp dynamic SMEM by device's opt-in limit
        let dyn_smem = dyn_smem_bytes.min(self.smem_optin_limit);

        // Opt-in to dynamic SMEM > 48 KiB and prefer SMEM carveout
        unsafe {
            let raw = func.to_raw();
            let _ = cu::cuFuncSetAttribute(
                raw,
                cu::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                cu::CUshared_carveout_enum::CU_SHAREDMEM_CARVEOUT_MAX_SHARED as i32,
            );
            let _ = cu::cuFuncSetAttribute(
                raw,
                cu::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                dyn_smem as i32,
            );
        }

        // Use CUDA-suggested block size considering dynamic shared memory
        let block_x: u32 = match env::var("CCI_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                let (_mg, suggest) = func
                    .suggested_launch_configuration(dyn_smem, BlockSize::xyz(0, 0, 0))?;
                suggest
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
        };
        // One block per combo (kernel uses 1 active lane per block for simplicity)
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x.max(64), 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                + (periods_offset * std::mem::size_of::<i32>()) as u64;
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw()
                + (out_offset_elems * std::mem::size_of::<f32>()) as u64;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, dyn_smem as u32, args)?;
        }
        Ok(())
    }

    pub fn cci_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &CciBatchRange,
    ) -> Result<DeviceArrayF32, CudaCciError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();
        // VRAM estimate: inputs + params + outputs with checked arithmetic
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_b = len.checked_mul(sz_f32).ok_or_else(|| CudaCciError::InvalidInput("series_len bytes overflow".into()))?;
        let params_b = rows.checked_mul(sz_i32).ok_or_else(|| CudaCciError::InvalidInput("params bytes overflow".into()))?;
        let out_elems = rows.checked_mul(len).ok_or_else(|| CudaCciError::InvalidInput("rows*len overflow".into()))?;
        let out_b = out_elems.checked_mul(sz_f32).ok_or_else(|| CudaCciError::InvalidInput("out bytes overflow".into()))?;
        let bytes = prices_b.checked_add(params_b).and_then(|x| x.checked_add(out_b)).ok_or_else(|| CudaCciError::InvalidInput("total bytes overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        Self::will_fit(bytes, headroom)?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let periods_u: Vec<usize> = periods.iter().map(|&p| p as usize).collect();

        if Self::use_async() {
            let h_prices = LockedBuffer::from_slice(data_f32).map_err(CudaCciError::Cuda)?;
            let h_periods = LockedBuffer::from_slice(&periods).map_err(CudaCciError::Cuda)?;
            let mut d_prices =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }?;
            let mut d_periods =
                unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }?;
            let mut d_out =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(out_elems, &self.stream) }?;
            unsafe {
                d_prices.async_copy_from(&h_prices, &self.stream).map_err(CudaCciError::Cuda)?;
                d_periods.async_copy_from(&h_periods, &self.stream).map_err(CudaCciError::Cuda)?;
            }
            // Chunk launches if rows is extremely large; keep <= 65_535 blocks as a conservative limit
            let max_blocks: usize = 65_535;
            let mut launched = 0usize;
            while launched < rows {
                let n_this = std::cmp::min(max_blocks, rows - launched);
                let periods_off = launched;
                let out_off = launched * len;
                // dynamic shared memory for this chunk = max(period) * sizeof(f32)
                let max_p_this = periods_u[launched..launched + n_this]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
                let dyn_smem_bytes = max_p_this * std::mem::size_of::<f32>();
                self.launch_batch_kernel(
                    &d_prices,
                    &d_periods,
                    len,
                    n_this,
                    first_valid,
                    &mut d_out,
                    periods_off,
                    out_off,
                    dyn_smem_bytes,
                )?;
                launched += n_this;
            }
            self.stream.synchronize().map_err(CudaCciError::Cuda)?;
            if !self.debug_batch_logged && env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
                eprintln!(
                    "[cci] batch kernel: Plain, block_x=auto, chunked={} rows",
                    rows
                );
            }
            Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            })
        } else {
            let d_prices = DeviceBuffer::from_slice(data_f32).map_err(CudaCciError::Cuda)?;
            let d_periods = DeviceBuffer::from_slice(&periods).map_err(CudaCciError::Cuda)?;
            let mut d_out =
                unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }.map_err(CudaCciError::Cuda)?;
            let max_blocks: usize = 65_535;
            let mut launched = 0usize;
            while launched < rows {
                let n_this = std::cmp::min(max_blocks, rows - launched);
                // dynamic shared memory for this chunk = max(period) * sizeof(f32)
                let max_p_this = periods_u[launched..launched + n_this]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
                let dyn_smem_bytes = max_p_this * std::mem::size_of::<f32>();
                self.launch_batch_kernel(
                    &d_prices,
                    &d_periods,
                    len,
                    n_this,
                    first_valid,
                    &mut d_out,
                    launched,
                    launched * len,
                    dyn_smem_bytes,
                )?;
                launched += n_this;
            }
            if !self.debug_batch_logged && env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
                eprintln!(
                    "[cci] batch kernel: Plain, block_x=auto, chunked={} rows",
                    rows
                );
            }
            Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            })
        }
    }

    fn prepare_many_series(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<(Vec<i32>, usize), CudaCciError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCciError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaCciError::InvalidInput(
                "time-major buffer shape mismatch".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaCciError::InvalidInput("invalid period".into()));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                if !data_tm_f32[r * cols + s].is_nan() {
                    fv = Some(r);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaCciError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < period {
                return Err(CudaCciError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail={})",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCciError> {
        let mut func: Function = self
            .module
            .get_function("cci_many_series_one_param_f32")
            .map_err(|_| CudaCciError::MissingKernelSymbol { name: "cci_many_series_one_param_f32" })?;

        let block_x: u32 = match env::var("CCI_MS_BLOCK_X").ok().as_deref() {
            Some("auto") => {
                let (_mg, s) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
                s
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_mg, s) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
                s
            }
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // Validate against device limits
        let dev = Device::get_device(self.device_id).map_err(CudaCciError::Cuda)?;
        let max_grid_x = dev.get_attribute(DeviceAttribute::MaxGridDimX).map_err(CudaCciError::Cuda)? as u32;
        let max_threads = dev.get_attribute(DeviceAttribute::MaxThreadsPerBlock).map_err(CudaCciError::Cuda)? as u32;
        if grid_x == 0 || grid_x > max_grid_x || block_x == 0 || block_x > max_threads {
            return Err(CudaCciError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    pub fn cci_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaCciError> {
        let (first_valids, period) = Self::prepare_many_series(data_tm_f32, cols, rows, period)?;
        // Memory check
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCciError::InvalidInput("cols*rows overflow".into()))?;
        let total = elems
            .checked_mul(sz_f32)
            .and_then(|x| x.checked_add(cols.checked_mul(sz_i32).unwrap_or(0)))
            .and_then(|x| x.checked_add(elems.checked_mul(sz_f32).unwrap_or(0)))
            .ok_or_else(|| CudaCciError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(total, 64 * 1024 * 1024)?;
        if Self::use_async() {
            let h_prices = LockedBuffer::from_slice(data_tm_f32).map_err(CudaCciError::Cuda)?;
            let h_first = LockedBuffer::from_slice(&first_valids).map_err(CudaCciError::Cuda)?;
            let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }.map_err(CudaCciError::Cuda)?;
            let mut d_first = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }.map_err(CudaCciError::Cuda)?;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }.map_err(CudaCciError::Cuda)?;
            let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }.map_err(CudaCciError::Cuda)?;
            let mut d_first = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }.map_err(CudaCciError::Cuda)?;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }.map_err(CudaCciError::Cuda)?;
            unsafe {
                d_prices.async_copy_from(&h_prices, &self.stream).map_err(CudaCciError::Cuda)?;
                d_first.async_copy_from(&h_first, &self.stream).map_err(CudaCciError::Cuda)?;
            }
            self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
            self.stream.synchronize().map_err(CudaCciError::Cuda)?;
            Ok(DeviceArrayF32 { buf: d_out, rows, cols })
        } else {
            let d_prices = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaCciError::Cuda)?;
            let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaCciError::Cuda)?;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }.map_err(CudaCciError::Cuda)?;
            self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
            Ok(DeviceArrayF32 { buf: d_out, rows, cols })
        }
    }
}

// ---------- Benches (wrapper-owned) ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 200; // keep runtime reasonable (O(period) MAD per step)

    fn bytes_required() -> usize {
        let in_bytes = SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct CciBatchState {
        cuda: CudaCci,
        data: Vec<f32>,
        sweep: CciBatchRange,
    }
    impl CudaBenchState for CciBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .cci_batch_dev(&self.data, &self.sweep)
                .expect("cci batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCci::new(0).expect("cuda cci");
        let data = gen_series(SERIES_LEN);
        let sweep = CciBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let sweep = CciBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(CciBatchState { cuda, data, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "cci",
            "one_series_many_params",
            "cci_cuda_batch_dev",
            "1m_x_200",
            prep_one_series_many_params,
        )
        .with_sample_size(8)
        .with_mem_required(bytes_required())]
    }
}
