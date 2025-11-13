//! CUDA support for the Average Sentiment Oscillator (ASO).
//!
//! Parity goals with ALMA wrapper:
//! - Batch and many-series entry points
//! - JIT options (DetermineTargetFromContext, OptLevel O2) and NON_BLOCKING stream
//! - Light kernel policy enums (kept simple for now; Auto → plain launches)
//! - VRAM estimation with a small safety headroom and graceful errors
//! - Warmup/NaN semantics identical to scalar (prefix NaNs of length first_valid+period-1)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::aso::{AsoBatchRange, AsoParams};
use crate::indicators::willr::build_willr_gpu_tables; // reuse min/max sparse tables
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
// Low-level CUDA driver syscalls for per-kernel attributes
use cust::sys;

#[derive(Debug)]
pub enum CudaAsoError {
    Cuda(CudaError),
    InvalidInput(String),
    MissingKernelSymbol { name: &'static str },
    OutOfMemory { required: usize, free: usize, headroom: usize },
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    InvalidPolicy(&'static str),
    DeviceMismatch { buf: u32, current: u32 },
    NotImplemented,
}

impl fmt::Display for CudaAsoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAsoError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAsoError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaAsoError::MissingKernelSymbol { name } => write!(f, "Missing kernel symbol: {}", name),
            CudaAsoError::OutOfMemory { required, free, headroom } => write!(f, "Out of memory on device: required={}B, free={}B, headroom={}B", required, free, headroom),
            CudaAsoError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz } => write!(f, "Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))"),
            CudaAsoError::InvalidPolicy(p) => write!(f, "Invalid policy: {}", p),
            CudaAsoError::DeviceMismatch { buf, current } => write!(f, "Device mismatch for buffer (buf device={} current={})", buf, current),
            CudaAsoError::NotImplemented => write!(f, "Not implemented"),
        }
    }
}
impl std::error::Error for CudaAsoError {}

// Kernel policy enums (kept for API parity; currently only Auto/Plain used)
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

#[derive(Debug)]
pub struct CudaAso {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    batch_policy: BatchKernelPolicy,
    many_policy: ManySeriesKernelPolicy,
}

impl CudaAso {
    pub fn new(device_id: usize) -> Result<Self, CudaAsoError> {
        cust::init(CudaFlags::empty()).map_err(CudaAsoError::Cuda)?;
        let device = Device::get_device(device_id as u32).map_err(CudaAsoError::Cuda)?;
        let context = Arc::new(Context::new(device).map_err(CudaAsoError::Cuda)?);
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/aso_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(CudaAsoError::Cuda)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(CudaAsoError::Cuda)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            batch_policy: BatchKernelPolicy::Auto,
            many_policy: ManySeriesKernelPolicy::Auto,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaAsoError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaAsoError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn validate_launch(&self, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Result<(), CudaAsoError> {
        let dev = Device::get_device(self.device_id).map_err(CudaAsoError::Cuda)?;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX).map_err(CudaAsoError::Cuda)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY).map_err(CudaAsoError::Cuda)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ).map_err(CudaAsoError::Cuda)? as u32;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX).map_err(CudaAsoError::Cuda)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY).map_err(CudaAsoError::Cuda)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ).map_err(CudaAsoError::Cuda)? as u32;
        let (gx, gy, gz) = grid; let (bx, by, bz) = block;
        if bx > max_bx || by > max_by || bz > max_bz || gx > max_gx || gy > max_gy || gz > max_gz {
            return Err(CudaAsoError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    // ---- Batch: one-series × many-params ----
    pub fn aso_batch_dev(
        &self,
        open_f32: &[f32],
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &AsoBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaAsoError> {
        let (combos, first_valid, series_len, max_period) =
            prepare_batch_inputs(open_f32, high_f32, low_f32, close_f32, sweep)?;
        let n_combos = combos.len();
        let periods: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let modes: Vec<i32> = combos.iter().map(|p| p.mode.unwrap() as i32).collect();

        let tables = build_willr_gpu_tables(high_f32, low_f32);

        // VRAM estimate (inputs + params + tables + outputs)
        let in_bytes = 4usize
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaAsoError::InvalidInput("size overflow".into()))?;
        let param_bytes = 2usize
            .checked_mul(n_combos)
            .and_then(|x| x.checked_mul(std::mem::size_of::<i32>()))
            .ok_or(CudaAsoError::InvalidInput("size overflow".into()))?;
        let table_bytes = tables.st_max.len()
            .checked_mul(4).and_then(|a| a.checked_add(tables.st_min.len().checked_mul(4).unwrap_or(usize::MAX)))
            .and_then(|a| a.checked_add(tables.level_offsets.len().checked_mul(4).unwrap_or(usize::MAX)))
            .and_then(|a| a.checked_add(tables.log2.len().checked_mul(4).unwrap_or(usize::MAX)))
            .ok_or(CudaAsoError::InvalidInput("size overflow".into()))?;
        let out_bytes = 2usize
            .checked_mul(n_combos)
            .and_then(|x| x.checked_mul(series_len))
            .and_then(|x| x.checked_mul(4))
            .ok_or(CudaAsoError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(param_bytes)
            .and_then(|a| a.checked_add(table_bytes))
            .and_then(|a| a.checked_add(out_bytes))
            .ok_or(CudaAsoError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        // Upload
        let d_open = DeviceBuffer::from_slice(open_f32).map_err(CudaAsoError::Cuda)?;
        let d_high = DeviceBuffer::from_slice(high_f32).map_err(CudaAsoError::Cuda)?;
        let d_low = DeviceBuffer::from_slice(low_f32).map_err(CudaAsoError::Cuda)?;
        let d_close = DeviceBuffer::from_slice(close_f32).map_err(CudaAsoError::Cuda)?;
        let d_periods = DeviceBuffer::from_slice(&periods).map_err(CudaAsoError::Cuda)?;
        let d_modes = DeviceBuffer::from_slice(&modes).map_err(CudaAsoError::Cuda)?;
        let d_log2 = DeviceBuffer::from_slice(&tables.log2).map_err(CudaAsoError::Cuda)?;
        let d_offsets = DeviceBuffer::from_slice(&tables.level_offsets).map_err(CudaAsoError::Cuda)?;
        let d_st_max = DeviceBuffer::from_slice(&tables.st_max).map_err(CudaAsoError::Cuda)?;
        let d_st_min = DeviceBuffer::from_slice(&tables.st_min).map_err(CudaAsoError::Cuda)?;
        // Stream-pooled async allocations; ordered before the kernel in this stream
        let mut d_bulls: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream)
        }
        .map_err(CudaAsoError::Cuda)?;
        let mut d_bears: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream)
        }
        .map_err(CudaAsoError::Cuda)?;

        self.launch_batch_kernel(
            &d_open,
            &d_high,
            &d_low,
            &d_close,
            &d_periods,
            &d_modes,
            &d_log2,
            &d_offsets,
            &d_st_max,
            &d_st_min,
            series_len,
            first_valid,
            combos.len(),
            max_period,
            &mut d_bulls,
            &mut d_bears,
        )?;
        // Ensure producing stream is synchronized before handing out device buffers
        self.stream.synchronize().map_err(CudaAsoError::Cuda)?;
        Ok((
            DeviceArrayF32 { buf: d_bulls, rows: n_combos, cols: series_len },
            DeviceArrayF32 { buf: d_bears, rows: n_combos, cols: series_len },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_modes: &DeviceBuffer<i32>,
        d_log2: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        d_st_max: &DeviceBuffer<f32>,
        d_st_min: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        max_period: usize,
        d_bulls: &mut DeviceBuffer<f32>,
        d_bears: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAsoError> {
        if n_combos == 0 || series_len == 0 { return Ok(()); }
        let mut func = self.module.get_function("aso_batch_f32")
            .map_err(|_| CudaAsoError::MissingKernelSymbol { name: "aso_batch_f32" })?;
        let block_x = match self.batch_policy {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let smem_bytes = (2 * max_period * std::mem::size_of::<f32>()) as u32;
        // Opt-in to larger dynamic shared memory and set cache preference heuristics
        set_kernel_smem_prefs(&mut func, smem_bytes)?;

        self.validate_launch((n_combos as u32, 1, 1), (block_x, 1, 1))?;
        unsafe {
            let mut open_ptr = d_open.as_device_ptr().as_raw();
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut modes_ptr = d_modes.as_device_ptr().as_raw();
            let mut log2_ptr = d_log2.as_device_ptr().as_raw();
            let mut offs_ptr = d_offsets.as_device_ptr().as_raw();
            let mut stmax_ptr = d_st_max.as_device_ptr().as_raw();
            let mut stmin_ptr = d_st_min.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut level_count_i = d_offsets.len() as i32 - 1;
            let mut n_combos_i = n_combos as i32;
            let mut bulls_ptr = d_bulls.as_device_ptr().as_raw();
            let mut bears_ptr = d_bears.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut open_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut modes_ptr as *mut _ as *mut c_void,
                &mut log2_ptr as *mut _ as *mut c_void,
                &mut offs_ptr as *mut _ as *mut c_void,
                &mut stmax_ptr as *mut _ as *mut c_void,
                &mut stmin_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut level_count_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut bulls_ptr as *mut _ as *mut c_void,
                &mut bears_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, smem_bytes, args).map_err(CudaAsoError::Cuda)?;
        }
        Ok(())
    }

    // ---- Many series × one param (time-major) ----
    pub fn aso_many_series_one_param_time_major_dev(
        &self,
        open_tm_f32: &[f32],
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        mode: usize,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaAsoError> {
        if cols == 0 || rows == 0 || period == 0 {
            return Err(CudaAsoError::InvalidInput("invalid shape or period".into()));
        }
        let expected = cols.checked_mul(rows).ok_or(CudaAsoError::InvalidInput("size overflow".into()))?;
        if open_tm_f32.len() != expected
            || high_tm_f32.len() != expected
            || low_tm_f32.len() != expected
            || close_tm_f32.len() != expected
        {
            return Err(CudaAsoError::InvalidInput("mismatched input sizes".into()));
        }
        if mode > 2 {
            return Err(CudaAsoError::InvalidInput("invalid mode".into()));
        }

        // first_valid per series
        let mut first_valids: Vec<i32> = vec![0; cols];
        for s in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let idx = t * cols + s;
                if !close_tm_f32[idx].is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv as usize >= rows {
                return Err(CudaAsoError::InvalidInput(
                    "all values NaN in a series".into(),
                ));
            }
            if rows - (fv as usize) < period {
                return Err(CudaAsoError::InvalidInput(
                    "not enough valid data in a series".into(),
                ));
            }
            first_valids[s] = fv;
        }

        let d_open = DeviceBuffer::from_slice(open_tm_f32).map_err(CudaAsoError::Cuda)?;
        let d_high = DeviceBuffer::from_slice(high_tm_f32).map_err(CudaAsoError::Cuda)?;
        let d_low = DeviceBuffer::from_slice(low_tm_f32).map_err(CudaAsoError::Cuda)?;
        let d_close = DeviceBuffer::from_slice(close_tm_f32).map_err(CudaAsoError::Cuda)?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaAsoError::Cuda)?;
        let mut d_bulls: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(expected, &self.stream)
        }
        .map_err(CudaAsoError::Cuda)?;
        let mut d_bears: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(expected, &self.stream)
        }
        .map_err(CudaAsoError::Cuda)?;

        self.launch_many_series_kernel(
            &d_open,
            &d_high,
            &d_low,
            &d_close,
            &d_first,
            cols,
            rows,
            period,
            mode,
            &mut d_bulls,
            &mut d_bears,
        )?;
        self.stream.synchronize().map_err(CudaAsoError::Cuda)?;
        Ok((
            DeviceArrayF32 { buf: d_bulls, rows, cols },
            DeviceArrayF32 { buf: d_bears, rows, cols },
        ))
    }

    fn launch_many_series_kernel(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_first: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        mode: usize,
        d_bulls: &mut DeviceBuffer<f32>,
        d_bears: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAsoError> {
        if cols == 0 || rows == 0 { return Ok(()); }
        let mut func = self.module.get_function("aso_many_series_one_param_f32")
            .map_err(|_| CudaAsoError::MissingKernelSymbol { name: "aso_many_series_one_param_f32" })?;
        let block_x = match self.many_policy {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // shared: ring_b/e (2*period*f32) + dq_min_idx/dq_max_idx (2*period*i32)
        let smem_bytes = (2 * period * std::mem::size_of::<f32>()
            + 2 * period * std::mem::size_of::<i32>()) as u32;
        set_kernel_smem_prefs(&mut func, smem_bytes)?;
        self.validate_launch((cols as u32, 1, 1), (block_x, 1, 1))?;
        unsafe {
            let mut open_ptr = d_open.as_device_ptr().as_raw();
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut mode_i = mode as i32;
            let mut out_b_ptr = d_bulls.as_device_ptr().as_raw();
            let mut out_e_ptr = d_bears.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut open_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut mode_i as *mut _ as *mut c_void,
                &mut out_b_ptr as *mut _ as *mut c_void,
                &mut out_e_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, smem_bytes, args).map_err(CudaAsoError::Cuda)?;
        }
        Ok(())
    }
}

// ---- Helpers ----
// Best-effort: opt-in to larger dynamic shared memory per block and hint cache config
#[inline(always)]
fn set_kernel_smem_prefs(func: &mut Function, smem_bytes: u32) -> Result<(), CudaAsoError> {
    unsafe {
        let raw = func.to_raw();
        // Request up to smem_bytes of dynamic shared memory per block
        let _ = sys::cuFuncSetAttribute(
            raw,
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes as i32,
        );
        if smem_bytes > 48 * 1024 {
            // Prefer shared-memory carve-out when using lots of dynamic smem
            let _ = sys::cuFuncSetAttribute(
                raw,
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            );
            let _ = func.set_cache_config(CacheConfig::PreferShared);
        } else {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }
    }
    Ok(())
}

fn expand_params(range: &AsoBatchRange) -> Result<Vec<AsoParams>, CudaAsoError> {
    fn axis((s, e, st): (usize, usize, usize)) -> Result<Vec<usize>, CudaAsoError> {
        if st == 0 || s == e { return Ok(vec![s]); }
        let mut v = Vec::new();
        if s < e {
            let mut cur = s;
            while cur <= e { v.push(cur); let next = cur.saturating_add(st); if next == cur { break; } cur = next; }
        } else {
            let mut cur = s;
            while cur >= e { v.push(cur); let next = cur.saturating_sub(st); if next == cur { break; } cur = next; if cur == 0 && e > 0 { break; } }
        }
        if v.is_empty() { return Err(CudaAsoError::InvalidInput("empty usize range".into())); }
        Ok(v)
    }
    let ps = axis(range.period)?;
    let ms = axis(range.mode)?;
    let mut v = Vec::with_capacity(ps.len().saturating_mul(ms.len()));
    for &p in &ps { for &m in &ms { v.push(AsoParams { period: Some(p), mode: Some(m) }); } }
    Ok(v)
}

fn prepare_batch_inputs(
    open: &[f32],
    high: &[f32],
    low: &[f32],
    close: &[f32],
    sweep: &AsoBatchRange,
) -> Result<(Vec<AsoParams>, usize, usize, usize), CudaAsoError> {
    let len = close.len();
    if len == 0 || high.len() != len || low.len() != len || open.len() != len {
        return Err(CudaAsoError::InvalidInput(
            "empty or mismatched inputs".into(),
        ));
    }
    let combos = expand_params(sweep)?;
    let first_valid = (0..len)
        .find(|&i| !close[i].is_nan())
        .ok_or_else(|| CudaAsoError::InvalidInput("all values are NaN".into()))?;
    let max_period = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first_valid < max_period {
        return Err(CudaAsoError::InvalidInput("not enough valid data".into()));
    }
    Ok((combos, first_valid, len, max_period))
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const N: usize = 1_000_000;
    const PARAMS: usize = 200;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 4 * N * 4; // open/high/low/close
        let out_bytes = 2 * N * PARAMS * 4;
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct AsoBatchState {
        cuda: CudaAso,
        o: Vec<f32>,
        h: Vec<f32>,
        l: Vec<f32>,
        c: Vec<f32>,
        sweep: AsoBatchRange,
    }
    impl CudaBenchState for AsoBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .aso_batch_dev(&self.o, &self.h, &self.l, &self.c, &self.sweep)
                .unwrap();
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaAso::new(0).expect("cuda aso");
        let c = gen_series(N);
        let mut o = c.clone();
        let mut h = c.clone();
        let mut l = c.clone();
        for i in 0..N {
            let v = c[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0031;
            let off = (0.0019 * x.sin()).abs() + 0.05;
            o[i] = v - 0.1;
            h[i] = v + off;
            l[i] = v - off;
        }
        let sweep = AsoBatchRange {
            period: (10, 10 + PARAMS - 1, 1),
            mode: (0, 2, 1),
        };
        Box::new(AsoBatchState { cuda, o, h, l, c, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "aso",
            "one_series_many_params",
            "aso_cuda_batch_dev",
            "1m_x_200",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
