//! CUDA wrapper for the Range Filter indicator.
//!
//! Parity goals (mirrors ALMA/CWMA wrappers):
//! - PTX load via DetermineTargetFromContext + OptLevel O2 with fallback
//! - NON_BLOCKING stream
//! - Policy enums (simple 1D launch for batch; 1D blocks per series for many-series)
//! - VRAM checks + grid.y chunking (<= 65_535) for batch
//! - Warmup/NaN semantics identical to scalar: warm = first_valid + max(range_period, smooth? smooth_period : 0)
//! - Compensated FP32 accumulators in kernels (no FP64); FP32 I/O

#![cfg(feature = "cuda")]

use crate::indicators::range_filter::{RangeFilterBatchRange, RangeFilterParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

#[derive(Debug)]
pub enum CudaRangeFilterError {
    Cuda(CudaError),
    InvalidInput(String),
    MissingKernelSymbol { name: &'static str },
    OutOfMemory { required: usize, free: usize, headroom: usize },
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    InvalidPolicy(&'static str),
    DeviceMismatch { buf: u32, current: u32 },
    NotImplemented,
}

impl fmt::Display for CudaRangeFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaRangeFilterError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaRangeFilterError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaRangeFilterError::MissingKernelSymbol { name } => {
                write!(f, "Missing kernel symbol: {}", name)
            }
            CudaRangeFilterError::OutOfMemory { required, free, headroom } => write!(
                f,
                "Out of memory on device: required={}B, free={}B, headroom={}B",
                required, free, headroom
            ),
            CudaRangeFilterError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz } => write!(
                f,
                "Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))"
            ),
            CudaRangeFilterError::InvalidPolicy(p) => write!(f, "Invalid policy: {}", p),
            CudaRangeFilterError::DeviceMismatch { buf, current } => write!(
                f,
                "Device mismatch: buffer device={} current device={}",
                buf, current
            ),
            CudaRangeFilterError::NotImplemented => write!(f, "Not implemented"),
        }
    }
}
impl std::error::Error for CudaRangeFilterError {}

/// Three VRAM-backed arrays for (filter, high, low)
pub struct DeviceRangeFilterTrio {
    pub filter: DeviceBuffer<f32>,
    pub high: DeviceBuffer<f32>,
    pub low: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}

impl DeviceRangeFilterTrio {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaRangeFilterPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaRangeFilter {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaRangeFilterPolicy,
    debug_logged: AtomicBool,
}

impl CudaRangeFilter {
    pub fn new(device_id: usize) -> Result<Self, CudaRangeFilterError> {
        cust::init(CudaFlags::empty()).map_err(CudaRangeFilterError::Cuda)?;
        let device = Device::get_device(device_id as u32).map_err(CudaRangeFilterError::Cuda)?;
        let context = Arc::new(Context::new(device).map_err(CudaRangeFilterError::Cuda)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/range_filter_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(CudaRangeFilterError::Cuda)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).map_err(CudaRangeFilterError::Cuda)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaRangeFilterPolicy::default(),
            debug_logged: AtomicBool::new(false),
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn set_policy(&mut self, p: CudaRangeFilterPolicy) {
        self.policy = p;
    }
    pub fn synchronize(&self) -> Result<(), CudaRangeFilterError> {
        self.stream.synchronize().map_err(CudaRangeFilterError::Cuda)?;
        Ok(())
    }

    #[inline]
    fn headroom_bytes() -> usize {
        env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024)
    }
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

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaRangeFilterError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaRangeFilterError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    // ---------------- Batch: one series × many params ----------------

    #[inline]
    fn pick_1d_launch_for_batch(
        &self,
        n: usize,
    ) -> Result<(GridSize, BlockSize), CudaRangeFilterError> {
        let dev = Device::get_device(self.device_id).map_err(CudaRangeFilterError::Cuda)?;
        let max_threads_per_block = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .map_err(CudaRangeFilterError::Cuda)? as u32;
        let sm_count = dev
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map_err(CudaRangeFilterError::Cuda)? as u32;

        let auto_block_x = 256u32.min(max_threads_per_block);
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(1).min(max_threads_per_block),
            BatchKernelPolicy::Auto => auto_block_x,
        };

        let n_u32: u32 = n
            .try_into()
            .map_err(|_| CudaRangeFilterError::InvalidInput("n exceeds u32".into()))?;
        let blocks_needed = (n_u32 + block_x - 1) / block_x;
        let blocks_cap = sm_count.saturating_mul(32);
        let grid_x = blocks_needed.min(blocks_cap).max(1);

        self.validate_launch((grid_x, 1, 1), (block_x, 1, 1))?;

        Ok(((grid_x, 1, 1).into(), (block_x, 1, 1).into()))
    }

    #[inline]
    fn validate_launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaRangeFilterError> {
        let dev = Device::get_device(self.device_id).map_err(CudaRangeFilterError::Cuda)?;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .map_err(CudaRangeFilterError::Cuda)? as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaRangeFilterError::Cuda)? as u32;
        if block.0 == 0 || block.0 > max_bx || grid.0 == 0 || grid.0 > max_gx {
            return Err(CudaRangeFilterError::LaunchConfigTooLarge {
                gx: grid.0,
                gy: grid.1,
                gz: grid.2,
                bx: block.0,
                by: block.1,
                bz: block.2,
            });
        }
        Ok(())
    }
    pub fn range_filter_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &RangeFilterBatchRange,
    ) -> Result<(DeviceRangeFilterTrio, Vec<RangeFilterParams>), CudaRangeFilterError> {
        if data_f32.is_empty() {
            return Err(CudaRangeFilterError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaRangeFilterError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // Validate periods and warmup coverage like scalar
        let max_needed = combos
            .iter()
            .map(|c| {
                let rp = c.range_period.unwrap_or(14);
                let sp = if c.smooth_range.unwrap_or(true) {
                    c.smooth_period.unwrap_or(27)
                } else {
                    0
                };
                rp.max(sp)
            })
            .max()
            .unwrap_or(0);
        let valid = len - first_valid;
        if valid < max_needed {
            return Err(CudaRangeFilterError::InvalidInput(format!(
                "not enough valid data: needed = {}, valid = {}",
                max_needed, valid
            )));
        }
        for p in &combos {
            let rs = p.range_size.unwrap_or(2.618);
            if !rs.is_finite() || rs <= 0.0 {
                return Err(CudaRangeFilterError::InvalidInput(
                    "invalid range_size".into(),
                ));
            }
            let rp = p.range_period.unwrap_or(14);
            if rp == 0 || rp > len {
                return Err(CudaRangeFilterError::InvalidInput(
                    "invalid range_period".into(),
                ));
            }
            let sr = p.smooth_range.unwrap_or(true);
            let sp = p.smooth_period.unwrap_or(27);
            if sr && (sp == 0 || sp > len) {
                return Err(CudaRangeFilterError::InvalidInput(
                    "invalid smooth_period".into(),
                ));
            }
        }

        // VRAM estimate
        let rows = combos.len();
        let elem_size = std::mem::size_of::<f32>();
        let in_bytes = len
            .checked_mul(elem_size)
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        let params_row_bytes = elem_size
            .checked_add(3usize.checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| {
                CudaRangeFilterError::InvalidInput("size overflow".into())
            })?)
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        let params_bytes = rows
            .checked_mul(params_row_bytes)
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        let out_elems_per_row = len
            .checked_mul(3)
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        let out_bytes = rows
            .checked_mul(out_elems_per_row)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::headroom_bytes())?;

        let d_prices = DeviceBuffer::from_slice(data_f32).map_err(CudaRangeFilterError::Cuda)?;
        let range_sizes_f32: Vec<f32> = combos
            .iter()
            .map(|c| c.range_size.unwrap_or(2.618) as f32)
            .collect();
        let range_periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| {
                let rp = c.range_period.unwrap_or(14);
                rp.try_into().map_err(|_| {
                    CudaRangeFilterError::InvalidInput("range_period exceeds i32".into())
                })
            })
            .collect::<Result<_, _>>()?;
        let smooth_flags_i32: Vec<i32> = combos
            .iter()
            .map(|c| if c.smooth_range.unwrap_or(true) { 1 } else { 0 })
            .collect();
        let smooth_periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| {
                let sp = c.smooth_period.unwrap_or(27);
                sp.try_into().map_err(|_| {
                    CudaRangeFilterError::InvalidInput("smooth_period exceeds i32".into())
                })
            })
            .collect::<Result<_, _>>()?;
        let d_rs = DeviceBuffer::from_slice(&range_sizes_f32).map_err(CudaRangeFilterError::Cuda)?;
        let d_rp = DeviceBuffer::from_slice(&range_periods_i32).map_err(CudaRangeFilterError::Cuda)?;
        let d_sf = DeviceBuffer::from_slice(&smooth_flags_i32).map_err(CudaRangeFilterError::Cuda)?;
        let d_sp = DeviceBuffer::from_slice(&smooth_periods_i32).map_err(CudaRangeFilterError::Cuda)?;

        let elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("rows*len overflow".into()))?;
        let mut d_f: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaRangeFilterError::Cuda)?;
        let mut d_h: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaRangeFilterError::Cuda)?;
        let mut d_l: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaRangeFilterError::Cuda)?;

        // --------------- Single 1D launch (grid-stride kernel) ---------------
        let func = self
            .module
            .get_function("range_filter_batch_f32")
            .map_err(|_| CudaRangeFilterError::MissingKernelSymbol {
                name: "range_filter_batch_f32",
            })?;

        let (grid, block) = self.pick_1d_launch_for_batch(rows)?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut rs_ptr = d_rs.as_device_ptr().as_raw();
            let mut rp_ptr = d_rp.as_device_ptr().as_raw();
            let mut sf_ptr = d_sf.as_device_ptr().as_raw();
            let mut sp_ptr = d_sp.as_device_ptr().as_raw();
            let mut len_i: i32 = len
                .try_into()
                .map_err(|_| CudaRangeFilterError::InvalidInput("length exceeds i32".into()))?;
            let mut nrows_i: i32 = rows
                .try_into()
                .map_err(|_| CudaRangeFilterError::InvalidInput("rows exceeds i32".into()))?;
            let mut first_i: i32 = first_valid
                .try_into()
                .map_err(|_| CudaRangeFilterError::InvalidInput("first_valid exceeds i32".into()))?;
            let mut f_ptr = d_f.as_device_ptr().as_raw();
            let mut h_ptr = d_h.as_device_ptr().as_raw();
            let mut l_ptr = d_l.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut rs_ptr as *mut _ as *mut c_void,
                &mut rp_ptr as *mut _ as *mut c_void,
                &mut sf_ptr as *mut _ as *mut c_void,
                &mut sp_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut nrows_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaRangeFilterError::Cuda)?;
        }

        self.stream
            .synchronize()
            .map_err(CudaRangeFilterError::Cuda)?;

        Ok((
            DeviceRangeFilterTrio {
                filter: d_f,
                high: d_h,
                low: d_l,
                rows,
                cols: len,
                ctx: self.context.clone(),
                device_id: self.device_id,
            },
            combos,
        ))
    }

    // ------------- Many series × one param (time‑major) -------------
    pub fn range_filter_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &RangeFilterParams,
    ) -> Result<DeviceRangeFilterTrio, CudaRangeFilterError> {
        if rows == 0 || cols == 0 {
            return Err(CudaRangeFilterError::InvalidInput("empty dims".into()));
        }
        let elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != elems {
            return Err(CudaRangeFilterError::InvalidInput(
                "time-major input must be rows*cols".into(),
            ));
        }
        let rs_f64 = params.range_size.unwrap_or(2.618);
        if !rs_f64.is_finite() || rs_f64 <= 0.0 {
            return Err(CudaRangeFilterError::InvalidInput(
                "invalid range_size".into(),
            ));
        }
        let rs = rs_f64 as f32;
        let rp_usize = params.range_period.unwrap_or(14);
        let rp: i32 = rp_usize
            .try_into()
            .map_err(|_| CudaRangeFilterError::InvalidInput("range_period exceeds i32".into()))?;
        let sr = params.smooth_range.unwrap_or(true);
        let sp_usize = params.smooth_period.unwrap_or(27);
        let sp: i32 = sp_usize
            .try_into()
            .map_err(|_| CudaRangeFilterError::InvalidInput("smooth_period exceeds i32".into()))?;
        if rp <= 0 || (sr && sp <= 0) {
            return Err(CudaRangeFilterError::InvalidInput("invalid period".into()));
        }

        // first_valid per series (column) scanning time-major layout
        if rows > i32::MAX as usize || cols > i32::MAX as usize {
            return Err(CudaRangeFilterError::InvalidInput(
                "rows or cols exceeds i32".into(),
            ));
        }
        let mut first_valids = vec![cols as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if data_tm_f32[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let elem_size = std::mem::size_of::<f32>();
        let in_bytes = elems
            .checked_mul(elem_size)
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(3)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        let aux_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(out_bytes)
            .and_then(|v| v.checked_add(aux_bytes))
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::headroom_bytes())?;

        let d_data = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaRangeFilterError::Cuda)?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaRangeFilterError::Cuda)?;
        let mut d_f: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaRangeFilterError::Cuda)?;
        let mut d_h: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaRangeFilterError::Cuda)?;
        let mut d_l: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaRangeFilterError::Cuda)?;

        let func = self
            .module
            .get_function("range_filter_many_series_one_param_f32")
            .map_err(|_| CudaRangeFilterError::MissingKernelSymbol {
                name: "range_filter_many_series_one_param_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
            _ => 1,
        };
        let grid: GridSize = (cols as u32, 1u32, 1u32).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch((cols as u32, 1, 1), (block_x, 1, 1))?;

        unsafe {
            let mut data_ptr = d_data.as_device_ptr().as_raw();
            let mut rs_f = rs;
            let mut rp_i = rp;
            let mut sf_i = if sr { 1i32 } else { 0i32 };
            let mut sp_i = sp;
            let mut cols_i: i32 = cols
                .try_into()
                .map_err(|_| CudaRangeFilterError::InvalidInput("cols exceeds i32".into()))?; // num_series
            let mut rows_i: i32 = rows
                .try_into()
                .map_err(|_| CudaRangeFilterError::InvalidInput("rows exceeds i32".into()))?; // series_len
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut f_ptr = d_f.as_device_ptr().as_raw();
            let mut h_ptr = d_h.as_device_ptr().as_raw();
            let mut l_ptr = d_l.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut rs_f as *mut _ as *mut c_void,
                &mut rp_i as *mut _ as *mut c_void,
                &mut sf_i as *mut _ as *mut c_void,
                &mut sp_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaRangeFilterError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaRangeFilterError::Cuda)?;

        Ok(DeviceRangeFilterTrio {
            filter: d_f,
            high: d_h,
            low: d_l,
            rows,
            cols,
            ctx: self.context.clone(),
            device_id: self.device_id,
        })
    }
}

// ---------------- Helpers ----------------
#[inline]
fn expand_grid(r: &RangeFilterBatchRange) -> Result<Vec<RangeFilterParams>, CudaRangeFilterError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaRangeFilterError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            loop {
                v.push(cur);
                match cur.checked_add(step) {
                    Some(next) if next <= end => {
                        cur = next;
                    }
                    Some(_) => break,
                    None => {
                        return Err(CudaRangeFilterError::InvalidInput(
                            "usize range overflow in range_period sweep".into(),
                        ));
                    }
                }
            }
        } else {
            let mut cur = start;
            loop {
                v.push(cur);
                match cur.checked_sub(step) {
                    Some(next) if next >= end => {
                        cur = next;
                    }
                    Some(_) => break,
                    None => {
                        return Err(CudaRangeFilterError::InvalidInput(
                            "usize range overflow in range_period sweep".into(),
                        ));
                    }
                }
            }
        }

        if v.is_empty() {
            return Err(CudaRangeFilterError::InvalidInput(
                "empty range_period sweep".into(),
            ));
        }

        Ok(v)
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, CudaRangeFilterError> {
        let eps = 1e-12;
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(CudaRangeFilterError::InvalidInput(
                "non-finite range_size sweep bounds".into(),
            ));
        }
        if step.abs() < eps || (start - end).abs() < eps {
            return Ok(vec![start]);
        }

        let mut v = Vec::new();
        let delta = end - start;
        let dir = if delta >= 0.0 { 1.0 } else { -1.0 };
        let mut step_eff = step;
        if step_eff.signum() == 0.0 {
            step_eff = dir * step.abs();
        } else if step_eff.signum() != dir {
            step_eff = dir * step.abs();
        }

        let mut x = start;
        if dir > 0.0 {
            while x <= end + eps {
                v.push(x);
                x += step_eff;
            }
        } else {
            while x >= end - eps {
                v.push(x);
                x += step_eff;
            }
        }

        if v.is_empty() {
            return Err(CudaRangeFilterError::InvalidInput(
                "empty range_size sweep".into(),
            ));
        }

        Ok(v)
    }

    let range_sizes = axis_f64(r.range_size)?;
    let range_periods = axis_usize(r.range_period)?;

    let combos_len = range_sizes
        .len()
        .checked_mul(range_periods.len())
        .ok_or_else(|| CudaRangeFilterError::InvalidInput("combos size overflow".into()))?;

    let mut out = Vec::with_capacity(combos_len);
    for &rs in &range_sizes {
        for &rp in &range_periods {
            out.push(RangeFilterParams {
                range_size: Some(rs),
                range_period: Some(rp),
                smooth_range: r.smooth_range,
                smooth_period: r.smooth_period,
            });
        }
    }
    Ok(out)
}

// ---------------- Bench profiles ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    struct RfBatchState {
        cuda: CudaRangeFilter,
    }
    impl CudaBenchState for RfBatchState {
        fn launch(&mut self) {
            let _ = &self.cuda;
        }
    }

    fn prep_rf_batch() -> Box<dyn CudaBenchState> {
        let len = 120_000usize;
        let mut data = vec![f32::NAN; len];
        
        for i in 5..len {
            let x = i as f32;
            data[i] = (x * 0.0021).sin() + 0.00021 * x;
        }
        let sweep = RangeFilterBatchRange {
            range_size: (2.0, 4.0, 0.2),
            range_period: (8, 64, 8),
            smooth_range: Some(true),
            smooth_period: Some(27),
        };
        let cuda = CudaRangeFilter::new(0).unwrap();
        let _ = cuda.range_filter_batch_dev(&data, &sweep).unwrap();
        Box::new(RfBatchState { cuda })
    }

    struct RfManySeriesState {
        cuda: CudaRangeFilter,
    }
    impl CudaBenchState for RfManySeriesState {
        fn launch(&mut self) {
            let _ = &self.cuda;
        }
    }

    fn prep_rf_many_series() -> Box<dyn CudaBenchState> {
        let cols = 256usize; // series
        let rows = 600_000usize; // time
        let mut tm = vec![f32::NAN; rows * cols];
        for s in 0..cols {
            for t in s..rows {
                let idx = t * cols + s;
                let x = t as f32 + s as f32 * 0.01;
                tm[idx] = (x * 0.0013).sin() + 0.00011 * x;
            }
        }
        let params = RangeFilterParams {
            range_size: Some(2.618),
            range_period: Some(14),
            smooth_range: Some(true),
            smooth_period: Some(27),
        };
        for s in 0..cols {
            for t in s..rows {
                let idx = t * cols + s;
                let x = t as f32 + s as f32 * 0.01;
                tm[idx] = (x * 0.0013).sin() + 0.00011 * x;
            }
        }
        let params = RangeFilterParams {
            range_size: Some(2.618),
            range_period: Some(14),
            smooth_range: Some(true),
            smooth_period: Some(27),
        };
        let cuda = CudaRangeFilter::new(0).unwrap();
        let _ = cuda
            .range_filter_many_series_one_param_time_major_dev(&tm, cols, rows, &params)
            .unwrap();
        let _ = cuda
            .range_filter_many_series_one_param_time_major_dev(&tm, cols, rows, &params)
            .unwrap();
        Box::new(RfManySeriesState { cuda })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "range_filter",
                "one_series_many_params",
                "cuda/range_filter",
                "batch",
                prep_rf_batch,
            )
            .with_sample_size(25),
            CudaBenchScenario::new(
                "range_filter",
                "many_series_one_param",
                "cuda/range_filter",
                "many_series",
                prep_rf_many_series,
            )
            .with_sample_size(15),
            CudaBenchScenario::new(
                "range_filter",
                "one_series_many_params",
                "cuda/range_filter",
                "batch",
                prep_rf_batch,
            )
            .with_sample_size(25),
            CudaBenchScenario::new(
                "range_filter",
                "many_series_one_param",
                "cuda/range_filter",
                "many_series",
                prep_rf_many_series,
            )
            .with_sample_size(15),
        ]
    }
}
