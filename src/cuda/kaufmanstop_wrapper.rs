//! CUDA support for the Kaufmanstop indicator (adaptive stop based on MA of range).
//!
//! Parity goals (mirrors ALMA wrapper patterns):
//! - PTX load via DetermineTargetFromContext + OptLevel O2 with fallback
//! - NON_BLOCKING stream
//! - VRAM checks and (lightweight) chunking where applicable
//! - Public entry points return VRAM-backed arrays (DeviceArrayF32)
//! - Warmup/NaN semantics identical to scalar implementation

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::cuda::moving_averages::ma_selector::{CudaMaData, CudaMaSelector, CudaMaSelectorError};
use crate::indicators::kaufmanstop::{
    expand_grid_wrapper, KaufmanstopBatchRange, KaufmanstopParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaKaufmanstopError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("not implemented")]
    NotImplemented,
    #[error("out of memory: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
}

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
pub struct CudaKaufmanstopPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaKaufmanstopPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

pub struct CudaKaufmanstop {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    policy: CudaKaufmanstopPolicy,
    device_id: u32,
}

impl CudaKaufmanstop {
    pub fn new(device_id: usize) -> Result<Self, CudaKaufmanstopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kaufmanstop_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) =
                    Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
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
            policy: CudaKaufmanstopPolicy::default(),
            device_id: device_id as u32,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaKaufmanstopPolicy,
    ) -> Result<Self, CudaKaufmanstopError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
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

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaKaufmanstopError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaKaufmanstopError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn validate_launch_dims(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaKaufmanstopError> {
        use cust::device::DeviceAttribute;
        let dev = Device::get_device(self.device_id)?;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaKaufmanstopError::InvalidInput(
                "zero-sized grid or block".into(),
            ));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz
        {
            return Err(CudaKaufmanstopError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        Ok(())
    }

    // -------------- Batch: one series × many params --------------
    pub fn kaufmanstop_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &KaufmanstopBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<KaufmanstopParams>), CudaKaufmanstopError> {
        // Validate inputs
        if high.is_empty() || low.is_empty() || high.len() != low.len() {
            return Err(CudaKaufmanstopError::InvalidInput(
                "high/low must be same non-zero length".into(),
            ));
        }
        let len = high.len();

        // First valid index (both high & low finite) — matches scalar
        let first = high
            .iter()
            .zip(low.iter())
            .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
            .ok_or_else(|| CudaKaufmanstopError::InvalidInput("all values are NaN".into()))?;

        // Expand parameter sweep
        let combos = expand_grid_wrapper(sweep)
            .map_err(|e| CudaKaufmanstopError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaKaufmanstopError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // Validate periods
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            if p == 0 || p > len {
                return Err(CudaKaufmanstopError::InvalidInput("invalid period".into()));
            }
            if len - first < p {
                return Err(CudaKaufmanstopError::InvalidInput(
                    "not enough valid data after first valid".into(),
                ));
            }
        }

        // VRAM check: base + full output must fit (MA slab is tiled)
        let head = Self::headroom_bytes();
        let base_bytes = 2usize
            .checked_mul(len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                CudaKaufmanstopError::InvalidInput("base_bytes overflow".into())
            })?;
        let out_bytes = combos
            .len()
            .checked_mul(len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                CudaKaufmanstopError::InvalidInput("out_bytes overflow".into())
            })?;
        let required = base_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| {
                CudaKaufmanstopError::InvalidInput("total bytes overflow".into())
            })?;
        Self::will_fit(required, head)?;

        // Upload base series once
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * len) }?;

        // Build range (high - low) with NaN propagation on host
        let mut range = vec![f32::NAN; len];
        for i in first..len {
            let (h, l) = (high[i], low[i]);
            range[i] = if h.is_nan() || l.is_nan() { f32::NAN } else { h - l };
        }

        // Main kernel (one series → many params)
        let mut ks_many_params_fn: Function = self
            .module
            .get_function("kaufmanstop_one_series_many_params_time_major_f32")
            .map_err(|_| CudaKaufmanstopError::MissingKernelSymbol {
                name: "kaufmanstop_one_series_many_params_time_major_f32",
            })?;

        // Robust default block geometry
        let bx_default: u32 = match ks_many_params_fn
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
        {
            Ok((_min_grid, suggested)) => suggested.max(128).min(512),
            Err(_) => 256,
        };
        let prefer_by = |tile_params: usize| -> u32 {
            if tile_params >= 8 {
                8
            } else if tile_params >= 4 {
                4
            } else {
                1
            }
        };

        // Determine a safe param tile size so the MA slab fits with headroom
        let per_param_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKaufmanstopError::InvalidInput("per_param_bytes overflow".into()))?;
        let free = mem_get_info().ok().map(|(free, _)| free).unwrap_or(usize::MAX);
        let mut max_tile_params = if free > head + base_bytes + out_bytes {
            ((free - head - base_bytes - out_bytes) / per_param_bytes).max(1)
        } else {
            1
        };
        max_tile_params = max_tile_params.min(combos.len()).max(1);

        // Selector for MA(range)
        let selector = CudaMaSelector::new(self.device_id as usize);

        // Process params in contiguous tiles, additionally splitting by direction to keep
        // base_is_low uniform per launch (kernel expects a single base selector).
        let mut p0 = 0usize;
        while p0 < combos.len() {
            // Establish direction for this sub-tile
            let dir_long = combos[p0]
                .direction
                .as_deref()
                .unwrap_or("long")
                .eq_ignore_ascii_case("long");

            // Extend tile with same direction (contiguous segment)
            let mut p1 = p0;
            let mut taken = 0usize;
            while p1 < combos.len() && taken < max_tile_params {
                let same_dir = combos[p1]
                    .direction
                    .as_deref()
                    .unwrap_or("long")
                    .eq_ignore_ascii_case(if dir_long { "long" } else { "short" });
                if !same_dir {
                    break;
                }
                p1 += 1;
                taken += 1;
            }
            let tile = &combos[p0..p1];
            let tile_n = tile.len();

            // Prepare warm indices and multipliers for this tile
            let mut warm_ps = Vec::<i32>::with_capacity(tile_n);
            let mut signed_mults = Vec::<f32>::with_capacity(tile_n);
            for prm in tile.iter() {
                let period = prm.period.unwrap();
                let mult = prm.mult.unwrap() as f32;
                warm_ps.push((first + period - 1) as i32);
                signed_mults.push(if dir_long { -mult } else { mult });
            }
            let d_warm = DeviceBuffer::from_slice(&warm_ps)?;
            let d_signed = DeviceBuffer::from_slice(&signed_mults)?;

            // Allocate MA slab for the tile: [tile_n * len], time-major per param
            let mut d_ma_tile = unsafe { DeviceBuffer::<f32>::uninitialized(tile_n * len) }?;

            // Stage MA rows into a single pinned host slab, then upload once.
            let mut h_ma_tile =
                unsafe { LockedBuffer::<f32>::uninitialized(tile_n * len) }?;
            let mut keep_alive: Vec<DeviceArrayF32> = Vec::with_capacity(tile_n);
            for (j, prm) in tile.iter().enumerate() {
                let period = prm.period.unwrap();
                let ma_type = prm.ma_type.as_deref().unwrap_or("sma");
                let ma_dev = selector
                    .ma_to_device(ma_type, CudaMaData::SliceF32(&range), period)
                    .map_err(|e| CudaKaufmanstopError::InvalidInput(format!("ma_to_device: {}", e)))?;
                debug_assert_eq!(ma_dev.rows, 1);
                debug_assert_eq!(ma_dev.cols, len);

                // Destination host slice for param j
                let offset = j * len;
                let slice = &mut h_ma_tile.as_mut_slice()[offset..offset + len];
                unsafe {
                    ma_dev.buf.async_copy_to(slice, &self.stream)?;
                }
                keep_alive.push(ma_dev);
            }
            // Ensure D2H fills complete before uploading to device, then free sources
            self.stream.synchronize()?;
            drop(keep_alive);
            unsafe {
                d_ma_tile.async_copy_from(&h_ma_tile, &self.stream)?;
            }

            // Launch the tile kernel
            let bx = bx_default;
            let by = prefer_by(tile_n);
            let grid_x = ((len as u32) + bx - 1) / bx;
            let grid_y = ((tile_n as u32) + by - 1) / by;
            let grid = GridSize::xyz(grid_x.max(1), grid_y.max(1), 1);
            let block = BlockSize::xyz(bx, by, 1);
            self.validate_launch_dims((grid_x, grid_y, 1), (bx, by, 1))?;
            let shmem = (bx as usize) * std::mem::size_of::<f32>();

            let out_offset = p0 * len;
            let launch_res = unsafe {
                let mut hp = d_high.as_device_ptr().as_raw();
                let mut lp = d_low.as_device_ptr().as_raw();
                let mut mp = d_ma_tile.as_device_ptr().as_raw();
                let mut wp = d_warm.as_device_ptr().as_raw();
                let mut sp = d_signed.as_device_ptr().as_raw();
                let mut rows_i = len as i32;
                let mut params_i = tile_n as i32;
                let mut bil = if dir_long { 1i32 } else { 0i32 };
                let mut out_ptr = d_out.as_device_ptr().add(out_offset).as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut hp as *mut _ as *mut c_void,
                    &mut lp as *mut _ as *mut c_void,
                    &mut mp as *mut _ as *mut c_void,
                    &mut wp as *mut _ as *mut c_void,
                    &mut sp as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut params_i as *mut _ as *mut c_void,
                    &mut bil as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream.launch(&ks_many_params_fn, grid, block, shmem as u32, args)
            };

            if let Err(_e) = launch_res {
                // Fallback: per-parameter AXPY row kernel (preserves correctness)
                let mut axpy_fn: Function = self
                    .module
                    .get_function("kaufmanstop_axpy_row_f32")
                    .map_err(|_| CudaKaufmanstopError::MissingKernelSymbol {
                        name: "kaufmanstop_axpy_row_f32",
                    })?;
                let block_x: u32 = match self.policy.batch {
                    BatchKernelPolicy::Auto => {
                        let (_min_grid, suggested) = axpy_fn
                            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                            ?;
                        suggested.max(128).min(512)
                    }
                    BatchKernelPolicy::Plain { block_x } => block_x.max(64).min(1024),
                };
                let grid_x = ((len as u32) + block_x - 1) / block_x;
                let grid_base: GridSize = (grid_x.max(1), 1, 1).into();
                let block_base: BlockSize = (block_x, 1, 1).into();

                for (j, prm) in tile.iter().enumerate() {
                    let period = prm.period.unwrap();
                    let mult = prm.mult.unwrap() as f32;
                    let is_long = prm
                        .direction
                        .as_deref()
                        .unwrap_or("long")
                        .eq_ignore_ascii_case("long");
                    let signed_mult = if is_long { -mult } else { mult };
                    let base_is_low = if is_long { 1i32 } else { 0i32 };
                    let warm = (first + period - 1) as i32;
                    let ma_type = prm.ma_type.as_deref().unwrap_or("sma");

                    let ma_dev = selector
                        .ma_to_device(ma_type, CudaMaData::SliceF32(&range), period)
                        .map_err(|e| {
                            CudaKaufmanstopError::InvalidInput(format!("ma_to_device: {}", e))
                        })?;

                    unsafe {
                        let mut hp = d_high.as_device_ptr().as_raw();
                        let mut lp = d_low.as_device_ptr().as_raw();
                        let mut mp = ma_dev.buf.as_device_ptr().as_raw();
                        let mut n = len as i32;
                        let mut sm = signed_mult;
                        let mut w = warm;
                        let mut bil = base_is_low;
                        let mut out_ptr = d_out
                            .as_device_ptr()
                            .add((p0 + j) * len)
                            .as_raw();

                        let args: &mut [*mut c_void] = &mut [
                            &mut hp as *mut _ as *mut c_void,
                            &mut lp as *mut _ as *mut c_void,
                            &mut mp as *mut _ as *mut c_void,
                            &mut n as *mut _ as *mut c_void,
                            &mut sm as *mut _ as *mut c_void,
                            &mut w as *mut _ as *mut c_void,
                            &mut bil as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&axpy_fn, grid_base, block_base, 0, args)?;
                    }
                    // Keep conservative ordering to avoid lifetime hazards
                    self.stream.synchronize()?;
                }
            }

            p0 = p1;
        }

        // Ensure all tiles are complete
        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    // -------------- Many-series: time-major, one param --------------
    pub fn kaufmanstop_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &KaufmanstopParams,
    ) -> Result<DeviceArrayF32, CudaKaufmanstopError> {
        if cols == 0 || rows == 0 || high_tm.len() != cols * rows || low_tm.len() != cols * rows {
            return Err(CudaKaufmanstopError::InvalidInput(
                "invalid dims for time-major inputs".into(),
            ));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 || period > rows {
            return Err(CudaKaufmanstopError::InvalidInput("invalid period".into()));
        }
        let mult = params.mult.unwrap_or(2.0) as f32;
        let is_long = params
            .direction
            .as_deref()
            .unwrap_or("long")
            .eq_ignore_ascii_case("long");
        let signed_mult = if is_long { -mult } else { mult };
        let base_is_low = if is_long { 1i32 } else { 0i32 };
        let ma_type = params.ma_type.as_deref().unwrap_or("sma");

        // Per-series first valid (both high & low must be non-NaN at t)
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaKaufmanstopError::InvalidInput(format!("series {} all NaN", s))
            })?;
            if rows - fv < period {
                return Err(CudaKaufmanstopError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }

        // Build range (high - low) time-major with NaN propagation
        let mut range_tm = vec![f32::NAN; cols * rows];
        for idx in 0..(cols * rows) {
            let h = high_tm[idx];
            let l = low_tm[idx];
            range_tm[idx] = if h.is_nan() || l.is_nan() {
                f32::NAN
            } else {
                h - l
            };
        }

        // Compute MA(range) on device. For now, use the selector in single-parameter
        // time-major mode by flattening; only a subset of MAs expose time-major kernels
        // through their wrappers. To keep behavior broad, compute per-series via selector
        // but write directly to the final output with our AXPY kernel.
        // However, to reduce kernel launches, if ma_type == "sma", use the SMA wrapper's
        // native many-series path.
        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaKaufmanstopError::InvalidInput("rows*cols overflow".into()))?;

        // VRAM check: 2×inputs + first_valids + outputs + MA slab (approximate)
        let head = Self::headroom_bytes();
        let bytes_series = total
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKaufmanstopError::InvalidInput("bytes overflow".into()))?;
        let bytes_first = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaKaufmanstopError::InvalidInput("first_valid bytes overflow".into()))?;
        let required = bytes_series
            .checked_mul(3)
            .and_then(|x| x.checked_add(bytes_first))
            .ok_or_else(|| CudaKaufmanstopError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, head)?;

        let mut d_high = DeviceBuffer::from_slice(high_tm)?;
        let mut d_low = DeviceBuffer::from_slice(low_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(total) }?;

        // Try native SMA many-series path; otherwise, fall back to per-series selector
        let ma_tm_dev: DeviceBuffer<f32> = if ma_type.eq_ignore_ascii_case("sma") {
            use crate::cuda::moving_averages::sma_wrapper::CudaSma;
            use crate::indicators::moving_averages::sma::SmaParams as SParams;
            let sma = CudaSma::new(self.device_id as usize)?;
            let sparams = SParams {
                period: Some(period),
            };
            let ma_dev = sma
                .sma_multi_series_one_param_time_major_dev(&range_tm, cols, rows, &sparams)
                .map_err(|e| {
                    CudaKaufmanstopError::InvalidInput(format!(
                        "sma_multi_series_one_param_time_major_dev: {}",
                        e
                    ))
                })?;
            ma_dev.buf
        } else {
            // Generic (slower) fallback: compute each series via selector and copy into a device buffer
            let selector = CudaMaSelector::new(self.device_id as usize);
            let mut d_ma = unsafe { DeviceBuffer::<f32>::uninitialized(total) }?;
            // Prepare a host pinned buffer for staging each row
            for s in 0..cols {
                // Gather one series contiguous
                let mut series = vec![f32::NAN; rows];
                for t in 0..rows {
                    series[t] = range_tm[t * cols + s];
                }
                let ma_dev = selector
                    .ma_to_device(ma_type, CudaMaData::SliceF32(&series), period)
                    .map_err(|e| {
                        CudaKaufmanstopError::InvalidInput(format!("ma_to_device: {}", e))
                    })?;
                debug_assert_eq!(ma_dev.rows, 1);
                debug_assert_eq!(ma_dev.cols, rows);
                // Copy into device buffer at column s (time-major)
                // We need to scatter to strided positions; use a small kernel-free host copy via staging
                // by pulling to pinned host and then writing into the strided layout.
                let mut pinned = unsafe { LockedBuffer::<f32>::uninitialized(rows) }?;
                unsafe {
                    ma_dev
                        .buf
                        .async_copy_to(&mut pinned.as_mut_slice(), &self.stream)?;
                }
                self.stream.synchronize()?;
                // Scatter into d_ma via host-side strided copy (one series at a time)
                // Note: For large matrices, a dedicated scatter kernel would be better.
                let mut host_scatter = vec![0f32; rows];
                host_scatter.copy_from_slice(pinned.as_slice());
                // Write into device at strided positions by building a temp time-major column
                // and copying as a contiguous slice per series by transposing logic.
                // Here we just copy to host and then to device slice with the same layout.
                for t in 0..rows {
                    let idx = t * cols + s;
                    // SAFETY: writing via staging host buffer; final upload after loop
                    // We'll accumulate into a full host buffer and upload once at the end.
                    range_tm[idx] = host_scatter[t];
                }
            }
            // Upload the filled time-major MA matrix
            d_ma.copy_from(&range_tm)?;
            d_ma
        };

        // Launch many-series AXPY kernel
        let mut func: Function = self
            .module
            .get_function("kaufmanstop_many_series_one_param_time_major_f32")
            .map_err(|_| CudaKaufmanstopError::MissingKernelSymbol {
                name: "kaufmanstop_many_series_one_param_time_major_f32",
            })?;
        // 2‑D grid: x=series tiles, y=time tiles (avoids per‑element div/mod)
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    ?;
                suggested.max(128).min(512)
            }
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64).min(1024),
        };
        let block_y: u32 = 1; // keep y simple; kernel strides by gridDim.y over time
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let time_tile: u32 = 256;
        let grid_y = ((rows as u32) + time_tile - 1) / time_tile;
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (block_x, block_y, 1).into();

        unsafe {
            let mut hp = d_high.as_device_ptr().as_raw();
            let mut lp = d_low.as_device_ptr().as_raw();
            let mut mp = ma_tm_dev.as_device_ptr().as_raw();
            let mut fp = d_first.as_device_ptr().as_raw();
            let mut c = cols as i32;
            let mut r = rows as i32;
            let mut sm = signed_mult;
            let mut bil = base_is_low;
            let mut p = period as i32;
            let mut op = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp as *mut _ as *mut c_void,
                &mut lp as *mut _ as *mut c_void,
                &mut mp as *mut _ as *mut c_void,
                &mut fp as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut sm as *mut _ as *mut c_void,
                &mut bil as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
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
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct KaufmanstopBatchState {
        cuda: CudaKaufmanstop,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: KaufmanstopBatchRange,
    }
    impl CudaBenchState for KaufmanstopBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .kaufmanstop_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("kaufmanstop batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKaufmanstop::new(0).expect("cuda kaufmanstop");
        let p = gen_series(ONE_SERIES_LEN);
        let mut high = vec![0f32; ONE_SERIES_LEN];
        let mut low = vec![0f32; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN {
            let r = 0.5f32 + ((i as f32) * 0.00037).cos().abs();
            high[i] = p[i] + 0.5 * r;
            low[i] = p[i] - 0.5 * r;
        }
        let sweep = KaufmanstopBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            mult: (2.0, 2.0, 0.0),
            direction: ("long".to_string(), "long".to_string(), 0.0),
            ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
        };
        Box::new(KaufmanstopBatchState {
            cuda,
            high,
            low,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "kaufmanstop",
            "one_series_many_params",
            "kaufmanstop_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
