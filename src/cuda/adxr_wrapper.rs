//! CUDA wrapper for ADXR (Average Directional Index Rating)
//!
//! - API mirrors ALMA/CWMA wrappers (DeviceArrayF32 output, NON_BLOCKING stream)
//! - Two entry points:
//!   - Batch: one series × many params (`adxr_batch_dev`)
//!   - Many-series × one param in time-major layout (`adxr_many_series_one_param_time_major_dev`)
//! - Warmup/NaN semantics identical to scalar: values before `first + 2*period` are NaN.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::adxr::{AdxrBatchRange, AdxrParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaAdxrError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaAdxr {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaAdxr {
    pub fn new(device_id: usize) -> Result<Self, CudaAdxrError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/adxr_kernel.ptx"));
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
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self { module, stream, _context: context, device_id: device_id as u32 })
    }

    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> { self._context.clone() }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaAdxrError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn will_fit(bytes_needed: usize, headroom: usize) -> Result<(), CudaAdxrError> {
        if let Ok((free, _total)) = mem_get_info() {
            let adj_free = free.saturating_sub(headroom);
            if bytes_needed <= adj_free { Ok(()) } else { Err(CudaAdxrError::OutOfMemory { required: bytes_needed, free, headroom }) }
        } else { Ok(()) }
    }

    #[inline]
    fn round_up(x: usize, align: usize) -> usize {
        (x + align - 1) / align * align
    }

    fn expand_periods(sweep: &AdxrBatchRange) -> Vec<AdxrParams> {
        let (start, end, step) = sweep.period;
        let ps: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else if start < end {
            (start..=end).step_by(step).collect()
        } else {
            // reversed bounds
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if let Some(next) = cur.checked_sub(step) { cur = next; } else { break; }
                if cur < end { break; }
            }
            v
        };
        ps.into_iter().map(|p| AdxrParams { period: Some(p) }).collect()
    }

    fn find_first_valid_close(close: &[f32]) -> Option<usize> {
        for (i, &v) in close.iter().enumerate() {
            if v == v {
                // not NaN
                return Some(i);
            }
        }
        None
    }

    /// Batch: one series × many params. Returns a VRAM-backed row-major matrix (rows=n_combos).
    pub fn adxr_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &AdxrBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<AdxrParams>), CudaAdxrError> {
        let n = close_f32.len();
        if n == 0 || high_f32.len() != n || low_f32.len() != n {
            return Err(CudaAdxrError::InvalidInput("input slices are empty or mismatched".into()));
        }
        let combos = Self::expand_periods(sweep);
        if combos.is_empty() { return Err(CudaAdxrError::InvalidInput("no period combinations".into())); }

        let first = Self::find_first_valid_close(close_f32).ok_or_else(|| CudaAdxrError::InvalidInput("all values are NaN".into()))?;
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if n - first < max_p + 1 { return Err(CudaAdxrError::InvalidInput(format!("not enough valid data: needed >= {}, have {}", max_p + 1, n - first))); }

        // Memory estimate with headroom
        let headroom = 64 * 1024 * 1024;
        let out_elems = n
            .checked_mul(combos.len())
            .ok_or_else(|| CudaAdxrError::InvalidInput("rows*cols overflow".into()))?;
        let bytes = (high_f32.len() + low_f32.len() + close_f32.len())
            .checked_mul(4)
            .and_then(|b| b.checked_add(combos.len().saturating_mul(core::mem::size_of::<i32>())))
            .and_then(|b| b.checked_add(out_elems.saturating_mul(4)))
            .ok_or_else(|| CudaAdxrError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(bytes, headroom)?;

        // Periods -> i32, async copy
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let n_combos = periods_i32.len();

        // Heuristic: prefer optimized kernel on large problems; otherwise use legacy batch kernel
        const MIN_COMBOS_FOR_OPT: usize = 64;
        const MIN_SERIES_LEN_FOR_OPT: usize = 100_000;
        let use_opt = n_combos >= MIN_COMBOS_FOR_OPT || n >= MIN_SERIES_LEN_FOR_OPT;

        if use_opt {
            // Global ring workspace sizing (n_combos × padded period)
            let ring_pitch = Self::round_up(max_p, 32);
            let ring_elems = ring_pitch
                .checked_mul(n_combos)
                .ok_or_else(|| CudaAdxrError::InvalidInput("ring workspace overflow".into()))?;

            // Memory fit check includes inputs + periods + ring + outputs
            let headroom = 64 * 1024 * 1024;
            let out_elems2 = n_combos
                .checked_mul(n)
                .ok_or_else(|| CudaAdxrError::InvalidInput("rows*cols overflow".into()))?;
            let bytes = (high_f32.len() + low_f32.len() + close_f32.len())
                .checked_mul(4)
                .and_then(|b| b.checked_add(periods_i32.len().saturating_mul(4)))
                .and_then(|b| b.checked_add(ring_elems.saturating_mul(4)))
                .and_then(|b| b.checked_add(out_elems2.saturating_mul(4)))
                .ok_or_else(|| CudaAdxrError::InvalidInput("byte size overflow".into()))?;
            Self::will_fit(bytes, headroom)?;

            // Upload inputs (async)
            let d_high  = unsafe { DeviceBuffer::from_slice_async(high_f32,  &self.stream) }?;
            let d_low   = unsafe { DeviceBuffer::from_slice_async(low_f32,   &self.stream) }?;
            let d_close = unsafe { DeviceBuffer::from_slice_async(close_f32, &self.stream) }?;
            let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }?;

            let mut d_ring: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(ring_elems, &self.stream) }
                    .map_err(|e| CudaAdxrError::Cuda(e))?;
            let mut d_out: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(out_elems2, &self.stream) }
                    .map_err(|e| CudaAdxrError::Cuda(e))?;

            // Optimized kernel with shared tiling
            let mut func = self
                .module
                .get_function("adxr_one_series_many_params_f32_opt")
                .map_err(|_| CudaAdxrError::MissingKernelSymbol { name: "adxr_one_series_many_params_f32_opt" })?;

            // Hint: prefer shared memory over L1 for this kernel
            func.set_cache_config(CacheConfig::PreferShared)?;

            // Dynamic shared memory for two f32 tiles of length 256
            let shmem_bytes: usize = 2 * 256 * core::mem::size_of::<f32>();

            // Occupancy-aware block size for given shmem
            let (_min_grid, suggested) = func
                .suggested_launch_configuration(shmem_bytes, BlockSize::xyz(0, 0, 0))
                .unwrap_or((0, 128));
            let mut block_x = if suggested > 0 { suggested } else { 128 } as u32;
            if block_x > 256 { block_x = 256; }

            // 1D launch: one thread per combo (period)
            let blocks_x = ((n_combos as u32 + block_x - 1) / block_x).max(1);
            let grid: GridSize = (blocks_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, (shmem_bytes as u32), stream>>>(
                        d_high.as_device_ptr(),
                        d_low.as_device_ptr(),
                        d_close.as_device_ptr(),
                        d_periods.as_device_ptr(),
                        n as i32,
                        first as i32,
                        n_combos as i32,
                        d_ring.as_device_ptr(),
                        ring_pitch as i32,
                        d_out.as_device_ptr()
                    )
                )?;
            }

            self.stream.synchronize()?;

            Ok((DeviceArrayF32 { buf: d_out, rows: n_combos, cols: n }, combos))
        } else {
            // Legacy kernel path (no shared tiles, no global ring required)
            let headroom = 64 * 1024 * 1024;
            let out_elems3 = n_combos
                .checked_mul(n)
                .ok_or_else(|| CudaAdxrError::InvalidInput("rows*cols overflow".into()))?;
            let bytes = (high_f32.len() + low_f32.len() + close_f32.len())
                .checked_mul(4)
                .and_then(|b| b.checked_add(periods_i32.len().saturating_mul(4)))
                .and_then(|b| b.checked_add(out_elems3.saturating_mul(4)))
                .ok_or_else(|| CudaAdxrError::InvalidInput("byte size overflow".into()))?;
            Self::will_fit(bytes, headroom)?;

            let d_high = unsafe { DeviceBuffer::from_slice_async(high_f32, &self.stream) }?;
            let d_low = unsafe { DeviceBuffer::from_slice_async(low_f32, &self.stream) }?;
            let d_close = unsafe { DeviceBuffer::from_slice_async(close_f32, &self.stream) }?;
            let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
            let mut d_out: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(out_elems3, &self.stream) }
                    .map_err(|e| CudaAdxrError::Cuda(e))?;

            let func = self
                .module
                .get_function("adxr_batch_f32")
                .map_err(|_| CudaAdxrError::MissingKernelSymbol { name: "adxr_batch_f32" })?;

            // Ask occupancy for a suggested block size; fall back to 128
            let (_, suggested) = func
                .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                .unwrap_or((0, 0));
            let block_x = if suggested > 0 { suggested } else { 128 } as u32;

            // Chunk over grid.y so grid.y <= 65_535
            let max_grid_y = 65_535usize;
            let mut launched = 0usize;
            while launched < n_combos {
                let chunk = (n_combos - launched).min(max_grid_y);
                let grid: GridSize = (1u32, chunk as u32, 1u32).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let stream = &self.stream;
                unsafe {
                    // Offset periods and out by launched
                    let d_periods_off = d_periods.as_device_ptr().add(launched);
                    let d_out_off = d_out.as_device_ptr().add(launched * n);
                    launch!(
                        func<<<grid, block, 0, stream>>>(
                            d_high.as_device_ptr(),
                            d_low.as_device_ptr(),
                            d_close.as_device_ptr(),
                            d_periods_off,
                            n as i32,
                            first as i32,
                            chunk as i32,
                            d_out_off
                        )
                    )?;
                }
                launched += chunk;
            }

            self.stream.synchronize()?;

            Ok((DeviceArrayF32 { buf: d_out, rows: n_combos, cols: n }, combos))
        }
    }

    /// Many-series × one param (time-major). Returns VRAM-backed matrix with shape rows×cols.
    pub fn adxr_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaAdxrError> {
        if cols == 0 || rows == 0 { return Err(CudaAdxrError::InvalidInput("empty matrix".into())); }
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAdxrError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm_f32.len() != n || low_tm_f32.len() != n || close_tm_f32.len() != n {
            return Err(CudaAdxrError::InvalidInput(
                "matrix inputs must have identical length".into(),
            ));
        }
        if period == 0 || period > rows { return Err(CudaAdxrError::InvalidInput("invalid period".into())); }

        // First-valid per series uses close only (match scalar semantics)
        let mut first_valids: Vec<i32> = vec![0; cols];
        for s in 0..cols {
            let mut fv = -1;
            for t in 0..rows {
                let v = close_tm_f32[t * cols + s];
                if v == v {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        let headroom = 64 * 1024 * 1024;
        let bytes = (high_tm_f32.len() + low_tm_f32.len() + close_tm_f32.len())
            .checked_mul(4)
            .and_then(|b| b.checked_add(first_valids.len().saturating_mul(4)))
            .and_then(|b| b.checked_add(n.saturating_mul(4)))
            .ok_or_else(|| CudaAdxrError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(bytes, headroom)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm_f32, &self.stream) }?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }
                .map_err(|e| CudaAdxrError::Cuda(e))?;

        // Try optimized time-major kernel first; fall back to legacy signature if missing
        if let Ok(func_opt) = self.module.get_function("adxr_many_series_one_param_time_major_f32_opt") {
            let ring_pitch = Self::round_up(period, 32);
            let mut d_ring: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(cols * ring_pitch, &self.stream) }
                    .map_err(|e| CudaAdxrError::Cuda(e))?;

            let grid: GridSize = (cols as u32, 1u32, 1u32).into();
            let block: BlockSize = (1u32, 1u32, 1u32).into();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func_opt<<<grid, block, 0, stream>>>(
                        d_high.as_device_ptr(),
                        d_low.as_device_ptr(),
                        d_close.as_device_ptr(),
                        d_first.as_device_ptr(),
                        period as i32,
                        cols as i32,
                        rows as i32,
                        d_ring.as_device_ptr(),
                        ring_pitch as i32,
                        d_out.as_device_ptr()
                    )
                )
                .map_err(|e| CudaAdxrError::Cuda(e))?;
            }
        } else {
            let func = self
                .module
                .get_function("adxr_many_series_one_param_f32")
                .map_err(|e| CudaAdxrError::Cuda(e))?;
            let grid: GridSize = (cols as u32, 1u32, 1u32).into();
            let block: BlockSize = (1u32, 1u32, 1u32).into();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_high.as_device_ptr(),
                        d_low.as_device_ptr(),
                        d_close.as_device_ptr(),
                        d_first.as_device_ptr(),
                        period as i32,
                        cols as i32,
                        rows as i32,
                        d_out.as_device_ptr()
                    )
                )
                .map_err(|e| CudaAdxrError::Cuda(e))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaAdxrError::Cuda(e))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    /// Host-copy helper returning a contiguous f32 vector (row-major)
    pub fn adxr_batch_into_host_f32(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &AdxrBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<AdxrParams>), CudaAdxrError> {
        let (arr, combos) = self.adxr_batch_dev(high_f32, low_f32, close_f32, sweep)?;
        if out.len() != arr.len() { return Err(CudaAdxrError::InvalidInput("out length mismatch".into())); }
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(arr.len())
                .map_err(|e| CudaAdxrError::Cuda(e))?
        };
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaAdxrError::Cuda(e))?;
        }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    struct AdxrBatchBench {
        cuda: CudaAdxr,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: AdxrBatchRange,
    }
    impl CudaBenchState for AdxrBatchBench {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .adxr_batch_dev(&self.high, &self.low, &self.close, &self.sweep);
        }
    }

    struct AdxrManySeriesBench {
        cuda: CudaAdxr,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for AdxrManySeriesBench {
        fn launch(&mut self) {
            let _ = self.cuda.adxr_many_series_one_param_time_major_dev(
                &self.high_tm,
                &self.low_tm,
                &self.close_tm,
                self.cols,
                self.rows,
                self.period,
            );
        }
    }

    fn make_series(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Generate a plausible OHLC path (noisy sine) in f32
        let mut h = vec![f32::NAN; n];
        let mut l = vec![f32::NAN; n];
        let mut c = vec![f32::NAN; n];
        for i in 1..n {
            let x = i as f32 * 0.00123;
            let base = x.sin() + 0.0003 * (i as f32);
            let hi = base + 0.5 + 0.05 * (x * 3.0).cos();
            let lo = base - 0.5 - 0.04 * (x * 1.7).sin();
            h[i] = hi;
            l[i] = lo;
            c[i] = (hi + lo) * 0.5;
        }
        (h, l, c)
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let (h, l, c) = make_series(50_000);
        let sweep = AdxrBatchRange { period: (5, 60, 5) };
        let cuda = CudaAdxr::new(0).expect("cuda");
        Box::new(AdxrBatchBench {
            cuda,
            high: h,
            low: l,
            close: c,
            sweep,
        })
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let cols = 128usize;
        let rows = 8192usize;
        let (mut h, mut l, mut c) = make_series(cols * rows);
        for s in 0..cols {
            h[s] = f32::NAN;
            l[s] = f32::NAN;
            c[s] = f32::NAN;
        }
        let cuda = CudaAdxr::new(0).expect("cuda");
        Box::new(AdxrManySeriesBench {
            cuda,
            high_tm: h,
            low_tm: l,
            close_tm: c,
            cols,
            rows,
            period: 14,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let bytes_batch = (50_000usize * 3 + (1 + (60 - 5) / 5) * 50_000) * 4;
        let bytes_many = 128usize * 8192usize * 4usize * 4usize; // 3 inputs + out + firsts
        vec![
            CudaBenchScenario::new("adxr", "one_series", "adxr_cuda_batch", "50k", prep_batch)
                .with_mem_required(bytes_batch),
            CudaBenchScenario::new("adxr", "many_series", "adxr_cuda_ms1p", "128x8k", prep_many)
                .with_mem_required(bytes_many),
        ]
    }
}
