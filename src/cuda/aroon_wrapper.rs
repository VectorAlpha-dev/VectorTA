#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::aroon::{AroonBatchRange, AroonParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cuda_sys;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaAroonError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid range: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
}

pub struct DeviceArrayF32Pair {
    pub first: DeviceArrayF32,  // up
    pub second: DeviceArrayF32, // down
}

impl DeviceArrayF32Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.first.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.first.cols
    }
}

pub struct CudaAroonBatchResult {
    pub outputs: DeviceArrayF32Pair,
    pub combos: Vec<AroonParams>,
}

pub struct CudaAroon {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaAroon {
    pub fn new(device_id: usize) -> Result<Self, CudaAroonError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/aroon_kernel.ptx"));
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
            _context: context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> { self._context.clone() }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"), Err(_) => true }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(bytes_needed: usize, headroom: usize) -> Result<(), CudaAroonError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _total)) = Self::device_mem_info() {
            if bytes_needed.saturating_add(headroom) <= free {
                Ok(())
            } else {
                Err(CudaAroonError::OutOfMemory { required: bytes_needed, free, headroom })
            }
        } else {
            Ok(())
        }
    }

    fn expand_lengths(sweep: &AroonBatchRange) -> Result<Vec<AroonParams>, CudaAroonError> {
        let (start, end, step) = sweep.length;
        let mut lens: Vec<usize> = Vec::new();
        if step == 0 || start == end {
            lens.push(start);
        } else if start < end {
            let mut v = start;
            while v <= end { lens.push(v); match v.checked_add(step) { Some(n) => { if n==v { break; } v = n; }, None => break } }
        } else {
            let mut v = start;
            loop { lens.push(v); if v == end { break; } let n = v.saturating_sub(step); if n==v { break; } v = n; if v < end { break; } }
        }
        if lens.is_empty() {
            return Err(CudaAroonError::InvalidRange { start, end, step });
        }
        Ok(lens.into_iter().map(|l| AroonParams { length: Some(l) }).collect())
    }

    fn find_first_valid_pair(high: &[f32], low: &[f32]) -> Option<usize> {
        for i in 0..high.len() {
            let h = high[i];
            let l = low[i];
            if h == h && l == l && h.is_finite() && l.is_finite() {
                return Some(i);
            }
        }
        None
    }

    /// Batch: one series × many params (row-major outputs)
    pub fn aroon_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &AroonBatchRange,
    ) -> Result<CudaAroonBatchResult, CudaAroonError> {
        let n = high_f32.len();
        if n == 0 || low_f32.len() != n {
            return Err(CudaAroonError::InvalidInput(
                "empty or mismatched inputs".into(),
            ));
        }
        let combos = Self::expand_lengths(sweep)?;
        let first = Self::find_first_valid_pair(high_f32, low_f32)
            .ok_or_else(|| CudaAroonError::InvalidInput("all values are NaN".into()))?;
        let max_len = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
        if n - first < max_len + 1 {
            return Err(CudaAroonError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_len + 1,
                n - first
            )));
        }

        let lengths_i32: Vec<i32> = combos.iter().map(|c| c.length.unwrap() as i32).collect();
        let out_elems = combos
            .len()
            .checked_mul(n)
            .ok_or_else(|| CudaAroonError::InvalidInput("rows*cols overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        let in_bytes = high_f32.len().saturating_mul(4) + low_f32.len().saturating_mul(4);
        let param_bytes = lengths_i32.len().saturating_mul(4);
        let out_bytes = out_elems
            .checked_mul(4)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaAroonError::InvalidInput("byte size overflow".into()))?;
        let bytes = in_bytes
            .checked_add(param_bytes)
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| CudaAroonError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(bytes, headroom)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_f32, &self.stream) }?;
        let d_lengths = unsafe { DeviceBuffer::from_slice_async(&lengths_i32, &self.stream) }?;
        let mut d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;
        let mut d_down: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        let _ = self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_lengths,
            n,
            first,
            max_len,
            combos.len(),
            &mut d_up,
            &mut d_down,
            0,
        )?;

        self.stream.synchronize()?;

        let outputs = DeviceArrayF32Pair {
            first: DeviceArrayF32 {
                buf: d_up,
                rows: combos.len(),
                cols: n,
            },
            second: DeviceArrayF32 {
                buf: d_down,
                rows: combos.len(),
                cols: n,
            },
        };
        Ok(CudaAroonBatchResult { outputs, combos })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        max_len: usize,
        n_combos: usize,
        d_up: &mut DeviceBuffer<f32>,
        d_down: &mut DeviceBuffer<f32>,
        block_x: u32,
    ) -> Result<u32, CudaAroonError> {
        if n_combos == 0 || series_len == 0 {
            return Ok(0);
        }

        let mut func = self
            .module
            .get_function("aroon_batch_f32")
            .map_err(|_| CudaAroonError::MissingKernelSymbol { name: "aroon_batch_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferShared);

        // Dynamic shared memory: two int deques of capacity (max_len+1)
        let shmem_bytes_usize: usize = 2 * (max_len + 1) * std::mem::size_of::<i32>();
        let shmem_bytes: u32 = shmem_bytes_usize as u32;

        let selected_block_x = if block_x > 0 { block_x } else { 1 };

        let max_grid_y = 65_535usize;
        let mut launched = 0usize;
        while launched < n_combos {
            let chunk = (n_combos - launched).min(max_grid_y);
            let grid: GridSize = (1u32, chunk as u32, 1u32).into();
            let block: BlockSize = (selected_block_x, 1, 1).into();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shmem_bytes, stream>>>(
                        d_high.as_device_ptr(),
                        d_low.as_device_ptr(),
                        d_lengths.as_device_ptr().add(launched),
                        series_len as i32,
                        first_valid as i32,
                        chunk as i32,
                        d_up.as_device_ptr().add(launched * series_len),
                        d_down.as_device_ptr().add(launched * series_len)
                    )
                )?;
            }
            launched += chunk;
        }

        Ok(selected_block_x)
    }

    /// Many-series × one-param (time-major). Returns two VRAM-backed matrices.
    pub fn aroon_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        length: usize,
    ) -> Result<DeviceArrayF32Pair, CudaAroonError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAroonError::InvalidInput("empty matrix".into()));
        }
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAroonError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm_f32.len() != n || low_tm_f32.len() != n {
            return Err(CudaAroonError::InvalidInput(
                "matrix inputs mismatch".into(),
            ));
            return Err(CudaAroonError::InvalidInput(
                "matrix inputs mismatch".into(),
            ));
        }
        if length == 0 || length > rows {
            // lookback cannot exceed series length
            return Err(CudaAroonError::InvalidInput("invalid length".into()));
        }

        // first-valid per series (both finite)
        let mut first_valids: Vec<i32> = vec![-1; cols];
        for s in 0..cols {
            for t in 0..rows {
                let h = high_tm_f32[t * cols + s];
                let l = low_tm_f32[t * cols + s];
                if h == h && l == l && h.is_finite() && l.is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let headroom = 64 * 1024 * 1024;
        let in_bytes =
            high_tm_f32.len().saturating_mul(4) + low_tm_f32.len().saturating_mul(4);
        let first_bytes = cols.saturating_mul(4);
        let out_bytes = n
            .checked_mul(4)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaAroonError::InvalidInput("byte size overflow".into()))?;
        let bytes = in_bytes
            .checked_add(first_bytes)
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| CudaAroonError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(bytes, headroom)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }?;
        let mut d_down: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }?;

        let mut func = self
            .module
            .get_function("aroon_many_series_one_param_f32")
            .map_err(|_| CudaAroonError::MissingKernelSymbol { name: "aroon_many_series_one_param_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        // Dynamic shared memory: two int deques of capacity (length+1)
        let shmem_bytes_usize: usize = 2 * (length + 1) * std::mem::size_of::<i32>();
        let (suggested_block, _min_grid) = func
            .suggested_launch_configuration(shmem_bytes_usize, BlockSize::xyz(0, 0, 0))
            .unwrap_or((128, 0));
        let block_x: u32 = if suggested_block > 0 { suggested_block } else { 128 } as u32;
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shmem_bytes: u32 = shmem_bytes_usize as u32;
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, shmem_bytes, stream>>>(
                    d_high.as_device_ptr(),
                    d_low.as_device_ptr(),
                    d_first.as_device_ptr(),
                    length as i32,
                    cols as i32,
                    rows as i32,
                    d_up.as_device_ptr(),
                    d_down.as_device_ptr()
                )
            )?;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Pair {
            first: DeviceArrayF32 {
                buf: d_up,
                rows,
                cols,
            },
            second: DeviceArrayF32 {
                buf: d_down,
                rows,
                cols,
            },
        })
    }

    /// Host-copy helper for batch (returns row-major up/down)
    pub fn aroon_batch_into_host_f32(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &AroonBatchRange,
        out_up: &mut [f32],
        out_down: &mut [f32],
    ) -> Result<(usize, usize, Vec<AroonParams>), CudaAroonError> {
        let CudaAroonBatchResult { outputs, combos } =
            self.aroon_batch_dev(high_f32, low_f32, sweep)?;
        let rows = outputs.rows();
        let cols = outputs.cols();
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAroonError::InvalidInput("rows*cols overflow".into()))?;
        if out_up.len() != expected || out_down.len() != expected {
            return Err(CudaAroonError::InvalidInput(
                "output length mismatch".into(),
            ));
        }
        outputs.first.buf.copy_to(out_up)?;
        outputs.second.buf.copy_to(out_down)?;
        Ok((rows, cols, combos))
    }

    /// Optional: return page-locked host buffers to overlap D->H without extra copies
    pub fn aroon_batch_into_pinned_host_f32(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &AroonBatchRange,
    ) -> Result<(LockedBuffer<f32>, LockedBuffer<f32>, usize, usize, Vec<AroonParams>), CudaAroonError> {
        let CudaAroonBatchResult { outputs, combos } = self.aroon_batch_dev(high_f32, low_f32, sweep)?;
        let rows = outputs.rows();
        let cols = outputs.cols();
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAroonError::InvalidInput("rows*cols overflow".into()))?;
        let mut pinned_up: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(expected)? };
        let mut pinned_dn: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(expected)? };
        unsafe {
            outputs
                .first
                .buf
                .async_copy_to(pinned_up.as_mut_slice(), &self.stream)?;
            outputs
                .second
                .buf
                .async_copy_to(pinned_dn.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        Ok((pinned_up, pinned_dn, rows, cols, combos))
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    fn gen_series(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![f32::NAN; n];
        let mut l = vec![f32::NAN; n];
        for i in 5..n {
            let x = i as f32 * 0.0031;
            let base = x.sin() * 0.7 + 0.0005 * (i as f32);
            let hi = base + 1.0 + 0.03 * (x * 2.0).cos();
            let lo = base - 1.0 - 0.02 * (x * 1.7).sin();
            h[i] = hi;
            l[i] = lo;
        }
        (h, l)
    }

    struct AroonBatchDevBench {
        cuda: CudaAroon,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_lengths: DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        max_len: usize,
        n_combos: usize,
        d_up: DeviceBuffer<f32>,
        d_down: DeviceBuffer<f32>,
    }
    impl CudaBenchState for AroonBatchDevBench {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .launch_batch_kernel(
                    &self.d_high,
                    &self.d_low,
                    &self.d_lengths,
                    self.series_len,
                    self.first_valid,
                    self.max_len,
                    self.n_combos,
                    &mut self.d_up,
                    &mut self.d_down,
                    0,
                )
                .expect("aroon batch kernel");
            self.cuda.stream.synchronize().expect("aroon sync");
        }
    }
    fn prep_batch_dev() -> Box<dyn CudaBenchState> {
        let (h, l) = gen_series(200_000);
        let sweep = AroonBatchRange { length: (10, 500, 1) };
        let cuda = CudaAroon::new(0).expect("cuda aroon");

        let combos = CudaAroon::expand_lengths(&sweep).expect("aroon lengths");
        let first_valid = CudaAroon::find_first_valid_pair(&h, &l).expect("aroon first");
        let max_len = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
        let lengths_i32: Vec<i32> = combos.iter().map(|c| c.length.unwrap() as i32).collect();

        let d_high = DeviceBuffer::from_slice(&h).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&l).expect("d_low");
        let d_lengths = DeviceBuffer::from_slice(&lengths_i32).expect("d_lengths");

        let series_len = h.len();
        let n_combos = combos.len();
        let elems = series_len * n_combos;
        let d_up = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("d_up");
        let d_down = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("d_down");

        Box::new(AroonBatchDevBench {
            cuda,
            d_high,
            d_low,
            d_lengths,
            series_len,
            first_valid,
            max_len,
            n_combos,
            d_up,
            d_down,
        })
    }

    struct AroonManyBench {
        cuda: CudaAroon,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        length: usize,
        block_x: u32,
        shmem_bytes: u32,
        d_up: DeviceBuffer<f32>,
        d_down: DeviceBuffer<f32>,
    }
    impl CudaBenchState for AroonManyBench {
        fn launch(&mut self) {
            let mut func = self
                .cuda
                .module
                .get_function("aroon_many_series_one_param_f32")
                .expect("aroon_many_series_one_param_f32");
            let grid: GridSize = (self.cols as u32, 1, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();
            let stream = &self.cuda.stream;
            unsafe {
                launch!(
                    func<<<grid, block, self.shmem_bytes, stream>>>(
                        self.d_high_tm.as_device_ptr(),
                        self.d_low_tm.as_device_ptr(),
                        self.d_first_valids.as_device_ptr(),
                        self.length as i32,
                        self.cols as i32,
                        self.rows as i32,
                        self.d_up.as_device_ptr(),
                        self.d_down.as_device_ptr()
                    )
                )
                .expect("aroon many-series launch");
            }
            self.cuda.stream.synchronize().expect("aroon sync");
        }
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cols = 256usize;
        let rows = 16_384usize;
        let mut high_tm = vec![f32::NAN; cols * rows];
        let mut low_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in (s % 7)..rows {
                let i = t * cols + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.0007;
                high_tm[i] = x.sin() + 1.1;
                low_tm[i] = x.sin() - 1.1;
            }
        }
        let cuda = CudaAroon::new(0).expect("cuda aroon");
        let mut first_valids: Vec<i32> = vec![-1; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");

        // Match wrapper kernel config selection (once in prep).
        let mut func = cuda
            .module
            .get_function("aroon_many_series_one_param_f32")
            .expect("aroon_many_series_one_param_f32");
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let shmem_bytes_usize: usize = 2 * (25 + 1) * std::mem::size_of::<i32>();
        let (suggested_block, _min_grid) = func
            .suggested_launch_configuration(shmem_bytes_usize, BlockSize::xyz(0, 0, 0))
            .unwrap_or((128, 0));
        let block_x: u32 = (if suggested_block > 0 { suggested_block } else { 128 }) as u32;
        let shmem_bytes: u32 = shmem_bytes_usize as u32;

        let elems = cols * rows;
        let d_up = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("d_up");
        let d_down = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.expect("d_down");
        cuda.stream.synchronize().expect("sync");
        Box::new(AroonManyBench {
            cuda,
            d_high_tm,
            d_low_tm,
            d_first_valids,
            cols,
            rows,
            length: 25,
            block_x,
            shmem_bytes,
            d_up,
            d_down,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let bytes_batch = (200_000usize * 2 + (500 - 10 + 1) * 200_000usize * 2) * 4; // rough
        let bytes_many = 256usize * 16_384usize * 2 * 4 * 3; // 2 inputs + 2 outputs + firsts
        vec![
            CudaBenchScenario::new(
                "aroon",
                "one_series_many_params",
                "aroon_cuda_batch_dev",
                "200k_x_491",
                prep_batch_dev,
            )
            .with_mem_required(bytes_batch),
            CudaBenchScenario::new(
                "aroon",
                "many_series_one_param",
                "aroon_cuda_ms1p",
                "256x16k_L25",
                prep_many,
            )
            .with_mem_required(bytes_many),
        ]
    }
}
