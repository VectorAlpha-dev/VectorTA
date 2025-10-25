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
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaAroonError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAroonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAroonError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAroonError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaAroonError {}

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
    _context: Context,
}

impl CudaAroon {
    pub fn new(device_id: usize) -> Result<Self, CudaAroonError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAroonError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/aroon_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    #[inline]
    fn will_fit(bytes_needed: usize, headroom: usize) -> bool {
        if let Ok((free, _total)) = mem_get_info() {
            let free = free.saturating_sub(headroom);
            (bytes_needed as u64) <= (free as u64)
        } else {
            true
        }
    }

    fn expand_lengths(sweep: &AroonBatchRange) -> Vec<AroonParams> {
        let (start, end, step) = sweep.length;
        let lens: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        };
        lens.into_iter()
            .map(|l| AroonParams { length: Some(l) })
            .collect()
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
        let combos = Self::expand_lengths(sweep);
        if combos.is_empty() {
            return Err(CudaAroonError::InvalidInput(
                "no length combinations".into(),
            ));
        }
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
        let out_elems = combos.len() * n;
        let headroom = 64 * 1024 * 1024;
        let bytes =
            (high_f32.len() + low_f32.len()) * 4 + lengths_i32.len() * 4 + out_elems * 4 * 2;
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaAroonError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_f32, &self.stream) }
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_f32, &self.stream) }
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let d_lengths = unsafe { DeviceBuffer::from_slice_async(&lengths_i32, &self.stream) }
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let mut d_up: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let mut d_down: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;

        // Launch kernel with y-chunking if needed
        let mut func = self
            .module
            .get_function("aroon_batch_f32")
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        // Dynamic shared memory: two int deques of capacity (max_len+1)
        let shmem_bytes_usize: usize = 2 * (max_len + 1) * std::mem::size_of::<i32>();
        // Ask occupancy suggester using our dynamic smem needs
        let (suggested_block, _min_grid) = func
            .suggested_launch_configuration(shmem_bytes_usize, BlockSize::xyz(0, 0, 0))
            .unwrap_or((128, 0));
        let block_x: u32 = if suggested_block > 0 { suggested_block } else { 128 } as u32;
        // Optional: opt-in to larger dynamic shared memory if supported
        // Optional large dynamic shared memory opt-in removed: requires raw CUfunction access
        let shmem_bytes: u32 = shmem_bytes_usize as u32;
        let max_grid_y = 65_535usize;
        let mut launched = 0usize;
        while launched < combos.len() {
            let chunk = (combos.len() - launched).min(max_grid_y);
            let grid: GridSize = (1u32, chunk as u32, 1u32).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shmem_bytes, stream>>>(
                        d_high.as_device_ptr(),
                        d_low.as_device_ptr(),
                        d_lengths.as_device_ptr().add(launched),
                        n as i32,
                        first as i32,
                        chunk as i32,
                        d_up.as_device_ptr().add(launched * n),
                        d_down.as_device_ptr().add(launched * n)
                    )
                )
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
            }
            launched += chunk;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;

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
        let n = cols * rows;
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
        let bytes = (high_tm_f32.len() + low_tm_f32.len()) * 4 + cols * 4 + n * 4 * 2;
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaAroonError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let mut d_up: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        let mut d_down: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;

        let mut func = self
            .module
            .get_function("aroon_many_series_one_param_f32")
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
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
            )
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;

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
        let expected = rows * cols;
        if out_up.len() != expected || out_down.len() != expected {
            return Err(CudaAroonError::InvalidInput(
                "output length mismatch".into(),
            ));
        }
        outputs.first.buf.copy_to(out_up).map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        outputs.second.buf.copy_to(out_down).map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
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
        let expected = rows * cols;
        let mut pinned_up: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(expected)
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?
        };
        let mut pinned_dn: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(expected)
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?
        };
        unsafe {
            outputs
                .first
                .buf
                .async_copy_to(pinned_up.as_mut_slice(), &self.stream)
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
            outputs
                .second
                .buf
                .async_copy_to(pinned_dn.as_mut_slice(), &self.stream)
                .map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
        }
        self.stream.synchronize().map_err(|e| CudaAroonError::Cuda(e.to_string()))?;
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

    struct AroonBatchBench {
        cuda: CudaAroon,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: AroonBatchRange,
    }
    impl CudaBenchState for AroonBatchBench {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .aroon_batch_dev(&self.high, &self.low, &self.sweep);
        }
    }
    fn prep_batch() -> Box<dyn CudaBenchState> {
        let (h, l) = gen_series(200_000);
        let sweep = AroonBatchRange {
            length: (10, 500, 1),
        };
        let cuda = CudaAroon::new(0).expect("cuda aroon");
        Box::new(AroonBatchBench {
            cuda,
            high: h,
            low: l,
            sweep,
        })
    }

    struct AroonManyBench {
        cuda: CudaAroon,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        length: usize,
    }
    impl CudaBenchState for AroonManyBench {
        fn launch(&mut self) {
            let _ = self.cuda.aroon_many_series_one_param_time_major_dev(
                &self.high_tm,
                &self.low_tm,
                self.cols,
                self.rows,
                self.length,
            );
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
        Box::new(AroonManyBench {
            cuda,
            high_tm,
            low_tm,
            cols,
            rows,
            length: 25,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let bytes_batch = (200_000usize * 2 + (500 - 10 + 1) * 200_000usize * 2) * 4; // rough
        let bytes_many = 256usize * 16_384usize * 2 * 4 * 3; // 2 inputs + 2 outputs + firsts
        vec![
            CudaBenchScenario::new(
                "aroon",
                "one_series_many_params",
                "aroon_cuda_batch",
                "200k_x_491",
                prep_batch,
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
