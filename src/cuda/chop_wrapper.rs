//! CUDA wrapper for Choppiness Index (CHOP)
//!
//! Goals:
//! - API parity with ALMA-style wrappers (PTX via OUT_DIR, NON_BLOCKING stream).
//! - Batch (one series × many params) and many-series × one-param (time-major).
//! - Warmup/NaN semantics identical to scalar implementation in `src/indicators/chop.rs`.
//! - Category-appropriate optimization: host builds sparse tables for H/L (batch)
//!   and host computes ATR prefix sums for many-series to keep kernels simple.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::chop::{ChopBatchRange, ChopParams};
use crate::indicators::willr::build_willr_gpu_tables; // reuse sparse-table prep for H/L
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::fmt;

#[derive(Debug)]
pub enum CudaChopError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaChopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaChopError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaChopError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaChopError {}

pub struct CudaChop {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaChop {
    pub fn new(device_id: usize) -> Result<Self, CudaChopError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/chop_kernel.ptx"));
        // O2 default with fallbacks for driver variance
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    // ---------- Batch (one series × many params) ----------

    fn expand_grid(range: &ChopBatchRange) -> Vec<ChopParams> {
        let (ps, pe, pt) = range.period;
        let (ss, se, st) = range.scalar;
        let (ds, de, dt) = range.drift;
        let periods: Vec<usize> = if pt == 0 || ps == pe {
            vec![ps]
        } else {
            (ps..=pe).step_by(pt).collect()
        };
        let scalars: Vec<f64> = if st == 0.0 || (ss - se).abs() < f64::EPSILON {
            vec![ss]
        } else {
            let mut v = Vec::new();
            let mut x = ss;
            while x <= se + 1e-12 {
                v.push(x);
                x += st;
            }
            v
        };
        let drifts: Vec<usize> = if dt == 0 || ds == de {
            vec![ds]
        } else {
            (ds..=de).step_by(dt).collect()
        };
        let mut combos = Vec::with_capacity(periods.len() * scalars.len() * drifts.len());
        for &p in &periods {
            for &s in &scalars {
                for &d in &drifts {
                    combos.push(ChopParams {
                        period: Some(p),
                        scalar: Some(s),
                        drift: Some(d),
                    });
                }
            }
        }
        combos
    }

    pub fn chop_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &ChopBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<ChopParams>), CudaChopError> {
        let n = close_f32.len();
        if n == 0 || high_f32.len() != n || low_f32.len() != n {
            return Err(CudaChopError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaChopError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // first_valid where H/L/C are all finite (match scalar)
        let mut first = -1isize;
        for i in 0..n {
            let h = high_f32[i];
            let l = low_f32[i];
            let c = close_f32[i];
            if h == h && l == l && c == c {
                first = i as isize;
                break;
            }
        }
        if first < 0 {
            return Err(CudaChopError::InvalidInput("all values are NaN".into()));
        }
        let first = first as usize;

        let max_period = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if n - first < max_period {
            return Err(CudaChopError::InvalidInput(format!(
                "not enough valid data: needed >= {}, have {}",
                max_period,
                n - first
            )));
        }

        // Host precompute: H/L sparse tables for O(1) window range queries
        let tables = build_willr_gpu_tables(high_f32, low_f32);

        // Pack params
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let drifts_i32: Vec<i32> = combos.iter().map(|c| c.drift.unwrap() as i32).collect();
        let scalars_f32: Vec<f32> = combos.iter().map(|c| c.scalar.unwrap() as f32).collect();

        // VRAM estimate (+64MB headroom)
        let out_elems = combos.len() * n;
        let bytes = (high_f32.len() + low_f32.len() + close_f32.len()) * 4
            + (periods_i32.len() + drifts_i32.len()) * 4
            + scalars_f32.len() * 4
            + tables.log2.len() * 4
            + tables.level_offsets.len() * 4
            + tables.st_max.len() * 4
            + tables.st_min.len() * 4
            + tables.nan_psum.len() * 4
            + out_elems * 4;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaChopError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload inputs/params
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_f32, &self.stream) }
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_f32, &self.stream) }
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_f32, &self.stream) }
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_drifts = DeviceBuffer::from_slice(&drifts_i32)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_scalars = DeviceBuffer::from_slice(&scalars_f32)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        let d_log2 = DeviceBuffer::from_slice(&tables.log2)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_offsets = DeviceBuffer::from_slice(&tables.level_offsets)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_st_max = DeviceBuffer::from_slice(&tables.st_max)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_st_min = DeviceBuffer::from_slice(&tables.st_min)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_nan_psum = DeviceBuffer::from_slice(&tables.nan_psum)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }
                .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        // Kernel
        let func = self
            .module
            .get_function("chop_batch_f32")
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        // Occupancy-based suggestion
        let (_, suggested) = func
            .suggested_launch_configuration(
                (max_period * std::mem::size_of::<f32>()) as usize,
                BlockSize::xyz(0, 0, 0),
            )
            .unwrap_or((0, 0));
        let block_x = if suggested > 0 { suggested } else { 256 };

        // Chunk rows to keep grid.x <= 65_535
        let rows = combos.len();
        let mut launched = 0usize;
        while launched < rows {
            let n_this = (rows - launched).min(65_535);
            let grid: GridSize = (n_this as u32, 1u32, 1u32).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let shared_bytes = max_period * std::mem::size_of::<f32>();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shared_bytes as u32, stream>>>(
                        d_high.as_device_ptr(),
                        d_low.as_device_ptr(),
                        d_close.as_device_ptr(),
                        d_periods.as_device_ptr().add(launched),
                        d_drifts.as_device_ptr().add(launched),
                        d_scalars.as_device_ptr().add(launched),
                        d_log2.as_device_ptr(),
                        d_offsets.as_device_ptr(),
                        d_st_max.as_device_ptr(),
                        d_st_min.as_device_ptr(),
                        d_nan_psum.as_device_ptr(),
                        n as i32,
                        first as i32,
                        (tables.level_offsets.len() - 1) as i32,
                        n_this as i32,
                        max_period as i32,
                        d_out.as_device_ptr().add(launched * n)
                    )
                )
                .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
            }
            launched += n_this;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: n,
            },
            combos,
        ))
    }

    // ---------- Many-series × one param (time-major) ----------

    pub fn chop_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ChopParams,
    ) -> Result<DeviceArrayF32, CudaChopError> {
        if cols == 0 || rows == 0 {
            return Err(CudaChopError::InvalidInput("empty matrix".into()));
        }
        let n = cols * rows;
        if high_tm_f32.len() != n || low_tm_f32.len() != n || close_tm_f32.len() != n {
            return Err(CudaChopError::InvalidInput(
                "matrix inputs must have identical length".into(),
            ));
        }
        let period = params.period.unwrap_or(14);
        let drift = params.drift.unwrap_or(1);
        let scalar = params.scalar.unwrap_or(100.0) as f32;
        if period == 0 || drift == 0 {
            return Err(CudaChopError::InvalidInput("invalid params".into()));
        }

        // first-valid per series: H/L/C all finite
        let mut first_valids: Vec<i32> = vec![-1; cols];
        for s in 0..cols {
            let mut fv = -1;
            for r in 0..rows {
                let h = high_tm_f32[r * cols + s];
                let l = low_tm_f32[r * cols + s];
                let c = close_tm_f32[r * cols + s];
                if h == h && l == l && c == c {
                    fv = r as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        // Validate max tail
        for s in 0..cols {
            let fv = first_valids[s];
            if fv < 0 {
                return Err(CudaChopError::InvalidInput("all values are NaN".into()));
            }
            if rows - (fv as usize) < period {
                return Err(CudaChopError::InvalidInput("not enough valid data".into()));
            }
        }

        // Host precompute: ATR series per series and prefix sums (rows+1 × cols)
        let mut atr_psum_tm = vec![0f32; (rows + 1) * cols];
        {
            let inv_drift = 1.0f64 / (drift as f64);
            for s in 0..cols {
                let fv = first_valids[s] as usize;
                let mut prev_close = close_tm_f32[fv * cols + s] as f64;
                let mut rma_atr: f64 = f64::NAN;
                let mut sum_tr = 0.0f64;
                let mut acc = 0.0f64; // prefix
                                      // prefix psum[0..fv] already zero
                for r in fv..rows {
                    let hi = high_tm_f32[r * cols + s] as f64;
                    let lo = low_tm_f32[r * cols + s] as f64;
                    let cl = close_tm_f32[r * cols + s] as f64;
                    let rel = r - fv;
                    let tr = if rel == 0 {
                        hi - lo
                    } else {
                        (hi - lo).max((hi - prev_close).abs().max((lo - prev_close).abs()))
                    };
                    if rel < drift {
                        sum_tr += tr;
                        if rel == drift - 1 {
                            rma_atr = sum_tr * inv_drift;
                        }
                    } else {
                        rma_atr += inv_drift * (tr - rma_atr);
                    }
                    prev_close = cl;
                    let current_atr = if rel < drift {
                        if rel == drift - 1 {
                            rma_atr
                        } else {
                            f64::NAN
                        }
                    } else {
                        rma_atr
                    };
                    let add = if current_atr.is_nan() {
                        0.0
                    } else {
                        current_atr
                    };
                    acc += add;
                    atr_psum_tm[(r + 1) * cols + s] = acc as f32;
                }
            }
        }

        // VRAM estimate (+64MB)
        let bytes = (high_tm_f32.len() + low_tm_f32.len()) * 4 // inputs used on device
            + atr_psum_tm.len() * 4
            + first_valids.len() * 4
            + n * 4;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaChopError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_psum = DeviceBuffer::from_slice(&atr_psum_tm)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }
                .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("chop_many_series_one_param_f32")
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        // 128 threads per block by default
        let block: BlockSize = (128u32, 1u32, 1u32).into();
        let grid: GridSize = (((cols as u32 + 127) / 128).max(1), 1u32, 1u32).into();
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_high.as_device_ptr(),
                    d_low.as_device_ptr(),
                    d_psum.as_device_ptr(),
                    d_first.as_device_ptr(),
                    cols as i32,
                    rows as i32,
                    period as i32,
                    scalar,
                    d_out.as_device_ptr()
                )
            )
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    // ---------- Host-copy helpers (optional) ----------
    pub fn chop_batch_into_host_f32(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &ChopBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<ChopParams>), CudaChopError> {
        let (arr, combos) = self.chop_batch_dev(high_f32, low_f32, close_f32, sweep)?;
        if arr.len() != out.len() {
            return Err(CudaChopError::InvalidInput("out length mismatch".into()));
        }
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(arr.len()) }
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaChopError::Cuda(e.to_string()))?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }
}

// ---------- Benches (wrapper-owned) ----------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    struct ChopBatchBench {
        cuda: CudaChop,
        h: Vec<f32>,
        l: Vec<f32>,
        c: Vec<f32>,
        sweep: ChopBatchRange,
    }
    impl CudaBenchState for ChopBatchBench {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .chop_batch_dev(&self.h, &self.l, &self.c, &self.sweep);
        }
    }

    struct ChopManyBench {
        cuda: CudaChop,
        h_tm: Vec<f32>,
        l_tm: Vec<f32>,
        c_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: ChopParams,
    }
    impl CudaBenchState for ChopManyBench {
        fn launch(&mut self) {
            let _ = self.cuda.chop_many_series_one_param_time_major_dev(
                &self.h_tm,
                &self.l_tm,
                &self.c_tm,
                self.cols,
                self.rows,
                &self.params,
            );
        }
    }

    fn make_ohlc(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut h = vec![f32::NAN; n];
        let mut l = vec![f32::NAN; n];
        let mut c = vec![f32::NAN; n];
        for i in 1..n {
            let x = i as f32 * 0.0013;
            let base = x.sin() + 0.0002 * (i as f32);
            let hi = base + 0.6 + 0.07 * (x * 2.4).cos();
            let lo = base - 0.6 - 0.06 * (x * 1.7).sin();
            h[i] = hi;
            l[i] = lo;
            c[i] = (hi + lo) * 0.5;
        }
        (h, l, c)
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let (h, l, c) = make_ohlc(50_000);
        let sweep = ChopBatchRange {
            period: (5, 55, 5),
            scalar: (100.0, 100.0, 0.0),
            drift: (1, 5, 2),
        };
        let cuda = CudaChop::new(0).expect("cuda chop");
        Box::new(ChopBatchBench {
            cuda,
            h,
            l,
            c,
            sweep,
        })
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cols = 128usize;
        let rows = 8192usize;
        let (mut h, mut l, mut c) = make_ohlc(cols * rows);
        for s in 0..cols {
            h[s] = f32::NAN;
            l[s] = f32::NAN;
            c[s] = f32::NAN;
        }
        let cuda = CudaChop::new(0).expect("cuda chop");
        let params = ChopParams {
            period: Some(14),
            scalar: Some(100.0),
            drift: Some(1),
        };
        Box::new(ChopManyBench {
            cuda,
            h_tm: h,
            l_tm: l,
            c_tm: c,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let bytes_batch = (50_000usize * 3 + (1 + (55 - 5) / 5) * 50_000 + 50_000 * 8) * 4; // rough
        let bytes_many = 128usize * 8192usize * 4usize * 4usize; // 3 inputs + psum + out
        vec![
            CudaBenchScenario::new("chop", "one_series", "chop_cuda_batch", "50k", prep_batch)
                .with_mem_required(bytes_batch),
            CudaBenchScenario::new(
                "chop",
                "many_series",
                "chop_cuda_many_series",
                "128x8k",
                prep_many,
            )
            .with_mem_required(bytes_many),
        ]
    }
}
