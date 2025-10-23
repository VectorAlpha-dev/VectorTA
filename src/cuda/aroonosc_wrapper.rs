//! CUDA support for the Aroon Oscillator (aroonosc).
//!
//! Mirrors the CPU batching API by accepting a single high/low series with many
//! parameter combinations (length), and a many-series × one-param time-major
//! entry. Kernels operate in FP32 and replicate scalar semantics:
//! - Warmup per row/series at: first_valid + length
//! - Warmup prefix filled with NaN
//! - Value = clamp(-100, 100, 100/length * (idx_high - idx_low))
//!
//! PTX symbols expected:
//! - "aroonosc_batch_f32"
//! - "aroonosc_many_series_one_param_f32"

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::aroonosc::{AroonOscBatchRange, AroonOscParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaAroonOscError {
    Cuda(String),
    InvalidInput(String),
    NotImplemented,
}

impl fmt::Display for CudaAroonOscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAroonOscError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAroonOscError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaAroonOscError::NotImplemented => write!(f, "CUDA AroonOsc not implemented"),
        }
    }
}

impl std::error::Error for CudaAroonOscError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaAroonOscPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaAroonOscPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaAroonOsc {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaAroonOscPolicy,
    last_batch_block: Option<u32>,
    last_many_block: Option<u32>,
}

impl CudaAroonOsc {
    pub fn new(device_id: usize) -> Result<Self, CudaAroonOscError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/aroonosc_kernel.ptx"));
        // Prefer context target + O2; retry with simpler options if needed
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaAroonOscPolicy::default(),
            last_batch_block: None,
            last_many_block: None,
        })
    }

    pub fn set_policy(&mut self, policy: CudaAroonOscPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaAroonOscPolicy {
        &self.policy
    }

    // ---------- Batch: one-series × many-params ----------

    pub fn aroonosc_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &AroonOscBatchRange,
    ) -> Result<DeviceArrayF32, CudaAroonOscError> {
        let (combos, first_valid, series_len) =
            Self::prepare_batch_inputs(high_f32, low_f32, sweep)?;
        let n_combos = combos.len();
        let lengths_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.length.unwrap_or(0) as i32)
            .collect();

        // VRAM estimate with 64 MB headroom
        let bytes = high_f32.len() * 4
            + low_f32.len() * 4
            + lengths_i32.len() * 4
            + n_combos * series_len * 4
            + 64 * 1024 * 1024;
        if let Ok((free, _)) = mem_get_info() {
            if bytes > free {
                return Err(CudaAroonOscError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (bytes as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let d_high = DeviceBuffer::from_slice(high_f32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_f32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let d_lengths = DeviceBuffer::from_slice(&lengths_i32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len * n_combos)
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?
        };

        let avg_len: f32 = lengths_i32
            .iter()
            .map(|&x| (x.max(1)) as f32)
            .sum::<f32>()
            / (n_combos as f32);

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_lengths,
            series_len as i32,
            first_valid as i32,
            n_combos as i32,
            &mut d_out,
            avg_len,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        series_len: i32,
        first_valid: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
        avg_len: f32,
    ) -> Result<(), CudaAroonOscError> {
        if n_combos <= 0 || series_len <= 0 {
            return Ok(());
        }

        let block_x = self.select_block_x_batch(avg_len);
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let func = self
                .module
                .get_function("aroonosc_batch_f32")
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut series_len_i = series_len;
            let mut first_valid_i = first_valid;
            let mut n_combos_i = n_combos;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut lengths_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        }
        // For debug logging parity; print once per process when BENCH_DEBUG=1
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if self.last_batch_block != Some(block_x) {
                eprintln!("[DEBUG] aroonosc batch block_x={}", block_x);
                unsafe {
                    (*(self as *const _ as *mut CudaAroonOsc)).last_batch_block = Some(block_x);
                }
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn select_block_x_batch(&self, avg_len: f32) -> u32 {
        if let BatchKernelPolicy::OneD { block_x } = self.policy.batch {
            if block_x > 0 {
                // Ensure at least one full warp and round to 32-multiple
                return ((block_x + 31) / 32) * 32;
            }
        }
        if avg_len >= 256.0 {
            256
        } else if avg_len >= 64.0 {
            256
        } else if avg_len >= 32.0 {
            128
        } else {
            64
        }
    }

    #[inline(always)]
    fn select_block_x_many(&self, series_len: i32) -> u32 {
        if let ManySeriesKernelPolicy::OneD { block_x } = self.policy.many_series {
            if block_x > 0 {
                return block_x;
            }
        }
        let s = series_len.max(1) as u32;
        let up_to_warp = ((s + 31) / 32) * 32;
        up_to_warp.clamp(32, 256)
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        sweep: &AroonOscBatchRange,
    ) -> Result<(Vec<AroonOscParams>, usize, usize), CudaAroonOscError> {
        let len = high.len();
        if len == 0 || low.len() != len {
            return Err(CudaAroonOscError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }

        let combos = expand_lengths(sweep);
        if combos.is_empty() {
            return Err(CudaAroonOscError::InvalidInput(
                "no length combinations".into(),
            ));
        }

        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan())
            .ok_or_else(|| CudaAroonOscError::InvalidInput("all values are NaN".into()))?;

        let max_len = combos
            .iter()
            .map(|p| p.length.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_len == 0 {
            return Err(CudaAroonOscError::InvalidInput(
                "length must be positive".into(),
            ));
        }
        let window = max_len + 1;
        let valid = len - first_valid;
        if valid < window {
            return Err(CudaAroonOscError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                window, valid
            )));
        }
        Ok((combos, first_valid, len))
    }

    // ---------- Many-series × one-param (time-major) ----------

    pub fn aroonosc_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        length: usize,
    ) -> Result<DeviceArrayF32, CudaAroonOscError> {
        if high_tm_f32.len() != low_tm_f32.len() {
            return Err(CudaAroonOscError::InvalidInput("mismatched inputs".into()));
        }
        if rows == 0 || cols == 0 || length == 0 {
            return Err(CudaAroonOscError::InvalidInput(
                "rows/cols/length must be positive".into(),
            ));
        }
        if high_tm_f32.len() != rows * cols {
            return Err(CudaAroonOscError::InvalidInput("shape mismatch".into()));
        }

        // first_valid per series (time-major layout)
        let mut first_valids: Vec<i32> = vec![0; rows];
        for s in 0..rows {
            let mut fv = -1i32;
            for t in 0..cols {
                let idx = t * rows + s;
                let h = high_tm_f32[idx];
                let l = low_tm_f32[idx];
                if !h.is_nan() && !l.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv.max(0);
        }

        // VRAM estimate (64 MB headroom)
        let bytes = (high_tm_f32.len() + low_tm_f32.len()) * 4
            + first_valids.len() * 4
            + rows * cols * 4
            + 64 * 1024 * 1024;
        if let Ok((free, _)) = mem_get_info() {
            if bytes > free {
                return Err(CudaAroonOscError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (bytes as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let d_high = DeviceBuffer::from_slice(high_tm_f32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_tm_f32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(rows * cols)
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_fv,
            rows as i32,
            cols as i32,
            length as i32,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: i32,
        series_len: i32,
        length: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAroonOscError> {
        if num_series <= 0 || series_len <= 0 || length <= 0 {
            return Ok(());
        }
        let block_x = self.select_block_x_many(series_len);
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let func = self
                .module
                .get_function("aroonosc_many_series_one_param_f32")
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut ns = num_series;
            let mut sl = series_len;
            let mut l = length;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut ns as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if self.last_many_block != Some(block_x) {
                eprintln!("[DEBUG] aroonosc many-series block_x={}", block_x);
                unsafe {
                    (*(self as *const _ as *mut CudaAroonOsc)).last_many_block = Some(block_x);
                }
            }
        }
        Ok(())
    }

    // Convenience: copy device output to host slice (FP32)
    pub fn aroonosc_batch_into_host_f32(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &AroonOscBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<AroonOscParams>), CudaAroonOscError> {
        let (combos, first_valid, series_len) =
            Self::prepare_batch_inputs(high_f32, low_f32, sweep)?;
        if out.len() != combos.len() * series_len {
            return Err(CudaAroonOscError::InvalidInput(format!(
                "out wrong length: got {}, expected {}",
                out.len(),
                combos.len() * series_len
            )));
        }
        let d_high = DeviceBuffer::from_slice(high_f32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_f32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let lengths_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.length.unwrap_or(0) as i32)
            .collect();
        let d_lengths = DeviceBuffer::from_slice(&lengths_i32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len * combos.len())
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?
        };

        let avg_len: f32 = lengths_i32
            .iter()
            .map(|&x| (x.max(1)) as f32)
            .sum::<f32>()
            / (combos.len() as f32);

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_lengths,
            series_len as i32,
            first_valid as i32,
            combos.len() as i32,
            &mut d_out,
            avg_len,
        )?;
        // Ensure the kernel finished, then single D2H copy
        self.stream
            .synchronize()
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        d_out
            .copy_to(out)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        Ok((combos.len(), series_len, combos))
    }

    /// Optional: write results directly into a pinned host buffer to enable copy/compute overlap.
    pub fn aroonosc_batch_into_pinned_host_f32(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &AroonOscBatchRange,
        pinned_out: &mut LockedBuffer<f32>,
    ) -> Result<(usize, usize, Vec<AroonOscParams>), CudaAroonOscError> {
        let (combos, first_valid, series_len) =
            Self::prepare_batch_inputs(high_f32, low_f32, sweep)?;
        if pinned_out.len() != combos.len() * series_len {
            return Err(CudaAroonOscError::InvalidInput(format!(
                "pinned_out wrong length: got {}, expected {}",
                pinned_out.len(),
                combos.len() * series_len
            )));
        }

        let d_high = DeviceBuffer::from_slice(high_f32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_f32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let lengths_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.length.unwrap_or(0) as i32)
            .collect();
        let d_lengths = DeviceBuffer::from_slice(&lengths_i32)
            .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len * combos.len())
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?
        };

        let avg_len: f32 = lengths_i32
            .iter()
            .map(|&x| (x.max(1)) as f32)
            .sum::<f32>()
            / (combos.len() as f32);

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_lengths,
            series_len as i32,
            first_valid as i32,
            combos.len() as i32,
            &mut d_out,
            avg_len,
        )?;

        unsafe {
            d_out
                .async_copy_to(pinned_out.as_mut_slice(), &self.stream)
                .map_err(|e| CudaAroonOscError::Cuda(e.to_string()))?;
        }
        Ok((combos.len(), series_len, combos))
    }
}

fn expand_lengths(range: &AroonOscBatchRange) -> Vec<AroonOscParams> {
    let (start, end, step) = range.length;
    if step == 0 || start == end {
        return vec![AroonOscParams {
            length: Some(start),
        }];
    }
    let mut v = Vec::new();
    let mut cur = start;
    while cur <= end {
        v.push(AroonOscParams { length: Some(cur) });
        cur = cur.saturating_add(step);
    }
    v
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * 4; // high+low
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * 4;
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0021;
            let off = (0.0033 * x.sin()).abs() + 0.1;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct AroonOscBatchState {
        cuda: CudaAroonOsc,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: AroonOscBatchRange,
    }
    impl CudaBenchState for AroonOscBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .aroonosc_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("aroonosc batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaAroonOsc::new(0).expect("cuda aroonosc");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_close(&close);
        let sweep = AroonOscBatchRange {
            length: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(AroonOscBatchState {
            cuda,
            high,
            low,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "aroonosc",
            "one_series_many_params",
            "aroonosc_cuda_batch_dev",
            "1m_x_128",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
