#![cfg(feature = "cuda")]

//! CUDA wrapper for Median Absolute Deviation (MEDIUM_AD).
//!
//! Parity with ALMA/CUDA wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/medium_ad_kernel.ptx")) with JIT
//!   options DetermineTargetFromContext + OptLevel O2, with simpler fallbacks if needed.
//! - NON_BLOCKING stream
//! - VRAM guard and grid.y chunking (<= 65_535)
//! - Public device entry points for batch (one-series × many-params) and
//!   many-series (time-major) × one-param.
//!
//! Semantics:
//! - Identical to scalar path in src/indicators/medium_ad.rs (warmup NaNs,
//!   window NaNs → NaN, period==1 → 0.0 on finite input).

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::medium_ad::MediumAdBatchRange;
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize, Function};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

const MEDIUM_AD_MAX_PERIOD: usize = 512; // must match kernels/cuda/medium_ad_kernel.cu

#[derive(Debug, Error)]
pub enum CudaMediumAdError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch configuration too large: grid=({gx},{gy},{gz}), block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer on {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

#[derive(Clone, Debug)]
struct MediumAdCombo {
    period: i32,
}

pub struct CudaMediumAd {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaMediumAd {
    pub fn new(device_id: usize) -> Result<Self, CudaMediumAdError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/medium_ad_kernel.ptx"));
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
            context,
            device_id: device_id as u32,
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

    #[inline]
    fn pick_block_x_from_occupancy(func: &Function) -> u32 {
        match func.suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0)) {
            Ok((suggested, _min_grid)) => suggested.clamp(64, 256),
            Err(_) => 128,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMediumAdError> {
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaMediumAdError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    fn expand_grid(range: &MediumAdBatchRange) -> Result<Vec<MediumAdCombo>, CudaMediumAdError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaMediumAdError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            if start < end {
                return Ok((start..=end).step_by(step.max(1)).collect());
            }
            let mut v = Vec::new();
            let mut x = start as isize;
            let end_i = end as isize;
            let st = (step as isize).max(1);
            while x >= end_i {
                v.push(x as usize);
                x = x.saturating_sub(st);
                if x < 0 {
                    break;
                }
            }
            if v.is_empty() {
                return Err(CudaMediumAdError::InvalidInput(format!(
                    "invalid period range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }

        let periods = axis_usize(range.period)?;
        if periods.is_empty() {
            return Err(CudaMediumAdError::InvalidInput(format!(
                "invalid period range: start={}, end={}, step={}",
                range.period.0, range.period.1, range.period.2
            )));
        }

        Ok(periods
            .into_iter()
            .map(|p| MediumAdCombo { period: p as i32 })
            .collect())
    }

    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaMediumAdError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)?
            .max(1) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)?
            .max(1) as u32;
        let max_grid_y = device
            .get_attribute(DeviceAttribute::MaxGridDimY)?
            .max(1) as u32;
        let max_grid_z = device
            .get_attribute(DeviceAttribute::MaxGridDimZ)?
            .max(1) as u32;

        let threads_per_block = bx
            .saturating_mul(by)
            .saturating_mul(bz);
        if threads_per_block > max_threads
            || gx > max_grid_x
            || gy > max_grid_y
            || gz > max_grid_z
        {
            return Err(CudaMediumAdError::LaunchConfigTooLarge {
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

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MediumAdBatchRange,
    ) -> Result<(Vec<MediumAdCombo>, usize), CudaMediumAdError> {
        if data_f32.is_empty() {
            return Err(CudaMediumAdError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaMediumAdError::InvalidInput("all NaN/INF".into()))?;
        let combos = Self::expand_grid(sweep)?;
        for c in &combos {
            let p = c.period as usize;
            if p == 0 || p > len {
                return Err(CudaMediumAdError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    p, len
                )));
            }
            if p > MEDIUM_AD_MAX_PERIOD {
                return Err(CudaMediumAdError::InvalidInput(format!(
                    "period {} exceeds MEDIUM_AD_MAX_PERIOD {} for CUDA path",
                    p, MEDIUM_AD_MAX_PERIOD
                )));
            }
            if len - first_valid < p {
                return Err(CudaMediumAdError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    p,
                    len - first_valid
                )));
            }
        }
        Ok((combos, first_valid))
    }

    fn launch_batch_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMediumAdError> {
        let mut func = self
            .module
            .get_function("medium_ad_batch_f32")
            .map_err(|_| CudaMediumAdError::MissingKernelSymbol {
                name: "medium_ad_batch_f32",
            })?;

        // Prefer L1 cache and pick launch size via occupancy
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x: u32 = Self::pick_block_x_from_occupancy(&func);
        let grid_x = ((len as u32) + block_x - 1) / block_x;

        // Chunk grid.y to avoid launch limits and large VRAM spikes
        const max_y: usize = 65_535;
        let chunk_rows = n_combos.min(max_y).max(1);
        let mut launched = 0usize;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk_rows);
            let gx = grid_x.max(1);
            let gy = cur as u32;
            let gz = 1u32;
            self.validate_launch(gx, gy, gz, block_x, 1, 1)?;
            let grid: GridSize = (gx, gy, gz).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut fv_i = first_valid as i32;
                let periods_byte_offset = launched
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or_else(|| {
                        CudaMediumAdError::InvalidInput(
                            "periods offset overflow in medium_ad batch kernel".into(),
                        )
                    })? as u64;
                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(periods_byte_offset);
                let mut ncomb_i = cur as i32;
                let out_elem_offset = launched
                    .checked_mul(len)
                    .ok_or_else(|| {
                        CudaMediumAdError::InvalidInput(
                            "output offset overflow in medium_ad batch kernel".into(),
                        )
                    })?;
                let out_byte_offset = out_elem_offset
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        CudaMediumAdError::InvalidInput(
                            "output byte offset overflow in medium_ad batch kernel".into(),
                        )
                    })? as u64;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(out_byte_offset);

                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream.launch(&func, grid, block, 0, args)?;
            }

            launched += cur;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[MediumAdCombo],
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaMediumAdError> {
        let len = data_f32.len();
        let elem_size = std::mem::size_of::<f32>();
        let n_elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| {
                CudaMediumAdError::InvalidInput("rows*cols overflow in medium_ad batch".into())
            })?;
        let out_bytes = n_elems
            .checked_mul(elem_size)
            .ok_or_else(|| {
                CudaMediumAdError::InvalidInput("output bytes overflow in medium_ad batch".into())
            })?;
        let in_bytes = len
            .checked_mul(elem_size)
            .ok_or_else(|| {
                CudaMediumAdError::InvalidInput("input bytes overflow in medium_ad batch".into())
            })?;
        let total_bytes = in_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| {
                CudaMediumAdError::InvalidInput("total bytes overflow in medium_ad batch".into())
            })?;
        Self::will_fit(total_bytes, 64 << 20)?;

        // Prefer async copies
        let h_data = LockedBuffer::from_slice(data_f32)?;
        let mut d_data =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }?;
        unsafe { d_data.async_copy_from(&h_data, &self.stream) }?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        // Pinned host memory + async copy for the smaller params vector too
        let h_periods = LockedBuffer::from_slice(&periods)?;
        let mut d_periods =
            unsafe { DeviceBuffer::<i32>::uninitialized_async(periods.len(), &self.stream) }?;
        unsafe { d_periods.async_copy_from(&h_periods, &self.stream) }?;

        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(n_elems, &self.stream) }?;

        self.launch_batch_kernel(
            &d_data,
            len,
            first_valid,
            &d_periods,
            combos.len(),
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn medium_ad_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MediumAdBatchRange,
    ) -> Result<DeviceArrayF32, CudaMediumAdError> {
        let (combos, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid)
    }

    // ---------------- Many-series × one-param (time-major) ----------------

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<(Vec<i32>, usize), CudaMediumAdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMediumAdError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if cols
            .checked_mul(rows)
            .map(|n| n != data_tm_f32.len())
            .unwrap_or(true)
        {
            return Err(CudaMediumAdError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols.checked_mul(rows).unwrap_or(usize::MAX),
                data_tm_f32.len()
            )));
        }
        if period == 0 || period > rows {
            return Err(CudaMediumAdError::InvalidInput(
                "invalid period for many-series".into(),
            ));
        }
        if period > MEDIUM_AD_MAX_PERIOD {
            return Err(CudaMediumAdError::InvalidInput(format!(
                "period {} exceeds MEDIUM_AD_MAX_PERIOD {} for CUDA path",
                period, MEDIUM_AD_MAX_PERIOD
            )));
        }

        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if v.is_finite() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }
        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMediumAdError> {
        let mut func = self
            .module
            .get_function("medium_ad_many_series_one_param_f32")
            .map_err(|_| CudaMediumAdError::MissingKernelSymbol {
                name: "medium_ad_many_series_one_param_f32",
            })?;

        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x: u32 = Self::pick_block_x_from_occupancy(&func);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let gx = grid_x.max(1);
        let gy = 1u32;
        let gz = 1u32;
        self.validate_launch(gx, gy, gz, block_x, 1, 1)?;
        let grid: GridSize = (gx, gy, gz).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut data_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    pub fn medium_ad_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMediumAdError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, period)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| {
                CudaMediumAdError::InvalidInput(
                    "cols*rows overflow in medium_ad many-series".into(),
                )
            })?;
        let elem_size = std::mem::size_of::<f32>();
        let in_bytes = elems
            .checked_mul(elem_size)
            .ok_or_else(|| {
                CudaMediumAdError::InvalidInput(
                    "input bytes overflow in medium_ad many-series".into(),
                )
            })?;
        let out_bytes = in_bytes;
        let total_bytes = in_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| {
                CudaMediumAdError::InvalidInput(
                    "total bytes overflow in medium_ad many-series".into(),
                )
            })?;
        Self::will_fit(total_bytes, 64 << 20)?;

        let h_prices = LockedBuffer::from_slice(data_tm_f32)?;
        let h_first = LockedBuffer::from_slice(&first_valids)?;

        let mut d_prices =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;
        let mut d_first =
            unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;

        unsafe { d_prices.async_copy_from(&h_prices, &self.stream) }?;
        unsafe { d_first.async_copy_from(&h_first, &self.stream) }?;

        self.launch_many_series_kernel(&d_prices, cols, rows, period, &d_first, &mut d_out)?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    // Keep sizes modest due to O(p log p) per output cost on GPU
    const ONE_SERIES_LEN: usize = 200_000;
    const PARAM_SWEEP: usize = 64;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + (64 << 20)
    }

    struct MediumAdBatchDeviceState {
        cuda: CudaMediumAd,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for MediumAdBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    self.len,
                    self.first_valid,
                    &self.d_periods,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("medium_ad batch");
            self.cuda.stream.synchronize().expect("sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMediumAd::new(0).expect("cuda medium_ad");
        let price = gen_series(ONE_SERIES_LEN);
        let start = 5usize;
        let end = start + PARAM_SWEEP - 1;
        let sweep = MediumAdBatchRange {
            period: (start, end, 1),
        };
        let (combos, first_valid) =
            CudaMediumAd::prepare_batch_inputs(&price, &sweep).expect("prep medium_ad");
        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(ONE_SERIES_LEN * combos.len()) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(MediumAdBatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            d_out,
            len: ONE_SERIES_LEN,
            first_valid,
            n_combos: combos.len(),
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "medium_ad",
            "one_series_many_params",
            "medium_ad_cuda_batch_dev",
            "200k_x_64",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
