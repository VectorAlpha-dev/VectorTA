//! CUDA wrapper for VOSS (Ehlers Voss Filter)
//!
//! Parity with ALMA/CWMA wrappers:
//! - PTX load via DetermineTargetFromContext + OptLevel O2, with simple fallbacks
//! - NON_BLOCKING stream
//! - Policy enums for batch and many-series
//! - VRAM checks + grid.y chunking (<= 65_535) for batch
//! - Warmup/NaN semantics identical to scalar: warm = first_valid + max(period, 5, 3*predict);
//!   filt[start-2..start) set to 0.0; rest of warmup are NaN.
//! - f64 accumulators in kernels; f32 I/O

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::voss::{expand_grid_voss, VossBatchRange, VossParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaVossError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("voss: invalid input: {0}")]
    InvalidInput(String),
    #[error("voss: invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("voss: out of memory on device: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("voss: missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("voss: launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("voss: arithmetic overflow when computing {what}")]
    ArithmeticOverflow { what: &'static str },
    #[error("voss: device mismatch for buffer (buf={buf}, current={current})")]
    DeviceMismatch { buf: i32, current: i32 },
    #[error("voss: not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32, block_y: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaVossPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaVossPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaVoss {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaVossPolicy,
}

impl CudaVoss {
    pub fn new(device_id: usize) -> Result<Self, CudaVossError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/voss_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?,
            },
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        // Prefer L1 cache for both kernels (no shared memory used by these).
        if let Ok(mut f) = module.get_function("voss_batch_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }
        if let Ok(mut f) = module.get_function("voss_many_series_one_param_time_major_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }

        Ok(Self {
            module,
            stream,
            context: context.clone(),
            device_id: device_id as u32,
            policy: CudaVossPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, p: CudaVossPolicy) { self.policy = p; }
    pub fn synchronize(&self) -> Result<(), CudaVossError> {
        self.stream.synchronize().map_err(Into::into)
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
    fn will_fit(required_bytes: usize, headroom: usize) -> Result<(), CudaVossError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        let (free, _total) = match Self::device_mem_info() {
            Some(v) => v,
            None => return Ok(()),
        };
        let need = required_bytes
            .checked_add(headroom)
            .ok_or(CudaVossError::ArithmeticOverflow {
                what: "required_bytes + headroom_bytes",
            })?;
        if need <= free {
            Ok(())
        } else {
            Err(CudaVossError::OutOfMemory {
                required: required_bytes,
                free,
                headroom,
            })
        }
    }

    // -------------------- Batch (one series × many params) --------------------
    pub fn voss_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VossBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, Vec<VossParams>), CudaVossError> {
        if data_f32.is_empty() {
            return Err(CudaVossError::InvalidInput("empty input".into()));
        }
        let len = data_f32.len();
        let first = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaVossError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_voss(sweep)
            .map_err(|e| CudaVossError::InvalidInput(e.to_string()))?;
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            let q = prm.predict.unwrap_or(0);
            let order = 3 * q;
            let min_index = p.max(5).max(order);
            if p == 0 || p > len {
                return Err(CudaVossError::InvalidInput("invalid period".into()));
            }
            if len - first < min_index {
                return Err(CudaVossError::InvalidInput("not enough valid data".into()));
            }
            let b = prm.bandwidth.unwrap_or(0.25);
            if !b.is_finite() || b <= 0.0 || b > 1.0 {
                return Err(CudaVossError::InvalidInput("invalid bandwidth".into()));
            }
        }

        let rows = combos.len();
        // VRAM estimate (inputs uploaded as f64) + params + outputs
        let in_bytes = len
            .checked_mul(8)
            .ok_or(CudaVossError::ArithmeticOverflow { what: "len * 8 (prices upload)" })?;
        let params_per_row = std::mem::size_of::<i32>()
            .checked_mul(2)
            .and_then(|x| x.checked_add(std::mem::size_of::<f64>()))
            .ok_or(CudaVossError::ArithmeticOverflow { what: "param bytes per row" })?;
        let params_bytes = rows
            .checked_mul(params_per_row)
            .ok_or(CudaVossError::ArithmeticOverflow { what: "rows * param bytes" })?;
        let elems = rows
            .checked_mul(len)
            .ok_or(CudaVossError::ArithmeticOverflow { what: "rows * len (outputs)" })?;
        let outs_bytes = elems
            .checked_mul(4)
            .and_then(|x| x.checked_mul(2))
            .ok_or(CudaVossError::ArithmeticOverflow { what: "2 * rows * len * 4 (outputs)" })?;
        let required_bytes = in_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(outs_bytes))
            .ok_or(CudaVossError::ArithmeticOverflow { what: "total batch bytes" })?;
        Self::will_fit(required_bytes, Self::headroom_bytes())?;

        // H2D (prices): page-locked host buffer + async copy on our NON_BLOCKING stream
        let mut h_prices = unsafe { LockedBuffer::<f64>::uninitialized(len) }?;
        for (dst, &src) in h_prices.as_mut_slice().iter_mut().zip(data_f32.iter()) {
            *dst = src as f64;
        }
        let mut d_prices: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        unsafe { d_prices.async_copy_from(h_prices.as_slice(), &self.stream) }?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap_or(20) as i32).collect();
        let predicts: Vec<i32> = combos.iter().map(|c| c.predict.unwrap_or(3) as i32).collect();
        let bws: Vec<f64> = combos.iter().map(|c| c.bandwidth.unwrap_or(0.25)).collect();
        let d_p = DeviceBuffer::from_slice(&periods)?;
        let d_q = DeviceBuffer::from_slice(&predicts)?;
        let d_bw = DeviceBuffer::from_slice(&bws)?;

        let mut d_voss: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_filt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        // Launch with grid.y chunking
        let func = self
            .module
            .get_function("voss_batch_f32")
            .map_err(|_| CudaVossError::MissingKernelSymbol { name: "voss_batch_f32" })?;
        // Only threadIdx.x==0 performs the scan; default block.x=1 avoids stranding lanes.
        let block_x = match self.policy.batch { BatchKernelPolicy::OneD { block_x } if block_x > 0 => block_x, _ => 1 };
        const MAX_GRID_Y: usize = 65_535;
        let mut start_row = 0usize;
        while start_row < rows {
            let count = (rows - start_row).min(MAX_GRID_Y);
            let grid: GridSize = (1u32, count as u32, 1u32).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut p_prices = d_prices.as_device_ptr().as_raw();
                let mut p_len = len as i32;
                let mut p_first = first as i32;
                let mut p_per = d_p.as_device_ptr().add(start_row).as_raw();
                let mut p_pre = d_q.as_device_ptr().add(start_row).as_raw();
                let mut p_bw = d_bw.as_device_ptr().add(start_row).as_raw();
                let mut p_nrows = count as i32;
                let base = start_row * len;
                let mut p_voss = d_voss.as_device_ptr().add(base).as_raw();
                let mut p_filt = d_filt.as_device_ptr().add(base).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_prices as *mut _ as *mut c_void,
                    &mut p_len as *mut _ as *mut c_void,
                    &mut p_first as *mut _ as *mut c_void,
                    &mut p_per as *mut _ as *mut c_void,
                    &mut p_pre as *mut _ as *mut c_void,
                    &mut p_bw as *mut _ as *mut c_void,
                    &mut p_nrows as *mut _ as *mut c_void,
                    &mut p_voss as *mut _ as *mut c_void,
                    &mut p_filt as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            start_row += count;
        }

        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_voss,
                rows,
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_filt,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    // -------------------- Many-series × one-param (time-major) --------------------
    pub fn voss_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VossParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaVossError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVossError::InvalidInput("empty matrix".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or(CudaVossError::ArithmeticOverflow { what: "cols * rows" })?;
        if data_tm_f32.len() != elems {
            return Err(CudaVossError::InvalidInput(
                "data must be time-major cols*rows".into(),
            ));
        }

        let p = params.period.unwrap_or(20);
        let q = params.predict.unwrap_or(3);
        let b = params.bandwidth.unwrap_or(0.25);
        if p == 0 || !b.is_finite() || b <= 0.0 || b > 1.0 {
            return Err(CudaVossError::InvalidInput("invalid params".into()));
        }

        // Per-series first-valid indices (walk column with +cols stride)
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut idx = s;
            for t in 0..rows {
                let v = data_tm_f32[idx];
                if !v.is_nan() { first_valids[s] = t as i32; break; }
                idx += cols;
            }
        }

        // VRAM estimate (inputs uploaded as f64)
        let in_bytes = elems
            .checked_mul(8)
            .ok_or(CudaVossError::ArithmeticOverflow { what: "elems * 8 (many-series input)" })?;
        let first_valid_bytes = cols
            .checked_mul(4)
            .ok_or(CudaVossError::ArithmeticOverflow { what: "cols * 4 (first_valids)" })?;
        let outs_bytes = elems
            .checked_mul(4)
            .and_then(|x| x.checked_mul(2))
            .ok_or(CudaVossError::ArithmeticOverflow { what: "2 * elems * 4 (many-series outputs)" })?;
        let required_bytes = in_bytes
            .checked_add(first_valid_bytes)
            .and_then(|x| x.checked_add(outs_bytes))
            .ok_or(CudaVossError::ArithmeticOverflow { what: "total many-series bytes" })?;
        Self::will_fit(required_bytes, Self::headroom_bytes())?;

        // H2D (large input): page-locked host buffer + async copy
        let mut h_data = unsafe { LockedBuffer::<f64>::uninitialized(elems) }?;
        for (dst, &src) in h_data.as_mut_slice().iter_mut().zip(data_tm_f32.iter()) { *dst = src as f64; }
        let mut d_data: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        unsafe { d_data.async_copy_from(h_data.as_slice(), &self.stream) }?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_voss: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_filt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        let func = self
            .module
            .get_function("voss_many_series_one_param_time_major_f32")
            .map_err(|_| CudaVossError::MissingKernelSymbol {
                name: "voss_many_series_one_param_time_major_f32",
            })?;

        let (block_x, block_y) = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x, block_y } if block_x > 0 && block_y > 0 => (block_x, block_y),
            _ => (1, u32::min(64, cols as u32)),
        };
        let grid: GridSize = (1, ((cols as u32) + block_y - 1) / block_y, 1).into();
        let block: BlockSize = (block_x, block_y, 1).into();

        unsafe {
            let mut p_data = d_data.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_cols = cols as i32;
            let mut p_rows = rows as i32;
            let mut p_p = p as i32;
            let mut p_q = q as i32;
            let mut p_bw = b as f64;
            let mut p_voss = d_voss.as_device_ptr().as_raw();
            let mut p_filt = d_filt.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_data as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_p as *mut _ as *mut c_void,
                &mut p_q as *mut _ as *mut c_void,
                &mut p_bw as *mut _ as *mut c_void,
                &mut p_voss as *mut _ as *mut c_void,
                &mut p_filt as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_voss,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_filt,
                rows,
                cols,
            },
        ))
    }
}

// ---------------- Benches ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_ROWS: usize = 1_000_000;

    fn bytes_batch(rows: usize) -> usize {
        let in_bytes = ONE_SERIES_LEN * 8;               // f64 upload
        let params   = rows * (4 + 4 + 8);               // i32, i32, f64
        let outs     = 2 * rows * ONE_SERIES_LEN * 4;    // two f32 outputs
        in_bytes + params + outs + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        elems * 8                                    // f64 upload
            + MANY_SERIES_COLS * 4                   // i32 first_valids
            + 2 * elems * 4                          // two f32 outputs
            + 64 * 1024 * 1024
    }

    struct VossBatchState {
        cuda: CudaVoss,
        d_prices: DeviceBuffer<f64>,
        d_p: DeviceBuffer<i32>,
        d_q: DeviceBuffer<i32>,
        d_bw: DeviceBuffer<f64>,
        len: usize,
        first: usize,
        rows: usize,
        block_x: u32,
        d_voss: DeviceBuffer<f32>,
        d_filt: DeviceBuffer<f32>,
    }
    impl CudaBenchState for VossBatchState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("voss_batch_f32")
                .expect("voss_batch_f32");

            const MAX_GRID_Y: usize = 65_535;
            let mut start_row = 0usize;
            while start_row < self.rows {
                let count = (self.rows - start_row).min(MAX_GRID_Y);
                let grid: GridSize = (1u32, count as u32, 1u32).into();
                let block: BlockSize = (self.block_x, 1, 1).into();
                unsafe {
                    let mut p_prices = self.d_prices.as_device_ptr().as_raw();
                    let mut p_len = self.len as i32;
                    let mut p_first = self.first as i32;
                    let mut p_per = self.d_p.as_device_ptr().add(start_row).as_raw();
                    let mut p_pre = self.d_q.as_device_ptr().add(start_row).as_raw();
                    let mut p_bw = self.d_bw.as_device_ptr().add(start_row).as_raw();
                    let mut p_nrows = count as i32;
                    let base = start_row * self.len;
                    let mut p_voss = self.d_voss.as_device_ptr().add(base).as_raw();
                    let mut p_filt = self.d_filt.as_device_ptr().add(base).as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_prices as *mut _ as *mut c_void,
                        &mut p_len as *mut _ as *mut c_void,
                        &mut p_first as *mut _ as *mut c_void,
                        &mut p_per as *mut _ as *mut c_void,
                        &mut p_pre as *mut _ as *mut c_void,
                        &mut p_bw as *mut _ as *mut c_void,
                        &mut p_nrows as *mut _ as *mut c_void,
                        &mut p_voss as *mut _ as *mut c_void,
                        &mut p_filt as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&func, grid, block, 0, args)
                        .expect("voss batch launch");
                }
                start_row += count;
            }
            self.cuda.stream.synchronize().expect("voss batch sync");
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaVoss::new(0).expect("cuda voss");
        let mut data = gen_series(ONE_SERIES_LEN);
        for i in 0..4 {
            data[i] = f32::NAN;
        }
        let sweep = VossBatchRange {
            period: (10, 34, 2),
            predict: (1, 4, 1),
            bandwidth: (0.1, 0.4, 0.05),
        };

        let len = data.len();
        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let combos = expand_grid_voss(&sweep).expect("expand_grid_voss");
        let rows = combos.len();

        // Upload prices as f64 once.
        let mut prices_f64 = vec![0f64; len];
        for (dst, &src) in prices_f64.iter_mut().zip(data.iter()) {
            *dst = src as f64;
        }
        let d_prices = DeviceBuffer::from_slice(&prices_f64).expect("d_prices");

        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap_or(20) as i32).collect();
        let predicts: Vec<i32> = combos.iter().map(|c| c.predict.unwrap_or(3) as i32).collect();
        let bws: Vec<f64> = combos.iter().map(|c| c.bandwidth.unwrap_or(0.25)).collect();
        let d_p = DeviceBuffer::from_slice(&periods).expect("d_p");
        let d_q = DeviceBuffer::from_slice(&predicts).expect("d_q");
        let d_bw = DeviceBuffer::from_slice(&bws).expect("d_bw");

        let elems = rows.checked_mul(len).expect("rows*len overflow");
        let d_voss: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_voss");
        let d_filt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_filt");

        let block_x = match cuda.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 1,
        };
        cuda.stream.synchronize().expect("voss prep sync");

        Box::new(VossBatchState {
            cuda,
            d_prices,
            d_p,
            d_q,
            d_bw,
            len,
            first,
            rows,
            block_x,
            d_voss,
            d_filt,
        })
    }

    struct VossManyState {
        cuda: CudaVoss,
        d_data: DeviceBuffer<f64>,
        d_fv: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        p: i32,
        q: i32,
        bw: f64,
        grid: GridSize,
        block: BlockSize,
        d_voss: DeviceBuffer<f32>,
        d_filt: DeviceBuffer<f32>,
    }
    impl CudaBenchState for VossManyState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("voss_many_series_one_param_time_major_f32")
                .expect("voss_many_series_one_param_time_major_f32");
            unsafe {
                let mut p_data = self.d_data.as_device_ptr().as_raw();
                let mut p_fv = self.d_fv.as_device_ptr().as_raw();
                let mut p_cols = self.cols as i32;
                let mut p_rows = self.rows as i32;
                let mut p_p = self.p;
                let mut p_q = self.q;
                let mut p_bw = self.bw;
                let mut p_voss = self.d_voss.as_device_ptr().as_raw();
                let mut p_filt = self.d_filt.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_data as *mut _ as *mut c_void,
                    &mut p_fv as *mut _ as *mut c_void,
                    &mut p_cols as *mut _ as *mut c_void,
                    &mut p_rows as *mut _ as *mut c_void,
                    &mut p_p as *mut _ as *mut c_void,
                    &mut p_q as *mut _ as *mut c_void,
                    &mut p_bw as *mut _ as *mut c_void,
                    &mut p_voss as *mut _ as *mut c_void,
                    &mut p_filt as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, args)
                    .expect("voss many-series launch");
            }
            self.cuda.stream.synchronize().expect("voss many-series sync");
        }
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaVoss::new(0).expect("cuda voss");
        let mut tm = gen_time_major_prices(MANY_SERIES_COLS, MANY_SERIES_ROWS);
        // ensure a few NaNs at start of each series
        // data tm NaNs and params set above
        for s in 0..MANY_SERIES_COLS {
            tm[s] = f32::NAN;
            tm[s + MANY_SERIES_COLS] = f32::NAN;
        }
        let params = VossParams {
            period: Some(20),
            predict: Some(3),
            bandwidth: Some(0.25),
        };
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let elems = cols.checked_mul(rows).expect("cols*rows overflow");

        // Per-series first_valid indices (time-major)
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut idx = s;
            for t in 0..rows {
                let v = tm[idx];
                if !v.is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
                idx += cols;
            }
        }
        let d_fv = DeviceBuffer::from_slice(&first_valids).expect("d_fv");

        // Upload as f64 once.
        let mut data_f64 = vec![0f64; elems];
        for (dst, &src) in data_f64.iter_mut().zip(tm.iter()) {
            *dst = src as f64;
        }
        let d_data = DeviceBuffer::from_slice(&data_f64).expect("d_data");

        let d_voss: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_voss_tm");
        let d_filt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_filt_tm");

        let p = params.period.unwrap_or(20) as i32;
        let q = params.predict.unwrap_or(3) as i32;
        let bw = params.bandwidth.unwrap_or(0.25);
        let (block_x, block_y) = match cuda.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x, block_y } if block_x > 0 && block_y > 0 => {
                (block_x, block_y)
            }
            _ => (1, u32::min(64, cols as u32)),
        };
        let grid: GridSize = (1, ((cols as u32) + block_y - 1) / block_y, 1).into();
        let block: BlockSize = (block_x, block_y, 1).into();

        cuda.stream.synchronize().expect("voss prep sync");
        Box::new(VossManyState {
            cuda,
            d_data,
            d_fv,
            cols,
            rows,
            p,
            q,
            bw,
            grid,
            block,
            d_voss,
            d_filt,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "voss",
                "one_series_many_params",
                "voss_batch",
                "voss_batch/one_series_many_params",
                prep_batch,
            )
            .with_mem_required({
                let sweep = VossBatchRange {
                    period: (10, 34, 2),
                    predict: (1, 4, 1),
                    bandwidth: (0.1, 0.4, 0.05),
                };
                let rows = expand_grid_voss(&sweep).map(|v| v.len()).unwrap_or(300);
                bytes_batch(rows)
            })
            .with_inner_iters(1),
            CudaBenchScenario::new(
                "voss",
                "one_param_time_major",
                "voss_many_series",
                "voss_many/one_param_time_major",
                prep_many,
            )
            .with_mem_required(bytes_many_series())
            .with_inner_iters(1),
        ]
    }
}
