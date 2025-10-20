//! CUDA support for the Accelerator Oscillator (ACOSC).
//!
//! Mirrors ALMA-style integration where applicable:
//! - PTX loaded via include_str!(.../acosc_kernel.ptx) with DetermineTargetFromContext + OptLevel O2
//! - Stream NON_BLOCKING
//! - Conservative VRAM checks with ~64MB headroom
//! - Public device entry points for one-series×batch (degenerate single row) and
//!   many-series×one-param (time-major)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::ffi::c_void;
use std::fmt;

// --- Kernel constants (must match CUDA kernels) ---
const P5: usize = 5;
const P34: usize = 34;
const WARP: usize = 32;
// Dynamic shared memory required by the warp kernel (bytes)
const SHMEM_WARP_BYTES: u32 = ((P34 + P5 + P5) * WARP * std::mem::size_of::<f32>()) as u32;

#[derive(Debug)]
pub enum CudaAcoscError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAcoscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAcoscError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAcoscError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaAcoscError {}

pub struct DeviceAcoscPair {
    pub osc: DeviceArrayF32,
    pub change: DeviceArrayF32,
}
impl DeviceAcoscPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.osc.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.osc.cols
    }
}

pub struct CudaAcosc {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaAcosc {
    pub fn new(device_id: usize) -> Result<Self, CudaAcoscError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/acosc_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    fn device_mem_info() -> Option<(usize, usize)> {
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if res == cu::CUresult::CUDA_SUCCESS {
                Some((free, total))
            } else {
                None
            }
        }
    }
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // -------- Batch: one series (degenerate single row) --------
    pub fn acosc_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
    ) -> Result<DeviceAcoscPair, CudaAcoscError> {
        let len = high_f32.len();
        if len == 0 || low_f32.len() != len {
            return Err(CudaAcoscError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }
        let first_valid = (0..len)
            .find(|&i| high_f32[i].is_finite() && low_f32[i].is_finite())
            .unwrap_or(len);

        // VRAM estimate: 2 inputs + 2 outputs
        let in_bytes = 2 * len * std::mem::size_of::<f32>();
        let out_bytes = 2 * len * std::mem::size_of::<f32>();
        let required = in_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaAcoscError::InvalidInput(
                "insufficient device memory for acosc batch".into(),
            ));
        }

        let d_high =
            DeviceBuffer::from_slice(high_f32).map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low_f32).map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let mut d_osc: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let mut d_change: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            len as i32,
            first_valid as i32,
            &mut d_osc,
            &mut d_change,
        )?;

        Ok(DeviceAcoscPair {
            osc: DeviceArrayF32 {
                buf: d_osc,
                rows: 1,
                cols: len,
            },
            change: DeviceArrayF32 {
                buf: d_change,
                rows: 1,
                cols: len,
            },
        })
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        series_len: i32,
        first_valid: i32,
        d_osc: &mut DeviceBuffer<f32>,
        d_change: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAcoscError> {
        if series_len <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("acosc_batch_f32")
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;

        // Single-threaded kernel: launch 1x1x1 and sync before dropping device temps.
        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut len_i = series_len;
            let mut fv_i = first_valid;
            let mut osc_ptr = d_osc.as_device_ptr().as_raw();
            let mut chg_ptr = d_change.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut osc_ptr as *mut _ as *mut c_void,
                &mut chg_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        }
        // Ensure temporaries on device live until the kernel completes.
        self.stream
            .synchronize()
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        Ok(())
    }

    // -------- Many-series × one-param (time-major) --------
    pub fn acosc_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
    ) -> Result<DeviceAcoscPair, CudaAcoscError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaAcoscError::InvalidInput("empty dimensions".into()));
        }
        if high_tm_f32.len() != low_tm_f32.len() || high_tm_f32.len() != num_series * series_len {
            return Err(CudaAcoscError::InvalidInput(
                "time-major inputs must be same length and match rows*cols".into(),
            ));
        }
        // first_valid per series
        let mut first_valids = vec![series_len as i32; num_series];
        for s in 0..num_series {
            for t in 0..series_len {
                let idx = t * num_series + s;
                if high_tm_f32[idx].is_finite() && low_tm_f32[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let in_bytes = 2 * num_series * series_len * std::mem::size_of::<f32>();
        let out_bytes = 2 * num_series * series_len * std::mem::size_of::<f32>();
        let aux_bytes = num_series * std::mem::size_of::<i32>();
        let required = in_bytes + out_bytes + aux_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaAcoscError::InvalidInput(
                "insufficient device memory for acosc many-series".into(),
            ));
        }

        let d_high = DeviceBuffer::from_slice(high_tm_f32)
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_tm_f32)
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let mut d_osc: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(num_series * series_len) }
                .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let mut d_change: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(num_series * series_len) }
                .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_first,
            num_series as i32,
            series_len as i32,
            &mut d_osc,
            &mut d_change,
        )?;

        Ok(DeviceAcoscPair {
            osc: DeviceArrayF32 {
                buf: d_osc,
                rows: num_series,
                cols: series_len,
            },
            change: DeviceArrayF32 {
                buf: d_change,
                rows: num_series,
                cols: series_len,
            },
        })
    }

    /// Many-series × one-param (time-major) using device-resident inputs to avoid H2D copies.
    /// `d_first_valids` must have length `num_series`.
    pub fn acosc_many_series_one_param_time_major_dev_device_inputs(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
    ) -> Result<DeviceAcoscPair, CudaAcoscError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaAcoscError::InvalidInput("empty dimensions".into()));
        }
        let elems = num_series * series_len;
        let mut d_osc: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        let mut d_change: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            d_high_tm,
            d_low_tm,
            d_first_valids,
            num_series as i32,
            series_len as i32,
            &mut d_osc,
            &mut d_change,
        )?;

        Ok(DeviceAcoscPair {
            osc: DeviceArrayF32 {
                buf: d_osc,
                rows: num_series,
                cols: series_len,
            },
            change: DeviceArrayF32 {
                buf: d_change,
                rows: num_series,
                cols: series_len,
            },
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
        d_osc_tm: &mut DeviceBuffer<f32>,
        d_change_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAcoscError> {
        if num_series <= 0 || series_len <= 0 {
            return Ok(());
        }
        // Heuristic: prefer warp-striped kernel for large series counts with enough time steps
        let use_warp = (num_series as usize) >= 64 && (series_len as usize) >= 128;
        unsafe {
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut ns = num_series;
            let mut sl = series_len;
            let mut osc_ptr = d_osc_tm.as_device_ptr().as_raw();
            let mut chg_ptr = d_change_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut ns as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut osc_ptr as *mut _ as *mut c_void,
                &mut chg_ptr as *mut _ as *mut c_void,
            ];

            if use_warp {
                if let Ok(func) = self
                    .module
                    .get_function("acosc_many_series_one_param_f32_warp")
                {
                    let grid: GridSize = (((num_series as u32) + 31) / 32, 1, 1).into();
                    let block: BlockSize = (32, 1, 1).into();
                    self.stream
                        .launch(&func, grid, block, SHMEM_WARP_BYTES, &mut args)
                        .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
                } else {
                    let func = self
                        .module
                        .get_function("acosc_many_series_one_param_f32")
                        .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
                    let grid: GridSize = (num_series as u32, 1, 1).into();
                    let block: BlockSize = (256, 1, 1).into();
                    self.stream
                        .launch(&func, grid, block, 0, &mut args)
                        .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
                }
            } else {
                let func = self
                    .module
                    .get_function("acosc_many_series_one_param_f32")
                    .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
                let grid: GridSize = (num_series as u32, 1, 1).into();
                let block: BlockSize = (256, 1, 1).into();
                self.stream
                    .launch(&func, grid, block, 0, &mut args)
                    .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
            }
        }
        // Ensure temporary inputs aren’t dropped until this work finishes.
        self.stream
            .synchronize()
            .map_err(|e| CudaAcoscError::Cuda(e.to_string()))?;
        Ok(())
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const NUM_SERIES: usize = 512;
    const SERIES_LEN: usize = 4096;

    fn bytes_one_series() -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let elems = NUM_SERIES * SERIES_LEN;
        let in_bytes = 2 * elems * std::mem::size_of::<f32>();
        let out_bytes = 2 * elems * std::mem::size_of::<f32>();
        let aux = NUM_SERIES * std::mem::size_of::<i32>();
        in_bytes + out_bytes + aux + 64 * 1024 * 1024
    }

    fn synth_hl_from_base(base: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = base.to_vec();
        let mut low = base.to_vec();
        for i in 0..base.len() {
            let v = base[i];
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0031;
            let off = (0.0049 * x.sin()).abs() + 0.13;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct OneSeriesState {
        cuda: CudaAcosc,
        high: Vec<f32>,
        low: Vec<f32>,
    }
    impl CudaBenchState for OneSeriesState {
        fn launch(&mut self) {
            let _ = self.cuda.acosc_batch_dev(&self.high, &self.low).unwrap();
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaAcosc::new(0).unwrap();
        let base = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_base(&base);
        Box::new(OneSeriesState { cuda, high, low })
    }

    struct ManySeriesState {
        cuda: CudaAcosc,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .acosc_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    NUM_SERIES,
                    SERIES_LEN,
                )
                .unwrap();
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaAcosc::new(0).unwrap();
        // Build time-major arrays
        let mut high_tm = vec![f32::NAN; NUM_SERIES * SERIES_LEN];
        let mut low_tm = vec![f32::NAN; NUM_SERIES * SERIES_LEN];
        for s in 0..NUM_SERIES {
            let base = gen_series(SERIES_LEN);
            let (h, l) = synth_hl_from_base(&base);
            for t in 0..SERIES_LEN {
                let idx = t * NUM_SERIES + s;
                high_tm[idx] = h[t];
                low_tm[idx] = l[t];
            }
        }
        Box::new(ManySeriesState {
            cuda,
            high_tm,
            low_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "acosc",
                "one_series",
                "acosc_cuda_batch_dev",
                "1m",
                prep_one_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "acosc",
                "many_series_one_param",
                "acosc_cuda_many_series_one_param_dev",
                "512x4096",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
