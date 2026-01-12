#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::ultosc::{UltOscBatchRange, UltOscParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, DeviceCopy};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaUltoscError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
struct Float2 {
    x: f32,
    y: f32,
}

unsafe impl DeviceCopy for Float2 {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Int3 {
    x: i32,
    y: i32,
    z: i32,
}

unsafe impl DeviceCopy for Int3 {}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
    Tiled2D {
        tx: u32,
        ty: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaUltoscPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

pub struct CudaUltosc {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaUltoscPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaUltosc {
    pub fn new(device_id: usize) -> Result<Self, CudaUltoscError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ultosc_kernel.ptx"));
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
            policy: CudaUltoscPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
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

    pub fn set_policy(&mut self, p: CudaUltoscPolicy) {
        self.policy = p;
    }
    pub fn policy(&self) -> &CudaUltoscPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required.saturating_add(headroom) <= free
        } else {
            true
        }
    }
    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaUltoscError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaUltoscError::LaunchConfigTooLarge {
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
    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] ultosc batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaUltosc)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] ultosc many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaUltosc)).debug_many_logged = true;
                }
            }
        }
    }

    pub fn ultosc_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &UltOscBatchRange,
    ) -> Result<DeviceArrayF32, CudaUltoscError> {
        let (combos, first_valid, len) =
            Self::prepare_batch_inputs(high_f32, low_f32, close_f32, sweep)?;
        let (pcmtl, ptr) = build_prefix_sums_ulthlc(high_f32, low_f32, close_f32, first_valid);

        let headroom = 64 * 1024 * 1024usize;
        let prefix_bytes = (len
            .checked_add(1)
            .and_then(|v| v.checked_mul(std::mem::size_of::<Float2>()))
            .and_then(|v| v.checked_mul(2)))
        .ok_or_else(|| CudaUltoscError::InvalidInput("prefix bytes overflow".into()))?;
        let combos_len = combos.len();
        let periods_bytes = combos_len
            .checked_mul(std::mem::size_of::<Int3>())
            .ok_or_else(|| CudaUltoscError::InvalidInput("period bytes overflow".into()))?;
        let out_elems = combos_len
            .checked_mul(len)
            .ok_or_else(|| CudaUltoscError::InvalidInput("output element count overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaUltoscError::InvalidInput("output bytes overflow".into()))?;
        let bytes_required = prefix_bytes
            .checked_add(periods_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaUltoscError::InvalidInput("total bytes overflow".into()))?;
        if !Self::will_fit(bytes_required, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaUltoscError::OutOfMemory {
                    required: bytes_required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaUltoscError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_pcmtl: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(&pcmtl, &self.stream)? };
        let d_ptr: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(&ptr, &self.stream)? };

        let mut periods = Vec::with_capacity(combos.len());
        for c in &combos {
            periods.push(Int3 {
                x: c.timeperiod1.unwrap_or(7) as i32,
                y: c.timeperiod2.unwrap_or(14) as i32,
                z: c.timeperiod3.unwrap_or(28) as i32,
            });
        }
        let d_periods: DeviceBuffer<Int3> =
            unsafe { DeviceBuffer::from_slice_async(&periods, &self.stream)? };

        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream)? };

        self.launch_batch(
            &d_pcmtl,
            &d_ptr,
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

    #[allow(clippy::too_many_arguments)]
    fn launch_batch(
        &self,
        d_pcmtl: &DeviceBuffer<Float2>,
        d_ptr: &DeviceBuffer<Float2>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<Int3>,
        nrows: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaUltoscError> {
        if nrows == 0 || len == 0 {
            return Ok(());
        }

        let func = self.module.get_function("ultosc_batch_f32").map_err(|_| {
            CudaUltoscError::MissingKernelSymbol {
                name: "ultosc_batch_f32",
            }
        })?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let max_grid_y: usize = 65_535;

        let mut launched: usize = 0;
        while launched < nrows {
            let chunk = (nrows - launched).min(max_grid_y);
            let gx = grid_x.max(1);
            let gy = chunk as u32;
            let gz = 1u32;
            self.validate_launch(gx, gy, gz, block_x, 1, 1)?;
            let grid: GridSize = (gx, gy, gz).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                (*(self as *const _ as *mut CudaUltosc)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
            }

            unsafe {
                let mut pcmtl_ptr = d_pcmtl.as_device_ptr().as_raw();
                let mut ptr_ptr = d_ptr.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                    + (launched as u64) * std::mem::size_of::<Int3>() as u64;
                let mut nrows_i = chunk as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw()
                    + (launched as u64) * (len as u64) * std::mem::size_of::<f32>() as u64;
                let args: &mut [*mut c_void] = &mut [
                    &mut pcmtl_ptr as *mut _ as *mut c_void,
                    &mut ptr_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut nrows_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += chunk;
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &UltOscBatchRange,
    ) -> Result<(Vec<UltOscParams>, usize, usize), CudaUltoscError> {
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaUltoscError::InvalidInput(
                "input length mismatch".into(),
            ));
        }
        if high.is_empty() {
            return Err(CudaUltoscError::InvalidInput("empty input".into()));
        }
        let len = high.len();

        let mut first_valid = None;
        for i in 1..len {
            if high[i - 1].is_finite()
                && low[i - 1].is_finite()
                && close[i - 1].is_finite()
                && high[i].is_finite()
                && low[i].is_finite()
                && close[i].is_finite()
            {
                first_valid = Some(i);
                break;
            }
        }
        let first = first_valid
            .ok_or_else(|| CudaUltoscError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_ultosc(sweep)?;
        if combos.is_empty() {
            return Err(CudaUltoscError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        for c in &combos {
            let p1 = c.timeperiod1.unwrap_or(7);
            let p2 = c.timeperiod2.unwrap_or(14);
            let p3 = c.timeperiod3.unwrap_or(28);
            let maxp = p1.max(p2).max(p3);
            if maxp == 0 {
                return Err(CudaUltoscError::InvalidInput("periods must be > 0".into()));
            }
            if maxp > len {
                return Err(CudaUltoscError::InvalidInput(
                    "period exceeds data length".into(),
                ));
            }
            if len - first < maxp {
                return Err(CudaUltoscError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
        }
        Ok((combos, first, len))
    }

    pub fn ultosc_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        p1: usize,
        p2: usize,
        p3: usize,
    ) -> Result<DeviceArrayF32, CudaUltoscError> {
        let prep = Self::prepare_many_series_inputs(
            high_tm_f32,
            low_tm_f32,
            close_tm_f32,
            cols,
            rows,
            p1,
            p2,
            p3,
        )?;
        let (pcmtl_tm, ptr_tm) = build_prefix_sums_time_major_ulthlc(
            high_tm_f32,
            low_tm_f32,
            close_tm_f32,
            cols,
            rows,
            &prep.first_valids,
        );

        let headroom = 64 * 1024 * 1024usize;
        let prefix_bytes = pcmtl_tm
            .len()
            .checked_add(ptr_tm.len())
            .and_then(|v| v.checked_mul(std::mem::size_of::<Float2>()))
            .ok_or_else(|| CudaUltoscError::InvalidInput("prefix bytes overflow".into()))?;
        let meta_bytes = prep
            .first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaUltoscError::InvalidInput("meta bytes overflow".into()))?;
        let out_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaUltoscError::InvalidInput("matrix size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaUltoscError::InvalidInput("output bytes overflow".into()))?;
        let bytes_required = prefix_bytes
            .checked_add(meta_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaUltoscError::InvalidInput("total bytes overflow".into()))?;
        if !Self::will_fit(bytes_required, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaUltoscError::OutOfMemory {
                    required: bytes_required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaUltoscError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_pcmtl_tm: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(&pcmtl_tm, &self.stream)? };
        let d_ptr_tm: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(&ptr_tm, &self.stream)? };
        let d_first: DeviceBuffer<i32> = DeviceBuffer::from_slice(&prep.first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream)? };

        self.launch_many_series(
            &d_pcmtl_tm,
            &d_ptr_tm,
            cols,
            rows,
            p1,
            p2,
            p3,
            &d_first,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series(
        &self,
        d_pcmtl_tm: &DeviceBuffer<Float2>,
        d_ptr_tm: &DeviceBuffer<Float2>,
        cols: usize,
        rows: usize,
        p1: usize,
        p2: usize,
        p3: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaUltoscError> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("ultosc_many_series_one_param_f32")
            .map_err(|_| CudaUltoscError::MissingKernelSymbol {
                name: "ultosc_many_series_one_param_f32",
            })?;

        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto | ManySeriesKernelPolicy::OneD { .. } => {
                let block_x: u32 = match self.policy.many_series {
                    ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
                    _ => 256,
                };
                let grid_x = ((rows as u32) + block_x - 1) / block_x;
                let gx = grid_x.max(1);
                let gy = cols as u32;
                let gz = 1u32;
                self.validate_launch(gx, gy, gz, block_x, 1, 1)?;
                let grid: GridSize = (gx, gy, gz).into();
                let block: BlockSize = (block_x, 1, 1).into();
                unsafe {
                    (*(self as *const _ as *mut CudaUltosc)).last_many =
                        Some(ManySeriesKernelSelected::OneD { block_x });
                }
                unsafe {
                    let mut pcmtl_ptr = d_pcmtl_tm.as_device_ptr().as_raw();
                    let mut ptr_ptr = d_ptr_tm.as_device_ptr().as_raw();
                    let mut cols_i = cols as i32;
                    let mut rows_i = rows as i32;
                    let mut p1_i = p1 as i32;
                    let mut p2_i = p2 as i32;
                    let mut p3_i = p3 as i32;
                    let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
                    let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut pcmtl_ptr as *mut _ as *mut c_void,
                        &mut ptr_ptr as *mut _ as *mut c_void,
                        &mut cols_i as *mut _ as *mut c_void,
                        &mut rows_i as *mut _ as *mut c_void,
                        &mut p1_i as *mut _ as *mut c_void,
                        &mut p2_i as *mut _ as *mut c_void,
                        &mut p3_i as *mut _ as *mut c_void,
                        &mut first_ptr as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)?;
                }
            }
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                let block: BlockSize = (tx, ty, 1).into();
                let grid_x = ((rows as u32) + tx - 1) / tx;
                let grid_y = ((cols as u32) + ty - 1) / ty;
                let gx = grid_x.max(1);
                let gy = grid_y.max(1);
                let gz = 1u32;
                self.validate_launch(gx, gy, gz, tx, ty, 1)?;
                let grid: GridSize = (gx, gy, gz).into();
                unsafe {
                    (*(self as *const _ as *mut CudaUltosc)).last_many =
                        Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
                }
                unsafe {
                    let mut pcmtl_ptr = d_pcmtl_tm.as_device_ptr().as_raw();
                    let mut ptr_ptr = d_ptr_tm.as_device_ptr().as_raw();
                    let mut cols_i = cols as i32;
                    let mut rows_i = rows as i32;
                    let mut p1_i = p1 as i32;
                    let mut p2_i = p2 as i32;
                    let mut p3_i = p3 as i32;
                    let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
                    let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut pcmtl_ptr as *mut _ as *mut c_void,
                        &mut ptr_ptr as *mut _ as *mut c_void,
                        &mut cols_i as *mut _ as *mut c_void,
                        &mut rows_i as *mut _ as *mut c_void,
                        &mut p1_i as *mut _ as *mut c_void,
                        &mut p2_i as *mut _ as *mut c_void,
                        &mut p3_i as *mut _ as *mut c_void,
                        &mut first_ptr as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)?;
                }
            }
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    fn prepare_many_series_inputs(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        p1: usize,
        p2: usize,
        p3: usize,
    ) -> Result<PreparedManySeries, CudaUltoscError> {
        if high_tm.len() != low_tm.len() || high_tm.len() != close_tm.len() {
            return Err(CudaUltoscError::InvalidInput(
                "matrix length mismatch".into(),
            ));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaUltoscError::InvalidInput(
                "matrix dims must be positive".into(),
            ));
        }
        let expected_len = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaUltoscError::InvalidInput("matrix size overflow".into()))?;
        if high_tm.len() != expected_len {
            return Err(CudaUltoscError::InvalidInput(
                "matrix shape mismatch".into(),
            ));
        }
        let maxp = p1.max(p2).max(p3);
        if maxp == 0 {
            return Err(CudaUltoscError::InvalidInput("periods must be > 0".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 1..rows {
                let idx0 = (t - 1) * cols + s;
                let idx1 = t * cols + s;
                if high_tm[idx0].is_finite()
                    && low_tm[idx0].is_finite()
                    && close_tm[idx0].is_finite()
                    && high_tm[idx1].is_finite()
                    && low_tm[idx1].is_finite()
                    && close_tm[idx1].is_finite()
                {
                    fv = Some(t);
                    break;
                }
            }
            let val =
                fv.ok_or_else(|| CudaUltoscError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - val < maxp {
                return Err(CudaUltoscError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
            first_valids[s] = val as i32;
        }
        Ok(PreparedManySeries { first_valids })
    }
}

struct PreparedManySeries {
    first_valids: Vec<i32>,
}

#[inline]
fn split_f64_to_float2_vec(src: &[f64]) -> Vec<Float2> {
    let mut v = Vec::with_capacity(src.len());
    for &d in src {
        let hi = d as f32;
        let lo = (d - (hi as f64)) as f32;
        v.push(Float2 { x: hi, y: lo });
    }
    v
}

fn build_prefix_sums_ulthlc(
    high: &[f32],
    low: &[f32],
    close: &[f32],
    first_valid: usize,
) -> (Vec<Float2>, Vec<Float2>) {
    let len = high.len();
    let mut pcmtl64 = vec![0.0f64; len + 1];
    let mut ptr64 = vec![0.0f64; len + 1];
    for i in 0..len {
        let (mut add_c, mut add_t) = (0.0f64, 0.0f64);
        if i >= first_valid {
            let hi = high[i] as f64;
            let lo = low[i] as f64;
            let ci = close[i] as f64;
            let pc = close[i - 1] as f64;
            let tl = if lo < pc { lo } else { pc };
            let mut trv = hi - lo;
            let d1 = (hi - pc).abs();
            if d1 > trv {
                trv = d1;
            }
            let d2 = (lo - pc).abs();
            if d2 > trv {
                trv = d2;
            }
            add_c = ci - tl;
            add_t = trv;
        }
        pcmtl64[i + 1] = pcmtl64[i] + add_c;
        ptr64[i + 1] = ptr64[i] + add_t;
    }
    (
        split_f64_to_float2_vec(&pcmtl64),
        split_f64_to_float2_vec(&ptr64),
    )
}

fn build_prefix_sums_time_major_ulthlc(
    high_tm: &[f32],
    low_tm: &[f32],
    close_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> (Vec<Float2>, Vec<Float2>) {
    let mut pcmtl_tm64 = vec![0.0f64; (rows + 1) * cols];
    let mut ptr_tm64 = vec![0.0f64; (rows + 1) * cols];
    for t in 0..rows {
        for s in 0..cols {
            let fv = first_valids[s] as usize;
            let (mut add_c, mut add_t) = (0.0f64, 0.0f64);
            if t >= fv {
                let idx = t * cols + s;
                let hi = high_tm[idx] as f64;
                let lo = low_tm[idx] as f64;
                let ci = close_tm[idx] as f64;
                let pc = close_tm[idx - cols] as f64;
                let tl = if lo < pc { lo } else { pc };
                let mut trv = hi - lo;
                let d1 = (hi - pc).abs();
                if d1 > trv {
                    trv = d1;
                }
                let d2 = (lo - pc).abs();
                if d2 > trv {
                    trv = d2;
                }
                add_c = ci - tl;
                add_t = trv;
            }
            let prev = t * cols + s;
            let cur = (t + 1) * cols + s;
            pcmtl_tm64[cur] = pcmtl_tm64[prev] + add_c;
            ptr_tm64[cur] = ptr_tm64[prev] + add_t;
        }
    }
    (
        split_f64_to_float2_vec(&pcmtl_tm64),
        split_f64_to_float2_vec(&ptr_tm64),
    )
}

fn expand_grid_ultosc(r: &UltOscBatchRange) -> Result<Vec<UltOscParams>, CudaUltoscError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaUltoscError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let s = step.max(1);
        let mut v = Vec::new();
        if start <= end {
            let mut cur = start;
            loop {
                v.push(cur);
                if cur == end {
                    break;
                }
                let next = cur.checked_add(s).ok_or_else(|| {
                    CudaUltoscError::InvalidInput(format!(
                        "axis overflow: start={} end={} step={}",
                        start, end, step
                    ))
                })?;
                if next <= cur || next > end {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            loop {
                v.push(cur);
                if cur == end {
                    break;
                }
                let next = match cur.checked_sub(s) {
                    Some(n) => n,
                    None => break,
                };
                if next < end {
                    break;
                }
                cur = next;
            }
        }
        if v.is_empty() {
            return Err(CudaUltoscError::InvalidInput(format!(
                "invalid range: start={} end={} step={}",
                start, end, step
            )));
        }
        Ok(v)
    }
    let t1 = axis(r.timeperiod1)?;
    let t2 = axis(r.timeperiod2)?;
    let t3 = axis(r.timeperiod3)?;
    let cap = t1
        .len()
        .checked_mul(t2.len())
        .and_then(|v| v.checked_mul(t3.len()))
        .ok_or_else(|| CudaUltoscError::InvalidInput("parameter grid size overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &a in &t1 {
        for &b in &t2 {
            for &c in &t3 {
                out.push(UltOscParams {
                    timeperiod1: Some(a),
                    timeperiod2: Some(b),
                    timeperiod3: Some(c),
                });
            }
        }
    }
    Ok(out)
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 60_000;
    const PARAM_SWEEP: usize = 125;
    const MS_COLS: usize = 128;
    const MS_ROWS: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 2 * (ONE_SERIES_LEN + 1) * std::mem::size_of::<Float2>();
        let params_bytes = PARAM_SWEEP * std::mem::size_of::<Int3>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let in_bytes = 2 * (MS_ROWS + 1) * MS_COLS * std::mem::size_of::<Float2>();
        let meta = MS_COLS * std::mem::size_of::<i32>();
        let out_bytes = MS_ROWS * MS_COLS * std::mem::size_of::<f32>();
        in_bytes + meta + out_bytes + 64 * 1024 * 1024
    }

    struct UltoscBatchState {
        cuda: CudaUltosc,
        d_pcmtl: DeviceBuffer<Float2>,
        d_ptr: DeviceBuffer<Float2>,
        d_periods: DeviceBuffer<Int3>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first: usize,
        nrows: usize,
    }
    impl CudaBenchState for UltoscBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_pcmtl,
                    &self.d_ptr,
                    self.len,
                    self.first,
                    &self.d_periods,
                    self.nrows,
                    &mut self.d_out,
                )
                .expect("ultosc batch launch");
            let _ = self.cuda.stream.synchronize();
        }
    }
    fn prep_ultosc_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaUltosc::new(0).expect("cuda ultosc");
        let mut high = vec![f32::NAN; ONE_SERIES_LEN];
        let mut low = vec![f32::NAN; ONE_SERIES_LEN];
        let mut close = gen_series(ONE_SERIES_LEN);
        for i in 1..ONE_SERIES_LEN {
            let x = i as f32;
            let base = close[i];
            let off = (0.0029 * x.sin()).abs() + 0.05;
            high[i] = base + off;
            low[i] = base - off;
        }
        let sweep = UltOscBatchRange {
            timeperiod1: (4, 32, 4),
            timeperiod2: (8, 64, 8),
            timeperiod3: (16, 128, 16),
        };
        let (combos, first, len) =
            CudaUltosc::prepare_batch_inputs(&high, &low, &close, &sweep).expect("prep");
        let (pcmtl, ptr) = build_prefix_sums_ulthlc(&high, &low, &close, first);
        let mut periods = Vec::with_capacity(combos.len());
        for c in &combos {
            periods.push(Int3 {
                x: c.timeperiod1.unwrap() as i32,
                y: c.timeperiod2.unwrap() as i32,
                z: c.timeperiod3.unwrap() as i32,
            });
        }
        let d_pcmtl: DeviceBuffer<Float2> = DeviceBuffer::from_slice(&pcmtl).expect("d_pcmtl");
        let d_ptr: DeviceBuffer<Float2> = DeviceBuffer::from_slice(&ptr).expect("d_ptr");
        let d_periods: DeviceBuffer<Int3> = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(combos.len() * len) }.expect("d_out");
        Box::new(UltoscBatchState {
            cuda,
            d_pcmtl,
            d_ptr,
            d_periods,
            d_out,
            len,
            first,
            nrows: combos.len(),
        })
    }

    struct UltoscManySeriesState {
        cuda: CudaUltosc,
        d_pcmtl_tm: DeviceBuffer<Float2>,
        d_ptr_tm: DeviceBuffer<Float2>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        p1: usize,
        p2: usize,
        p3: usize,
    }
    impl CudaBenchState for UltoscManySeriesState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series(
                    &self.d_pcmtl_tm,
                    &self.d_ptr_tm,
                    self.cols,
                    self.rows,
                    self.p1,
                    self.p2,
                    self.p3,
                    &self.d_first,
                    &mut self.d_out_tm,
                )
                .expect("ultosc many-series launch");
            let _ = self.cuda.stream.synchronize();
        }
    }

    fn synth_hlc_tm(cols: usize, rows: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut close_tm = vec![f32::NAN; cols * rows];
        let mut high_tm = vec![f32::NAN; cols * rows];
        let mut low_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in 1..rows {
                let x = t as f32 + s as f32 * 0.41;
                let base = (x * 0.002).sin() + 0.0003 * x;
                let spread = (x * 0.0013).cos().abs() + 0.04;
                let idx = t * cols + s;
                close_tm[idx] = base;
                high_tm[idx] = base + spread;
                low_tm[idx] = base - spread;
            }
        }
        (high_tm, low_tm, close_tm)
    }

    fn prep_ultosc_many_series() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaUltosc::new(0).expect("cuda ultosc");
        cuda.set_policy(CudaUltoscPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 4 },
        });
        let (high_tm, low_tm, close_tm) = synth_hlc_tm(MS_COLS, MS_ROWS);
        let (p1, p2, p3) = (7usize, 14usize, 28usize);
        let prep = CudaUltosc::prepare_many_series_inputs(
            &high_tm, &low_tm, &close_tm, MS_COLS, MS_ROWS, p1, p2, p3,
        )
        .expect("prep ms");
        let (pcmtl_tm, ptr_tm) = build_prefix_sums_time_major_ulthlc(
            &high_tm,
            &low_tm,
            &close_tm,
            MS_COLS,
            MS_ROWS,
            &prep.first_valids,
        );
        let d_pcmtl_tm = DeviceBuffer::from_slice(&pcmtl_tm).expect("d_pcmtl_tm");
        let d_ptr_tm = DeviceBuffer::from_slice(&ptr_tm).expect("d_ptr_tm");
        let d_first = DeviceBuffer::from_slice(&prep.first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(MS_COLS * MS_ROWS) }.expect("d_out_tm");
        Box::new(UltoscManySeriesState {
            cuda,
            d_pcmtl_tm,
            d_ptr_tm,
            d_first,
            d_out_tm,
            cols: MS_COLS,
            rows: MS_ROWS,
            p1,
            p2,
            p3,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "ultosc",
                "one_series_many_params",
                "ultosc_cuda_batch_dev",
                "60k_x_125",
                || prep_ultosc_batch(),
            )
            .with_mem_required(bytes_one_series_many_params())
            .with_sample_size(10),
            CudaBenchScenario::new(
                "ultosc",
                "many_series_one_param",
                "ultosc_cuda_many_series_one_param",
                "128x1m",
                || prep_ultosc_many_series(),
            )
            .with_mem_required(bytes_many_series_one_param())
            .with_sample_size(10),
        ]
    }
}
