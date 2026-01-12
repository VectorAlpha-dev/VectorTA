#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::vi::{ViBatchRange, ViParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, DeviceCopy, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaViError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error(
        "out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes"
    )]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
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
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct DeviceArrayF32Pair {
    pub a: DeviceArrayF32,
    pub b: DeviceArrayF32,
}
impl DeviceArrayF32Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.a.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.a.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaViPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaVi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaViPolicy,
}

impl CudaVi {
    pub fn new(device_id: usize) -> Result<Self, CudaViError> {
        Self::new_with_policy(device_id, CudaViPolicy::default())
    }

    pub fn new_with_policy(device_id: usize, policy: CudaViPolicy) -> Result<Self, CudaViError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vi_kernel.ptx"));
        let jit = [
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, &jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy,
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
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> Result<bool, CudaViError> {
        if !Self::mem_check_enabled() {
            return Ok(true);
        }
        if let Ok((free, _total)) = mem_get_info() {
            return Ok(required_bytes.saturating_add(headroom) <= free);
        }
        Ok(true)
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline(always)]
    fn use_pinned(bytes: usize) -> bool {
        bytes >= (1 << 20)
    }

    fn h2d_upload<T: DeviceCopy>(&self, src: &[T]) -> Result<DeviceBuffer<T>, CudaViError> {
        let bytes = src.len() * std::mem::size_of::<T>();
        if Self::use_pinned(bytes) {
            let h = LockedBuffer::from_slice(src)?;
            let mut d = unsafe { DeviceBuffer::<T>::uninitialized(src.len()) }?;
            d.copy_from(&h)?;
            Ok(d)
        } else {
            Ok(DeviceBuffer::from_slice(src)?)
        }
    }

    #[inline]
    fn choose_launch_1d(
        &self,
        func: &Function,
        n_items: usize,
    ) -> Result<(GridSize, BlockSize), CudaViError> {
        let (min_grid_suggest, block_suggest) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .unwrap_or((0, 256));
        let block_x = block_suggest.clamp(64, 1024);
        let mut grid_x = ((n_items as u32) + block_x - 1) / block_x;
        if min_grid_suggest > 0 {
            grid_x = grid_x.max(min_grid_suggest);
        }
        let gx = grid_x.max(1);

        if let Ok(device) = Device::get_device(self.device_id) {
            if let Ok(max_grid_x) = device.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            {
                if gx > max_grid_x as u32 {
                    return Err(CudaViError::LaunchConfigTooLarge {
                        gx,
                        gy: 1,
                        gz: 1,
                        bx: block_x,
                        by: 1,
                        bz: 1,
                    });
                }
            }
            if let Ok(max_block_x) =
                device.get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            {
                if block_x > max_block_x as u32 {
                    return Err(CudaViError::LaunchConfigTooLarge {
                        gx,
                        gy: 1,
                        gz: 1,
                        bx: block_x,
                        by: 1,
                        bz: 1,
                    });
                }
            }
        }
        Ok(((gx, 1, 1).into(), (block_x, 1, 1).into()))
    }

    fn build_prefix_single(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<(usize, Vec<f32>, Vec<f32>, Vec<f32>), CudaViError> {
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaViError::InvalidInput("length mismatch".into()));
        }
        let n = high.len();
        if n == 0 {
            return Err(CudaViError::InvalidInput("empty input".into()));
        }
        let first = (0..n)
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .ok_or_else(|| CudaViError::InvalidInput("all values NaN".into()))?;

        let mut pfx_tr64 = vec![0.0f64; n];
        let mut pfx_vp64 = vec![0.0f64; n];
        let mut pfx_vm64 = vec![0.0f64; n];

        pfx_tr64[first] = (high[first] - low[first]) as f64;
        pfx_vp64[first] = 0.0;
        pfx_vm64[first] = 0.0;
        let mut prev_h = high[first];
        let mut prev_l = low[first];
        let mut prev_c = close[first];
        for i in (first + 1)..n {
            let hi = high[i];
            let lo = low[i];
            let hl = hi - lo;
            let hc = (hi - prev_c).abs();
            let lc = (lo - prev_c).abs();
            let tr_i = hl.max(hc.max(lc)) as f64;
            let vp_i = (hi - prev_l).abs() as f64;
            let vm_i = (lo - prev_h).abs() as f64;
            pfx_tr64[i] = pfx_tr64[i - 1] + tr_i;
            pfx_vp64[i] = pfx_vp64[i - 1] + vp_i;
            pfx_vm64[i] = pfx_vm64[i - 1] + vm_i;
            prev_h = hi;
            prev_l = lo;
            prev_c = close[i];
        }
        let pfx_tr: Vec<f32> = pfx_tr64.into_iter().map(|v| v as f32).collect();
        let pfx_vp: Vec<f32> = pfx_vp64.into_iter().map(|v| v as f32).collect();
        let pfx_vm: Vec<f32> = pfx_vm64.into_iter().map(|v| v as f32).collect();
        Ok((first, pfx_tr, pfx_vp, pfx_vm))
    }

    fn build_prefix_time_major(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<(Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>), CudaViError> {
        if high_tm.len() != low_tm.len() || high_tm.len() != close_tm.len() {
            return Err(CudaViError::InvalidInput("length mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaViError::InvalidInput("invalid dims".into()));
        }
        if high_tm.len() != cols * rows {
            return Err(CudaViError::InvalidInput(
                "dims do not match data length".into(),
            ));
        }

        let mut first_valids = vec![-1i32; cols];
        let mut pfx_tr64 = vec![0.0f64; cols * rows];
        let mut pfx_vp64 = vec![0.0f64; cols * rows];
        let mut pfx_vm64 = vec![0.0f64; cols * rows];

        for s in 0..cols {
            let mut first = None;
            for r in 0..rows {
                let idx = r * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if h.is_finite() && l.is_finite() && c.is_finite() {
                    first = Some(r);
                    break;
                }
            }
            if let Some(fv) = first {
                first_valids[s] = fv as i32;

                let base = fv * cols + s;
                pfx_tr64[base] = (high_tm[base] - low_tm[base]) as f64;
                pfx_vp64[base] = 0.0;
                pfx_vm64[base] = 0.0;
                let mut prev_h = high_tm[base];
                let mut prev_l = low_tm[base];
                let mut prev_c = close_tm[base];
                for r in (fv + 1)..rows {
                    let idx = r * cols + s;
                    let hi = high_tm[idx];
                    let lo = low_tm[idx];
                    let hl = hi - lo;
                    let hc = (hi - prev_c).abs();
                    let lc = (lo - prev_c).abs();
                    let tr_i = hl.max(hc.max(lc));
                    let vp_i = (hi - prev_l).abs();
                    let vm_i = (lo - prev_h).abs();
                    pfx_tr64[idx] = pfx_tr64[idx - cols] + tr_i as f64;
                    pfx_vp64[idx] = pfx_vp64[idx - cols] + vp_i as f64;
                    pfx_vm64[idx] = pfx_vm64[idx - cols] + vm_i as f64;
                    prev_h = hi;
                    prev_l = lo;
                    prev_c = close_tm[idx];
                }
            } else {
                first_valids[s] = -1;
            }
        }
        let pfx_tr: Vec<f32> = pfx_tr64.into_iter().map(|v| v as f32).collect();
        let pfx_vp: Vec<f32> = pfx_vp64.into_iter().map(|v| v as f32).collect();
        let pfx_vm: Vec<f32> = pfx_vm64.into_iter().map(|v| v as f32).collect();
        Ok((first_valids, pfx_tr, pfx_vp, pfx_vm))
    }

    fn launch_vi_batch_f32(
        &self,
        d_tr: &DeviceBuffer<f32>,
        d_vp: &DeviceBuffer<f32>,
        d_vm: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        rows: usize,
        first_valid: usize,
        d_plus: &mut DeviceBuffer<f32>,
        d_minus: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaViError> {
        let mut func: Function = self.module.get_function("vi_batch_f32").map_err(|_| {
            CudaViError::MissingKernelSymbol {
                name: "vi_batch_f32",
            }
        })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let (min_grid_suggest, block_suggest) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .unwrap_or((0, 256));
        let bx: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
            _ => block_suggest.clamp(64, 1024),
        };

        let mut gx = ((len as u32) + bx - 1) / bx;
        if min_grid_suggest > 0 {
            gx = gx.max(min_grid_suggest);
        }
        let mut gy = rows as u32;

        if let Ok(device) = Device::get_device(self.device_id) {
            if let Ok(max_block_x) =
                device.get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            {
                if bx > max_block_x as u32 {
                    return Err(CudaViError::LaunchConfigTooLarge {
                        gx,
                        gy,
                        gz: 1,
                        bx,
                        by: 1,
                        bz: 1,
                    });
                }
            }
            if let Ok(max_grid_y) = device.get_attribute(cust::device::DeviceAttribute::MaxGridDimY)
            {
                if gy > max_grid_y as u32 {
                    let total = rows
                        .checked_mul(len)
                        .ok_or_else(|| CudaViError::InvalidInput("rows*len overflow".into()))?;
                    let gx1 = ((total as u64) + (bx as u64) - 1) / (bx as u64);
                    let gx1: u32 = gx1
                        .try_into()
                        .map_err(|_| CudaViError::InvalidInput("grid_x overflow".into()))?;
                    gx = gx1.max(1);
                    gy = 1;
                }
            }
            if let Ok(max_grid_x) = device.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            {
                if gx > max_grid_x as u32 {
                    return Err(CudaViError::LaunchConfigTooLarge {
                        gx,
                        gy,
                        gz: 1,
                        bx,
                        by: 1,
                        bz: 1,
                    });
                }
            }
        }

        let grid: GridSize = (gx.max(1), gy.max(1), 1).into();
        let block: BlockSize = (bx, 1, 1).into();

        unsafe {
            let mut tr_ptr = d_tr.as_device_ptr().as_raw();
            let mut vp_ptr = d_vp.as_device_ptr().as_raw();
            let mut vm_ptr = d_vm.as_device_ptr().as_raw();
            let mut pr_ptr = d_periods.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut rows_i = rows as i32;
            let mut first_i = first_valid as i32;
            let mut plus_ptr = d_plus.as_device_ptr().as_raw();
            let mut minus_ptr = d_minus.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut vp_ptr as *mut _ as *mut c_void,
                &mut vm_ptr as *mut _ as *mut c_void,
                &mut pr_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut plus_ptr as *mut _ as *mut c_void,
                &mut minus_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_vi_many_series_one_param_f32(
        &self,
        d_tr: &DeviceBuffer<f32>,
        d_vp: &DeviceBuffer<f32>,
        d_vm: &DeviceBuffer<f32>,
        d_first: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_plus: &mut DeviceBuffer<f32>,
        d_minus: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaViError> {
        let mut func: Function = self
            .module
            .get_function("vi_many_series_one_param_f32")
            .map_err(|_| CudaViError::MissingKernelSymbol {
                name: "vi_many_series_one_param_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let (min_grid_suggest, block_suggest) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .unwrap_or((0, 256));
        let bx: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
            _ => block_suggest.clamp(64, 1024),
        };
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaViError::InvalidInput("rows*cols overflow".into()))?;
        let mut gx = ((total as u64) + (bx as u64) - 1) / (bx as u64);
        if min_grid_suggest > 0 {
            gx = gx.max(min_grid_suggest as u64);
        }
        let gx: u32 = gx
            .try_into()
            .map_err(|_| CudaViError::InvalidInput("grid_x overflow".into()))?;
        let grid: GridSize = (gx.max(1), 1, 1).into();
        let block: BlockSize = (bx, 1, 1).into();

        unsafe {
            let mut tr_ptr = d_tr.as_device_ptr().as_raw();
            let mut vp_ptr = d_vp.as_device_ptr().as_raw();
            let mut vm_ptr = d_vm.as_device_ptr().as_raw();
            let mut fv_ptr = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p_i = period as i32;
            let mut plus_ptr = d_plus.as_device_ptr().as_raw();
            let mut minus_ptr = d_minus.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut vp_ptr as *mut _ as *mut c_void,
                &mut vm_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut plus_ptr as *mut _ as *mut c_void,
                &mut minus_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn vi_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &ViBatchRange,
    ) -> Result<(DeviceArrayF32Pair, Vec<ViParams>), CudaViError> {
        if high_f32.len() != low_f32.len() || high_f32.len() != close_f32.len() {
            return Err(CudaViError::InvalidInput("length mismatch".into()));
        }
        let len = high_f32.len();
        if len == 0 {
            return Err(CudaViError::InvalidInput("empty input".into()));
        }

        fn expand_grid_local(r: &ViBatchRange) -> Vec<ViParams> {
            let (start, end, step) = r.period;
            if step == 0 || start == end {
                return vec![ViParams {
                    period: Some(start),
                }];
            }
            if start < end {
                return (start..=end)
                    .step_by(step)
                    .map(|p| ViParams { period: Some(p) })
                    .collect();
            }
            let mut out = Vec::new();
            let mut cur = start;
            loop {
                out.push(ViParams { period: Some(cur) });
                if cur == end {
                    break;
                }
                cur = match cur.checked_sub(step) {
                    Some(v) => v,
                    None => break,
                };
                if cur < end {
                    break;
                }
            }
            out
        }
        let combos = expand_grid_local(sweep);
        if combos.is_empty() {
            return Err(CudaViError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

        let (first_valid, pfx_tr, pfx_vp, pfx_vm) =
            self.build_prefix_single(high_f32, low_f32, close_f32)?;
        if len - first_valid < max_p {
            return Err(CudaViError::InvalidInput(
                "insufficient valid data after first_valid".into(),
            ));
        }

        let rows = combos.len();
        let bytes = 3usize
            .checked_mul(len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|v| {
                rows.checked_mul(std::mem::size_of::<i32>())
                    .and_then(|p| v.checked_add(p))
            })
            .and_then(|v| {
                rows.checked_mul(len)
                    .and_then(|rc| rc.checked_mul(std::mem::size_of::<f32>()))
                    .and_then(|out| v.checked_add(out))
            })
            .ok_or_else(|| CudaViError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes, headroom)? {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaViError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaViError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_tr = self.h2d_upload(&pfx_tr)?;
        let d_vp = self.h2d_upload(&pfx_vp)?;
        let d_vm = self.h2d_upload(&pfx_vm)?;

        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = self.h2d_upload(&periods_host)?;

        let total = rows
            .checked_mul(len)
            .ok_or_else(|| CudaViError::InvalidInput("rows*len overflow".into()))?;
        let mut d_plus = unsafe { DeviceBuffer::<f32>::uninitialized(total) }?;
        let mut d_minus = unsafe { DeviceBuffer::<f32>::uninitialized(total) }?;

        self.launch_vi_batch_f32(
            &d_tr,
            &d_vp,
            &d_vm,
            &d_periods,
            len,
            rows,
            first_valid,
            &mut d_plus,
            &mut d_minus,
        )?;

        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32Pair {
                a: DeviceArrayF32 {
                    buf: d_plus,
                    rows,
                    cols: len,
                },
                b: DeviceArrayF32 {
                    buf: d_minus,
                    rows,
                    cols: len,
                },
            },
            combos,
        ))
    }

    pub fn vi_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ViParams,
    ) -> Result<DeviceArrayF32Pair, CudaViError> {
        if cols == 0 || rows == 0 {
            return Err(CudaViError::InvalidInput("invalid dims".into()));
        }
        if high_tm_f32.len() != cols * rows
            || low_tm_f32.len() != cols * rows
            || close_tm_f32.len() != cols * rows
        {
            return Err(CudaViError::InvalidInput("dims do not match data".into()));
        }
        let period = params.period.unwrap_or(14);
        if period == 0 || period > rows {
            return Err(CudaViError::InvalidInput("invalid period".into()));
        }

        let (first_valids, pfx_tr, pfx_vp, pfx_vm) =
            self.build_prefix_time_major(high_tm_f32, low_tm_f32, close_tm_f32, cols, rows)?;

        let n = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaViError::InvalidInput("rows*cols overflow".into()))?;
        let bytes = 3usize
            .checked_mul(n)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|v| {
                cols.checked_mul(std::mem::size_of::<i32>())
                    .and_then(|p| v.checked_add(p))
            })
            .and_then(|v| {
                n.checked_mul(std::mem::size_of::<f32>() * 2)
                    .and_then(|out| v.checked_add(out))
            })
            .ok_or_else(|| CudaViError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes, headroom)? {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaViError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaViError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_tr = self.h2d_upload(&pfx_tr)?;
        let d_vp = self.h2d_upload(&pfx_vp)?;
        let d_vm = self.h2d_upload(&pfx_vm)?;
        let d_first = self.h2d_upload(&first_valids)?;

        let mut d_plus = unsafe { DeviceBuffer::<f32>::uninitialized(n) }?;
        let mut d_minus = unsafe { DeviceBuffer::<f32>::uninitialized(n) }?;

        self.launch_vi_many_series_one_param_f32(
            &d_tr,
            &d_vp,
            &d_vm,
            &d_first,
            cols,
            rows,
            period,
            &mut d_plus,
            &mut d_minus,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Pair {
            a: DeviceArrayF32 {
                buf: d_plus,
                rows,
                cols,
            },
            b: DeviceArrayF32 {
                buf: d_minus,
                rows,
                cols,
            },
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    struct BatchState {
        cuda: CudaVi,
        d_tr: DeviceBuffer<f32>,
        d_vp: DeviceBuffer<f32>,
        d_vm: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        len: usize,
        rows: usize,
        first_valid: usize,
        out_plus: DeviceBuffer<f32>,
        out_minus: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.launch_vi_batch_f32(
                &self.d_tr,
                &self.d_vp,
                &self.d_vm,
                &self.d_periods,
                self.len,
                self.rows,
                self.first_valid,
                &mut self.out_plus,
                &mut self.out_minus,
            );
            let _ = self.cuda.stream.synchronize();
        }
    }

    struct ManyState {
        cuda: CudaVi,
        d_tr: DeviceBuffer<f32>,
        d_vp: DeviceBuffer<f32>,
        d_vm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        out_plus: DeviceBuffer<f32>,
        out_minus: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self.cuda.launch_vi_many_series_one_param_f32(
                &self.d_tr,
                &self.d_vp,
                &self.d_vm,
                &self.d_first,
                self.cols,
                self.rows,
                self.period,
                &mut self.out_plus,
                &mut self.out_minus,
            );
            let _ = self.cuda.stream.synchronize();
        }
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        v.push(CudaBenchScenario::new(
            "vi",
            "one_series_many_params",
            "vi_batch",
            "vi_batch/len100k/periods[7..35]",
            || {
                let len = 100_000usize;
                let mut h = vec![f32::NAN; len];
                let mut l = vec![f32::NAN; len];
                let mut c = vec![f32::NAN; len];
                for i in 5..len {
                    let x = i as f32;
                    h[i] = x.sin() + 0.01 * x;
                    l[i] = h[i] - 0.5;
                    c[i] = (0.5 * x).cos() + 0.02 * x;
                }
                let sweep = ViBatchRange { period: (7, 35, 2) };
                let cuda = CudaVi::new(0).unwrap();
                let (start, end, step) = sweep.period;
                let mut periods = Vec::new();
                if step == 0 || start == end {
                    periods.push(start);
                } else if start < end {
                    periods.extend((start..=end).step_by(step));
                } else {
                    let mut cur = start;
                    loop {
                        periods.push(cur);
                        if cur == end {
                            break;
                        }
                        cur = match cur.checked_sub(step) {
                            Some(v) => v,
                            None => break,
                        };
                        if cur < end {
                            break;
                        }
                    }
                }
                let rows = periods.len();
                let max_p = periods.iter().copied().max().unwrap_or(1);
                let (first_valid, pfx_tr, pfx_vp, pfx_vm) =
                    cuda.build_prefix_single(&h, &l, &c).unwrap();
                assert!(len - first_valid >= max_p);
                let d_tr = cuda.h2d_upload(&pfx_tr).unwrap();
                let d_vp = cuda.h2d_upload(&pfx_vp).unwrap();
                let d_vm = cuda.h2d_upload(&pfx_vm).unwrap();
                let periods_host: Vec<i32> = periods.into_iter().map(|p| p as i32).collect();
                let d_periods = cuda.h2d_upload(&periods_host).unwrap();
                let total = rows.checked_mul(len).unwrap();
                let out_plus = unsafe { DeviceBuffer::<f32>::uninitialized(total) }.unwrap();
                let out_minus = unsafe { DeviceBuffer::<f32>::uninitialized(total) }.unwrap();
                Box::new(BatchState {
                    cuda,
                    d_tr,
                    d_vp,
                    d_vm,
                    d_periods,
                    len,
                    rows,
                    first_valid,
                    out_plus,
                    out_minus,
                }) as Box<dyn CudaBenchState>
            },
        ));

        v.push(CudaBenchScenario::new(
            "vi",
            "many_series_one_param",
            "vi_many",
            "vi_many/rows65536xcols64/period14",
            || {
                let rows = 65_536usize;
                let cols = 64usize;
                let mut h = vec![f32::NAN; rows * cols];
                let mut l = vec![f32::NAN; rows * cols];
                let mut c = vec![f32::NAN; rows * cols];
                for s in 0..cols {
                    for r in s..rows {
                        let idx = r * cols + s;
                        let x = (r as f32) * 0.002 + (s as f32) * 0.01;
                        h[idx] = x.sin() + 0.01 * x;
                        l[idx] = h[idx] - 0.4;
                        c[idx] = 0.5 * x.cos() + 0.02 * x;
                    }
                }
                let cuda = CudaVi::new(0).unwrap();
                let (first_valids, pfx_tr, pfx_vp, pfx_vm) = cuda
                    .build_prefix_time_major(&h, &l, &c, cols, rows)
                    .unwrap();
                let d_tr = cuda.h2d_upload(&pfx_tr).unwrap();
                let d_vp = cuda.h2d_upload(&pfx_vp).unwrap();
                let d_vm = cuda.h2d_upload(&pfx_vm).unwrap();
                let d_first = cuda.h2d_upload(&first_valids).unwrap();
                let n = rows.checked_mul(cols).unwrap();
                let out_plus = unsafe { DeviceBuffer::<f32>::uninitialized(n) }.unwrap();
                let out_minus = unsafe { DeviceBuffer::<f32>::uninitialized(n) }.unwrap();
                Box::new(ManyState {
                    cuda,
                    d_tr,
                    d_vp,
                    d_vm,
                    d_first,
                    cols,
                    rows,
                    period: 14,
                    out_plus,
                    out_minus,
                }) as Box<dyn CudaBenchState>
            },
        ));

        v
    }
}
