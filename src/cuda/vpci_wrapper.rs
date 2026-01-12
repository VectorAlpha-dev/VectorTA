#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::vpci::{VpciBatchRange, VpciParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, DeviceCopy, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaVpciError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
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

#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
struct Float2 {
    hi: f32,
    lo: f32,
}
unsafe impl DeviceCopy for Float2 {}

#[inline(always)]
fn pack_ds(v: f64) -> Float2 {
    let hi = v as f32;
    let lo = (v - (hi as f64)) as f32;
    Float2 { hi, lo }
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
pub struct CudaVpciPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaVpci {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaVpciPolicy,
}

impl CudaVpci {
    pub fn new(device_id: usize) -> Result<Self, CudaVpciError> {
        Self::new_with_policy(device_id, CudaVpciPolicy::default())
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaVpciPolicy,
    ) -> Result<Self, CudaVpciError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vpci_kernel.ptx"));
        let jit = [
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, &jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaVpciPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaVpciPolicy {
        &self.policy
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaVpciError> {
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaVpciError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaVpciError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    fn build_prefix_single(
        &self,
        close: &[f32],
        volume: &[f32],
    ) -> Result<
        (
            usize,
            LockedBuffer<Float2>,
            LockedBuffer<Float2>,
            LockedBuffer<Float2>,
        ),
        CudaVpciError,
    > {
        if close.len() != volume.len() {
            return Err(CudaVpciError::InvalidInput("length mismatch".into()));
        }
        let n = close.len();
        if n == 0 {
            return Err(CudaVpciError::InvalidInput("empty input".into()));
        }
        let first = (0..n)
            .find(|&i| close[i].is_finite() && volume[i].is_finite())
            .ok_or_else(|| CudaVpciError::InvalidInput("all values are NaN".into()))?;

        let mut pfx_c = unsafe { LockedBuffer::<Float2>::uninitialized(n) }?;
        let mut pfx_v = unsafe { LockedBuffer::<Float2>::uninitialized(n) }?;
        let mut pfx_cv = unsafe { LockedBuffer::<Float2>::uninitialized(n) }?;

        for i in 0..first {
            pfx_c.as_mut_slice()[i] = Float2::default();
            pfx_v.as_mut_slice()[i] = Float2::default();
            pfx_cv.as_mut_slice()[i] = Float2::default();
        }

        let mut sc = 0.0f64;
        let mut sv = 0.0f64;
        let mut scv = 0.0f64;
        for i in first..n {
            let c = if close[i].is_finite() {
                close[i] as f64
            } else {
                0.0
            };
            let v = if volume[i].is_finite() {
                volume[i] as f64
            } else {
                0.0
            };
            sc += c;
            sv += v;
            scv += c * v;
            pfx_c.as_mut_slice()[i] = pack_ds(sc);
            pfx_v.as_mut_slice()[i] = pack_ds(sv);
            pfx_cv.as_mut_slice()[i] = pack_ds(scv);
        }
        Ok((first, pfx_c, pfx_v, pfx_cv))
    }

    fn build_prefix_tm(
        &self,
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<
        (
            Vec<i32>,
            LockedBuffer<Float2>,
            LockedBuffer<Float2>,
            LockedBuffer<Float2>,
        ),
        CudaVpciError,
    > {
        if close_tm.len() != volume_tm.len() {
            return Err(CudaVpciError::InvalidInput("length mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaVpciError::InvalidInput("invalid dims".into()));
        }
        let total = cols.checked_mul(rows).ok_or_else(|| {
            CudaVpciError::InvalidInput("rows*cols overflow in build_prefix_tm".into())
        })?;
        if close_tm.len() != total {
            return Err(CudaVpciError::InvalidInput(
                "dims do not match data length".into(),
            ));
        }

        let mut first_valids = vec![-1i32; cols];
        let mut pfx_c = unsafe { LockedBuffer::<Float2>::uninitialized(total) }?;
        let mut pfx_v = unsafe { LockedBuffer::<Float2>::uninitialized(total) }?;
        let mut pfx_cv = unsafe { LockedBuffer::<Float2>::uninitialized(total) }?;
        pfx_c.as_mut_slice().fill(Float2::default());
        pfx_v.as_mut_slice().fill(Float2::default());
        pfx_cv.as_mut_slice().fill(Float2::default());

        for s in 0..cols {
            let mut first = None;
            for r in 0..rows {
                let idx = r * cols + s;
                if close_tm[idx].is_finite() && volume_tm[idx].is_finite() {
                    first = Some(r);
                    break;
                }
                if close_tm[idx].is_finite() && volume_tm[idx].is_finite() {
                    first = Some(r);
                    break;
                }
            }
            if let Some(fv) = first {
                first_valids[s] = fv as i32;
                let mut sc = 0.0f64;
                let mut sv = 0.0f64;
                let mut scv = 0.0f64;
                for r in fv..rows {
                    let idx = r * cols + s;
                    let c = if close_tm[idx].is_finite() {
                        close_tm[idx] as f64
                    } else {
                        0.0
                    };
                    let v = if volume_tm[idx].is_finite() {
                        volume_tm[idx] as f64
                    } else {
                        0.0
                    };
                    sc += c;
                    sv += v;
                    scv += c * v;
                    pfx_c.as_mut_slice()[idx] = pack_ds(sc);
                    pfx_v.as_mut_slice()[idx] = pack_ds(sv);
                    pfx_cv.as_mut_slice()[idx] = pack_ds(scv);
                }
            } else {
            }
        }
        Ok((first_valids, pfx_c, pfx_v, pfx_cv))
    }

    pub fn vpci_batch_dev(
        &self,
        close_f32: &[f32],
        volume_f32: &[f32],
        sweep: &VpciBatchRange,
    ) -> Result<(DeviceArrayF32Pair, Vec<VpciParams>), CudaVpciError> {
        if close_f32.len() != volume_f32.len() {
            return Err(CudaVpciError::InvalidInput("length mismatch".into()));
        }
        let len = close_f32.len();
        if len == 0 {
            return Err(CudaVpciError::InvalidInput("empty input".into()));
        }

        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                return vec![s];
            }
            let mut out = Vec::new();
            if s < e {
                let mut v = s;
                loop {
                    out.push(v);
                    match v.checked_add(st) {
                        Some(next) if next <= e => v = next,
                        _ => break,
                    }
                }
            } else {
                let mut v = s;
                loop {
                    out.push(v);
                    if v == e {
                        break;
                    }
                    match v.checked_sub(st) {
                        Some(next) if next >= e => v = next,
                        _ => break,
                    }
                }
            }
            out
        }

        let shorts = axis_usize(sweep.short_range);
        let longs = axis_usize(sweep.long_range);
        let rows = shorts.len().checked_mul(longs.len()).ok_or_else(|| {
            CudaVpciError::InvalidInput("rows*cols overflow in vpci_batch_dev".into())
        })?;
        if rows == 0 {
            return Err(CudaVpciError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut combos = Vec::with_capacity(rows);
        for &s in &shorts {
            for &l in &longs {
                combos.push(VpciParams {
                    short_range: Some(s),
                    long_range: Some(l),
                });
            }
        }
        let max_long = combos.iter().map(|p| p.long_range.unwrap()).max().unwrap();

        let (first_valid, h_pfx_c, h_pfx_v, h_pfx_cv) =
            self.build_prefix_single(close_f32, volume_f32)?;
        if len - first_valid < max_long {
            return Err(CudaVpciError::InvalidInput(
                "insufficient valid data after first_valid".into(),
            ));
        }

        let float2_size = std::mem::size_of::<Float2>();
        let f32_size = std::mem::size_of::<f32>();
        let i32_size = std::mem::size_of::<i32>();

        let prefix_bytes = len
            .checked_mul(float2_size)
            .and_then(|b| b.checked_mul(3))
            .ok_or_else(|| {
                CudaVpciError::InvalidInput("prefix byte size overflow in vpci_batch_dev".into())
            })?;
        let vol_bytes = len.checked_mul(f32_size).ok_or_else(|| {
            CudaVpciError::InvalidInput("volume byte size overflow in vpci_batch_dev".into())
        })?;
        let params_bytes = rows
            .checked_mul(2)
            .and_then(|n| n.checked_mul(i32_size))
            .ok_or_else(|| {
                CudaVpciError::InvalidInput("params byte size overflow in vpci_batch_dev".into())
            })?;
        let out_elems = rows.checked_mul(len).ok_or_else(|| {
            CudaVpciError::InvalidInput("rows*len overflow in vpci_batch_dev".into())
        })?;
        let out_bytes = out_elems
            .checked_mul(f32_size)
            .and_then(|b| b.checked_mul(2))
            .ok_or_else(|| {
                CudaVpciError::InvalidInput("output byte size overflow in vpci_batch_dev".into())
            })?;

        let bytes = prefix_bytes
            .checked_add(vol_bytes)
            .and_then(|b| b.checked_add(params_bytes))
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| {
                CudaVpciError::InvalidInput("total byte size overflow in vpci_batch_dev".into())
            })?;

        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit(bytes, headroom)?;

        let d_pfx_c: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_c.as_slice(), &self.stream) }?;
        let d_pfx_v: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_v.as_slice(), &self.stream) }?;
        let d_pfx_cv: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_cv.as_slice(), &self.stream) }?;
        let h_vol = LockedBuffer::from_slice(volume_f32)?;
        let d_vol: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(h_vol.as_slice(), &self.stream) }?;

        let shorts_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.short_range.unwrap() as i32)
            .collect();
        let longs_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.long_range.unwrap() as i32)
            .collect();
        let h_shorts = LockedBuffer::from_slice(&shorts_i32)?;
        let h_longs = LockedBuffer::from_slice(&longs_i32)?;
        let d_shorts: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::from_slice_async(h_shorts.as_slice(), &self.stream) }?;
        let d_longs: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::from_slice_async(h_longs.as_slice(), &self.stream) }?;

        let mut d_vpci: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems) }?;
        let mut d_vpcis: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        let func = self.module.get_function("vpci_batch_f32").map_err(|_| {
            CudaVpciError::MissingKernelSymbol {
                name: "vpci_batch_f32",
            }
        })?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let gx = grid_x.max(1);

        if let Ok(device) = Device::get_device(self.device_id) {
            if let Ok(max_grid_x) = device.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            {
                if gx > max_grid_x as u32 {
                    return Err(CudaVpciError::LaunchConfigTooLarge {
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
                    return Err(CudaVpciError::LaunchConfigTooLarge {
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

        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut pfx_c_ptr = d_pfx_c.as_device_ptr().as_raw();
            let mut pfx_v_ptr = d_pfx_v.as_device_ptr().as_raw();
            let mut pfx_cv_ptr = d_pfx_cv.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol.as_device_ptr().as_raw();
            let mut shorts_ptr = d_shorts.as_device_ptr().as_raw();
            let mut longs_ptr = d_longs.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut n_rows_i = rows as i32;
            let mut first_valid_i = (first_valid.min(len)) as i32;
            let mut out_vpci_ptr = d_vpci.as_device_ptr().as_raw();
            let mut out_vpcis_ptr = d_vpcis.as_device_ptr().as_raw();

            let mut args: [*mut c_void; 11] = [
                &mut pfx_c_ptr as *mut _ as *mut c_void,
                &mut pfx_v_ptr as *mut _ as *mut c_void,
                &mut pfx_cv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut shorts_ptr as *mut _ as *mut c_void,
                &mut longs_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_rows_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_vpci_ptr as *mut _ as *mut c_void,
                &mut out_vpcis_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, &mut args[..])?;
        }

        self.synchronize()?;

        Ok((
            DeviceArrayF32Pair {
                a: DeviceArrayF32 {
                    buf: d_vpci,
                    rows,
                    cols: len,
                },
                b: DeviceArrayF32 {
                    buf: d_vpcis,
                    rows,
                    cols: len,
                },
            },
            combos,
        ))
    }

    pub fn vpci_many_series_one_param_time_major_dev(
        &self,
        close_tm_f32: &[f32],
        volume_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpciParams,
    ) -> Result<DeviceArrayF32Pair, CudaVpciError> {
        let short_p = params.short_range.unwrap_or(5);
        let long_p = params.long_range.unwrap_or(25);
        if short_p == 0 || long_p == 0 || short_p > long_p {
            return Err(CudaVpciError::InvalidInput("invalid params".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaVpciError::InvalidInput("invalid dims".into()));
        }

        let total = rows.checked_mul(cols).ok_or_else(|| {
            CudaVpciError::InvalidInput(
                "rows*cols overflow in vpci_many_series_one_param_time_major_dev".into(),
            )
        })?;
        if close_tm_f32.len() != total || volume_tm_f32.len() != total {
            return Err(CudaVpciError::InvalidInput(
                "dims do not match data length".into(),
            ));
        }

        let (first_valids, h_pfx_c, h_pfx_v, h_pfx_cv) =
            self.build_prefix_tm(close_tm_f32, volume_tm_f32, cols, rows)?;

        let float2_size = std::mem::size_of::<Float2>();
        let f32_size = std::mem::size_of::<f32>();
        let i32_size = std::mem::size_of::<i32>();

        let prefix_bytes = total
            .checked_mul(float2_size)
            .and_then(|b| b.checked_mul(3))
            .ok_or_else(|| {
                CudaVpciError::InvalidInput(
                    "prefix byte size overflow in vpci_many_series_one_param_time_major_dev".into(),
                )
            })?;
        let firsts_bytes = cols.checked_mul(i32_size).ok_or_else(|| {
            CudaVpciError::InvalidInput(
                "firsts byte size overflow in vpci_many_series_one_param_time_major_dev".into(),
            )
        })?;
        let vol_bytes = total.checked_mul(f32_size).ok_or_else(|| {
            CudaVpciError::InvalidInput(
                "volume byte size overflow in vpci_many_series_one_param_time_major_dev".into(),
            )
        })?;
        let out_bytes = total
            .checked_mul(f32_size)
            .and_then(|b| b.checked_mul(2))
            .ok_or_else(|| {
                CudaVpciError::InvalidInput(
                    "output byte size overflow in vpci_many_series_one_param_time_major_dev".into(),
                )
            })?;

        let bytes = prefix_bytes
            .checked_add(firsts_bytes)
            .and_then(|b| b.checked_add(vol_bytes))
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| {
                CudaVpciError::InvalidInput(
                    "total byte size overflow in vpci_many_series_one_param_time_major_dev".into(),
                )
            })?;

        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit(bytes, headroom)?;

        let h_firsts = LockedBuffer::from_slice(&first_valids)?;
        let d_pfx_c: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_c.as_slice(), &self.stream) }?;
        let d_pfx_v: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_v.as_slice(), &self.stream) }?;
        let d_pfx_cv: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_cv.as_slice(), &self.stream) }?;
        let d_firsts: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::from_slice_async(h_firsts.as_slice(), &self.stream) }?;
        let h_vol = LockedBuffer::from_slice(volume_tm_f32)?;
        let d_vol: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(h_vol.as_slice(), &self.stream) }?;

        let mut d_vpci: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;
        let mut d_vpcis: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;

        let func = self
            .module
            .get_function("vpci_many_series_one_param_f32")
            .map_err(|_| CudaVpciError::MissingKernelSymbol {
                name: "vpci_many_series_one_param_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let gx = grid_x.max(1);

        if let Ok(device) = Device::get_device(self.device_id) {
            if let Ok(max_grid_x) = device.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            {
                if gx > max_grid_x as u32 {
                    return Err(CudaVpciError::LaunchConfigTooLarge {
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
                    return Err(CudaVpciError::LaunchConfigTooLarge {
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

        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut pfx_c_ptr = d_pfx_c.as_device_ptr().as_raw();
            let mut pfx_v_ptr = d_pfx_v.as_device_ptr().as_raw();
            let mut pfx_cv_ptr = d_pfx_cv.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol.as_device_ptr().as_raw();
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut short_i = short_p as i32;
            let mut long_i = long_p as i32;
            let mut out_vpci_ptr = d_vpci.as_device_ptr().as_raw();
            let mut out_vpcis_ptr = d_vpcis.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 11] = [
                &mut pfx_c_ptr as *mut _ as *mut c_void,
                &mut pfx_v_ptr as *mut _ as *mut c_void,
                &mut pfx_cv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut short_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut out_vpci_ptr as *mut _ as *mut c_void,
                &mut out_vpcis_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args[..])?;
        }

        self.synchronize()?;

        Ok(DeviceArrayF32Pair {
            a: DeviceArrayF32 {
                buf: d_vpci,
                rows,
                cols,
            },
            b: DeviceArrayF32 {
                buf: d_vpcis,
                rows,
                cols,
            },
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series(rows: usize) -> usize {
        2 * ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + 2 * rows * ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + 3 * ONE_SERIES_LEN * std::mem::size_of::<super::Float2>()
            + rows * 2 * std::mem::size_of::<i32>()
            + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaVpci,
        d_pfx_c: DeviceBuffer<Float2>,
        d_pfx_v: DeviceBuffer<Float2>,
        d_pfx_cv: DeviceBuffer<Float2>,
        d_vol: DeviceBuffer<f32>,
        d_shorts: DeviceBuffer<i32>,
        d_longs: DeviceBuffer<i32>,
        d_vpci: DeviceBuffer<f32>,
        d_vpcis: DeviceBuffer<f32>,
        series_len: usize,
        n_rows: usize,
        first_valid: usize,
        grid: GridSize,
        block: BlockSize,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("vpci_batch_f32")
                .expect("vpci_batch_f32");
            unsafe {
                let mut pfx_c_ptr = self.d_pfx_c.as_device_ptr().as_raw();
                let mut pfx_v_ptr = self.d_pfx_v.as_device_ptr().as_raw();
                let mut pfx_cv_ptr = self.d_pfx_cv.as_device_ptr().as_raw();
                let mut vol_ptr = self.d_vol.as_device_ptr().as_raw();
                let mut shorts_ptr = self.d_shorts.as_device_ptr().as_raw();
                let mut longs_ptr = self.d_longs.as_device_ptr().as_raw();
                let mut series_len_i = self.series_len as i32;
                let mut n_rows_i = self.n_rows as i32;
                let mut first_valid_i = (self.first_valid.min(self.series_len)) as i32;
                let mut out_vpci_ptr = self.d_vpci.as_device_ptr().as_raw();
                let mut out_vpcis_ptr = self.d_vpcis.as_device_ptr().as_raw();

                let mut args: [*mut c_void; 11] = [
                    &mut pfx_c_ptr as *mut _ as *mut c_void,
                    &mut pfx_v_ptr as *mut _ as *mut c_void,
                    &mut pfx_cv_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut shorts_ptr as *mut _ as *mut c_void,
                    &mut longs_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_rows_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_vpci_ptr as *mut _ as *mut c_void,
                    &mut out_vpcis_ptr as *mut _ as *mut c_void,
                ];

                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, &mut args)
                    .expect("vpci launch");
            }
            self.cuda.stream.synchronize().expect("vpci sync");
        }
    }

    fn prep_vpci_one_series_many_params() -> Box<dyn CudaBenchState> {
        let mut close = gen_series(ONE_SERIES_LEN);
        let mut vol = gen_series(ONE_SERIES_LEN);

        for i in 0..1024 {
            close[i] = f32::NAN;
            vol[i] = f32::NAN;
        }

        let sweep = VpciBatchRange {
            short_range: (5, 20, 1),
            long_range: (25, 60, 5),
        };

        let shorts: Vec<usize> = (sweep.short_range.0..=sweep.short_range.1).collect();
        let longs: Vec<usize> = (sweep.long_range.0..=sweep.long_range.1)
            .step_by(sweep.long_range.2.max(1))
            .collect();
        let n_rows = shorts.len() * longs.len();
        assert_eq!(n_rows, ((60 - 25) / 5 + 1) * ((20 - 5) / 1 + 1));

        let mut shorts_i32 = Vec::with_capacity(n_rows);
        let mut longs_i32 = Vec::with_capacity(n_rows);
        for &s in &shorts {
            for &l in &longs {
                shorts_i32.push(s as i32);
                longs_i32.push(l as i32);
            }
        }

        let cuda = CudaVpci::new(0).expect("cuda vpci");
        let (first_valid, h_pfx_c, h_pfx_v, h_pfx_cv) = cuda
            .build_prefix_single(&close, &vol)
            .expect("build_prefix_single");

        let d_pfx_c: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_c.as_slice(), &cuda.stream) }
                .expect("d_pfx_c");
        let d_pfx_v: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_v.as_slice(), &cuda.stream) }
                .expect("d_pfx_v");
        let d_pfx_cv: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::from_slice_async(h_pfx_cv.as_slice(), &cuda.stream) }
                .expect("d_pfx_cv");
        let h_vol = LockedBuffer::from_slice(&vol).expect("h_vol");
        let d_vol: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(h_vol.as_slice(), &cuda.stream) }
                .expect("d_vol");
        let d_shorts =
            unsafe { DeviceBuffer::from_slice_async(&shorts_i32, &cuda.stream) }.expect("d_shorts");
        let d_longs =
            unsafe { DeviceBuffer::from_slice_async(&longs_i32, &cuda.stream) }.expect("d_longs");

        let out_elems = n_rows
            .checked_mul(ONE_SERIES_LEN)
            .expect("vpci out_elems overflow");
        let d_vpci: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &cuda.stream) }.expect("d_vpci");
        let d_vpcis: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &cuda.stream) }.expect("d_vpcis");

        let block_x: u32 = 128;
        let grid_x = ((n_rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        cuda.stream.synchronize().expect("vpci prep sync");

        Box::new(BatchState {
            cuda,
            d_pfx_c,
            d_pfx_v,
            d_pfx_cv,
            d_vol,
            d_shorts,
            d_longs,
            d_vpci,
            d_vpcis,
            series_len: ONE_SERIES_LEN,
            n_rows,
            first_valid,
            grid,
            block,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "vpci",
            "one_series_many_params",
            "vpci_batch",
            "vpci/batch/1e6",
            prep_vpci_one_series_many_params,
        )
        .with_mem_required(bytes_one_series(((60 - 25) / 5 + 1) * ((20 - 5) / 1 + 1)))]
    }
}
